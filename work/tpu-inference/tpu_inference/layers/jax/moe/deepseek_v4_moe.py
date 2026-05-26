# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DeepSeek-V4 MoE math for the JAX TPU backend (functional layer).

Differences vs. V3:
  - score_func is `"sqrtsoftplus"`: scores = sqrt(softplus(linear(x))).
  - Hash-routing for first `n_hash_layers` MoE layers: indices come from a
    fixed `tid2eid[input_id]` lookup (not from scores). Routing weights still
    use the score path.
  - No expert-group structure (no `n_group`/`topk_group` like V3).
  - Bias is added BEFORE topk for selection but is NOT used in routing weights.
  - Optional `swiglu_limit` clamp on Expert(SwiGLU) gate/up activations.

See INVARIANTS.md (I18-I20) for routing semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2


def _shard_e_first(x: jnp.ndarray) -> jnp.ndarray:
    """Constrain a stacked-experts tensor `[E, ...]` to be sharded on E
    across the `attn_dp` mesh axis. No-op outside a mesh context (CPU
    unit tests run without `jax.set_mesh`)."""
    if jax.sharding.get_abstract_mesh().empty:
        return x
    return jax.lax.with_sharding_constraint(x, P('attn_dp', None, None))


def _shard_e_last(x: jnp.ndarray) -> jnp.ndarray:
    """Constrain a `[N, E]` per-(token, expert) tensor to be E-sharded so
    it broadcasts cleanly against E-sharded `h_NEi`."""
    if jax.sharding.get_abstract_mesh().empty:
        return x
    return jax.lax.with_sharding_constraint(x, P(None, 'attn_dp'))


def _shard_e_mid(x: jnp.ndarray) -> jnp.ndarray:
    """Constrain a `[N, E, K]` per-(token, expert, feat) tensor to be
    E-sharded on the middle axis. Without this XLA may decide to all-gather
    the einsum output back to fully-replicated `[N, E, K]` per chip
    (~256 MiB for [128, 256, 2048] bf16) and blow per-chip activation HBM
    in BACKEND_PASSES, even though the W tensor is E-sharded."""
    if jax.sharding.get_abstract_mesh().empty:
        return x
    return jax.lax.with_sharding_constraint(x, P(None, 'attn_dp', None))


# Gate.

@dataclass
class GateParams:
    weight: jnp.ndarray             # [n_routed_experts, dim] fp32
    bias: Optional[jnp.ndarray]     # [n_routed_experts] fp32, or None on hash layers
    tid2eid: Optional[jnp.ndarray]  # [vocab_size, top_k] int32, or None on non-hash layers
    score_func: str
    route_scale: float
    top_k: int


def gate_forward(
    x: jnp.ndarray,           # [N, dim] fp32 (already flattened)
    input_ids: jnp.ndarray,   # [N] int (only used in hash mode)
    params: GateParams,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (weights, indices) each of shape [N, top_k].

    Reference: `Gate.forward` in `/mnt/scratch/v4_pro/inference/model.py`.
    """
    scores = x @ params.weight.T  # [N, n_routed_experts]
    if params.score_func == "softmax":
        scores = jax.nn.softmax(scores, axis=-1)
    elif params.score_func == "sigmoid":
        scores = jax.nn.sigmoid(scores)
    elif params.score_func == "sqrtsoftplus":
        scores = jnp.sqrt(jax.nn.softplus(scores))
    else:
        raise ValueError(f"Unknown score_func: {params.score_func}")
    original_scores = scores
    is_hash = params.tid2eid is not None
    if not is_hash:
        scores = scores + params.bias  # bias for selection only
    if is_hash:
        indices = params.tid2eid[input_ids]  # [N, top_k]
        indices = indices.astype(jnp.int32)
    else:
        _, indices = jax.lax.top_k(scores, params.top_k)
        indices = indices.astype(jnp.int32)
    # Routing weights: gather original_scores at indices.
    weights = jnp.take_along_axis(original_scores, indices, axis=-1)  # [N, top_k]
    if params.score_func != "softmax":
        weights = weights / (weights.sum(axis=-1, keepdims=True))
    weights = weights * params.route_scale
    return weights, indices


# Expert (SwiGLU FFN).

@dataclass
class ExpertParams:
    w1: jnp.ndarray  # [inter_dim, dim] (gate proj)
    w2: jnp.ndarray  # [dim, inter_dim] (down proj)
    w3: jnp.ndarray  # [inter_dim, dim] (up proj)
    swiglu_limit: float


def expert_forward(
    x: jnp.ndarray,             # [N, dim]
    weights: Optional[jnp.ndarray],  # [N, 1] or None — token-wise gating weight
    params: ExpertParams,
) -> jnp.ndarray:
    """SwiGLU FFN. Reference: `Expert.forward`."""
    dtype = x.dtype
    gate = (x.astype(jnp.float32) @ params.w1.astype(jnp.float32).T)
    up = (x.astype(jnp.float32) @ params.w3.astype(jnp.float32).T)
    if params.swiglu_limit > 0:
        up = jnp.clip(up, -params.swiglu_limit, params.swiglu_limit)
        gate = jnp.minimum(gate, params.swiglu_limit)
    h = jax.nn.silu(gate) * up
    if weights is not None:
        h = h * weights.astype(jnp.float32)
    return (h.astype(dtype) @ params.w2.astype(dtype).T).astype(dtype)


# MoE forward.

@dataclass
class MoEParams:
    gate: GateParams
    experts: list  # list[ExpertParams], length n_routed_experts
    shared_expert: ExpertParams
    n_routed_experts: int
    dim: int
    # Pre-stacked weights (built at load time); skips the per-call
    # `jnp.stack(experts[*].wN)` whose all-to-all storm OOMs HLO temp.
    w1_stacked: Optional[jnp.ndarray] = None  # [E, inter, dim]
    w2_stacked: Optional[jnp.ndarray] = None  # [E, dim, inter]
    w3_stacked: Optional[jnp.ndarray] = None  # [E, inter, dim]
    swiglu_limit: Optional[float] = None      # uniform across experts


def moe_forward(
    x: jnp.ndarray,           # [B, S, dim] or [N, dim]
    input_ids: jnp.ndarray,   # [B, S] int
    params: MoEParams,
    layer_idx: int = -1,
    n_real=None,              # traced int32 scalar: # real (non-pad) flat tokens
) -> jnp.ndarray:
    """MoE forward = sum_i (route_weight_i * expert_i(x)) + shared_expert(x).
    Vectorized dense: per-expert weights stacked into `[E, ...]` and
    computed as three einsums; per-(token, expert) routing weight masks
    non-top_k experts. O(E * N) flops vs O(top_k * N) for true sparse
    dispatch (megablox); follow-up B2 in the backlog.
    """
    orig_shape = x.shape
    flat_x = x.reshape(-1, params.dim)             # [N, dim]
    flat_ids = input_ids.reshape(-1)

    def _ckR(name, q):
        # S16: REAL-ROWS-ONLY (global rows < n_real) checksum. The global [ckS]
        # sums include per-process-CONSTANT idle/pad-row uninit-HBM garbage from
        # the shard_map collective, so a raw cross-engine A!=B on [ckS] is NOT
        # proof of a real-row bug. This masks rows>=n_real to disambiguate.
        if layer_idx == 0 and n_real is not None:
            mask = (jnp.arange(q.shape[0]) < jnp.asarray(n_real, jnp.int32))[:, None]
            qm = mask * q.astype(jnp.float32)
            jax.debug.print("[ckR] L{l} {n}: rsum={s:.9e} rabsmax={a:.9e}",
                            l=layer_idx, n=name, s=jnp.sum(qm),
                            a=jnp.max(jnp.abs(qm)))
    _ckR("moe_input", flat_x)

    weights, indices = gate_forward(
        flat_x.astype(jnp.float32), flat_ids, params.gate)
    # weights: [N, top_k] fp32, indices: [N, top_k] int32

    E = params.n_routed_experts
    fp32 = jnp.float32
    dtype = x.dtype

    # Two paths: pre-stacked (production; param tree carries E-sharded
    # `wN_stacked` from load) or per-expert stack inside JIT (tests).
    # The per-expert path needs `_shard_e_first` to avoid all-gathering
    # each stacked operand to its full [E, inter, dim] shape per-chip.
    if params.w1_stacked is not None:
        W1 = params.w1_stacked
        W2 = params.w2_stacked
        W3 = params.w3_stacked
        swiglu_limit = (params.swiglu_limit
                        if params.swiglu_limit is not None
                        else params.experts[0].swiglu_limit)
    else:
        W1 = _shard_e_first(jnp.stack([e.w1 for e in params.experts]))  # [E,i,d]
        W2 = _shard_e_first(jnp.stack([e.w2 for e in params.experts]))  # [E,d,i]
        W3 = _shard_e_first(jnp.stack([e.w3 for e in params.experts]))  # [E,i,d]
        swiglu_limit = params.experts[0].swiglu_limit  # uniform across experts

    # Per-(token, expert) routing weight: zero for non-top_k experts.
    # Round-tripped through `dtype` to match expert_forward's bf16 cast.
    one_hot = jax.nn.one_hot(indices, E, dtype=fp32)        # [N, top_k, E]
    per_expert_weight = jnp.einsum('nke,nk->ne', one_hot, weights)  # [N, E]
    per_expert_weight = per_expert_weight.astype(dtype).astype(fp32)
    # Route the routed-expert collective by the activation's token sharding:
    # explicit shard_map for token-sharded PREFILL (token axis N divisible across
    # attn_dp — the S1-buggy case where idle DP ranks feed uninit HBM into the
    # implicit collective-matmul), else the dense einsum (CPU/no-mesh AND
    # single-token DECODE, whose activation is replicated by _v4_decode_replicate
    # so N=1 cannot — and need not — be sharded over attn_dp).
    mesh = jax.sharding.get_abstract_mesh()
    axis = (mesh.shape['attn_dp'] if not mesh.empty else 1)
    N = flat_x.shape[0]
    use_shard_map = (not mesh.empty) and axis > 1 and N >= axis and N % axis == 0
    # E-sharded only for the dense path; the shard_map path keeps it
    # token-sharded so each rank carries its tokens' full [N/axis, E] weights.
    if not use_shard_map:
        per_expert_weight = _shard_e_last(per_expert_weight)
    if layer_idx == 0:
        jax.debug.print("[ckS] L{l} {n}: sum={s:.9e} absmax={m:.9e}",
                        l=layer_idx, n="moe_perexpw",
                        s=jnp.sum(per_expert_weight),
                        m=jnp.max(jnp.abs(per_expert_weight)))

    if not use_shard_map:
        # Dense einsum path (CPU/no-mesh + replicated decode): bit-for-bit
        # unchanged from the original implementation.
        x_fp32 = flat_x.astype(fp32)
        W1_fp32 = W1.astype(fp32)
        W3_fp32 = W3.astype(fp32)
        gate_NEi = _shard_e_mid(jnp.einsum('nd,eid->nei', x_fp32, W1_fp32))
        up_NEi = _shard_e_mid(jnp.einsum('nd,eid->nei', x_fp32, W3_fp32))
        if swiglu_limit > 0:
            up_NEi = jnp.clip(up_NEi, -swiglu_limit, swiglu_limit)
            gate_NEi = jnp.minimum(gate_NEi, swiglu_limit)
        h_NEi = jax.nn.silu(gate_NEi) * up_NEi
        h_NEi = _shard_e_mid(h_NEi * per_expert_weight[..., None])
        out_NEd = _shard_e_mid(jnp.einsum(
            'nei,edi->ned', h_NEi.astype(dtype), W2.astype(dtype)))  # [N, E, dim]
        # [ckE] S25: DECODE takes THIS dense path (N=1 replicated => use_shard_map
        # False); S24's gmm non-det was PREFILL-only. 2-engine logs show decode
        # moe_input + moe_shared + routing BYTE-IDENTICAL but moe_routed_y DIVERGES
        # => the divergence is born in lines 238-250 from identical input. Split it:
        # a tiny read-only shard_map hands each rank its [N,E//axis,dim] shard so
        # jnp.sum is rank-LOCAL (no collective); [ckEY] is the post-all-reduce y.
        # [ckE] per-rank diverges x2 => local einsum/buffer; [ckE] identical but
        # [ckEY] diverges => the attn_dp E-reduction collective. y unchanged.
        if layer_idx == 0 and not mesh.empty:
            def _ckE_local(o):                       # o: [N, E//axis, dim] LOCAL shard
                of = o.astype(fp32)
                jax.debug.print("[ckE] r={r} eloc_gsum={s:.9e} eloc_sqsum={q:.9e} "
                                "eloc_absmax={m:.9e}",
                                r=jax.lax.axis_index('attn_dp'),
                                s=jnp.sum(of), q=jnp.sum(of * of),
                                m=jnp.max(jnp.abs(of)))
                return jnp.sum(of)[None]
            jax.shard_map(_ckE_local, mesh=mesh,
                          in_specs=(P(None, 'attn_dp', None),),
                          out_specs=P('attn_dp'), check_vma=False)(out_NEd)
        y = out_NEd.astype(fp32).sum(axis=1)                # [N, dim] fp32 routed sum
        if layer_idx == 0:                                  # [ckEY] post-all-reduce y
            jax.debug.print("[ckEY] y_gsum={s:.9e} y_sqsum={q:.9e} y_absmax={m:.9e}",
                            s=jnp.sum(y), q=jnp.sum(y * y), m=jnp.max(jnp.abs(y)))
    else:
        # Sharded PREFILL path. Explicit shard_map over 'attn_dp' that dispatches
        # each token to its top_k experts and runs the per-rank expert FFN through
        # the production grouped-matmul kernel gmm_v2 with zero_initialize=True.
        # S18: the prior bespoke dense per-rank einsum here read UNINITIALISED HBM
        # in its matmul output buffer (identical [ckG] x_full input but divergent
        # [ckL] output across processes, real rows hit; weights + all_gather both
        # exonerated). gmm_v2 DMA-zeroes its unvisited output rows — that is the
        # determinism lever: tokens NOT routed to a rank's experts produce a
        # deterministic 0 instead of garbage. Dispatch mirrors fused_moe_gmm.py
        # (_process_tokens_locally + moe_gmm_local), using plain-JAX gather/scatter
        # (the ragged_* kernels fall back to exactly this) and V4's own gate.
        EP = E // axis                       # experts per rank (256 // 32 = 8)
        top_k = params.gate.top_k

        def _routed_local(x_l, w_l, idx_l, W1_l, W3_l, W2_l):
            x_full = jax.lax.all_gather(x_l, 'attn_dp', axis=0, tiled=True)   # [N,dim]
            x_full = jax.lax.optimization_barrier(x_full)
            w_full = jax.lax.all_gather(w_l, 'attn_dp', axis=0, tiled=True)   # [N,top_k]
            idx_full = jax.lax.all_gather(idx_l, 'attn_dp', axis=0,
                                          tiled=True)                        # [N,top_k]
            r = jax.lax.axis_index('attn_dp')
            Nf = x_full.shape[0]
            if layer_idx == 0:
                # [ckR0] routing determinism + [ckR1] gmm-input baseline. If idx/w
                # gsum DIFFER across 2 fresh engines, the corruptor is the GATE
                # (sharded-contraction router all-reduce), not the gmm.
                jax.debug.print("[ckR0] r={r} idx_gsum={i:.9e} w_gsum={w:.9e} "
                                "w_absmax={wm:.9e} xfull_gsum={x:.9e} xfull_absmax={xm:.9e}",
                                r=r, i=jnp.sum(idx_full.astype(jnp.float32)),
                                w=jnp.sum(w_full.astype(jnp.float32)),
                                wm=jnp.max(jnp.abs(w_full.astype(jnp.float32))),
                                x=jnp.sum(x_full.astype(jnp.float32)),
                                xm=jnp.max(jnp.abs(x_full.astype(jnp.float32))))
            # Dispatch: sort the (token, slot) pairs by global expert id; gmm then
            # processes THIS rank's EP experts via group_offset (megablox pattern).
            idx_flat = idx_full.reshape(-1)                                  # [N*top_k]
            argsort_idx = jnp.argsort(idx_flat)
            token_idx = jnp.arange(Nf, dtype=jnp.int32).repeat(top_k)
            token_idx_sorted = token_idx[argsort_idx]                        # [N*top_k]
            group_sizes = jax.nn.one_hot(idx_flat, E, dtype=jnp.int32).sum(0)  # [E]
            revert_idx = jnp.argsort(argsort_idx)
            # S24: gmm lhs/rhs in bf16 `dtype` (NOT fp32). fp32 lhs is UNIQUE to V4
            # among all gmm_v2 callers (prod fused_moe_gmm uses bf16) and selects an
            # untested fp32 sublane/tile path (get_sublane_tiling(lhs.dtype) gmm_v2:951,
            # tile_m gmm_v2:860) that drives the partial_out_ref carry. g1(fp32) is the
            # non-det ORIGINATOR; g2 already bf16. Match the known-good prod path.
            x_sorted = x_full[token_idx_sorted].astype(dtype)                # [N*top_k,dim]
            group_offset = jnp.asarray([r * EP], jnp.int32)
            # Local weights -> gmm rhs [EP,k,n]; transpose is rank-local (no comm).
            W13_l = jnp.concatenate(
                [W1_l.transpose(0, 2, 1), W3_l.transpose(0, 2, 1)],
                axis=2).astype(dtype)                         # [EP, dim, 2*inter]
            g1 = gmm_v2(x_sorted, W13_l, group_sizes, group_offset=group_offset,
                        zero_initialize=False,
                        preferred_element_type=fp32)          # [N*top_k, 2*inter]
            if layer_idx == 0:
                # [ckR2] g1 (gate/up) masked to THIS rank's OWNED sorted rows. Owned
                # rows = clean x_sorted @ owned W -> MUST be deterministic. Divergence
                # here ==> gmm computes owned rows non-deterministically (uninit read).
                _eid = idx_flat[argsort_idx]
                _om = ((_eid >= r * EP) & (_eid < (r + 1) * EP))[:, None]
                _g1o = jnp.where(_om, g1, 0.0)
                # [ckR1b] gmm INPUT (x_sorted) on owned rows. If this is byte-equal x2
                # engines but [ckR2] g1own diverges x2 => kernel non-det given identical
                # input (decisive); if xso itself diverges => input/gather is the source.
                _xso = jnp.where(_om, x_sorted.astype(fp32), 0.0)
                jax.debug.print("[ckR1b] r={r} xso_gsum={g:.9e} xso_sqsum={q:.9e} "
                                "xso_absmax={m:.9e}", r=r, g=jnp.sum(_xso),
                                q=jnp.sum(_xso * _xso), m=jnp.max(jnp.abs(_xso)))
                jax.debug.print("[ckR2] r={r} g1own_gsum={s:.9e} g1own_sqsum={q:.9e} "
                                "g1own_absmax={m:.9e}", r=r, s=jnp.sum(_g1o),
                                q=jnp.sum(_g1o * _g1o), m=jnp.max(jnp.abs(_g1o)))
            gate, up = jnp.split(g1, 2, axis=-1)
            if swiglu_limit > 0:
                up = jnp.clip(up, -swiglu_limit, swiglu_limit)
                gate = jnp.minimum(gate, swiglu_limit)
            h = (jax.nn.silu(gate) * up).astype(dtype)        # [N*top_k, inter]
            W2g_l = W2_l.transpose(0, 2, 1).astype(dtype)     # [EP, inter, dim]
            g2 = gmm_v2(h, W2g_l, group_sizes, group_offset=group_offset,
                        zero_initialize=False,
                        preferred_element_type=fp32)          # [N*top_k, dim]
            if layer_idx == 0:
                # [ckR3] g2 (down) masked to OWNED sorted rows, BEFORE revert/mask.
                # ckR2 clean & ckR3 divergent ==> g2 gmm; both divergent ==> g1 gmm.
                _eid3 = idx_flat[argsort_idx]
                _om3 = ((_eid3 >= r * EP) & (_eid3 < (r + 1) * EP))[:, None]
                _g2o = jnp.where(_om3, g2, 0.0)
                jax.debug.print("[ckR3] r={r} g2own_gsum={s:.9e} g2own_absmax={m:.9e}",
                                r=r, s=jnp.sum(_g2o), m=jnp.max(jnp.abs(_g2o)))
            # Revert to (token, slot) order. Each (token,slot) has exactly one
            # expert, owned by exactly one rank, so weight-combine + psum must give
            # each token its exact routed sum with no double count -- PROVIDED every
            # row whose expert is NOT on this rank contributes 0.
            # S22: gmm zero_initialize only zeros the OUTER ends of the sorted
            # buffer, NOT the interior rows of OTHER ranks' experts (which interleave
            # among real tokens); those read uninit HBM and the combine pulls that
            # garbage into REAL tokens (the S1 cross-process nondeterminism). So
            # explicitly zero every non-owned (token,slot) here, mirroring production
            # fused_moe_gmm's valid_rows_mask. idx_flat is already in (token,slot)
            # order, aligning row-for-row with token_hidden after the revert.
            token_hidden = g2[revert_idx]                     # [N*top_k, dim]
            _owned = (idx_flat >= r * EP) & (idx_flat < (r + 1) * EP)  # [N*top_k]
            token_hidden = jnp.where(_owned[:, None], token_hidden, 0)
            token_topk = token_hidden.reshape(Nf, top_k, -1) * w_full[..., None]
            local = token_topk.sum(axis=1)                    # [N, dim] fp32
            if layer_idx == 0:
                jax.debug.print("[ckL] r={r} local_gsum={s:.9e} local_absmax={m:.9e}",
                                r=r, s=jnp.sum(local), m=jnp.max(jnp.abs(local)))
            y_full = jax.lax.psum(local, 'attn_dp')           # [N, dim] full sum
            NP = Nf // axis
            return jax.lax.dynamic_slice_in_dim(y_full, r * NP, NP, axis=0)

        # Pad rows (global rows >= n_real) carry uninit-HBM garbage in flat_x (the
        # gate ran on it); zero their activation + routing weight (deterministic 0
        # contribution) and route them deterministically to expert 0.
        if n_real is not None:
            _keep = (jnp.arange(N) < jnp.asarray(n_real, jnp.int32))[:, None]
            flat_x_sm = jnp.where(_keep, flat_x, 0)
            weights_sm = jnp.where(_keep, weights, 0)
            indices_sm = jnp.where(_keep, indices, 0)
        else:
            flat_x_sm, weights_sm, indices_sm = flat_x, weights, indices
        y = jax.shard_map(
            _routed_local, mesh=mesh,
            in_specs=(P('attn_dp', None), P('attn_dp', None), P('attn_dp', None),
                      P('attn_dp', None, None), P('attn_dp', None, None),
                      P('attn_dp', None, None)),
            out_specs=P('attn_dp', None), check_vma=False,
        )(flat_x_sm, weights_sm, indices_sm, W1, W3, W2)     # [N, dim] fp32 routed sum
    if layer_idx == 0:
        jax.debug.print("[ckS] L{l} {n}: sum={s:.9e} absmax={m:.9e}",
                        l=layer_idx, n="moe_routed_y",
                        s=jnp.sum(y), m=jnp.max(jnp.abs(y)))
    _ckR("moe_routed", y)
    shared = expert_forward(flat_x, None, params.shared_expert)
    if layer_idx == 0:
        jax.debug.print("[ckS] L{l} {n}: sum={s:.9e} absmax={m:.9e}",
                        l=layer_idx, n="moe_shared",
                        s=jnp.sum(shared.astype(fp32)),
                        m=jnp.max(jnp.abs(shared.astype(fp32))))
    _ckR("moe_shared", shared)
    y = y + shared.astype(fp32)
    # NOTE: the S22 owned-expert mask inside _routed_local IS the "idle-rank mask
    # before the psum" deferred here. S13's post-gather y>=n_real mask was REFUTED
    # (garbage is not pad-row VALUES); the working fix masks by expert OWNERSHIP in
    # the sorted gmm output, mirroring fused_moe_gmm's valid_rows_mask.
    return y.astype(dtype).reshape(orig_shape)

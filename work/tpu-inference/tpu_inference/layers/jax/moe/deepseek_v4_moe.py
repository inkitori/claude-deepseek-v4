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
    per_expert_weight = _shard_e_last(per_expert_weight)
    if layer_idx == 0:
        jax.debug.print("[ckS] L{l} {n}: sum={s:.9e} absmax={m:.9e}",
                        l=layer_idx, n="moe_perexpw",
                        s=jnp.sum(per_expert_weight),
                        m=jnp.max(jnp.abs(per_expert_weight)))

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
        'nei,edi->ned',
        h_NEi.astype(dtype),
        W2.astype(dtype),
    ))                                                       # [N, E, dim] E-sharded

    # Sum routed experts in fp32 to match the original loop's
    # `y` accumulator dtype, then add the always-on shared expert.
    y = out_NEd.astype(fp32).sum(axis=1)                    # [N, dim] fp32 routed sum
    if layer_idx == 0:
        jax.debug.print("[ckS] L{l} {n}: sum={s:.9e} absmax={m:.9e}",
                        l=layer_idx, n="moe_routed_y",
                        s=jnp.sum(y), m=jnp.max(jnp.abs(y)))
    shared = expert_forward(flat_x, None, params.shared_expert)
    if layer_idx == 0:
        jax.debug.print("[ckS] L{l} {n}: sum={s:.9e} absmax={m:.9e}",
                        l=layer_idx, n="moe_shared",
                        s=jnp.sum(shared.astype(fp32)),
                        m=jnp.max(jnp.abs(shared.astype(fp32))))
    y = y + shared.astype(fp32)
    # NOTE: n_real is plumbed here (seed-path only) for a future idle-rank fix.
    # S13 tried zeroing y rows >= n_real (post-gather replicated mask): REFUTED —
    # decode still collapsed (incoherent), so the garbage is NOT confined to
    # pad-row VALUES; it enters the real rows via the token-axis all-gather /
    # expert all-reduce collective itself (same class as the seed _linear no-op,
    # commit 83f74395). Next: shard_map idle-RANK mask before the psum
    # (fused_moe_gmm.py:230-245 template), not output row-masking.
    return y.astype(dtype).reshape(orig_shape)

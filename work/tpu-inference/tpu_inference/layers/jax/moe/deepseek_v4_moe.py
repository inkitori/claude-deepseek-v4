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


# --------------------- gate ---------------------

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


# --------------------- expert (SwiGLU FFN) ---------------------

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


# --------------------- moe ---------------------

@dataclass
class MoEParams:
    gate: GateParams
    experts: list  # list[ExpertParams], length n_routed_experts
    shared_expert: ExpertParams
    n_routed_experts: int
    dim: int


def moe_forward(
    x: jnp.ndarray,           # [B, S, dim] or [N, dim]
    input_ids: jnp.ndarray,   # [B, S] int
    params: MoEParams,
) -> jnp.ndarray:
    """MoE forward = sum_i (route_weight_i * expert_i(x)) + shared_expert(x).

    Routing is mathematically the same top-k pattern as the PyTorch reference:
    for each token n, we compute a sum over experts of
    ``route_weight[n, e] * expert_e(x[n])`` where ``route_weight[n, e]`` is
    nonzero only for the top_k experts the gate selected for n. The compute
    here is "vectorized dense": rather than launching one Python-unrolled
    expert kernel per expert (which made jit_run_model emit
    ``n_routed_experts * 3`` separate matmul HLOs and blew up XLA compile
    to 30+ minutes for V4-Flash), we stack the per-expert weights into a
    single ``[E, ...]`` tensor and compute gate / up / down as three big
    einsums. The masking is folded into a per-(token, expert) routing
    weight that is zero for non-top_k experts, so the algebra is identical
    to the per-expert loop modulo XLA reduction ordering.

    This is still O(n_routed_experts * N) in flops — true sparse dispatch
    (grouped matmul over only the experts each token actually picks) would
    be ~32x cheaper for top_k=8 / E=256. That's a follow-up; this change
    fixes the compile-time blowup without touching the param tree.
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

    # Stack the 256 ExpertParams entries into [E, ...] tensors. The new
    # leading "expert" dim has no natural sharding from the per-leaf source
    # (each w_e is sharded on `dim`), so without a constraint XLA satisfies
    # the einsum by all-gathering each stacked operand to its full
    # bf16[256, 2048, 4096] = 4 GiB shape on every chip — that's 12 GiB of
    # HLO temp (W1+W2+W3) on top of the ~17 GiB resident weights, which
    # OOMs the 31.25 GiB chip HBM budget at compile time on v6e-32.
    #
    # Constrain to E-sharded: each chip ends up holding 8 of the 256 experts
    # (256 / attn_dp=32 = 8). All downstream tensors that carry the E axis
    # then stay E-sharded, and the only cross-chip comm is a single
    # all-reduce over `attn_dp` on the final `[N, dim]` accumulator (E is
    # the contracting axis when we sum over experts). The reshard at each
    # stack is an all-to-all from "dim-sharded across stacked leaves" to
    # "E-sharded", with the same per-chip footprint (~128 MiB bf16).
    W1 = _shard_e_first(jnp.stack([e.w1 for e in params.experts]))  # [E,i,d]
    W2 = _shard_e_first(jnp.stack([e.w2 for e in params.experts]))  # [E,d,i]
    W3 = _shard_e_first(jnp.stack([e.w3 for e in params.experts]))  # [E,i,d]
    swiglu_limit = params.experts[0].swiglu_limit  # uniform across experts

    # Per-(token, expert) routing weight: nonzero only for the top_k experts
    # the gate picked for each token. Built via one_hot + einsum so it
    # reduces to a single fp32 [N, E] matrix. We round-trip through `dtype`
    # to preserve the bf16-cast precision behavior of the original loop's
    # `per_token_weight.astype(flat_x.dtype)` step.
    one_hot = jax.nn.one_hot(indices, E, dtype=fp32)        # [N, top_k, E]
    per_expert_weight = jnp.einsum('nke,nk->ne', one_hot, weights)  # [N, E]
    per_expert_weight = per_expert_weight.astype(dtype).astype(fp32)
    per_expert_weight = _shard_e_last(per_expert_weight)    # E-sharded

    # Gate / up projections in fp32 (matches expert_forward's fp32 path).
    x_fp32 = flat_x.astype(fp32)
    W1_fp32 = W1.astype(fp32)
    W3_fp32 = W3.astype(fp32)
    gate_NEi = _shard_e_mid(jnp.einsum('nd,eid->nei', x_fp32, W1_fp32))  # [N,E,i]
    up_NEi = _shard_e_mid(jnp.einsum('nd,eid->nei', x_fp32, W3_fp32))    # [N,E,i]
    if swiglu_limit > 0:
        up_NEi = jnp.clip(up_NEi, -swiglu_limit, swiglu_limit)
        gate_NEi = jnp.minimum(gate_NEi, swiglu_limit)
    h_NEi = jax.nn.silu(gate_NEi) * up_NEi                  # [N, E, inter]

    # Apply per-(token, expert) routing weight (mid-apply, matches
    # expert_forward's `h = h * weights` between SwiGLU and w2).
    h_NEi = _shard_e_mid(h_NEi * per_expert_weight[..., None])  # [N,E,i] E-sharded

    # Down projection in the activation dtype (bf16), matching
    # `(h.astype(dtype) @ w2.astype(dtype).T)`.
    out_NEd = _shard_e_mid(jnp.einsum(
        'nei,edi->ned',
        h_NEi.astype(dtype),
        W2.astype(dtype),
    ))                                                       # [N, E, dim] E-sharded

    # Sum routed experts in fp32 to match the original loop's
    # `y` accumulator dtype, then add the always-on shared expert.
    y = out_NEd.astype(fp32).sum(axis=1)                    # [N, dim] fp32
    shared = expert_forward(flat_x, None, params.shared_expert)
    y = y + shared.astype(fp32)
    return y.astype(dtype).reshape(orig_shape)

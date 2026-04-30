# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DeepSeek-V4 attention math for the JAX TPU backend.

Functional-only — every public function takes arrays + weights and returns
arrays. Stateful wrappers (NNX modules, KV caches) live in
`tpu_inference/models/jax/deepseek_v4.py`. This split makes the math testable
in isolation against the PyTorch reference at
`tests/models/jax/_deepseek_v4_reference/`.

The three attention flavors are selected by `compress_ratio`:
  - `0`   → pure sliding-window attention (SWA), window `args.window_size`.
  - `4`   → CSA  (compressor with overlap, indexer-driven top-k).
  - `128` → HCA  (compressor without overlap, deterministic top-k).

See V3_TO_V4_DIFF.md and INVARIANTS.md (I5–I12) for shape conventions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P


# --------------------- general helpers ---------------------

def _replicate(x: jnp.ndarray) -> jnp.ndarray:
    """Force `x` fully replicated. Used on tiny constant tables (ape)
    that the loader heuristically sharded along `attn_dp`; without this
    constraint XLA emits a 32-way reshard at every broadcast site, which
    `Involuntary full rematerialization`-warns and wastes activation HBM.
    No-op outside a mesh context (CPU unit tests run without `jax.set_mesh`).
    """
    if jax.sharding.get_abstract_mesh().empty:
        return x
    return jax.lax.with_sharding_constraint(x, P())

def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float) -> jnp.ndarray:
    """RMSNorm: out = weight * x / sqrt(mean(x*x) + eps). Computation in fp32,
    cast back to x.dtype. Matches PyTorch reference RMSNorm exactly."""
    dtype = x.dtype
    xf = x.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    xf = xf * lax.rsqrt(var + eps)
    return (weight.astype(jnp.float32) * xf).astype(dtype)


def precompute_freqs_cis(
    rope_head_dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> jnp.ndarray:
    """Returns a complex64 tensor of shape [seqlen, rope_head_dim/2]. Matches
    the YaRN-augmented frequencies in the PyTorch reference exactly."""
    def find_correction_dim(num_rotations, dim, base_, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base_))

    def find_correction_range(low_rot, high_rot, dim, base_, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base_, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base_, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(lo, hi, dim):
        if lo == hi:
            hi += 0.001
        lf = (jnp.arange(dim, dtype=jnp.float32) - lo) / (hi - lo)
        return jnp.clip(lf, 0.0, 1.0)

    dim = rope_head_dim
    freqs = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    if original_seq_len > 0:
        lo, hi = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1.0 - linear_ramp_factor(lo, hi, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = jnp.arange(seqlen, dtype=jnp.float32)
    freqs_outer = jnp.outer(t, freqs)  # [seqlen, dim/2]
    return jnp.exp(1j * freqs_outer.astype(jnp.float64)).astype(jnp.complex64)


def apply_rotary_emb(x: jnp.ndarray, freqs_cis: jnp.ndarray, inverse: bool = False) -> jnp.ndarray:
    """RoPE on the last axis of x interpreted as complex pairs. Returns a NEW
    tensor (JAX is immutable, unlike the in-place PyTorch reference). The
    caller should splice the result back into x's last `rope_head_dim` slots.
    """
    orig_dtype = x.dtype
    # x: [..., seqlen, ..., rope_head_dim]. Reshape to complex.
    xf = x.astype(jnp.float32)
    head = xf.reshape(*xf.shape[:-1], -1, 2)
    xc = jax.lax.complex(head[..., 0], head[..., 1])  # complex64
    if inverse:
        freqs_cis = jnp.conj(freqs_cis)
    if x.ndim == 3:
        # [B, S, rope_head_dim/2]
        fc = freqs_cis.reshape(1, xc.shape[1], xc.shape[-1])
    else:
        # [B, S, n_heads, rope_head_dim/2]
        fc = freqs_cis.reshape(1, xc.shape[1], 1, xc.shape[-1])
    yc = xc * fc
    yr = jnp.stack([jnp.real(yc), jnp.imag(yc)], axis=-1)
    out = yr.reshape(*xf.shape)
    return out.astype(orig_dtype)


def splice_rope(x: jnp.ndarray, rope_dim: int, freqs_cis: jnp.ndarray, inverse: bool = False) -> jnp.ndarray:
    """Replace the last `rope_dim` slots of x with their RoPE-rotated values.
    The PyTorch reference uses in-place `apply_rotary_emb(x[..., -rd:], ...)`;
    this function is the JAX-functional equivalent."""
    nope = x[..., :-rope_dim]
    rope = x[..., -rope_dim:]
    rope_rotated = apply_rotary_emb(rope, freqs_cis, inverse)
    return jnp.concatenate([nope, rope_rotated], axis=-1)


# --------------------- sparse attention ---------------------

def sparse_attn(
    q: jnp.ndarray,        # [B, M, H, D]
    kv: jnp.ndarray,       # [B, N, D]
    attn_sink: jnp.ndarray,  # [H] fp32
    topk_idxs: jnp.ndarray,  # [B, M, K] int32; -1 means "ignore"
    softmax_scale: float,
) -> jnp.ndarray:
    """Multi-head attention restricted to top-k KV positions per query, with a
    learnable per-head sink term added to the softmax denominator.

    Reference: sparse_attn_kernel in
    `/mnt/scratch/v4_pro/inference/kernel.py`. The math is:
      logits[b,m,h,k] = q[b,m,h] · kv[b, topk_idxs[b,m,k]] * scale
      mask out logits where topk_idxs[b,m,k] == -1
      m_max = max over valid logits and attn_sink
      out[b,m,h] = sum_k exp(logits-m_max) * kv[b,topk_idxs[b,m,k]] /
                   (sum_k exp(logits-m_max) + exp(attn_sink-m_max))
    """
    B, M, H, D = q.shape
    K = topk_idxs.shape[-1]
    qf = q.astype(jnp.float32)
    kvf = kv.astype(jnp.float32)
    safe_idx = jnp.maximum(topk_idxs, 0).astype(jnp.int32)
    # Gather: kv[b, safe_idx[b,m,k]] -> [B, M, K, D]
    idx_expanded = jnp.broadcast_to(
        safe_idx.reshape(B, M * K, 1), (B, M * K, D))
    kv_gathered = jnp.take_along_axis(kvf, idx_expanded, axis=1)
    kv_gathered = kv_gathered.reshape(B, M, K, D)
    valid = (topk_idxs != -1)  # [B, M, K]
    logits = jnp.einsum("bmhd,bmkd->bmhk", qf, kv_gathered) * softmax_scale
    logits = jnp.where(valid[:, :, None, :], logits, jnp.full_like(logits, -jnp.inf))
    sink = attn_sink.astype(jnp.float32).reshape(1, 1, H, 1)
    m_max = jnp.maximum(logits.max(axis=-1, keepdims=True), sink)
    m_max = jnp.where(jnp.isfinite(m_max), m_max, jnp.zeros_like(m_max))
    p = jnp.exp(logits - m_max)
    p = jnp.where(valid[:, :, None, :], p, jnp.zeros_like(p))
    sink_term = jnp.exp(sink - m_max)
    denom = p.sum(axis=-1, keepdims=True) + sink_term
    p = p / denom
    out = jnp.einsum("bmhk,bmkd->bmhd", p, kv_gathered)
    return out.astype(q.dtype)


# --------------------- mHC sinkhorn ---------------------

def hc_split_sinkhorn(
    mixes: jnp.ndarray,     # [N, mix_hc] fp32
    hc_scale: jnp.ndarray,  # [3] fp32
    hc_base: jnp.ndarray,   # [mix_hc] fp32
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (pre, post, comb) shaped [N, hc], [N, hc], [N, hc, hc].
    Doubly-stochastic-ish `comb` via Sinkhorn iterations. Reference:
    `hc_split_sinkhorn_kernel` in `/mnt/scratch/v4_pro/inference/kernel.py`."""
    H = hc_mult
    pre = jax.nn.sigmoid(mixes[:, :H] * hc_scale[0] + hc_base[:H]) + eps
    post = 2.0 * jax.nn.sigmoid(mixes[:, H:2 * H] * hc_scale[1] + hc_base[H:2 * H])
    comb_lin = mixes[:, 2 * H:] * hc_scale[2] + hc_base[2 * H:]
    comb = comb_lin.reshape(mixes.shape[0], H, H)
    # Initial: row-softmax + eps, then col-normalize with eps.
    comb = jax.nn.softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)

    def body(_, c):
        c = c / (c.sum(axis=-1, keepdims=True) + eps)
        c = c / (c.sum(axis=-2, keepdims=True) + eps)
        return c

    comb = lax.fori_loop(0, sinkhorn_iters - 1, body, comb)
    return pre, post, comb


# --------------------- compressor / indexer (PREFILL ONLY) ---------------------
#
# The PyTorch reference Compressor maintains decode-time state buffers in
# `kv_state` / `score_state`. For Tier 1/2 numerical equivalence we use only
# prefill (start_pos=0); decode-state plumbing lives in the model module.

@dataclass
class CompressorParams:
    ape: jnp.ndarray         # [ratio, coff*head_dim] fp32
    wkv: jnp.ndarray         # [coff*head_dim, dim] fp32
    wgate: jnp.ndarray       # [coff*head_dim, dim] fp32
    norm_w: jnp.ndarray      # [head_dim] fp32
    head_dim: int
    rope_head_dim: int
    compress_ratio: int
    norm_eps: float
    rotate: bool


def _overlap_transform(t: jnp.ndarray, ratio: int, d: int, fill_value: float) -> jnp.ndarray:
    """Mirror of Compressor.overlap_transform.
    t: [B, S, ratio, 2*head_dim] -> [B, S, 2*ratio, head_dim] with the second
    half of head_dim of the *previous* sequence position interleaved into the
    first ratio slots, and the first half of head_dim of the *current*
    sequence position in the last ratio slots."""
    B, S, _, _ = t.shape
    new = jnp.full((B, S, 2 * ratio, d), fill_value, dtype=t.dtype)
    # second-half of head_dim → first ratio slots (will be overwritten by
    # previous-step values below).
    new = new.at[:, :, ratio:, :].set(t[:, :, :, d:])
    # previous step's first-half of head_dim → first ratio slots, shifted by 1.
    prev_first_half = t[:, :-1, :, :d]
    new = new.at[:, 1:, :ratio, :].set(prev_first_half)
    return new


def compressor_prefill(
    x: jnp.ndarray,            # [B, S, dim]
    params: CompressorParams,
    freqs_cis_full: jnp.ndarray,  # [max_seq_len, rope_head_dim/2] complex64
) -> jnp.ndarray:
    """Prefill-only compressor forward.

    Returns the compressed KV `[B, S//ratio, head_dim]` after RoPE on the
    rope tail. Matches the start_pos==0 branch of `Compressor.forward`.

    The math:
      kv  = wkv(x.float())       [B, S, coff*head_dim]
      sc  = wgate(x.float())     [B, S, coff*head_dim]
      Drop trailing remainder S % ratio (its compressor output is undefined
      until decode finishes the window). Then unflatten S→(S//ratio, ratio).
      If overlap (ratio==4): apply overlap_transform that stacks current and
      previous windows.
      Add per-window APE bias to scores; softmax over the ratio axis;
      sum-pool kv weighted by softmax(scores) → [B, S//ratio, head_dim].
      RMSNorm(head_dim), then RoPE on last rope_head_dim slots (using the
      *first* position of each ratio-group's freqs).
    """
    B, S, _ = x.shape
    ratio = params.compress_ratio
    overlap = (ratio == 4)
    coff = 2 if overlap else 1
    d = params.head_dim
    rd = params.rope_head_dim
    if S < ratio:
        # Nothing to compress yet; return an empty [B, 0, head_dim] tensor.
        return jnp.zeros((B, 0, d), dtype=x.dtype)
    xf = x.astype(jnp.float32)
    # Linear (no bias). wkv stored as [out, in], so x @ wkv.T.
    kv = xf @ params.wkv.T
    score = xf @ params.wgate.T
    cutoff = (S // ratio) * ratio
    # Drop remainder. (For S divisible by ratio this is a no-op.)
    kv = kv[:, :cutoff, :]
    score = score[:, :cutoff, :]
    # Reshape to ratio groups.
    kv = kv.reshape(B, cutoff // ratio, ratio, coff * d)
    score = score.reshape(B, cutoff // ratio, ratio, coff * d) + _replicate(params.ape)  # [ratio, coff*d] broadcasts
    if overlap:
        # Insert prev-step + current-step into a doubled-ratio bin.
        kv = _overlap_transform(kv, ratio, d, fill_value=0.0)
        score = _overlap_transform(score, ratio, d, fill_value=-jnp.inf)
        # After overlap_transform, the kv/score tensors are [B, S//ratio, 2*ratio, head_dim].
        kv_pooled = (kv * jax.nn.softmax(score, axis=2)).sum(axis=2)
    else:
        # No overlap: pool over the ratio axis directly with score softmax.
        kv_pooled = (kv * jax.nn.softmax(score, axis=2)).sum(axis=2)
    # cast back to original x dtype, RMSNorm.
    kv_pooled = kv_pooled.astype(x.dtype)
    kv_norm = rms_norm(kv_pooled, params.norm_w, params.norm_eps)
    # RoPE: pick freqs at positions [0, ratio, 2*ratio, ...] (per `freqs_cis[:cutoff:ratio]`)
    fc = freqs_cis_full[:cutoff:ratio]
    kv_norm = splice_rope(kv_norm, rd, fc, inverse=False)
    return kv_norm


# --------------------- topk index helpers ---------------------

def get_window_topk_idxs_prefill(window_size: int, bsz: int, seqlen: int) -> jnp.ndarray:
    """Window-attention topk indices for prefill (start_pos=0).
    Returns [B, S, min(S, window_size)] int32, with -1 for masked entries.
    (When seqlen < window_size, the trailing window slots are absent — this
    matches the torch reference's `arange(min(seqlen, window_size))` width.)
    """
    K = min(seqlen, window_size)
    base = jnp.arange(seqlen)[:, None]               # [S, 1]
    matrix = jnp.maximum(base - window_size + 1, 0) + jnp.arange(K)
    matrix = jnp.where(matrix > base, -1, matrix)    # [S, K]
    return jnp.broadcast_to(matrix[None, :, :], (bsz, seqlen, K)).astype(jnp.int32)


def get_compress_topk_idxs_prefill(ratio: int, bsz: int, seqlen: int, offset: int) -> jnp.ndarray:
    """Compress-attention topk indices for prefill, used when there is no
    indexer (HCA). Each query position s attends to all compressed positions
    `0..(s+1)//ratio - 1` (with offset added)."""
    matrix = jnp.broadcast_to(jnp.arange(seqlen // ratio)[None, :], (seqlen, seqlen // ratio))
    mask = matrix >= (jnp.arange(1, seqlen + 1)[:, None] // ratio)
    matrix = jnp.where(mask, -1, matrix + offset)
    return jnp.broadcast_to(matrix[None, :, :], (bsz, seqlen, matrix.shape[-1])).astype(jnp.int32)


# --------------------- indexer (PREFILL ONLY) ---------------------

@dataclass
class IndexerParams:
    wq_b: jnp.ndarray         # [n_heads*head_dim, q_lora_rank] bf16
    weights_proj: jnp.ndarray  # [n_heads, dim] bf16
    compressor: CompressorParams
    n_heads: int
    head_dim: int
    rope_head_dim: int
    index_topk: int
    softmax_scale: float
    norm_eps: float


def indexer_prefill(
    x: jnp.ndarray,                      # [B, S, dim]
    qr: jnp.ndarray,                     # [B, S, q_lora_rank]
    params: IndexerParams,
    freqs_cis_full: jnp.ndarray,
    offset: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (topk_idxs, indexer_kv_cache_prefill).

    `topk_idxs`: [B, S, K] int32 — top-K compressed positions per query (with
    -1 for invalid causal slots). K = min(index_topk, S//ratio).
    `indexer_kv_cache_prefill`: [B, S//ratio, head_dim] — the indexer's
    compressed KV (used only by tests checking equivalence, not by
    downstream attention).
    """
    B, S, _ = x.shape
    H = params.n_heads
    Dh = params.head_dim
    rd = params.rope_head_dim
    ratio = params.compressor.compress_ratio

    fc = freqs_cis_full[:S]
    # q = wq_b(qr): qr is [B, S, q_lora_rank], wq_b is [H*Dh, q_lora_rank]
    q = (qr.astype(jnp.float32) @ params.wq_b.astype(jnp.float32).T).astype(qr.dtype)
    q = q.reshape(B, S, H, Dh)
    q = splice_rope(q, rd, fc, inverse=False)
    # `rotate_activation` is identity (DECISIONS.md D3); fp4_act_quant is no-op.
    # Compressor produces [B, S//ratio, head_dim].
    kv = compressor_prefill(x, params.compressor, freqs_cis_full)
    end_pos_div_ratio = S // ratio
    # weights_proj: [B, S, H], scaled by softmax_scale * 1/sqrt(H)
    weights = (x.astype(jnp.float32) @ params.weights_proj.astype(jnp.float32).T)
    weights = weights * params.softmax_scale * (H ** -0.5)
    # index_score: [B, S, H, T] = einsum("bshd,btd->bsht", q, kv)
    qf = q.astype(jnp.float32)
    kvf = kv.astype(jnp.float32)
    index_score = jnp.einsum("bshd,btd->bsht", qf, kvf)
    index_score = jax.nn.relu(index_score) * weights[..., None]  # weights broadcast over T
    index_score = index_score.sum(axis=2)  # [B, S, T]
    # Causal mask on compressed positions:
    # T = end_pos_div_ratio. For query position s (0-indexed), valid t range is
    # [0, (s+1)//ratio). Otherwise -inf.
    s_arange = jnp.arange(S)
    t_arange = jnp.arange(end_pos_div_ratio)
    mask = t_arange[None, :] >= ((s_arange + 1)[:, None] // ratio)  # [S, T]
    index_score = jnp.where(mask[None, :, :], -jnp.inf, index_score)
    # Top-k. K = min(index_topk, end_pos_div_ratio).
    K = min(params.index_topk, end_pos_div_ratio)
    if K == 0:
        topk_idxs = jnp.zeros((B, S, 0), dtype=jnp.int32)
    else:
        # `top_k` returns (values, indices); we want indices.
        _, topk_idxs = lax.top_k(index_score, K)
        # Apply causal mask AGAIN: any returned idx that is itself causally
        # invalid (because real top-k may include slots where score==-inf
        # that we got via tie-breaking) -> -1.
        topk_invalid = topk_idxs >= ((s_arange + 1)[None, :, None] // ratio)
        topk_idxs = jnp.where(topk_invalid, -1, topk_idxs + offset)
    return topk_idxs.astype(jnp.int32), kv


# --------------------- attention (PREFILL ONLY) ---------------------

@dataclass
class AttentionParams:
    # core projections
    attn_sink: jnp.ndarray   # [n_heads] fp32
    wq_a: jnp.ndarray        # [q_lora_rank, dim] bf16
    q_norm_w: jnp.ndarray    # [q_lora_rank] fp32
    wq_b: jnp.ndarray        # [n_heads*head_dim, q_lora_rank] bf16
    wkv: jnp.ndarray         # [head_dim, dim] bf16
    kv_norm_w: jnp.ndarray   # [head_dim] fp32
    wo_a: jnp.ndarray        # [n_groups*o_lora_rank, n_heads*head_dim/n_groups] bf16
    wo_b: jnp.ndarray        # [dim, n_groups*o_lora_rank] bf16

    # config
    n_heads: int
    head_dim: int
    rope_head_dim: int
    n_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratio: int       # 0 / 4 / 128
    norm_eps: float
    softmax_scale: float

    # optional sub-modules
    compressor: object = None  # CompressorParams | None
    indexer: object = None     # IndexerParams | None


def _linear(x, w):
    """Convenience: x @ w.T using w's dtype-aware path. We always upcast to fp32
    for accumulation here, then cast back. Matches the PyTorch reference's
    behavior (F.linear in bf16 typically uses fp32 accumulation under the hood
    but the input/output are bf16)."""
    return (x.astype(jnp.float32) @ w.astype(jnp.float32).T).astype(x.dtype)


# --------------------- compressor / indexer (DECODE) ---------------------
#
# Decode-time variants. The reference Compressor maintains two state buffers
# (`kv_state`, `score_state`) and a `kv_cache` of compressed positions. The
# functional JAX equivalents thread these as input/output arrays:
#
#   compressor_decode_step:   manages (kv_state, score_state) and emits a new
#                             compressed kv when (start_pos+1) % ratio == 0.
#                             It does NOT manage a kv_cache buffer — the caller
#                             owns that and decides where to write.
#   indexer_decode_step:      runs its own compressor_decode_step on its
#                             internal compressor params, writes the new
#                             compressed kv into a private kv_cache buffer at
#                             slot start_pos//ratio, then computes top-k.
#   attention_decode_step:    runs its own compressor_decode_step, writes the
#                             new compressed kv into kv_cache[:, win + ...],
#                             plus runs the indexer (if ratio==4) to get
#                             top-K compressed positions, then sparse_attn.
#
# State shapes (per batch, single layer):
#   compressor.kv_state:    [B, coff*ratio, coff*head_dim]  fp32
#   compressor.score_state: [B, coff*ratio, coff*head_dim]  fp32, init -inf
# where coff = 2 if ratio == 4 else 1.


def compressor_init_state(
    batch_size: int,
    head_dim: int,
    compress_ratio: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (kv_state, score_state) for a Compressor."""
    coff = 2 if compress_ratio == 4 else 1
    kv_state = jnp.zeros((batch_size, coff * compress_ratio, coff * head_dim), dtype=jnp.float32)
    score_state = jnp.full((batch_size, coff * compress_ratio, coff * head_dim),
                            -jnp.inf, dtype=jnp.float32)
    return kv_state, score_state


def compressor_decode_step(
    x_step: jnp.ndarray,           # [B, 1, dim]
    start_pos: int,                # absolute decoded position
    params: CompressorParams,
    freqs_cis_full: jnp.ndarray,
    kv_state: jnp.ndarray,         # [B, coff*ratio, coff*head_dim] fp32
    score_state: jnp.ndarray,      # [B, coff*ratio, coff*head_dim] fp32
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, bool]:
    """One decode step. Returns (kv_state', score_state', kv_compressed, did_compress).
    `kv_compressed` is `[B, 1, head_dim]` (the new compressed position) when
    `did_compress` is True. When `did_compress` is False, the returned
    `kv_compressed` is undefined (zeros-of-correct-shape); the caller must
    not use it.

    `did_compress` is a Python bool determined by `start_pos`, so it is
    safe to branch on at trace time.
    """
    B = x_step.shape[0]
    ratio = params.compress_ratio
    overlap = (ratio == 4)
    coff = 2 if overlap else 1
    d = params.head_dim
    rd = params.rope_head_dim

    xf = x_step.astype(jnp.float32)
    kv = xf @ params.wkv.T          # [B, 1, coff*d]
    score = xf @ params.wgate.T     # [B, 1, coff*d]

    pos_in_ratio = start_pos % ratio
    score = score + _replicate(params.ape)[pos_in_ratio]

    kv_one = kv.squeeze(1)
    score_one = score.squeeze(1)

    if overlap:
        # Reference (paraphrased):
        #   kv_state[:, ratio + pos_in_ratio] = kv.squeeze(1)
        #   score_state[:, ratio + pos_in_ratio] = score.squeeze(1)
        kv_state = kv_state.at[:, ratio + pos_in_ratio].set(kv_one)
        score_state = score_state.at[:, ratio + pos_in_ratio].set(score_one)
    else:
        kv_state = kv_state.at[:, pos_in_ratio].set(kv_one)
        score_state = score_state.at[:, pos_in_ratio].set(score_one)

    did_compress = ((start_pos + 1) % ratio) == 0

    if did_compress:
        if overlap:
            kv_concat = jnp.concatenate(
                [kv_state[:, :ratio, :d], kv_state[:, ratio:, d:]], axis=1)
            score_concat = jnp.concatenate(
                [score_state[:, :ratio, :d], score_state[:, ratio:, d:]], axis=1)
        else:
            kv_concat = kv_state[..., :d]
            score_concat = score_state
        softmax_score = jax.nn.softmax(score_concat, axis=1)
        kv_compressed = (kv_concat * softmax_score).sum(axis=1, keepdims=True)  # [B, 1, d]

        # RMSNorm + RoPE on the new compressed position.
        kv_norm = rms_norm(kv_compressed.astype(x_step.dtype), params.norm_w, params.norm_eps)
        rope_pos = start_pos + 1 - ratio
        fc = freqs_cis_full[rope_pos:rope_pos + 1]
        kv_norm = splice_rope(kv_norm, rd, fc, inverse=False)

        # In overlap mode, slide the front half of the buffer up by one ratio
        # group: kv_state[:, :ratio] = kv_state[:, ratio:].
        if overlap:
            kv_state = kv_state.at[:, :ratio].set(kv_state[:, ratio:])
            score_state = score_state.at[:, :ratio].set(score_state[:, ratio:])

        return kv_state, score_state, kv_norm, True
    else:
        # No new compressed position this step. Return zeros placeholder.
        kv_norm = jnp.zeros((B, 1, d), dtype=x_step.dtype)
        return kv_state, score_state, kv_norm, False


def indexer_decode_step(
    x_step: jnp.ndarray,                  # [B, 1, dim]
    qr_step: jnp.ndarray,                 # [B, 1, q_lora_rank]
    start_pos: int,
    params: IndexerParams,
    freqs_cis_full: jnp.ndarray,
    offset: int,
    inner_kv_state: jnp.ndarray,          # compressor's kv_state for this indexer
    inner_score_state: jnp.ndarray,
    inner_kv_cache: jnp.ndarray,          # [B, max/ratio, index_head_dim] — the indexer's compressed cache
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (inner_kv_state, inner_score_state, inner_kv_cache, topk_idxs).

    `topk_idxs` shape: [B, 1, K] where K = min(index_topk, end_pos//ratio).
    K depends on start_pos statically.

    The indexer maintains its OWN kv_cache (separate from attention's). Real
    width is `[B, max_seq_len // ratio, index_head_dim]`. This function reads
    and writes that cache.
    """
    B = x_step.shape[0]
    H = params.n_heads
    Dh = params.head_dim
    rd = params.rope_head_dim
    ratio = params.compressor.compress_ratio

    fc = freqs_cis_full[start_pos:start_pos + 1]
    q = (qr_step.astype(jnp.float32) @ params.wq_b.astype(jnp.float32).T).astype(qr_step.dtype)
    q = q.reshape(B, 1, H, Dh)
    q = splice_rope(q, rd, fc, inverse=False)
    # rotate_activation = identity (D3); fp4_act_quant no-op.

    # Run inner compressor step.
    inner_kv_state, inner_score_state, kv_compressed, did = compressor_decode_step(
        x_step, start_pos, params.compressor, freqs_cis_full,
        inner_kv_state, inner_score_state,
    )
    if did:
        write_idx = start_pos // ratio
        inner_kv_cache = inner_kv_cache.at[:, write_idx].set(kv_compressed.squeeze(1))

    end_pos = start_pos + 1
    end_pos_div_ratio = end_pos // ratio  # number of compressed positions filled

    weights = (x_step.astype(jnp.float32) @ params.weights_proj.astype(jnp.float32).T)
    weights = weights * params.softmax_scale * (H ** -0.5)
    qf = q.astype(jnp.float32)
    kvf = inner_kv_cache.astype(jnp.float32)
    index_score = jnp.einsum("bshd,btd->bsht", qf, kvf)
    index_score = jax.nn.relu(index_score) * weights[..., None]
    index_score = index_score.sum(axis=2)
    t_arange = jnp.arange(inner_kv_cache.shape[1])
    valid_mask = t_arange[None, None, :] < end_pos_div_ratio
    index_score = jnp.where(valid_mask, index_score, -jnp.inf)

    K = min(params.index_topk, end_pos_div_ratio)
    if K == 0:
        topk_idxs = jnp.zeros((B, 1, 0), dtype=jnp.int32)
    else:
        _, topk_idxs = lax.top_k(index_score, K)
        topk_idxs = topk_idxs + offset
    return inner_kv_state, inner_score_state, inner_kv_cache, topk_idxs.astype(jnp.int32)


# --------------------- attention decode helpers ---------------------

def get_window_topk_idxs_decode(window_size: int, bsz: int, start_pos: int) -> jnp.ndarray:
    """Decode-time window topk indices. Shape [B, 1, window_size].

    Mirrors `get_window_topk_idxs(start_pos > 0)` from the reference. We
    require `start_pos` to be a Python int so the index pattern can be built
    at trace time. (vLLM's scheduler knows absolute positions per request.)
    """
    win = window_size
    if start_pos >= win - 1:
        sp = start_pos % win
        # arange(sp+1, win) ++ arange(0, sp+1)
        front = jnp.arange(sp + 1, win, dtype=jnp.int32)
        back = jnp.arange(0, sp + 1, dtype=jnp.int32)
        matrix = jnp.concatenate([front, back], axis=0)
    elif start_pos > 0:
        front = jnp.arange(start_pos + 1, dtype=jnp.int32)
        pad = jnp.full((win - start_pos - 1,), -1, dtype=jnp.int32)
        matrix = jnp.concatenate([front, pad], axis=0)
    else:
        head = jnp.zeros((1,), dtype=jnp.int32)
        pad = jnp.full((win - 1,), -1, dtype=jnp.int32)
        matrix = jnp.concatenate([head, pad], axis=0)
    return jnp.broadcast_to(matrix.reshape(1, 1, win), (bsz, 1, win))


def get_compress_topk_idxs_decode(
    ratio: int, bsz: int, start_pos: int, offset: int) -> jnp.ndarray:
    """Decode-time compressed topk indices for HCA.
    Mirrors `get_compress_topk_idxs(start_pos>0)`:
        arange(0, (start_pos+1) // ratio) + offset
    Returns shape [B, 1, T] where T = (start_pos+1) // ratio.
    """
    T = (start_pos + 1) // ratio
    matrix = jnp.arange(T, dtype=jnp.int32) + offset
    return jnp.broadcast_to(matrix.reshape(1, 1, T), (bsz, 1, T))


@dataclass
class AttentionDecodeState:
    """Per-layer mutable decode state. All fields are explicit JAX arrays so
    we can `lax.scan` over layers / decode steps.

    Fields are uniform across layers (regardless of compress_ratio) so the
    pytree shape is consistent. For layers with ratio==0, the compressor /
    indexer fields are zero-sized placeholders.
    """
    kv_cache: jnp.ndarray            # [B, win + extra, head_dim]; extra=max/ratio if ratio else 0
    compressor_kv_state: jnp.ndarray
    compressor_score_state: jnp.ndarray
    indexer_kv_state: jnp.ndarray
    indexer_score_state: jnp.ndarray
    indexer_kv_cache: jnp.ndarray    # [B, max/ratio, index_head_dim] (only when ratio==4)


def attention_decode_init_state(
    batch_size: int,
    cfg_max_seq_len: int,
    params: "AttentionParams",
    cfg_index_head_dim: int = 0,
    dtype=jnp.bfloat16,
) -> AttentionDecodeState:
    """Allocate the per-layer decode state."""
    win = params.window_size
    ratio = params.compress_ratio
    Dh = params.head_dim
    extra = (cfg_max_seq_len // ratio) if ratio else 0
    kvc = jnp.zeros((batch_size, win + extra, Dh), dtype=dtype)
    if ratio > 0:
        c_kv, c_sc = compressor_init_state(batch_size, Dh, ratio)
    else:
        c_kv = jnp.zeros((batch_size, 0, 0), dtype=jnp.float32)
        c_sc = jnp.full((batch_size, 0, 0), -jnp.inf, dtype=jnp.float32)
    if ratio == 4 and cfg_index_head_dim > 0:
        i_kv, i_sc = compressor_init_state(batch_size, cfg_index_head_dim, ratio)
        i_cache = jnp.zeros((batch_size, cfg_max_seq_len // ratio, cfg_index_head_dim), dtype=dtype)
    else:
        i_kv = jnp.zeros((batch_size, 0, 0), dtype=jnp.float32)
        i_sc = jnp.full((batch_size, 0, 0), -jnp.inf, dtype=jnp.float32)
        i_cache = jnp.zeros((batch_size, 0, 0), dtype=dtype)
    return AttentionDecodeState(
        kv_cache=kvc,
        compressor_kv_state=c_kv,
        compressor_score_state=c_sc,
        indexer_kv_state=i_kv,
        indexer_score_state=i_sc,
        indexer_kv_cache=i_cache,
    )


def attention_decode_step(
    x_step: jnp.ndarray,           # [B, 1, dim]
    start_pos: int,
    params: AttentionParams,
    freqs_cis_full: jnp.ndarray,
    state: AttentionDecodeState,
) -> Tuple[AttentionDecodeState, jnp.ndarray]:
    """One decode step of full attention.

    Mirrors `Attention.forward(x, start_pos>0)` from the reference. Writes:
      - state.kv_cache[:, start_pos % win] = current step's kv (SWA write).
      - state.kv_cache[:, win + start_pos // ratio] = newly-compressed kv,
        when (start_pos+1) % ratio == 0.

    Returns (new_state, y_step) with y_step shape [B, 1, dim].
    """
    B = x_step.shape[0]
    H = params.n_heads
    Dh = params.head_dim
    rd = params.rope_head_dim
    win = params.window_size
    ratio = params.compress_ratio
    eps = params.norm_eps
    fc = freqs_cis_full[start_pos:start_pos + 1]

    # q
    qr = _linear(x_step, params.wq_a)
    qr = rms_norm(qr, params.q_norm_w, eps)
    q = _linear(qr, params.wq_b).reshape(B, 1, H, Dh)
    q_f = q.astype(jnp.float32)
    q = (q_f * lax.rsqrt(jnp.square(q_f).mean(-1, keepdims=True) + eps)).astype(q.dtype)
    q = splice_rope(q, rd, fc, inverse=False)

    # kv (single shared head)
    kv = _linear(x_step, params.wkv)
    kv = rms_norm(kv, params.kv_norm_w, eps)
    kv = splice_rope(kv, rd, fc, inverse=False)

    # SWA write to kv_cache[:, start_pos % win].
    new_kv_cache = state.kv_cache.at[:, start_pos % win].set(kv.squeeze(1))

    topk_idxs = get_window_topk_idxs_decode(win, B, start_pos)

    if ratio > 0:
        offset = win
        # Run attention's compressor step (separate state from indexer).
        c_kvst, c_scst, kv_compressed, did = compressor_decode_step(
            x_step, start_pos, params.compressor, freqs_cis_full,
            state.compressor_kv_state, state.compressor_score_state,
        )
        if did:
            write_idx = win + (start_pos // ratio)
            new_kv_cache = new_kv_cache.at[:, write_idx].set(kv_compressed.squeeze(1))

        if params.indexer is not None:
            i_kvst, i_scst, i_kvcache, compress_topk = indexer_decode_step(
                x_step, qr, start_pos, params.indexer, freqs_cis_full,
                offset, state.indexer_kv_state, state.indexer_score_state,
                state.indexer_kv_cache,
            )
        else:
            compress_topk = get_compress_topk_idxs_decode(ratio, B, start_pos, offset)
            i_kvst = state.indexer_kv_state
            i_scst = state.indexer_score_state
            i_kvcache = state.indexer_kv_cache

        topk_idxs = jnp.concatenate([topk_idxs, compress_topk], axis=-1)
    else:
        c_kvst = state.compressor_kv_state
        c_scst = state.compressor_score_state
        i_kvst = state.indexer_kv_state
        i_scst = state.indexer_score_state
        i_kvcache = state.indexer_kv_cache

    topk_idxs = topk_idxs.astype(jnp.int32)

    o = sparse_attn(q, new_kv_cache, params.attn_sink, topk_idxs, params.softmax_scale)
    o = splice_rope(o, rd, fc, inverse=True)

    G = params.n_groups
    R = params.o_lora_rank
    o_grouped = o.reshape(B, 1, G, -1)
    in_per_group = (H * Dh) // G
    wo_a_view = params.wo_a.reshape(G, R, in_per_group).astype(jnp.float32)
    o_proj = jnp.einsum("bsgd,grd->bsgr", o_grouped.astype(jnp.float32), wo_a_view)
    o_flat = o_proj.reshape(B, 1, G * R).astype(x_step.dtype)
    y = _linear(o_flat, params.wo_b)

    new_state = AttentionDecodeState(
        kv_cache=new_kv_cache,
        compressor_kv_state=c_kvst,
        compressor_score_state=c_scst,
        indexer_kv_state=i_kvst,
        indexer_score_state=i_scst,
        indexer_kv_cache=i_kvcache,
    )
    return new_state, y


def attention_prefill(
    x: jnp.ndarray,                # [B, S, dim]
    params: AttentionParams,
    freqs_cis_full: jnp.ndarray,  # [max_seq_len, rope_head_dim/2] complex64 — for current layer's rope
) -> jnp.ndarray:
    """Full attention forward for prefill (start_pos=0). Returns [B, S, dim].

    Implements the prefill path of `Attention.forward` from
    `/mnt/scratch/v4_pro/inference/model.py`, including all three flavors
    (SWA / CSA / HCA) selected via `params.compress_ratio`.
    """
    B, S, _ = x.shape
    H = params.n_heads
    Dh = params.head_dim
    rd = params.rope_head_dim
    win = params.window_size
    ratio = params.compress_ratio
    eps = params.norm_eps
    fc = freqs_cis_full[:S]

    # q
    qr = _linear(x, params.wq_a)
    qr = rms_norm(qr, params.q_norm_w, eps)  # q_norm
    q = _linear(qr, params.wq_b).reshape(B, S, H, Dh)
    # second RMS-style scaling on q (no learnable weight): q *= rsqrt(mean(q^2)+eps)
    q_f = q.astype(jnp.float32)
    q = (q_f * lax.rsqrt(jnp.square(q_f).mean(-1, keepdims=True) + eps)).astype(q.dtype)
    q = splice_rope(q, rd, fc, inverse=False)

    # kv (single shared head)
    kv = _linear(x, params.wkv)
    kv = rms_norm(kv, params.kv_norm_w, eps)  # kv_norm
    kv = splice_rope(kv, rd, fc, inverse=False)
    # act_quant on kv[..., :-rd] is no-op (DECISIONS.md D2)

    # window topk indices
    topk_idxs = get_window_topk_idxs_prefill(win, B, S)

    # compressed kv + compress topk indices (CSA / HCA only)
    if ratio > 0:
        offset = S  # kv.size(1) at concat time is S (before adding compressed)
        if params.indexer is not None:
            compress_topk, _indexer_kv = indexer_prefill(x, qr, params.indexer, freqs_cis_full, offset)
        else:
            compress_topk = get_compress_topk_idxs_prefill(ratio, B, S, offset)
        topk_idxs = jnp.concatenate([topk_idxs, compress_topk], axis=-1)
    topk_idxs = topk_idxs.astype(jnp.int32)

    # build kv buffer: for prefill, kv_cache[:bsz, :S] = current kv; for compress,
    # the compressed kv is appended after position S. (The PyTorch reference
    # writes to kv_cache + concats the compressor output to a temp tensor;
    # we just use the temp tensor directly since there is no decode follow-up.)
    if ratio > 0:
        kv_compressed = compressor_prefill(x, params.compressor, freqs_cis_full)
        # kv_compressed: [B, S//ratio, head_dim]. Append.
        kv_full = jnp.concatenate([kv, kv_compressed], axis=1)
    else:
        kv_full = kv

    o = sparse_attn(q, kv_full, params.attn_sink, topk_idxs, params.softmax_scale)

    # inverse RoPE on rope dims of o
    o = splice_rope(o, rd, fc, inverse=True)

    # grouped low-rank output projection
    G = params.n_groups
    R = params.o_lora_rank
    o_grouped = o.reshape(B, S, G, -1)
    # wo_a: [G*R, n_heads*Dh/G] -> view as [G, R, ...]. NOTE: PyTorch stores
    # weights as [out, in] where out=G*R and in=n_heads*Dh/G. .view(G, R, in)
    # treats the [out] axis as [G, R].
    in_per_group = (H * Dh) // G
    wo_a_view = params.wo_a.reshape(G, R, in_per_group).astype(jnp.float32)
    o_proj = jnp.einsum("bsgd,grd->bsgr", o_grouped.astype(jnp.float32), wo_a_view)
    o_flat = o_proj.reshape(B, S, G * R).astype(x.dtype)
    return _linear(o_flat, params.wo_b)


# --------------------- prefill→decode state init (S1) ---------------------
#
# Closed-form derivation of `AttentionDecodeState` from a prefill input. The
# state is what `attention_decode_step`'s rolling buffers would contain after
# T steps from zero state. Used by the model wrapper to seed decode state
# after running `attention_prefill` once on a new sequence — subsequent
# decode calls then advance one position at a time, O(1)/step instead of
# O(T²)/step. (Backlog item S1.)
#
# Why closed-form rather than iterating attention_decode_step T times: the
# decode kernel takes start_pos as a Python int and uses static control
# flow (`if did_compress`, `kv_state.at[:, pos_in_ratio]`). Iterating it
# inside `lax.scan` would require a refactor to traced start_pos; iterating
# in Python unrolls T copies into HLO at compile time, which scales 50×T
# across layers and blows up compile cost. The closed-form below mirrors
# the reference state semantics directly. Pinned by parity tests in
# `test_deepseek_v4.py::TestPrefillToDecodeStateParity`.

def _compressor_state_from_prefill(
    x: jnp.ndarray,             # [B, T, dim]
    params: CompressorParams,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (kv_state, score_state) matching torch reference's
    `Compressor.forward(start_pos=0)` post-state at end of T prefill steps.

    Layout (B = x.shape[0], ratio = params.compress_ratio, d = head_dim,
    cutoff = T - T%ratio, remainder = T % ratio):
      overlap (ratio==4), shape [B, 8, 2*d]:
        slots [0, ratio):           kv at positions [cutoff-ratio, cutoff)
                                    + ape (only if cutoff >= ratio)
        slots [ratio, ratio+rem):   kv at positions [cutoff, T) + ape[:rem]
        slots [ratio+rem, 2*ratio): init (zero kv, -inf score)
      no-overlap (ratio==128), shape [B, 128, d]:
        slots [0, remainder):       kv at positions [cutoff, T) + ape[:rem]
        slots [remainder, ratio):   init

    NB: this state DIFFERS at the field level from running
    `compressor_decode_step` T times from zero state — torch leaves the
    "back" slots (overlap) or "right" slots (no-overlap) at init values
    that the iterative path would have populated with stale data — but
    both produce identical compression outputs at the next compression
    boundary because those slots are fully overwritten before then.
    Pin via `TestPrefillToDecodeStateParity::test_field_parity`.
    """
    B, T, _ = x.shape
    ratio = params.compress_ratio
    overlap = (ratio == 4)
    coff = 2 if overlap else 1
    d = params.head_dim

    if T == 0:
        return compressor_init_state(B, d, ratio)

    xf = x.astype(jnp.float32)
    kv_full = xf @ params.wkv.T          # [B, T, coff*d]
    score_full = xf @ params.wgate.T     # [B, T, coff*d]

    full_d = coff * d
    n_slots = 2 * ratio if overlap else ratio
    kv_state = jnp.zeros((B, n_slots, full_d), dtype=jnp.float32)
    score_state = jnp.full((B, n_slots, full_d), -jnp.inf, dtype=jnp.float32)

    remainder = T % ratio
    cutoff = T - remainder
    offset = ratio if overlap else 0
    ape = _replicate(params.ape)  # [ratio, coff*d]

    # Most-recent completed window (overlap only): slots [:ratio] = kv at
    # positions [cutoff-ratio, cutoff) + ape (full ratio slice).
    if overlap and cutoff >= ratio:
        kv_state = kv_state.at[:, :ratio, :].set(kv_full[:, cutoff - ratio:cutoff, :])
        score_state = score_state.at[:, :ratio, :].set(
            score_full[:, cutoff - ratio:cutoff, :] + ape)

    # In-progress window: slots [offset, offset+remainder) = kv at
    # positions [cutoff, T) + ape[:remainder].
    if remainder > 0:
        kv_state = kv_state.at[:, offset:offset + remainder, :].set(
            kv_full[:, cutoff:T, :])
        score_state = score_state.at[:, offset:offset + remainder, :].set(
            score_full[:, cutoff:T, :] + ape[:remainder])

    return kv_state, score_state


def _swa_kv_cache_from_prefill(
    kv: jnp.ndarray,             # [B, T, head_dim] — already rms_normed + RoPE'd
    win: int,
) -> jnp.ndarray:
    """Build the SWA portion of attention's kv_cache after T prefill steps.

    Mirrors what `attention_decode_step.kv_cache[:, :win]` holds after T
    decode calls from zero state. Slot i holds the most recent kv at position
    p with `p % win == i`. Returns shape [B, win, head_dim]. dtype matches kv.
    """
    B, T, D = kv.shape
    if T == 0:
        return jnp.zeros((B, win, D), dtype=kv.dtype)
    if T < win:
        out = jnp.zeros((B, win, D), dtype=kv.dtype)
        return out.at[:, :T, :].set(kv)
    # T >= win: cache[i] = kv[T-1 - ((T-1-i) % win)].
    # Equivalent: take the last win positions and roll by T % win so slot
    # (T-win+k) % win = (k + T) % win lands at index (k + T) % win.
    last = kv[:, T - win:T, :]                  # [B, win, D]
    return jnp.roll(last, shift=T % win, axis=1)


def attention_init_state_from_prefill(
    x: jnp.ndarray,                # [B, T, dim]
    params: AttentionParams,
    freqs_cis_full: jnp.ndarray,
    cfg_max_seq_len: int,
    cfg_index_head_dim: int = 0,
    dtype=jnp.bfloat16,
) -> AttentionDecodeState:
    """Closed-form construction of `AttentionDecodeState` after a prefill of
    length T = x.shape[1]. Equivalent to running `attention_decode_step` T
    times from `attention_decode_init_state(...)` zero state.

    The decode state can then drive `attention_decode_step` at start_pos=T
    without re-running prefill. This is the missing primitive that lets S1
    convert vLLM's "every step is a fresh prefill" path into "first call
    prefills and seeds state, subsequent calls do O(1) decode steps."
    """
    B, T, _ = x.shape
    win = params.window_size
    ratio = params.compress_ratio
    Dh = params.head_dim
    rd = params.rope_head_dim
    eps = params.norm_eps

    # SWA kv (matches attention_prefill's kv computation).
    kv = _linear(x, params.wkv)
    kv = rms_norm(kv, params.kv_norm_w, eps)
    fc = freqs_cis_full[:T] if T > 0 else freqs_cis_full[:0]
    if T > 0:
        kv = splice_rope(kv, rd, fc, inverse=False)
    kv = kv.astype(dtype)

    extra = (cfg_max_seq_len // ratio) if ratio else 0
    kv_cache = jnp.zeros((B, win + extra, Dh), dtype=dtype)
    swa = _swa_kv_cache_from_prefill(kv, win)
    kv_cache = kv_cache.at[:, :win, :].set(swa)

    if ratio > 0:
        # Compressed positions [win, win + T//ratio) come from compressor_prefill.
        kv_compressed = compressor_prefill(x, params.compressor, freqs_cis_full).astype(dtype)
        Tcomp = kv_compressed.shape[1]  # = T // ratio
        if Tcomp > 0:
            kv_cache = kv_cache.at[:, win:win + Tcomp, :].set(kv_compressed)
        c_kv, c_sc = _compressor_state_from_prefill(x, params.compressor)
    else:
        c_kv = jnp.zeros((B, 0, 0), dtype=jnp.float32)
        c_sc = jnp.full((B, 0, 0), -jnp.inf, dtype=jnp.float32)

    if ratio == 4 and params.indexer is not None:
        # Indexer state: same compressor logic on params.indexer.compressor.
        i_kv, i_sc = _compressor_state_from_prefill(x, params.indexer.compressor)
        # indexer_kv_cache: [B, max/ratio, index_head_dim] with [:T//ratio]
        # populated from the indexer's compressor_prefill output.
        max_iidx = cfg_max_seq_len // ratio
        i_cache = jnp.zeros((B, max_iidx, cfg_index_head_dim), dtype=dtype)
        if T >= ratio:
            idx_kv_compressed = compressor_prefill(
                x, params.indexer.compressor, freqs_cis_full).astype(dtype)
            Ti = idx_kv_compressed.shape[1]
            i_cache = i_cache.at[:, :Ti, :].set(idx_kv_compressed)
    else:
        i_kv = jnp.zeros((B, 0, 0), dtype=jnp.float32)
        i_sc = jnp.full((B, 0, 0), -jnp.inf, dtype=jnp.float32)
        i_cache = jnp.zeros((B, 0, 0), dtype=dtype)

    return AttentionDecodeState(
        kv_cache=kv_cache,
        compressor_kv_state=c_kv,
        compressor_score_state=c_sc,
        indexer_kv_state=i_kv,
        indexer_score_state=i_sc,
        indexer_kv_cache=i_cache,
    )

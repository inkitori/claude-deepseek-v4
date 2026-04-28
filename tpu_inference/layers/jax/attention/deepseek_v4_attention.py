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


# --------------------- general helpers ---------------------

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
    score = score.reshape(B, cutoff // ratio, ratio, coff * d) + params.ape  # [ratio, coff*d] broadcasts
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
    Returns [B, S, window_size] int32, with -1 for masked entries."""
    base = jnp.arange(seqlen)[:, None]               # [S, 1]
    matrix = jnp.maximum(base - window_size + 1, 0) + jnp.arange(min(seqlen, window_size))
    matrix = jnp.where(matrix > base, -1, matrix)    # [S, window_size]
    return jnp.broadcast_to(matrix[None, :, :], (bsz, seqlen, window_size)).astype(jnp.int32)


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

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pure-PyTorch replacements for the tilelang kernels in
/mnt/scratch/v4_pro/inference/kernel.py.

Only the math-relevant kernels are replicated. FP4/FP8 GEMMs are NOT replicated
(they only fire when weights are stored in those dtypes; for tests with random
BF16 weights, F.linear is used directly inside `linear()`).

Reference behavior is taken from the kernel docstrings and from the
sparse_attn_kernel and hc_split_sinkhorn_kernel sources.
"""
from typing import Optional

import torch
import torch.nn.functional as F


def act_quant_noop(x: torch.Tensor, block_size: int = 128, scale_fmt=None,
                   scale_dtype=torch.float32, inplace: bool = False):
    """Stub for `act_quant`. With `inplace=True`, the real kernel does
    quant→dequant in-place, simulating FP8 round-trip noise; we leave x
    unchanged, matching the no-op semantics documented in DECISIONS.md D2.
    Returns (x, scales) when `inplace=False` (used at GEMM call sites for
    real FP8 weight matmul — not exercised in our tests).
    """
    if inplace:
        return None
    # Fallback for completeness. Block-wise abs-max quantization to fp8.
    M = x.shape[:-1].numel()
    N = x.shape[-1]
    flat = x.reshape(M, N).float()
    n_blocks = (N + block_size - 1) // block_size
    scales = torch.empty(M, n_blocks, dtype=scale_dtype)
    out = torch.empty_like(flat, dtype=torch.float32)
    for j in range(n_blocks):
        s, e = j * block_size, min(j * block_size + block_size, N)
        amax = flat[:, s:e].abs().max(dim=-1, keepdim=True).values.clamp_min(1e-4)
        scale = amax / 448.0
        scales[:, j:j + 1] = scale
        out[:, s:e] = flat[:, s:e] / scale
    return out.reshape(x.shape).to(torch.float8_e4m3fn) if hasattr(torch, "float8_e4m3fn") else out, scales


def fp4_act_quant_noop(x: torch.Tensor, fp4_block_size: int = 32,
                       inplace: bool = False):
    """Stub for `fp4_act_quant`. We only ever use this with `inplace=True`,
    which is a no-op in our setting (DECISIONS.md D2)."""
    if inplace:
        return None
    return x, None


def sparse_attn_torch(q: torch.Tensor, kv: torch.Tensor,
                      attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                      softmax_scale: float) -> torch.Tensor:
    """Pure-PyTorch implementation of sparse multi-head attention with
    learnable sink. Reference: sparse_attn_kernel in
    /mnt/scratch/v4_pro/inference/kernel.py.

    q: [B, M, H, D] BF16
    kv: [B, N, D]   BF16   (single KV head shared across all H query heads)
    attn_sink: [H] FP32
    topk_idxs: [B, M, K] int (entries == -1 mean "ignore this slot")
    softmax_scale: float

    For each (b, m), compute attention against `kv[b, topk_idxs[b, m]]` only,
    masking out -1 slots, with an extra `exp(attn_sink[h] - max)` term added
    to the softmax denominator (as the sink contribution).
    """
    B, M, H, D = q.shape
    K = topk_idxs.shape[-1]
    # Compute on float32 to match kernel's online-softmax FP32 accumulation.
    q_fp = q.float()  # [B, M, H, D]
    kv_fp = kv.float()  # [B, N, D]
    # Build masked KV gather: -1 indices map to a zero-vector and -inf logit.
    # gather KV: shape [B, M, K, D]
    safe_idx = topk_idxs.clamp(min=0).long()
    # kv_gathered[b, m, k] = kv[b, safe_idx[b, m, k]]
    kv_gathered = torch.gather(
        kv_fp.unsqueeze(1).expand(B, M, kv_fp.shape[1], D),
        2,
        safe_idx.unsqueeze(-1).expand(B, M, K, D),
    )  # [B, M, K, D]
    # logits: [B, M, H, K]
    logits = torch.einsum("bmhd,bmkd->bmhk", q_fp, kv_gathered) * softmax_scale
    # Mask out -1 slots.
    valid = (topk_idxs != -1)  # [B, M, K]
    logits = torch.where(valid.unsqueeze(2), logits, torch.full_like(logits, float("-inf")))
    # Numerically stable softmax with sink contribution.
    m_max = torch.maximum(
        logits.max(dim=-1, keepdim=True).values,
        attn_sink.view(1, 1, H, 1).float(),
    )
    # Replace any all-(-inf) rows' max with sink so we don't get NaN: when the
    # max is -inf and sink is also small, set max=0 to avoid (-inf - -inf).
    m_max = torch.where(torch.isfinite(m_max), m_max, torch.zeros_like(m_max))
    p = torch.exp(logits - m_max)  # zeros where logit=-inf
    p = torch.where(valid.unsqueeze(2), p, torch.zeros_like(p))
    sink = torch.exp(attn_sink.view(1, 1, H, 1).float() - m_max)
    denom = p.sum(dim=-1, keepdim=True) + sink  # [B, M, H, 1]
    p = p / denom  # [B, M, H, K]
    out = torch.einsum("bmhk,bmkd->bmhd", p, kv_gathered).to(q.dtype)
    return out


def hc_split_sinkhorn_torch(mixes: torch.Tensor, hc_scale: torch.Tensor,
                            hc_base: torch.Tensor, hc_mult: int = 4,
                            sinkhorn_iters: int = 20, eps: float = 1e-6):
    """Pure-PyTorch port of `hc_split_sinkhorn_kernel`. Returns (pre, post, comb).

    mixes: [N, mix_hc] FP32 where N is flattened batch/seq and mix_hc=(2+hc_mult)*hc_mult
    hc_scale: [3] FP32
    hc_base: [mix_hc] FP32

    Returns:
      pre:  [N, hc_mult]            = sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
      post: [N, hc_mult]            = 2 * sigmoid(mixes[:, hc:2hc] * scale[1] + base[hc:2hc])
      comb: [N, hc_mult, hc_mult]   = doubly-stochastic-ish via Sinkhorn iterations on
                                      reshape(mixes[:, 2hc:] * scale[2] + base[2hc:], (hc, hc)).
    """
    N, _ = mixes.shape
    H = hc_mult
    pre_lin = mixes[:, :H] * hc_scale[0] + hc_base[:H]
    pre = torch.sigmoid(pre_lin) + eps
    post_lin = mixes[:, H:2 * H] * hc_scale[1] + hc_base[H:2 * H]
    post = 2.0 * torch.sigmoid(post_lin)
    comb_lin = mixes[:, 2 * H:] * hc_scale[2] + hc_base[2 * H:]
    comb = comb_lin.reshape(N, H, H)
    # Initial step: row-softmax + eps, then col-normalize with eps.
    comb = F.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    # iters-1 more rounds of (row-norm, col-norm).
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb

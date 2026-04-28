# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pure-PyTorch reference for DeepSeek-V4. Sourced from the official DeepSeek
inference/model.py (copy at /mnt/scratch/v4_pro/inference/model.py) and patched
to run on CPU without tilelang/CUDA.

Modifications vs. upstream:
  - `world_size`/`rank` are forced to 1 (no torch.distributed).
  - Linear() uses torch.float32 weights for FP32-typed params (matches upstream).
    For non-fp32 weight dtypes other than bf16 we fall back to bf16 storage and
    a normal F.linear (no FP4/FP8 GEMMs).
  - Custom kernels are replaced by pure-PyTorch equivalents from kernel_stubs:
      * sparse_attn -> sparse_attn_torch
      * hc_split_sinkhorn -> hc_split_sinkhorn_torch
      * act_quant(inplace=True), fp4_act_quant(inplace=True) -> no-op
      * rotate_activation -> identity (Hadamard rotation; see DECISIONS.md D3).

The math should be bit-similar to the upstream reference up to floating-point
accumulation order, with all FP4/FP8 quantization disabled.
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .kernel_stubs import (act_quant_noop, fp4_act_quant_noop,
                           hc_split_sinkhorn_torch, sparse_attn_torch)

WORLD_SIZE = 1
RANK = 0
default_dtype = torch.bfloat16
scale_fmt = None
scale_dtype = torch.float32


@contextmanager
def set_dtype(dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@dataclass
class ModelArgs:
    """Field names match V4 config.json keys (with PyTorch ref aliases). All FP4
    / FP8 quantization options are accepted but ignored — see DECISIONS.md D2."""
    max_batch_size: int = 4
    max_seq_len: int = 256
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 1024
    dim: int = 256
    moe_inter_dim: int = 64
    n_layers: int = 6
    n_hash_layers: int = 1
    n_mtp_layers: int = 1
    n_heads: int = 4
    n_kv_heads: int = 1  # informational (V4 always 1 KV head)
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 2.5
    swiglu_limit: float = 0.0
    # mqa
    q_lora_rank: int = 64
    head_dim: int = 32
    rope_head_dim: int = 16
    norm_eps: float = 1e-6
    o_groups: int = 2
    o_lora_rank: int = 64
    window_size: int = 8
    # length must be n_layers + n_mtp_layers (last entry is for the MTP block).
    compress_ratios: Tuple[int, ...] = (0, 0, 4, 128, 4, 0, 0)
    # yarn / rope
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1
    # index
    index_n_heads: int = 4
    index_head_dim: int = 16
    index_topk: int = 8
    # hc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6


# --------------------- core building blocks ---------------------

class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


def linear(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    """No-op around F.linear. Real reference dispatches to FP4/FP8 GEMMs based
    on weight dtype, which we don't exercise here (DECISIONS.md D2)."""
    return F.linear(x, weight, bias)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Honor caller-requested dtype for fp32 params; otherwise force bf16.
        if dtype == torch.float32:
            stored_dtype = torch.float32
        else:
            stored_dtype = torch.bfloat16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=stored_dtype))
        self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__(in_features, out_features, bias, dtype)


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__(in_features, out_features, bias, dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(8)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """In-place RoPE on the trailing rope_head_dim of x (treated as complex pairs)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x.to(y.dtype))
    return y


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Identity. See DECISIONS.md D3 — Hadamard rotation does not change topk
    rankings, and we do not run real fast_hadamard_transform."""
    return x


# --------------------- compressor / indexer ---------------------

@lru_cache(2)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size),
                            torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 32, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap
        self.coff = coff

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: Optional[torch.Tensor] = None
        self.register_buffer(
            "kv_state",
            torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full((args.max_batch_size, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32),
            persistent=False,
        )
        self.freqs_cis: Optional[torch.Tensor] = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio: cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio: cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset: offset + remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset: offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score = score + self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant_noop(kv, 32, True)
        else:
            act_quant_noop(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads  # world_size=1
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio
        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim),
            persistent=False,
        )
        self.freqs_cis: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        fp4_act_quant_noop(q, 32, True)
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        index_score = torch.einsum("bshd,btd->bsht", q.float(), self.kv_cache[:bsz, :end_pos // ratio].float())
        index_score = (index_score.relu_() * weights.float().unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            index_score = index_score + torch.where(mask, float("-inf"), 0.0)
        topk = min(self.index_topk, end_pos // ratio)
        if topk > 0:
            topk_idxs = index_score.topk(topk, dim=-1)[1]
        else:
            topk_idxs = index_score.new_zeros(bsz, seqlen, 0, dtype=torch.long)
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs


# --------------------- attention ---------------------

class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * args.o_lora_rank, dtype=torch.bfloat16)
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None

        kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim),
            persistent=False,
        )
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim, args.max_seq_len, original_seq_len,
            rope_theta, args.rope_factor, args.beta_fast, args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        # q
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps).to(q.dtype)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # kv
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant_noop(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # write to kv_cache and concat for prefill
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn_torch(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = sparse_attn_torch(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # o projection
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float())
        x = self.wo_b(o.flatten(2).to(x.dtype))
        return x


# --------------------- moe ---------------------

class Gate(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids].long()
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit=0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)
        self.w2 = Linear(inter_dim, dim, dtype=dtype)
        self.w3 = Linear(dim, inter_dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = 0
        self.experts_end_idx = self.n_local_experts
        self.gate = Gate(layer_id, args)
        # Always allocate experts in bf16 for the reference (DECISIONS.md D2).
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim, dtype=None, swiglu_limit=args.swiglu_limit)
            for _ in range(self.n_routed_experts)
        ])
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, dtype=None, swiglu_limit=args.swiglu_limit)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] = y[idx] + expert(x[idx], weights[idx, top, None].to(x.dtype)).float()
        y = y + self.shared_experts(x).float()
        return y.type_as(x).view(shape)


# --------------------- block ---------------------

class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        mixes_flat = mixes.reshape(-1, mixes.size(-1))
        pre, post, comb = hc_split_sinkhorn_torch(
            mixes_flat, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps,
        )
        pre = pre.view(*shape[:2], self.hc_mult)
        post = post.view(*shape[:2], self.hc_mult)
        comb = comb.view(*shape[:2], self.hc_mult, self.hc_mult)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class ParallelHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x_f = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_f.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_f, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.float().view(shape), dim=2)
        return y.to(dtype)

    def get_logits_all(self, x):
        return F.linear(x.float(), self.weight)

    def forward(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: RMSNorm):
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        return self.get_logits_all(norm(x))


class MTPBlock(Block):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        self.e_proj = Linear(args.dim, args.dim)
        self.h_proj = Linear(args.dim, args.dim)
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.embed: Optional[ParallelEmbedding] = None
        self.head: Optional[ParallelHead] = None

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed is not None and self.head is not None
        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)
        return self.head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)


# --------------------- transformer ---------------------

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        self.mtp = nn.ModuleList()
        for layer_id in range(args.n_mtp_layers):
            self.mtp.append(MTPBlock(args.n_layers + layer_id, args))
            self.mtp[-1].embed = self.embed
            self.mtp[-1].head = self.head
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, return_logits_all: bool = True):
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        return self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)

    def reset_state(self):
        """Zero out all per-layer kv caches and compressor state buffers. Call
        between independent inputs in tests."""
        for m in self.modules():
            if hasattr(m, "kv_cache") and isinstance(m.kv_cache, torch.Tensor):
                m.kv_cache.zero_()
            if hasattr(m, "kv_state") and isinstance(m.kv_state, torch.Tensor):
                m.kv_state.zero_()
            if hasattr(m, "score_state") and isinstance(m.score_state, torch.Tensor):
                m.score_state.fill_(float("-inf"))


def init_model_random(model: Transformer, seed: int = 0):
    """Fill all parameters with deterministic small random values matching the
    dtype each parameter declared. Buffers are left as their initial values."""
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if p.dtype == torch.int32:
            # tid2eid lookup table (n_routed_experts ranges)
            p.data.copy_(torch.randint(0, model.args.n_routed_experts, p.shape, generator=g, dtype=torch.int32))
        else:
            t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02, generator=g)
            p.data.copy_(t.to(p.dtype))

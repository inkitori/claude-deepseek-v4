# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DeepSeek-V4 model assembly for the JAX TPU backend.

Functional core (`deepseek_v4_forward_prefill` / `deepseek_v4_mtp_forward`)
operates on a flat parameter dict — easy to test, easy to compile.

A thin `DeepseekV4` JAX class wraps the parameter dict into a structure
compatible with the existing `tpu_inference` model registry, so vLLM can
construct it via `--model deepseek-ai/DeepSeek-V4-Pro` and `--model
deepseek-ai/DeepSeek-V4-Flash`. The class deliberately does NOT use
`tpu_inference.kernels.ragged_paged_attention` — it computes attention with
fully materialized topk-sparse softmax (correctness > performance; see
DECISIONS.md D5).

Modules and submodules:
  * `Transformer.embed → ParallelEmbedding` (rep across mesh)
  * `Transformer.layers[i] → Block` (mHC + Attention + MoE)
  * `Transformer.head → ParallelHead` (HC mixer + linear)
  * `Transformer.mtp[0] → MTPBlock`

See V3_TO_V4_DIFF.md for what changed vs DeepSeekV3.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from tpu_inference.layers.jax.attention.deepseek_v4_attention import (
    AttentionParams, CompressorParams, IndexerParams,
    attention_prefill, hc_split_sinkhorn, precompute_freqs_cis, rms_norm,
    splice_rope,
)
from tpu_inference.layers.jax.moe.deepseek_v4_moe import (
    ExpertParams, GateParams, MoEParams, gate_forward, moe_forward,
)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class DeepseekV4Config:
    """JAX-side mirror of the V4 HuggingFace config. Field names follow the
    HF config.json keys; aliases match the inference/model.py ModelArgs."""
    vocab_size: int
    hidden_size: int
    intermediate_size: int  # only used by MTP if it had a dense FFN — V4 doesn't
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    num_hash_layers: int
    num_nextn_predict_layers: int
    sliding_window: int
    swiglu_limit: float
    score_func: str
    routed_scaling_factor: float
    rms_norm_eps: float
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    rope_beta_fast: int
    rope_beta_slow: int
    rope_original_seq_len: int
    max_position_embeddings: int
    compress_ratios: Tuple[int, ...]   # length = num_hidden_layers + num_nextn_predict_layers
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float

    @classmethod
    def from_hf_dict(cls, d: Dict[str, Any]) -> "DeepseekV4Config":
        rs = d.get("rope_scaling") or {}
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d.get("intermediate_size", d.get("moe_intermediate_size", 0)),
            moe_intermediate_size=d["moe_intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d.get("num_key_value_heads", 1),
            head_dim=d["head_dim"],
            qk_rope_head_dim=d["qk_rope_head_dim"],
            q_lora_rank=d["q_lora_rank"],
            o_lora_rank=d["o_lora_rank"],
            o_groups=d["o_groups"],
            n_routed_experts=d["n_routed_experts"],
            n_shared_experts=d["n_shared_experts"],
            num_experts_per_tok=d["num_experts_per_tok"],
            num_hash_layers=d.get("num_hash_layers", 0),
            num_nextn_predict_layers=d.get("num_nextn_predict_layers", 0),
            sliding_window=d.get("sliding_window", 128),
            swiglu_limit=d.get("swiglu_limit", 0.0),
            score_func=d.get("scoring_func", "sqrtsoftplus"),
            routed_scaling_factor=d.get("routed_scaling_factor", 1.0),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            rope_theta=d.get("rope_theta", 10000.0),
            compress_rope_theta=d.get("compress_rope_theta", 160000.0),
            rope_factor=rs.get("factor", 1.0),
            rope_beta_fast=rs.get("beta_fast", 32),
            rope_beta_slow=rs.get("beta_slow", 1),
            rope_original_seq_len=rs.get("original_max_position_embeddings", 0),
            max_position_embeddings=d["max_position_embeddings"],
            compress_ratios=tuple(d["compress_ratios"]),
            index_n_heads=d["index_n_heads"],
            index_head_dim=d["index_head_dim"],
            index_topk=d["index_topk"],
            hc_mult=d.get("hc_mult", 4),
            hc_sinkhorn_iters=d.get("hc_sinkhorn_iters", 20),
            hc_eps=d.get("hc_eps", 1e-6),
        )

    @property
    def expected_compress_ratios_len(self) -> int:
        return self.num_hidden_layers + self.num_nextn_predict_layers

    def __post_init__(self):
        # Sanity: compress_ratios length must match num_hidden_layers + n_mtp.
        # If config.json provides only num_hidden_layers entries we extend.
        if len(self.compress_ratios) == self.num_hidden_layers:
            self.compress_ratios = self.compress_ratios + (0,) * self.num_nextn_predict_layers
        assert len(self.compress_ratios) >= self.expected_compress_ratios_len, (
            f"compress_ratios length {len(self.compress_ratios)} < expected "
            f"{self.expected_compress_ratios_len}")


# ------------------------------------------------------------
# Param-tree types
# ------------------------------------------------------------

@dataclass
class BlockParams:
    attn: AttentionParams
    moe: MoEParams
    attn_norm_w: jnp.ndarray
    ffn_norm_w: jnp.ndarray
    hc_attn_fn: jnp.ndarray
    hc_ffn_fn: jnp.ndarray
    hc_attn_base: jnp.ndarray
    hc_ffn_base: jnp.ndarray
    hc_attn_scale: jnp.ndarray
    hc_ffn_scale: jnp.ndarray
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    norm_eps: float


@dataclass
class MTPBlockParams:
    block: BlockParams
    e_proj: jnp.ndarray   # [dim, dim]
    h_proj: jnp.ndarray   # [dim, dim]
    enorm_w: jnp.ndarray
    hnorm_w: jnp.ndarray
    final_norm_w: jnp.ndarray
    hc_head_fn: jnp.ndarray   # [hc_mult, hc_mult*dim]
    hc_head_base: jnp.ndarray  # [hc_mult]
    hc_head_scale: jnp.ndarray  # [1]


@dataclass
class TransformerParams:
    embed_w: jnp.ndarray
    layers: List[BlockParams]
    final_norm_w: jnp.ndarray
    head_w: jnp.ndarray
    hc_head_fn: jnp.ndarray
    hc_head_base: jnp.ndarray
    hc_head_scale: jnp.ndarray
    mtp: List[MTPBlockParams]
    hc_mult: int


# ------------------------------------------------------------
# Block math
# ------------------------------------------------------------

def hc_pre(
    x: jnp.ndarray,            # [B, S, hc, D] (residual stream)
    hc_fn: jnp.ndarray,        # [(2+hc)*hc, hc*D] fp32
    hc_scale: jnp.ndarray,     # [3] fp32
    hc_base: jnp.ndarray,      # [(2+hc)*hc] fp32
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (y, post, comb).
      y    [B, S, D]      fed into attn or ffn
      post [B, S, hc]     applied to sub-output in hc_post
      comb [B, S, hc, hc] applied to residual stream in hc_post
    """
    dtype = x.dtype
    B, S, hc, D = x.shape
    xf = x.reshape(B, S, hc * D).astype(jnp.float32)
    rsqrt = lax.rsqrt(jnp.square(xf).mean(-1, keepdims=True) + norm_eps)
    mixes = (xf @ hc_fn.T) * rsqrt
    mixes_flat = mixes.reshape(-1, mixes.shape[-1])
    pre, post, comb = hc_split_sinkhorn(
        mixes_flat, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps,
    )
    pre = pre.reshape(B, S, hc)
    post = post.reshape(B, S, hc)
    comb = comb.reshape(B, S, hc, hc)
    # y = sum over hc dim of (pre[..., None] * x)  -> [B, S, D]
    y = jnp.sum(pre[..., None] * x.astype(jnp.float32), axis=2)
    return y.astype(dtype), post, comb


def hc_post(
    x: jnp.ndarray,        # [B, S, D] (sub-output)
    residual: jnp.ndarray,  # [B, S, hc, D]
    post: jnp.ndarray,      # [B, S, hc]
    comb: jnp.ndarray,      # [B, S, hc, hc]
) -> jnp.ndarray:
    """Returns [B, S, hc, D]. Reference: see V3_TO_V4_DIFF.md and the upstream
    `Block.hc_post`.

    Math (fp32 throughout to match upstream):
      y[b,s,j,d] = post[b,s,j] * x[b,s,d]
                 + sum_i comb[b,s,i,j] * residual[b,s,i,d]
    """
    dtype = x.dtype
    xf = x.astype(jnp.float32)
    rf = residual.astype(jnp.float32)
    pf = post.astype(jnp.float32)
    cf = comb.astype(jnp.float32)
    a = pf[..., None] * xf[..., None, :]
    b = jnp.einsum("bsij,bsid->bsjd", cf, rf)
    return (a + b).astype(dtype)


def block_forward(
    x: jnp.ndarray,           # [B, S, hc, D]
    input_ids: jnp.ndarray,
    params: BlockParams,
    freqs_cis_full: jnp.ndarray,
) -> jnp.ndarray:
    """Block forward, prefill only."""
    # attn sub-step
    residual = x
    y, post, comb = hc_pre(
        x, params.hc_attn_fn, params.hc_attn_scale, params.hc_attn_base,
        params.hc_mult, params.hc_sinkhorn_iters, params.norm_eps, params.hc_eps,
    )
    y = rms_norm(y, params.attn_norm_w, params.norm_eps)
    y = attention_prefill(y, params.attn, freqs_cis_full)
    x = hc_post(y, residual, post, comb)

    # ffn sub-step
    residual = x
    y, post, comb = hc_pre(
        x, params.hc_ffn_fn, params.hc_ffn_scale, params.hc_ffn_base,
        params.hc_mult, params.hc_sinkhorn_iters, params.norm_eps, params.hc_eps,
    )
    y = rms_norm(y, params.ffn_norm_w, params.norm_eps)
    y = moe_forward(y, input_ids, params.moe)
    return hc_post(y, residual, post, comb)


# ------------------------------------------------------------
# Head
# ------------------------------------------------------------

def head_hc(
    x: jnp.ndarray,            # [B, S, hc, D]
    hc_fn: jnp.ndarray,        # [hc, hc*D] fp32
    hc_scale: jnp.ndarray,     # [1] fp32
    hc_base: jnp.ndarray,      # [hc] fp32
    norm_eps: float,
    hc_eps: float,
) -> jnp.ndarray:
    """Sigmoid-gated HC mix used by `ParallelHead.hc_head` and
    `MTPBlock.hc_head_fn` / scale / base. Returns [B, S, D]."""
    dtype = x.dtype
    B, S, hc, D = x.shape
    xf = x.reshape(B, S, hc * D).astype(jnp.float32)
    rsqrt = lax.rsqrt(jnp.square(xf).mean(-1, keepdims=True) + norm_eps)
    mixes = (xf @ hc_fn.T) * rsqrt
    pre = jax.nn.sigmoid(mixes * hc_scale + hc_base) + hc_eps  # [B, S, hc]
    y = jnp.sum(pre[..., None] * x.astype(jnp.float32), axis=2)
    return y.astype(dtype)


def head_forward(
    x: jnp.ndarray,            # [B, S, hc, D]
    head_w: jnp.ndarray,       # [vocab, D] fp32
    final_norm_w: jnp.ndarray,
    hc_head_fn: jnp.ndarray,
    hc_head_scale: jnp.ndarray,
    hc_head_base: jnp.ndarray,
    norm_eps: float,
    hc_eps: float,
) -> jnp.ndarray:
    """Returns logits [B, S, vocab_size] (fp32)."""
    x = head_hc(x, hc_head_fn, hc_head_scale, hc_head_base, norm_eps, hc_eps)
    x = rms_norm(x, final_norm_w, norm_eps)
    return (x.astype(jnp.float32) @ head_w.T)


# ------------------------------------------------------------
# Top-level prefill
# ------------------------------------------------------------

def deepseek_v4_forward_prefill(
    input_ids: jnp.ndarray,        # [B, S] int32
    params: TransformerParams,
    freqs_cis_swa: jnp.ndarray,    # plain rope_theta freqs
    freqs_cis_compressed: jnp.ndarray,  # compress_rope_theta freqs (YaRN)
    cfg: DeepseekV4Config,
) -> jnp.ndarray:
    """Returns logits [B, S, vocab_size]."""
    h = params.embed_w[input_ids]  # [B, S, D]
    h = jnp.broadcast_to(h[:, :, None, :], (*h.shape[:2], cfg.hc_mult, h.shape[-1]))
    for li, layer in enumerate(params.layers):
        # Attention layer's freqs depends on its compress_ratio.
        cr = layer.attn.compress_ratio
        fc = freqs_cis_compressed if cr > 0 else freqs_cis_swa
        h = block_forward(h, input_ids, layer, fc)
    return head_forward(
        h, params.head_w, params.final_norm_w,
        params.hc_head_fn, params.hc_head_scale, params.hc_head_base,
        cfg.rms_norm_eps, cfg.hc_eps,
    )


def deepseek_v4_mtp_forward(
    h: jnp.ndarray,                # [B, S, hc, D] from main stack
    input_ids: jnp.ndarray,        # [B, S] int32
    mtp_params: MTPBlockParams,
    embed_w: jnp.ndarray,
    head_w: jnp.ndarray,
    freqs_cis_swa: jnp.ndarray,
    freqs_cis_compressed: jnp.ndarray,
    cfg: DeepseekV4Config,
) -> jnp.ndarray:
    """MTP block forward. Returns logits [B, S, vocab_size]."""
    # e = embed(input_ids); enorm; hnorm; combine.
    e = embed_w[input_ids]
    e = rms_norm(e, mtp_params.enorm_w, cfg.rms_norm_eps)
    h_normed = rms_norm(h, mtp_params.hnorm_w, cfg.rms_norm_eps)
    # e_proj: [dim, dim]; h_proj: [dim, dim]
    e_proj = (e.astype(jnp.float32) @ mtp_params.e_proj.astype(jnp.float32).T).astype(e.dtype)
    h_proj = (h_normed.astype(jnp.float32) @ mtp_params.h_proj.astype(jnp.float32).T).astype(h.dtype)
    h = e_proj[:, :, None, :] + h_proj
    # Run the inner Block.
    cr = mtp_params.block.attn.compress_ratio
    fc = freqs_cis_compressed if cr > 0 else freqs_cis_swa
    h = block_forward(h, input_ids, mtp_params.block, fc)
    # Head (uses MTP's own hc_head_fn/scale/base, the parent head's weight,
    # and MTP's own final norm).
    h = head_hc(h, mtp_params.hc_head_fn, mtp_params.hc_head_scale,
                mtp_params.hc_head_base, cfg.rms_norm_eps, cfg.hc_eps)
    h = rms_norm(h, mtp_params.final_norm_w, cfg.rms_norm_eps)
    return (h.astype(jnp.float32) @ head_w.T)


# ------------------------------------------------------------
# Helpers for tests / weight loading
# ------------------------------------------------------------

def make_freqs_cis(cfg: DeepseekV4Config, max_seq_len: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pre-compute the two RoPE freq tables: SWA (plain) and compressed (YaRN)."""
    swa = precompute_freqs_cis(
        cfg.qk_rope_head_dim, max_seq_len, 0, cfg.rope_theta,
        cfg.rope_factor, cfg.rope_beta_fast, cfg.rope_beta_slow,
    )
    compressed = precompute_freqs_cis(
        cfg.qk_rope_head_dim, max_seq_len, cfg.rope_original_seq_len,
        cfg.compress_rope_theta, cfg.rope_factor,
        cfg.rope_beta_fast, cfg.rope_beta_slow,
    )
    return swa, compressed

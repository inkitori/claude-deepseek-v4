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


# ------------------------------------------------------------
# Abstract param-tree construction (for Tier 3 compile-only tests)
# ------------------------------------------------------------

def _shape_struct(shape, dtype) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(tuple(int(s) for s in shape), dtype)


def make_abstract_attention_params(cfg: DeepseekV4Config, layer_id: int) -> AttentionParams:
    """Returns an AttentionParams populated with `jax.ShapeDtypeStruct`s
    matching the real param shapes for `layer_id`. No allocation."""
    H = cfg.num_attention_heads
    Dh = cfg.head_dim
    rd = cfg.qk_rope_head_dim
    G = cfg.o_groups
    R = cfg.o_lora_rank
    qLR = cfg.q_lora_rank
    cr = cfg.compress_ratios[layer_id]
    bf16 = jnp.bfloat16
    fp32 = jnp.float32

    compressor = None
    indexer = None
    if cr > 0:
        coff = 2 if cr == 4 else 1
        compressor = CompressorParams(
            ape=_shape_struct((cr, coff * Dh), fp32),
            wkv=_shape_struct((coff * Dh, cfg.hidden_size), fp32),
            wgate=_shape_struct((coff * Dh, cfg.hidden_size), fp32),
            norm_w=_shape_struct((Dh,), fp32),
            head_dim=Dh,
            rope_head_dim=rd,
            compress_ratio=cr,
            norm_eps=cfg.rms_norm_eps,
            rotate=False,
        )
        if cr == 4:
            ic = CompressorParams(
                ape=_shape_struct((cr, coff * cfg.index_head_dim), fp32),
                wkv=_shape_struct((coff * cfg.index_head_dim, cfg.hidden_size), fp32),
                wgate=_shape_struct((coff * cfg.index_head_dim, cfg.hidden_size), fp32),
                norm_w=_shape_struct((cfg.index_head_dim,), fp32),
                head_dim=cfg.index_head_dim,
                rope_head_dim=rd,
                compress_ratio=cr,
                norm_eps=cfg.rms_norm_eps,
                rotate=True,
            )
            indexer = IndexerParams(
                wq_b=_shape_struct((cfg.index_n_heads * cfg.index_head_dim, qLR), bf16),
                weights_proj=_shape_struct((cfg.index_n_heads, cfg.hidden_size), bf16),
                compressor=ic,
                n_heads=cfg.index_n_heads,
                head_dim=cfg.index_head_dim,
                rope_head_dim=rd,
                index_topk=cfg.index_topk,
                softmax_scale=cfg.index_head_dim ** -0.5,
                norm_eps=cfg.rms_norm_eps,
            )
    return AttentionParams(
        attn_sink=_shape_struct((H,), fp32),
        wq_a=_shape_struct((qLR, cfg.hidden_size), bf16),
        q_norm_w=_shape_struct((qLR,), fp32),
        wq_b=_shape_struct((H * Dh, qLR), bf16),
        wkv=_shape_struct((Dh, cfg.hidden_size), bf16),
        kv_norm_w=_shape_struct((Dh,), fp32),
        wo_a=_shape_struct((G * R, (H * Dh) // G), bf16),
        wo_b=_shape_struct((cfg.hidden_size, G * R), bf16),
        n_heads=H,
        head_dim=Dh,
        rope_head_dim=rd,
        n_groups=G,
        o_lora_rank=R,
        window_size=cfg.sliding_window,
        compress_ratio=cr,
        norm_eps=cfg.rms_norm_eps,
        softmax_scale=Dh ** -0.5,
        compressor=compressor,
        indexer=indexer,
    )


def make_abstract_moe_params(cfg: DeepseekV4Config, layer_id: int) -> MoEParams:
    bf16 = jnp.bfloat16
    fp32 = jnp.float32
    is_hash = layer_id < cfg.num_hash_layers
    gate = GateParams(
        weight=_shape_struct((cfg.n_routed_experts, cfg.hidden_size), fp32),
        bias=None if is_hash else _shape_struct((cfg.n_routed_experts,), fp32),
        tid2eid=_shape_struct((cfg.vocab_size, cfg.num_experts_per_tok), jnp.int32) if is_hash else None,
        score_func=cfg.score_func,
        route_scale=cfg.routed_scaling_factor,
        top_k=cfg.num_experts_per_tok,
    )
    expert_dtype = bf16  # See DECISIONS.md D2 — we treat all experts as bf16 (real V4 is fp4, dequantized at load).
    expert_template = lambda: ExpertParams(
        w1=_shape_struct((cfg.moe_intermediate_size, cfg.hidden_size), expert_dtype),
        w2=_shape_struct((cfg.hidden_size, cfg.moe_intermediate_size), expert_dtype),
        w3=_shape_struct((cfg.moe_intermediate_size, cfg.hidden_size), expert_dtype),
        swiglu_limit=cfg.swiglu_limit,
    )
    experts = [expert_template() for _ in range(cfg.n_routed_experts)]
    shared = expert_template()
    return MoEParams(
        gate=gate,
        experts=experts,
        shared_expert=shared,
        n_routed_experts=cfg.n_routed_experts,
        dim=cfg.hidden_size,
    )


def make_abstract_block_params(cfg: DeepseekV4Config, layer_id: int) -> BlockParams:
    fp32 = jnp.float32
    mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
    hc_dim = cfg.hc_mult * cfg.hidden_size
    return BlockParams(
        attn=make_abstract_attention_params(cfg, layer_id),
        moe=make_abstract_moe_params(cfg, layer_id),
        attn_norm_w=_shape_struct((cfg.hidden_size,), fp32),
        ffn_norm_w=_shape_struct((cfg.hidden_size,), fp32),
        hc_attn_fn=_shape_struct((mix_hc, hc_dim), fp32),
        hc_ffn_fn=_shape_struct((mix_hc, hc_dim), fp32),
        hc_attn_base=_shape_struct((mix_hc,), fp32),
        hc_ffn_base=_shape_struct((mix_hc,), fp32),
        hc_attn_scale=_shape_struct((3,), fp32),
        hc_ffn_scale=_shape_struct((3,), fp32),
        hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
        norm_eps=cfg.rms_norm_eps,
    )


def make_abstract_transformer_params(cfg: DeepseekV4Config) -> TransformerParams:
    bf16 = jnp.bfloat16
    fp32 = jnp.float32
    layers = [make_abstract_block_params(cfg, i) for i in range(cfg.num_hidden_layers)]
    mtp_blocks = []
    for i in range(cfg.num_nextn_predict_layers):
        layer_id = cfg.num_hidden_layers + i
        block = make_abstract_block_params(cfg, layer_id)
        mtp_blocks.append(MTPBlockParams(
            block=block,
            e_proj=_shape_struct((cfg.hidden_size, cfg.hidden_size), bf16),
            h_proj=_shape_struct((cfg.hidden_size, cfg.hidden_size), bf16),
            enorm_w=_shape_struct((cfg.hidden_size,), fp32),
            hnorm_w=_shape_struct((cfg.hidden_size,), fp32),
            final_norm_w=_shape_struct((cfg.hidden_size,), fp32),
            hc_head_fn=_shape_struct((cfg.hc_mult, cfg.hc_mult * cfg.hidden_size), fp32),
            hc_head_base=_shape_struct((cfg.hc_mult,), fp32),
            hc_head_scale=_shape_struct((1,), fp32),
        ))
    return TransformerParams(
        embed_w=_shape_struct((cfg.vocab_size, cfg.hidden_size), bf16),
        layers=layers,
        final_norm_w=_shape_struct((cfg.hidden_size,), fp32),
        head_w=_shape_struct((cfg.vocab_size, cfg.hidden_size), fp32),
        hc_head_fn=_shape_struct((cfg.hc_mult, cfg.hc_mult * cfg.hidden_size), fp32),
        hc_head_base=_shape_struct((cfg.hc_mult,), fp32),
        hc_head_scale=_shape_struct((1,), fp32),
        mtp=mtp_blocks,
        hc_mult=cfg.hc_mult,
    )


# ------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------

def count_param_bytes(params_struct: TransformerParams) -> int:
    """Total bytes of all parameters when using the dtypes stored in the
    ShapeDtypeStruct tree. Each parameter contributes prod(shape) * itemsize."""
    total = 0
    for leaf in jax.tree_util.tree_leaves(params_struct, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct)):
        if isinstance(leaf, jax.ShapeDtypeStruct):
            total += int(jnp.dtype(leaf.dtype).itemsize) * int(_prod(leaf.shape))
    return total


def _prod(xs):
    p = 1
    for v in xs:
        p *= int(v)
    return p


def kv_cache_bytes_per_layer(cfg: DeepseekV4Config, max_seq_len: int, dtype_bytes: int = 2) -> Dict[str, int]:
    """Per-layer KV cache size for a single sequence at `max_seq_len`. Reports
    SWA-only, SWA+CSA-compressed, and SWA+HCA-compressed sizes separately so
    the caller can sum the right ones based on `cfg.compress_ratios`."""
    Dh = cfg.head_dim
    out = {}
    out["swa_only"] = cfg.sliding_window * Dh * dtype_bytes
    if 4 in cfg.compress_ratios:
        out["csa_layer"] = (cfg.sliding_window + max_seq_len // 4) * Dh * dtype_bytes
    if 128 in cfg.compress_ratios:
        out["hca_layer"] = (cfg.sliding_window + max_seq_len // 128) * Dh * dtype_bytes
    return out


# ------------------------------------------------------------
# Pytree registration so jax.eval_shape can traverse our dataclasses
# ------------------------------------------------------------

def _register_pytree(cls, fields):
    def flatten(obj):
        children = tuple(getattr(obj, f) for f in fields)
        # static metadata: any non-array fields of the dataclass.
        meta = tuple((f, getattr(obj, f)) for f in obj.__dataclass_fields__ if f not in fields)
        return children, meta

    def unflatten(meta, children):
        kw = dict(zip(fields, children))
        for k, v in meta:
            kw[k] = v
        return cls(**kw)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)


# Register dataclasses with jnp.ndarray / ShapeDtypeStruct fields. Static
# metadata (ints, floats, strings, etc.) is preserved across tree_map.
_register_pytree(CompressorParams,
                 ("ape", "wkv", "wgate", "norm_w"))
_register_pytree(IndexerParams,
                 ("wq_b", "weights_proj", "compressor"))
_register_pytree(AttentionParams,
                 ("attn_sink", "wq_a", "q_norm_w", "wq_b", "wkv",
                  "kv_norm_w", "wo_a", "wo_b", "compressor", "indexer"))
_register_pytree(GateParams,
                 ("weight", "bias", "tid2eid"))
_register_pytree(ExpertParams,
                 ("w1", "w2", "w3"))
_register_pytree(MoEParams,
                 ("gate", "experts", "shared_expert"))
_register_pytree(BlockParams,
                 ("attn", "moe", "attn_norm_w", "ffn_norm_w",
                  "hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base",
                  "hc_attn_scale", "hc_ffn_scale"))
_register_pytree(MTPBlockParams,
                 ("block", "e_proj", "h_proj", "enorm_w", "hnorm_w",
                  "final_norm_w", "hc_head_fn", "hc_head_base", "hc_head_scale"))
_register_pytree(TransformerParams,
                 ("embed_w", "layers", "final_norm_w", "head_w",
                  "hc_head_fn", "hc_head_base", "hc_head_scale", "mtp"))


# ------------------------------------------------------------
# HF safetensors name → JAX param-tree path mapping (Tier 4)
# ------------------------------------------------------------

# Mapping schema:
#   key   : a (regex, jax-path-template) pair
#   regex : Python regex matching the HF parameter name (groups: layer, expert)
#   path  : a list of (segment-template, kind) where kind is 'attr', 'index'
#           and segment-template may use {layer}, {expert} placeholders.
#
# We build this mapping once and use it both to validate that every name in
# the HF index has a destination in our param tree (Tier 4) and as the
# basis for a real-weight loader (out of scope for this PR — see
# PROD_TOPOLOGY_RISKS.md item 6).

import re

# Each entry: (regex, jax-path-template-string).
# In path templates, {L} = layer index, {E} = expert index, {M} = mtp index.
_HF_TO_JAX_RULES = [
    # Top-level
    (re.compile(r"^embed\.weight$"), "embed_w"),
    (re.compile(r"^head\.weight$"), "head_w"),
    (re.compile(r"^norm\.weight$"), "final_norm_w"),
    (re.compile(r"^hc_head_fn$"), "hc_head_fn"),
    (re.compile(r"^hc_head_base$"), "hc_head_base"),
    (re.compile(r"^hc_head_scale$"), "hc_head_scale"),
    # Layer mHC
    (re.compile(r"^layers\.(?P<L>\d+)\.hc_attn_fn$"), "layers[{L}].hc_attn_fn"),
    (re.compile(r"^layers\.(?P<L>\d+)\.hc_ffn_fn$"), "layers[{L}].hc_ffn_fn"),
    (re.compile(r"^layers\.(?P<L>\d+)\.hc_attn_base$"), "layers[{L}].hc_attn_base"),
    (re.compile(r"^layers\.(?P<L>\d+)\.hc_ffn_base$"), "layers[{L}].hc_ffn_base"),
    (re.compile(r"^layers\.(?P<L>\d+)\.hc_attn_scale$"), "layers[{L}].hc_attn_scale"),
    (re.compile(r"^layers\.(?P<L>\d+)\.hc_ffn_scale$"), "layers[{L}].hc_ffn_scale"),
    # Layer norms
    (re.compile(r"^layers\.(?P<L>\d+)\.attn_norm\.weight$"), "layers[{L}].attn_norm_w"),
    (re.compile(r"^layers\.(?P<L>\d+)\.ffn_norm\.weight$"), "layers[{L}].ffn_norm_w"),
    # Layer attn core
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.attn_sink$"), "layers[{L}].attn.attn_sink"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.wq_a\.weight$"), "layers[{L}].attn.wq_a"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.wq_b\.weight$"), "layers[{L}].attn.wq_b"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.q_norm\.weight$"), "layers[{L}].attn.q_norm_w"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.wkv\.weight$"), "layers[{L}].attn.wkv"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.kv_norm\.weight$"), "layers[{L}].attn.kv_norm_w"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.wo_a\.weight$"), "layers[{L}].attn.wo_a"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.wo_b\.weight$"), "layers[{L}].attn.wo_b"),
    # Layer compressor
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.compressor\.ape$"), "layers[{L}].attn.compressor.ape"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.compressor\.norm\.weight$"), "layers[{L}].attn.compressor.norm_w"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.compressor\.wkv\.weight$"), "layers[{L}].attn.compressor.wkv"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.compressor\.wgate\.weight$"), "layers[{L}].attn.compressor.wgate"),
    # Layer indexer
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.indexer\.wq_b\.weight$"), "layers[{L}].attn.indexer.wq_b"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.indexer\.weights_proj\.weight$"), "layers[{L}].attn.indexer.weights_proj"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.indexer\.compressor\.ape$"), "layers[{L}].attn.indexer.compressor.ape"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.indexer\.compressor\.norm\.weight$"), "layers[{L}].attn.indexer.compressor.norm_w"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.indexer\.compressor\.wkv\.weight$"), "layers[{L}].attn.indexer.compressor.wkv"),
    (re.compile(r"^layers\.(?P<L>\d+)\.attn\.indexer\.compressor\.wgate\.weight$"), "layers[{L}].attn.indexer.compressor.wgate"),
    # Layer ffn (MoE) gate
    (re.compile(r"^layers\.(?P<L>\d+)\.ffn\.gate\.weight$"), "layers[{L}].moe.gate.weight"),
    (re.compile(r"^layers\.(?P<L>\d+)\.ffn\.gate\.bias$"), "layers[{L}].moe.gate.bias"),
    (re.compile(r"^layers\.(?P<L>\d+)\.ffn\.gate\.tid2eid$"), "layers[{L}].moe.gate.tid2eid"),
    # Layer ffn experts
    (re.compile(r"^layers\.(?P<L>\d+)\.ffn\.experts\.(?P<E>\d+)\.w1\.weight$"), "layers[{L}].moe.experts[{E}].w1"),
    (re.compile(r"^layers\.(?P<L>\d+)\.ffn\.experts\.(?P<E>\d+)\.w2\.weight$"), "layers[{L}].moe.experts[{E}].w2"),
    (re.compile(r"^layers\.(?P<L>\d+)\.ffn\.experts\.(?P<E>\d+)\.w3\.weight$"), "layers[{L}].moe.experts[{E}].w3"),
    # Layer ffn shared experts
    (re.compile(r"^layers\.(?P<L>\d+)\.ffn\.shared_experts\.w1\.weight$"), "layers[{L}].moe.shared_expert.w1"),
    (re.compile(r"^layers\.(?P<L>\d+)\.ffn\.shared_experts\.w2\.weight$"), "layers[{L}].moe.shared_expert.w2"),
    (re.compile(r"^layers\.(?P<L>\d+)\.ffn\.shared_experts\.w3\.weight$"), "layers[{L}].moe.shared_expert.w3"),
    # MTP block — layout mirrors layers but rooted at mtp[M]
    (re.compile(r"^mtp\.(?P<M>\d+)\.attn\.attn_sink$"), "mtp[{M}].block.attn.attn_sink"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.attn\.wq_a\.weight$"), "mtp[{M}].block.attn.wq_a"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.attn\.wq_b\.weight$"), "mtp[{M}].block.attn.wq_b"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.attn\.q_norm\.weight$"), "mtp[{M}].block.attn.q_norm_w"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.attn\.wkv\.weight$"), "mtp[{M}].block.attn.wkv"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.attn\.kv_norm\.weight$"), "mtp[{M}].block.attn.kv_norm_w"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.attn\.wo_a\.weight$"), "mtp[{M}].block.attn.wo_a"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.attn\.wo_b\.weight$"), "mtp[{M}].block.attn.wo_b"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.attn_norm\.weight$"), "mtp[{M}].block.attn_norm_w"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.ffn_norm\.weight$"), "mtp[{M}].block.ffn_norm_w"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.hc_attn_fn$"), "mtp[{M}].block.hc_attn_fn"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.hc_ffn_fn$"), "mtp[{M}].block.hc_ffn_fn"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.hc_attn_base$"), "mtp[{M}].block.hc_attn_base"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.hc_ffn_base$"), "mtp[{M}].block.hc_ffn_base"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.hc_attn_scale$"), "mtp[{M}].block.hc_attn_scale"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.hc_ffn_scale$"), "mtp[{M}].block.hc_ffn_scale"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.ffn\.gate\.weight$"), "mtp[{M}].block.moe.gate.weight"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.ffn\.gate\.bias$"), "mtp[{M}].block.moe.gate.bias"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.ffn\.experts\.(?P<E>\d+)\.w1\.weight$"), "mtp[{M}].block.moe.experts[{E}].w1"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.ffn\.experts\.(?P<E>\d+)\.w2\.weight$"), "mtp[{M}].block.moe.experts[{E}].w2"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.ffn\.experts\.(?P<E>\d+)\.w3\.weight$"), "mtp[{M}].block.moe.experts[{E}].w3"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.ffn\.shared_experts\.w1\.weight$"), "mtp[{M}].block.moe.shared_expert.w1"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.ffn\.shared_experts\.w2\.weight$"), "mtp[{M}].block.moe.shared_expert.w2"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.ffn\.shared_experts\.w3\.weight$"), "mtp[{M}].block.moe.shared_expert.w3"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.e_proj\.weight$"), "mtp[{M}].e_proj"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.h_proj\.weight$"), "mtp[{M}].h_proj"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.enorm\.weight$"), "mtp[{M}].enorm_w"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.hnorm\.weight$"), "mtp[{M}].hnorm_w"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.norm\.weight$"), "mtp[{M}].final_norm_w"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.hc_head_fn$"), "mtp[{M}].hc_head_fn"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.hc_head_base$"), "mtp[{M}].hc_head_base"),
    (re.compile(r"^mtp\.(?P<M>\d+)\.hc_head_scale$"), "mtp[{M}].hc_head_scale"),
]

# Suffixes that indicate FP4/FP8 quantization scales — present alongside .weight
# in the HF checkpoint. The weight loader needs them paired with their .weight
# counterpart for dequantization. For Tier 4 we just verify they are recognized.
_QUANT_SUFFIXES = {".scale"}


# ------------------------------------------------------------
# vLLM model registry wrapper
# ------------------------------------------------------------
#
# This class makes `DeepseekV4ForCausalLM` discoverable via the
# tpu_inference model registry. It is intentionally a thin shim — the
# math is in the functional core above (block_forward / attention_prefill /
# moe_forward). The class exists primarily so that vLLM dispatch on the
# `DeepseekV4ForCausalLM` architecture string finds *something* registered
# and does NOT fall back to the vLLM-native PyTorch path. Full runtime
# integration (paged-KV plumbing, sharded weight loading with FP4/FP8
# dequantization, mesh-aware sharding annotations) is documented as
# Phase 7+ work in PROD_TOPOLOGY_RISKS.md item 7.

class DeepseekV4ForCausalLM:
    """Stub wrapper class for vLLM dispatch. Holds a parsed
    `DeepseekV4Config` and the abstract param tree for shape verification.

    The class is intentionally lightweight: it does NOT subclass `JaxModule`
    or own `nnx.Variable`s. Until the full vLLM-runtime integration lands,
    this class lives only to be importable + registry-discoverable.
    """
    def __init__(self, vllm_config, rng_key=None, mesh=None):
        self.vllm_config = vllm_config
        self.mesh = mesh
        # Build the V4 config from the HF config dict if available.
        hf_config = getattr(vllm_config.model_config, "hf_config", None)
        cfg_dict = None
        if hf_config is not None:
            # vLLM may store config either as a dict or as a HuggingFace
            # PretrainedConfig instance — try both.
            cfg_dict = (hf_config.to_dict() if hasattr(hf_config, "to_dict")
                        else dict(hf_config))
        if cfg_dict is None:
            raise ValueError("DeepseekV4ForCausalLM needs a model_config.hf_config")
        self.config = DeepseekV4Config.from_hf_dict(cfg_dict)
        # Abstract param tree — useful for shape probes and the weight-name
        # mapping. Real (allocated) params would be too large; we defer that
        # to a future weight loader.
        self.params = make_abstract_transformer_params(self.config)

    def map_weight_name(self, hf_name: str):
        """Returns the JAX param-tree path for an HF param name, or None."""
        return map_hf_name_to_jax_path(hf_name)

    def __call__(self, *args, **kwargs):
        # Defer to the functional forward. This signature would need to
        # match the rest of the tpu-inference runtime; for now it raises a
        # clear error so callers know what's missing.
        raise NotImplementedError(
            "DeepseekV4ForCausalLM.__call__ is not yet wired into the "
            "tpu_inference runtime. The functional forward path "
            "(`deepseek_v4_forward_prefill`) is fully tested and can be "
            "called directly with a TransformerParams pytree. See "
            "tests/models/jax/test_deepseek_v4.py for usage."
        )


def map_hf_name_to_jax_path(name: str) -> Optional[str]:
    """Returns the JAX param-tree path string for an HF parameter name, or
    None if no rule matches.

    Names ending in `.scale` (FP4/FP8 quantization scales) return the path of
    the corresponding `.weight` plus a ".scale" suffix — the caller must
    dequantize using the scale and then place the dequantized array at the
    base path. (For our Tier 4 smoke test we just verify name-coverage; we
    do NOT dequantize — see PROD_TOPOLOGY_RISKS.md item 6.)
    """
    base = name
    suffix = ""
    if name.endswith(".scale"):
        base = name[:-len(".scale")] + ".weight"
        suffix = "<scale>"
    for pat, path in _HF_TO_JAX_RULES:
        m = pat.match(base)
        if m:
            kw = {k.upper(): v for k, v in m.groupdict().items()}
            try:
                resolved = path.format(**kw)
            except KeyError:
                continue
            return resolved + suffix
    return None

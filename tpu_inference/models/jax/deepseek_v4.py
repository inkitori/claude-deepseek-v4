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

def transformer_body_forward(
    input_ids: jnp.ndarray,        # [B, S] int32
    params: TransformerParams,
    freqs_cis_swa: jnp.ndarray,
    freqs_cis_compressed: jnp.ndarray,
    cfg: DeepseekV4Config,
) -> jnp.ndarray:
    """Body-only prefill: embed → all layers → return [B, S, hc, D] residual stream.
    No final HC mix, no norm, no lm_head. Used by the nnx wrapper so it can
    return hidden states from __call__ and apply the head in compute_logits."""
    h = params.embed_w[input_ids]  # [B, S, D]
    h = jnp.broadcast_to(h[:, :, None, :], (*h.shape[:2], cfg.hc_mult, h.shape[-1]))
    for layer in params.layers:
        cr = layer.attn.compress_ratio
        fc = freqs_cis_compressed if cr > 0 else freqs_cis_swa
        h = block_forward(h, input_ids, layer, fc)
    return h


def deepseek_v4_forward_prefill(
    input_ids: jnp.ndarray,        # [B, S] int32
    params: TransformerParams,
    freqs_cis_swa: jnp.ndarray,    # plain rope_theta freqs
    freqs_cis_compressed: jnp.ndarray,  # compress_rope_theta freqs (YaRN)
    cfg: DeepseekV4Config,
) -> jnp.ndarray:
    """Returns logits [B, S, vocab_size]."""
    h = transformer_body_forward(
        input_ids, params, freqs_cis_swa, freqs_cis_compressed, cfg,
    )
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

def _materialize_param_tree(abstract_tree):
    """Replace every ShapeDtypeStruct leaf with jnp.zeros(shape, dtype)."""
    return jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype),
        abstract_tree,
        is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
    )


def _extract_hf_config_dict(vllm_config):
    """vLLM stores hf_config as either a PretrainedConfig or a SimpleNamespace.
    Both expose `.to_dict()` in our test harness. Real vllm uses
    PretrainedConfig.to_dict() too."""
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    if hf_config is None:
        raise ValueError("DeepseekV4ForCausalLM needs vllm_config.model_config.hf_config")
    if hasattr(hf_config, "to_dict"):
        return hf_config.to_dict()
    return dict(hf_config)


# Lazy nnx import — the wrapper class is created at import time but nnx
# initialization touches JAX which we want to defer behind the tpu_inference
# import path.
def _build_class():
    from flax import nnx
    from tpu_inference.layers.jax import JaxModule

    class DeepseekV4ForCausalLM(JaxModule):
        """Minimum-viable nnx.Module wrapper for vLLM dispatch.

        Subclasses `JaxModule` (== `nnx.Module`) so it survives
        `nnx.eval_shape(create_abstract_model)` in
        `tpu_inference/models/common/model_loader.py:244` (BLOCKERS B2).

        Storage strategy: the entire V4 parameter pytree
        (`TransformerParams` registered as a pytree above) is held inside a
        single `nnx.Param` whose `.value` is the dataclass tree. This
        sidesteps the V3-style "every weight is its own JaxEinsum/JaxRmsNorm"
        ceremony — V4's math runs through the existing functional core
        (`transformer_body_forward` / `head_forward`), which only needs the
        raw arrays. JaxAutoWeightsLoader-compatible per-weight Modules are
        deferred (see PROD_TOPOLOGY_RISKS item 7).

        Forward contract:
          * `__call__(kv_caches, input_ids, attention_metadata, ...)` runs
            the transformer body for the prefill case where there is exactly
            ONE sequence in the batch and `attention_metadata` describes a
            full prefill (every token is a new query at its own position).
            Returns `(kv_caches, hidden_TM, [])` with `hidden_TM` of shape
            `[T, hc * D]` (T = total tokens; M = hc*D).
          * `compute_logits(hidden_states)` reshapes back to `[T, hc, D]`
            and runs the V4 head (HC mix → final norm → matmul against
            `head_w`).

        What this wrapper INTENTIONALLY does NOT do:
          * Multi-sequence concurrent decode (per-batch state plumbing
            requires kv-cache schema changes — BLOCKERS B1).
          * paged-KV interoperation: `kv_caches` is passed through unchanged.
            V4's per-layer state lives in the model dataclass tree (no
            mutation across __call__ invocations).
          * Sharding annotations on individual params (V3 has them; V4
            does not yet — see PROD_TOPOLOGY_RISKS item 1).
        """

        def __init__(self, vllm_config, rng_key=None, mesh=None):
            cfg_dict = _extract_hf_config_dict(vllm_config)
            self.config = DeepseekV4Config.from_hf_dict(cfg_dict)
            self.vllm_config = vllm_config
            self.mesh = mesh
            # Build abstract param tree, then materialize each leaf as a
            # zero array. `nnx.eval_shape` will trace these to
            # `ShapeDtypeStruct` automatically.
            abs_tree = make_abstract_transformer_params(self.config)
            real_tree = _materialize_param_tree(abs_tree)
            # Holding the full tree inside one nnx.Param lets nnx walk the
            # registered TransformerParams pytree (see _register_pytree
            # block above) when computing state / partition specs.
            self.params_v = nnx.Param(real_tree)
            # Freqs are fp32 lookup tables; build them up-front so the
            # attribute slot is registered as data (an nnx.Variable) and
            # later mutations don't trip nnx's static/data sentinel.
            swa, comp = make_freqs_cis(self.config, self.config.max_position_embeddings)
            self._freqs_swa_v = nnx.Variable(swa)
            self._freqs_compressed_v = nnx.Variable(comp)

        # -- nnx housekeeping ----------------------------------------------

        def initialize_cache(self):
            """Pre-compute RoPE freq tables. Called by the loader after
            weights are populated. Idempotent."""
            swa, comp = make_freqs_cis(self.config, self.config.max_position_embeddings)
            self._freqs_swa_v = nnx.Variable(swa)
            self._freqs_compressed_v = nnx.Variable(comp)

        # -- public helpers (back-compat with v2 forward_prefill API) ------

        @property
        def params(self):
            """Convenience accessor — returns the underlying TransformerParams
            tree. Useful for tests that expect the v2 API."""
            return self.params_v.get_value()

        @params.setter
        def params(self, new_tree):
            self.params_v = nnx.Param(new_tree)

        def map_weight_name(self, hf_name: str):
            """Returns the JAX param-tree path for an HF param name, or None."""
            return map_hf_name_to_jax_path(hf_name)

        def load_weights_from_dir(self, checkpoint_dir: str):
            """Load real V4 weights from a checkpoint directory."""
            from tpu_inference.models.jax.deepseek_v4_loader import (
                apply_weights_to_param_tree, load_v4_safetensors_to_dict,
            )
            current = self.params_v.get_value()
            current = jax.tree_util.tree_map(
                lambda x: jnp.zeros(x.shape, dtype=x.dtype) if isinstance(x, jax.ShapeDtypeStruct) else x,
                current,
                is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
            )
            weights = load_v4_safetensors_to_dict(checkpoint_dir)
            new_tree = apply_weights_to_param_tree(current, weights, self.config)
            self.params_v = nnx.Param(new_tree)
            self.initialize_cache()

        def forward_prefill(self, input_ids: jnp.ndarray) -> jnp.ndarray:
            """Functional prefill helper — returns logits [B, S, vocab]."""
            return deepseek_v4_forward_prefill(
                input_ids, self.params_v.get_value(),
                self._freqs_swa_v.get_value(),
                self._freqs_compressed_v.get_value(),
                self.config,
            )

        # -- vLLM-runtime contract -----------------------------------------

        def __call__(
            self,
            kv_caches,
            input_ids,
            attention_metadata=None,
            inputs_embeds=None,
            _input_positions=None,
            _layer_name_to_kv_cache=None,
            _lora_metadata=None,
            intermediate_tensors=None,
            is_first_rank: bool = True,
            is_last_rank: bool = True,
            *args,
            **kwargs,
        ):
            """vLLM-runtime forward.

            Returns `(kv_caches, hidden_TM, [])`.

            For now this only correctly supports the **single-sequence
            prefill** case: input_ids is treated as one contiguous sequence
            of T tokens at positions [0, T). Multi-sequence batches and
            decode steps with start_pos>0 require the per-layer V4 state
            plumbing tracked in BLOCKERS B1.
            """
            ids = jnp.asarray(input_ids)
            if ids.ndim == 1:
                ids_2d = ids.reshape(1, -1)
            elif ids.ndim == 2:
                ids_2d = ids
            else:
                raise ValueError(f"input_ids must be 1D or 2D, got shape {ids.shape}")
            params = self.params_v.get_value()
            h = transformer_body_forward(
                ids_2d, params,
                self._freqs_swa_v.get_value(),
                self._freqs_compressed_v.get_value(),
                self.config,
            )
            # h: [B, S, hc, D]. vLLM expects (T, M) where T = B*S.
            B, S, hc, D = h.shape
            hidden_TM = h.reshape(B * S, hc * D)
            return kv_caches, hidden_TM, []

        def compute_logits(self, hidden_states):
            """Apply the V4 head to hidden states. hidden_states is [T, hc*D]
            (output of __call__) or [T_sampled, hc*D] (sampled subset)."""
            params = self.params_v.get_value()
            T = hidden_states.shape[0]
            hc = self.config.hc_mult
            D = self.config.hidden_size
            assert hidden_states.shape[-1] == hc * D, (
                f"compute_logits expected last dim {hc*D}, got {hidden_states.shape[-1]}"
            )
            h = hidden_states.reshape(1, T, hc, D)
            logits_BSV = head_forward(
                h, params.head_w, params.final_norm_w,
                params.hc_head_fn, params.hc_head_scale, params.hc_head_base,
                self.config.rms_norm_eps, self.config.hc_eps,
            )
            return logits_BSV.reshape(T, -1)

        def load_weights(self, weights=None, *args, **kwargs):
            """vLLM weight-loading entry. Streamed weights iterators are
            handled by the v4 loader — for now this method stays a no-op
            stub for callers that pass dummy/random weights. Real
            HF-streamer integration is BLOCKERS B5 (new in v4)."""
            # No-op: dummy / random init is already in place via __init__.
            return set()

    return DeepseekV4ForCausalLM


DeepseekV4ForCausalLM = _build_class()


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

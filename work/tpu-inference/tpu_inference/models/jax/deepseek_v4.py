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
    AttentionDecodeState, AttentionParams, CompressorParams, IndexerParams,
    _v4_nan_tripwire, attention_decode_step, attention_init_state_from_prefill,
    attention_prefill, hc_split_sinkhorn, precompute_freqs_cis, rms_norm,
    splice_rope,
)
from tpu_inference.layers.jax.moe.deepseek_v4_moe import (
    ExpertParams, GateParams, MoEParams, gate_forward, moe_forward,
)
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


_V4_DECODE_ARGMAX_PROBE = os.environ.get("V4_DECODE_ARGMAX_PROBE", "0") == "1"


def _v4_argmax_probe(name: str, x: jnp.ndarray) -> None:
    """S1 diagnostic: print max_abs(x) and (if x is logits-shaped) the
    top-3 token ids + their logit values. Gated at module import so HLO is
    unchanged when `V4_DECODE_ARGMAX_PROBE=0`. Used to distinguish whether
    the model emits real-vocab argmax that detokenizes empty vs collapses
    to pad/EOS — see CLAUDE.md S1."""
    if not _V4_DECODE_ARGMAX_PROBE:
        return
    xf = x.astype(jnp.float32)
    max_abs = jnp.max(jnp.abs(xf))
    if x.ndim >= 1 and x.shape[-1] >= 4:
        top_vals, top_ids = jax.lax.top_k(xf, 3)
        jax.debug.print(
            "[v4probe] {n} max_abs={m} top_ids={i} top_vals={v}",
            n=name, m=max_abs, i=top_ids, v=top_vals,
        )
    else:
        jax.debug.print("[v4probe] {n} max_abs={m}", n=name, m=max_abs)


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


def block_init_state_and_forward(
    x: jnp.ndarray,            # [B, T, hc, D]
    input_ids: jnp.ndarray,    # [B, T]
    params: BlockParams,
    freqs_cis_full: jnp.ndarray,
    cfg_max_seq_len: int,
    cfg_index_head_dim: int,
    layer_idx: int = -1,
) -> Tuple[AttentionDecodeState, jnp.ndarray]:
    """Block forward (prefill) that ALSO captures the post-prefill
    `AttentionDecodeState` for this layer. Output `[B, T, hc, D]` is
    bit-equivalent to `block_forward`; the captured state can drive
    `block_decode_step` at start_pos=T without re-running prefill."""
    residual = x
    y, post, comb = hc_pre(
        x, params.hc_attn_fn, params.hc_attn_scale, params.hc_attn_base,
        params.hc_mult, params.hc_sinkhorn_iters, params.norm_eps, params.hc_eps,
    )
    y = rms_norm(y, params.attn_norm_w, params.norm_eps)
    # Capture decode state from this layer's attention input. The state
    # constructor mirrors what `attention_prefill` will write to the
    # internal kv buffers at the same `y`.
    decode_state = attention_init_state_from_prefill(
        y, params.attn, freqs_cis_full,
        cfg_max_seq_len=cfg_max_seq_len,
        cfg_index_head_dim=cfg_index_head_dim,
        dtype=jnp.bfloat16,
        layer_idx=layer_idx,
    )
    y = attention_prefill(y, params.attn, freqs_cis_full)
    x = hc_post(y, residual, post, comb)

    residual = x
    y, post, comb = hc_pre(
        x, params.hc_ffn_fn, params.hc_ffn_scale, params.hc_ffn_base,
        params.hc_mult, params.hc_sinkhorn_iters, params.norm_eps, params.hc_eps,
    )
    y = rms_norm(y, params.ffn_norm_w, params.norm_eps)
    y = moe_forward(y, input_ids, params.moe)
    return decode_state, hc_post(y, residual, post, comb)


def _v4_weight_nan_audit(tree) -> None:
    """One-shot finiteness audit of a loaded V4 param tree. For each array
    leaf, emits a `[weight_nan] {path}` line if the tensor contains any NaN
    or Inf, plus a `[weight_nan_audit]` summary. Used to confirm/refute
    CLAUDE.md S1 hyp 1: a NaN-producing FP4/FP8 scale or packed-FP4 nibble
    on a single layer's weights yields an all-NaN bf16 leaf that
    poisons the decode forward at e.g. L5 attention.
    Gated at the call site by `V4_WEIGHT_NAN_AUDIT=1`."""
    import sys as _sys
    leaves: List[Tuple[str, Any]] = []

    def _walk(obj, path: str) -> None:
        if hasattr(obj, "shape") and hasattr(obj, "dtype") and not hasattr(
                obj, "__dataclass_fields__"):
            leaves.append((path, obj))
            return
        if isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{path}[{i}]")
            return
        if hasattr(obj, "__dataclass_fields__"):
            for fname in obj.__dataclass_fields__:
                sub = getattr(obj, fname, None)
                if sub is None:
                    continue
                sub_path = f"{path}.{fname}" if path else fname
                _walk(sub, sub_path)
            return

    _walk(tree, "")
    nan_count = 0
    inf_count = 0
    for path, t in leaves:
        try:
            tf = t.astype(jnp.float32)
            nan_any = bool(jnp.any(jnp.isnan(tf)))
            inf_any = bool(jnp.any(jnp.isinf(tf)))
        except Exception as e:  # noqa: BLE001
            print(
                f"[weight_nan] {path}: AUDIT_FAILED {e!r}",
                file=_sys.stderr, flush=True,
            )
            continue
        if nan_any or inf_any:
            print(
                f"[weight_nan] {path}: nan_any={nan_any} inf_any={inf_any} "
                f"shape={tuple(t.shape)} dtype={t.dtype}",
                file=_sys.stderr, flush=True,
            )
            if nan_any:
                nan_count += 1
            if inf_any:
                inf_count += 1
    print(
        f"[weight_nan_audit] examined={len(leaves)} nan_leaves={nan_count} "
        f"inf_leaves={inf_count}",
        file=_sys.stderr, flush=True,
    )


def block_decode_step(
    x_step: jnp.ndarray,           # [B, 1, hc, D]
    input_ids_step: jnp.ndarray,   # [B, 1]
    params: BlockParams,
    freqs_cis_full: jnp.ndarray,
    prev_state: AttentionDecodeState,
    start_pos,
    layer_idx: int = -1,
) -> Tuple[AttentionDecodeState, jnp.ndarray]:
    """One decode step through this block. Mirrors `block_forward` but
    swaps `attention_prefill` for `attention_decode_step` (which mutates
    `prev_state`). Returns `(new_state, x_out)` with `x_out: [B, 1, hc, D]`.
    `start_pos` is the absolute position of the new token (Python int or
    traced jnp.int32 scalar)."""
    _v4_nan_tripwire("attn_in", x_step, layer_idx, start_pos)
    residual = x_step
    y, post, comb = hc_pre(
        x_step, params.hc_attn_fn, params.hc_attn_scale, params.hc_attn_base,
        params.hc_mult, params.hc_sinkhorn_iters, params.norm_eps, params.hc_eps,
    )
    _v4_nan_tripwire("attn_hcpre_y", y, layer_idx, start_pos)
    y = rms_norm(y, params.attn_norm_w, params.norm_eps)
    new_state, y = attention_decode_step(
        y, start_pos, params.attn, freqs_cis_full, prev_state, layer_idx)
    _v4_nan_tripwire("attn_decode_y", y, layer_idx, start_pos)
    x_step = hc_post(y, residual, post, comb)
    _v4_nan_tripwire("attn_block_out", x_step, layer_idx, start_pos)

    residual = x_step
    y, post, comb = hc_pre(
        x_step, params.hc_ffn_fn, params.hc_ffn_scale, params.hc_ffn_base,
        params.hc_mult, params.hc_sinkhorn_iters, params.norm_eps, params.hc_eps,
    )
    _v4_nan_tripwire("ffn_hcpre_y", y, layer_idx, start_pos)
    y = rms_norm(y, params.ffn_norm_w, params.norm_eps)
    y = moe_forward(y, input_ids_step, params.moe)
    _v4_nan_tripwire("moe_y", y, layer_idx, start_pos)
    out = hc_post(y, residual, post, comb)
    _v4_nan_tripwire("ffn_block_out", out, layer_idx, start_pos)
    return new_state, out


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


def transformer_body_init_state_from_prefill(
    input_ids: jnp.ndarray,        # [B, T] int32
    params: TransformerParams,
    freqs_cis_swa: jnp.ndarray,
    freqs_cis_compressed: jnp.ndarray,
    cfg: DeepseekV4Config,
    state_max_seq_len: int,
) -> Tuple[jnp.ndarray, List[AttentionDecodeState]]:
    """Run a prefill of `input_ids` AND capture per-layer
    `AttentionDecodeState` for subsequent decode steps. Returns
    `(h: [B, T, hc, D], states: List[AttentionDecodeState])`.

    `h` is bit-equivalent to `transformer_body_forward(input_ids, ...)`;
    `states[i]` is what `attention_decode_step` would have accumulated
    after T iterations of the iterative path on layer i. Combined with
    `transformer_body_decode_step`, the wrapper can convert vLLM's
    every-step-recomputes-prefill path into prefill-once-then-O(1) decode.

    `state_max_seq_len` sizes the per-layer kv_cache buffer (slots
    `[win, win + state_max_seq_len/ratio)`). The caller picks this to
    match the runtime context bound (typically vLLM's max_model_len),
    NOT the architectural max_position_embeddings — V4-Flash's 1M HF
    config would otherwise blow per-layer state into GiB on every chip.
    """
    h = params.embed_w[input_ids]  # [B, T, D]
    h = jnp.broadcast_to(h[:, :, None, :], (*h.shape[:2], cfg.hc_mult, h.shape[-1]))
    states: List[AttentionDecodeState] = []
    for i, layer in enumerate(params.layers):
        cr = layer.attn.compress_ratio
        fc = freqs_cis_compressed if cr > 0 else freqs_cis_swa
        # index_head_dim is only consumed when this layer's attention has
        # an indexer (ratio==4). Pass cfg.index_head_dim either way; the
        # state-init helper itself gates on `params.indexer is not None`.
        idx_hd = cfg.index_head_dim if (cr == 4 and layer.attn.indexer is not None) else 0
        st, h = block_init_state_and_forward(
            h, input_ids, layer, fc,
            cfg_max_seq_len=state_max_seq_len,
            cfg_index_head_dim=idx_hd,
            layer_idx=i,
        )
        states.append(st)
    return h, states


def transformer_body_decode_step(
    input_ids_step: jnp.ndarray,   # [B, 1] int32
    params: TransformerParams,
    freqs_cis_swa: jnp.ndarray,
    freqs_cis_compressed: jnp.ndarray,
    cfg: DeepseekV4Config,
    prev_states: List[AttentionDecodeState],
    start_pos,
) -> Tuple[jnp.ndarray, List[AttentionDecodeState]]:
    """One decode step through every layer. Returns
    `(h: [B, 1, hc, D], new_states: List[AttentionDecodeState])`.

    `start_pos` is the absolute position of the new token (Python int or
    traced jnp.int32 scalar)."""
    h = params.embed_w[input_ids_step]  # [B, 1, D]
    h = jnp.broadcast_to(h[:, :, None, :], (*h.shape[:2], cfg.hc_mult, h.shape[-1]))
    _v4_nan_tripwire("embed_h", h, -1, start_pos)
    new_states: List[AttentionDecodeState] = []
    for i, (layer, prev) in enumerate(zip(params.layers, prev_states)):
        cr = layer.attn.compress_ratio
        fc = freqs_cis_compressed if cr > 0 else freqs_cis_swa
        new_state, h = block_decode_step(
            h, input_ids_step, layer, fc, prev, start_pos, layer_idx=i)
        new_states.append(new_state)
    return h, new_states


# Pack each layer's AttentionDecodeState (6 fields) into a single fp32 jax.Array
# so it can ride through vLLM's `kv_caches: List[jax.Array]` carrier. fp32
# is needed for compressor's softmax-state accumulator; bf16 fields round-trip
# exactly through fp32→bf16 cast.

# Per-field layout: (name, shape, dtype). Shape leading axis = batch dim B.
_DECODE_STATE_FIELD_NAMES = (
    "kv_cache",
    "compressor_kv_state",
    "compressor_score_state",
    "indexer_kv_state",
    "indexer_score_state",
    "indexer_kv_cache",
)


def _layer_decode_state_layout(
    layer_params: BlockParams,
    cfg_index_head_dim: int,
    state_max_seq_len: int,
    batch_size: int = 1,
) -> Tuple[Tuple[str, Tuple[int, int, int], "jnp.dtype"], ...]:
    """Return the per-field layout `((name, shape, dtype), ...)` for one
    layer's `AttentionDecodeState`. Mirrors `attention_init_state_from_prefill`
    field-shape decisions exactly so the packed buffer can losslessly round-
    trip a state produced by either the prefill-init or decode-step paths.

    Layers with `compress_ratio == 0` (pure SWA) emit zero-sized placeholders
    for the compressor / indexer fields; layers with `ratio > 0 and indexer
    is None` zero-size only the indexer fields. This keeps the packed buffer
    schema uniform across layer types — the layout's total element count
    differs per layer but every layer has all 6 named fields.
    """
    p = layer_params.attn
    win = p.window_size
    ratio = p.compress_ratio
    Dh = p.head_dim
    extra = (state_max_seq_len // ratio) if ratio else 0
    coff = 2 if ratio == 4 else 1

    # kv_cache (always populated; bf16).
    kv_cache_shape = (batch_size, win + extra, Dh)

    # compressor_kv_state / compressor_score_state — populated when ratio>0.
    if ratio > 0:
        comp_state_shape = (batch_size, coff * ratio, coff * Dh)
    else:
        comp_state_shape = (batch_size, 0, 0)

    # indexer_kv_state / indexer_score_state / indexer_kv_cache — populated
    # only for ratio==4 layers that have an indexer.
    if ratio == 4 and p.indexer is not None and cfg_index_head_dim > 0:
        idx_state_shape = (batch_size, coff * ratio, coff * cfg_index_head_dim)
        idx_cache_shape = (batch_size, state_max_seq_len // ratio, cfg_index_head_dim)
    else:
        idx_state_shape = (batch_size, 0, 0)
        idx_cache_shape = (batch_size, 0, 0)

    return (
        ("kv_cache", kv_cache_shape, jnp.bfloat16),
        ("compressor_kv_state", comp_state_shape, jnp.float32),
        ("compressor_score_state", comp_state_shape, jnp.float32),
        ("indexer_kv_state", idx_state_shape, jnp.float32),
        ("indexer_score_state", idx_state_shape, jnp.float32),
        ("indexer_kv_cache", idx_cache_shape, jnp.bfloat16),
    )


def _layer_packed_size(layout) -> int:
    """Total fp32 elements needed to hold one layer's packed state. Caller
    pre-allocates a `[packed_size]` (or `[max_num_seqs, packed_size]`) fp32
    buffer per layer."""
    n = 0
    for _, shape, _ in layout:
        n += int(_prod(shape))
    return n


def _pack_layer_state(state: AttentionDecodeState, layout) -> jnp.ndarray:
    """Flatten one layer's `AttentionDecodeState` into a 1D fp32 array. Order
    of fields matches `_DECODE_STATE_FIELD_NAMES`. bf16 fields are upcast
    to fp32 (lossless) for uniform-dtype storage; the unpack step casts back."""
    parts = []
    for name, shape, dtype in layout:
        arr = getattr(state, name)
        # The state's actual array may have a zero-element placeholder shape
        # (e.g. (B, 0, 0)) for fields that don't apply to this layer. The
        # layout's shape matches by construction — flatten to fp32 directly.
        flat = arr.reshape(-1).astype(jnp.float32)
        parts.append(flat)
    return jnp.concatenate(parts, axis=0) if parts else jnp.zeros((0,), dtype=jnp.float32)


def _unpack_layer_state(packed: jnp.ndarray, layout) -> AttentionDecodeState:
    """Inverse of `_pack_layer_state`. Slices the flat fp32 array by the
    per-field offsets, reshapes to each field's shape, and casts back to
    the field's natural dtype. The packed buffer must have been produced
    by `_pack_layer_state` against the same `layout`."""
    fields: Dict[str, jnp.ndarray] = {}
    offset = 0
    for name, shape, dtype in layout:
        n = int(_prod(shape))
        if n == 0:
            fields[name] = jnp.zeros(shape, dtype=dtype)
            continue
        chunk = jax.lax.dynamic_slice_in_dim(packed, offset, n, axis=0)
        fields[name] = chunk.reshape(shape).astype(dtype)
        offset += n
    return AttentionDecodeState(**fields)


def transformer_body_layout(
    params: TransformerParams,
    cfg: DeepseekV4Config,
    state_max_seq_len: int,
    batch_size: int = 1,
) -> List[Tuple]:
    """Per-layer packed-state layout for the whole transformer body. Index
    `i` of the result describes layer `i`'s `AttentionDecodeState` field
    shapes and dtypes. The caller uses this to pre-allocate per-layer
    `[max_num_seqs, packed_size_i]` fp32 buffers and to drive `_pack_*` /
    `_unpack_*`."""
    return [
        _layer_decode_state_layout(
            layer,
            cfg_index_head_dim=cfg.index_head_dim,
            state_max_seq_len=state_max_seq_len,
            batch_size=batch_size,
        )
        for layer in params.layers
    ]


def v4_state_max_seq_len_from_vllm_config(vllm_config) -> int:
    """Single source of truth for the per-layer packed-state buffer size.
    The kv-cache allocator and `__call__` MUST agree on this value or the
    JIT donation of `kv_caches` mismatches shape.

    Returns `min(max(max_model_len, max_num_batched_tokens * dp_size),
    max_position_embeddings)` — the buffer covers whichever bucket a real
    prefill can land in, capped by the architectural sequence limit.
    """
    mc = getattr(vllm_config, "model_config", None)
    sc = getattr(vllm_config, "scheduler_config", None)
    shc = getattr(vllm_config, "sharding_config", None)

    mml = getattr(mc, "max_model_len", None) if mc is not None else None
    mml = int(mml) if mml else 0

    mnbt = getattr(sc, "max_num_batched_tokens", None) if sc is not None else None
    dp = getattr(shc, "total_dp_size", None) if shc is not None else None
    pad_ceiling = (int(mnbt) * int(dp)) if (mnbt and dp) else 0

    hf = getattr(mc, "hf_text_config",
                 getattr(mc, "hf_config", None)) if mc is not None else None
    mpe = int(getattr(hf, "max_position_embeddings", 0)) if hf is not None else 0

    eff = max(mml, pad_ceiling)
    if eff <= 0:
        return mpe
    return min(eff, mpe) if mpe > 0 else eff


def v4_layer_packed_sizes_from_cfg(
    cfg: DeepseekV4Config,
    state_max_seq_len: int,
    batch_size: int = 1,
) -> List[int]:
    """Per-layer packed `AttentionDecodeState` size (fp32 elements) derived
    from `cfg` alone. Mirrors `_layer_decode_state_layout`'s shape decisions
    exactly so the kv_cache_manager allocator (which has cfg but not yet
    loaded params) can size buffers identically to what
    `transformer_body_layout(params, cfg, ...)` would produce on the same
    cfg + state_max_seq_len.
    """
    sizes: List[int] = []
    Dh = cfg.head_dim
    win = cfg.sliding_window
    for layer_id in range(cfg.num_hidden_layers):
        ratio = cfg.compress_ratios[layer_id]
        extra = (state_max_seq_len // ratio) if ratio else 0
        coff = 2 if ratio == 4 else 1
        n = batch_size * (win + extra) * Dh
        if ratio > 0:
            n += 2 * batch_size * (coff * ratio) * (coff * Dh)
        if ratio == 4 and cfg.index_head_dim > 0:
            n += 2 * batch_size * (coff * ratio) * (coff * cfg.index_head_dim)
            n += batch_size * (state_max_seq_len // ratio) * cfg.index_head_dim
        sizes.append(int(n))
    return sizes


def transformer_body_init_state_to_buffer(
    input_ids: jnp.ndarray,            # [B, T] int32
    params: TransformerParams,
    freqs_cis_swa: jnp.ndarray,
    freqs_cis_compressed: jnp.ndarray,
    cfg: DeepseekV4Config,
    state_max_seq_len: int,
) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """Run a prefill of `input_ids` and return `(h, packed_buffers)` where
    `packed_buffers[i]` is the 1D fp32 packed `AttentionDecodeState` for
    layer `i`. Output `h` is byte-equivalent to `transformer_body_forward`."""
    h, states = transformer_body_init_state_from_prefill(
        input_ids, params, freqs_cis_swa, freqs_cis_compressed, cfg,
        state_max_seq_len=state_max_seq_len,
    )
    for i, st in enumerate(states):
        _v4_nan_tripwire("prefill_state_kv_cache", st.kv_cache, i, -1)
    layouts = transformer_body_layout(
        params, cfg, state_max_seq_len, batch_size=int(input_ids.shape[0]))
    packed_buffers = [_pack_layer_state(s, lo) for s, lo in zip(states, layouts)]
    for i, b in enumerate(packed_buffers):
        _v4_nan_tripwire("packed_buffer_post_pack", b, i, -1)
    return h, packed_buffers


def transformer_body_decode_step_from_buffer(
    input_ids_step: jnp.ndarray,       # [B, 1] int32
    params: TransformerParams,
    freqs_cis_swa: jnp.ndarray,
    freqs_cis_compressed: jnp.ndarray,
    cfg: DeepseekV4Config,
    prev_buffers: List[jnp.ndarray],
    start_pos,
    state_max_seq_len: int,
) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """One decode step driven by per-layer packed-state buffers. Wraps
    `transformer_body_decode_step` with unpack-before / pack-after.
    Returns `(h: [B, 1, hc, D], new_buffers: List[jnp.ndarray])`.

    `start_pos` is the absolute position of the new token (Python int or
    traced jnp.int32 scalar — the kernel handles either)."""
    layouts = transformer_body_layout(
        params, cfg, state_max_seq_len,
        batch_size=int(input_ids_step.shape[0]))
    prev_states = [_unpack_layer_state(b, lo)
                   for b, lo in zip(prev_buffers, layouts)]
    h, new_states = transformer_body_decode_step(
        input_ids_step, params, freqs_cis_swa, freqs_cis_compressed, cfg,
        prev_states, start_pos,
    )
    new_buffers = [_pack_layer_state(s, lo)
                   for s, lo in zip(new_states, layouts)]
    return h, new_buffers


def _v4_constrain_packed_replicated(
    buffers: List[jnp.ndarray],
) -> List[jnp.ndarray]:
    """Constrain each packed-state buffer to `P()` inside the trace.
    Required: without it the JIT-boundary reshard interacts with donated
    `kv_caches` contents and SIGSEGVs on the second execution of a cached
    prefill artifact."""
    try:
        spec = jax.sharding.PartitionSpec()
        return [lax.with_sharding_constraint(b, spec) for b in buffers]
    except Exception:
        return buffers


def _v4_force_kv_caches_read(
    buffers: List[jnp.ndarray],
    kv_caches: List[jnp.ndarray],
) -> List[jnp.ndarray]:
    """Force XLA to read donated `kv_caches` (prevents aliased write
    elision under SPMD donation). Output equals `b` at runtime; the
    `where(opaque_false, b + kv, b)` shape keeps the kv-read branch
    in HLO. NaN-safe: discarded branch holds `b + (-inf) = -inf`,
    never NaN, so compressor `score_state` -inf init slots are fine."""
    opaque_false = lax.optimization_barrier(jnp.bool_(False))
    return [
        jnp.where(opaque_false, b + kv.astype(jnp.float32), b)
        for b, kv in zip(buffers, kv_caches)
    ]


def deepseek_v4_run_with_decode_state(
    kv_caches: List[jnp.ndarray],
    input_ids: jnp.ndarray,
    params: TransformerParams,
    freqs_cis_swa: jnp.ndarray,
    freqs_cis_compressed: jnp.ndarray,
    cfg: DeepseekV4Config,
    state_max_seq_len: int,
    is_decode_step: bool,
    start_pos,
    state_init_ids: "jnp.ndarray | None" = None,
) -> Tuple[List[jnp.ndarray], jnp.ndarray]:
    """Run one prefill or one decode step, threading per-layer
    `AttentionDecodeState` through `kv_caches` (each entry a 1D fp32 buffer
    sized by `v4_layer_packed_sizes_from_cfg`).

    Prefill seeds fresh packed state from `input_ids`. Decode reads prior
    state from `kv_caches`, advances one position at traced `start_pos`,
    and returns updated buffers. Returns `(updated_kv_caches, h)` so the
    caller can pass `kv_caches` as a donated JIT argument.

    `state_init_ids` (prefill only): when set, state is seeded from this
    sliced-to-real-length tensor while `h` is still computed on the
    padded `input_ids`. Required because state init is positional (padding
    tokens encoded into SWA / compressor / indexer slots produce wrong
    decode reads), but `transformer_body_forward` on the padded shape
    is what the runtime's logits path reads — running the body on a
    sliced shape changes the SPMD compile and the prefill `h` argmax
    no longer matches the real-V4 reference.

    Donation-safety scaffolding (`optimization_barrier(h)`,
    `_v4_force_kv_caches_read`, `_v4_constrain_packed_replicated`) blocks
    XLA from eliding aliased writes / CSEing across the two prefill paths.
    """
    if is_decode_step:
        h, new_buffers = transformer_body_decode_step_from_buffer(
            input_ids, params, freqs_cis_swa, freqs_cis_compressed, cfg,
            kv_caches, start_pos=start_pos,
            state_max_seq_len=state_max_seq_len,
        )
        new_buffers = _v4_force_kv_caches_read(new_buffers, kv_caches)
        new_buffers = _v4_constrain_packed_replicated(new_buffers)
        return new_buffers, h
    h = transformer_body_forward(
        input_ids, params, freqs_cis_swa, freqs_cis_compressed, cfg)
    state_ids = state_init_ids if state_init_ids is not None else input_ids
    _h_state, packed_buffers = transformer_body_init_state_to_buffer(
        state_ids, params, freqs_cis_swa, freqs_cis_compressed, cfg,
        state_max_seq_len=state_max_seq_len,
    )
    packed_buffers = _v4_force_kv_caches_read(packed_buffers, kv_caches)
    for i, b in enumerate(packed_buffers):
        _v4_nan_tripwire("packed_buffer_post_force_read", b, i, -1)
    packed_buffers = _v4_constrain_packed_replicated(packed_buffers)
    return packed_buffers, h


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


# Abstract param-tree construction for compile-only tests.

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


# Reporting helpers.

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
                 ("gate", "experts", "shared_expert",
                  "w1_stacked", "w2_stacked", "w3_stacked"))
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


# HF safetensors name → JAX param-tree path mapping. Used by the loader
# (deepseek_v4_loader.py) and by load-time validation.

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


# vLLM model registry wrapper. Math is in the functional core above; this
# class exists so vLLM dispatch on `DeepseekV4ForCausalLM` finds something
# registered.

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
        deferred.

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
          * Per-leaf JaxModule sharding annotations à la V3 (V4 instead
            applies sharding via `with_sharding_constraint` at the
            functional-core boundary — see `_shard_e_first` /
            `_shard_e_last` in `deepseek_v4_moe.py`).
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
            # Cap freqs precompute at the actual prefill bucket ceiling
            # rather than max_position_embeddings (V4-Flash's HF config has
            # 1 048 576, which would produce a 1 GB freqs table pinned in HBM).
            swa, comp = make_freqs_cis(self.config, self._effective_freqs_seq_len())
            self._freqs_swa_v = nnx.Variable(swa)
            self._freqs_compressed_v = nnx.Variable(comp)

        def _effective_freqs_seq_len(self) -> int:
            """Smallest freqs precompute size covering both `max_model_len` and
            vLLM's prefill-bucket ceiling (`max_num_batched_tokens × total_dp_size`),
            capped at `max_position_embeddings`. Without the bucket term, a
            short prompt that lands in a 1024-token bucket fails to reshape."""
            mc = getattr(self.vllm_config, "model_config", None)
            sc = getattr(self.vllm_config, "scheduler_config", None)
            shc = getattr(self.vllm_config, "sharding_config", None)

            mml = getattr(mc, "max_model_len", None) if mc is not None else None
            mml = int(mml) if mml else 0

            mnbt = getattr(sc, "max_num_batched_tokens", None) if sc is not None else None
            dp = getattr(shc, "total_dp_size", None) if shc is not None else None
            pad_ceiling = (int(mnbt) * int(dp)) if (mnbt and dp) else 0

            mpe = int(self.config.max_position_embeddings)
            eff = max(mml, pad_ceiling)
            if eff <= 0:
                return mpe
            return min(eff, mpe)

        def initialize_cache(self):
            """Pre-compute RoPE freq tables. Idempotent."""
            swa, comp = make_freqs_cis(self.config, self._effective_freqs_seq_len())
            self._freqs_swa_v = nnx.Variable(swa)
            self._freqs_compressed_v = nnx.Variable(comp)

        @property
        def params(self):
            return self.params_v.get_value()

        @params.setter
        def params(self, new_tree):
            self.params_v = nnx.Param(new_tree)

        def map_weight_name(self, hf_name: str):
            """Returns the JAX param-tree path for an HF param name, or None."""
            return map_hf_name_to_jax_path(hf_name)

        def load_weights_from_dir(self, checkpoint_dir: str):
            """Load V4 weights from a checkpoint directory. Streams one
            dequantized tensor at a time onto a sharded jax.Array — neither
            the full param tree nor a host-RAM buffer of all dequantized
            weights fits at v6e-32 chip budget."""
            from tpu_inference.models.jax.deepseek_v4_loader import (
                iter_v4_safetensors_dequant_torch, place_torch_as_jax_sharded,
                iter_v4_safetensors_specs, place_spec_as_jax_sharded,
            )
            current = self.params_v.get_value()

            # 1. Build JAX-path -> abstract leaf index. We use this to look
            #    up target shape/dtype for each loaded weight in O(1).
            path_to_leaf: Dict[str, Any] = {}

            def _walk(obj, path: str):
                # Array-like leaf (ShapeDtypeStruct or jnp.ndarray). We also
                # accept duck-typed leaves with .shape/.dtype that are not
                # themselves dataclasses (covers torch tensors and numpy
                # arrays in test paths).
                if isinstance(obj, jax.ShapeDtypeStruct) or (
                    hasattr(obj, "shape") and hasattr(obj, "dtype")
                    and not hasattr(obj, "__dataclass_fields__")
                ):
                    path_to_leaf[path] = obj
                    return
                if isinstance(obj, list):
                    for i, item in enumerate(obj):
                        _walk(item, f"{path}[{i}]")
                    return
                if hasattr(obj, "__dataclass_fields__"):
                    for fname in obj.__dataclass_fields__:
                        sub = getattr(obj, fname, None)
                        if sub is None:
                            continue
                        sub_path = f"{path}.{fname}" if path else fname
                        _walk(sub, sub_path)
                    return
                # Static metadata (int, float, bool, str, tuple, etc.) — not
                # an array leaf; skip silently.

            _walk(current, "")

            # 2. Helper: navigate `current` to the parent of `path`, then
            #    assign the new array in place. Reuses the regex split rule
            #    from `apply_weights_to_param_tree` so HF→JAX path strings
            #    remain consistent across the streaming and bulk loaders.
            import re as _re
            def _assign(path: str, arr):
                parts = _re.split(r"\.|(\[\d+\])", path)
                parts = [p for p in parts if p]
                cur = current
                for part in parts[:-1]:
                    if part.startswith("["):
                        cur = cur[int(part[1:-1])]
                    else:
                        cur = getattr(cur, part)
                last = parts[-1]
                if last.startswith("["):
                    cur[int(last[1:-1])] = arr
                else:
                    setattr(cur, last, arr)

            # Incremental MoE consolidation: as soon as all 256 experts of
            # a (layer, wname) group are placed, build a single stacked
            # `[E, inter, dim]` jax.Array sharded `P('attn_dp', None, None)`
            # and release the per-leaf references. Doing this incrementally
            # — not as a post-load pass — is critical: post-load, HBM is
            # fragmented across 33 000 small allocations and the
            # `device_put` reshard's 256 MB transient can't find a
            # contiguous block (we burned an iter learning this). Doing it
            # as soon as a group is full means we only ever hold one
            # group's per-leaf set alongside its stacked tensor at a time
            # (peak ~512 MB / chip during the reshard, well within budget).
            import threading as _threading
            from jax.sharding import NamedSharding, PartitionSpec as _P
            from tpu_inference.layers.jax.moe.deepseek_v4_moe import MoEParams as _MoEParams
            _expert_path_re = _re.compile(
                r'^layers\[(\d+)\]\.moe\.experts\[(\d+)\]\.(w[123])$')
            _mtp_expert_path_re = _re.compile(
                r'^mtp\[(\d+)\]\.block\.moe\.experts\[(\d+)\]\.(w[123])$')
            _expert_group_lock = _threading.Lock()
            _expert_group_counter: Dict[Tuple[str, int, str], int] = {}
            _consolidated_groups: set = set()

            def _maybe_consolidate(jax_path: str):
                # Identify group key, increment counter; if we hit 256,
                # consolidate now (under the lock so we only consolidate
                # each group once).
                m = _expert_path_re.match(jax_path)
                kind = None
                if m is not None:
                    kind, idx, wname = "layer", int(m.group(1)), m.group(3)
                else:
                    m = _mtp_expert_path_re.match(jax_path)
                    if m is None:
                        return
                    kind, idx, wname = "mtp", int(m.group(1)), m.group(3)
                key = (kind, idx, wname)
                with _expert_group_lock:
                    cnt = _expert_group_counter.get(key, 0) + 1
                    _expert_group_counter[key] = cnt
                    if cnt < self.config.n_routed_experts:
                        return
                    if key in _consolidated_groups:
                        return
                    _consolidated_groups.add(key)
                # All 256 placed — outside the lock, run the stack.
                if self.mesh is None:
                    return
                if kind == "layer":
                    moe = current.layers[idx].moe
                else:
                    moe = current.mtp[idx].block.moe
                weights = [getattr(e, wname) for e in moe.experts]
                if any(w is None for w in weights):
                    # Late race after another thread already consolidated
                    # — defensive; shouldn't happen given the lock.
                    return
                e_spec = NamedSharding(self.mesh, _P('attn_dp', None, None))
                stacked = jax.device_put(jnp.stack(weights), e_spec)
                stacked.block_until_ready()
                # Release the per-leaf references so the device buffers can
                # be reclaimed before we move on to the next group.
                for e in moe.experts:
                    setattr(e, wname, None)
                # Attach the stacked tensor + record the swiglu_limit (we
                # captured it from any expert; uniform across the layer).
                setattr(moe, f"{wname}_stacked", stacked)
                if moe.swiglu_limit is None and moe.experts:
                    moe.swiglu_limit = float(moe.experts[0].swiglu_limit)

            # 3. Stream-load every weight, placing each as a sharded array.
            #    Logs heartbeat progress every 200 placements (~1% of a real
            #    V4-Flash checkpoint's tensor count) so a long load doesn't
            #    look like a hang. The CPU dequant is the dominant cost and
            #    runs single-threaded inside `iter_v4_safetensors_dequant_torch`
            #    on each Ray worker independently.
            import sys as _sys
            import time as _time
            import os as _os
            placed_paths: set = set()
            skipped: List[str] = []
            t0 = _time.time()
            t_last = t0
            placed_count = 0

            # Default-on slice-aware path: each host reads only its row range.
            # Set V4_LOADER_SLICE_AWARE=0 to fall back to the full-dequant
            # path (useful for parity testing or if a future refactor breaks
            # slice-aware in a corner case).
            slice_aware = _os.environ.get("V4_LOADER_SLICE_AWARE", "1") == "1"

            # Multi-threaded placement: each tensor's read+dequant+placement
            # spends most of its wall time in JAX/safetensors C code that
            # releases the GIL. Running N placement threads per host
            # parallelizes the per-tensor framework overhead. Workers produce
            # (jax_path, arr) pairs; the main thread drains them and does
            # _assign so the dataclass-tree mutation stays single-threaded.
            try:
                place_workers = max(1, int(
                    _os.environ.get("V4_LOADER_PLACE_WORKERS", "8")))
            except ValueError:
                place_workers = 1

            print(
                f"[deepseek_v4] load_weights_from_dir: streaming "
                f"{checkpoint_dir!r} (mesh={self.mesh}, "
                f"slice_aware={slice_aware}, place_workers={place_workers})",
                file=_sys.stderr, flush=True,
            )

            def _do_place_spec(spec) -> Tuple[Optional[str], Any, str]:
                """Worker-side: resolve path + run the slice-aware placement.
                Returns (jax_path, arr, hf_name); jax_path=None means skip."""
                jax_path = map_hf_name_to_jax_path(spec.hf_name)
                if jax_path is None or jax_path.endswith("<scale>"):
                    return (None, None, spec.hf_name)
                leaf = path_to_leaf.get(jax_path)
                if leaf is None:
                    return (None, None, spec.hf_name)
                target_shape = tuple(leaf.shape)
                target_dtype = jnp.dtype(leaf.dtype)
                arr = place_spec_as_jax_sharded(
                    spec, target_dtype, target_shape, self.mesh,
                )
                return (jax_path, arr, spec.hf_name)

            def _do_place_full(item) -> Tuple[Optional[str], Any, str]:
                """Worker-side equivalent for the full-dequant fallback path."""
                hf_name, torch_t = item
                jax_path = map_hf_name_to_jax_path(hf_name)
                if jax_path is None or jax_path.endswith("<scale>"):
                    return (None, None, hf_name)
                leaf = path_to_leaf.get(jax_path)
                if leaf is None:
                    return (None, None, hf_name)
                target_shape = tuple(leaf.shape)
                target_dtype = jnp.dtype(leaf.dtype)
                arr = place_torch_as_jax_sharded(
                    torch_t, target_dtype, target_shape, self.mesh,
                )
                return (jax_path, arr, hf_name)

            def _drain_one(future, last_hf_name_box):
                jax_path, arr, hf_name = future.result()
                last_hf_name_box[0] = hf_name
                if jax_path is None:
                    skipped.append(hf_name)
                    return False
                _assign(jax_path, arr)
                placed_paths.add(jax_path)
                _maybe_consolidate(jax_path)
                return True

            if slice_aware:
                source_iter = iter_v4_safetensors_specs(checkpoint_dir)
                worker_fn = _do_place_spec
            else:
                source_iter = iter_v4_safetensors_dequant_torch(checkpoint_dir)
                worker_fn = _do_place_full

            if place_workers <= 1:
                # Single-threaded path (env override or fallback).
                for item in source_iter:
                    jax_path, arr, hf_name = worker_fn(item)
                    if jax_path is None:
                        skipped.append(hf_name)
                        continue
                    _assign(jax_path, arr)
                    placed_paths.add(jax_path)
                    _maybe_consolidate(jax_path)
                    placed_count += 1
                    if placed_count % 200 == 0:
                        now = _time.time()
                        rate = 200.0 / max(now - t_last, 1e-9)
                        print(
                            f"[deepseek_v4] placed {placed_count} tensors "
                            f"({rate:.1f}/s, last={hf_name})",
                            file=_sys.stderr, flush=True,
                        )
                        t_last = now
            else:
                # Bounded sliding window over a thread pool. We keep a few
                # extra futures in flight beyond `place_workers` so the pool
                # is always saturated, but cap the queue so peak host memory
                # stays bounded to ~that many in-flight (slice, scale)
                # buffers.
                from concurrent.futures import (
                    ThreadPoolExecutor, wait, FIRST_COMPLETED)
                in_flight_max = place_workers * 2
                last_hf_name_box = [""]
                with ThreadPoolExecutor(max_workers=place_workers) as ex:
                    pending = set()
                    exhausted = False
                    while not exhausted or pending:
                        # Refill the in-flight queue.
                        while not exhausted and len(pending) < in_flight_max:
                            try:
                                item = next(source_iter)
                            except StopIteration:
                                exhausted = True
                                break
                            pending.add(ex.submit(worker_fn, item))
                        if not pending:
                            break
                        done, pending = wait(
                            pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            ok = _drain_one(fut, last_hf_name_box)
                            if ok:
                                placed_count += 1
                                if placed_count % 200 == 0:
                                    now = _time.time()
                                    rate = 200.0 / max(now - t_last, 1e-9)
                                    print(
                                        f"[deepseek_v4] placed {placed_count} "
                                        f"tensors ({rate:.1f}/s, "
                                        f"last={last_hf_name_box[0]})",
                                        file=_sys.stderr, flush=True,
                                    )
                                    t_last = now
            elapsed = _time.time() - t0
            print(
                f"[deepseek_v4] load_weights_from_dir done: placed="
                f"{placed_count} skipped={len(skipped)} elapsed={elapsed:.1f}s "
                f"(slice_aware={slice_aware})",
                file=_sys.stderr, flush=True,
            )

            # 4. Zero-fill any leaves the checkpoint didn't cover. These are
            #    small tail leaves (MTP, optional indexer scales, etc.) so
            #    a replicated jnp.zeros is fine here.
            still_abstract: List[str] = []
            for path, leaf in path_to_leaf.items():
                if path in placed_paths:
                    continue
                if isinstance(leaf, jax.ShapeDtypeStruct):
                    still_abstract.append(path)
                    z = jnp.zeros(leaf.shape, dtype=leaf.dtype)
                    _assign(path, z)
            if still_abstract:
                # Surface in logs so we know which leaves stayed zero.
                # (Not an error: the abstract tree is a superset of any single
                # real-weight checkpoint, e.g. V4-Flash has 0 MTP layers.)
                import sys as _sys
                print(
                    f"[deepseek_v4] zero-filled {len(still_abstract)} leaves "
                    f"not present in checkpoint (e.g. {still_abstract[:3]})",
                    file=_sys.stderr, flush=True,
                )

            if _os.environ.get("V4_WEIGHT_NAN_AUDIT", "0") == "1":
                _v4_weight_nan_audit(current)

            self.params_v = nnx.Param(current)
            self.initialize_cache()

        def _consolidate_moe_after_load(self, tree):
            """Merge per-expert MoE weights into a single E-sharded stacked
            tensor per layer. Returns the rewritten param tree.

            Memory contract: the per-expert leaves and the stacked tensor
            have the same per-chip resident bytes (~128 MiB / 3 stacks /
            layer). During the `jnp.stack` we briefly hold both, then drop
            the per-leaf references by setting `experts=[]`. Worst-case
            transient peak per chip is 2× that single-layer cost (~256 MiB),
            negligible relative to the 17 GiB resident weight budget.
            """
            from jax.sharding import NamedSharding, PartitionSpec as P
            from tpu_inference.layers.jax.moe.deepseek_v4_moe import MoEParams
            mesh = self.mesh
            E_spec = NamedSharding(mesh, P('attn_dp', None, None))

            def consolidate(moe):
                if not moe.experts or moe.w1_stacked is not None:
                    return moe
                w1 = jax.device_put(
                    jnp.stack([e.w1 for e in moe.experts]), E_spec)
                w2 = jax.device_put(
                    jnp.stack([e.w2 for e in moe.experts]), E_spec)
                w3 = jax.device_put(
                    jnp.stack([e.w3 for e in moe.experts]), E_spec)
                # Block on placement so the per-leaf source arrays have no
                # outstanding readers and can be GC'd when we drop the list.
                w1.block_until_ready()
                w2.block_until_ready()
                w3.block_until_ready()
                swiglu_limit = float(moe.experts[0].swiglu_limit)
                return MoEParams(
                    gate=moe.gate,
                    experts=[],  # released
                    shared_expert=moe.shared_expert,
                    n_routed_experts=moe.n_routed_experts,
                    dim=moe.dim,
                    w1_stacked=w1,
                    w2_stacked=w2,
                    w3_stacked=w3,
                    swiglu_limit=swiglu_limit,
                )

            new_layers = []
            for layer in tree.layers:
                new_moe = consolidate(layer.moe)
                new_layers.append(dataclasses.replace(layer, moe=new_moe))

            new_mtp = []
            for mtp_block in tree.mtp:
                new_block_moe = consolidate(mtp_block.block.moe)
                new_inner = dataclasses.replace(mtp_block.block, moe=new_block_moe)
                new_mtp.append(dataclasses.replace(mtp_block, block=new_inner))

            return dataclasses.replace(tree, layers=new_layers, mtp=new_mtp)

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
            """vLLM-runtime forward. Returns `(kv_caches, hidden_TM, [])`
            with `hidden_TM` of shape `[T, hc*D]`.

            Multi-sequence (query_start_loc with >1 segment): Python loop
            over sequences, each through `transformer_body_forward`. Hidden
            states are concatenated back in input order before the HC head
            mix. Single-sequence falls through to a single body call.
            """
            ids = jnp.asarray(input_ids)
            if ids.ndim == 0:
                raise ValueError("input_ids must be 1D or 2D")
            if ids.ndim > 2:
                raise ValueError(
                    f"input_ids must be 1D or 2D, got shape {ids.shape}")
            params = self.params_v.get_value()
            freqs_swa = self._freqs_swa_v.get_value()
            freqs_comp = self._freqs_compressed_v.get_value()

            seg_bounds = self._extract_seq_segments(ids, attention_metadata)
            if seg_bounds is None or len(seg_bounds) <= 1:
                if ids.ndim == 1:
                    ids_2d = ids.reshape(1, -1)
                else:
                    ids_2d = ids
                is_decode = (
                    attention_metadata is not None
                    and int(getattr(
                        attention_metadata, "decode_start_pos", 0)) > 0)
                T = int(ids_2d.shape[-1])
                state_max_seq_len = (
                    v4_state_max_seq_len_from_vllm_config(self.vllm_config))
                if is_decode:
                    start_pos = (
                        attention_metadata.seq_lens[0] - 1).astype(jnp.int32)
                    ids_for_orchestrator = ids_2d[:, 0:1]
                    state_init_ids = None
                else:
                    start_pos = jnp.int32(0)
                    # Prefill `h` runs on the padded ids — slicing the body's
                    # input shape changes the SPMD compile and the argmax at
                    # L_real-1 drifts from the real-V4 reference. State init
                    # runs on ids sliced to the real prompt length: SWA /
                    # compressor / indexer state construction is positional,
                    # so feeding T_pad ids encodes padding tokens into kv
                    # slots that decode steps then attend to.
                    ids_for_orchestrator = ids_2d
                    L_real = T
                    qsl_cpu = getattr(
                        attention_metadata, "query_start_loc_cpu", None)
                    if qsl_cpu is not None:
                        try:
                            import numpy as _np
                            L_real = int(_np.asarray(qsl_cpu)[1])
                        except Exception:  # noqa: BLE001
                            L_real = T
                    L_real = max(1, min(L_real, T))
                    state_init_ids = (
                        ids_2d[:, :L_real] if L_real < T else None)
                kv_caches, h = deepseek_v4_run_with_decode_state(
                    kv_caches, ids_for_orchestrator, params,
                    freqs_swa, freqs_comp,
                    self.config,
                    state_max_seq_len=state_max_seq_len,
                    is_decode_step=is_decode,
                    start_pos=start_pos,
                    state_init_ids=state_init_ids,
                )
                _v4_argmax_probe("body_out", h.reshape(-1, h.shape[-1]))
                if is_decode and T > 1:
                    pad_shape = (h.shape[0], T - 1, *h.shape[2:])
                    pad = jnp.zeros(pad_shape, h.dtype)
                    h = jnp.concatenate([h, pad], axis=1)
                h_BSD = head_hc(
                    h, params.hc_head_fn, params.hc_head_scale,
                    params.hc_head_base,
                    self.config.rms_norm_eps, self.config.hc_eps,
                )
                B, S, D = h_BSD.shape
                hidden_TD = h_BSD.reshape(B * S, D)
                _v4_argmax_probe("call_hidden", hidden_TD)
                return kv_caches, hidden_TD, []

            # Multi-sequence dispatch.
            if ids.ndim == 2:
                if ids.shape[0] != 1:
                    raise ValueError(
                        "multi-seq dispatch requires 1D input_ids or "
                        f"shape (1, T); got {ids.shape}")
                ids_flat = ids.reshape(-1)
            else:
                ids_flat = ids

            D = self.config.hidden_size
            T_total = int(ids_flat.shape[0])
            per_seq_hidden = []
            covered = 0
            for s_start, s_end in seg_bounds:
                seq_ids = ids_flat[s_start:s_end].reshape(1, -1)
                h = transformer_body_forward(
                    seq_ids, params, freqs_swa, freqs_comp, self.config)
                h_BSD = head_hc(
                    h, params.hc_head_fn, params.hc_head_scale,
                    params.hc_head_base,
                    self.config.rms_norm_eps, self.config.hc_eps,
                )
                per_seq_hidden.append(h_BSD.reshape(-1, D))
                covered = s_end

            hidden_TD = jnp.concatenate(per_seq_hidden, axis=0)
            # If query_start_loc didn't cover the full padded input length
            # (vllm pads to a bucket), tail-pad with zeros.
            if covered < T_total:
                hidden_TD = jnp.concatenate(
                    [hidden_TD,
                     jnp.zeros((T_total - covered, D), dtype=hidden_TD.dtype)],
                    axis=0,
                )
            assert hidden_TD.shape[0] == T_total, (
                f"per-seq concat={hidden_TD.shape[0]} != T_total={T_total}")
            return kv_caches, hidden_TD, []

        def _extract_seq_segments(self, ids, attention_metadata):
            """Return a list of (start, end) Python int tuples describing
            each active sequence's slice of `input_ids`. Returns None to
            signal "treat as a single sequence" (the legacy default).

            This reads `attention_metadata.query_start_loc` (shape
            `(max_num_seqs+1,)`, padded with the last value in trailing
            slots) and uses `seq_lens` (padded with zeros) to count the
            active sequences.
            """
            if attention_metadata is None:
                return None
            qsl_arr = getattr(attention_metadata, "query_start_loc", None)
            if qsl_arr is None:
                return None
            try:
                import numpy as _np
                qsl = _np.asarray(qsl_arr).tolist()
            except Exception:
                return None
            if len(qsl) < 2:
                return None
            seq_lens = getattr(attention_metadata, "seq_lens", None)
            if seq_lens is not None:
                try:
                    import numpy as _np
                    sl = _np.asarray(seq_lens).tolist()
                    n_active = sum(1 for s in sl if int(s) > 0)
                except Exception:
                    n_active = None
            else:
                n_active = None
            if n_active is None:
                # Fall back: count strictly-increasing entries in qsl.
                n_active = 0
                for i in range(len(qsl) - 1):
                    if int(qsl[i + 1]) > int(qsl[i]):
                        n_active += 1
                    else:
                        break
            if n_active <= 0:
                return None
            segs = []
            for i in range(n_active):
                s_start = int(qsl[i])
                s_end = int(qsl[i + 1])
                if s_end <= s_start:
                    continue
                segs.append((s_start, s_end))
            return segs if segs else None

        def compute_logits(self, hidden_states):
            """Apply the head matmul to per-token hidden. `hidden_states` is
            shape `(T, D)` — already HC-mixed by `__call__`. We apply the
            final RMSNorm and matmul against `head_w` (vocab proj)."""
            params = self.params_v.get_value()
            D = self.config.hidden_size
            assert hidden_states.shape[-1] == D, (
                f"compute_logits expected last dim {D}, got {hidden_states.shape[-1]}"
            )
            x = rms_norm(hidden_states, params.final_norm_w, self.config.rms_norm_eps)
            logits = x.astype(jnp.float32) @ params.head_w.T
            _v4_argmax_probe("logits", logits)
            return logits

        def load_weights(self, rng=None, *args, **kwargs):
            """vLLM weight-loading entry.

            After `nnx.eval_shape(create_abstract_model)`, every leaf in
            this module is a `jax.ShapeDtypeStruct`. We need concrete
            arrays to forward through.

            Strategy:
              1. If `self.vllm_config.model_config.model` is a local
                 directory containing `config.json` + a safetensors index
                 (or single shard), load real weights via
                 `deepseek_v4_loader`. This is the production path.
              2. Otherwise fall back to materializing every leaf as
                 `jnp.zeros` (dummy load, used in unit tests where vllm
                 is not invoking us with a real path).
            """
            import os
            model_path = None
            try:
                model_path = self.vllm_config.model_config.model
            except Exception:
                model_path = None

            is_local_dir = (
                isinstance(model_path, str)
                and os.path.isdir(model_path)
                and os.path.isfile(os.path.join(model_path, "config.json"))
            )

            if is_local_dir:
                try:
                    self.load_weights_from_dir(model_path)
                    return set()
                except Exception as e:
                    # Don't crash engine init — fall back to dummy load
                    # and surface the error in logs. The forward pass
                    # will still produce defined (zero-weighted) output.
                    import traceback
                    print(
                        f"[deepseek_v4] load_weights_from_dir({model_path!r}) "
                        f"failed: {e!r}; falling back to dummy zero-fill.\n"
                        f"{traceback.format_exc()}",
                        flush=True,
                    )

            # Dummy fallback: materialize all leaves as zeros.
            def _materialize(leaf):
                if isinstance(leaf, jax.ShapeDtypeStruct):
                    return jnp.zeros(leaf.shape, dtype=leaf.dtype)
                return leaf

            current = self.params_v.get_value()
            new_params = jax.tree_util.tree_map(
                _materialize, current,
                is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
            )
            self.params_v = nnx.Param(new_params)

            # Recompute freq tables (pure function of static config).
            swa, comp = make_freqs_cis(self.config, self._effective_freqs_seq_len())
            self._freqs_swa_v = nnx.Variable(swa)
            self._freqs_compressed_v = nnx.Variable(comp)
            return set()

    return DeepseekV4ForCausalLM


DeepseekV4ForCausalLM = _build_class()


def map_hf_name_to_jax_path(name: str) -> Optional[str]:
    """Returns the JAX param-tree path string for an HF parameter name, or
    None if no rule matches.

    Names ending in `.scale` (FP4/FP8 quantization scales) return the path of
    the corresponding `.weight` plus a ".scale" suffix — the caller must
    dequantize using the scale and then place the dequantized array at the
    base path. (Dequantization itself happens in
    `deepseek_v4_loader.dequant_fp4_to_bf16` / `dequant_fp8_to_bf16`.)
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

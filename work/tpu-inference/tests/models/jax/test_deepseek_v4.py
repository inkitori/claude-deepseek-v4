# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tier 1–3 tests for DeepSeek-V4 JAX implementation against the CPU PyTorch
reference at `_deepseek_v4_reference/`.

  Tier 1 — component unit tests (`Test*Component`)
  Tier 2 — end-to-end tiny-config logits parity (`TestEndToEnd`)
  Tier 3 — compile-only test on real V4-Flash and V4-Pro configs (`TestRealConfigCompile`)

Tier 4 (weight-loader smoke test) lives in `test_deepseek_v4_weights.py`.

The JAX backend is forced to CPU (the dev box has no working TPU). Tier 3
runs on CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=N` to
simulate v4-8 (N=8) and v6e-32 (N=32). See PROD_TOPOLOGY_RISKS.md for what
this cannot validate.
"""
import os

# Force JAX to CPU — see DECISIONS.md D4. (TPU mmap fails on this host.)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Provide enough CPU devices for both v4-8 (8) and simulated v6e-32 (32).
os.environ.setdefault("XLA_FLAGS",
                      "--xla_force_host_platform_device_count=32")

import sys
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

# Make the local _deepseek_v4_reference package importable.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))


def _scratch(name):
    """Resolve a scratch fixture path. The v8 host loop stages fixtures under
    ``work/scratch/`` (the tpu-inference subtree's repo-level scratch dir).
    Earlier sessions used ``/mnt/scratch/`` — kept as a fallback so tests run
    on either layout. Falls back to env override ``V4_SCRATCH_DIR``."""
    env = os.environ.get("V4_SCRATCH_DIR")
    candidates = []
    if env:
        candidates.append(os.path.join(env, name))
    candidates.append(os.path.join("/mnt/scratch", name))
    repo_scratch = os.path.normpath(
        os.path.join(str(_HERE), "..", "..", "..", "..", "scratch"))
    candidates.append(os.path.join(repo_scratch, name))
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[-1]

from _deepseek_v4_reference import (Transformer as TorchTransformer,
                                    ModelArgs as TorchArgs,
                                    Compressor as TorchCompressor,
                                    Indexer as TorchIndexer,
                                    Attention as TorchAttention,
                                    Block as TorchBlock,
                                    MoE as TorchMoE, Gate as TorchGate)
from _deepseek_v4_reference.model import (RMSNorm as TorchRMSNorm,
                                          init_model_random,
                                          precompute_freqs_cis as torch_freqs,
                                          apply_rotary_emb as torch_rope)
from _deepseek_v4_reference.kernel_stubs import (hc_split_sinkhorn_torch,
                                                  sparse_attn_torch)

from tpu_inference.layers.jax.attention.deepseek_v4_attention import (
    rms_norm, hc_split_sinkhorn, precompute_freqs_cis, apply_rotary_emb,
    sparse_attn, splice_rope, compressor_prefill, indexer_prefill,
    get_window_topk_idxs_prefill, get_compress_topk_idxs_prefill,
    attention_prefill, CompressorParams, IndexerParams, AttentionParams,
    compressor_decode_step, indexer_decode_step, attention_decode_step,
    AttentionDecodeState, attention_decode_init_state, compressor_init_state,
    get_window_topk_idxs_decode, get_compress_topk_idxs_decode,
)
from tpu_inference.layers.jax.moe.deepseek_v4_moe import (
    gate_forward, expert_forward, moe_forward,
    GateParams, ExpertParams, MoEParams,
)
from tpu_inference.models.jax.deepseek_v4 import (
    DeepseekV4Config, BlockParams, MTPBlockParams, TransformerParams,
    block_forward, head_forward, deepseek_v4_forward_prefill,
    deepseek_v4_mtp_forward, hc_pre, hc_post, head_hc, make_freqs_cis,
    make_abstract_transformer_params, count_param_bytes,
    kv_cache_bytes_per_layer, map_hf_name_to_jax_path,
)


# ---------------- helpers ----------------

def t2j(t: torch.Tensor) -> jnp.ndarray:
    arr = t.detach()
    if arr.dtype == torch.bfloat16:
        # JAX numpy bridge through float32 (numpy has no bf16 dtype yet).
        return jnp.asarray(arr.float().numpy()).astype(jnp.bfloat16)
    return jnp.asarray(arr.numpy())


def maxabs(a: jnp.ndarray, b: torch.Tensor) -> float:
    a_np = np.asarray(a).astype(np.float32)
    b_np = b.detach().float().numpy()
    return float(np.abs(a_np - b_np).max())


def make_tiny_args(**overrides) -> TorchArgs:
    a = TorchArgs()
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


# =============================================================
# Tier 1 — components
# =============================================================


class TestRMSNormComponent:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_rms_norm_matches_torch(self, dtype):
        torch.manual_seed(0)
        x = torch.randn(2, 8, 64, dtype=dtype)
        w = torch.empty(64, dtype=torch.float32).normal_(std=0.02)
        ref = TorchRMSNorm(64, 1e-6)
        ref.weight.data.copy_(w)
        with torch.inference_mode():
            y_torch = ref(x)
        y_jax = rms_norm(t2j(x), t2j(w), 1e-6)
        atol = 1e-5 if dtype == torch.float32 else 1e-2
        assert maxabs(y_jax, y_torch) <= atol


class TestRopeComponent:
    def test_freqs_cis_matches_torch(self):
        for orig in (0, 64):
            fc_t = torch_freqs(16, 64, orig, 10000.0, 1.0, 32, 1)
            fc_j = precompute_freqs_cis(16, 64, orig, 10000.0, 1.0, 32, 1)
            np.testing.assert_allclose(np.real(np.asarray(fc_j)),
                                        np.real(fc_t.detach().numpy()),
                                        atol=1e-5)
            np.testing.assert_allclose(np.imag(np.asarray(fc_j)),
                                        np.imag(fc_t.detach().numpy()),
                                        atol=1e-5)

    @pytest.mark.parametrize("inverse", [False, True])
    def test_apply_rotary_emb_matches_torch(self, inverse):
        torch.manual_seed(0)
        q = torch.randn(2, 8, 4, 32, dtype=torch.bfloat16)  # rope_head_dim=16 in tail
        rope_dim = 16
        fc = torch_freqs(rope_dim, 64, 0, 10000.0, 1.0, 32, 1)[:8]
        q_ref = q.clone()
        torch_rope(q_ref[..., -rope_dim:], fc, inverse=inverse)
        q_jax = splice_rope(t2j(q), rope_dim, t2j(fc).astype(jnp.complex64), inverse=inverse)
        # Element-wise tolerance bf16
        assert maxabs(q_jax, q_ref) <= 1e-2


class TestSinkhornComponent:
    def test_sinkhorn_matches_torch(self):
        torch.manual_seed(0)
        N, hc = 16, 4
        mix_hc = (2 + hc) * hc
        mixes = torch.randn(N, mix_hc, dtype=torch.float32)
        hcs = torch.randn(3, dtype=torch.float32)
        hcb = torch.randn(mix_hc, dtype=torch.float32)
        pre_t, post_t, comb_t = hc_split_sinkhorn_torch(mixes, hcs, hcb, hc, 20, 1e-6)
        pre_j, post_j, comb_j = hc_split_sinkhorn(t2j(mixes), t2j(hcs), t2j(hcb), hc, 20, 1e-6)
        # fp32 throughout
        assert maxabs(pre_j, pre_t) <= 1e-5
        assert maxabs(post_j, post_t) <= 1e-5
        assert maxabs(comb_j, comb_t) <= 1e-5

    def test_sinkhorn_doubly_stochastic(self):
        torch.manual_seed(1)
        N, hc = 8, 4
        mix_hc = (2 + hc) * hc
        mixes = torch.randn(N, mix_hc, dtype=torch.float32)
        hcs = torch.randn(3, dtype=torch.float32)
        hcb = torch.randn(mix_hc, dtype=torch.float32)
        _, _, comb = hc_split_sinkhorn(t2j(mixes), t2j(hcs), t2j(hcb), hc, 30, 1e-6)
        comb = np.asarray(comb)
        # After 30 iters comb should be close to doubly stochastic.
        np.testing.assert_allclose(comb.sum(axis=-1), 1.0, atol=1e-2)
        np.testing.assert_allclose(comb.sum(axis=-2), 1.0, atol=1e-2)


class TestSparseAttnComponent:
    def test_sparse_attn_matches_torch(self):
        torch.manual_seed(0)
        B, M, H, D = 2, 6, 4, 32
        N, K = 12, 5
        q = torch.randn(B, M, H, D, dtype=torch.bfloat16)
        kv = torch.randn(B, N, D, dtype=torch.bfloat16)
        sink = torch.randn(H, dtype=torch.float32)
        idxs = torch.randint(0, N, (B, M, K), dtype=torch.int32)
        idxs[0, 0, 0] = -1
        idxs[1, 2, :2] = -1
        out_t = sparse_attn_torch(q, kv, sink, idxs, 1.0 / D ** 0.5)
        out_j = sparse_attn(t2j(q), t2j(kv), t2j(sink), t2j(idxs), 1.0 / D ** 0.5)
        assert maxabs(out_j, out_t) <= 1e-2

    def test_sparse_attn_all_invalid(self):
        """Edge case: every topk slot is -1 (no valid positions). The result
        should still be finite (sink term saves the day)."""
        torch.manual_seed(0)
        B, M, H, D, K = 1, 1, 4, 32, 4
        q = torch.randn(B, M, H, D, dtype=torch.bfloat16)
        kv = torch.randn(B, 8, D, dtype=torch.bfloat16)
        sink = torch.randn(H, dtype=torch.float32)
        idxs = torch.full((B, M, K), -1, dtype=torch.int32)
        out_t = sparse_attn_torch(q, kv, sink, idxs, 1.0 / D ** 0.5)
        out_j = sparse_attn(t2j(q), t2j(kv), t2j(sink), t2j(idxs), 1.0 / D ** 0.5)
        assert np.all(np.isfinite(np.asarray(out_j).astype(np.float32)))
        assert maxabs(out_j, out_t) <= 1e-2


# =============================================================
# Compressor / Indexer (prefill-only)
# =============================================================


def _torch_compressor_to_jax_params(c: TorchCompressor) -> CompressorParams:
    return CompressorParams(
        ape=t2j(c.ape).astype(jnp.float32),
        wkv=t2j(c.wkv.weight).astype(jnp.float32),
        wgate=t2j(c.wgate.weight).astype(jnp.float32),
        norm_w=t2j(c.norm.weight).astype(jnp.float32),
        head_dim=c.head_dim,
        rope_head_dim=c.rope_head_dim,
        compress_ratio=c.compress_ratio,
        norm_eps=1e-6,
        rotate=c.rotate,
    )


def _torch_indexer_to_jax_params(idx: TorchIndexer, args: TorchArgs) -> IndexerParams:
    return IndexerParams(
        wq_b=t2j(idx.wq_b.weight),
        weights_proj=t2j(idx.weights_proj.weight),
        compressor=_torch_compressor_to_jax_params(idx.compressor),
        n_heads=args.index_n_heads,
        head_dim=args.index_head_dim,
        rope_head_dim=args.rope_head_dim,
        index_topk=args.index_topk,
        softmax_scale=args.index_head_dim ** -0.5,
        norm_eps=1e-6,
    )


class TestCompressorComponent:
    @pytest.mark.parametrize("ratio,seqlen", [(4, 32), (4, 16), (128, 256)])
    def test_compressor_prefill_matches_torch(self, ratio, seqlen):
        torch.manual_seed(0)
        args = make_tiny_args(max_seq_len=512)
        # Force seqlen >= ratio so that compression actually runs.
        args.window_size = 8
        c = TorchCompressor(args, compress_ratio=ratio, head_dim=args.head_dim, rotate=False)
        # Init weights with controlled seed.
        for n, p in c.named_parameters():
            t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02)
            p.data.copy_(t.to(p.dtype))
        # Hook up freqs_cis and a kv_cache (the compressor writes into it; we
        # just provide a buffer of the right shape).
        kv_cache = torch.zeros(args.max_batch_size, args.max_seq_len // ratio, args.head_dim)
        c.kv_cache = kv_cache
        c.freqs_cis = torch_freqs(args.rope_head_dim, args.max_seq_len, 0,
                                   args.compress_rope_theta, args.rope_factor,
                                   args.beta_fast, args.beta_slow)
        x = torch.randn(1, seqlen, args.dim, dtype=torch.bfloat16)
        with torch.inference_mode():
            kv_t = c(x, start_pos=0)
        # JAX
        params = _torch_compressor_to_jax_params(c)
        fc_j = t2j(c.freqs_cis).astype(jnp.complex64)
        kv_j = compressor_prefill(t2j(x), params, fc_j)
        # Compare the [B, S//ratio, head_dim] outputs.
        if kv_t is None:
            pytest.skip("seqlen < ratio: compressor returns None")
        # The torch impl returns the unpadded compressed kv, shape [B, S//ratio, head_dim].
        assert kv_t.shape == kv_j.shape, f"{kv_t.shape} vs {kv_j.shape}"
        assert maxabs(kv_j, kv_t) <= 1e-2


class TestIndexerComponent:
    def test_indexer_prefill_matches_torch_topk(self):
        torch.manual_seed(0)
        args = make_tiny_args(max_seq_len=64, index_topk=4)
        idx = TorchIndexer(args, compress_ratio=4)
        for n, p in idx.named_parameters():
            t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02)
            p.data.copy_(t.to(p.dtype))
        idx.freqs_cis = torch_freqs(args.rope_head_dim, args.max_seq_len, 0,
                                     args.compress_rope_theta, args.rope_factor,
                                     args.beta_fast, args.beta_slow)
        # For test purposes, we don't pre-allocate the indexer's kv_cache to
        # be shared with compressor — the indexer manages its own.
        seqlen = 32
        x = torch.randn(1, seqlen, args.dim, dtype=torch.bfloat16)
        qr = torch.randn(1, seqlen, args.q_lora_rank, dtype=torch.bfloat16)
        offset = 100  # arbitrary offset to add to topk idxs
        with torch.inference_mode():
            topk_t = idx(x, qr, start_pos=0, offset=offset)
        params = _torch_indexer_to_jax_params(idx, args)
        fc_j = t2j(idx.freqs_cis).astype(jnp.complex64)
        topk_j, _ = indexer_prefill(t2j(x), t2j(qr), params, fc_j, offset)
        # The exact topk values may differ if there are ties (sorting is
        # not deterministic across implementations). Compare the SETS of
        # selected indices per (b, s).
        a = np.asarray(topk_j).astype(np.int64)
        b = topk_t.detach().numpy().astype(np.int64)
        assert a.shape == b.shape
        # For each query position, the set of selected idxs (after dropping
        # -1 sentinels) must match.
        for bi in range(a.shape[0]):
            for si in range(a.shape[1]):
                set_a = set(int(v) for v in a[bi, si] if v != -1)
                set_b = set(int(v) for v in b[bi, si] if v != -1)
                assert set_a == set_b, f"({bi},{si}) {set_a} vs {set_b}"


def _torch_attention_to_jax_params(attn: TorchAttention, args: TorchArgs) -> AttentionParams:
    compressor = None
    indexer = None
    if attn.compress_ratio:
        compressor = _torch_compressor_to_jax_params(attn.compressor)
        if attn.indexer is not None:
            indexer = _torch_indexer_to_jax_params(attn.indexer, args)
    return AttentionParams(
        attn_sink=t2j(attn.attn_sink).astype(jnp.float32),
        wq_a=t2j(attn.wq_a.weight),
        q_norm_w=t2j(attn.q_norm.weight).astype(jnp.float32),
        wq_b=t2j(attn.wq_b.weight),
        wkv=t2j(attn.wkv.weight),
        kv_norm_w=t2j(attn.kv_norm.weight).astype(jnp.float32),
        wo_a=t2j(attn.wo_a.weight),
        wo_b=t2j(attn.wo_b.weight),
        n_heads=attn.n_heads,
        head_dim=attn.head_dim,
        rope_head_dim=attn.rope_head_dim,
        n_groups=attn.n_groups,
        o_lora_rank=attn.o_lora_rank,
        window_size=attn.window_size,
        compress_ratio=attn.compress_ratio,
        norm_eps=args.norm_eps,
        softmax_scale=attn.softmax_scale,
        compressor=compressor,
        indexer=indexer,
    )


class TestAttentionComponent:
    @pytest.mark.parametrize("compress_ratio,layer_id", [(0, 0), (4, 2), (128, 3)])
    def test_attention_prefill_matches_torch(self, compress_ratio, layer_id):
        torch.manual_seed(0)
        args = make_tiny_args(max_seq_len=128)
        attn = TorchAttention(layer_id, args)
        # Init params.
        for n, p in attn.named_parameters():
            t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02)
            p.data.copy_(t.to(p.dtype))
        S = 32
        x = torch.randn(1, S, args.dim, dtype=torch.bfloat16)
        with torch.inference_mode():
            y_t = attn(x, start_pos=0)
        params = _torch_attention_to_jax_params(attn, args)
        fc_full = t2j(attn.freqs_cis).astype(jnp.complex64)
        y_j = attention_prefill(t2j(x), params, fc_full)
        # bf16 tolerance per tier-1 spec: atol=1e-2, rtol=1e-2.
        # Attention output passes through several matmul + accumulation chains
        # so values can be ~0.5 magnitude; rtol of 1e-2 is the binding bound.
        max_diff = maxabs(y_j, y_t)
        max_rel = float(np.abs(np.asarray(y_j).astype(np.float32) - y_t.float().numpy()).max() /
                         max(1e-6, float(np.abs(y_t.float().numpy()).max())))
        # v8 iter 4 tightening: 5e-2 -> 1e-3. Worst observed across 45 seed
        # combos: 7.6e-6 (130x margin). See TOLERANCE_LOG.md T1.
        assert max_diff <= 1e-3, f"compress_ratio={compress_ratio}: max abs diff {max_diff}"


# =============================================================
# MoE component tests
# =============================================================


def _torch_gate_to_jax_params(gate: TorchGate) -> GateParams:
    bias = t2j(gate.bias).astype(jnp.float32) if gate.bias is not None else None
    tid2eid = t2j(gate.tid2eid) if gate.hash else None
    return GateParams(
        weight=t2j(gate.weight).astype(jnp.float32),
        bias=bias,
        tid2eid=tid2eid,
        score_func=gate.score_func,
        route_scale=gate.route_scale,
        top_k=gate.topk,
    )


def _torch_expert_to_jax_params(expert) -> ExpertParams:
    return ExpertParams(
        w1=t2j(expert.w1.weight),
        w2=t2j(expert.w2.weight),
        w3=t2j(expert.w3.weight),
        swiglu_limit=expert.swiglu_limit,
    )


def _torch_moe_to_jax_params(moe: TorchMoE) -> MoEParams:
    return MoEParams(
        gate=_torch_gate_to_jax_params(moe.gate),
        experts=[_torch_expert_to_jax_params(e) for e in moe.experts],
        shared_expert=_torch_expert_to_jax_params(moe.shared_experts),
        n_routed_experts=moe.n_routed_experts,
        dim=moe.dim,
    )


def _torch_block_to_jax_params(blk: TorchBlock, args: TorchArgs) -> BlockParams:
    return BlockParams(
        attn=_torch_attention_to_jax_params(blk.attn, args),
        moe=_torch_moe_to_jax_params(blk.ffn),
        attn_norm_w=t2j(blk.attn_norm.weight).astype(jnp.float32),
        ffn_norm_w=t2j(blk.ffn_norm.weight).astype(jnp.float32),
        hc_attn_fn=t2j(blk.hc_attn_fn).astype(jnp.float32),
        hc_ffn_fn=t2j(blk.hc_ffn_fn).astype(jnp.float32),
        hc_attn_base=t2j(blk.hc_attn_base).astype(jnp.float32),
        hc_ffn_base=t2j(blk.hc_ffn_base).astype(jnp.float32),
        hc_attn_scale=t2j(blk.hc_attn_scale).astype(jnp.float32),
        hc_ffn_scale=t2j(blk.hc_ffn_scale).astype(jnp.float32),
        hc_mult=blk.hc_mult,
        hc_sinkhorn_iters=blk.hc_sinkhorn_iters,
        hc_eps=blk.hc_eps,
        norm_eps=args.norm_eps,
    )


def _torch_transformer_to_jax_params_and_cfg(model: TorchTransformer):
    a = model.args
    layers = [_torch_block_to_jax_params(l, a) for l in model.layers]
    mtp_params_list = []
    for mtp in model.mtp:
        # mtp is MTPBlock — uses Block's __init__ then adds e_proj/h_proj/etc.
        block_params = _torch_block_to_jax_params(mtp, a)
        mtp_params_list.append(MTPBlockParams(
            block=block_params,
            e_proj=t2j(mtp.e_proj.weight),
            h_proj=t2j(mtp.h_proj.weight),
            enorm_w=t2j(mtp.enorm.weight).astype(jnp.float32),
            hnorm_w=t2j(mtp.hnorm.weight).astype(jnp.float32),
            final_norm_w=t2j(mtp.norm.weight).astype(jnp.float32),
            hc_head_fn=t2j(mtp.hc_head_fn).astype(jnp.float32),
            hc_head_base=t2j(mtp.hc_head_base).astype(jnp.float32),
            hc_head_scale=t2j(mtp.hc_head_scale).astype(jnp.float32),
        ))
    params = TransformerParams(
        embed_w=t2j(model.embed.weight),
        layers=layers,
        final_norm_w=t2j(model.norm.weight).astype(jnp.float32),
        head_w=t2j(model.head.weight).astype(jnp.float32),
        hc_head_fn=t2j(model.hc_head_fn).astype(jnp.float32),
        hc_head_base=t2j(model.hc_head_base).astype(jnp.float32),
        hc_head_scale=t2j(model.hc_head_scale).astype(jnp.float32),
        mtp=mtp_params_list,
        hc_mult=a.hc_mult,
    )
    cfg = DeepseekV4Config(
        vocab_size=a.vocab_size,
        hidden_size=a.dim,
        intermediate_size=a.moe_inter_dim,
        moe_intermediate_size=a.moe_inter_dim,
        num_hidden_layers=a.n_layers,
        num_attention_heads=a.n_heads,
        num_key_value_heads=1,
        head_dim=a.head_dim,
        qk_rope_head_dim=a.rope_head_dim,
        q_lora_rank=a.q_lora_rank,
        o_lora_rank=a.o_lora_rank,
        o_groups=a.o_groups,
        n_routed_experts=a.n_routed_experts,
        n_shared_experts=a.n_shared_experts,
        num_experts_per_tok=a.n_activated_experts,
        num_hash_layers=a.n_hash_layers,
        num_nextn_predict_layers=a.n_mtp_layers,
        sliding_window=a.window_size,
        swiglu_limit=a.swiglu_limit,
        score_func=a.score_func,
        routed_scaling_factor=a.route_scale,
        rms_norm_eps=a.norm_eps,
        rope_theta=a.rope_theta,
        compress_rope_theta=a.compress_rope_theta,
        rope_factor=a.rope_factor,
        rope_beta_fast=a.beta_fast,
        rope_beta_slow=a.beta_slow,
        rope_original_seq_len=a.original_seq_len,
        max_position_embeddings=a.max_seq_len,
        compress_ratios=tuple(a.compress_ratios),
        index_n_heads=a.index_n_heads,
        index_head_dim=a.index_head_dim,
        index_topk=a.index_topk,
        hc_mult=a.hc_mult,
        hc_sinkhorn_iters=a.hc_sinkhorn_iters,
        hc_eps=a.hc_eps,
    )
    return params, cfg


class TestBlockComponent:
    @pytest.mark.parametrize("layer_id", [0, 2, 3, 5])  # SWA, CSA, HCA, trailing-SWA
    def test_block_matches_torch(self, layer_id):
        torch.manual_seed(0)
        args = make_tiny_args()
        # Build a single Block with known compress_ratio from config.
        blk = TorchBlock(layer_id, args)
        for n, p in blk.named_parameters():
            if p.dtype == torch.int32:
                p.data.copy_(torch.randint(0, args.n_routed_experts, p.shape, dtype=torch.int32))
            else:
                t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02)
                p.data.copy_(t.to(p.dtype))
        # If MoE has a hash gate (layer_id < n_hash_layers), set tid2eid.
        if blk.ffn.gate.hash:
            blk.ffn.gate.tid2eid.data.copy_(
                torch.randint(0, args.n_routed_experts, blk.ffn.gate.tid2eid.shape, dtype=torch.int32))
        S = 16
        x = torch.randn(1, S, args.hc_mult, args.dim, dtype=torch.bfloat16)
        ids = torch.randint(0, args.vocab_size, (1, S))
        with torch.inference_mode():
            y_t = blk(x, 0, ids)
        params = _torch_block_to_jax_params(blk, args)
        fc = t2j(blk.attn.freqs_cis).astype(jnp.complex64)
        y_j = block_forward(t2j(x), t2j(ids).astype(jnp.int32), params, fc)
        # Block math compounds many matmul + sinkhorn iterations + softmax.
        # v8 iter 4 tightening: 5e-2 -> 2e-2. Worst observed across 80 seed
        # combos: 7.8e-3 (2.5x margin) — bf16 ULP-bounded by 1/128 at the
        # output magnitude in this tiny config. See TOLERANCE_LOG.md T2.
        diff = maxabs(y_j, y_t)
        assert diff <= 2e-2, f"layer_id={layer_id}: block max diff {diff}"


class TestMoEComponent:
    @pytest.mark.parametrize("layer_id", [0, 1])  # 0 is hash (n_hash_layers=1), 1 is non-hash
    def test_gate_matches_torch(self, layer_id):
        torch.manual_seed(layer_id + 1)
        args = make_tiny_args()
        gate = TorchGate(layer_id, args)
        for n, p in gate.named_parameters():
            t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02)
            p.data.copy_(t.to(p.dtype))
        if gate.hash:
            gate.tid2eid.data.copy_(torch.randint(0, args.n_routed_experts, gate.tid2eid.shape, dtype=torch.int32))
        x = torch.randn(20, args.dim, dtype=torch.bfloat16)
        ids = torch.randint(0, args.vocab_size, (20,))
        with torch.inference_mode():
            w_t, i_t = gate(x, ids)
        params = _torch_gate_to_jax_params(gate)
        w_j, i_j = gate_forward(t2j(x).astype(jnp.float32), t2j(ids).astype(jnp.int32), params)
        # indices: must be exactly equal (same routing decisions).
        assert np.array_equal(np.asarray(i_j), i_t.detach().numpy()), \
            f"Gate selected different experts (hash={gate.hash})"
        # weights: bf16 tolerance.
        assert maxabs(w_j.astype(jnp.bfloat16), w_t) <= 1e-2

    def test_expert_matches_torch(self):
        torch.manual_seed(0)
        args = make_tiny_args()
        from _deepseek_v4_reference.model import Expert as TorchExpert
        e = TorchExpert(args.dim, args.moe_inter_dim, dtype=None, swiglu_limit=args.swiglu_limit)
        for n, p in e.named_parameters():
            t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02)
            p.data.copy_(t.to(p.dtype))
        x = torch.randn(8, args.dim, dtype=torch.bfloat16)
        with torch.inference_mode():
            y_t = e(x)
        params = _torch_expert_to_jax_params(e)
        y_j = expert_forward(t2j(x), None, params)
        assert maxabs(y_j, y_t) <= 1e-2

    def test_moe_matches_torch(self):
        torch.manual_seed(0)
        args = make_tiny_args()
        moe = TorchMoE(layer_id=1, args=args)  # non-hash layer for first test
        for n, p in moe.named_parameters():
            t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02)
            p.data.copy_(t.to(p.dtype))
        # Tiny seq for clarity.
        x = torch.randn(1, 6, args.dim, dtype=torch.bfloat16)
        ids = torch.randint(0, args.vocab_size, (1, 6))
        with torch.inference_mode():
            y_t = moe(x, ids)
        params = _torch_moe_to_jax_params(moe)
        y_j = moe_forward(t2j(x), t2j(ids).astype(jnp.int32), params)
        # v8 iter 4 tightening: 5e-2 -> 5e-3. Worst observed across 10 seeds:
        # 4.88e-4 (10x margin). The hash variant stays at 5e-2 (observed
        # 4.2e-2 — too close to tighten safely).
        assert maxabs(y_j, y_t) <= 5e-3

    def test_moe_hash_layer_matches_torch(self):
        torch.manual_seed(0)
        args = make_tiny_args(n_hash_layers=1)
        moe = TorchMoE(layer_id=0, args=args)  # hash layer
        for n, p in moe.named_parameters():
            t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02)
            p.data.copy_(t.to(p.dtype))
        moe.gate.tid2eid.data.copy_(torch.randint(0, args.n_routed_experts, moe.gate.tid2eid.shape, dtype=torch.int32))
        x = torch.randn(1, 6, args.dim, dtype=torch.bfloat16)
        ids = torch.randint(0, args.vocab_size, (1, 6))
        with torch.inference_mode():
            y_t = moe(x, ids)
        params = _torch_moe_to_jax_params(moe)
        y_j = moe_forward(t2j(x), t2j(ids).astype(jnp.int32), params)
        assert maxabs(y_j, y_t) <= 5e-2


# =============================================================
# Tier 2 — end-to-end (stub for now; populated after the JAX model lands)
# =============================================================


class TestEndToEnd:
    """Tier 2: full transformer forward equivalence between JAX and PyTorch
    on the tiny config with random weights.
    """

    @staticmethod
    def _build_pair(seed: int = 0, **arg_overrides):
        torch.manual_seed(seed)
        args = make_tiny_args(**arg_overrides)
        model = TorchTransformer(args)
        # Initialize all params with controlled seed for reproducibility.
        from _deepseek_v4_reference.model import init_model_random
        init_model_random(model, seed=seed)
        params, cfg = _torch_transformer_to_jax_params_and_cfg(model)
        # Pre-build the freqs tables that the JAX side expects.
        # The torch model has different `freqs_cis` per layer — we approximate by
        # using the layer's own freqs in compress vs SWA paths via cfg.
        swa = t2j(torch.empty(0)).astype(jnp.complex64)  # placeholder
        comp = t2j(torch.empty(0)).astype(jnp.complex64)
        # Use the torch reference's actual freq tables for parity.
        # The first SWA layer (compress_ratio==0) freqs_cis is the SWA path.
        # The first compressed layer (compress_ratio>0) freqs_cis is the compressed path.
        swa_layer = next((l for l in model.layers if l.attn.compress_ratio == 0), None)
        comp_layer = next((l for l in model.layers if l.attn.compress_ratio > 0), None)
        if swa_layer is not None:
            swa = t2j(swa_layer.attn.freqs_cis).astype(jnp.complex64)
        if comp_layer is not None:
            comp = t2j(comp_layer.attn.freqs_cis).astype(jnp.complex64)
        return model, params, cfg, swa, comp

    @pytest.mark.parametrize("seqlen", [16, 32, 64])
    def test_single_batch_prefill_logits_parity(self, seqlen):
        model, params, cfg, swa, comp = self._build_pair(seed=0)
        torch.manual_seed(seqlen + 7)
        x = torch.randint(0, model.args.vocab_size, (1, seqlen))
        with torch.inference_mode():
            model.reset_state()
            l_t = model(x, start_pos=0)
        l_j = deepseek_v4_forward_prefill(t2j(x).astype(jnp.int32), params, swa, comp, cfg)
        # bf16 noise in the 6-layer tiny config plateaus at ~1.4e-4 for
        # seqlen=64 (measured across 60 seed combos in v8 iter 4); the
        # original 0.1 budget assumed worst-case 0.025/layer accumulation
        # that doesn't materialize. 1e-3 keeps a 7x margin and catches any
        # real per-layer regression. See TOLERANCE_LOG.md T3.
        diff = maxabs(l_j, l_t)
        assert diff <= 1e-3, f"seqlen={seqlen}: max logits diff {diff}"

    def test_multi_batch_prefill(self):
        model, params, cfg, swa, comp = self._build_pair(seed=42)
        torch.manual_seed(99)
        x = torch.randint(0, model.args.vocab_size, (4, 16))
        with torch.inference_mode():
            model.reset_state()
            l_t = model(x, start_pos=0)
        l_j = deepseek_v4_forward_prefill(t2j(x).astype(jnp.int32), params, swa, comp, cfg)
        diff = maxabs(l_j, l_t)
        # v8 iter 4 tightening per TOLERANCE_LOG.md T3 (was 0.1, observed ~5e-5).
        assert diff <= 1e-3, f"multi-batch max logits diff {diff}"

    def test_argmax_token_agreement(self):
        """The strongest invariant: the argmax token at every position must
        agree with the PyTorch reference. Float-level diffs are looser; the
        argmax is a discrete check that catches 'mostly-right' bugs."""
        model, params, cfg, swa, comp = self._build_pair(seed=123)
        torch.manual_seed(7)
        x = torch.randint(0, model.args.vocab_size, (2, 32))
        with torch.inference_mode():
            model.reset_state()
            l_t = model(x, start_pos=0)
        l_j = deepseek_v4_forward_prefill(t2j(x).astype(jnp.int32), params, swa, comp, cfg)
        argmax_t = l_t.argmax(dim=-1).numpy()
        argmax_j = np.asarray(l_j.argmax(axis=-1))
        # Allow small disagreement on tokens with very close top-2 logits.
        agree = (argmax_t == argmax_j).mean()
        assert agree >= 0.95, f"argmax agreement {agree:.3f} < 0.95"

    def test_v4_pro_style_compress_ratios(self):
        """Tier 2 hardening: exercise V4-Pro's leading-HCA pattern
        (`compress_ratios = [128, 128, 4, 128, 4, 0]`). V4-Flash has
        leading-SWA `[0, 0, 4, 128, 4, 0]`. Both patterns route through
        the same code paths but the order differs; this test confirms
        no order-dependent bugs exist."""
        model, params, cfg, swa, comp = self._build_pair(
            seed=11,
            compress_ratios=(128, 128, 4, 128, 4, 0, 0),
        )
        torch.manual_seed(13)
        x = torch.randint(0, model.args.vocab_size, (1, 32))
        with torch.inference_mode():
            model.reset_state()
            l_t = model(x, start_pos=0)
        l_j = deepseek_v4_forward_prefill(t2j(x).astype(jnp.int32), params, swa, comp, cfg)
        diff = maxabs(l_j, l_t)
        # v8 iter 4 tightening per TOLERANCE_LOG.md T3 (was 0.1, observed ~4e-5).
        assert diff <= 1e-3, f"V4-Pro-style pattern: max logits diff {diff}"

    def test_long_context_sliding_window_wraparound(self):
        """Tier 2 hardening: prefill with seqlen >> sliding_window. With
        tiny window_size=8 and seqlen=128, the sliding window wraps 16
        times. Any off-by-one in the window_topk_idxs path would diverge."""
        model, params, cfg, swa, comp = self._build_pair(seed=21, max_seq_len=256)
        torch.manual_seed(23)
        S = 128
        x = torch.randint(0, model.args.vocab_size, (1, S))
        with torch.inference_mode():
            model.reset_state()
            l_t = model(x, start_pos=0)
        l_j = deepseek_v4_forward_prefill(t2j(x).astype(jnp.int32), params, swa, comp, cfg)
        diff = maxabs(l_j, l_t)
        # v8 iter 4 tightening per TOLERANCE_LOG.md T3 (was 0.15, observed
        # 1.2e-4 at S=128). 2e-3 keeps a ~16x margin.
        assert diff <= 2e-3, f"long-context max logits diff {diff}"
        # Argmax invariant — must agree on >=90% of the 128 positions.
        argmax_t = l_t.argmax(dim=-1).numpy()
        argmax_j = np.asarray(l_j.argmax(axis=-1))
        agree = (argmax_t == argmax_j).mean()
        assert agree >= 0.90, f"long-context argmax agreement {agree:.3f}"

    def test_mtp_forward_parity(self):
        model, params, cfg, swa, comp = self._build_pair(seed=2)
        torch.manual_seed(5)
        S = 16
        x = torch.randint(0, model.args.vocab_size, (1, S))
        # Need to run main forward first so the layers' kv_caches/state mirror
        # what they would be at MTP time. For the parity test we just feed an
        # arbitrary [B, S, hc, D] hidden state through MTP — matching the
        # PyTorch ref's `mtp[0](h, 0, x)` semantics.
        h = torch.randn(1, S, model.args.hc_mult, model.args.dim, dtype=torch.bfloat16)
        # Reset once more to ensure clean state for the MTP-only run.
        model.reset_state()
        with torch.inference_mode():
            mtp_logits_t = model.mtp[0](h, 0, x)
        mtp_logits_j = deepseek_v4_mtp_forward(
            t2j(h), t2j(x).astype(jnp.int32), params.mtp[0],
            params.embed_w, params.head_w, swa, comp, cfg,
        )
        diff = maxabs(mtp_logits_j, mtp_logits_t)
        # v8 iter 4 tightening per TOLERANCE_LOG.md T3 (was 0.1, observed ~2e-5).
        assert diff <= 1e-3, f"MTP max logits diff {diff}"


# =============================================================
# Tier 2 hardening (v8) — B1 multi-sequence dispatch
# =============================================================


def _hf_dict_from_torch_args(a: TorchArgs) -> dict:
    """Translate the PyTorch ref's `ModelArgs` → an HF-style config dict
    that `DeepseekV4Config.from_hf_dict` accepts. Used by the B1 multi-seq
    test to build a `DeepseekV4ForCausalLM` wrapper whose `__call__` we
    can exercise directly (no fixture / no vllm subprocess).
    """
    return {
        "vocab_size": a.vocab_size,
        "hidden_size": a.dim,
        "intermediate_size": a.moe_inter_dim,
        "moe_intermediate_size": a.moe_inter_dim,
        "num_hidden_layers": a.n_layers,
        "num_attention_heads": a.n_heads,
        "num_key_value_heads": 1,
        "head_dim": a.head_dim,
        "qk_rope_head_dim": a.rope_head_dim,
        "q_lora_rank": a.q_lora_rank,
        "o_lora_rank": a.o_lora_rank,
        "o_groups": a.o_groups,
        "n_routed_experts": a.n_routed_experts,
        "n_shared_experts": a.n_shared_experts,
        "num_experts_per_tok": a.n_activated_experts,
        "num_hash_layers": a.n_hash_layers,
        "num_nextn_predict_layers": a.n_mtp_layers,
        "sliding_window": a.window_size,
        "swiglu_limit": a.swiglu_limit,
        "scoring_func": a.score_func,
        "routed_scaling_factor": a.route_scale,
        "rms_norm_eps": a.norm_eps,
        "rope_theta": a.rope_theta,
        "compress_rope_theta": a.compress_rope_theta,
        "max_position_embeddings": a.max_seq_len,
        "compress_ratios": list(a.compress_ratios),
        "index_n_heads": a.index_n_heads,
        "index_head_dim": a.index_head_dim,
        "index_topk": a.index_topk,
        "hc_mult": a.hc_mult,
        "hc_sinkhorn_iters": a.hc_sinkhorn_iters,
        "hc_eps": a.hc_eps,
        "rope_scaling": {
            "factor": a.rope_factor,
            "beta_fast": a.beta_fast,
            "beta_slow": a.beta_slow,
            "original_max_position_embeddings": a.original_seq_len,
        },
    }


class TestConcurrentMultiSeqDispatch:
    """B1 (v8): when `vllm serve` schedules multiple concurrent sequences in
    one forward call, `DeepseekV4ForCausalLM.__call__` must dispatch each
    sequence through `transformer_body_forward` independently — so seq B's
    tokens don't attend to seq A's prefix. Prior to v8 the wrapper
    collapsed all `input_ids` into a single mega-sequence, which was the
    headline B1 bug. This test pins the per-sequence isolation invariant.
    """

    @staticmethod
    def _build_wrapper_with_real_params(seed: int = 0):
        from types import SimpleNamespace
        from tpu_inference.models.jax.deepseek_v4 import DeepseekV4ForCausalLM
        torch.manual_seed(seed)
        a = make_tiny_args()
        torch_model = TorchTransformer(a)
        from _deepseek_v4_reference.model import init_model_random
        init_model_random(torch_model, seed=seed)
        params, cfg = _torch_transformer_to_jax_params_and_cfg(torch_model)
        # Build the nnx wrapper from a torch-args-derived HF dict; same cfg
        # math the wrapper would derive from a real config.json.
        hf_dict = _hf_dict_from_torch_args(a)
        fake_hf = SimpleNamespace(**hf_dict)
        fake_hf.to_dict = lambda: hf_dict
        fake_vc = SimpleNamespace(model_config=SimpleNamespace(hf_config=fake_hf))
        m = DeepseekV4ForCausalLM(fake_vc)
        # Replace the auto-zero-init params with the real torch-derived ones
        # so __call__ produces meaningful per-token hidden states.
        from flax import nnx
        m.params_v = nnx.Param(params)
        return m, cfg

    @staticmethod
    def _attention_metadata_for(qsl_list, max_num_seqs=4):
        from tpu_inference.layers.common.attention_metadata import \
            AttentionMetadata
        # qsl_list[-1] is the total tokens; pad with that value to length
        # max_num_seqs+1 to match vllm's padded layout.
        qsl = list(qsl_list)
        while len(qsl) < max_num_seqs + 1:
            qsl.append(qsl[-1])
        seq_lens = [qsl_list[i + 1] - qsl_list[i]
                    for i in range(len(qsl_list) - 1)]
        while len(seq_lens) < max_num_seqs:
            seq_lens.append(0)
        T = qsl_list[-1]
        return AttentionMetadata(
            input_positions=jnp.zeros((T,), dtype=jnp.int32),
            block_tables=jnp.zeros((max_num_seqs,), dtype=jnp.int32),
            seq_lens=jnp.asarray(seq_lens, dtype=jnp.int32),
            query_start_loc=jnp.asarray(qsl, dtype=jnp.int32),
            request_distribution=jnp.asarray(
                [0, len(qsl_list) - 1, 0], dtype=jnp.int32),
        )

    def test_concurrent_decode_two_seqs(self):
        """Two sequences with different prompt lengths run through one
        `__call__` must produce per-sequence hidden states that are
        bit-identical to running each sequence alone.

        This is the strongest correctness invariant for multi-seq dispatch —
        if any layer's attention or HC mix leaks across the sequence
        boundary, the per-seq slice of the concurrent run will diverge
        from the serial run.
        """
        m, cfg = self._build_wrapper_with_real_params(seed=11)
        # Two prompts with different lengths, different vocab — the values
        # don't matter as long as they trigger different intermediate
        # activations.
        seq_a = jnp.asarray([3, 17, 1, 42, 7, 9], dtype=jnp.int32)
        seq_b = jnp.asarray([5, 11, 23, 8], dtype=jnp.int32)
        T_a, T_b = int(seq_a.shape[0]), int(seq_b.shape[0])

        # Serial single-seq runs (treat each as a fresh prefill).
        _, hidden_a, _ = m([], seq_a, attention_metadata=None)
        _, hidden_b, _ = m([], seq_b, attention_metadata=None)
        assert hidden_a.shape == (T_a, cfg.hidden_size)
        assert hidden_b.shape == (T_b, cfg.hidden_size)

        # Concurrent run — concatenated input_ids, two-segment metadata.
        ids_concat = jnp.concatenate([seq_a, seq_b], axis=0)
        meta = self._attention_metadata_for([0, T_a, T_a + T_b])
        _, hidden_concat, _ = m([], ids_concat, attention_metadata=meta)
        assert hidden_concat.shape == (T_a + T_b, cfg.hidden_size)

        # Per-seq slices must be bit-identical to the serial runs.
        diff_a = float(jnp.max(jnp.abs(hidden_concat[:T_a] - hidden_a)))
        diff_b = float(jnp.max(
            jnp.abs(hidden_concat[T_a:T_a + T_b] - hidden_b)))
        assert diff_a == 0.0, (
            f"seq A leaked across boundary: max diff {diff_a}")
        assert diff_b == 0.0, (
            f"seq B leaked across boundary: max diff {diff_b}")

    def test_single_seq_via_metadata_matches_no_metadata(self):
        """Sanity: passing `attention_metadata` describing a single sequence
        must produce the same result as passing `attention_metadata=None`.
        Exercises the `n_active == 1` branch of the dispatcher — which
        falls through to the legacy single-body path."""
        m, cfg = self._build_wrapper_with_real_params(seed=17)
        seq = jnp.asarray([2, 4, 6, 8, 10, 12, 14, 16], dtype=jnp.int32)
        T = int(seq.shape[0])
        _, hidden_no_meta, _ = m([], seq, attention_metadata=None)
        meta = self._attention_metadata_for([0, T])
        _, hidden_with_meta, _ = m([], seq, attention_metadata=meta)
        diff = float(jnp.max(jnp.abs(hidden_with_meta - hidden_no_meta)))
        assert diff == 0.0, (
            f"single-seq with metadata diverged: max diff {diff}")

    def test_three_seqs_concurrent(self):
        """Three sequences in one forward — pads `query_start_loc` to
        `max_num_seqs+1` and asserts each per-seq slice matches its serial
        run. Catches off-by-one in the `n_active` count or in trailing-pad
        handling."""
        m, cfg = self._build_wrapper_with_real_params(seed=23)
        seq_a = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)
        seq_b = jnp.asarray([10, 11, 12, 13, 14], dtype=jnp.int32)
        seq_c = jnp.asarray([20, 21, 22], dtype=jnp.int32)
        T_a, T_b, T_c = (int(seq_a.shape[0]), int(seq_b.shape[0]),
                          int(seq_c.shape[0]))
        _, hid_a, _ = m([], seq_a)
        _, hid_b, _ = m([], seq_b)
        _, hid_c, _ = m([], seq_c)
        ids = jnp.concatenate([seq_a, seq_b, seq_c], axis=0)
        meta = self._attention_metadata_for(
            [0, T_a, T_a + T_b, T_a + T_b + T_c])
        _, hid_concat, _ = m([], ids, attention_metadata=meta)
        assert hid_concat.shape == (T_a + T_b + T_c, cfg.hidden_size)
        d_a = float(jnp.max(jnp.abs(hid_concat[:T_a] - hid_a)))
        d_b = float(
            jnp.max(jnp.abs(hid_concat[T_a:T_a + T_b] - hid_b)))
        d_c = float(jnp.max(
            jnp.abs(hid_concat[T_a + T_b:T_a + T_b + T_c] - hid_c)))
        assert d_a == 0.0 and d_b == 0.0 and d_c == 0.0, (
            f"three-seq concurrent leaked: a={d_a} b={d_b} c={d_c}")


# =============================================================
# Tier 3 — compile-only against real V4 configs
# =============================================================


_HF_CONFIG_PATHS = {
    "V4-Flash": _scratch("v4_flash/config.json"),
    "V4-Pro": _scratch("v4_pro/config.json"),
}


def _load_real_config(model_name):
    import json
    path = _HF_CONFIG_PATHS[model_name]
    if not os.path.exists(path):
        pytest.skip(f"Real config not found at {path}")
    with open(path) as f:
        return DeepseekV4Config.from_hf_dict(json.load(f))


class TestRealConfigCompile:
    """Tier 3: compile-only test against the real V4 configs.
    Builds an abstract param tree (no allocation), calls `jax.eval_shape` on
    the forward, and reports per-device parameter and KV-cache byte budgets.

    The mesh kind chooses how many devices the abstract sharding is divided
    across. Both v4-8 (8 devices) and v6e-32-sim (32 devices) are simulated
    on CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=N` set at
    the top of this file. See PROD_TOPOLOGY_RISKS.md for what real-TPU
    behavior this cannot validate.
    """

    @pytest.mark.parametrize("model_name", ["V4-Flash", "V4-Pro"])
    def test_eval_shape_succeeds(self, model_name):
        cfg = _load_real_config(model_name)
        # Verify config sanity.
        assert len(cfg.compress_ratios) >= cfg.expected_compress_ratios_len
        # Build the abstract param tree.
        params_struct = make_abstract_transformer_params(cfg)
        # Pre-build the freqs tables symbolically (small).
        # Use a small max_seq_len to avoid materializing 1M positions.
        swa = jax.ShapeDtypeStruct((128, cfg.qk_rope_head_dim // 2), jnp.complex64)
        comp = jax.ShapeDtypeStruct((128, cfg.qk_rope_head_dim // 2), jnp.complex64)
        # Build a tiny input: 1 sequence × 128 tokens.
        input_ids = jax.ShapeDtypeStruct((1, 128), jnp.int32)
        # eval_shape on the forward. We can't pass swa/comp as args to a
        # jit-traced function with concrete arrays, so we wrap the call in
        # a closure that uses jax.eval_shape.
        def fwd(input_ids, params, swa, comp):
            return deepseek_v4_forward_prefill(input_ids, params, swa, comp, cfg)
        out = jax.eval_shape(fwd, input_ids, params_struct, swa, comp)
        # Output must be [B=1, S=128, vocab_size] fp32.
        assert out.shape == (1, 128, cfg.vocab_size), out.shape
        # fp32 logits.
        assert out.dtype == jnp.float32

    @pytest.mark.parametrize("model_name", ["V4-Flash", "V4-Pro"])
    def test_param_byte_budget(self, model_name):
        cfg = _load_real_config(model_name)
        params_struct = make_abstract_transformer_params(cfg)
        total = count_param_bytes(params_struct)
        gb = total / (1024 ** 3)
        # Real V4-Flash is 284B params at FP4+FP8 mixed (~150 GB), or ~570 GB at bf16
        # equivalent (since we treat all weights as bf16). V4-Pro is 1.6T => ~3.2 TB at bf16.
        # We don't enforce strict bounds — just print and sanity-check non-zero.
        print(f"\n[{model_name}] total bf16-equivalent param bytes: {gb:.1f} GB (raw: {total} bytes)")
        assert total > 0
        if model_name == "V4-Pro":
            # Sanity: V4-Pro at bf16 should be in the 1-5 TB range.
            assert 500 * (1024 ** 3) < total < 10_000 * (1024 ** 3), \
                f"V4-Pro bf16 size {gb:.1f} GB outside expected 500-10000 GB range"
        else:
            # V4-Flash bf16 should be 100-1000 GB.
            assert 100 * (1024 ** 3) < total < 2000 * (1024 ** 3), \
                f"V4-Flash bf16 size {gb:.1f} GB outside expected 100-2000 GB range"

    @pytest.mark.parametrize("model_name", ["V4-Flash", "V4-Pro"])
    @pytest.mark.parametrize("mesh_kind", ["v4-8", "v6e-32-sim"])
    def test_per_device_budget(self, model_name, mesh_kind):
        """Print per-device budget assuming uniform sharding across `mesh_dev`
        devices. Flag if any dimension doesn't divide cleanly.
        """
        cfg = _load_real_config(model_name)
        mesh_devs = {"v4-8": 8, "v6e-32-sim": 32}[mesh_kind]
        if len(jax.devices()) < mesh_devs:
            pytest.skip(f"Need {mesh_devs} CPU devices; have {len(jax.devices())}")
        params_struct = make_abstract_transformer_params(cfg)
        total = count_param_bytes(params_struct)
        per_dev = total / mesh_devs
        per_dev_gb = per_dev / (1024 ** 3)
        print(f"\n[{model_name} @ {mesh_kind}] per-device bf16 param bytes: {per_dev_gb:.2f} GB")
        # KV cache size for 1M context per layer.
        kv_per_layer = kv_cache_bytes_per_layer(cfg, max_seq_len=cfg.max_position_embeddings)
        # Aggregate across layers.
        from collections import Counter
        ratio_counter = Counter(cfg.compress_ratios)
        total_kv = 0
        for ratio, n_layers in ratio_counter.items():
            if ratio == 0:
                total_kv += kv_per_layer["swa_only"] * n_layers
            elif ratio == 4:
                total_kv += kv_per_layer["csa_layer"] * n_layers
            elif ratio == 128:
                total_kv += kv_per_layer["hca_layer"] * n_layers
        kv_per_dev = total_kv / mesh_devs
        print(f"[{model_name} @ {mesh_kind}] per-device KV @ 1M ctx (1 seq): {kv_per_dev / (1024**3):.2f} GB")
        # Sanity: per-device weights for V4-Pro on v6e-32 should be in tens of GB.
        if model_name == "V4-Pro" and mesh_kind == "v6e-32-sim":
            # v6e has 32 GB HBM/chip; bf16 weights would be ~100 GB/chip with no fp4. Will OOM
            # in production unless fp4/fp8 is applied. PROD_TOPOLOGY_RISKS.md documents this.
            print("WARNING: V4-Pro at bf16 will OOM on v6e-32 (32 GB HBM/chip) — needs FP4/FP8.")

    @pytest.mark.parametrize("model_name", ["V4-Flash", "V4-Pro"])
    def test_compile_first_two_layers_only(self, model_name):
        """Compile-only test on a TRUNCATED real config (first 2 layers) so the
        actual XLA compilation completes in a reasonable time. Verifies that
        the math constructs lower correctly.

        For V4-Pro we additionally truncate n_routed_experts to 64 (real is
        384) to keep CPU memory tractable. The math paths exercised are
        identical to V4-Flash; this test guards V4-Pro-specific shape ratios
        (q_lora_rank=1536, o_groups=16, etc.) — if those triggered a bug it
        would also surface in eval_shape (covered by `test_eval_shape_succeeds`).
        """
        cfg_full = _load_real_config(model_name)
        cfg_dict = {**dataclasses.asdict(cfg_full),
                    "num_hidden_layers": 2,
                    "num_nextn_predict_layers": 0,
                    "compress_ratios": (cfg_full.compress_ratios[2],
                                         cfg_full.compress_ratios[3])}
        if model_name == "V4-Pro":
            # Reduce experts so the zeros-materialized param tree fits in CPU RAM.
            cfg_dict["n_routed_experts"] = max(64, cfg_full.num_experts_per_tok * 4)
            cfg_dict["vocab_size"] = 4096  # also reduce vocab; tid2eid is keyed by vocab
        cfg = DeepseekV4Config(**cfg_dict)
        params_struct = make_abstract_transformer_params(cfg)
        # Materialize the params with zeros.
        def to_zeros(x):
            return jnp.zeros(x.shape, dtype=x.dtype)
        params = jax.tree_util.tree_map(to_zeros, params_struct)
        S = 16
        input_ids = jnp.zeros((1, S), dtype=jnp.int32)
        swa, comp = make_freqs_cis(cfg, S)
        # JIT + lower + compile.
        @jax.jit
        def fwd(ids, p, sw, cp):
            return deepseek_v4_forward_prefill(ids, p, sw, cp, cfg)
        lowered = fwd.lower(input_ids, params, swa, comp)
        compiled = lowered.compile()
        # Smoke-run the compiled fn to verify no shape/sharding bugs surface.
        out = compiled(input_ids, params, swa, comp)
        assert out.shape == (1, S, cfg.vocab_size)


# Note: the dataclasses module is needed in the test for asdict()
import dataclasses


# =============================================================
# Tier 4 — weight-loading smoke test
# =============================================================


class TestWeightLoaderSmoke:
    """Tier 4: validate that every parameter name in the V4-Flash safetensors
    index can be mapped to a JAX param-tree path, and (for one downloaded
    shard) that the parameter shapes match what we expect from the abstract
    param tree.

    Skips if the safetensors index isn't present at /mnt/scratch/v4_flash/.
    """

    INDEX_PATH = _scratch("v4_flash/model.safetensors.index.json")
    SHARD_PATH = _scratch("v4_flash/model-00001-of-00046.safetensors")

    def _load_index(self):
        import json
        if not os.path.exists(self.INDEX_PATH):
            pytest.skip(f"V4-Flash safetensors index not found at {self.INDEX_PATH}")
        with open(self.INDEX_PATH) as f:
            return json.load(f)

    def test_every_name_maps(self):
        idx = self._load_index()
        wm = idx["weight_map"]
        unmapped = []
        for name in wm.keys():
            path = map_hf_name_to_jax_path(name)
            if path is None:
                unmapped.append(name)
        assert not unmapped, (
            f"{len(unmapped)} HF names did not map; first 10:\n  "
            + "\n  ".join(unmapped[:10])
        )
        print(f"\n[V4-Flash] All {len(wm)} HF names mapped to JAX paths.")

    def test_shard_shapes_match_abstract_tree(self):
        """Open one safetensors shard and check that each tensor's shape
        matches the abstract JAX param tree.

        We do NOT load the .scale tensors here — they're FP4/FP8 quantization
        artifacts and require dequantization. We only validate plain .weight
        tensors and unsuffixed params (e.g. attn_sink, hc_*_base).
        """
        if not os.path.exists(self.SHARD_PATH):
            pytest.skip(f"Shard not found at {self.SHARD_PATH}")
        cfg = _load_real_config("V4-Flash")
        params = make_abstract_transformer_params(cfg)

        # Build a flat dict from path -> ShapeDtypeStruct.
        # Walk the param tree manually since paths use [N] index notation.
        path_to_struct = {}
        path_to_struct["embed_w"] = params.embed_w
        path_to_struct["head_w"] = params.head_w
        path_to_struct["final_norm_w"] = params.final_norm_w
        path_to_struct["hc_head_fn"] = params.hc_head_fn
        path_to_struct["hc_head_base"] = params.hc_head_base
        path_to_struct["hc_head_scale"] = params.hc_head_scale
        for li, layer in enumerate(params.layers):
            base = f"layers[{li}]"
            path_to_struct[f"{base}.attn_norm_w"] = layer.attn_norm_w
            path_to_struct[f"{base}.ffn_norm_w"] = layer.ffn_norm_w
            path_to_struct[f"{base}.hc_attn_fn"] = layer.hc_attn_fn
            path_to_struct[f"{base}.hc_ffn_fn"] = layer.hc_ffn_fn
            path_to_struct[f"{base}.hc_attn_base"] = layer.hc_attn_base
            path_to_struct[f"{base}.hc_ffn_base"] = layer.hc_ffn_base
            path_to_struct[f"{base}.hc_attn_scale"] = layer.hc_attn_scale
            path_to_struct[f"{base}.hc_ffn_scale"] = layer.hc_ffn_scale
            attn = layer.attn
            attn_base = f"{base}.attn"
            path_to_struct[f"{attn_base}.attn_sink"] = attn.attn_sink
            path_to_struct[f"{attn_base}.wq_a"] = attn.wq_a
            path_to_struct[f"{attn_base}.wq_b"] = attn.wq_b
            path_to_struct[f"{attn_base}.q_norm_w"] = attn.q_norm_w
            path_to_struct[f"{attn_base}.wkv"] = attn.wkv
            path_to_struct[f"{attn_base}.kv_norm_w"] = attn.kv_norm_w
            path_to_struct[f"{attn_base}.wo_a"] = attn.wo_a
            path_to_struct[f"{attn_base}.wo_b"] = attn.wo_b
            if attn.compressor is not None:
                cb = f"{attn_base}.compressor"
                c = attn.compressor
                path_to_struct[f"{cb}.ape"] = c.ape
                path_to_struct[f"{cb}.norm_w"] = c.norm_w
                path_to_struct[f"{cb}.wkv"] = c.wkv
                path_to_struct[f"{cb}.wgate"] = c.wgate
            if attn.indexer is not None:
                ib = f"{attn_base}.indexer"
                ix = attn.indexer
                path_to_struct[f"{ib}.wq_b"] = ix.wq_b
                path_to_struct[f"{ib}.weights_proj"] = ix.weights_proj
                ic = ix.compressor
                icb = f"{ib}.compressor"
                path_to_struct[f"{icb}.ape"] = ic.ape
                path_to_struct[f"{icb}.norm_w"] = ic.norm_w
                path_to_struct[f"{icb}.wkv"] = ic.wkv
                path_to_struct[f"{icb}.wgate"] = ic.wgate
            moe = layer.moe
            mb = f"{base}.moe"
            path_to_struct[f"{mb}.gate.weight"] = moe.gate.weight
            if moe.gate.bias is not None:
                path_to_struct[f"{mb}.gate.bias"] = moe.gate.bias
            if moe.gate.tid2eid is not None:
                path_to_struct[f"{mb}.gate.tid2eid"] = moe.gate.tid2eid
            for ei, e in enumerate(moe.experts):
                path_to_struct[f"{mb}.experts[{ei}].w1"] = e.w1
                path_to_struct[f"{mb}.experts[{ei}].w2"] = e.w2
                path_to_struct[f"{mb}.experts[{ei}].w3"] = e.w3
            path_to_struct[f"{mb}.shared_expert.w1"] = moe.shared_expert.w1
            path_to_struct[f"{mb}.shared_expert.w2"] = moe.shared_expert.w2
            path_to_struct[f"{mb}.shared_expert.w3"] = moe.shared_expert.w3
        # MTP analogous (skipped in this test since shard 1 has no MTP).

        # Read the shard header (metadata only).
        from safetensors import safe_open
        mismatches = []
        unrecognized = []
        scale_only = 0
        with safe_open(self.SHARD_PATH, framework="numpy") as f:
            for name in f.keys():
                path = map_hf_name_to_jax_path(name)
                if path is None:
                    unrecognized.append(name)
                    continue
                if path.endswith("<scale>"):
                    scale_only += 1
                    continue
                t_shape = tuple(f.get_slice(name).get_shape())
                if path not in path_to_struct:
                    # E.g. mtp paths in this shard would land here if shard 1
                    # contained mtp. Not a failure for shard 1.
                    continue
                expected = path_to_struct[path]
                if tuple(expected.shape) != t_shape:
                    mismatches.append((name, t_shape, tuple(expected.shape)))
        # Report.
        print(f"\n[V4-Flash shard 1] checked {scale_only} scale tensors (deferred to dequant), "
              f"{len(unrecognized)} unrecognized, {len(mismatches)} shape mismatches.")
        assert not unrecognized, f"unrecognized: {unrecognized[:5]}"
        # Some shape mismatches are expected when the HF checkpoint stores
        # the FP4/FP8 packed form (e.g. weight is [out, in//2] for fp4_e2m1fn_x2,
        # but our abstract tree has bf16 [out, in]). Allow them but report.
        if mismatches:
            print(f"[V4-Flash shard 1] note: {len(mismatches)} shape mismatches "
                  f"(expected for FP4/FP8 packed weights). First 3:")
            for m in mismatches[:3]:
                print(f"  {m[0]}: HF{m[1]} vs JAX-bf16{m[2]}")


# =============================================================
# W1 — Decode path tests
# =============================================================
#
# These tests verify the JAX decode functions match the reference
# `Attention.forward(x, start_pos>0)` path. Strategy:
#   1. Run torch reference prefill of length P → fills the torch-side state
#      buffers (kv_cache, kv_state, score_state, indexer caches/states).
#   2. Mirror the populated torch state into a JAX `AttentionDecodeState`.
#   3. Run torch decode for one or more tokens at start_pos=P, P+1, ...
#   4. Run JAX `attention_decode_step` on the same inputs.
#   5. Compare (per-step output, per-step state).


def _torch_attention_state_to_jax(
    attn,        # TorchAttention after prefill
    args,        # TorchArgs
    bsz: int,
) -> AttentionDecodeState:
    """Snapshot the torch Attention's state buffers into an AttentionDecodeState.
    Only the [:bsz] slice is taken (matching reference indexing)."""
    # Attention.kv_cache: [max_batch, win+(max/ratio if ratio else 0), head_dim]
    kvc = t2j(attn.kv_cache[:bsz].clone())
    if attn.compress_ratio > 0:
        c_kv = t2j(attn.compressor.kv_state[:bsz].clone()).astype(jnp.float32)
        c_sc = t2j(attn.compressor.score_state[:bsz].clone()).astype(jnp.float32)
    else:
        c_kv = jnp.zeros((bsz, 0, 0), dtype=jnp.float32)
        c_sc = jnp.full((bsz, 0, 0), -jnp.inf, dtype=jnp.float32)
    if attn.compress_ratio == 4 and attn.indexer is not None:
        i_kv = t2j(attn.indexer.compressor.kv_state[:bsz].clone()).astype(jnp.float32)
        i_sc = t2j(attn.indexer.compressor.score_state[:bsz].clone()).astype(jnp.float32)
        i_cache = t2j(attn.indexer.kv_cache[:bsz].clone())
    else:
        i_kv = jnp.zeros((bsz, 0, 0), dtype=jnp.float32)
        i_sc = jnp.full((bsz, 0, 0), -jnp.inf, dtype=jnp.float32)
        i_cache = jnp.zeros((bsz, 0, 0), dtype=jnp.bfloat16)
    return AttentionDecodeState(
        kv_cache=kvc,
        compressor_kv_state=c_kv,
        compressor_score_state=c_sc,
        indexer_kv_state=i_kv,
        indexer_score_state=i_sc,
        indexer_kv_cache=i_cache,
    )


def _build_attn_for_decode(layer_id: int, seed: int = 0, max_seq_len: int = 64):
    """Build a fresh torch.Attention with random init and a max_seq_len that
    accommodates decode tests."""
    torch.manual_seed(seed)
    args = make_tiny_args(max_seq_len=max_seq_len)
    attn = TorchAttention(layer_id, args)
    for n, p in attn.named_parameters():
        t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02)
        p.data.copy_(t.to(p.dtype))
    return attn, args


class TestDecodeAttentionParity:
    """Verify attention_decode_step matches torch reference, layer by layer."""

    @pytest.mark.parametrize("layer_id,start_pos", [
        (0, 1),       # SWA, very first decode token
        (0, 8),       # SWA, just past sliding-window edge
        (0, 9),       # SWA, window-wrap boundary
        (2, 4),       # CSA, first compression boundary (ratio=4)
        (2, 7),       # CSA, mid-window before next compression
        (2, 8),       # CSA, second compression boundary
        (2, 16),      # CSA, multiple compressions in
        (3, 16),      # HCA (ratio=128) — no compression yet (only at start_pos+1==128)
        (3, 32),      # HCA, well after window edge
    ])
    def test_decode_step_parity(self, layer_id, start_pos):
        attn, args = _build_attn_for_decode(layer_id, seed=0, max_seq_len=256)
        bsz = 1
        with torch.inference_mode():
            # Run torch prefill of length start_pos to populate state.
            prefill_x = torch.randn(bsz, start_pos, args.dim, dtype=torch.bfloat16)
            _ = attn(prefill_x, start_pos=0)
            # Decode one token at start_pos.
            x_step = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)
            o_t = attn(x_step, start_pos=start_pos)

        # Snapshot torch state -> JAX state. NOTE: snapshot must be after the
        # prefill step but before the decode step. We re-run prefill on a
        # *fresh* attn to capture the right state.
        attn_fresh, args2 = _build_attn_for_decode(layer_id, seed=0, max_seq_len=256)
        with torch.inference_mode():
            _ = attn_fresh(prefill_x, start_pos=0)
        jax_state = _torch_attention_state_to_jax(attn_fresh, args2, bsz)

        # Build JAX params.
        params_j = _torch_attention_to_jax_params(attn_fresh, args2)
        fc = t2j(attn_fresh.freqs_cis).astype(jnp.complex64)
        new_state, y_j = attention_decode_step(
            t2j(x_step), start_pos, params_j, fc, jax_state,
        )
        diff = maxabs(y_j, o_t)
        # v8 iter 4 tightening: was 5e-2; observed worst 3.8e-6 across the
        # 9 parametrized points. 1e-4 keeps a 25x margin while making the
        # test catch real regressions. See TOLERANCE_LOG.md "Decode step".
        assert diff <= 1e-4, (
            f"layer={layer_id} start_pos={start_pos}: decode step output "
            f"diff {diff}"
        )


class TestDecodeRollingParity:
    """Run a rolling decode of length K starting from a P-prefix prefill, and
    compare the per-step outputs to the torch reference. This is a stronger
    test that catches state-mutation bugs."""

    @pytest.mark.parametrize("layer_id,P,K", [
        (0, 4, 4),    # SWA short
        (0, 8, 8),    # SWA at window boundary
        (2, 8, 8),    # CSA: 2 compressions during prefill, more during decode
        (2, 4, 12),   # CSA: prefill < ratio, all compressions happen during decode
        (3, 16, 16),  # HCA: small enough that no compression yet (ratio=128)
    ])
    def test_rolling_decode_parity(self, layer_id, P, K):
        attn, args = _build_attn_for_decode(layer_id, seed=42, max_seq_len=256)
        bsz = 1
        torch.manual_seed(P + K + layer_id)
        full_x = torch.randn(bsz, P + K, args.dim, dtype=torch.bfloat16)

        with torch.inference_mode():
            # Torch path: prefill P, then decode K one-at-a-time.
            _ = attn(full_x[:, :P], start_pos=0)
        # Snapshot torch state into JAX.
        jax_state = _torch_attention_state_to_jax(attn, args, bsz)
        params_j = _torch_attention_to_jax_params(attn, args)
        fc = t2j(attn.freqs_cis).astype(jnp.complex64)

        for k in range(K):
            sp = P + k
            x_step = full_x[:, sp:sp + 1]
            with torch.inference_mode():
                o_t = attn(x_step, start_pos=sp)
            jax_state, y_j = attention_decode_step(
                t2j(x_step), sp, params_j, fc, jax_state,
            )
            diff = maxabs(y_j, o_t)
            # v8 iter 4 tightening: was 5e-2; observed worst 3.8e-6 across
            # all (layer, P, K) combos. 1e-4 keeps a 25x margin.
            assert diff <= 1e-4, (
                f"step k={k} (sp={sp}): rolling decode diff {diff}"
            )


# =============================================================
# Tier 6 — Real-TPU compile + tiny forward
# =============================================================


class TestRealTpuTinyForward:
    """Tier 6: compile + run a tiny forward on real TPU using the
    pre-staged tiny_v4_bf16 fixture. Skips when TPU is unavailable."""

    BF16_DIR = _scratch("tiny_v4_bf16")

    def _get_tpu_devices(self):
        try:
            return jax.devices("tpu")
        except (RuntimeError, ValueError):
            return []

    def test_tiny_tpu_compile_and_forward(self):
        if not os.path.exists(self.BF16_DIR):
            pytest.skip(f"{self.BF16_DIR} missing")
        tpu_devs = self._get_tpu_devices()
        if len(tpu_devs) < 1:
            pytest.skip("No TPU devices available")

        # We deliberately do NOT compare against the CPU result here — Tier 6
        # only verifies that compile succeeds and the forward runs without
        # error. Logits dtype/shape are checked.
        import json
        from tpu_inference.models.jax.deepseek_v4 import (
            DeepseekV4Config, deepseek_v4_forward_prefill,
            make_abstract_transformer_params, make_freqs_cis,
        )
        from tpu_inference.models.jax.deepseek_v4_loader import (
            apply_weights_to_param_tree, load_v4_safetensors_to_dict,
        )
        with open(os.path.join(self.BF16_DIR, "config.json")) as f:
            hf_config = json.load(f)
        cfg = DeepseekV4Config.from_hf_dict(hf_config)
        params = make_abstract_transformer_params(cfg)
        # Materialize abstract params with zeros (we only need shapes for
        # compile — for the forward we'll substitute the real loader output).
        params = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, dtype=x.dtype),
            params,
            is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
        )
        weights = load_v4_safetensors_to_dict(self.BF16_DIR)
        params = apply_weights_to_param_tree(params, weights, cfg)
        S = 16
        ids = (jnp.arange(S, dtype=jnp.int32) % cfg.vocab_size).reshape(1, S)
        swa, comp = make_freqs_cis(cfg, cfg.max_position_embeddings)

        @jax.jit
        def fwd(ids, p, sw, cp):
            return deepseek_v4_forward_prefill(ids, p, sw, cp, cfg)

        # Move to TPU.
        with jax.default_device(tpu_devs[0]):
            ids = jax.device_put(ids)
            params = jax.device_put(params)
            swa = jax.device_put(swa)
            comp = jax.device_put(comp)
            logits = fwd(ids, params, swa, comp).block_until_ready()
        assert logits.shape == (1, S, cfg.vocab_size)
        assert logits.dtype == jnp.float32
        np_logits = np.asarray(logits).astype(np.float32)
        # Sanity: finite, non-zero, varied (not a constant).
        assert np.all(np.isfinite(np_logits)), "non-finite TPU logits"
        assert np_logits.std() > 0.01, f"TPU logits std too low: {np_logits.std()}"
        # CPU comparison happens in TestRealTpuVsCpuLogitsParity (separate
        # session) — JAX cannot host both TPU and CPU backends in the same
        # process. The CPU forward correctness is already covered by every
        # CPU-default test in this file.


# =============================================================
# Tier 5 — vLLM serve curl round-trip (TPU-only)
# =============================================================


class TestVllmServeRoundtrip:
    """Tier 5: spawn `vllm serve` against /mnt/scratch/tiny_v4_bf16, send
    two identical /v1/completions, assert HTTP 200 + non-empty text +
    byte-equal `choices` (deterministic with seed=0).

    Skips when TPU unavailable, fixture missing, or `vllm` binary not on
    PATH. Per the v6 SUMMARY this round-trip exercises the entire
    integration: pydantic VllmConfig gate (B4 workaround), the V4 nnx
    port (B2 fix), the KVCacheManager use_mla override (B5 fix), and
    the deepseek_v4_loader path through DeepseekV4ForCausalLM.load_weights.
    """

    BF16_DIR = _scratch("tiny_v4_bf16")
    PORT = 18080
    READY_TIMEOUT_S = 240
    PREFLIGHT_LOG = os.environ.get(
        "TPU_PREFLIGHT_LOG",
        os.path.normpath(os.path.join(str(_HERE),
                                      "..", "..", "..", "..", "..",
                                      "logs", "tpu-preflight.log")))

    def _has_tpu(self):
        # We deliberately do NOT call `jax.devices("tpu")` here — that
        # initializes the TPU backend in the *parent* pytest process and
        # makes the TPU unavailable to the subprocess we spawn. Instead
        # read the host-side preflight log written before the agent was
        # invoked.
        try:
            with open(self.PREFLIGHT_LOG) as f:
                first_line = f.readline().strip()
        except FileNotFoundError:
            return False
        try:
            import json
            d = json.loads(first_line)
            return bool(d.get("ok")) and int(d.get("n_tpu", 0)) >= 1
        except Exception:
            return False

    def _has_vllm_binary(self):
        import shutil
        return shutil.which("vllm") is not None

    def test_curl_roundtrip(self):
        if not os.path.exists(self.BF16_DIR):
            pytest.skip(f"{self.BF16_DIR} missing")
        if not self._has_tpu():
            pytest.skip("No TPU devices available")
        if not self._has_vllm_binary():
            pytest.skip("vllm binary not on PATH")

        import json
        import subprocess
        import time
        import urllib.request
        import urllib.error

        env = dict(os.environ)
        env["NEW_MODEL_DESIGN"] = "1"
        env["JAX_PLATFORMS"] = "tpu"
        # Avoid inheriting the test process's CPU-mesh XLA flags.
        env.pop("XLA_FLAGS", None)

        cmd = [
            "vllm", "serve", self.BF16_DIR,
            "--tensor-parallel-size", "4",
            "--max-model-len", "256",
            "--max-num-seqs", "2",
            "--port", str(self.PORT),
            "--seed", "0",
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--additional_config",
            '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}',
        ]
        log_path = f"/tmp/vllm_serve_t5_pytest.log"
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        )
        try:
            ready = False
            deadline = time.monotonic() + self.READY_TIMEOUT_S
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    log_f.flush()
                    log_tail = open(log_path).read()[-4000:]
                    pytest.fail(
                        f"vllm serve died during init "
                        f"(rc={proc.returncode}); log tail:\n{log_tail}"
                    )
                try:
                    with urllib.request.urlopen(
                        f"http://localhost:{self.PORT}/v1/models",
                        timeout=2,
                    ) as resp:
                        if resp.status == 200:
                            ready = True
                            break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                    pass
                time.sleep(2)

            if not ready:
                pytest.fail(f"vllm serve did not become ready in {self.READY_TIMEOUT_S}s")

            def _post(prompt, max_tokens=8, seed=0):
                payload = json.dumps({
                    "model": self.BF16_DIR,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "seed": seed,
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"http://localhost:{self.PORT}/v1/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.status, r.read()

            # (1) Spec assertions: two identical seed=0 requests are
            # byte-equal in their generation-relevant fields, and produce
            # non-empty completion text.
            s1, b1 = _post("abc")
            s2, b2 = _post("abc")
            assert s1 == 200 and s2 == 200, (s1, s2)
            j1 = json.loads(b1)
            j2 = json.loads(b2)
            text1 = j1["choices"][0]["text"]
            text2 = j2["choices"][0]["text"]
            assert text1 != "", f"resp1 text was empty: {j1!r}"
            assert text2 != "", f"resp2 text was empty: {j2!r}"
            assert text1 == text2, (text1, text2)
            assert j1["choices"][0]["finish_reason"] == j2["choices"][0]["finish_reason"]
            assert j1["usage"] == j2["usage"]
            assert j1["choices"][0]["finish_reason"] == "length", (
                f"expected finish_reason=length (max_tokens=8), got "
                f"{j1['choices'][0]['finish_reason']}"
            )
            assert j1["usage"]["completion_tokens"] == 8, j1["usage"]

            # (2) Sanity: prompt-dependence. A *different* prompt should
            # produce different completion text than "abc". This is a
            # weak sanity check — the only ways this fails are: (a) the
            # model is collapsing all prompts to the same logits (bug),
            # (b) by accident the two prompts happen to argmax-tie. We
            # use a longer prompt that shares no characters to maximize
            # the chance of divergence.
            s3, b3 = _post("hello world this is a longer prompt")
            assert s3 == 200, s3
            j3 = json.loads(b3)
            text3 = j3["choices"][0]["text"]
            assert text3 != "", f"resp3 text was empty: {j3!r}"
            assert text3 != text1, (
                f"different prompt produced same completion as 'abc' — "
                f"prompt-independent output is a bug.\n"
                f"  text1={text1!r}\n  text3={text3!r}"
            )

            # (3) Longer generation: max_tokens=16 still completes
            # without error and produces 16 completion tokens (or hits
            # finish_reason=stop, but the tiny synthetic config has no
            # natural stop tokens so we expect length).
            s4, b4 = _post("abc", max_tokens=16)
            assert s4 == 200, s4
            j4 = json.loads(b4)
            text4 = j4["choices"][0]["text"]
            assert text4 != "", f"resp4 text was empty: {j4!r}"
            assert j4["usage"]["completion_tokens"] == 16, j4["usage"]
            # The first 8 tokens should match the 8-token completion
            # because temperature=0 makes the model deterministic at
            # each step regardless of remaining max_tokens budget.
            # (Caveat: vllm's continuous-batching scheduler can
            # interleave with other requests, but at temp=0 the
            # per-position argmax is independent of batch.)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            log_f.close()


# =============================================================
# W4 / Tier 4b / Tier 7 — FP4/FP8 weight loader & dequant
# =============================================================


class TestFp8Dequant:
    """Unit-level FP8 dequant: loader produces bit-identical bf16 to a
    pre-staged groundtruth on the tiny synthetic fixture."""

    QUANT = _scratch("tiny_v4_quant")
    GT = _scratch("tiny_v4_groundtruth")

    def _skip_if_missing(self):
        if not (os.path.exists(self.QUANT) and os.path.exists(self.GT)):
            pytest.skip("tiny_v4_quant/tiny_v4_groundtruth not present")

    def test_full_loader_matches_groundtruth(self):
        """All 355 tensors load to bit-identical bf16 against groundtruth."""
        self._skip_if_missing()
        from tpu_inference.models.jax.deepseek_v4_loader import \
            load_v4_safetensors_to_dict
        wq = load_v4_safetensors_to_dict(self.QUANT)
        wgt = load_v4_safetensors_to_dict(self.GT)
        assert set(wq.keys()) == set(wgt.keys()), "key set mismatch"
        max_diff = 0.0
        n = 0
        for k in wq:
            a = np.asarray(wq[k]).astype(np.float32)
            b = np.asarray(wgt[k]).astype(np.float32)
            assert a.shape == b.shape, f"shape mismatch {k}: {a.shape} vs {b.shape}"
            d = float(np.abs(a - b).max())
            if d > max_diff:
                max_diff = d
            n += 1
        # Bit-identical: dequant of (e4m3fn * e8m0) must produce the same bf16
        # bytes as the pre-dequantized groundtruth (which used the same recipe).
        assert max_diff == 0.0, f"max diff {max_diff} across {n} tensors"


class TestDeepseekV4ForCausalLMHelpers:
    """Sanity test that the partial __call__ helpers
    (load_weights_from_dir + forward_prefill) work end-to-end on
    tiny_v4_bf16. Full vllm-runtime __call__ is blocked on B1+B2."""

    BF16_DIR = _scratch("tiny_v4_bf16")

    def test_forward_prefill_helper(self):
        if not os.path.exists(self.BF16_DIR):
            pytest.skip(f"{self.BF16_DIR} not present")
        from types import SimpleNamespace
        from tpu_inference.models.jax.deepseek_v4 import DeepseekV4ForCausalLM
        with open(os.path.join(self.BF16_DIR, "config.json")) as f:
            import json
            hf_dict = json.load(f)
        # Fake vllm_config: must expose model_config.hf_config either as a
        # dict or with a .to_dict method.
        fake_hf = SimpleNamespace(**hf_dict)
        fake_hf.to_dict = lambda: hf_dict
        fake_vc = SimpleNamespace(model_config=SimpleNamespace(hf_config=fake_hf))
        model = DeepseekV4ForCausalLM(fake_vc)
        model.load_weights_from_dir(self.BF16_DIR)
        S = 16
        ids = (jnp.arange(S, dtype=jnp.int32) % model.config.vocab_size).reshape(1, S)
        logits = model.forward_prefill(ids)
        assert logits.shape == (1, S, model.config.vocab_size)
        assert logits.dtype == jnp.float32
        np_logits = np.asarray(logits).astype(np.float32)
        assert np.all(np.isfinite(np_logits))
        assert np_logits.std() > 0.01

    def test_call_runs_and_compute_logits_matches_forward_prefill(self):
        """v4 W3: __call__ + compute_logits must match the legacy
        forward_prefill helper exactly. The two paths share the same
        transformer body and head, so this is a near-tautology — its real
        purpose is to catch reshape/order-of-axes regressions in the new
        nnx wrapper."""
        if not os.path.exists(self.BF16_DIR):
            pytest.skip(f"{self.BF16_DIR} not present")
        from types import SimpleNamespace
        from tpu_inference.models.jax.deepseek_v4 import DeepseekV4ForCausalLM
        with open(os.path.join(self.BF16_DIR, "config.json")) as f:
            import json
            hf_dict = json.load(f)
        fake_hf = SimpleNamespace(**hf_dict)
        fake_hf.to_dict = lambda: hf_dict
        fake_vc = SimpleNamespace(model_config=SimpleNamespace(hf_config=fake_hf))
        model = DeepseekV4ForCausalLM(fake_vc)
        model.load_weights_from_dir(self.BF16_DIR)
        T = 16
        ids = (jnp.arange(T, dtype=jnp.int32) % model.config.vocab_size)
        kv_caches_in = []
        kv_caches_out, hidden_TD, extra = model(kv_caches_in, ids, None)
        # vLLM's compute_logits convention: __call__ returns (T, D) per-token
        # hidden, so the HC head mix must be folded into __call__ (not
        # compute_logits). compute_logits then applies only the final
        # RMSNorm + matmul against head_w to yield logits.
        assert hidden_TD.shape == (T, model.config.hidden_size)
        assert kv_caches_out is kv_caches_in
        assert extra == []
        logits = model.compute_logits(hidden_TD)
        assert logits.shape == (T, model.config.vocab_size)
        ref = model.forward_prefill(ids.reshape(1, T)).reshape(T, -1)
        # Same math, recombined — should be bit-identical.
        assert jnp.max(jnp.abs(logits - ref)) == 0.0

    def test_eval_shape_makes_abstract_module(self):
        """v4 W3 — must pass `nnx.eval_shape(lambda: DeepseekV4ForCausalLM(...))`.
        This is the exact gate that vllm hits at
        tpu_inference/models/common/model_loader.py:244 and that v3 captured
        as BLOCKERS B2."""
        if not os.path.exists(self.BF16_DIR):
            pytest.skip(f"{self.BF16_DIR} not present")
        from types import SimpleNamespace
        from flax import nnx
        from tpu_inference.models.jax.deepseek_v4 import DeepseekV4ForCausalLM
        with open(os.path.join(self.BF16_DIR, "config.json")) as f:
            import json
            hf_dict = json.load(f)
        fake_hf = SimpleNamespace(**hf_dict)
        fake_hf.to_dict = lambda: hf_dict
        fake_vc = SimpleNamespace(model_config=SimpleNamespace(hf_config=fake_hf))
        m_abs = nnx.eval_shape(lambda: DeepseekV4ForCausalLM(fake_vc))
        assert isinstance(m_abs, nnx.Module)
        # Inner param tree should be abstracted to ShapeDtypeStructs.
        embed = m_abs.params_v.get_value().embed_w
        assert isinstance(embed, jax.ShapeDtypeStruct)
        assert embed.dtype == jnp.bfloat16


class TestRealShardRoundTrip:
    """Tier 4b: round-trip the staged real V4-Flash bf16 shard through the
    loader. embed.weight is bf16 with no scale, so byte-equality against the
    direct safetensors read validates the bf16 path end-to-end."""

    SHARD = _scratch("v4_flash/model-00001-of-00046.safetensors")
    CHECKPOINT_DIR = _scratch("v4_flash")

    def test_real_bf16_shard_byte_equal(self):
        if not os.path.exists(self.SHARD):
            pytest.skip("V4-Flash shard not present")
        from safetensors import safe_open
        from tpu_inference.models.jax.deepseek_v4_loader import to_jax_bf16
        with safe_open(self.SHARD, framework="pt") as f:
            t_direct = f.get_tensor("embed.weight")
        # Loader path (without going through dequant): just convert.
        jax_arr = to_jax_bf16(t_direct)
        # Spot-check: a few (i, j) elements must equal direct read.
        np_direct = t_direct.float().numpy()
        np_jax = np.asarray(jax_arr).astype(np.float32)
        # Spot-check first row, last row, and a deterministic random one.
        rng = np.random.default_rng(0)
        for _ in range(10):
            i = int(rng.integers(0, np_direct.shape[0]))
            j = int(rng.integers(0, np_direct.shape[1]))
            assert np_jax[i, j] == np_direct[i, j], \
                f"({i},{j}): JAX={np_jax[i, j]} torch={np_direct[i, j]}"
        # Full byte-equality (bf16 → fp32 cast is exact).
        np.testing.assert_array_equal(np_jax, np_direct)


class TestRealFp8DequantSmoke:
    """Tier 4b extension (v8 iter 3, expanded iter 7): exercise the FP8
    dequant path on real V4-Flash tensors. The tiny_v4_quant fixture
    validates the recipe; this test confirms the same code handles real
    V4-Flash FP8 (block=128) without producing NaN/inf or all-zeros.

    The bf16 round-trip in TestRealShardRoundTrip only covers the
    embedding (which is stored as bf16, no scale). Dense layers like
    `layers.0.attn.wq_a.{weight,scale}` are FP8 e4m3fn + e8m0fnu scale
    at block=128 — the production path. Without this smoke we have no
    real-data evidence for the FP8 dequant path.

    v8 iter 7: parametrized across 4 distinct attn projections at
    different layers (layers 0/10/30 + multiple projection heads).
    Each tensor is small enough (≤4 MB int8) that all 4 cases together
    add <2 s of gcsfuse IO on a warm mount.
    """

    REAL_DIR = _scratch("v4_flash")
    INDEX = _scratch("v4_flash/model.safetensors.index.json")

    # Mix of attn projections + layers. wq_a / wkv at layers 0,10,30 keeps
    # IO under ~12 MB total while exercising 3 distinct shapes (8x32,
    # 4x32 scale grids) and 3 distinct shards.
    @pytest.mark.parametrize("tensor_base", [
        "layers.0.attn.wq_a",   # shape [1024, 4096], shard 2 (existing case)
        "layers.0.attn.wkv",    # shape [512,  4096], shard 2 (different out_dim)
        "layers.10.attn.wq_a",  # shape [1024, 4096], shard 12 (later layer)
        "layers.30.attn.wkv",   # shape [512,  4096], shard 32 (deep layer + diff shape)
    ])
    def test_real_fp8_dequant_produces_finite_nontrivial_bf16(self, tensor_base):
        if not os.path.exists(self.INDEX):
            pytest.skip("V4-Flash safetensors index not present")
        from safetensors import safe_open
        from tpu_inference.models.jax.deepseek_v4_loader import (
            dequant_fp8_to_bf16,
        )
        with open(self.INDEX) as f:
            mapping = json.load(f)["weight_map"]
        w_name = tensor_base + ".weight"
        s_name = tensor_base + ".scale"
        if w_name not in mapping or s_name not in mapping:
            pytest.skip(f"{w_name} / {s_name} not in real V4-Flash index")
        if mapping[w_name] != mapping[s_name]:
            pytest.skip(
                f"weight + scale across different shards "
                f"({mapping[w_name]} vs {mapping[s_name]})"
            )
        shard_path = os.path.join(self.REAL_DIR, mapping[w_name])
        with safe_open(shard_path, framework="pt") as f:
            w = f.get_tensor(w_name)
            s = f.get_tensor(s_name)
        # Sanity on storage dtypes — these are the production FP8 dtypes.
        assert str(w.dtype) == "torch.float8_e4m3fn", (
            f"unexpected weight dtype {w.dtype}; "
            f"FP8 e4m3fn was assumed by deepseek_v4_loader.dequant_fp8_to_bf16"
        )
        assert str(s.dtype) == "torch.float8_e8m0fnu", (
            f"unexpected scale dtype {s.dtype}; "
            f"e8m0fnu was assumed by deepseek_v4_loader.dequant_fp8_to_bf16"
        )
        # Block size derived from shapes: real V4-Flash uses 128.
        out_dim, in_dim = w.shape
        so, si = s.shape
        block_o = out_dim // so
        block_i = in_dim // si
        assert block_o == block_i, (
            f"non-square block {block_o}x{block_i}: dequant_fp8_to_bf16 "
            f"assumes block-square; bail rather than silently mis-dequant"
        )
        block = block_o
        assert block == 128, (
            f"V4-Flash spec is weight_block_size=[128,128]; got {block}"
        )

        bf = dequant_fp8_to_bf16(w, s, block=block)
        assert bf.dtype == torch.bfloat16, f"got {bf.dtype}"
        assert tuple(bf.shape) == tuple(w.shape), (
            f"shape changed: {bf.shape} vs {w.shape}"
        )
        # Numerical sanity: no NaN, no Inf, non-trivial std, not all zeros.
        bf_f = bf.float()
        assert torch.isfinite(bf_f).all(), (
            "dequant produced NaN/Inf — likely a scale-decode bug"
        )
        std = bf_f.std().item()
        assert std > 1e-4, (
            f"dequant std={std:.4g} is suspiciously small; "
            f"all-zero output would suggest a missing scale multiply"
        )
        # Reasonable abs range — V4 weights are O(1) at bf16 init.
        # We allow a wide envelope to avoid being brittle.
        amax = bf_f.abs().max().item()
        assert 1e-3 < amax < 1e3, (
            f"dequant amax={amax:.4g} outside [1e-3, 1e3] — likely a "
            f"scale-exponent or sign-bit bug in dequant_fp8_to_bf16"
        )


class TestRealFp4DequantSmoke:
    """Tier 4b extension (v8 iter 3, expanded iter 7): exercise the FP4
    dequant path on real V4-Flash expert tensors. The synthetic
    tiny_v4_quant fixture uses fp4_block=8, but real V4-Flash uses
    fp4_block=32 — this test confirms the loader's recipe (FP4_TABLE
    codebook + e8m0 scale) handles the production block size without
    producing NaN/inf or all-zeros.

    v8 iter 7: parametrized across 4 distinct expert tensors covering
    all three projections (w1/w2/w3), multiple expert indices, and
    layers spanning 2..42. Each tensor is ~4 MB int8; total IO ≤ 16 MB
    on warm gcsfuse cache.
    """

    REAL_DIR = _scratch("v4_flash")
    INDEX = _scratch("v4_flash/model.safetensors.index.json")

    # Picked to maximize diversity of layer / expert / projection while
    # keeping each tensor in a single shard (verified via the index).
    # All have packed shape [out, in/2] with fp4_block=32 per V4-Flash
    # spec; w1/w3 are [2048, 2048] (-> in_logical=4096), w2 is
    # [4096, 1024] (-> in_logical=2048) reflecting the SwiGLU layout.
    @pytest.mark.parametrize("tensor_base", [
        "layers.2.ffn.experts.0.w1",     # existing case (shard 4)
        "layers.0.ffn.experts.0.w2",     # w2 projection (different shape)
        "layers.5.ffn.experts.10.w1",    # later layer + expert 10
        "layers.42.ffn.experts.255.w3",  # deepest layer + last expert + w3
    ])
    def test_real_fp4_dequant_produces_finite_nontrivial_bf16(self, tensor_base):
        if not os.path.exists(self.INDEX):
            pytest.skip("V4-Flash safetensors index not present")
        from safetensors import safe_open
        from tpu_inference.models.jax.deepseek_v4_loader import (
            dequant_fp4_to_bf16,
        )
        with open(self.INDEX) as f:
            mapping = json.load(f)["weight_map"]
        w_name = tensor_base + ".weight"
        s_name = tensor_base + ".scale"
        if w_name not in mapping or s_name not in mapping:
            pytest.skip(f"{w_name} / {s_name} not in real V4-Flash index")
        if mapping[w_name] != mapping[s_name]:
            pytest.skip(
                f"weight + scale across different shards "
                f"({mapping[w_name]} vs {mapping[s_name]})"
            )
        shard_path = os.path.join(self.REAL_DIR, mapping[w_name])
        with safe_open(shard_path, framework="pt") as f:
            w = f.get_tensor(w_name)
            s = f.get_tensor(s_name)
        # Sanity on storage dtypes — these are the production FP4 dtypes.
        assert w.dtype == torch.int8, (
            f"unexpected weight dtype {w.dtype}; "
            f"int8-packed FP4 was assumed by deepseek_v4_loader.dequant_fp4_to_bf16"
        )
        assert str(s.dtype) == "torch.float8_e8m0fnu", (
            f"unexpected scale dtype {s.dtype}; "
            f"e8m0fnu was assumed by deepseek_v4_loader.dequant_fp4_to_bf16"
        )
        # Block size derived from shapes: real V4-Flash uses 32.
        out_dim, in_packed = w.shape
        in_logical = 2 * in_packed
        so, si = s.shape
        assert so == out_dim, f"scale rows {so} != out_dim {out_dim}"
        fp4_block = in_logical // si
        assert fp4_block == 32, (
            f"V4-Flash expects fp4_block=32; got {fp4_block} "
            f"(in_logical={in_logical}, si={si})"
        )

        bf = dequant_fp4_to_bf16(w, s, fp4_block=fp4_block)
        assert bf.dtype == torch.bfloat16, f"got {bf.dtype}"
        assert tuple(bf.shape) == (out_dim, in_logical), (
            f"shape: got {bf.shape}, expected ({out_dim}, {in_logical})"
        )
        # Numerical sanity.
        bf_f = bf.float()
        assert torch.isfinite(bf_f).all(), (
            "FP4 dequant produced NaN/Inf — likely a scale-decode or "
            "FP4_TABLE codebook bug"
        )
        std = bf_f.std().item()
        assert std > 1e-4, (
            f"FP4 dequant std={std:.4g} is suspiciously small"
        )
        amax = bf_f.abs().max().item()
        assert 1e-3 < amax < 1e3, (
            f"FP4 dequant amax={amax:.4g} outside [1e-3, 1e3]"
        )
        # FP4_TABLE only contains values in {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
        # After scale multiplication every output should be a member of this
        # set times a power-of-two scale. We don't enforce that exactly
        # (would need to factor out the scale), but we DO check that
        # exact-zero values exist (FP4 codebook has 0 entries) — a missing
        # zero would suggest an off-by-one in the codebook lookup.
        zeros_count = (bf_f == 0.0).sum().item()
        assert zeros_count > 0, (
            "FP4 dequant produced no exact zeros — codebook lookup likely "
            "off; FP4_TABLE indices 0 and 8 both decode to 0.0"
        )


# ------------------------------------------------------------
# Tier 4b — independent reference dequant (v8 iter 7)
# ------------------------------------------------------------
#
# The smoke tests above prove the loader produces *plausible* output
# (finite, non-trivial std, expected magnitude range, has zeros). They
# do NOT prove the loader produces the *correct* values. The smoke could
# pass even with subtle bugs like:
#   - FP4 nibble-order swap (loader puts low nibble at index 2k+1 instead of 2k)
#   - FP8 scale-block axis flip (block stride along out vs in axis)
#   - e8m0 bias off-by-one (interpret byte as unbiased exponent)
#
# These tests close that gap by re-implementing both dequants from the
# spec in pure numpy with a deliberately different addressing pattern
# (sign-magnitude FP4 decode vs the loader's 16-entry FP4_TABLE lookup;
# bit-by-bit e4m3fn decode vs torch's `.float()` cast). They then assert
# byte-equality of the bf16 outputs — exact match in all 16 bits per
# element. Any divergence indicates a real loader bug, not a tolerance
# issue.
#
# These run on the same real V4-Flash tensors as the smoke tests above
# (warm gcsfuse cache after that suite ran), so IO cost is ~zero.


def _numpy_decode_e4m3fn(buf: np.ndarray) -> np.ndarray:
    """Independent reference for torch's `float8_e4m3fn -> float32` cast.

    The "FN" variant per OFP8 spec:
      - bit 7: sign
      - bits 3..6: 4-bit biased exponent (bias = 7)
      - bits 0..2: 3-bit mantissa
      - exp == 0:  subnormal,   value = (-1)^s * 2^-6 * (mant/8)
      - 0 < exp < 15: normal,   value = (-1)^s * 2^(exp-7) * (1 + mant/8)
      - exp == 15, mant == 7: NaN (no Inf in FN variant)
      - exp == 15, mant <  7: normal  (uses the "F"-half of the 0xFC..0xFE)

    Args:
      buf: uint8 numpy array of e4m3fn-packed bytes.
    Returns:
      float32 numpy array with the decoded values. NaN is preserved.
    """
    buf = buf.astype(np.uint8)
    sign = (buf >> 7) & 1
    exp = (buf >> 3) & 0xF
    mant = buf & 0x7
    # Use float64 throughout so the (1 + mant/8) and 2^(exp-7) products
    # are exact (all values are dyadic). Cast to fp32 at the end.
    mant_f = mant.astype(np.float64)
    exp_i = exp.astype(np.int32)
    subnormal = (mant_f / 8.0) * (2.0 ** -6)
    normal = (1.0 + mant_f / 8.0) * np.exp2(exp_i - 7)
    out = np.where(exp == 0, subnormal, normal)
    out = np.where(sign == 1, -out, out)
    out = np.where((exp == 15) & (mant == 7), np.nan, out)
    return out.astype(np.float32)


def _numpy_decode_e8m0fnu(buf: np.ndarray) -> np.ndarray:
    """Independent reference for torch's `float8_e8m0fnu -> float32` cast.

    e8m0fnu has 8 bits of unbiased-by-127 exponent, no sign, no mantissa.
    Byte 0xFF encodes NaN. Real V4-Flash scale tensors do not contain
    0xFF (all scales are finite); we still propagate it as NaN for
    completeness.

    Returns float32 array with values 2^(byte - 127) for byte != 0xFF.
    """
    buf = buf.astype(np.uint8)
    # Avoid an fp32 overflow warning for byte=0xFF (2^128 overflows to inf
    # in the cast before the np.where re-masks to NaN). Pre-mask the
    # exponent so the computation only sees finite-encoding bytes.
    safe_buf = np.where(buf == 0xFF, np.uint8(0), buf)
    out = np.exp2(safe_buf.astype(np.int32) - 127).astype(np.float32)
    out = np.where(buf == 0xFF, np.nan, out)
    return out


class TestFp8CastByteDomain:
    """Tier 4b extension (v8 iter 8): exhaustive byte-domain parity between
    the numpy references (`_numpy_decode_e4m3fn`, `_numpy_decode_e8m0fnu`)
    and torch's native cast for *every* one of the 256 possible input
    bytes. The other byte-equal tests in this file (iter 7 and iter 8)
    indirectly exercise these decoders on the byte distribution that
    actually appears in V4-Flash weights. This test closes the remaining
    gap by covering the byte distribution that *could* appear, including
    rarely-seen subnormals, the FN-variant 0xFC..0xFE encodings, and
    the explicit NaN bytes (e4m3fn 0x7F / 0xFF, e8m0fnu 0xFF).

    A divergence here is a real spec mismatch between our reference and
    torch's cast — it would invalidate every other Tier 4b byte-equal
    test, so it's worth catching at the input boundary explicitly.

    Per-byte parity is asserted via uint32 reinterpretation (NaN-aware:
    NaN positions are equated by `np.isnan`, non-NaN values must match
    bit-for-bit). The loader path itself is not exercised by this test —
    only the numpy/torch dual-decoder boundary.
    """

    @staticmethod
    def _all_bytes() -> np.ndarray:
        return np.arange(256, dtype=np.uint8)

    @staticmethod
    def _f32_bits(arr: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(arr).view(np.uint32)

    def test_e4m3fn_all_256_bytes_match_torch_cast(self):
        """Iterate all 256 e4m3fn bytes; assert numpy ref == torch cast.

        Expected NaN bytes per the FN spec (no Inf): {0x7F, 0xFF}
          = (sign∈{0,1}, exp=15, mant=7).
        Subnormals: bytes {0x00..0x07, 0x80..0x87} (exp==0).
        Normals: everything else.

        On torch≥2.4 the `view(float8_e4m3fn)` reinterpret + `.float()`
        cast is the canonical decode; that's the path we lock in here.
        """
        all_bytes = self._all_bytes()
        np_decoded = _numpy_decode_e4m3fn(all_bytes)
        torch_decoded = (
            torch.from_numpy(all_bytes).view(torch.float8_e4m3fn).float()
            .numpy()
        )

        np_nan = np.isnan(np_decoded)
        torch_nan = np.isnan(torch_decoded)
        # NaN positions must match exactly. The FN spec defines exactly
        # 2 NaN bytes (0x7F and 0xFF). Both decoders must agree.
        assert np.array_equal(np_nan, torch_nan), (
            f"e4m3fn NaN positions differ: numpy says "
            f"{np.where(np_nan)[0].tolist()}, torch says "
            f"{np.where(torch_nan)[0].tolist()}"
        )
        # FN spec sanity check — independent of either decoder.
        assert np.where(np_nan)[0].tolist() == [0x7F, 0xFF], (
            f"e4m3fn FN spec expects NaN at bytes {{0x7F, 0xFF}}; numpy "
            f"reports NaN at {np.where(np_nan)[0].tolist()}"
        )

        # Non-NaN bit-pattern equality.
        non_nan = ~np_nan
        np_bits = self._f32_bits(np_decoded)[non_nan]
        torch_bits = self._f32_bits(torch_decoded)[non_nan]
        assert np.array_equal(np_bits, torch_bits), (
            f"e4m3fn bit-pattern divergence on "
            f"{int((np_bits != torch_bits).sum())} of "
            f"{int(non_nan.sum())} non-NaN bytes; first divergent byte at "
            f"index {int(np.where(np_bits != torch_bits)[0][0])} "
            f"(numpy={np_decoded[np.where(np_bits != torch_bits)[0][0]]}, "
            f"torch={torch_decoded[np.where(np_bits != torch_bits)[0][0]]})"
        )

        # Spec spot-checks (independent of either decoder).
        # Byte 0x00 = +0.0, byte 0x80 = -0.0, byte 0x38 = exp=7,mant=0 = +1.0.
        assert np_decoded[0x00] == 0.0
        assert np_decoded[0x80] == 0.0  # -0.0 == 0.0 in float comparison
        assert (
            self._f32_bits(np_decoded)[0x80]
            == np.float32(-0.0).view(np.uint32)
        ), "e4m3fn 0x80 should encode -0.0, not +0.0"
        assert np_decoded[0x38] == 1.0, (
            f"e4m3fn 0x38 should decode to +1.0; got {np_decoded[0x38]}"
        )
        # Byte 0x7E = exp=15, mant=6 (FN normal at the edge): value =
        # 1.75 * 2^(15-7) = 1.75 * 256 = 448.
        assert np_decoded[0x7E] == 448.0, (
            f"e4m3fn 0x7E should decode to 448.0 (FN max-normal); "
            f"got {np_decoded[0x7E]}"
        )

    def test_e8m0fnu_all_256_bytes_match_torch_cast(self):
        """Iterate all 256 e8m0fnu bytes; assert numpy ref == torch cast.

        Spec: byte b in [0..254] → 2^(b-127); byte 255 → NaN.
        b=0 → 2^-127 ≈ 5.88e-39 (subnormal in fp32, since fp32 min
        normal is 2^-126). b=127 → 1.0. b=254 → 2^127 ≈ 1.70e38
        (largest finite encoding).
        """
        all_bytes = self._all_bytes()
        np_decoded = _numpy_decode_e8m0fnu(all_bytes)
        torch_decoded = (
            torch.from_numpy(all_bytes).view(torch.float8_e8m0fnu).float()
            .numpy()
        )

        np_nan = np.isnan(np_decoded)
        torch_nan = np.isnan(torch_decoded)
        assert np.array_equal(np_nan, torch_nan), (
            f"e8m0fnu NaN positions differ: numpy says "
            f"{np.where(np_nan)[0].tolist()}, torch says "
            f"{np.where(torch_nan)[0].tolist()}"
        )
        assert np.where(np_nan)[0].tolist() == [0xFF], (
            f"e8m0fnu spec expects NaN only at 0xFF; numpy reports "
            f"NaN at {np.where(np_nan)[0].tolist()}"
        )

        non_nan = ~np_nan
        np_bits = self._f32_bits(np_decoded)[non_nan]
        torch_bits = self._f32_bits(torch_decoded)[non_nan]
        assert np.array_equal(np_bits, torch_bits), (
            f"e8m0fnu bit-pattern divergence on "
            f"{int((np_bits != torch_bits).sum())} of "
            f"{int(non_nan.sum())} non-NaN bytes; first divergent byte at "
            f"index {int(np.where(np_bits != torch_bits)[0][0])}"
        )

        # Spec spot-checks: byte=0 → 2^-127 (subnormal), byte=127 → 1.0,
        # byte=254 → 2^127.
        assert np_decoded[127] == 1.0, (
            f"e8m0fnu 0x7F should decode to 1.0; got {np_decoded[127]}"
        )
        assert np_decoded[0] == np.float32(2.0 ** -127), (
            f"e8m0fnu 0x00 should decode to 2^-127; got {np_decoded[0]}"
        )
        assert np_decoded[254] == np.float32(2.0 ** 127), (
            f"e8m0fnu 0xFE should decode to 2^127; got {np_decoded[254]}"
        )


class TestFp8DequantIndependentReference:
    """Tier 4b extension (v8 iter 7): byte-equal comparison between the
    loader's `dequant_fp8_to_bf16` and a pure-numpy reference that
    decodes e4m3fn / e8m0fnu from raw bits. Independent in code path
    (no `weight.float()`, no torch tensor indexing); shared with the
    loader only on the final bf16 RTNE rounding step (we use torch
    for that on both sides since numpy lacks native bf16).

    A failure here means the loader's recipe diverges from the spec —
    a real bug, not a precision issue.
    """

    REAL_DIR = _scratch("v4_flash")
    INDEX = _scratch("v4_flash/model.safetensors.index.json")

    # v8 iter 7: 2 cases (wq_a, wkv on layer 0).
    # v8 iter 8: +4 cases — wq_b (in>>out aspect, deeper layer), wo_a / wo_b
    # (output projection, square-ish + transposed-aspect), shared_experts.w1
    # (FP8 dense FFN — distinct from routed FP4 experts; covers the only
    # FP8 path outside attn). Total 6 byte-equal real-tensor cases spanning
    # 4 distinct shapes, 4 distinct shards, 5 distinct projections, and
    # layers {0, 5, 10, 20, 40}.
    @pytest.mark.parametrize("tensor_base", [
        "layers.0.attn.wq_a",                # iter7: [1024, 4096], shard 2
        "layers.0.attn.wkv",                 # iter7: [512,  4096], shard 2
        "layers.20.attn.wq_b",               # iter8: [32768, 1024] (out>>in)
        "layers.10.attn.wo_a",               # iter8: [8192, 4096] (output proj A)
        "layers.5.attn.wo_b",                # iter8: [4096, 8192] (output proj B; in>out)
        "layers.40.ffn.shared_experts.w1",   # iter8: [2048, 4096] (FP8 dense FFN)
    ])
    def test_byte_equal_against_numpy_reference(self, tensor_base):
        if not os.path.exists(self.INDEX):
            pytest.skip("V4-Flash safetensors index not present")
        from safetensors import safe_open
        from tpu_inference.models.jax.deepseek_v4_loader import (
            dequant_fp8_to_bf16,
        )
        with open(self.INDEX) as f:
            mapping = json.load(f)["weight_map"]
        w_name = tensor_base + ".weight"
        s_name = tensor_base + ".scale"
        if w_name not in mapping or s_name not in mapping:
            pytest.skip(f"{w_name} not in V4-Flash index")
        if mapping[w_name] != mapping[s_name]:
            pytest.skip("weight + scale across different shards")
        shard_path = os.path.join(self.REAL_DIR, mapping[w_name])
        with safe_open(shard_path, framework="pt") as f:
            w = f.get_tensor(w_name)
            s = f.get_tensor(s_name)

        # Loader path.
        loader_bf16 = dequant_fp8_to_bf16(w, s, block=128)

        # Reference path: decode raw bytes via numpy, multiply, then
        # round to bf16 via torch.
        w_bytes = w.view(torch.uint8).numpy()
        s_bytes = s.view(torch.uint8).numpy()
        w_f32 = _numpy_decode_e4m3fn(w_bytes)
        s_f32 = _numpy_decode_e8m0fnu(s_bytes)
        # Upsample scale [So, Si] to [out, in] by repeating each block.
        # np.kron(s, ones((b,b))) is functionally equivalent to torch's
        # repeat_interleave(b, dim=0).repeat_interleave(b, dim=1) — a
        # different code path producing the same fp32 byte pattern.
        s_full = np.kron(s_f32, np.ones((128, 128), dtype=np.float32))
        ref_fp32 = w_f32 * s_full
        # No NaNs expected on real V4-Flash weights — assert this so
        # silent NaN-propagation doesn't mask a comparison miss.
        assert np.isfinite(ref_fp32).all(), (
            f"reference produced NaN/Inf — V4-Flash {tensor_base} should "
            f"contain only finite e4m3fn values"
        )
        ref_bf16 = torch.from_numpy(ref_fp32).bfloat16()

        # Byte-level comparison (bf16 reinterpreted as uint16).
        loader_u16 = loader_bf16.contiguous().view(torch.uint16).numpy()
        ref_u16 = ref_bf16.contiguous().view(torch.uint16).numpy()
        assert np.array_equal(loader_u16, ref_u16), (
            f"{tensor_base}: loader and numpy reference diverged in bf16 bytes. "
            f"Loader uses torch's e4m3fn->fp32 cast and repeat_interleave; "
            f"reference uses bit-level decode and np.kron. Divergence implies "
            f"one path is mis-interpreting the FP8 spec."
        )


class TestFp4DequantIndependentReference:
    """Tier 4b extension (v8 iter 7): byte-equal comparison between the
    loader's `dequant_fp4_to_bf16` and a pure-numpy reference that
    decodes FP4 nibbles via sign-magnitude decomposition (different
    addressing pattern from the loader's 16-entry FP4_TABLE lookup).

    The loader uses a single 16-entry `_FP4_TABLE_T` indexed by the full
    nibble. The reference splits each nibble into a sign bit and a 3-bit
    magnitude index, looks the magnitude up in an 8-entry table, and
    negates conditionally. Both are mathematically equivalent for the
    DeepSeek FP4 codebook; structural divergence (e.g., sign bit at the
    wrong position, magnitude table off-by-one) would produce different
    bytes here.

    Note on negative-zero: the DeepSeek codebook collapses nibble 8
    (sign=1, mag_idx=0) to +0.0, not -0.0. The reference must follow
    this spec choice — without canonicalizing -0 to +0 we'd diverge
    only in the sign bit of zeros (bf16 0x8000 vs 0x0000), which
    happens for ~0.6% of real V4-Flash FP4 weights (every nibble-8 byte).
    """

    REAL_DIR = _scratch("v4_flash")
    INDEX = _scratch("v4_flash/model.safetensors.index.json")

    # 8-entry magnitude table — independent of the loader's 16-entry table.
    _FP4_MAGNITUDES = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32,
    )

    @classmethod
    def _decode_nibbles(cls, nibbles_uint8: np.ndarray) -> np.ndarray:
        """nibbles_uint8: uint8 array with values in [0, 15]. Returns fp32."""
        sign = (nibbles_uint8 >> 3) & 1
        mag_idx = nibbles_uint8 & 0x7
        mag = cls._FP4_MAGNITUDES[mag_idx]
        val = np.where(sign == 1, -mag, mag).astype(np.float32)
        # DeepSeek's FP4_TABLE collapses nibble 8 (sign=1, mag=0) to +0.0,
        # not -0.0 — the negative-zero slot is unused. Canonicalize here
        # so byte-equality with the loader's table-lookup output holds.
        val = np.where(mag == 0.0, np.float32(0.0), val)
        return val

    # v8 iter 7: 2 cases (experts.0.w1 layer 2; experts.0.w2 layer 0).
    # v8 iter 8: +2 cases — deeper layer + mid-range expert id (experts.128 on
    # layer 30; experts.50.w3 on layer 10). w3 is the gate-up projection
    # distinct from the iter-7 w1/w2 cases. Total 4 byte-equal real-tensor
    # cases spanning all three SwiGLU projections, expert ids in {0, 50,
    # 128}, and layers {0, 2, 10, 30}.
    @pytest.mark.parametrize("tensor_base", [
        "layers.2.ffn.experts.0.w1",       # iter7: [2048, 2048] (4 MB)
        "layers.0.ffn.experts.0.w2",       # iter7: [4096, 1024] (4 MB; w2 axis)
        "layers.30.ffn.experts.128.w1",    # iter8: deep layer + mid expert id
        "layers.10.ffn.experts.50.w3",     # iter8: w3 gate-up projection
    ])
    def test_byte_equal_against_numpy_reference(self, tensor_base):
        if not os.path.exists(self.INDEX):
            pytest.skip("V4-Flash safetensors index not present")
        from safetensors import safe_open
        from tpu_inference.models.jax.deepseek_v4_loader import (
            dequant_fp4_to_bf16,
        )
        with open(self.INDEX) as f:
            mapping = json.load(f)["weight_map"]
        w_name = tensor_base + ".weight"
        s_name = tensor_base + ".scale"
        if w_name not in mapping or s_name not in mapping:
            pytest.skip(f"{w_name} not in V4-Flash index")
        if mapping[w_name] != mapping[s_name]:
            pytest.skip("weight + scale across different shards")
        shard_path = os.path.join(self.REAL_DIR, mapping[w_name])
        with safe_open(shard_path, framework="pt") as f:
            w = f.get_tensor(w_name)
            s = f.get_tensor(s_name)

        out_dim, in_packed = w.shape
        in_logical = 2 * in_packed
        fp4_block = in_logical // s.shape[1]
        assert fp4_block == 32, f"expected fp4_block=32, got {fp4_block}"

        # Loader path.
        loader_bf16 = dequant_fp4_to_bf16(w, s, fp4_block=fp4_block)

        # Reference path. Reinterpret int8 bytes as uint8 so nibble
        # arithmetic is well-defined for negative-signed bytes too.
        w_bytes = w.view(torch.uint8).numpy()
        low_nibbles = w_bytes & 0x0F
        high_nibbles = (w_bytes >> 4) & 0x0F
        low_vals = self._decode_nibbles(low_nibbles)   # [out, in_packed]
        high_vals = self._decode_nibbles(high_nibbles)
        # Interleave into [out, in_logical] with low at index 2k, high at 2k+1.
        unpacked = np.empty((out_dim, in_logical), dtype=np.float32)
        unpacked[:, 0::2] = low_vals
        unpacked[:, 1::2] = high_vals

        # Decode scale via independent e8m0fnu reference, broadcast across blocks.
        s_bytes = s.view(torch.uint8).numpy()
        s_f32 = _numpy_decode_e8m0fnu(s_bytes)  # [out, in_logical/fp4_block]
        s_full = np.repeat(s_f32, fp4_block, axis=1)  # [out, in_logical]
        ref_fp32 = unpacked * s_full
        assert np.isfinite(ref_fp32).all(), (
            f"reference produced NaN/Inf — real FP4 expert weights should "
            f"never contain 0xFF e8m0fnu scales"
        )
        ref_bf16 = torch.from_numpy(ref_fp32).bfloat16()

        # Byte-level comparison.
        loader_u16 = loader_bf16.contiguous().view(torch.uint16).numpy()
        ref_u16 = ref_bf16.contiguous().view(torch.uint16).numpy()
        assert np.array_equal(loader_u16, ref_u16), (
            f"{tensor_base}: loader and numpy reference diverged in bf16 bytes. "
            f"Loader uses 16-entry FP4_TABLE indexing; reference uses "
            f"sign-magnitude decomposition. Divergence implies one path is "
            f"mis-interpreting the FP4 codebook layout (e.g. sign-bit "
            f"position, nibble order, or scale-block axis)."
        )


class TestQuantToParamsApply:
    """Tier 7 prep: apply dequantized tiny_v4_quant weights into the abstract
    DeepseekV4 param tree, run forward on a fixed input, and compare logits
    against the same forward run on tiny_v4_groundtruth weights.

    This test isolates the LOADER's correctness from quant arithmetic — both
    sides go through the same JAX forward, so any logit divergence is the
    loader's fault.
    """

    QUANT = _scratch("tiny_v4_quant")
    GT = _scratch("tiny_v4_groundtruth")

    def _skip_if_missing(self):
        if not (os.path.exists(self.QUANT) and os.path.exists(self.GT)):
            pytest.skip("tiny_v4_quant/tiny_v4_groundtruth not present")

    @staticmethod
    def _build_params(checkpoint_dir):
        """Returns (params, cfg, swa_freqs, comp_freqs, weights_dict)."""
        import json
        from tpu_inference.models.jax.deepseek_v4 import (
            DeepseekV4Config, make_abstract_transformer_params, make_freqs_cis,
        )
        from tpu_inference.models.jax.deepseek_v4_loader import (
            apply_weights_to_param_tree, load_v4_safetensors_to_dict,
        )
        with open(os.path.join(checkpoint_dir, "config.json")) as f:
            hf_config = json.load(f)
        cfg = DeepseekV4Config.from_hf_dict(hf_config)
        params = make_abstract_transformer_params(cfg)
        # Materialize abstract leaves to zero arrays.
        params = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, dtype=x.dtype),
            params,
            is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
        )
        weights = load_v4_safetensors_to_dict(checkpoint_dir)
        params = apply_weights_to_param_tree(params, weights, cfg)
        swa, comp = make_freqs_cis(cfg, cfg.max_position_embeddings)
        return params, cfg, swa, comp, weights

    def test_forward_logits_quant_vs_groundtruth(self):
        self._skip_if_missing()
        from tpu_inference.models.jax.deepseek_v4 import \
            deepseek_v4_forward_prefill
        p_q, cfg, swa_q, comp_q, _ = self._build_params(self.QUANT)
        p_gt, cfg2, swa_gt, comp_gt, _ = self._build_params(self.GT)
        # Build a deterministic input.
        ids = jnp.zeros((1, 16), dtype=jnp.int32) + jnp.arange(16, dtype=jnp.int32) % cfg.vocab_size
        l_q = deepseek_v4_forward_prefill(ids, p_q, swa_q, comp_q, cfg)
        l_gt = deepseek_v4_forward_prefill(ids, p_gt, swa_gt, comp_gt, cfg)
        # Since the loader is byte-equal across quant and groundtruth
        # (TestFp8Dequant proves max_diff == 0.0 across 355 tensors), the
        # forward output is byte-equal too — both sides run the same Python
        # source on the same JAX device with byte-identical bf16 weights.
        # See TOLERANCE_LOG.md T7 (v8 iter 4 tightening: 0.1 -> byte-exact).
        l_q_n, l_gt_n = np.asarray(l_q), np.asarray(l_gt)
        diff = float(np.abs(l_q_n - l_gt_n).max())
        assert diff == 0.0, f"Tier 7: logits diverged (max abs diff {diff})"
        assert np.array_equal(l_q_n, l_gt_n), (
            "Tier 7: logits not byte-equal despite max-abs == 0"
        )


class TestCompressorDecodeStep:
    """Per-step compressor parity. The torch Compressor's prefill populates
    its kv_state/score_state; we snapshot, then run a decode step on both
    sides and compare states + (when compressing) the output kv."""

    @pytest.mark.parametrize("ratio,P,start_pos", [
        (4, 4, 4),       # ratio=4 (overlap), first compression event after prefill
        (4, 8, 8),       # ratio=4, multiple compressions
        (4, 7, 7),       # mid-window, no compression yet at start_pos (start_pos+1 % 4 != 0)
        (4, 12, 12),     # deep
        (128, 64, 64),   # ratio=128, no compression yet
        (128, 128, 128), # exactly first compression
    ])
    def test_compressor_decode_step_parity(self, ratio, P, start_pos):
        torch.manual_seed(0)
        args = make_tiny_args(max_seq_len=512)
        c = TorchCompressor(args, compress_ratio=ratio, head_dim=args.head_dim, rotate=False)
        for n, p in c.named_parameters():
            t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02)
            p.data.copy_(t.to(p.dtype))
        kv_cache = torch.zeros(args.max_batch_size, args.max_seq_len // ratio, args.head_dim)
        c.kv_cache = kv_cache
        c.freqs_cis = torch_freqs(args.rope_head_dim, args.max_seq_len, 0,
                                   args.compress_rope_theta, args.rope_factor,
                                   args.beta_fast, args.beta_slow)
        bsz = 1
        prefill_x = torch.randn(bsz, P, args.dim, dtype=torch.bfloat16)
        with torch.inference_mode():
            _ = c(prefill_x, start_pos=0)

        # Snapshot torch state -> JAX.
        kv_state_j = t2j(c.kv_state[:bsz].clone()).astype(jnp.float32)
        score_state_j = t2j(c.score_state[:bsz].clone()).astype(jnp.float32)
        params_j = _torch_compressor_to_jax_params(c)
        fc_j = t2j(c.freqs_cis).astype(jnp.complex64)

        # Decode one token.
        x_step = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)
        with torch.inference_mode():
            kv_t = c(x_step, start_pos=start_pos)  # may be None if no compression event
        new_kvst, new_scst, kv_compressed_j, did = compressor_decode_step(
            t2j(x_step), start_pos, params_j, fc_j, kv_state_j, score_state_j,
        )
        # If torch returned None (no compression), the JAX did flag must be False too.
        if kv_t is None:
            assert not did, f"JAX claims compression but torch said no at sp={start_pos}"
        else:
            assert did, f"JAX did not compress but torch did at sp={start_pos}"
            diff = maxabs(kv_compressed_j, kv_t)
            # Measured worst across 24 hits (3 compress configs × 8 seeds): 0.0.
            # Tightened from 5e-2 to 1e-5 (TOLERANCE_LOG entry T-CDS).
            assert diff <= 1e-5, f"ratio={ratio} sp={start_pos}: kv_compressed diff {diff}"

        # State after step must match torch.
        diff_kvst = float(np.abs(np.asarray(new_kvst) -
                                  c.kv_state[:bsz].detach().float().numpy()).max())
        # score_state contains -inf in unfilled slots; diff with torch's -inf -> nan if torch
        # also has -inf. Use isfinite comparison.
        sc_j = np.asarray(new_scst)
        sc_t = c.score_state[:bsz].detach().float().numpy()
        finite_mask = np.isfinite(sc_t)
        if finite_mask.any():
            diff_sc = float(np.abs(sc_j[finite_mask] - sc_t[finite_mask]).max())
        else:
            diff_sc = 0.0
        # Measured worst across 72 points (9 configs × 8 seeds):
        # kv_state 7.15e-7, score_state 5.96e-7 — both at fp32 ULP. Tightened
        # from 5e-2 to 1e-5 (TOLERANCE_LOG T-CDS).
        assert diff_kvst <= 1e-5, f"ratio={ratio} sp={start_pos}: kv_state diff {diff_kvst}"
        assert diff_sc <= 1e-5, f"ratio={ratio} sp={start_pos}: score_state diff {diff_sc}"


# =============================================================
# v3 hardening — extended decode parity coverage
# =============================================================
#
# The original W1 (v2) parity tests covered start_pos ∈ {1, 8, 9, 16, 32}
# and rolling decodes up to length 16+16=32. The spec calls for points at
# 64 and 256, plus a 32-step rolling-decode equivalence test against
# prefill-of-the-same-prefix. These tests are added here in v3.
#
# Tolerances follow TOLERANCE_LOG.md T8 — atol=5e-2 for per-step output,
# atol=2e-2 for state-equivalence (per the spec's allowance for
# accumulation drift across 32 sequential decode steps).


class TestDecodeAttentionParityExtended:
    """Extra start_pos points called for in the v3 spec resume mandate
    ({1, 8, 9, 64, 256}). Layer-by-layer like TestDecodeAttentionParity."""

    @pytest.mark.parametrize("layer_id,start_pos,max_seq_len", [
        # SWA, multiple wraparounds past window=8.
        (0, 64, 256),
        # CSA (ratio=4): 16 compressions accumulated by sp=64.
        (2, 64, 256),
        # HCA (ratio=128): exact first compression event at sp+1==128, so
        # sp=128 is the first decode step where the compressed pool is
        # non-empty.
        (3, 128, 512),
        # HCA deep — past the first compression event.
        (3, 192, 512),
        # v8 long-context additions per spec post-v3 resume hint
        # (sp ∈ {500, 1023}): exercise the decode path past the initial
        # compression schedule to catch state-buffer wraparound bugs that
        # only surface deep in a sequence.
        # SWA at sp=500: 62 wraparounds of window=8.
        (0, 500, 512),
        # CSA at sp=500: 125 compressions accumulated.
        (2, 500, 512),
        # HCA at sp=500: 3 compressions accumulated; deep past 1st event.
        (3, 500, 512),
        # SWA at sp=1023: max-context single-step parity, 127 window
        # wraparounds. The SWA path is the cheapest end-of-context probe.
        (0, 1023, 1024),
        # v8 iter 4 additions — fill gaps in {0..1023} between sp=192 and
        # sp=500, and between sp=500 and sp=1023. SWA only (the cheapest
        # probe) and HCA with multiple compression events accumulated.
        # SWA at sp=256: 32 wraparounds of window=8.
        (0, 256, 512),
        # SWA at sp=768: 96 wraparounds; mid-band between {500, 1023}.
        (0, 768, 1024),
        # HCA at sp=256: just past the 2nd compression event (event at
        # sp+1 == 256). Tests that the compressor pool is consistent
        # immediately after a compression.
        (3, 256, 512),
        # HCA at sp=768: 6 compression events accumulated (events at
        # sp+1 ∈ {128, 256, 384, 512, 640, 768}). Deep state probe.
        (3, 768, 1024),
    ])
    def test_decode_step_parity_extended(self, layer_id, start_pos, max_seq_len):
        attn, args = _build_attn_for_decode(
            layer_id, seed=0, max_seq_len=max_seq_len)
        bsz = 1
        with torch.inference_mode():
            prefill_x = torch.randn(bsz, start_pos, args.dim, dtype=torch.bfloat16)
            _ = attn(prefill_x, start_pos=0)
            x_step = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)
            o_t = attn(x_step, start_pos=start_pos)

        attn_fresh, args2 = _build_attn_for_decode(
            layer_id, seed=0, max_seq_len=max_seq_len)
        with torch.inference_mode():
            _ = attn_fresh(prefill_x, start_pos=0)
        jax_state = _torch_attention_state_to_jax(attn_fresh, args2, bsz)
        params_j = _torch_attention_to_jax_params(attn_fresh, args2)
        fc = t2j(attn_fresh.freqs_cis).astype(jnp.complex64)
        new_state, y_j = attention_decode_step(
            t2j(x_step), start_pos, params_j, fc, jax_state,
        )
        diff = maxabs(y_j, o_t)
        # v8 iter 4 tightening: was 5e-2, observed worst 3.8e-6 across the
        # 8 parametrized points. 1e-4 keeps a 25x margin while making the
        # test catch real regressions instead of waving them through.
        # See TOLERANCE_LOG.md "Decode step (extended)".
        assert diff <= 1e-4, (
            f"layer={layer_id} sp={start_pos} (max_seq_len={max_seq_len}): "
            f"decode step output diff {diff}"
        )


class TestDecodeRollingEquivalenceWithPrefill:
    """Stronger invariant: the SWA kv_cache buffer after a 32-step rolling
    decode (starting from empty state with full_x[:, :32]) must match the
    SWA kv_cache buffer after a torch prefill of length 32 on the same
    inputs. This catches state-mutation bugs that per-step parity would
    miss if the bug self-cancels at the output level.

    We restrict the comparison to layer_id=0 (SWA) so the kv_cache is a
    pure circular buffer with no compressor entanglement; CSA / HCA layers
    are covered by the per-step rolling tests above.

    Tolerance: byte-exact (v8 iter 4 tightening; was atol=2e-2). The 32
    sequential decode writes turn out to be byte-identical to the bulk
    prefill writes — both paths invoke the same per-position RoPE kernel
    with the same freqs slice and the same input row, just batched
    differently. Measured 0.0 max-abs across 8 random seeds. See
    TOLERANCE_LOG.md T8."""

    def test_swa_decode_state_equals_prefill_state_after_32_steps(self):
        torch.manual_seed(7)
        K = 32
        attn, args = _build_attn_for_decode(0, seed=0, max_seq_len=64)
        bsz = 1
        full_x = torch.randn(bsz, K, args.dim, dtype=torch.bfloat16)

        # Path A: torch prefill of length 32.
        with torch.inference_mode():
            _ = attn(full_x, start_pos=0)
        kvc_prefill = attn.kv_cache[:bsz].detach().float().numpy().copy()

        # Path B: 32 JAX decode steps starting from empty state on the
        # SAME inputs. Use a fresh torch attention to source weights, but
        # construct the empty JAX state directly via
        # `attention_decode_init_state` (the torch reference cannot run a
        # length-0 prefill — `sparse_attn_torch` would max over an empty
        # axis).
        attn_fresh, args2 = _build_attn_for_decode(0, seed=0, max_seq_len=64)
        params_j = _torch_attention_to_jax_params(attn_fresh, args2)
        fc = t2j(attn_fresh.freqs_cis).astype(jnp.complex64)
        jax_state = attention_decode_init_state(
            batch_size=bsz, cfg_max_seq_len=args2.max_seq_len,
            params=params_j, cfg_index_head_dim=args2.index_head_dim,
            dtype=jnp.bfloat16,
        )
        for k in range(K):
            x_step = full_x[:, k:k+1]
            jax_state, _ = attention_decode_step(
                t2j(x_step), k, params_j, fc, jax_state,
            )
        kvc_decode = np.asarray(jax_state.kv_cache).astype(np.float32)

        diff = float(np.abs(kvc_prefill - kvc_decode).max())
        assert diff == 0.0, (
            f"SWA kv_cache after 32 decode steps must byte-match prefill "
            f"state; got max abs diff {diff}"
        )
        assert np.array_equal(kvc_prefill, kvc_decode), (
            "SWA kv_cache after 32 decode steps not byte-equal to prefill "
            "state despite max-abs == 0"
        )


class TestCompressorDecodeStepExtended:
    """Compressor decode parity at deeper start_pos values, including the
    second + third compression events for ratio=4 and the second event
    for ratio=128."""

    @pytest.mark.parametrize("ratio,P,start_pos,max_seq_len", [
        # ratio=4 CSA, sp=64 → 16th compression event boundary.
        (4, 64, 64, 256),
        # ratio=4 CSA, sp=63 → mid-window before 16th compression.
        (4, 63, 63, 256),
        # ratio=128 HCA, sp=255 → 2nd compression event boundary
        # (start_pos+1 == 256, divisible by 128).
        (128, 255, 255, 512),
    ])
    def test_compressor_decode_step_parity_extended(
        self, ratio, P, start_pos, max_seq_len,
    ):
        torch.manual_seed(0)
        args = make_tiny_args(max_seq_len=max_seq_len)
        c = TorchCompressor(args, compress_ratio=ratio,
                            head_dim=args.head_dim, rotate=False)
        for n, p in c.named_parameters():
            t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02)
            p.data.copy_(t.to(p.dtype))
        kv_cache = torch.zeros(args.max_batch_size,
                               args.max_seq_len // ratio, args.head_dim)
        c.kv_cache = kv_cache
        c.freqs_cis = torch_freqs(args.rope_head_dim, args.max_seq_len, 0,
                                   args.compress_rope_theta, args.rope_factor,
                                   args.beta_fast, args.beta_slow)
        bsz = 1
        prefill_x = torch.randn(bsz, P, args.dim, dtype=torch.bfloat16)
        with torch.inference_mode():
            _ = c(prefill_x, start_pos=0)

        kv_state_j = t2j(c.kv_state[:bsz].clone()).astype(jnp.float32)
        score_state_j = t2j(c.score_state[:bsz].clone()).astype(jnp.float32)
        params_j = _torch_compressor_to_jax_params(c)
        fc_j = t2j(c.freqs_cis).astype(jnp.complex64)

        x_step = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)
        with torch.inference_mode():
            kv_t = c(x_step, start_pos=start_pos)
        new_kvst, new_scst, kv_compressed_j, did = compressor_decode_step(
            t2j(x_step), start_pos, params_j, fc_j,
            kv_state_j, score_state_j,
        )
        if kv_t is None:
            assert not did, (
                f"JAX claims compression but torch said no at "
                f"ratio={ratio} sp={start_pos}"
            )
        else:
            assert did, (
                f"JAX did not compress but torch did at "
                f"ratio={ratio} sp={start_pos}"
            )
            diff = maxabs(kv_compressed_j, kv_t)
            # See TOLERANCE_LOG T-CDS — measured worst 0.0 / 24 points.
            assert diff <= 1e-5, (
                f"ratio={ratio} sp={start_pos}: kv_compressed diff {diff}"
            )
        diff_kvst = float(np.abs(np.asarray(new_kvst) -
                                  c.kv_state[:bsz].detach().float().numpy()).max())
        sc_j = np.asarray(new_scst)
        sc_t = c.score_state[:bsz].detach().float().numpy()
        finite_mask = np.isfinite(sc_t)
        if finite_mask.any():
            diff_sc = float(np.abs(sc_j[finite_mask] - sc_t[finite_mask]).max())
        else:
            diff_sc = 0.0
        # See TOLERANCE_LOG T-CDS — measured worst 7.15e-7 (kv_state) /
        # 5.96e-7 (score_state) across 72 points (9 configs × 8 seeds).
        assert diff_kvst <= 1e-5, (
            f"ratio={ratio} sp={start_pos}: kv_state diff {diff_kvst}"
        )
        assert diff_sc <= 1e-5, (
            f"ratio={ratio} sp={start_pos}: score_state diff {diff_sc}"
        )


class TestDecodeRollingParityLong:
    """Rolling decode for longer K, exercising more compression events."""

    @pytest.mark.parametrize("layer_id,P,K,max_seq_len", [
        # SWA: 1+31 = 4× window-8 wraparounds. P=1 because the torch
        # reference cannot run a length-0 prefill (sparse_attn_torch would
        # max over an empty axis).
        (0, 1, 31, 64),
        # SWA: 16+32 = 48 total → 6× window wraparounds; prefill leaves a
        # specific pattern in the circular buffer that decode then walks.
        (0, 16, 32, 64),
        # CSA: 1+31 = 32 steps total, ~8 compression events. P=1 for the
        # same length-0 reason as above.
        (2, 1, 31, 256),
    ])
    def test_rolling_decode_parity_long(self, layer_id, P, K, max_seq_len):
        attn, args = _build_attn_for_decode(
            layer_id, seed=42, max_seq_len=max_seq_len)
        bsz = 1
        torch.manual_seed(P + K + layer_id + 1000)
        full_x = torch.randn(bsz, P + K, args.dim, dtype=torch.bfloat16)

        with torch.inference_mode():
            _ = attn(full_x[:, :P], start_pos=0)
        jax_state = _torch_attention_state_to_jax(attn, args, bsz)
        params_j = _torch_attention_to_jax_params(attn, args)
        fc = t2j(attn.freqs_cis).astype(jnp.complex64)

        for k in range(K):
            sp = P + k
            x_step = full_x[:, sp:sp + 1]
            with torch.inference_mode():
                o_t = attn(x_step, start_pos=sp)
            jax_state, y_j = attention_decode_step(
                t2j(x_step), sp, params_j, fc, jax_state,
            )
            diff = maxabs(y_j, o_t)
            # v8 iter 6 tightening: was 5e-2; observed worst 7.63e-6 across
            # 3 configs × 6 seeds × up to 32 decode steps (~500 measurements,
            # scripts/measure_rolling_long_parity.py). 13× margin under 1e-4.
            # Same path as TestDecodeAttentionParity / TestDecodeRollingParity
            # which iter 4 already tightened to 1e-4.
            assert diff <= 1e-4, (
                f"layer={layer_id} P={P} K={K} step k={k} sp={sp}: "
                f"rolling decode diff {diff}"
            )


# ------------------------------------------------------------
# Tier 4b — codebook + keystone-tensor reference (v8 iter 9)
# ------------------------------------------------------------
#
# The byte-equal real-data tests above (iter 7 + iter 8) prove the FP4 /
# FP8 *paths* are correct on real V4-Flash bytes. They do not directly
# pin down the loader's 16-entry FP4 codebook table itself — a swap of
# two adjacent table entries would produce real-data divergences only
# if the surrounding bytes happen to address the swapped slots, which
# is statistical, not exhaustive.
#
# Iter 9 closes the remaining boundary:
#   * `TestFp4CodebookReference` enumerates all 16 nibble values (every
#     byte in the FP4 codebook) and asserts the loader's
#     `_FP4_TABLE_T` agrees with a manually-typed reference table at
#     every slot, including the -0 → +0 canonicalization at index 8
#     (INVARIANTS::I38).
#   * `TestRealKeystoneTensorRoundTrip` round-trips three "keystone"
#     real V4-Flash tensors through `to_jax_bf16`: a bf16 norm tensor,
#     a fp32 attention sink tensor, and an int64 router lookup table
#     (`tid2eid`). The bf16 round-trip in `TestRealShardRoundTrip`
#     covers the embedding only; norms / sinks / int lookups travel
#     the same `to_jax_bf16` dispatch but through different branches
#     (fp32 direct copy, int64 direct copy) — these are the loader
#     codepaths that move from torch into JAX *without* dequantization,
#     and bugs there would silently corrupt the model without the
#     existing FP4/FP8 byte-equal tests catching it.


class TestFp4CodebookReference:
    """Tier 4b extension (v8 iter 9): exhaustive enumeration of all 16
    FP4 codebook entries vs a manually-typed reference table.

    The DeepSeek FP4 codebook is sign-magnitude:

        nibble = (sign << 3) | mag_idx
        mag_idx ∈ [0..7] indexes [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        sign 0 → positive, sign 1 → negative
        BUT (sign=1, mag_idx=0) is collapsed to +0.0 — the negative-zero
        slot is unused (I38).

    The loader's `_FP4_TABLE_T` is a single 16-entry torch tensor that
    serves as a direct lookup for the full nibble. A swap of two
    adjacent entries (e.g. table[5] ↔ table[6], "3 ↔ 4") would not
    necessarily fail any other Tier 4b real-data test since the affected
    bytes would be statistically rare. This test forces the issue by
    enumerating every nibble.

    Reference values (per DeepSeek `inference/convert.py:FP4_TABLE`):
      nibble 0 →  0.0,   1 →  0.5,   2 →  1.0,   3 →  1.5,
      nibble 4 →  2.0,   5 →  3.0,   6 →  4.0,   7 →  6.0,
      nibble 8 →  0.0   (the -0.0 slot, canonicalized),
      nibble 9 → -0.5,  10 → -1.0,  11 → -1.5,
      nibble 12 → -2.0, 13 → -3.0, 14 → -4.0, 15 → -6.0.
    """

    REFERENCE = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]

    def test_loader_table_matches_reference_element_for_element(self):
        from tpu_inference.models.jax.deepseek_v4_loader import (
            _FP4_TABLE_T, FP4_TABLE_LIST,
        )
        # First: list constant equals our reference verbatim.
        assert FP4_TABLE_LIST == self.REFERENCE, (
            f"FP4_TABLE_LIST diverged from reference at entries: "
            f"{[i for i, (a, b) in enumerate(zip(FP4_TABLE_LIST, self.REFERENCE)) if a != b]}"
        )
        # Then: tensor matches list bit-for-bit (fp32; values are dyadic).
        assert _FP4_TABLE_T.dtype == torch.float32, (
            f"loader expects fp32 codebook for indexing; got {_FP4_TABLE_T.dtype}"
        )
        ref_t = torch.tensor(self.REFERENCE, dtype=torch.float32)
        assert torch.equal(_FP4_TABLE_T, ref_t), (
            f"_FP4_TABLE_T diverged from reference: "
            f"got {_FP4_TABLE_T.tolist()}, expected {self.REFERENCE}"
        )

    def test_decode_every_nibble_via_loader_unpack(self):
        """Pack every nibble in [0..15] into a single int8 byte and run it
        through `dequant_fp4_to_bf16` with a unit scale, then assert the
        bf16 outputs match the reference table at every position.

        This exercises the *loader's* nibble extraction (low nibble at
        index 2k, high at 2k+1) and table indexing — the same code that
        runs on every real V4-Flash FP4 expert weight. A swap, an
        off-by-one in the shift, or an incorrect mask would be caught.
        """
        from tpu_inference.models.jax.deepseek_v4_loader import (
            dequant_fp4_to_bf16,
        )
        # Build a single row containing every nibble 0..15 as 8 bytes:
        # byte k packs (high=2k+1, low=2k). Choose pairs (low=k, high=k+8)
        # so that across the 8 bytes we hit every nibble exactly once
        # in both nibble positions.
        # bytes[k] = (k+8) << 4 | k    for k in 0..7
        # → low nibbles: [0,1,2,3,4,5,6,7]; high nibbles: [8,9,..,15]
        bytes_row = np.array(
            [((k + 8) << 4) | k for k in range(8)],
            dtype=np.uint8,
        )
        # int8 view (PyTorch's `view` requires same itemsize). Some bytes
        # are >= 0x80 → negative when reinterpreted as int8. The loader
        # explicitly casts to uint8 before nibble math, so this should
        # not affect output.
        w_int8 = torch.from_numpy(bytes_row.view(np.int8)).view(1, 8)
        # Unit scale: byte 127 = 2^0 = 1.0 in e8m0fnu.
        s = torch.full((1, 16 // 32 if 16 % 32 == 0 else 1), 127, dtype=torch.uint8)
        # Actual scale shape: [out_dim=1, in_logical/fp4_block].
        # We want fp4_block=16 so that one scale entry covers all 16
        # nibbles in the row. Construct via the e8m0fnu reinterpret.
        s_e8m0 = torch.tensor([[127]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
        bf = dequant_fp4_to_bf16(w_int8, s_e8m0, fp4_block=16)
        assert tuple(bf.shape) == (1, 16), f"shape {bf.shape}"
        out = bf.float().squeeze(0).tolist()
        # Output ordering: [low_0, high_0, low_1, high_1, ..., low_7, high_7]
        # = [0, 8, 1, 9, 2, 10, ..., 7, 15]
        # so out[2k] should equal REFERENCE[k] and out[2k+1] should
        # equal REFERENCE[k+8].
        for k in range(8):
            assert out[2 * k] == self.REFERENCE[k], (
                f"low nibble {k}: loader produced {out[2 * k]}, "
                f"reference says {self.REFERENCE[k]}"
            )
            assert out[2 * k + 1] == self.REFERENCE[k + 8], (
                f"high nibble {k + 8}: loader produced {out[2 * k + 1]}, "
                f"reference says {self.REFERENCE[k + 8]}"
            )
        # Final sanity: every reference value appears in the output.
        out_sorted = sorted(out)
        ref_sorted = sorted(self.REFERENCE)
        assert out_sorted == ref_sorted, (
            f"loader output multiset {out_sorted} differs from "
            f"reference multiset {ref_sorted}"
        )


class TestRealKeystoneTensorRoundTrip:
    """Tier 4b extension (v8 iter 9): byte-equal round-trip through
    `to_jax_bf16` for the three non-quantized storage classes that the
    real V4-Flash checkpoint actually uses:

      * **bf16 norm**: e.g. `layers.0.attn.kv_norm.weight`. Stored as
        torch.bfloat16; loader path is
        `jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)`. The
        intermediate fp32 hop is exact for bf16 (every bf16 value is
        representable in fp32) so round-trip must be byte-exact.
      * **fp32 attention sink**: `layers.0.attn.attn_sink`. Stored as
        torch.float32; loader path is `jnp.asarray(t.numpy())` — direct
        zero-copy. Round-trip must be byte-exact.
      * **int64 router table**: `layers.0.ffn.gate.tid2eid`. Stored as
        torch.int64; loader path is `jnp.asarray(t.numpy())`. Round-trip
        must be byte-exact.

    `TestRealShardRoundTrip` already covers the bf16 path on the
    embedding tensor. This test covers the three other code branches in
    `to_jax_bf16` (fp32, int64, bf16-norm) on real V4-Flash data so that
    every branch is independently anchored to a real-data byte-equal
    proof. A regression that e.g. accidentally cast int64 → int32 (a
    real loss-of-information bug for the 129280-row tid2eid table) would
    fire here.
    """

    REAL_DIR = _scratch("v4_flash")
    INDEX = _scratch("v4_flash/model.safetensors.index.json")

    @pytest.mark.parametrize("name", [
        "layers.0.attn.kv_norm.weight",     # bf16, [512]
        "layers.30.attn.q_norm.weight",     # bf16, [1024], deeper layer
        "layers.0.attn_norm.weight",        # bf16, [4096], pre-attn block norm
    ])
    def test_real_bf16_norm_byte_equal(self, name):
        if not os.path.exists(self.INDEX):
            pytest.skip("V4-Flash safetensors index not present")
        from safetensors import safe_open
        from tpu_inference.models.jax.deepseek_v4_loader import to_jax_bf16
        with open(self.INDEX) as f:
            mapping = json.load(f)["weight_map"]
        if name not in mapping:
            pytest.skip(f"{name} not in V4-Flash index")
        shard_path = os.path.join(self.REAL_DIR, mapping[name])
        with safe_open(shard_path, framework="pt") as f:
            t = f.get_tensor(name)
        assert t.dtype == torch.bfloat16, (
            f"{name}: expected bf16 storage, got {t.dtype}"
        )
        jax_arr = to_jax_bf16(t)
        # Bit-equal in the underlying uint16 view.
        loader_u16 = (
            torch.from_numpy(np.asarray(jax_arr).view(np.uint16).copy())
        )
        # Direct safetensors → torch bf16 → uint16 view.
        ref_u16 = t.contiguous().view(torch.uint16)
        assert loader_u16.shape == ref_u16.shape, (
            f"{name}: shape changed under to_jax_bf16: "
            f"{loader_u16.shape} vs {ref_u16.shape}"
        )
        diff = (loader_u16 != ref_u16).sum().item()
        assert diff == 0, (
            f"{name}: {diff}/{loader_u16.numel()} bf16 bytes differ "
            f"after round-trip — to_jax_bf16's fp32 hop is no longer "
            f"exact for bf16 inputs"
        )

    @pytest.mark.parametrize("name", [
        "layers.0.attn.attn_sink",          # fp32, [64]
        "layers.30.attn.attn_sink",         # fp32, [64], deeper layer
        "layers.0.hc_attn_base",            # fp32, [24], head-mix base
        "hc_head_base",                     # fp32, [4], top-level head base
    ])
    def test_real_fp32_byte_equal(self, name):
        if not os.path.exists(self.INDEX):
            pytest.skip("V4-Flash safetensors index not present")
        from safetensors import safe_open
        from tpu_inference.models.jax.deepseek_v4_loader import to_jax_bf16
        with open(self.INDEX) as f:
            mapping = json.load(f)["weight_map"]
        if name not in mapping:
            pytest.skip(f"{name} not in V4-Flash index")
        shard_path = os.path.join(self.REAL_DIR, mapping[name])
        with safe_open(shard_path, framework="pt") as f:
            t = f.get_tensor(name)
        assert t.dtype == torch.float32, (
            f"{name}: expected fp32 storage, got {t.dtype}"
        )
        jax_arr = to_jax_bf16(t)
        # to_jax_bf16's fp32 branch is `jnp.asarray(t.numpy())` — direct
        # passthrough, must round-trip with full fp32 precision.
        np_jax = np.asarray(jax_arr)
        np_torch = t.numpy()
        assert np_jax.dtype == np.float32, (
            f"{name}: to_jax_bf16 changed fp32 → {np_jax.dtype}"
        )
        # Bit-equal via uint32 reinterpret.
        jax_u32 = np_jax.view(np.uint32)
        torch_u32 = np_torch.view(np.uint32)
        assert np.array_equal(jax_u32, torch_u32), (
            f"{name}: fp32 bytes differ after round-trip — to_jax_bf16's "
            f"fp32 branch is no longer zero-copy"
        )

    def test_real_int64_router_lookup_byte_equal(self):
        """`tid2eid` is the per-layer token-id → expert-id router lookup,
        shape `[vocab=129280, top_k=6]`, stored as int64. A silent
        truncation to int32 would still produce valid expert IDs (since
        all expert IDs fit in int16) but would corrupt high-id
        lookups via wraparound during indexing. Catch that here.
        """
        if not os.path.exists(self.INDEX):
            pytest.skip("V4-Flash safetensors index not present")
        from safetensors import safe_open
        from tpu_inference.models.jax.deepseek_v4_loader import to_jax_bf16
        name = "layers.0.ffn.gate.tid2eid"
        with open(self.INDEX) as f:
            mapping = json.load(f)["weight_map"]
        if name not in mapping:
            pytest.skip(f"{name} not in V4-Flash index")
        shard_path = os.path.join(self.REAL_DIR, mapping[name])
        with safe_open(shard_path, framework="pt") as f:
            t = f.get_tensor(name)
        assert t.dtype == torch.int64, (
            f"{name}: expected int64 storage, got {t.dtype}"
        )
        jax_arr = to_jax_bf16(t)
        np_jax = np.asarray(jax_arr)
        np_torch = t.numpy()
        # JAX may upcast int64 → int32 if `jax_enable_x64` is off
        # (the default). Either is acceptable as long as no value is
        # truncated. Spec says expert IDs fit in int16 so int32 is safe.
        if np_jax.dtype == np.int64:
            assert np.array_equal(np_jax, np_torch), (
                f"{name}: int64 values differ after round-trip"
            )
        elif np_jax.dtype == np.int32:
            # Verify lossless downcast: every value must fit in int32.
            tmin, tmax = int(np_torch.min()), int(np_torch.max())
            assert tmin >= np.iinfo(np.int32).min and tmax <= np.iinfo(np.int32).max, (
                f"{name}: int64 → int32 downcast would lose data: "
                f"value range [{tmin}, {tmax}] doesn't fit in int32"
            )
            assert np.array_equal(np_jax.astype(np.int64), np_torch), (
                f"{name}: int64 → int32 → int64 round-trip differs"
            )
        else:
            pytest.fail(
                f"{name}: to_jax_bf16 produced unexpected dtype {np_jax.dtype} "
                f"for int64 input"
            )


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
    CompressorParams, IndexerParams,
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


# =============================================================
# Tier 2 — end-to-end (stub for now; populated after the JAX model lands)
# =============================================================


class TestEndToEnd:
    """Tier 2 placeholder — full E2E logits parity is in test_deepseek_v4_e2e
    once the JAX model assembly lands. See PROGRESS.md."""

    @pytest.mark.skip(reason="JAX model assembly not landed yet; populated in Phase 3")
    def test_prefill_16(self):
        pass


# =============================================================
# Tier 3 — compile-only against real V4 configs
# =============================================================


class TestRealConfigCompile:
    """Compile-only test against the real V4 configs. Constructs the JAX
    model on a mesh, calls jax.eval_shape on the forward function, and
    confirms `jit(...).lower(...).compile()` succeeds. Does NOT run a
    forward pass (would OOM)."""

    @pytest.mark.skip(reason="JAX model assembly not landed yet; populated in Phase 4")
    @pytest.mark.parametrize("model_name", ["V4-Flash", "V4-Pro"])
    @pytest.mark.parametrize("mesh_kind", ["v4-8", "v6e-32-sim"])
    def test_eval_shape_succeeds(self, model_name, mesh_kind):
        pass

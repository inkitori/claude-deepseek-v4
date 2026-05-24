"""S1 bf16-vs-structural disambiguation (CPU, seconds).

The per-layer bisect shows scattered, position-dependent decode-vs-prefill
divergence across ALL layer types (incl. pure-SWA L1), with wildly varying
amplification. That smells like bf16 near-tie amplification, not a clean
structural indexing bug. DECISIVE TEST: rerun the same per-layer comparison in
pure fp32 (params fp32, activations fp32, decode-state fp32). If divergence
collapses to ~1e-6 -> S1-in-CPU-repro is bf16-precision-driven (the CPU repro
may NOT capture the real hard-collapse S1). If it persists -> structural.

Usage: python scripts/s1_cpu_dtype_disambig.py [dtype=fp32|bf16] [n_layers=4] [T=8] [N=12] [seed=0]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp

from s1_cpu_repro_v4flash import make_v4_flash_truncated_cfg
from s1_cpu_repro_peaked import make_scaled_params
from tpu_inference.models.jax import deepseek_v4 as D
from tpu_inference.layers.jax.attention import deepseek_v4_attention as A


def relerr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def main():
    dt_arg = sys.argv[1] if len(sys.argv) > 1 else "fp32"
    n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    DT = jnp.float32 if dt_arg == "fp32" else jnp.bfloat16

    cfg = make_v4_flash_truncated_cfg(n_layers=n_layers, n_experts=8)
    params = make_scaled_params(cfg, 0.5, seed=seed)
    if DT == jnp.float32:
        # Cast ALL weights to fp32 so the whole forward is fp32.
        params = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float32) if hasattr(x, "dtype") and
            jnp.issubdtype(x.dtype, jnp.floating) else x, params)
        # Force the decode-state dtype to fp32 everywhere it's hardcoded bf16.
        _orig = A.attention_init_state_from_prefill
        def _init_fp32(*a, **k):
            k["dtype"] = jnp.float32
            return _orig(*a, **k)
        A.attention_init_state_from_prefill = _init_fp32
        D.attention_init_state_from_prefill = _init_fp32

    max_seq = 4096
    swa, comp = D.make_freqs_cis(cfg, max_seq)
    ratios = [params.layers[i].attn.compress_ratio for i in range(n_layers)]
    print(f"Setup: dtype={dt_arg} n_layers={n_layers} ratios={ratios} T={T} N={N}")

    rng = np.random.default_rng(seed=1234)
    ids_full = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(1, T + N)),
                           dtype=jnp.int32)

    def fc_of(layer):
        return comp if layer.attn.compress_ratio > 0 else swa

    def embed(ids):
        h = params.embed_w[ids].astype(DT)
        return jnp.broadcast_to(h[:, :, None, :], (*h.shape[:2], cfg.hc_mult, h.shape[-1]))

    def ref_per_layer(ids):
        h = embed(ids)
        outs = []
        for layer in params.layers:
            h = D.block_forward(h, ids, layer, fc_of(layer))
            outs.append(h)
        return outs

    def init_states(ids):
        h = embed(ids)
        states = []
        for i, layer in enumerate(params.layers):
            cr = layer.attn.compress_ratio
            idx_hd = cfg.index_head_dim if (cr == 4 and layer.attn.indexer is not None) else 0
            st, h = D.block_init_state_and_forward(
                h, ids, layer, fc_of(layer), cfg_max_seq_len=max_seq,
                cfg_index_head_dim=idx_hd, layer_idx=i)
            states.append(st)
        return states

    def decode_step(ids_step, states, P):
        h = embed(ids_step)
        new, outs = [], []
        for i, (layer, prev) in enumerate(zip(params.layers, states)):
            ns, h = D.block_decode_step(
                h, ids_step, layer, fc_of(layer), prev, jnp.int32(P), layer_idx=i)
            new.append(ns)
            outs.append(h)
        return new, outs

    states = init_states(ids_full[:, :T])
    print("\npos  " + "".join(f" L{i}(r{ratios[i]})".rjust(12) for i in range(n_layers)))
    worst = 0.0
    n_bad = 0
    for step in range(N):
        P = T + step
        ref = ref_per_layer(ids_full[:, :P + 1])
        states, dec = decode_step(ids_full[:, P:P + 1], states, P)
        cells = [relerr(dec[i][:, 0], ref[i][:, P]) for i in range(n_layers)]
        worst = max(worst, max(cells))
        if max(cells) > 0.05:
            n_bad += 1
        row = f"{P:>3}  " + "".join(
            (("**%.4f" % c) if c > 0.05 else (" %.5f" % c)).rjust(12) for c in cells)
        print(row)

    print(f"\ndtype={dt_arg}: worst per-layer relErr={worst:.5f}, positions with any layer>0.05: {n_bad}/{N}")
    if DT == jnp.float32 and worst < 0.02:
        print("=> fp32 CLEAN: the CPU-repro divergence is bf16/precision-driven, NOT structural.")
    elif DT == jnp.float32:
        print("=> fp32 STILL diverges: STRUCTURAL bug (indexing/positions/values).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

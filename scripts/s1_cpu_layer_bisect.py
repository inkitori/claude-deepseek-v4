"""S1 per-LAYER teacher-forcing bisection (CPU, seconds).

The isolated single-layer attention decode matches prefill (s1_cpu_integration_test).
So S1 emerges only in the FULL multi-layer decode with REAL threaded activations.
This test threads the real state list (no synthetic per-layer input) and, at each
decode position P, compares the per-layer hidden state h from:
  - DECODE: block_init_state_and_forward(prefill[:T]) then block_decode_step T..P
  - REFERENCE: block_forward over the full prefix ids[:P+1], taking row P
reporting relErr per (position, layer). The first (layer, position) where relErr
blows up is the component that breaks. Also splits each block into its attention
sub-output vs the full block output to separate attention from MoE/residual.

Usage: python scripts/s1_cpu_layer_bisect.py [scale=0.5] [n_layers=4] [T=8] [N=12] [seed=0]
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


def relerr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def main():
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    cfg = make_v4_flash_truncated_cfg(n_layers=n_layers, n_experts=8)
    params = make_scaled_params(cfg, scale, seed=seed)
    max_seq = 4096
    swa, comp = D.make_freqs_cis(cfg, max_seq)
    ratios = [params.layers[i].attn.compress_ratio for i in range(n_layers)]
    print(f"Setup: n_layers={n_layers} ratios={ratios} scale={scale} T={T} N={N}")

    rng = np.random.default_rng(seed=1234)
    ids_full = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(1, T + N)),
                           dtype=jnp.int32)

    def fc_of(layer):
        return comp if layer.attn.compress_ratio > 0 else swa

    def embed(ids):
        h = params.embed_w[ids]
        return jnp.broadcast_to(h[:, :, None, :], (*h.shape[:2], cfg.hc_mult, h.shape[-1]))

    def ref_per_layer(ids):
        """block_forward over prefix; return list of per-layer outputs [B,S,hc,D]."""
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
    hdr = "pos  " + "".join(f" L{i}(r{ratios[i]})".rjust(12) for i in range(n_layers))
    print("\n=== per-layer relErr(h_decode[:,0], h_prefill_ref[:,P]) ===")
    print(hdr)
    first = None
    for step in range(N):
        P = T + step
        ref = ref_per_layer(ids_full[:, :P + 1])
        states, dec = decode_step(ids_full[:, P:P + 1], states, P)
        cells = []
        for i in range(n_layers):
            e = relerr(dec[i][:, 0], ref[i][:, P])
            cells.append(e)
            if e > 0.05 and first is None:
                first = (P, i, e)
        row = f"{P:>3}  " + "".join(
            (("**%.3f" % c) if c > 0.05 else (" %.4f" % c)).rjust(12) for c in cells)
        print(row)

    print()
    if first is None:
        print("NO per-layer divergence > 0.05 — decode matches prefill across all layers.")
    else:
        P, i, e = first
        print(f"FIRST DIVERGENCE: layer {i} (ratio={ratios[i]}) at pos {P}, relErr={e:.4f}")
        print("=> the bug enters at this layer's block_decode_step with real activations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""S1 CPU reproducer with PEAKED weights.

The existing s1_cpu_repro_v4flash.py uses normal*0.02 weights -> the compressor/
indexer internal softmaxes are ~uniform, which AVERAGES OUT any prefill-vs-decode
parity discrepancy (decode == prefill). Real trained weights make those softmaxes
PEAKED, which is the regime where S1 appears (decode collapses, prefill is fine).

This script reuses the repro machinery but scales weights by SCALE (arg) to peak
the internal softmaxes, and runs EAGER ONLY (donation is already exonerated; we
test the pure decode-vs-prefill math). If decode argmax diverges from the
fresh-prefill (transformer_body_forward) reference as SCALE grows, S1 is
reproduced on CPU -> fast fix-iteration loop.

Usage: python scripts/s1_cpu_repro_peaked.py [scale=0.5] [n_layers=4] [T=8] [N=12] [seed=0]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp

from s1_cpu_repro_v4flash import (
    make_v4_flash_truncated_cfg, run_test)
from tpu_inference.models.jax.deepseek_v4 import (
    make_abstract_transformer_params, make_freqs_cis)


def make_scaled_params(cfg, scale, seed=0):
    ps = make_abstract_transformer_params(cfg)
    leaves, treedef = jax.tree_util.tree_flatten(
        ps, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct))
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    out = [(jax.random.normal(k, x.shape, dtype=jnp.float32) * scale).astype(x.dtype)
           for k, x in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(treedef, out)


if __name__ == "__main__":
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    print(f"Setup: scale={scale} n_layers={n_layers} T={T} N={N} seed={seed}")
    cfg = make_v4_flash_truncated_cfg(n_layers=n_layers, n_experts=8)
    params = make_scaled_params(cfg, scale, seed=seed)
    max_seq = 4096
    swa, comp = make_freqs_cis(cfg, max_seq)
    n_bad, _ = run_test(f"peaked@{scale}", jit_donate=False, cfg=cfg, params=params,
                        swa=swa, comp=comp, max_seq=max_seq, T=T, N=N)
    print()
    if n_bad > 0:
        print(f"S1-LIKE REPRODUCED on CPU: decode diverges from prefill {n_bad}/{N} at scale={scale}")
        sys.exit(2)
    print(f"no divergence at scale={scale} (decode==prefill)")
    sys.exit(0)

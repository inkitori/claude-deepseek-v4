#!/usr/bin/env python3
"""TPU micro-benchmark: the PREFILL MoE rhs-prep CROSS-LAYER HBM PEAK (roadmap STEP 2).

P.15 REFUTED dual-residency: the real-model flag-OFF prefill forward `jit_run_model`
reserves ~18.34 GiB/chip (smoke log full-slice-v4-smoke-20260529T085830Z), so +16 GiB
fp8 can't coexist. STEP 2 asks: what dominates that ~18 GiB, and can it be shrunk
LOSSLESSLY (to unblock roadmap #4 token-sharded-o, currently HBM-OOM, DO-NOT-RETRY #21)?

HYPOTHESIS (DO-NOT-RETRY #21 + a code audit): the peak is the rhs-prep FP4->fp8 UNPACK
held concurrently across ALL 43 layers. The unpack (deepseek_v4_moe.py:351-358) runs
OUTSIDE the shard_map, once per layer in the UNROLLED layer loop, and depends ONLY on
the resident fp4 weights -- NOT the activation. So XLA is free to hoist all 43 unpacks
to the program start => ~43 x (W13 256MiB + W2t 128MiB + scales) ~= 18 GiB live at once.
The :350 comment "Only one layer's experts are live at a time" is the INTENT, not reality.

THE LEVER (LOSSLESS, value-identity): tie each layer's unpack to that layer's INCOMING
ACTIVATION via a bundled `jax.lax.optimization_barrier((W1,W3,W2, x))`. The barrier op
consumes x (which is on the residual chain, so depends on layer i-1) and re-emits the
weights, so the unpack transitively depends on x => can't hoist above layer i-1 => only
~1-2 layers' fp8 rhs live at once. Bit-identical (barrier is identity) => GATE-safe.

This bench A/Bs that at tier-2 (NO full-model load): build a K-layer MoE region (each
layer: fp4->fp8 unpack [the big buffers] -> the 2 gmm_v2 calls [consumer] -> residual
chain), compile it WITH and WITHOUT the per-layer barrier, and read XLA's compiled
`memory_analysis().temp_size_in_bytes` (the peak transient HBM) + min latency. If
no-barrier temp grows ~linearly in K (-> ~18 GiB @K=43) and barrier temp stays ~flat,
the lever is PROVEN before spending a smoke; the latency delta is the overlap-loss cost.

Mirrors the real path's rhs build (rhs_prep) + gmm consumer (gmm_core), imported from
perf_microbench_moe_prefill. Replicated per-rank (P()) -- per-chip peak is what matters.

Run (TPU FREE; SYNC first -- mh_run runs each host's own clone):
    scripts/full_slice_v4_sync.sh
    MH_TIMEOUT=1200 scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_moe_prefill_hbm.py --distributed
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_microbench_moe_decode import (  # noqa: E402
    DIM, INTER, E, TOP_K, N_MOE_LAYERS, measure)
from perf_microbench_moe_prefill import (  # noqa: E402
    make_fp4_local, make_replicated, rhs_prep, gmm_core)

GiB = 1024 ** 3


def build_layer_weights(mesh, K):
    """K layers of synthetic per-rank fp4 expert leaves (W1,S1,W3,S3,W2,S2), replicated.
    Distinct seeds per layer so XLA cannot CSE-merge the unpacks across layers."""
    ws = []
    for i in range(K):
        W1u, S1 = make_fp4_local(INTER, DIM, mesh, 10 + i * 7)   # [EP,inter,dim/2]
        W3u, S3 = make_fp4_local(INTER, DIM, mesh, 20 + i * 7)
        W2u, S2 = make_fp4_local(DIM, INTER, mesh, 30 + i * 7)   # [EP,dim,inter/2]
        ws.append((W1u, S1, W3u, S3, W2u, S2))
    return ws


def stack_forward(x0, weights, gsz, goff, mesh, barrier):
    """K-layer prefill MoE region. Each layer: (optionally barrier-tie the fp4 weights
    to the incoming activation x) -> unpack fp4->fp8 rhs (the big buffers) -> 2 gmm_v2
    (consumer) -> residual chain. barrier=True caps cross-layer unpack concurrency."""
    x = x0                                              # [N, dim] bf16, replicated
    N = x0.shape[0]
    for (W1u, S1, W3u, S3, W2u, S2) in weights:
        if barrier:
            # Bundle the fp4 weights with this layer's activation: the barrier op
            # consumes x (on the residual chain) and re-emits the weights, so the
            # unpack transitively depends on x and cannot hoist above layer i-1.
            W1u, W3u, W2u, x = jax.lax.optimization_barrier((W1u, W3u, W2u, x))
        W13, S13, W2t, S2t = rhs_prep(W1u, S1, W3u, S3, W2u, S2)   # fp8 rhs (live buffers)
        lhs = jnp.tile(x, (TOP_K, 1))                   # [N*top_k, dim] bf16 (gmm lhs)
        g2 = gmm_core(lhs, W13, S13, W2t, S2t, gsz, goff, mesh)    # [N*top_k, dim] fp32
        x = x + g2[:N].astype(x.dtype)                  # residual chain -> [N, dim]
    return x


def analyze(mem):
    """Pull (temp_GiB, arg_GiB) from a CompiledMemoryStats, tolerant of field names."""
    g = lambda a: getattr(mem, a, 0) / GiB
    return g('temp_size_in_bytes'), g('argument_size_in_bytes')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--N", type=int, default=256,
                    help="prefill seq_len (256 matches the MAX_LEN=256 smoke peak)")
    ap.add_argument("--layers", type=str, default="4,16,43",
                    help="K (#MoE layers) sweep; 43 = the real model")
    a = ap.parse_args()
    if a.distributed:
        jax.distributed.initialize()
    nd = jax.device_count()
    p0 = (not a.distributed) or jax.process_index() == 0
    devices = np.array(jax.devices()).reshape(nd)
    mesh = jax.sharding.Mesh(devices, ('attn_dp',))
    N = a.N
    M = N * TOP_K
    assert M % E == 0, f"N={N}: N*top_k={M} not divisible by E={E}"
    per_expert = M // E
    Ks = [int(s) for s in a.layers.split(",")]

    rows = []
    with jax.set_mesh(mesh):
        x0 = make_replicated((N, DIM), mesh, 1)                  # [N,dim] bf16
        gsz = jnp.full(E, per_expert, dtype=jnp.int32)
        goff = jnp.asarray([0], jnp.int32)
        gsz = jax.device_put(gsz, NamedSharding(mesh, P()))
        goff = jax.device_put(goff, NamedSharding(mesh, P()))

        for K in Ks:
            weights = build_layer_weights(mesh, K)
            res = {}
            for barrier in (False, True):
                fn = (lambda x, w, g, o, _b=barrier:
                      stack_forward(x, w, g, o, mesh, _b))
                compiled = jax.jit(fn).lower(x0, weights, gsz, goff).compile()
                temp, arg = analyze(compiled.memory_analysis())
                jax.block_until_ready(compiled(x0, weights, gsz, goff))
                _, mn = measure(lambda: compiled(x0, weights, gsz, goff), a.iters)
                res['bar' if barrier else 'none'] = (temp, arg, mn)
            rows.append((K, res))
            # free the K-layer weights before building the next (avoid HBM pile-up).
            del weights
            if p0:
                (t0, _, m0), (t1, a1, m1) = res['none'], res['bar']
                print(f"  K={K:>3}: noBar temp {t0:6.2f} GiB ({m0:7.1f} ms)  | "
                      f"bar temp {t1:6.2f} GiB ({m1:7.1f} ms)  | "
                      f"saved {t0 - t1:5.2f} GiB, lat {m1 - m0:+6.1f} ms "
                      f"({(m1/m0 - 1)*100:+.0f}%); args {a1:.2f} GiB")

    if p0:
        print(f"\n=== V4 PREFILL MoE rhs-prep CROSS-LAYER HBM PEAK (16-chip EP={E//nd}, "
              f"N={N}, M={M}) ===")
        print(f"dims dim={DIM} inter={INTER} | backend={jax.default_backend()} devices={nd}")
        print(f"per-layer fp8 rhs ~= W13 {16*DIM*2*INTER/GiB*1024:.0f}MiB + "
              f"W2t {16*INTER*DIM/GiB*1024:.0f}MiB + scales (EP={E//nd})")
        print(f"\n{'K':>4} {'noBar GiB':>10} {'noBar ms':>9} {'bar GiB':>9} {'bar ms':>8} "
              f"{'saved GiB':>10} {'lat delta':>10}")
        for K, res in rows:
            (t0, _, m0), (t1, _, m1) = res['none'], res['bar']
            print(f"{K:>4} {t0:>10.2f} {m0:>9.1f} {t1:>9.2f} {m1:>8.1f} "
                  f"{t0 - t1:>10.2f} {m1 - m0:>+9.1f}")
        print(f"\nREAD: if noBar temp grows ~linearly in K (-> ~18 GiB @K=43, matching the "
              f"real-model smoke peak) and bar temp stays ~flat, the rhs-prep unpack-"
              f"concurrency hypothesis is CONFIRMED and the optimization_barrier cap is a "
              f"LOSSLESS HBM lever. 'saved GiB' @K=43 ~= the headroom freed for roadmap #4 "
              f"(token-sharded-o, DO-NOT-RETRY #21) / a future residency play. 'lat delta' = "
              f"the overlap-loss cost (the rhs-prep VPU work no longer hides behind other "
              f"layers' gmm MXU). NOTE: the bench's gmm consumer is replicated + skips the "
              f"sort/gather/collectives (tiny at N={N}); the PEAK it measures is unpack-"
              f"dominated and faithful, the latency is a LOWER bound on prod's prefill wall.")


if __name__ == "__main__":
    main()

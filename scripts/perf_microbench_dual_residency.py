#!/usr/bin/env python3
"""TPU micro-benchmark: HBM FEASIBILITY of MoE expert DUAL-RESIDENCY (roadmap #2).

The prefill MoE path unpacks FP4->fp8 IN-TRACE every forward (`_fp4_rhs_and_scale`
x3, the rhs-prep, ~194-225 ms/fwd -- the single biggest prefill lever; P.9/P.12). It
CANNOT be made faster (P.12: XLA's native float4_e2m1fn.astype(fp8) is the VPU floor),
only ELIMINATED by keeping the experts fp8-RESIDENT (pre-unpacked at LOAD) for prefill,
ALONGSIDE the fp4-resident copy DECODE keeps reading (decode UNCHANGED => decode-neutral).

The ONLY make-or-break question this answers: does the dual copy FIT? The arithmetic
(per-chip, EP=16, all 43 MoE layers):
    fp4 codes+scales  ~8.57 GiB  (decode residency, kept)
    fp8 weights        ~16.5 GiB  (prefill residency, NEW = 2x the fp4 codes)
    non-expert model   ~1.6 GiB
    ------------------------------
    dual resident     ~26.7 GiB  / ~4.5 GiB free of the 31.25 budget  -- HBM-MARGINAL.

This ALLOCATES the full dual residency on the real mesh (NO full-model load), reads
`memory_stats()` for the ACTUAL resident/free, then runs the prefill gmm reading the
fp8-resident weights at a sweep of N to find the max prefill length whose MoE transient
still fits -- and times gmm-core ALONE (the post-dual cost) vs the rhs-prep it removes.
Tier-2: ~1 min vs a 25-45 min OOM-risk smoke.

The fp8 store mirrors what a load-time pre-build would hold: the fully-prepared gmm rhs
W13=[EP,dim,2*inter] + W2t=[EP,inter,dim] (fp8). The per-block SCALES stay e8m0 (part of
the fp4 residency) and the cheap e8m0->fp32+concat is left in-trace (P.9: rhs-prep cost
IS the weight unpack; scales are ~free) -- so the NEW resident bytes are JUST the fp8
weights (option b: +16.5, not +18.6 with fp32 scales resident).

CAVEAT: the transient measured here is MoE-ONLY. A real prefill forward ALSO holds the
attention working set + KV cache, so "max N that fits here" is an UPPER bound on the real
serving length. But the RESIDENT term (26.7 GiB, the dominant + exact part) is faithful:
if the resident copy alone leaves too little free, roadmap #2 is dead regardless.

Run (TPU FREE; SYNC first -- mh_run runs each host's own clone):
    scripts/full_slice_v4_sync.sh
    MH_TIMEOUT=900 scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_dual_residency.py --distributed
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
    DIM, INTER, E, TOP_K, N_MOE_LAYERS, BLK, measure)
from perf_microbench_moe_prefill import gmm_core, rhs_prep  # noqa: E402

GiB = 2 ** 30


def dev_zeros(shape, dtype, mesh):
    """A REPLICATED P() device array of `shape` (per-chip bytes = full shape). Values
    are irrelevant -- this measures HBM footprint + kernel time, not numerics."""
    return jax.device_put(jnp.zeros(shape, dtype), NamedSharding(mesh, P()))


def hbm(dev):
    st = dev.memory_stats()
    return (st.get('bytes_in_use', 0) / GiB,
            st.get('peak_bytes_in_use', 0) / GiB,
            st.get('bytes_limit', 0) / GiB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--seqs", type=str, default="256,512,1024,2048,4096")
    ap.add_argument("--nonexpert-gib", type=float, default=1.6,
                    help="dummy alloc standing in for the non-expert model residency")
    ap.add_argument("--layers", type=int, default=N_MOE_LAYERS,
                    help="how many MoE layers of dual residency to allocate (default 43)")
    a = ap.parse_args()
    if a.distributed:
        jax.distributed.initialize()
    nd = jax.device_count()
    p0 = (not a.distributed) or jax.process_index() == 0
    devices = np.array(jax.devices()).reshape(nd)
    mesh = jax.sharding.Mesh(devices, ('attn_dp',))
    axis = nd
    EP = E // axis
    f8 = jnp.float8_e4m3fn
    u8 = jnp.uint8
    seqs = [int(s) for s in a.seqs.split(",")]
    dev0 = jax.local_devices()[0]

    with jax.set_mesh(mesh):
        base_use, _, limit = hbm(dev0)

        # ---- (1) fp4 residency: the decode copy, KEPT. Per layer: w1/w3 [EP,inter,
        # dim/2] + w2 [EP,dim,inter/2] codes (u8 packed) + e8m0 scales [.., ../BLK]. ----
        fp4 = []
        for _ in range(a.layers):
            fp4.append((
                dev_zeros((EP, INTER, DIM // 2), u8, mesh),     # w1 codes
                dev_zeros((EP, INTER, DIM // BLK), u8, mesh),   # w1 scale (e8m0)
                dev_zeros((EP, INTER, DIM // 2), u8, mesh),     # w3 codes
                dev_zeros((EP, INTER, DIM // BLK), u8, mesh),   # w3 scale
                dev_zeros((EP, DIM, INTER // 2), u8, mesh),     # w2 codes
                dev_zeros((EP, DIM, INTER // BLK), u8, mesh),   # w2 scale
            ))
        jax.block_until_ready([x for t in fp4 for x in t])
        fp4_use, _, _ = hbm(dev0)

        # ---- (2) fp8 residency: the prefill copy, NEW. The fully-prepared gmm rhs
        # W13 [EP,dim,2*inter] + W2t [EP,inter,dim] (fp8). 2x the fp4 codes. ----
        fp8 = []
        for _ in range(a.layers):
            fp8.append((dev_zeros((EP, DIM, 2 * INTER), f8, mesh),   # W13
                        dev_zeros((EP, INTER, DIM), f8, mesh)))      # W2t
        jax.block_until_ready([x for t in fp8 for x in t])
        fp8_use, _, _ = hbm(dev0)

        # ---- (3) non-expert residency stand-in (dense/attn/embed/lm_head/norm). ----
        ne = None
        if a.nonexpert_gib > 0:
            ne = dev_zeros((int(a.nonexpert_gib * GiB),), u8, mesh)
            jax.block_until_ready(ne)
        resident_use, _, _ = hbm(dev0)
        free = limit - resident_use

        # ---- (4) prefill gmm reading the fp8-RESIDENT weights, swept over N. The
        # transient (lhs/g1/h/g2) on top of the 26.7 GiB resident -> max-N that fits. ----
        gmm_f = jax.jit(lambda l, w13, s13, w2t, s2t, g, o:
                        gmm_core(l, w13, s13, w2t, s2t, g, o, mesh))
        W13_0, W2t_0 = fp8[0]
        results = []
        for N in seqs:
            M = N * TOP_K
            if M % E != 0:
                results.append((N, 'skip(N*top_k%E)', 0.0, 0.0)); continue
            per_expert = M // E
            gsz = jnp.full(E, per_expert, dtype=jnp.int32)
            goff = jnp.asarray([0], jnp.int32)
            # transient scales (option b: NOT resident; cheap in-trace from e8m0).
            S13 = dev_zeros((EP, DIM // BLK, 1, 2 * INTER), jnp.float32, mesh)
            S2t = dev_zeros((EP, INTER // BLK, 1, DIM), jnp.float32, mesh)
            lhs = dev_zeros((M, DIM), jnp.bfloat16, mesh)
            try:
                jax.block_until_ready(gmm_f(lhs, W13_0, S13, W2t_0, S2t, gsz, goff))
                _, mn = measure(lambda: gmm_f(lhs, W13_0, S13, W2t_0, S2t, gsz, goff),
                                a.iters)
                _, peak, _ = hbm(dev0)
                results.append((N, 'FIT', mn, peak))
            except Exception as e:  # noqa: BLE001 -- OOM / RESOURCE_EXHAUSTED
                results.append((N, 'OOM:' + repr(e)[:80], 0.0, 0.0))
            del S13, S2t, lhs

        # current rhs-prep cost (the per-forward unpack dual-residency REMOVES).
        W1u, S1 = fp4[0][0], fp4[0][1]
        W3u, S3 = fp4[0][2], fp4[0][3]
        W2u, S2 = fp4[0][4], fp4[0][5]
        prep_f = jax.jit(rhs_prep)
        try:
            jax.block_until_ready(prep_f(W1u, S1, W3u, S3, W2u, S2))
            _, prep_min = measure(lambda: prep_f(W1u, S1, W3u, S3, W2u, S2), a.iters)
        except Exception as e:  # noqa: BLE001
            prep_min = float('nan')

    if p0:
        L = a.layers
        print(f"\n=== MoE DUAL-RESIDENCY HBM feasibility (roadmap #2) "
              f"(16-chip EP={EP}, layers={L}) ===")
        print(f"dims dim={DIM} inter={INTER} E={E} | backend={jax.default_backend()} "
              f"devices={nd} | HBM limit {limit:.2f} GiB/chip")
        print(f"\nRESIDENT footprint (per chip, cumulative bytes_in_use):")
        print(f"  baseline (pre-alloc)     {base_use:7.2f} GiB")
        print(f"  + fp4 codes (decode)     {fp4_use:7.2f} GiB  (+{fp4_use-base_use:.2f})")
        print(f"  + fp8 weights (prefill)  {fp8_use:7.2f} GiB  (+{fp8_use-fp4_use:.2f})")
        print(f"  + non-expert ({a.nonexpert_gib} GiB)  {resident_use:7.2f} GiB"
              f"  (+{resident_use-fp8_use:.2f})")
        print(f"  ==> dual resident = {resident_use:.2f} GiB ; FREE = {free:.2f} GiB")
        print(f"\nPREFILL gmm reading fp8-RESIDENT weights (transient on top of resident):")
        print(f"{'seq_N':>6} {'status':>14} {'gmm ms/layer':>13} {'peak GiB':>10} "
              f"{'transient':>10}")
        for (N, status, mn, peak) in results:
            tr = (peak - resident_use) if peak > 0 else 0.0
            print(f"{N:>6} {status[:14]:>14} {mn:>13.3f} {peak:>10.2f} {tr:>10.2f}")
        print(f"\nSAVING: rhs-prep (the per-forward FP4->fp8 unpack dual-residency "
              f"ELIMINATES) = {prep_min:.3f} ms/layer => x{L} = {prep_min*L:.1f} ms/forward.")
        print(f"  Projected prefill MoE = (gmm + dispatch + collective)*{L} WITHOUT the "
              f"+{prep_min*L:.0f} ms rhs-prep (see perf_microbench_moe_prefill for the rest).")
        print(f"\nVERDICT: dual fits if FREE ({free:.2f}) comfortably exceeds the real "
              f"prefill transient (MoE transient above + attention working set + KV). "
              f"OOM in the MoE-only sweep => roadmap #2 is DEAD (refuted cheaply).")


if __name__ == "__main__":
    main()

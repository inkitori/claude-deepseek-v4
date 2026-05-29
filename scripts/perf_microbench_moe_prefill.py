#!/usr/bin/env python3
"""TPU micro-benchmark: the V4 PREFILL MoE expert-FFN device cost (roadmap #2, fork B).

The decode campaign (P.1-P.8) drove decode to its ~lossless floor (146 ms/step, MoE
~106 ms) and proved it LAUNCH-bound at N=1. PREFILL is the untapped half of the goal
and is COMPUTE-bound (real MXU work through the sharded gmm_v2 FP4 path), NOT launch-
bound. This is the FIRST prefill measurement in the campaign -- a blank slate.

Mirrors the REAL prefill MoE path (deepseek_v4_moe.py:331-433, `_routed_local`): per
rank (attn_dp EP=16), all_gather tokens -> sort the (token,slot) pairs by expert ->
gather the sorted lhs [N*top_k, dim] -> two gmm_v2 grouped matmuls over THIS rank's 16
experts (group_offset) reading fp8 codes + per-block fp32 rhs_scale -> revert/mask/
combine/psum. Decomposed into the pieces a prefill roadmap would attack, at the real
V4-Flash dims, BALANCED routing, swept over prefill seq_len N (per-rank, replicated --
the per-chip cost is identical to one shard_map rank; collectives measured separately):

  RHS-PREP  : `_fp4_rhs_and_scale` x3 + concat + layout_constraint -- the in-trace
              FP4->fp8 unpack + swapaxes, run PER LAYER PER FORWARD (SEQ-INDEPENDENT).
              Candidate LOSSLESS lever: fp8 codes are lossless (e2m1 subset of e4m3),
              so pre-unpacking to fp8-RESIDENT at load (17.1 GiB/chip, FITS) removes
              this from EVERY forward -- DISTINCT from roadmap #4's LOSSY scaled-fp8.
  GMM-CORE  : the two gmm_v2 grouped matmuls + silu on M_owned=N*top_k/axis rows. The
              dominant prefill MoE COMPUTE. Compared to a same-FLOP dense batched fp8
              einsum (the MXU floor) -- is gmm at its floor, or is there headroom?
  DISPATCH  : argsort([N*top_k]) + gather lhs[N*top_k,dim] + revert gather (rank-local).
              The handoff's old "48% copy/transpose" prefill suspect lives HERE.
  COLLECTIVE: all_gather x[N/axis,dim]->[N,dim] + psum[N,dim] (the only true collectives).

Per-layer pieces -> x N_MOE_LAYERS(43) -> projected prefill MoE device-ms at each N.
Both gmm calls mirror the production call EXACTLY (rhs=[EP,k,n] fp8, group_sizes=[E]
balanced summing to lhs rows, group_offset=[0], bf16 lhs auto-quantized to fp8).

Run (TPU FREE; SYNC first -- mh_run runs each host's own clone):
    scripts/full_slice_v4_sync.sh
    MH_TIMEOUT=900 scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_moe_prefill.py --distributed
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.layout import Layout, with_layout_constraint
from jax.sharding import NamedSharding, PartitionSpec as P

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2
from tpu_inference.layers.jax.moe.deepseek_v4_moe import _fp4_rhs_and_scale
from tpu_inference.layers.common.quantization import u8_unpack_e2m1

# Reuse the decode bench's dims, the timing harness, and BLK.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_microbench_moe_decode import (  # noqa: E402
    DIM, INTER, E, TOP_K, N_MOE_LAYERS, BLK, measure)


def make_fp4_local(out_dim, in_dim, mesh, seed):
    """Synthetic [EP, out, in/2] u8 packed-FP4 + [EP, out, in/BLK] u8 e8m0 scale for
    ONE rank's EP=16 local experts, REPLICATED (P()). Value ranges mirror the decode
    bench's make_fp4_stacked: random bytes are valid e2m1; scale exponents in [125,131)
    keep dequant magnitude O(1). The per-chip gmm cost is identical to one sharded rank."""
    EP = E // mesh.shape['attn_dp']
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, size=(EP, out_dim, in_dim // 2), dtype=np.uint8)
    s = rng.integers(125, 131, size=(EP, out_dim, in_dim // BLK), dtype=np.uint8)
    sh = NamedSharding(mesh, P())
    return (jax.make_array_from_callback(w.shape, sh, lambda idx: w[idx]),
            jax.make_array_from_callback(s.shape, sh, lambda idx: s[idx]))


def make_replicated(shape, mesh, seed, dtype=jnp.bfloat16):
    sh = NamedSharding(mesh, P())
    rng = np.random.default_rng(seed)
    base = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    return jax.make_array_from_callback(shape, sh,
                                        lambda idx, _b=base: _b[idx]).astype(dtype)


# ---- The decomposed prefill MoE pieces (each freshly jitted in main) ----

def rhs_prep(W1u, S1, W3u, S3, W2u, S2):
    """In-trace FP4->fp8 unpack + swapaxes + concat + layout_constraint -- EXACTLY the
    per-layer, per-forward rhs build at deepseek_v4_moe.py:351-358. SEQ-INDEPENDENT."""
    r1, s1q = _fp4_rhs_and_scale(W1u, S1)        # [EP,dim,inter] fp8
    r3, s3q = _fp4_rhs_and_scale(W3u, S3)
    W13 = jnp.concatenate([r1, r3], axis=2)      # [EP,dim,2*inter] fp8
    S13 = jnp.concatenate([s1q, s3q], axis=3)    # [EP,dim/blk,1,2*inter] fp32
    W2t, S2t = _fp4_rhs_and_scale(W2u, S2)       # [EP,inter,dim] fp8
    W13 = with_layout_constraint(W13, Layout((0, 1, 2)))
    W2t = with_layout_constraint(W2t, Layout((0, 1, 2)))
    return W13, S13, W2t, S2t


def rhs_unpack_only(W1u, W3u, W2u):
    """Isolate just the FP4->fp8 CODE unpack (u8_unpack_e2m1 + fp8 cast), no swapaxes/
    concat -- the pure dtype work the load-time pre-build would also do."""
    f8 = jnp.float8_e4m3fn
    return (u8_unpack_e2m1(W1u).astype(f8), u8_unpack_e2m1(W3u).astype(f8),
            u8_unpack_e2m1(W2u).astype(f8))


def rhs_unpack_swap(W1u, W3u, W2u):
    """Unpack + swapaxes (the [E,out,in]->[E,in,out] LAYOUT transpose), no concat --
    isolates how much of rhs-prep is the transpose vs the concat+layout_constraint."""
    f8 = jnp.float8_e4m3fn
    return (u8_unpack_e2m1(W1u).astype(f8).swapaxes(1, 2),
            u8_unpack_e2m1(W3u).astype(f8).swapaxes(1, 2),
            u8_unpack_e2m1(W2u).astype(f8).swapaxes(1, 2))


def gmm_core(lhs, W13, S13, W2t, S2t, gsz, goff, mesh):
    """The two gmm_v2 grouped matmuls + silu, mirroring _routed_local g1/g2 EXACTLY.
    lhs=[N*top_k,dim] bf16 (the full sorted buffer); gmm reads only this rank's rows.
    Wrapped in shard_map: gmm_v2 is a Mosaic kernel that CANNOT be auto-partitioned;
    all inputs replicated (P()) -> each rank runs the full per-rank gmm, timing =
    per-chip cost (the 16 identical ranks run in parallel)."""
    def _body(lhs, W13, S13, W2t, S2t, gsz, goff):
        fp32, bf16 = jnp.float32, jnp.bfloat16
        g1 = gmm_v2(lhs, W13, gsz, rhs_scale=S13, group_offset=goff,
                    zero_initialize=False, preferred_element_type=fp32)   # [M, 2*inter]
        gate, up = jnp.split(g1, 2, axis=-1)
        h = (jax.nn.silu(gate) * up).astype(bf16)                          # [M, inter]
        g2 = gmm_v2(h, W2t, gsz, rhs_scale=S2t, group_offset=goff,
                    zero_initialize=False, preferred_element_type=fp32)   # [M, dim]
        return g2
    return jax.shard_map(_body, mesh=mesh, in_specs=(P(),) * 7, out_specs=P(),
                         check_vma=False)(lhs, W13, S13, W2t, S2t, gsz, goff)


def dense_fp8_floor(lhs_g, W13b, W2tb):
    """Same-FLOP dense BATCHED fp8 einsum = the MXU floor gmm's own-rows compute could
    hit with NO ragged/sort/per-block-scale overhead (fp8xfp8 to match gmm's MXU path).
    lhs_g=[EP,m,dim] fp8, W13b=[EP,dim,2*inter] fp8, W2tb=[EP,inter,dim] fp8."""
    fp32, bf16 = jnp.float32, jnp.bfloat16
    g1 = jnp.einsum('gmd,gdn->gmn', lhs_g, W13b, preferred_element_type=fp32)
    gate, up = jnp.split(g1, 2, axis=-1)
    h = (jax.nn.silu(gate) * up).astype(jnp.float8_e4m3fn)
    g2 = jnp.einsum('gmi,gid->gmd', h, W2tb, preferred_element_type=fp32)
    return g2


def dispatch_local(x_full, idx_flat, g2_like):
    """Rank-local dispatch: 2 argsorts + the two big [N*top_k,dim] gathers (the sort +
    sorted-lhs gather + revert gather of _routed_local). NO collective (that's separate)."""
    Nf = x_full.shape[0]
    argsort_idx = jnp.argsort(idx_flat)                                # [N*top_k]
    token_idx = jnp.arange(Nf, dtype=jnp.int32).repeat(TOP_K)
    token_idx_sorted = token_idx[argsort_idx]
    x_sorted = x_full[token_idx_sorted]                                # [N*top_k,dim] gather
    revert_idx = jnp.argsort(argsort_idx)
    token_hidden = g2_like[revert_idx]                                 # [N*top_k,dim] gather
    return x_sorted.astype(jnp.float32).sum() + token_hidden.sum()


def collective(x_sharded, mesh):
    """all_gather x[N/axis,dim]->[N,dim] (the dispatch collective) + psum[N,dim] (the
    combine collective). The two true ICI collectives of the prefill MoE shard_map."""
    def _local(x_l):
        xf = jax.lax.all_gather(x_l, 'attn_dp', axis=0, tiled=True)    # [N,dim]
        xf = jax.lax.optimization_barrier(xf)
        y = jax.lax.psum(xf.astype(jnp.float32), 'attn_dp')            # [N,dim] fp32
        NP = xf.shape[0] // mesh.shape['attn_dp']
        r = jax.lax.axis_index('attn_dp')
        return jax.lax.dynamic_slice_in_dim(y, r * NP, NP, axis=0)
    return jax.shard_map(_local, mesh=mesh, in_specs=(P('attn_dp', None),),
                         out_specs=P('attn_dp', None), check_vma=False)(x_sharded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--seqs", type=str, default="512,1024,2048,4096",
                    help="prefill seq_len sweep (each must be a multiple of 128 so "
                         "N*top_k is divisible by E for balanced group_sizes)")
    a = ap.parse_args()
    if a.distributed:
        jax.distributed.initialize()
    nd = jax.device_count()
    p0 = (not a.distributed) or jax.process_index() == 0
    devices = np.array(jax.devices()).reshape(nd)
    mesh = jax.sharding.Mesh(devices, ('attn_dp',))
    axis = nd
    EP = E // axis
    seqs = [int(s) for s in a.seqs.split(",")]

    rows = []
    with jax.set_mesh(mesh):
        # rhs-prep is seq-independent -> build the fp8 rhs ONCE (also feeds gmm_core).
        W1u, S1 = make_fp4_local(INTER, DIM, mesh, 10)   # [EP,inter,dim/2]
        W3u, S3 = make_fp4_local(INTER, DIM, mesh, 20)
        W2u, S2 = make_fp4_local(DIM, INTER, mesh, 30)   # [EP,dim,inter/2]
        prep_f = jax.jit(rhs_prep)
        W13, S13, W2t, S2t = jax.block_until_ready(prep_f(W1u, S1, W3u, S3, W2u, S2))
        prep_med, prep_min = measure(
            lambda: prep_f(W1u, S1, W3u, S3, W2u, S2), a.iters)
        # Decompose rhs-prep: pure unpack vs +swapaxes(transpose) vs +concat+layout.
        unp_f = jax.jit(rhs_unpack_only)
        jax.block_until_ready(unp_f(W1u, W3u, W2u))
        unp_med, unp_min = measure(lambda: unp_f(W1u, W3u, W2u), a.iters)
        swp_f = jax.jit(rhs_unpack_swap)
        jax.block_until_ready(swp_f(W1u, W3u, W2u))
        swp_med, swp_min = measure(lambda: swp_f(W1u, W3u, W2u), a.iters)

        gmm_f = jax.jit(lambda l, w13, s13, w2t, s2t, g, o:
                        gmm_core(l, w13, s13, w2t, s2t, g, o, mesh))
        floor_f = jax.jit(dense_fp8_floor)
        disp_f = jax.jit(dispatch_local)
        coll_f = jax.jit(lambda xs: collective(xs, mesh))
        f8 = jnp.float8_e4m3fn

        for N in seqs:
            M = N * TOP_K                                # total (token,slot) rows
            per_expert = M // E                          # balanced rows/expert
            assert per_expert * E == M, f"N={N}: N*top_k not divisible by E"
            M_owned = per_expert * EP                    # this rank's gmm rows
            gsz = jnp.full(E, per_expert, dtype=jnp.int32)
            goff = jnp.asarray([0], jnp.int32)
            lhs = make_replicated((M, DIM), mesh, 100 + N)            # [N*top_k,dim] bf16

            # gmm-core (the dominant compute).
            jax.block_until_ready(gmm_f(lhs, W13, S13, W2t, S2t, gsz, goff))
            g_med, g_min = measure(
                lambda: gmm_f(lhs, W13, S13, W2t, S2t, gsz, goff), a.iters)

            # dense fp8 floor: this rank's owned rows, batched per expert (same FLOP).
            lhs_owned = make_replicated((EP, per_expert, DIM), mesh, 200 + N, f8)
            W13b = make_replicated((EP, DIM, 2 * INTER), mesh, 300, f8)
            W2tb = make_replicated((EP, INTER, DIM), mesh, 400, f8)
            jax.block_until_ready(floor_f(lhs_owned, W13b, W2tb))
            fl_med, fl_min = measure(lambda: floor_f(lhs_owned, W13b, W2tb), a.iters)

            # dispatch (rank-local sort + gathers).
            x_full = make_replicated((N, DIM), mesh, 500 + N)
            idx_flat = jax.device_put(
                np.random.default_rng(N).integers(0, E, size=M, dtype=np.int32),
                NamedSharding(mesh, P()))
            g2_like = make_replicated((M, DIM), mesh, 600 + N)
            jax.block_until_ready(disp_f(x_full, idx_flat, g2_like))
            d_med, d_min = measure(lambda: disp_f(x_full, idx_flat, g2_like), a.iters)

            # collective (all_gather + psum), x sharded on the token axis.
            x_sh = make_replicated((N, DIM), mesh, 700 + N)
            x_sh = jax.device_put(x_sh, NamedSharding(mesh, P('attn_dp', None)))
            jax.block_until_ready(coll_f(x_sh))
            c_med, c_min = measure(lambda: coll_f(x_sh), a.iters)

            # FLOP for gmm-core's owned rows: g1 M*dim*2inter + g2 M*inter*dim, x2.
            flop = M_owned * 2 * (DIM * 2 * INTER + INTER * DIM)
            rows.append(dict(N=N, M_owned=M_owned, per_expert=per_expert,
                             gmm=g_min, floor=fl_min, disp=d_min, coll=c_min,
                             gmm_tf=flop / g_min / 1e9, floor_tf=flop / fl_min / 1e9))

    if p0:
        L = N_MOE_LAYERS
        print(f"\n=== V4 PREFILL MoE expert-FFN microbench (16-chip EP={EP}, E={E}, "
              f"top_k={TOP_K}) ===")
        print(f"dims dim={DIM} inter={INTER} | backend={jax.default_backend()} devices={nd}")
        print(f"\nRHS-PREP (fp4->fp8 unpack+swapaxes+concat, SEQ-INDEPENDENT): "
              f"min {prep_min:6.3f} ms/layer  => x{L} = {prep_min*L:6.1f} ms/forward")
        print(f"  decomp: unpack-only {unp_min:.3f} | +swapaxes {swp_min:.3f} | "
              f"+concat+layout {prep_min:.3f} ms/layer "
              f"(transpose +{swp_min-unp_min:.3f}, concat/layout +{prep_min-swp_min:.3f})")
        print(f"  (candidate LOSSLESS lever: fp8-codes-RESIDENT at load removes this; "
              f"17.1 GiB/chip FITS; lossless e2m1 subset of e4m3)")
        print(f"\n{'seq_N':>6} {'M_owned':>8} {'gmm':>8} {'floor':>8} {'gmm/fl':>7} "
              f"{'gmmTF':>7} {'disp':>8} {'coll':>8}  (min ms/layer, /rank)")
        for r in rows:
            print(f"{r['N']:>6} {r['M_owned']:>8} {r['gmm']:>8.3f} {r['floor']:>8.3f} "
                  f"{r['gmm']/r['floor']:>6.2f}x {r['gmm_tf']:>6.0f} {r['disp']:>8.3f} "
                  f"{r['coll']:>8.3f}")
        print(f"\nPROJECTED prefill MoE device-ms (per-layer x {L} + rhs-prep once):")
        for r in rows:
            moe = (r['gmm'] + r['disp'] + r['coll']) * L + prep_min * L
            comp = r['gmm'] * L
            print(f"  N={r['N']:>5}: gmm {r['gmm']*L:6.1f} + disp {r['disp']*L:6.1f} + "
                  f"coll {r['coll']*L:6.1f} + rhsprep {prep_min*L:5.1f} = ~{moe:6.1f} ms "
                  f"(gmm@floor would be {r['floor']*L:6.1f}; compute={comp:.0f})")
        print(f"\nREAD: gmm/floor ratio -> grouping/sort/per-block-scale overhead vs the "
              f"dense fp8 MXU floor. disp (the [N*top_k,dim] gathers) = the old '48% "
              f"copy' suspect. rhs-prep is the SEQ-INDEPENDENT lossless lever.")


if __name__ == "__main__":
    main()

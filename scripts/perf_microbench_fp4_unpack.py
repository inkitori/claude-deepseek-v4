#!/usr/bin/env python3
"""TPU micro-bench: FASTER FP4->FP8 unpack for the prefill MoE rhs-prep (roadmap #2).

The prefill MoE rhs-prep (`deepseek_v4_moe.py:_fp4_rhs_and_scale`, :351-358) is the
dominant SEQ-INDEPENDENT prefill cost: 5.24-5.35 ms/layer x43 = ~225 ms/forward, and
the prefill-bench decomposition pinned the cost on the UNPACK itself (not swapaxes/
concat). The production unpack is
    jax.lax.bitcast_convert_type(u8, jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)
i.e. it routes through the sub-byte `float4_e2m1fn` dtype, whose XLA .astype() lowering
is ~9x its HBM floor => VPU-compute-bound. e2m1 has only 16 values and e2m1 (subset of)
e4m3, so the convert is re-expressible by integer bit-math / a 16-entry LUT that NEVER
materializes float4_e2m1fn. This bench compares candidates at the real V4-Flash dims on
the real 16-chip mesh, EACH proven bit-identical to production on-device.

Candidates (uint8 packed-fp4 [...,N] -> fp8 e4m3 [...,2N], low-nibble-first):
  prod     : production `u8_unpack_e2m1(w).astype(fp8)` (the float4_e2m1fn path).
  lut16    : bitcast u8->uint4 (free reinterpret, same split) -> gather a 16-entry fp8 LUT.
  intarith : bitcast u8->uint4 -> branchless integer formula -> bitcast bytes to fp8.
             sign=(n&8)<<4; mag=n&7; magbyte = 0x30+4*mag if mag>=2 else 0x30 if mag==1
             else 0. (CPU-verified bit-exact vs production for all 256 input bytes.)

Both lut16/intarith use the uint4 bitcast for the nibble interleave -> identical element
ORDER to production's float4 bitcast (low nibble = index 0), differing only in HOW the
4-bit code becomes an fp8 byte (gather vs arith). Neither touches float4_e2m1fn.

Run (TPU FREE; SYNC first -- mh_run runs each host's own clone):
    scripts/full_slice_v4_sync.sh
    MH_TIMEOUT=900 scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_fp4_unpack.py --distributed
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from tpu_inference.layers.common.quantization import u8_unpack_e2m1

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_microbench_moe_decode import (  # noqa: E402
    DIM, INTER, E, N_MOE_LAYERS, measure)

F8 = jnp.float8_e4m3fn


def _u4_split(w):
    """bitcast uint8 [...,N] -> uint4 [...,2N], low nibble first -- identical bit-split
    and element order to u8_unpack_e2m1's float4 bitcast, but to an INTEGER dtype (no
    float4_e2m1fn). Pure reinterpret (no VPU compute)."""
    u4 = jax.lax.bitcast_convert_type(w, jnp.uint4)        # [...,N,2] uint4
    return u4.reshape(*w.shape[:-1], -1)                   # [...,2N] uint4


# ---- candidates: packed-fp4 uint8 [...,N] -> fp8 e4m3 [...,2N] ----

def unpack_prod(w):
    return u8_unpack_e2m1(w).astype(F8)


def build_lut():
    """16-entry fp8 LUT (nibble code -> fp8 value) built FROM production at runtime
    (no hardcoded constant): byte b in 0..15 has low nibble b, high nibble 0, so the
    even-index outputs are exactly code b's fp8 value."""
    probe = jnp.arange(16, dtype=jnp.uint8)
    return u8_unpack_e2m1(probe).astype(F8)[0::2]          # [16] fp8


def unpack_lut(w, lut):
    return lut[_u4_split(w).astype(jnp.int32)]             # gather


def unpack_intarith(w):
    n = _u4_split(w).astype(jnp.int32)                     # 0..15
    sign = (n & 0x8) << 4                                  # -> bit 7
    mag = n & 0x7
    magbyte = jnp.where(mag >= 2, 0x30 + (mag << 2),
                        jnp.where(mag == 1, 0x30, 0))
    byte = (sign | magbyte).astype(jnp.uint8)
    return jax.lax.bitcast_convert_type(byte, F8)


def unpack_intarith8(w):
    """Same bit-math but ENTIRELY in uint8 (no int32 widen -> no 4x intermediate)."""
    u8 = jnp.uint8
    n = _u4_split(w).astype(u8)                            # 0..15 in uint8
    sign = (n & u8(0x8)) << u8(4)                          # 0x80 or 0
    mag = n & u8(0x7)
    base = u8(0x30) + (mag << u8(2))                       # 0x30..0x4C (mag>=2 correct)
    magbyte = jnp.where(mag >= u8(2), base,
                        jnp.where(mag == u8(1), u8(0x30), u8(0)))
    return jax.lax.bitcast_convert_type(sign | magbyte, F8)


def unpack_closed8(w):
    """Closed-form NORMAL e4m3 byte ((EE+6)<<3 | M<<2) + one select for the e2m1
    subnormal region (mag<2), all uint8 -- minimizes the select count to 1+sign."""
    u8 = jnp.uint8
    n = _u4_split(w).astype(u8)
    sign = (n & u8(0x8)) << u8(4)
    mag = n & u8(0x7)
    closed = (((mag >> u8(1)) + u8(6)) << u8(3)) | ((mag & u8(1)) << u8(2))
    # subnormal: mag==0 -> 0x00, mag==1 -> 0x30 (closed gives 0x30/0x34 resp.)
    magbyte = jnp.where(mag >= u8(2), closed,
                        jnp.where(mag == u8(1), u8(0x30), u8(0)))
    return jax.lax.bitcast_convert_type(sign | magbyte, F8)


def make_w(out_dim, in_dim, mesh, seed):
    """Synthetic [EP, out, in/2] u8 packed-FP4 for one rank's EP local experts,
    REPLICATED (P()). Random bytes are all valid e2m1 (every byte = 2 codes)."""
    EP = E // mesh.shape['attn_dp']
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, size=(EP, out_dim, in_dim // 2), dtype=np.uint8)
    return jax.make_array_from_callback(w.shape, NamedSharding(mesh, P()),
                                        lambda idx: w[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--distributed", action="store_true")
    a = ap.parse_args()
    if a.distributed:
        jax.distributed.initialize()
    nd = jax.device_count()
    p0 = (not a.distributed) or jax.process_index() == 0
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(nd), ('attn_dp',))
    EP = E // nd

    with jax.set_mesh(mesh):
        # The three real per-layer expert leaves (W1/W3 = [EP,inter,dim/2], W2 = [EP,dim,inter/2]).
        W1u = make_w(INTER, DIM, mesh, 10)
        W3u = make_w(INTER, DIM, mesh, 20)
        W2u = make_w(DIM, INTER, mesh, 30)
        lut = jax.block_until_ready(build_lut())

        cands = {
            "prod": lambda w: unpack_prod(w),
            "intarith": lambda w: unpack_intarith(w),     # int32 domain
            "intarith8": lambda w: unpack_intarith8(w),   # uint8 domain
            "closed8": lambda w: unpack_closed8(w),       # uint8, 1 fewer select
        }
        _ = lut  # (lut16 gather REFUTED: 3406 ms/layer -- TPU small-table gather pathological)

        # All-three unpack (matches rhs_unpack_only: the apples-to-apples per-layer cost).
        def all3(fn):
            return lambda: (fn(W1u), fn(W3u), fn(W2u))

        results = {}
        ref = None
        for name, fn in cands.items():
            f = jax.jit(all3(fn))
            out = jax.block_until_ready(f())
            med, mn = measure(f, a.iters)
            # bit-identity vs production (raw fp8 byte compare on all three tensors).
            bc = lambda t: jax.lax.bitcast_convert_type(t, jnp.uint8)
            bytes_now = [np.asarray(bc(t)) for t in out]
            if name == "prod":
                ref = bytes_now
                mism = 0
            else:
                mism = int(sum(int((a_ != b_).sum())
                               for a_, b_ in zip(bytes_now, ref)))
            results[name] = (mn, med, mism)

    if p0:
        L = N_MOE_LAYERS
        base = results["prod"][0]
        print(f"\n=== V4 FP4->FP8 unpack candidates (16-chip EP={EP}, dim={DIM} "
              f"inter={INTER}) backend={jax.default_backend()} ===")
        print(f"per-layer = unpack W1+W3+W2 (= rhs_unpack_only); x{L} = per-forward\n")
        print(f"{'candidate':>10} {'ms/layer':>9} {'x43 ms':>8} {'vs prod':>8} "
              f"{'bit-mismatch':>13}")
        for name, (mn, med, mism) in results.items():
            tag = "BASELINE" if name == "prod" else f"{base/mn:.2f}x"
            ok = "IDENTICAL" if mism == 0 else f"DIFF({mism})"
            print(f"{name:>10} {mn:>9.3f} {mn*L:>8.1f} {tag:>8} {ok:>13}")
        print(f"\nREAD: a candidate is a usable lever ONLY if bit-mismatch=0 (lossless) "
              f"AND ms/layer < {base:.3f}. x43 is the per-forward rhs-prep unpack cut "
              f"(prefill MoE rhs-prep is ~225 ms/fwd; unpack is ~all of it).")


if __name__ == "__main__":
    main()

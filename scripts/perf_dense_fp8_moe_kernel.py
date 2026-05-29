"""Dense FP8-code MoE matvec Pallas/Mosaic kernel for TPU v6e (N=1 decode path).

WHY: at N=1 decode the V4 MoE FFN is HBM-bound. XLA materializes a bf16 dequant of
the FP4 experts (a full HBM round-trip of the [E, out, in] weight), costing ~2.47
ms/layer vs a 0.79 ms/layer bf16-matmul floor. This kernel reads the FP8 codes
(1 byte/elem) ONCE and applies the e8m0 block-scale IN-REGISTER (never materializing
bf16 to HBM), so it should approach the fp8-read floor.

v6e CANNOT unpack native fp4 in-kernel (`tpu.unpack_subelements` on f4E2M1FN needs
v7), so the codes are fed as FP8 e4m3, unpacked LOSSLESSLY in-trace (the e2m1 codebook
is a subset of e4m3). The kernel transports them bitcast-to-uint32 outside and
`pltpu.bitcast(..., float8_e4m3fn)` inside (the gmm_v2 fp8 transport trick).

Numerics: scale-AFTER-per-block (matches the gmm_v2 structure), fp32 accumulate. This
is NOT bit-identical to the bf16 scale-before path (~1e-2 rel err expected, fine); the
real GATE is a later slice smoke.

CPU-validate (no slice, interpret=True):
    PYTHONPATH=work/tpu-inference:work/vllm work/vllm_env/bin/python3 \
        scripts/perf_dense_fp8_moe_kernel.py
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.layers.common.quantization import (MXFP4_BLOCK_SIZE,
                                                      e8m0_to_fp32,
                                                      u8_unpack_e2m1)
from tpu_inference.layers.jax.moe.deepseek_v4_moe import _dequant_fp4_experts

BLOCK = MXFP4_BLOCK_SIZE  # 32


# ----------------------------------------------------------------------------- kernel
def _moe_matvec_kernel(
    # Inputs (one grid tile each)
    x_ref,            # f32 [in, 1]                  (full contracting col, replicated)
    w_ref,            # u32 [1, in//4, tile_out]     (fp8 codes packed 4/elem along IN)
    s_ref,            # f32 [1, num_blocks, tile_out] (decoded e8m0 block scale)
    # Output
    o_ref,            # f32 [1, 1, tile_out]  (3D: N=1 is the 2nd-minor, see pallas_call)
    *,
    in_size: int,
):
    """One (expert, out-tile) program. Computes a matrix-vector for the tile.

    out[o] = sum_b  s[b,o] * sum_{k in block b}  fp8code[k,o] * x[k]

    Layout mirrors gmm_v2: contracting axis IN is the 2nd-minor (sublane) dim so
    pltpu.bitcast (which unpacks along the 2nd-minor) expands the packed uint32
    weight x4 back to fp8 along IN. OUT is the minor (lane) dim.
    """
    num_blocks = in_size // BLOCK

    # Unpack the fp8 codes IN-KERNEL: u32 -> fp8 e4m3 along the 2nd-minor (IN) axis.
    # [in//4, tile_out] u32 -> [in, tile_out] fp8 (32b / 8b = x4 expansion).
    w_fp8 = pltpu.bitcast(w_ref[0], jnp.float8_e4m3fn)   # [in, tile_out] fp8
    x = x_ref[...]                                        # [in, 1] f32

    # Per-block partial dot, scaled after the block matmul (gmm_v2 structure),
    # fp32 accumulate. Reshape the IN axis to [num_blocks, BLOCK] and contract BLOCK.
    tile_out = w_fp8.shape[-1]
    w_blk = w_fp8.reshape(num_blocks, BLOCK, tile_out).astype(jnp.float32)
    x_blk = x.reshape(num_blocks, BLOCK, 1)              # [num_blocks, BLOCK, 1] f32
    # block partial dot over BLOCK: [num_blocks, tile_out]
    block_dot = jnp.sum(w_blk * x_blk, axis=1)
    # apply per-block e8m0 scale (scale-after-per-block) then sum over blocks.
    scaled = block_dot * s_ref[0]                        # [num_blocks, tile_out]
    o_ref[0] = jnp.sum(scaled, axis=0, keepdims=True).astype(o_ref.dtype)  # write [1,tile_out]


def dense_fp8_moe_matvec(
    x: jax.Array,            # bf16/f32 [N=1, in]
    w_fp8: jax.Array,        # float8_e4m3fn [E, out, in]  (lossless fp4->fp8 codes)
    scale_e8m0_u8: jax.Array,  # uint8 [E, out, in//BLOCK] (raw e8m0 exponent bytes)
    *,
    tile_out: int = 256,
    interpret: bool = False,
) -> jax.Array:
    """Dense FP8-code MoE gate/up (or down) matvec at N=1.

    Computes  out[e, o] = sum_b scale[e,o,b] * sum_{k in blk b} w_fp8[e,o,k] * x[0,k]
    i.e. the einsum 'nd,eod->neo' (n=1) with block-scale along the contracting axis d.
    Covers BOTH MoE projections (gate/up: in=dim, out=inter; down: in=inter, out=dim).

    Returns:
        out f32 [N=1, E, out].
    Grid: (E, out // tile_out). N=1 single row; a matrix-vector per (expert, out-tile).
    """
    N, in_size = x.shape
    assert N == 1, "decode N=1 path"
    E, out_size, in_w = w_fp8.shape
    assert in_w == in_size
    assert in_size % BLOCK == 0
    num_blocks = in_size // BLOCK
    assert scale_e8m0_u8.shape == (E, out_size, num_blocks)
    assert out_size % tile_out == 0, f"{out_size=} must be divisible by {tile_out=}"
    num_out_tiles = out_size // tile_out
    assert in_size % 4 == 0, "fp8 packs 4/uint32 along IN"

    # Orient weight to [E, in, out] (gmm 'group, k=IN, n=OUT'): IN must be the
    # 2nd-minor (sublane) axis so pltpu.bitcast unpacks along it. The block scale
    # follows: [E, num_blocks, out].
    w_fp8 = w_fp8.swapaxes(1, 2)                              # [E, in, out] fp8
    s_f32 = e8m0_to_fp32(scale_e8m0_u8).astype(jnp.float32)   # [E, out, num_blocks]
    s_f32 = s_f32.swapaxes(1, 2)                             # [E, num_blocks, out]

    # Transport fp8 codes as uint32 PACKED ALONG IN (the 2nd-minor axis): pack 4
    # consecutive IN elems into one uint32 so the in-kernel pltpu.bitcast expands
    # the sublane axis back x4. lax.bitcast_convert_type consumes the TRAILING axis,
    # so put the size-4 group last: [E,in,out] -> [E,in//4,out,4] -> u32 [E,in//4,out].
    # (Round-trip vs pltpu.bitcast's 2nd-minor expansion verified IN-order.)
    w_grp = jnp.transpose(w_fp8.reshape(E, in_size // 4, 4, out_size),
                          (0, 1, 3, 2))                # [E, in//4, out, 4]
    w_u32 = jax.lax.bitcast_convert_type(w_grp, jnp.uint32)  # [E, in//4, out]

    # x: replicate the full contracting column [in, 1] to every program.
    x_col = x.astype(jnp.float32).reshape(in_size, 1)

    grid = (E, num_out_tiles)

    out = pl.pallas_call(
        functools.partial(_moe_matvec_kernel, in_size=in_size),
        grid=grid,
        in_specs=[
            # x: replicated full column [in, 1] to every program.
            pl.BlockSpec((in_size, 1), lambda e, n: (0, 0)),
            # w_u32: this expert's out-tile, full (packed) contracting axis.
            pl.BlockSpec((1, in_size // 4, tile_out), lambda e, n: (e, 0, n)),
            # scale: this expert's out-tile, all blocks.
            pl.BlockSpec((1, num_blocks, tile_out), lambda e, n: (e, 0, n)),
        ],
        # Output is 3D [E, N=1, out]: Pallas TPU requires a block's last-two dims be
        # div by (8,128) OR equal the array dims. A 2D [E,out] block (1,tile_out) fails
        # (sublane=1 expert, not div 8, != E). Making N=1 the explicit 2nd-minor => block
        # 2nd-minor 1 == array 1 (complies); the expert axis E stays a leading grid dim.
        out_specs=pl.BlockSpec((1, 1, tile_out), lambda e, n: (e, 0, n)),
        out_shape=jax.ShapeDtypeStruct((E, 1, out_size), jnp.float32),
        interpret=interpret,
        # grid axis 0 = expert, axis 1 = out-tile; both parallel (no cross-tile dep).
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")),
    )(x_col, w_u32, s_f32)

    return out[:, 0, :][None]  # [E,1,out] -> [1, E, out]


# ----------------------------------------------------------------------------- validate
def _build_synthetic(E, out_size, in_size, *, seed=0):
    """Random packed-FP4 weight (uint8, 2 e2m1/byte along IN) + e8m0 scale + bf16 x."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    assert in_size % 2 == 0 and in_size % BLOCK == 0
    # packed uint8 weight: [E, out, in//2], each byte = 2 e2m1 codes (low nibble first).
    w_u8 = jax.random.randint(k1, (E, out_size, in_size // 2), 0, 256,
                              dtype=jnp.int32).astype(jnp.uint8)
    # e8m0 scale bytes in [125, 131] -> 2^(b-127) in {0.25 .. 16}: [E, out, in//BLOCK].
    s_u8 = jax.random.randint(k2, (E, out_size, in_size // BLOCK), 125, 132,
                              dtype=jnp.int32).astype(jnp.uint8)
    x = (jax.random.normal(k3, (1, in_size), dtype=jnp.float32) * 0.5).astype(
        jnp.bfloat16)
    return w_u8, s_u8, x


def _run_case(E, out_size, in_size, *, tile_out, seed=0):
    w_u8, s_u8, x = _build_synthetic(E, out_size, in_size, seed=seed)

    # Reference: bf16 dequant (scale-before, jnp.repeat-free) then the einsum the
    # production decode path runs ('nd,eid->nei' with i=out, d=in).
    w_bf16 = _dequant_fp4_experts(w_u8, s_u8)  # [E, out, in] bf16
    ref = jnp.einsum('nd,eid->nei', x, w_bf16,
                     preferred_element_type=jnp.float32)  # [1, E, out] f32

    # Kernel input: lossless fp4->fp8 codes + raw e8m0 scale bytes.
    w_fp8 = u8_unpack_e2m1(w_u8).astype(jnp.float8_e4m3fn)  # [E, out, in] fp8
    got = dense_fp8_moe_matvec(x, w_fp8, s_u8, tile_out=tile_out,
                               interpret=True)

    ref = jnp.asarray(ref, jnp.float32)
    got = jnp.asarray(got, jnp.float32)
    max_abs = float(jnp.max(jnp.abs(ref)))
    max_dabs = float(jnp.max(jnp.abs(got - ref)))
    rel = max_dabs / max(max_abs, 1e-30)
    return rel, max_dabs, max_abs, ref.shape, got.shape


def main():
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    cases = [
        # (E, out, in, tile_out)  -- gate/up-like and down-like, small + realistic ratios
        dict(E=2, out_size=64, in_size=64, tile_out=64),
        dict(E=2, out_size=64, in_size=128, tile_out=64),
        dict(E=2, out_size=128, in_size=64, tile_out=64),
        # realistic V4 shapes (scaled E): gate/up dim=4096->inter=2048; down inter->dim
        dict(E=4, out_size=2048, in_size=4096, tile_out=512),  # gate/up
        dict(E=4, out_size=4096, in_size=2048, tile_out=512),  # down
    ]
    worst = 0.0
    for c in cases:
        rel, dabs, mabs, rshape, gshape = _run_case(**c, seed=1)
        ok = rel <= 5e-2 and rshape == gshape
        worst = max(worst, rel)
        tag = "OK " if ok else "FAIL"
        print(f"[{tag}] E={c['E']:>2} out={c['out_size']:>4} in={c['in_size']:>4} "
              f"tile_out={c['tile_out']:>4} | shapes {rshape}=={gshape} "
              f"| max|d|={dabs:.4e} max|ref|={mabs:.4e} rel={rel:.4e}")
    print(f"\nWORST rel err across cases: {worst:.4e}  "
          f"(threshold 5e-2 -> {'PASS' if worst <= 5e-2 else 'FAIL'})")


if __name__ == "__main__":
    main()

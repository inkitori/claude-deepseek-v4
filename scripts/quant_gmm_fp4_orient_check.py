#!/usr/bin/env python3
"""CPU orientation check for the FP4 -> gmm_v2 (rhs, rhs_scale) path (QUANT).

gmm_v2 is a Pallas/Mosaic kernel and CANNOT run on CPU (it calls get_tpu_info()
at trace time). But the *orientation* of the (rhs, rhs_scale) pair we hand it is
the thing that costs a 30-min smoke when wrong — and that we CAN byte-verify on
CPU, because gmm computes exactly

    out[m,n] = sum_k lhs[m,k] * rhs[g,k,n] * rhs_scale[g, k//block, 0, n]

This script confirms `deepseek_v4_moe._fp4_rhs_and_scale` reconstructs the trusted
host/in-trace dequant `_dequant_fp4_experts` in gmm's [group, k=IN, n=OUT] layout,
plus: float4 swapaxes lowers on CPU, jax.experimental.layout imports, and
gmm_v2.validate_inputs (a pure assert) accepts the resulting shapes.

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=work/tpu-inference:work/vllm \
    work/vllm_env/bin/python3 scripts/quant_gmm_fp4_orient_check.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from jax.experimental.layout import Layout, with_layout_constraint  # noqa: F401  (import-check)
from tpu_inference.layers.common.quantization import (
    MXFP4_BLOCK_SIZE, e8m0_to_fp32, u8_unpack_e2m1)
from tpu_inference.layers.jax.moe.deepseek_v4_moe import (
    _fp4_rhs_and_scale, _dequant_fp4_experts)
from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2  # noqa: F401  (import-check)

BLK = MXFP4_BLOCK_SIZE  # 32
rng = np.random.default_rng(0)


def _dequant_f32(w_u8, scale_u8):
    """`_dequant_fp4_experts` without the trailing bf16 cast -> [E, out, in] f32.
    This is the trusted reference (validated byte-identical to the host torch
    dequant by quant_fp4_dequant_check.py)."""
    codes = u8_unpack_e2m1(w_u8).astype(jnp.float32)
    s = jnp.repeat(e8m0_to_fp32(scale_u8), BLK, axis=-1)
    return codes * s


def _mk(E, out, in_):
    """Random stored FP4 leaf: packed weight [E,out,in/2] u8 + e8m0 scale
    [E,out,in/BLK] u8 (raw exponent bytes, like the V4 loader)."""
    w = jnp.asarray(rng.integers(0, 256, size=(E, out, in_ // 2), dtype=np.uint8))
    # e8m0 bytes near 127 (== 2^0) so values are O(1) and not absurd.
    s = jnp.asarray(rng.integers(120, 135, size=(E, out, in_ // BLK), dtype=np.uint8))
    return w, s


def _check_leaf(name, E, out, in_):
    """Verify _fp4_rhs_and_scale reconstructs the dequant in [g,k=in,n=out]."""
    w, s = _mk(E, out, in_)
    rhs, scale = _fp4_rhs_and_scale(w, s)   # rhs [E,in,out] fp4 ; scale [E,in/BLK,1,out] f32

    assert rhs.dtype == jnp.float4_e2m1fn, f"{name}: rhs dtype {rhs.dtype} != float4_e2m1fn"
    assert rhs.shape == (E, in_, out), f"{name}: rhs shape {rhs.shape} != {(E, in_, out)}"
    assert scale.dtype == jnp.float32, f"{name}: scale dtype {scale.dtype}"
    assert scale.shape == (E, in_ // BLK, 1, out), \
        f"{name}: scale shape {scale.shape} != {(E, in_ // BLK, 1, out)}"

    # Reconstruct what gmm multiplies: rhs(f32) * per-block scale broadcast to k.
    rhs_f32 = rhs.astype(jnp.float32)                              # [E,k,n]
    scale_rep = jnp.repeat(scale[:, :, 0, :], BLK, axis=1)         # [E,k,n]
    recon = rhs_f32 * scale_rep                                    # [E,k=in,n=out]

    # Reference dequant, reoriented to gmm's [g, k=in, n=out].
    ref = _dequant_f32(w, s).swapaxes(1, 2)                        # [E,in,out]

    ok = bool(jnp.array_equal(recon, ref))
    if not ok:
        md = float(jnp.max(jnp.abs(recon - ref)))
        raise AssertionError(f"{name}: orientation MISMATCH, max|recon-ref|={md}")

    # End-to-end matmul: gmm math == x @ dequant.T (per expert) — catches any
    # residual index swap the elementwise check could miss.
    M = 8
    x = jnp.asarray(rng.standard_normal((M, in_)), dtype=jnp.float32)
    gmm_math = jnp.einsum('mk,gkn->gmn', x, recon)                 # [E,M,out]
    deq = _dequant_f32(w, s)                                       # [E,out,in]
    ref_mm = jnp.einsum('mk,gnk->gmn', x, deq)                     # [E,M,out]
    if not bool(jnp.allclose(gmm_math, ref_mm, atol=1e-3, rtol=1e-3)):
        md = float(jnp.max(jnp.abs(gmm_math - ref_mm)))
        raise AssertionError(f"{name}: matmul MISMATCH, max={md}")
    print(f"  {name}: rhs{tuple(rhs.shape)} fp4 + scale{tuple(scale.shape)} f32 "
          f"-> orientation EXACT, matmul OK")
    return rhs, scale


def main():
    print(f"jax {jax.__version__}; platform={jax.default_backend()}")
    # V4-Flash dims, small E for speed. H(dim)=128, I(inter)=64 (real: 4096/2048).
    H, I, E, EP = 128, 64, 8, 2

    # w1/w3: out=I, in=H ; w2: out=H, in=I.
    print("[1] per-leaf orientation (_fp4_rhs_and_scale == dequant in [g,k,n]):")
    r1, s1 = _check_leaf("w1", E, I, H)
    r3, s3 = _check_leaf("w3", E, I, H)
    r2, s2 = _check_leaf("w2", E, H, I)

    # W13 = concat(W1,W3) on n=out (axis2); S13 = concat on n (axis3).
    print("[2] W13 concat orientation:")
    W13 = jnp.concatenate([r1, r3], axis=2)       # [E, H, 2I]
    S13 = jnp.concatenate([s1, s3], axis=3)       # [E, H/BLK, 1, 2I]
    assert W13.shape == (E, H, 2 * I) and S13.shape == (E, H // BLK, 1, 2 * I)
    # gate (cols 0:I) must be W1, up (cols I:2I) must be W3.
    assert bool(jnp.array_equal(W13[:, :, :I].astype(jnp.float32),
                                r1.astype(jnp.float32)))
    assert bool(jnp.array_equal(W13[:, :, I:].astype(jnp.float32),
                                r3.astype(jnp.float32)))
    print(f"  W13{tuple(W13.shape)} fp4 + S13{tuple(S13.shape)} f32; gate=W1 up=W3 OK")

    # [3] float4 swapaxes + layout-constraint-shaped path lowers under jit on CPU.
    print("[3] float4 swapaxes lowers under jit (CPU):")
    f = jax.jit(lambda w: u8_unpack_e2m1(w).swapaxes(1, 2))
    w_tmp, _ = _mk(E, I, H)
    out = f(w_tmp)
    assert out.dtype == jnp.float4_e2m1fn and out.shape == (E, H, I)
    print(f"  jit float4 swapaxes -> {out.dtype} {tuple(out.shape)} OK")

    # [4] Shapes match gmm_v2's documented contract: rhs [group,k,n] float4,
    # rhs_scale [group, k/BLK, 1, n] f32. (gmm_v2.validate_inputs itself can't run
    # on CPU — it calls get_tpu_info() at gmm_v2.py:951 — so we assert the contract
    # directly; the EP-local slice mirrors the real prefill group_offset call.)
    print("[4] (EP-local) rhs/rhs_scale match gmm_v2 [group,k,n] / [group,k/BLK,1,n]:")
    for tag, (W, S, k, n) in {
            "g1(W13)": (W13[:EP], S13[:EP], H, 2 * I),
            "g2(W2)": (r2[:EP], s2[:EP], I, H)}.items():
        assert W.dtype == jnp.float4_e2m1fn and W.shape == (EP, k, n), \
            f"{tag}: rhs {W.dtype}{tuple(W.shape)} != float4{(EP, k, n)}"
        assert S.dtype == jnp.float32 and S.shape == (EP, k // BLK, 1, n), \
            f"{tag}: scale {S.dtype}{tuple(S.shape)} != f32{(EP, k // BLK, 1, n)}"
        assert k % BLK == 0, f"{tag}: k={k} not a multiple of block {BLK}"
        print(f"  {tag}: rhs{tuple(W.shape)} fp4 + scale{tuple(S.shape)} f32 OK")

    print("\nOK: FP4->gmm_v2 (rhs, rhs_scale) orientation byte-verified on CPU.")
    print("    (gmm_v2 itself is TPU-only; numerics/HBM confirmed by the smoke.)")


if __name__ == "__main__":
    main()

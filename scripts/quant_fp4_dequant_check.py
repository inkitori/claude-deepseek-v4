"""QUANT cheap-tier check: the in-trace JAX FP4 dequant == the trusted host
torch dequant, bit-for-bit, over the FULL codebook + e8m0 range.

`moe_forward` now keeps the 256 routed experts as packed FP4 (uint8) + e8m0
block scale in HBM (Strategy C) and dequants to bf16 in-trace via
`deepseek_v4_moe._dequant_fp4_experts`. This script proves that helper produces
output IDENTICAL to `deepseek_v4_loader.dequant_fp4_to_bf16` (the host path that
was trusted on v6e-32) on the SAME packed bytes + e8m0 scale — which also
empirically confirms the campaign premise that V4's FP4 codebook is
bit-identical to `jnp.float4_e2m1fn`.

Run (CPU, ~seconds):
  PYTHONPATH=work/tpu-inference:work/vllm work/vllm_env/bin/python3 \
      scripts/quant_fp4_dequant_check.py
Exit 0 = MATCH; non-zero = mismatch (do NOT smoke until this passes).
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import sys
import numpy as np
import jax
import jax.numpy as jnp
import torch

from tpu_inference.layers.common.quantization import MXFP4_BLOCK_SIZE
from tpu_inference.models.jax.deepseek_v4_loader import dequant_fp4_to_bf16
from tpu_inference.layers.jax.moe.deepseek_v4_moe import _dequant_fp4_experts

BLK = MXFP4_BLOCK_SIZE  # 32


def _check(shape_logical, seed):
    """shape_logical = (..., out, in). Returns (max_abs_diff, n_mismatch, n)."""
    *lead, out, in_dim = shape_logical
    assert in_dim % 2 == 0 and in_dim % BLK == 0
    rng = np.random.default_rng(seed)
    # Full-range packed bytes -> exercises all 16 codebook entries, both nibbles.
    w_u8 = rng.integers(0, 256, size=(*lead, out, in_dim // 2), dtype=np.uint8)
    # e8m0 bytes in [110,146): scales 2^-17..2^18, all finite (255 would be NaN).
    s_u8 = rng.integers(110, 146, size=(*lead, out, in_dim // BLK), dtype=np.uint8)

    # --- mine: in-trace JAX dequant (uint8 weight + float8_e8m0fnu scale, the
    #     exact dtypes the model's param tree carries) ---
    s_e8m0 = jax.lax.bitcast_convert_type(jnp.asarray(s_u8), jnp.float8_e8m0fnu)
    mine = np.asarray(
        _dequant_fp4_experts(jnp.asarray(w_u8), s_e8m0).astype(jnp.float32))

    # --- reference: host torch dequant, looped over any leading (expert) dims ---
    ref = np.empty((*lead, out, in_dim), dtype=np.float32)
    for idx in np.ndindex(*lead) if lead else [()]:
        w_t = torch.from_numpy(np.ascontiguousarray(w_u8[idx])).view(torch.int8)
        s_t = torch.from_numpy(np.ascontiguousarray(s_u8[idx])).view(torch.float8_e8m0fnu)
        ref[idx] = dequant_fp4_to_bf16(w_t, s_t, BLK).float().numpy()

    diff = np.abs(mine - ref)
    return float(diff.max()), int((diff != 0).sum()), mine.size


if __name__ == "__main__":
    # Real V4-Flash expert shapes (in = contracting axis), plus the 3D stacked
    # form moe_forward actually feeds, and a tiny full-coverage case.
    cases = [
        ("w1/w3 [inter,dim]", (2048, 4096)),
        ("w2    [dim,inter]", (4096, 2048)),
        ("tiny full-coverage", (16, 64)),
        ("stacked [E,inter,dim]", (4, 256, 512)),
    ]
    ok = True
    for name, shape in cases:
        max_diff, n_bad, n = _check(shape, seed=1234)
        status = "MATCH" if n_bad == 0 else f"MISMATCH ({n_bad}/{n})"
        print(f"  {status:>18}  max|Δ|={max_diff:.3e}  {name} {shape}")
        ok = ok and (n_bad == 0)
    print()
    if ok:
        print("OK: in-trace JAX fp4 dequant is byte-identical to host torch dequant "
              "(V4 FP4 codebook == jnp.float4_e2m1fn confirmed).")
        sys.exit(0)
    print("FAIL: in-trace dequant diverges from the trusted host dequant.")
    sys.exit(1)

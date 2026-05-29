#!/usr/bin/env python3
"""QUANT Q.2 — cheap CPU loader check (no slice).

Validates that the V4 loader keeps the routed FP4 experts COMPRESSED instead of
dequantizing them to bf16:
  * `iter_v4_safetensors_specs` emits, per routed expert weight, a `kind=="fp4"`
    weight spec AND a sibling `kind=="e8m0"` scale spec (fp8 dense stays fused —
    no e8m0 sibling).
  * `read_dequant_slice` returns a packed **uint8** weight `[out, in/2]` and a
    raw **float8_e8m0fnu** scale `[out, in/fp4_block]` (no host dequant).
  * `place_spec_as_jax_sharded(mesh=None)` preserves those dtypes (the e8m0
    `_torch_to_numpy_preserve` bitcast fix — NOT a numeric astype).
  * `map_hf_name_to_jax_path` routes `...wN.weight -> ...wN` and
    `...wN.scale -> ...wN_scale` (no `<scale>` drop).
  * The emitted (uint8 weight, e8m0 scale) dequant — via the shared MXFP4
    primitives the in-trace consumer uses — bit-identically to the
    byte-faithful bf16 groundtruth.

This is the loader's CHEAPEST validation tier; the slice smoke (Q.5) remains the
real gate. Build the fixtures first (once):
  V4_REAL_FLASH=<gcs snapshot dir> V4_SCRATCH_DIR=work/scratch \
    PYTHONPATH=work/tpu-inference:work/vllm work/vllm_env/bin/python3 \
    scripts/make_tiny_v4_checkpoint.py
Run:
  PYTHONPATH=work/tpu-inference:work/vllm work/vllm_env/bin/python3 \
    scripts/quant_loader_fp4_check.py
"""
import json
import os
import sys
from pathlib import Path

# Force CPU BEFORE importing jax — this is a no-slice check; never touch the TPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import numpy as np
import ml_dtypes
import torch
import jax
import jax.numpy as jnp
from safetensors import safe_open

ROOT = Path(os.environ.get("V4_REPO_ROOT", str(Path(__file__).resolve().parent.parent)))
QUANT = Path(os.environ.get("V4_TINY_QUANT", str(ROOT / "work" / "scratch" / "tiny_v4_quant")))
GT = Path(os.environ.get("V4_TINY_GT", str(ROOT / "work" / "scratch" / "tiny_v4_groundtruth")))

from tpu_inference.models.jax.deepseek_v4_loader import (  # noqa: E402
    iter_v4_safetensors_specs, read_dequant_slice, place_spec_as_jax_sharded,
    detect_quant_config,
)
from tpu_inference.models.jax.deepseek_v4 import map_hf_name_to_jax_path  # noqa: E402
from tpu_inference.layers.common.quantization import (  # noqa: E402
    u8_unpack_e2m1, e8m0_to_fp32)


def _to_np(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.float8_e8m0fnu:
        return t.view(torch.uint8).numpy().view(ml_dtypes.float8_e8m0fnu)
    if t.dtype == torch.int8:
        return t.view(torch.uint8).numpy()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


def _fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    if not QUANT.exists() or not GT.exists():
        _fail(f"fixtures missing ({QUANT}, {GT}); run scripts/make_tiny_v4_checkpoint.py")
    qc = detect_quant_config(json.loads((QUANT / "config.json").read_text()))
    fp4_block = qc["fp4_block"]
    print(f"fp4_block={fp4_block} fp8_block={qc['fp8_block']}")

    specs = list(iter_v4_safetensors_specs(str(QUANT)))
    by_name = {}
    for s in specs:
        by_name.setdefault(s.hf_name, []).append(s)
    dups = {n: len(v) for n, v in by_name.items() if len(v) > 1}
    if dups:
        _fail(f"duplicate specs (a tensor yielded twice): {dups}")

    fp4 = [s for s in specs if s.kind == "fp4"]
    e8 = [s for s in specs if s.kind == "e8m0"]
    fp8 = [s for s in specs if s.kind == "fp8"]
    print(f"specs: total={len(specs)} fp4={len(fp4)} e8m0={len(e8)} fp8={len(fp8)}")
    if not fp4:
        _fail("no fp4 weight specs emitted")
    if len(e8) != len(fp4):
        _fail(f"e8m0 scale specs ({len(e8)}) != fp4 weight specs ({len(fp4)})")

    e8_names = {s.hf_name for s in e8}
    e8_by_name = {s.hf_name: s for s in e8}
    # every fp4 weight has a matching e8m0 scale spec; the scale spec carries
    # the right shard path and no nested scale_key.
    for s in fp4:
        sn = s.hf_name[:-len(".weight")] + ".scale"
        if sn not in e8_names:
            _fail(f"fp4 weight {s.hf_name!r} has no sibling e8m0 scale spec")
    # fp8 dense must stay fused — NO e8m0 sibling spec.
    for s in fp8:
        sn = s.hf_name[:-len(".weight")] + ".scale"
        if sn in e8_names:
            _fail(f"fp8 weight {s.hf_name!r} wrongly emitted an e8m0 scale spec")
    # e8m0 specs must look like routed-expert scales and carry no nested scale.
    for s in e8:
        if ".experts." not in s.hf_name or not s.hf_name.endswith(".scale"):
            _fail(f"unexpected e8m0 spec name {s.hf_name!r}")
        if s.scale_key is not None or s.scale_shard_path is not None:
            _fail(f"e8m0 spec {s.hf_name!r} should not carry a nested scale")

    gt = safe_open(str(GT / "model-00001-of-00001.safetensors"), framework="pt")
    gt_names = set(gt.keys())

    n_checked = 0
    for ws in fp4:
        sn = ws.hf_name[:-len(".weight")] + ".scale"
        ss = e8_by_name[sn]
        wf = safe_open(ws.shard_path, framework="pt")
        out_dim = wf.get_slice(ws.hf_name).get_shape()[0]

        w = read_dequant_slice(ws, 0, out_dim)
        s = read_dequant_slice(ss, 0, out_dim)
        if w.dtype != torch.uint8:
            _fail(f"{ws.hf_name}: weight dtype {w.dtype} != uint8 (still dequanting?)")
        if s.dtype != torch.float8_e8m0fnu:
            _fail(f"{sn}: scale dtype {s.dtype} != float8_e8m0fnu")
        in_packed = w.shape[1]
        in_logical = 2 * in_packed
        if s.shape != (out_dim, in_logical // fp4_block):
            _fail(f"{sn}: scale shape {tuple(s.shape)} != {(out_dim, in_logical // fp4_block)}")

        # mapping: weight -> ...wN ; scale -> ...wN_scale
        wpath = map_hf_name_to_jax_path(ws.hf_name)
        spath = map_hf_name_to_jax_path(sn)
        if wpath is None or wpath.endswith("_scale") or "<scale>" in (wpath or ""):
            _fail(f"weight {ws.hf_name!r} mapped to bad path {wpath!r}")
        if spath != wpath + "_scale":
            _fail(f"scale {sn!r} mapped to {spath!r}, expected {wpath + '_scale'!r}")

        # placement (mesh=None) must PRESERVE dtypes (the E1 bitcast fix).
        warr = place_spec_as_jax_sharded(ws, jnp.uint8, (out_dim, in_packed), None)
        sarr = place_spec_as_jax_sharded(
            ss, jnp.float8_e8m0fnu, (out_dim, in_logical // fp4_block), None)
        if warr.dtype != jnp.uint8:
            _fail(f"{ws.hf_name}: placed weight dtype {warr.dtype} != uint8")
        if sarr.dtype != jnp.float8_e8m0fnu:
            _fail(f"{sn}: placed scale dtype {sarr.dtype} != float8_e8m0fnu "
                  f"(numeric astype corrupted the e8m0 bits?)")

        # dequant the emitted leaves via the shared MXFP4 primitives (the same
        # ones _dequant_fp4_experts uses) at the fixture block, compare to GT.
        w_u8 = jnp.asarray(_to_np(w))
        s_u8 = jax.lax.bitcast_convert_type(jnp.asarray(_to_np(s)), jnp.uint8)
        codes = u8_unpack_e2m1(w_u8).astype(jnp.float32)
        sc = jnp.repeat(e8m0_to_fp32(s_u8), fp4_block, axis=-1)
        deq = (codes * sc).astype(jnp.bfloat16)

        if ws.hf_name not in gt_names:
            _fail(f"groundtruth missing {ws.hf_name}")
        gt_np = _to_np(gt.get_tensor(ws.hf_name)).astype(np.float32)
        deq_np = np.asarray(deq).astype(np.float32)
        if deq_np.shape != gt_np.shape:
            _fail(f"{ws.hf_name}: dequant shape {deq_np.shape} != GT {gt_np.shape}")
        if not np.array_equal(deq_np, gt_np):
            maxd = float(np.abs(deq_np - gt_np).max())
            _fail(f"{ws.hf_name}: dequant != groundtruth, max|Δ|={maxd}")
        n_checked += 1

    # fp8 dense still dequants to bf16 (one spec, no sibling) — spot check one.
    if fp8:
        f8 = fp8[0]
        wf = safe_open(f8.shard_path, framework="pt")
        out_dim = wf.get_slice(f8.hf_name).get_shape()[0]
        b = read_dequant_slice(f8, 0, out_dim)
        if b.dtype != torch.bfloat16:
            _fail(f"fp8 {f8.hf_name}: dequant dtype {b.dtype} != bf16")

    print(f"OK: {n_checked} FP4 experts kept compressed (uint8 + e8m0), "
          f"routed to wN/wN_scale, dequant byte-identical to groundtruth; "
          f"fp8 dense still bf16.")


if __name__ == "__main__":
    main()

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DeepSeek-V4 weight loader: FP4 / FP8 / bf16 dequantization to bf16.

Storage formats in V4 safetensors (per `quant_meta.json`):

  * **bf16 / fp32**: stored verbatim. Loader copies into JAX bf16 / fp32.
  * **FP8 (dense)**: weight stored as `float8_e4m3fn`, scale stored as
    `float8_e8m0fnu` with shape `[out/block, in/block]`. `block` (typ. 32 in
    tiny fixtures, 128 in real V4-Flash/Pro) per `quantization_config.weight_block_size`.
    Dequantization:
        bf16[o, i] = w_e4m3fn[o, i].float() * 2 ** (s_e8m0fnu[o//b, i//b].uint8() - 127)

  * **FP4 (experts)**: weight stored packed as `int8` with shape
    `[out, in/2]` (two FP4 values per byte), scale stored as
    `float8_e8m0fnu` with shape `[out, in/fp4_block_size]`.
    `fp4_block_size` (typ. 8 in tiny, 32 in real). Each byte's low/high
    nibble looks up in:

        FP4_TABLE = [0, 0.5, 1, 1.5, 2, 3, 4, 6,
                     0, -0.5, -1, -1.5, -2, -3, -4, -6]

    so dequant is:

        unpacked[o, 2k]   = FP4_TABLE[w[o, k] & 0xF]
        unpacked[o, 2k+1] = FP4_TABLE[(w[o, k] >> 4) & 0xF]
        bf16[o, i] = unpacked[o, i] * 2 ** (s[o, i//fp4_block_size].uint8() - 127)

The dequant runs in PyTorch on CPU at load time (dtypes are well-supported
there), then the bf16 result is converted to a JAX array.

This is the canonical recipe from `/mnt/scratch/v4_pro/inference/convert.py`
(see `cast_e2m1fn_to_e4m3fn` for the FP4→FP8→bf16 path that real V4 uses;
our path is simpler since we go straight to bf16).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import NamedSharding, PartitionSpec as P

# Map JAX dtypes to PyTorch raw-view dtypes so we can convert a torch tensor
# (whose dtype numpy doesn't understand, e.g. bfloat16) to numpy via a
# byte-equivalent integer view, then re-view the resulting numpy buffer back
# to the JAX dtype. This avoids a costly fp32 detour for every weight.
_DTYPE_VIEW_MAP = {
    jnp.dtype(jnp.bfloat16): torch.uint16,
    jnp.dtype(jnp.float32): torch.uint32,
    jnp.dtype(jnp.float16): torch.uint16,
    jnp.dtype(jnp.float8_e4m3fn): torch.uint8,
}

# Canonical FP4 codebook. Indices 0..7 = positives; 8..15 = negatives. Order
# matches DeepSeek's `convert.py:FP4_TABLE`.
FP4_TABLE_LIST = [
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

_FP4_TABLE_T = torch.tensor(FP4_TABLE_LIST, dtype=torch.float32)


# ------------------------------------------------------------
# Per-tensor dequant
# ------------------------------------------------------------

def dequant_fp8_to_bf16(weight: torch.Tensor, scale: torch.Tensor,
                         block: int) -> torch.Tensor:
    """Dequantize an FP8 (e4m3fn) weight + e8m0fnu scale to bf16.

    Args:
        weight: 2-D `float8_e4m3fn` tensor of shape [out, in].
        scale: 2-D `float8_e8m0fnu` tensor of shape [out/block, in/block].
        block: block size (same in both dims).
    Returns:
        bf16 tensor of shape [out, in].
    """
    assert weight.dtype == torch.float8_e4m3fn, f"got {weight.dtype}"
    assert scale.dtype == torch.float8_e8m0fnu, f"got {scale.dtype}"
    out_dim, in_dim = weight.shape
    assert out_dim % block == 0, f"out_dim {out_dim} not divisible by block {block}"
    assert in_dim % block == 0, f"in_dim {in_dim} not divisible by block {block}"
    so, si = scale.shape
    assert so == out_dim // block and si == in_dim // block, \
        f"scale shape {scale.shape} expected ({out_dim // block}, {in_dim // block})"
    w_f32 = weight.float()
    # e8m0 cast → fp32 gives 2^(byte-127) directly.
    s_f32 = scale.float()
    # Broadcast scale [So, Si] to [out, in] by upsampling each block.
    s_full = s_f32.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return (w_f32 * s_full).bfloat16()


def dequant_fp4_to_bf16(weight_packed: torch.Tensor, scale: torch.Tensor,
                         fp4_block: int, logical_in_dim: Optional[int] = None) -> torch.Tensor:
    """Dequantize a packed FP4 weight + e8m0fnu scale to bf16.

    Args:
        weight_packed: int8 tensor of shape [out, in_packed]. Each byte holds
            two FP4 values (low nibble at index 2k, high nibble at index 2k+1).
        scale: e8m0fnu tensor of shape [out, in_logical / fp4_block].
        fp4_block: number of FP4 values per scale entry (along the IN axis).
            (Typically 8 in tiny fixtures, 32 in real V4.)
        logical_in_dim: optional override of the logical in-dimension. Default
            is `2 * weight_packed.shape[1]`.
    Returns:
        bf16 tensor of shape [out, logical_in_dim].
    """
    assert weight_packed.dtype == torch.int8, f"got {weight_packed.dtype}"
    assert scale.dtype == torch.float8_e8m0fnu, f"got {scale.dtype}"
    out_dim, in_packed = weight_packed.shape
    in_logical = logical_in_dim if logical_in_dim is not None else 2 * in_packed
    assert in_logical == 2 * in_packed, "FP4 packing must be 2:1 along IN axis"
    assert in_logical % fp4_block == 0, \
        f"in_logical {in_logical} not divisible by fp4_block {fp4_block}"
    so, si = scale.shape
    assert so == out_dim and si == in_logical // fp4_block, \
        f"scale shape {scale.shape} expected ({out_dim}, {in_logical // fp4_block})"

    # Unpack: low nibble first, then high nibble (per convert.py).
    w_u8 = weight_packed.view(torch.uint8)
    low = (w_u8 & 0x0F).long()
    high = ((w_u8 >> 4) & 0x0F).long()
    table = _FP4_TABLE_T.to(weight_packed.device)
    # stack along last axis so [..., 2k] = low, [..., 2k+1] = high.
    unpacked = torch.stack([table[low], table[high]], dim=-1).flatten(-2)  # [out, in_logical]
    # Apply per-block scale.
    s_f32 = scale.float()
    s_full = s_f32.repeat_interleave(fp4_block, dim=1)  # [out, in_logical]
    return (unpacked * s_full).bfloat16()


def to_jax_bf16(t: torch.Tensor) -> jnp.ndarray:
    """Convert a torch bf16 tensor to a JAX bf16 array via fp32 numpy bridge."""
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)
    if t.dtype == torch.float32:
        return jnp.asarray(t.numpy())
    if t.dtype == torch.int32 or t.dtype == torch.int64 or t.dtype == torch.int8:
        return jnp.asarray(t.numpy())
    # Catch-all — convert via fp32 numpy.
    return jnp.asarray(t.float().numpy())


# ------------------------------------------------------------
# Driver: read an HF V4 checkpoint and emit a dict {jax_name → jnp.ndarray}.
# ------------------------------------------------------------

def detect_quant_config(hf_config: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a dict with parsed quant params.

    Keys returned:
      fp8_block (int or None): weight_block_size for FP8 dense layers.
      fp4_block (int or None): fp4_block_size for FP4 expert layers.
      expert_dtype (str): "fp4" or "fp8" or "bf16".
    """
    qc = hf_config.get("quantization_config") or {}
    fp8_block = None
    if qc.get("quant_method") == "fp8":
        wbs = qc.get("weight_block_size", [128, 128])
        fp8_block = int(wbs[0])
        assert wbs[0] == wbs[1], f"non-square fp8 block size {wbs}"
    fp4_block = hf_config.get("fp4_block_size", None)
    expert_dtype = hf_config.get("expert_dtype", "bf16")
    if fp4_block is None and expert_dtype == "fp4":
        # The HF V4-Flash / V4-Pro public config.json doesn't carry
        # fp4_block_size — it's hardcoded to 32 in DeepSeek's reference
        # `inference/model.py:18`. Default to 32 so our loader matches the
        # upstream FP4 layout convention.
        fp4_block = 32
    return {
        "fp8_block": fp8_block,
        "fp4_block": fp4_block,
        "expert_dtype": expert_dtype,
    }


def dequant_weight(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    kind: str,
    fp8_block: Optional[int] = None,
    fp4_block: Optional[int] = None,
) -> torch.Tensor:
    """Dispatch on `kind` ∈ {"bf16", "fp8", "fp4", "raw"} to produce a tensor.

    For "bf16" / "raw": passthrough (raw means return as-is, e.g. integer
    lookup tables like tid2eid). Scale must be None.
    For "fp8": call `dequant_fp8_to_bf16(weight, scale, fp8_block)`.
    For "fp4": call `dequant_fp4_to_bf16(weight, scale, fp4_block)`.
    """
    if kind in ("bf16", "raw"):
        if scale is not None:
            raise ValueError(f"{kind} weight unexpectedly has a scale")
        if kind == "bf16" and weight.dtype not in (torch.bfloat16, torch.float32):
            return weight.bfloat16()
        return weight
    if kind == "fp8":
        if fp8_block is None:
            raise ValueError("fp8 dequant requires fp8_block")
        return dequant_fp8_to_bf16(weight, scale, fp8_block)
    if kind == "fp4":
        if fp4_block is None:
            raise ValueError("fp4 dequant requires fp4_block")
        return dequant_fp4_to_bf16(weight, scale, fp4_block)
    raise ValueError(f"Unknown kind {kind!r}")


def load_v4_safetensors_to_dict(
    checkpoint_dir: str,
    quant_meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, jnp.ndarray]:
    """Read every safetensors shard in `checkpoint_dir` and return a flat dict
    `{name → jnp.ndarray}` of dequantized weights.

    The HF parameter naming convention is preserved (no remapping). Pair
    `name` + `name.scale` are merged into one dequantized output at `name`.

    Args:
        checkpoint_dir: directory containing config.json, model*.safetensors,
            (optional) quant_meta.json.
        quant_meta: optional override for the quant_meta dict. If None and a
            `quant_meta.json` is present in the checkpoint dir, it is loaded.
            Otherwise we infer from `quantization_config` in `config.json`.

    Returns:
        dict mapping HF param name → jnp.ndarray (bf16 for weights, fp32 for
        norms/scales/biases per their stored dtype).
    """
    from safetensors import safe_open
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    with open(cfg_path) as f:
        hf_config = json.load(f)
    qc = detect_quant_config(hf_config)
    qmeta_path = os.path.join(checkpoint_dir, "quant_meta.json")
    if quant_meta is None and os.path.exists(qmeta_path):
        with open(qmeta_path) as f:
            quant_meta = json.load(f)
    quant_meta = quant_meta or {}

    # Discover shard files.
    shard_paths = []
    idx_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        shard_set = set(idx["weight_map"].values())
        for s in shard_set:
            shard_paths.append(os.path.join(checkpoint_dir, s))
    else:
        # Single-file fallback.
        for fn in os.listdir(checkpoint_dir):
            if fn.endswith(".safetensors"):
                shard_paths.append(os.path.join(checkpoint_dir, fn))
    shard_paths.sort()

    # Read every name + dtype + raw tensor; group weight/scale pairs.
    raw: Dict[str, Tuple[torch.Tensor, str]] = {}   # name → (tensor, kind)
    scales: Dict[str, torch.Tensor] = {}            # name without ".scale" → scale tensor
    for path in shard_paths:
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name.endswith(".scale"):
                    scales[name[:-len(".scale")]] = t
                    continue
                # Decide kind. Prefer quant_meta entry if present.
                meta_entry = quant_meta.get(name)
                if meta_entry is not None:
                    kind = meta_entry["kind"]
                else:
                    # Heuristic fallback when quant_meta.json is absent.
                    if t.dtype == torch.float8_e4m3fn:
                        kind = "fp8"
                    elif t.dtype == torch.int8 and "experts" in name:
                        kind = "fp4"
                    else:
                        kind = "bf16"
                raw[name] = (t, kind)

    # Now dequant each name.
    out: Dict[str, jnp.ndarray] = {}
    for name, (t, kind) in raw.items():
        # Scale's key is the weight's key without ".weight". Many V4 weights
        # end in ".weight" (e.g. "layers.0.attn.wkv.weight"); their scale is
        # at "layers.0.attn.wkv.scale".
        scale_key = name[:-len(".weight")] if name.endswith(".weight") else name
        sc = scales.get(scale_key)
        if kind in ("bf16", "raw"):
            deq = dequant_weight(t, None, kind)
        else:
            if sc is None:
                raise ValueError(f"Missing scale for {kind!r} weight {name!r}")
            deq = dequant_weight(t, sc, kind, fp8_block=qc["fp8_block"],
                                  fp4_block=qc["fp4_block"])
        out[name] = to_jax_bf16(deq)
    return out


# ------------------------------------------------------------
# Apply loaded weights into our abstract param tree (Tier 4b / Tier 7).
# ------------------------------------------------------------

def apply_weights_to_param_tree(
    params,                # TransformerParams (with abstract or real arrays)
    weights: Dict[str, jnp.ndarray],
    cfg,                   # DeepseekV4Config
):
    """Substitute every leaf in `params` whose JAX-path is reachable from a
    name in `weights`. Returns a new TransformerParams with real arrays.

    Uses `map_hf_name_to_jax_path` to map HF names → JAX paths, then walks
    the tree and replaces matching leaves.

    Note: this is a "best-effort" loader. The abstract param tree expects
    bf16 / fp32 arrays at every leaf. For weights we substitute bf16
    arrays (post-dequant). Some norms/biases are fp32 in the abstract tree
    and we preserve fp32. Shapes must match exactly (or transpose-equivalent).
    """
    from tpu_inference.models.jax.deepseek_v4 import map_hf_name_to_jax_path
    # We mutate a dict-of-paths first, then write back via dataclasses.replace.
    # To keep things simple we walk the tree manually using attribute paths.
    import re

    # Build path→array dict from weights (resolving "<scale>" suffixes — we
    # already merged scale into the weight at dequant time, so any remaining
    # "<scale>" path is a no-op).
    by_jax_path: Dict[str, jnp.ndarray] = {}
    for hf_name, arr in weights.items():
        path = map_hf_name_to_jax_path(hf_name)
        if path is None or path.endswith("<scale>"):
            continue
        by_jax_path[path] = arr

    # Helpers to apply a path like "layers[3].attn.wq_a" to params.
    def _navigate(obj, path: str):
        """Returns (parent_obj, attr_name, list_idx_or_None) for the LAST
        segment of `path`. The caller assigns parent_obj.{attr} = new_value
        (or parent_obj.{attr}[idx] = new_value)."""
        parts = re.split(r"\.|(\[\d+\])", path)
        parts = [p for p in parts if p]
        cur = obj
        for i, part in enumerate(parts[:-1]):
            if part.startswith("["):
                idx = int(part[1:-1])
                cur = cur[idx]
            else:
                cur = getattr(cur, part)
        last = parts[-1]
        if last.startswith("["):
            # Last segment is an index — parent is `cur`, leaf is cur[idx].
            return cur, None, int(last[1:-1])
        return cur, last, None

    # Apply.
    for path, arr in by_jax_path.items():
        try:
            parent, attr, idx = _navigate(params, path)
        except (AttributeError, IndexError, KeyError):
            # Path not present in this config (e.g. mtp params absent).
            continue
        if attr is not None:
            cur_leaf = getattr(parent, attr)
            target_dtype = jnp.dtype(cur_leaf.dtype) if hasattr(cur_leaf, "dtype") else None
            if target_dtype is not None and arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)
            # Sanity-check shape.
            if hasattr(cur_leaf, "shape") and tuple(cur_leaf.shape) != tuple(arr.shape):
                # fp4 packed weight may have HF shape [O, I/2] vs JAX-bf16 [O, I];
                # but we already dequantized, so this should not happen.
                raise ValueError(f"Shape mismatch for {path}: expected {tuple(cur_leaf.shape)} got {tuple(arr.shape)}")
            setattr(parent, attr, arr)
        else:
            # idx into a list field (e.g. layers[L] or experts[E])
            cur_leaf = parent[idx]
            target_dtype = jnp.dtype(cur_leaf.dtype) if hasattr(cur_leaf, "dtype") else None
            if target_dtype is not None and arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)
            parent[idx] = arr
    return params


# ------------------------------------------------------------
# Streaming + sharded loader (production path for v6e-32 / TP=32).
#
# `load_v4_safetensors_to_dict` above buffers every dequantized weight in CPU
# RAM and then materializes the full tree on TPU device — that approach
# transient-doubles HBM and OOMs at the v6e-32 chip budget. The streaming
# path below yields one dequantized torch tensor at a time so the caller can
# place it directly onto a sharded global jax.Array and free the CPU buffer
# before reading the next safetensors entry.
# ------------------------------------------------------------


def _torch_to_numpy_preserve(t: torch.Tensor) -> np.ndarray:
    """Convert a CPU torch tensor to a numpy array of the same dtype + shape.

    For torch dtypes that numpy doesn't natively support (bfloat16,
    float8_e4m3fn, float8_e8m0fnu) we go through an integer raw-view bridge
    and immediately re-view the resulting buffer to the matching `ml_dtypes`
    dtype. The end result is a numpy array semantically equal to the torch
    tensor — shape preserved, dtype preserved (or its ml_dtypes equivalent).

    The previous implementation returned an integer-typed numpy buffer and
    relied on the caller doing a `.view(target_dtype)` later; that was
    correct only when the caller's target dtype had the same itemsize as
    the source. For bf16 -> fp32 (e.g. a norm stored bf16 on disk but with
    an fp32 abstract leaf), the byte-view halved the array's logical
    length. Doing the conversion semantically here makes the caller's
    `.astype(target)` always shape-preserving.
    """
    if not t.is_contiguous():
        t = t.contiguous()
    if t.dtype == torch.bfloat16:
        import ml_dtypes
        return t.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    if t.dtype == torch.float8_e4m3fn:
        import ml_dtypes
        return t.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
    if t.dtype == torch.float8_e8m0fnu:
        # ml_dtypes doesn't carry e8m0fnu. We don't yield raw e8m0 tensors
        # to the caller (they're consumed via `dequant_weight` -> bf16) so
        # this branch is only here for paranoia / debug. Return uint8.
        return t.view(torch.uint8).numpy()
    return t.numpy()


def pick_partition_spec(shape: Tuple[int, ...], mesh) -> P:
    """Heuristically pick a `PartitionSpec` for a weight of `shape` on `mesh`.

    Goal: shard the weight along its largest dim that's divisible by the
    largest available mesh axis, so a 543 GB bf16 V4-Flash model spreads
    evenly across the 32-chip slice (~17 GB / chip). When no dim divides
    cleanly, fall back to fully replicated — that's safe for small leaves
    (norms, biases, hc mix tables).

    The axis preference order favors `attn_dp` first because that's the axis
    that holds all 32 chips when `enable_dp_attention=True` is set, which is
    the deploy shape for V4-Flash. Other axes are considered only if they
    happen to carry size > 1.
    """
    if mesh is None or not shape:
        return P()
    sizes: Dict[str, int] = {}
    try:
        # jax.sharding.Mesh.shape is an OrderedDict-like mapping of name -> size.
        for name, sz in mesh.shape.items():
            sizes[str(name)] = int(sz)
    except (AttributeError, TypeError):
        return P()

    preference = ["attn_dp", "model", "expert", "attn_dp_expert", "data"]
    chosen_axis = None
    chosen_size = 1
    for name in preference:
        if sizes.get(name, 1) > 1:
            chosen_axis = name
            chosen_size = sizes[name]
            break
    if chosen_axis is None:
        return P()

    # Pick the largest dim divisible by chosen_size. Tie-break: earlier dim.
    best_dim = -1
    best_dim_size = -1
    for i, d in enumerate(shape):
        if d % chosen_size == 0 and d >= chosen_size and d > best_dim_size:
            best_dim = i
            best_dim_size = d
    if best_dim == -1:
        return P()  # replicate
    spec = [None] * len(shape)
    spec[best_dim] = chosen_axis
    return P(*spec)


def place_torch_as_jax_sharded(
    t: torch.Tensor,
    target_dtype,
    target_shape: Tuple[int, ...],
    mesh,
) -> jnp.ndarray:
    """Convert a CPU torch tensor to a sharded global `jax.Array` on `mesh`.

    The placement uses `jax.make_array_from_callback`, so each host populates
    only its local shards (the callback receives a per-device index into the
    global array). With multi-host JAX (the v6e-32 deploy shape), this means
    we never replicate the full weight onto every chip.

    Args:
        t: CPU torch tensor in any dtype the loader produces (bf16, fp32,
            int32, int64, etc.).
        target_dtype: JAX dtype the abstract param tree expects at this leaf.
            We cast on the host side via numpy view + `astype` rather than on
            device, so the on-device buffer is the right size from the start.
        target_shape: shape the abstract param tree expects. Must equal
            `tuple(t.shape)`.
        mesh: JAX mesh (e.g. from `tpu_runner._init_mesh`). If `None`, the
            tensor is placed on a single device (test path / fallback).
    """
    if tuple(t.shape) != tuple(target_shape):
        raise ValueError(
            f"Shape mismatch placing tensor: torch={tuple(t.shape)} "
            f"target={tuple(target_shape)}")
    np_arr = _torch_to_numpy_preserve(t)
    target = jnp.dtype(target_dtype)
    if np_arr.dtype != target:
        # Semantic dtype conversion (e.g. bf16 norm on disk -> fp32 leaf).
        # `astype` preserves shape, unlike a raw-byte `.view(target)`.
        np_arr = np_arr.astype(target)

    if mesh is None:
        return jnp.asarray(np_arr)

    spec = pick_partition_spec(tuple(target_shape), mesh)
    sharding = NamedSharding(mesh, spec)

    def _callback(idx):
        # idx is a tuple of slices — extract this device's shard from the
        # full host-resident numpy buffer. JAX guarantees the slice covers
        # only the local devices' data.
        return np.asarray(np_arr[idx])

    return jax.make_array_from_callback(tuple(target_shape), sharding, _callback)


def iter_v4_safetensors_dequant_torch(
    checkpoint_dir: str,
    quant_meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Stream-yield `(hf_name, dequantized_torch_tensor)` one at a time.

    Reads safetensors shards in a fixed order, fetches each weight (and its
    paired `.scale`, possibly from a different shard) lazily, dequantizes on
    CPU, then yields. The caller MUST consume each tensor before requesting
    the next: returned torch tensors are not retained inside this iterator,
    so memory is released as soon as the caller drops its reference.

    `.scale` entries are not yielded directly — they are merged into their
    paired `.weight` via `dequant_weight`.
    """
    from safetensors import safe_open
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    with open(cfg_path) as f:
        hf_config = json.load(f)
    qc = detect_quant_config(hf_config)
    qmeta_path = os.path.join(checkpoint_dir, "quant_meta.json")
    if quant_meta is None and os.path.exists(qmeta_path):
        with open(qmeta_path) as f:
            quant_meta = json.load(f)
    quant_meta = quant_meta or {}

    # Build name -> shard path mapping from the official index. Falls back to
    # scanning shard keys when an index is absent (single-file checkpoints).
    name_to_shard: Dict[str, str] = {}
    idx_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        for nm, sh in idx["weight_map"].items():
            name_to_shard[nm] = os.path.join(checkpoint_dir, sh)
    else:
        for fn in sorted(os.listdir(checkpoint_dir)):
            if fn.endswith(".safetensors"):
                p = os.path.join(checkpoint_dir, fn)
                with safe_open(p, framework="pt") as fh:
                    for nm in fh.keys():
                        name_to_shard[nm] = p

    # Group names by shard so we can open each shard once and amortize the
    # mmap cost across all its tensors.
    shard_to_names: Dict[str, List[str]] = {}
    for nm, sp in name_to_shard.items():
        shard_to_names.setdefault(sp, []).append(nm)

    def _determine_kind(name: str, dtype) -> str:
        meta_entry = quant_meta.get(name)
        if meta_entry is not None:
            return meta_entry["kind"]
        if dtype == torch.float8_e4m3fn:
            return "fp8"
        if dtype == torch.int8 and "experts" in name:
            return "fp4"
        return "bf16"

    def _fetch_scale(scale_key: str) -> Optional[torch.Tensor]:
        """Cross-shard scale lookup. The vast majority of V4 .scale entries
        live in the same shard as their .weight, but we don't assume so."""
        sh = name_to_shard.get(scale_key)
        if sh is None:
            return None
        with safe_open(sh, framework="pt") as fh:
            return fh.get_tensor(scale_key)

    for sp in sorted(shard_to_names.keys()):
        with safe_open(sp, framework="pt") as f:
            local_keys = set(f.keys())
            for hf_name in sorted(shard_to_names[sp]):
                if hf_name.endswith(".scale"):
                    continue
                t = f.get_tensor(hf_name)
                kind = _determine_kind(hf_name, t.dtype)
                if kind in ("bf16", "raw"):
                    deq = dequant_weight(t, None, kind)
                else:
                    if hf_name.endswith(".weight"):
                        scale_key = hf_name[:-len(".weight")] + ".scale"
                    else:
                        scale_key = hf_name + ".scale"
                    if scale_key in local_keys:
                        scale_t = f.get_tensor(scale_key)
                    else:
                        scale_t = _fetch_scale(scale_key)
                    if scale_t is None:
                        raise ValueError(
                            f"Missing scale for {kind!r} weight {hf_name!r}")
                    deq = dequant_weight(t, scale_t, kind,
                                         fp8_block=qc["fp8_block"],
                                         fp4_block=qc["fp4_block"])
                    del scale_t
                del t
                yield hf_name, deq
                # Caller is expected to have consumed `deq` and let it fall
                # out of scope before the next iteration. We don't hold a
                # reference here past the yield.
                del deq

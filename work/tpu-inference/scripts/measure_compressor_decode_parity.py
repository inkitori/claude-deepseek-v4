"""Measure observed atol on TestCompressorDecodeStep + TestCompressorDecodeStepExtended.

The existing tests assert <= 5e-2 (a placeholder budget from v3). This script
runs the same configurations across multiple seeds and prints the worst-case
observed atol on:
  - kv_compressed (when compression event fires)
  - kv_state (whole tensor)
  - score_state (over finite torch positions only)
"""
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# Import via the test module's helpers.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tests", "models", "jax"))

from _deepseek_v4_reference import (  # noqa: E402
    Compressor as TorchCompressor,
    ModelArgs as TorchArgs,
)
from _deepseek_v4_reference.model import (  # noqa: E402
    precompute_freqs_cis as torch_freqs,
)
from tpu_inference.layers.jax.attention.deepseek_v4_attention import (  # noqa: E402
    compressor_decode_step, CompressorParams,
)


def t2j(t):
    arr = t.detach()
    if arr.dtype == torch.bfloat16:
        return jnp.asarray(arr.float().numpy()).astype(jnp.bfloat16)
    return jnp.asarray(arr.numpy())


def make_tiny_args(**overrides):
    a = TorchArgs()
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _torch_compressor_to_jax_params(c) -> CompressorParams:
    return CompressorParams(
        ape=t2j(c.ape).astype(jnp.float32),
        wkv=t2j(c.wkv.weight).astype(jnp.float32),
        wgate=t2j(c.wgate.weight).astype(jnp.float32),
        norm_w=t2j(c.norm.weight).astype(jnp.float32),
        head_dim=c.head_dim,
        rope_head_dim=c.rope_head_dim,
        compress_ratio=c.compress_ratio,
        norm_eps=1e-6,
        rotate=c.rotate,
    )


def measure_one(ratio, P, start_pos, max_seq_len, seed):
    torch.manual_seed(seed)
    args = make_tiny_args(max_seq_len=max_seq_len)
    c = TorchCompressor(args, compress_ratio=ratio, head_dim=args.head_dim, rotate=False)
    for n, p in c.named_parameters():
        t = torch.empty_like(p, dtype=torch.float32).normal_(0, 0.02)
        p.data.copy_(t.to(p.dtype))
    kv_cache = torch.zeros(args.max_batch_size, args.max_seq_len // ratio, args.head_dim)
    c.kv_cache = kv_cache
    c.freqs_cis = torch_freqs(args.rope_head_dim, args.max_seq_len, 0,
                              args.compress_rope_theta, args.rope_factor,
                              args.beta_fast, args.beta_slow)
    bsz = 1
    prefill_x = torch.randn(bsz, P, args.dim, dtype=torch.bfloat16)
    with torch.inference_mode():
        _ = c(prefill_x, start_pos=0)

    kv_state_j = t2j(c.kv_state[:bsz].clone()).astype(jnp.float32)
    score_state_j = t2j(c.score_state[:bsz].clone()).astype(jnp.float32)
    params_j = _torch_compressor_to_jax_params(c)
    fc_j = t2j(c.freqs_cis).astype(jnp.complex64)

    x_step = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)
    with torch.inference_mode():
        kv_t = c(x_step, start_pos=start_pos)
    new_kvst, new_scst, kv_compressed_j, did = compressor_decode_step(
        t2j(x_step), start_pos, params_j, fc_j, kv_state_j, score_state_j,
    )

    kv_compressed_diff = None
    if kv_t is not None:
        assert did
        kv_compressed_diff = float(np.abs(np.asarray(kv_compressed_j) -
                                          kv_t.detach().cpu().float().numpy()).max())
    diff_kvst = float(np.abs(np.asarray(new_kvst) -
                              c.kv_state[:bsz].detach().float().numpy()).max())
    sc_j = np.asarray(new_scst)
    sc_t = c.score_state[:bsz].detach().float().numpy()
    finite_mask = np.isfinite(sc_t)
    if finite_mask.any():
        diff_sc = float(np.abs(sc_j[finite_mask] - sc_t[finite_mask]).max())
    else:
        diff_sc = 0.0
    return {
        "ratio": ratio, "P": P, "start_pos": start_pos,
        "max_seq_len": max_seq_len, "seed": seed,
        "did_compress": bool(did),
        "kv_compressed_diff": kv_compressed_diff,
        "kv_state_diff": diff_kvst,
        "score_state_diff": diff_sc,
    }


def main():
    points_basic = [
        (4, 4, 4, 512),
        (4, 8, 8, 512),
        (4, 7, 7, 512),
        (4, 12, 12, 512),
        (128, 64, 64, 512),
        (128, 128, 128, 512),
    ]
    points_extended = [
        (4, 64, 64, 256),
        (4, 63, 63, 256),
        (128, 255, 255, 512),
    ]
    seeds = [0, 1, 2, 3, 5, 7, 11, 13]
    results = []
    for ratio, P, sp, msl in points_basic + points_extended:
        for seed in seeds:
            r = measure_one(ratio, P, sp, msl, seed)
            results.append(r)

    worst_kv_compressed = max(
        (r["kv_compressed_diff"] for r in results if r["kv_compressed_diff"] is not None),
        default=None)
    worst_kv_state = max(r["kv_state_diff"] for r in results)
    worst_score_state = max(r["score_state_diff"] for r in results)
    worst_kv_compressed_row = max(
        (r for r in results if r["kv_compressed_diff"] is not None),
        key=lambda r: r["kv_compressed_diff"], default=None)
    worst_kv_state_row = max(results, key=lambda r: r["kv_state_diff"])
    worst_score_state_row = max(results, key=lambda r: r["score_state_diff"])

    summary = {
        "n_points": len(results),
        "n_seeds": len(seeds),
        "worst_kv_compressed": worst_kv_compressed,
        "worst_kv_compressed_at": (
            None if worst_kv_compressed_row is None
            else {k: worst_kv_compressed_row[k] for k in ("ratio", "P", "start_pos", "seed")}),
        "worst_kv_state": worst_kv_state,
        "worst_kv_state_at": {k: worst_kv_state_row[k] for k in ("ratio", "P", "start_pos", "seed")},
        "worst_score_state": worst_score_state,
        "worst_score_state_at": {k: worst_score_state_row[k] for k in ("ratio", "P", "start_pos", "seed")},
        "current_budget": 5e-2,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

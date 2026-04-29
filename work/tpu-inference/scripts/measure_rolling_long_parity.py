"""Measure observed atol on TestDecodeRollingParityLong.

The existing test asserts <= 5e-2 (placeholder). It calls attention_decode_step
inside a rolling-K loop, which is the same path the v8 iter 4 tightening took
to atol=1e-4 on TestDecodeAttentionParity / TestDecodeRollingParity /
TestDecodeAttentionParityExtended. This script re-runs the parametrized
configs across multiple seeds and prints worst-case observed atol.
"""
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tests", "models", "jax"))

from _deepseek_v4_reference import (  # noqa: E402
    Attention as TorchAttention,
    ModelArgs as TorchArgs,
)
from _deepseek_v4_reference.model import (  # noqa: E402
    precompute_freqs_cis as torch_freqs,
)
from tpu_inference.layers.jax.attention.deepseek_v4_attention import (  # noqa: E402
    attention_decode_step,
)


def t2j(t):
    arr = t.detach()
    if arr.dtype == torch.bfloat16:
        return jnp.asarray(arr.float().numpy()).astype(jnp.bfloat16)
    return jnp.asarray(arr.numpy())


def maxabs(a, b):
    a_np = np.asarray(a).astype(np.float32)
    b_np = b.detach().float().numpy()
    return float(np.abs(a_np - b_np).max())


def make_tiny_args(**overrides):
    a = TorchArgs()
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


# Replicate the test-helper machinery for building an attention layer +
# converting to JAX state/params.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tests", "models", "jax"))


def main():
    # Lazy-import the helpers from the test module rather than re-implementing.
    # The pytest collector won't run; we just want the module-level helpers.
    import importlib.util
    test_path = os.path.join(os.path.dirname(__file__), os.pardir,
                             "tests", "models", "jax", "test_deepseek_v4.py")
    spec = importlib.util.spec_from_file_location("test_deepseek_v4_module", test_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _build_attn_for_decode = mod._build_attn_for_decode
    _torch_attention_state_to_jax = mod._torch_attention_state_to_jax
    _torch_attention_to_jax_params = mod._torch_attention_to_jax_params

    points = [
        # (layer_id, P, K, max_seq_len)
        (0, 1, 31, 64),
        (0, 16, 32, 64),
        (2, 1, 31, 256),
    ]
    seeds = [0, 1, 2, 3, 5, 7]
    rows = []
    for layer_id, P, K, msl in points:
        for seed in seeds:
            attn, args = _build_attn_for_decode(
                layer_id, seed=42, max_seq_len=msl)
            bsz = 1
            torch.manual_seed(P + K + layer_id + 1000 + seed)
            full_x = torch.randn(bsz, P + K, args.dim, dtype=torch.bfloat16)
            with torch.inference_mode():
                _ = attn(full_x[:, :P], start_pos=0)
            jax_state = _torch_attention_state_to_jax(attn, args, bsz)
            params_j = _torch_attention_to_jax_params(attn, args)
            fc = t2j(attn.freqs_cis).astype(jnp.complex64)

            worst = 0.0
            worst_k = -1
            for k in range(K):
                sp = P + k
                x_step = full_x[:, sp:sp + 1]
                with torch.inference_mode():
                    o_t = attn(x_step, start_pos=sp)
                jax_state, y_j = attention_decode_step(
                    t2j(x_step), sp, params_j, fc, jax_state,
                )
                d = maxabs(y_j, o_t)
                if d > worst:
                    worst = d
                    worst_k = k
            rows.append({
                "layer_id": layer_id, "P": P, "K": K, "max_seq_len": msl,
                "seed": seed, "worst_step_atol": worst, "worst_step_k": worst_k,
            })

    worst_overall = max(r["worst_step_atol"] for r in rows)
    worst_row = max(rows, key=lambda r: r["worst_step_atol"])
    summary = {
        "n_configs": len(points),
        "n_seeds": len(seeds),
        "n_rows": len(rows),
        "worst_overall": worst_overall,
        "worst_at": {k: worst_row[k] for k in
                     ("layer_id", "P", "K", "seed", "worst_step_k")},
        "current_budget": 5e-2,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

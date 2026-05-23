"""Compare TPU compiled HLO with vs without _v4_anchor_output_buffers.

Confirms whether the optimization_barrier in `_v4_anchor_output_buffers`
actually changes anything in the compiled program on TPU XLA. Empirically
the answer is NO: TPU XLA strips lax.optimization_barrier in compile, and
the with-anchor compiled HLO is byte-identical to the no-anchor version
(see HLO op-count diff at end of run).

Run on a single-host TPU (e.g. worker 0 of a v6e-N slice):

    TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 \\
    TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \\
    PYTHONPATH=work/vllm:work/tpu-inference JAX_PLATFORMS=tpu \\
    work/vllm_env/bin/python3.12 scripts/s1_tpu_anchor_compare.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import functools
import jax
import jax.numpy as jnp

from tpu_inference.models.jax.deepseek_v4 import (
    DeepseekV4Config,
    make_abstract_transformer_params,
    make_freqs_cis,
    deepseek_v4_run_with_decode_state,
    v4_layer_packed_sizes_from_cfg,
)
import tpu_inference.models.jax.deepseek_v4 as v4mod


cfg = DeepseekV4Config(
    vocab_size=1024, hidden_size=256, intermediate_size=64,
    moe_intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
    num_key_value_heads=1, head_dim=32, qk_rope_head_dim=16, q_lora_rank=64,
    o_lora_rank=64, o_groups=2, n_routed_experts=8, n_shared_experts=1,
    num_experts_per_tok=2, num_hash_layers=1, num_nextn_predict_layers=1,
    sliding_window=8, swiglu_limit=0.0, score_func="sqrtsoftplus",
    routed_scaling_factor=2.5, rms_norm_eps=1e-6, rope_theta=10000.0,
    compress_rope_theta=160000.0, rope_factor=16.0, rope_beta_fast=32,
    rope_beta_slow=1, rope_original_seq_len=0, max_position_embeddings=256,
    compress_ratios=(0, 0, 4, 128, 4, 0, 0), index_n_heads=4,
    index_head_dim=16, index_topk=8, hc_mult=4, hc_sinkhorn_iters=20,
    hc_eps=1e-6)

params_struct = make_abstract_transformer_params(cfg)
leaves, treedef = jax.tree_util.tree_flatten(
    params_struct, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct))
keys = jax.random.split(jax.random.PRNGKey(0), len(leaves))
randomized = [
    (jax.random.normal(k, x.shape, dtype=jnp.float32) * 0.02).astype(x.dtype)
    for k, x in zip(keys, leaves)
]
params = jax.tree_util.tree_unflatten(treedef, randomized)
max_seq = 64
swa, comp = make_freqs_cis(cfg, max_seq)
sizes = v4_layer_packed_sizes_from_cfg(cfg, max_seq, batch_size=1)
ids = jnp.zeros((1, 1), dtype=jnp.int32)


def count_ops(compiled_hlo):
    return {
        "optimization-barrier": compiled_hlo.count("optimization-barrier"),
        "dynamic-update-slice": compiled_hlo.count("dynamic-update-slice"),
        "copy": compiled_hlo.count(" copy("),
        "scatter": compiled_hlo.count("scatter"),
        "concatenate": compiled_hlo.count("concatenate"),
    }


@functools.partial(jax.jit, donate_argnums=0)
def jit_decode_anchored(kv_caches, ids, start_pos):
    return deepseek_v4_run_with_decode_state(
        kv_caches, ids, params, swa, comp, cfg,
        state_max_seq_len=max_seq, is_decode_step=True,
        start_pos=start_pos)


# Monkey-patch _v4_anchor_output_buffers to identity for the unanchored case.
orig_anchor = v4mod._v4_anchor_output_buffers
v4mod._v4_anchor_output_buffers = lambda bufs: list(bufs)


@functools.partial(jax.jit, donate_argnums=0)
def jit_decode_unanchored(kv_caches, ids, start_pos):
    return deepseek_v4_run_with_decode_state(
        kv_caches, ids, params, swa, comp, cfg,
        state_max_seq_len=max_seq, is_decode_step=True,
        start_pos=start_pos)


v4mod._v4_anchor_output_buffers = orig_anchor

kv_caches = [jnp.zeros((s,), dtype=jnp.float32) for s in sizes]

print("=== Case A (with _v4_anchor_output_buffers) ===")
ca_hlo = jit_decode_anchored.lower(kv_caches, ids, jnp.int32(8)).compile().as_text()
ca_counts = count_ops(ca_hlo)
for k, v in ca_counts.items():
    print(f"  [A] {k}: {v}")

print("=== Case B (anchor = identity) ===")
cb_hlo = jit_decode_unanchored.lower(kv_caches, ids, jnp.int32(8)).compile().as_text()
cb_counts = count_ops(cb_hlo)
for k, v in cb_counts.items():
    print(f"  [B] {k}: {v}")

print()
print("=== DIFF (A - B) ===")
n_diff = 0
for k in ca_counts:
    diff = ca_counts[k] - cb_counts[k]
    marker = "  " if diff == 0 else "!="
    if diff != 0:
        n_diff += 1
    print(f"  {marker} {k}: A={ca_counts[k]}, B={cb_counts[k]}, diff={diff}")

if n_diff == 0:
    print("\nOK: anchor produces ZERO compile-time difference on TPU "
          "— `_v4_anchor_output_buffers` is a JIT no-op at this scale.")
else:
    print(f"\nWARN: {n_diff} HLO op-counts differ — anchor is doing "
          "something compile-visible after all.")

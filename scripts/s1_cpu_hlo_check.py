"""Verify lax.optimization_barrier appears in the compiled HLO for V4 decode."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import functools
import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.models.jax.deepseek_v4 import (
    DeepseekV4Config,
    make_abstract_transformer_params,
    make_freqs_cis,
    deepseek_v4_run_with_decode_state,
    v4_layer_packed_sizes_from_cfg,
)


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


@functools.partial(jax.jit, donate_argnums=0)
def jit_decode(kv_caches, ids, start_pos):
    return deepseek_v4_run_with_decode_state(
        kv_caches, ids, params, swa, comp, cfg,
        state_max_seq_len=max_seq, is_decode_step=True,
        start_pos=start_pos)


kv_caches = [jnp.zeros((s,), dtype=jnp.float32) for s in sizes]
lowered = jit_decode.lower(kv_caches, ids, jnp.int32(8))
print("=== LOWERED HLO (pre-compile) tail ===")
print(lowered.as_text()[-3000:])
print("=== END LOWERED HLO ===")
compiled = lowered.compile()
hlo = compiled.as_text()

# Count various op kinds.
lowered_text = lowered.as_text()
kinds = {
    "[lowered] stablehlo.optimization_barrier": lowered_text.count("optimization_barrier"),
    "[compiled] optimization-barrier": hlo.count("optimization-barrier"),
    "[compiled] copy": hlo.count(" copy("),
    "[compiled] dynamic-update-slice": hlo.count("dynamic-update-slice"),
    "[compiled] scatter": hlo.count("scatter"),
    "[compiled] concatenate": hlo.count("concatenate"),
    "[compiled] CustomCall HostCallback": hlo.count('CustomCall("HostCallback'),
}
print("HLO op counts:")
for k, v in kinds.items():
    print(f"  {k}: {v}")

# Show the tail of HLO where the output is constructed
print("\n--- HLO tail (last 4 KB) ---")
print(hlo[-4000:])

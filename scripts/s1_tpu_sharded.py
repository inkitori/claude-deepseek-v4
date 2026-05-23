"""V4-Flash-truncated decode parity on a SHARDED TPU mesh.

Exercises the sharding-axis interaction with at[].set writes that the
unsharded single-device repro doesn't catch. Uses a single-host TPU
with `attn_dp = num_local_chips` (matches the real-V4 production
sharding orientation but at smaller scale).

Run on a single-host TPU (e.g. worker 0 of a v6e-N slice):

    TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 \\
    TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \\
    PYTHONPATH=work/vllm:work/tpu-inference JAX_PLATFORMS=tpu \\
    work/vllm_env/bin/python3.12 scripts/s1_tpu_sharded.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "tpu")
import functools
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tpu_inference.models.jax.deepseek_v4 import (
    DeepseekV4Config,
    make_abstract_transformer_params,
    make_freqs_cis,
    transformer_body_forward,
    head_forward,
    deepseek_v4_run_with_decode_state,
    v4_layer_packed_sizes_from_cfg,
)
from tpu_inference.models.jax.deepseek_v4_loader import pick_partition_spec


def make_cfg(n_layers=4, n_experts=8):
    return DeepseekV4Config(
        vocab_size=129280, hidden_size=4096, intermediate_size=2048,
        moe_intermediate_size=2048, num_hidden_layers=n_layers,
        num_attention_heads=64, num_key_value_heads=1, head_dim=512,
        qk_rope_head_dim=64, q_lora_rank=1024, o_lora_rank=1024,
        o_groups=8, n_routed_experts=n_experts, n_shared_experts=1,
        num_experts_per_tok=min(6, n_experts), num_hash_layers=3,
        num_nextn_predict_layers=0, sliding_window=128, swiglu_limit=10.0,
        score_func="sqrtsoftplus", routed_scaling_factor=2.5,
        rms_norm_eps=1e-6, rope_theta=10000.0, compress_rope_theta=160000.0,
        rope_factor=16.0, rope_beta_fast=32, rope_beta_slow=1,
        rope_original_seq_len=0, max_position_embeddings=4096,
        compress_ratios=(0, 0, 4, 128)[:n_layers], index_n_heads=64,
        index_head_dim=128, index_topk=512, hc_mult=4, hc_sinkhorn_iters=20,
        hc_eps=1e-6)


n_chips = jax.local_device_count()
assert n_chips >= 1, "no TPU chips visible"
devices = create_device_mesh((1, n_chips, 1, 1, 1), allow_split_physical_axes=True)
mesh = Mesh(devices, axis_names=("data", "attn_dp", "attn_dp_expert", "expert", "model"))
print(f"mesh = {mesh}  (single-host, {n_chips} chips, attn_dp={n_chips})")

cfg = make_cfg(n_layers=4, n_experts=8)

params_struct = make_abstract_transformer_params(cfg)
leaves, treedef = jax.tree_util.tree_flatten(
    params_struct, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct))
keys = jax.random.split(jax.random.PRNGKey(0), len(leaves))

t0 = time.time()
randomized = []
for k, x in zip(keys, leaves):
    spec = pick_partition_spec(x.shape, mesh)
    arr = (jax.random.normal(k, x.shape, dtype=jnp.float32) * 0.02).astype(x.dtype)
    arr = jax.device_put(arr, NamedSharding(mesh, spec))
    randomized.append(arr)
params = jax.tree_util.tree_unflatten(treedef, randomized)
print(f"params sharded in {time.time() - t0:.1f}s")

max_seq = 4096
swa, comp = make_freqs_cis(cfg, max_seq)
sizes = v4_layer_packed_sizes_from_cfg(cfg, max_seq, batch_size=1)

kv_caches = [jax.device_put(jnp.zeros((s,), dtype=jnp.float32), NamedSharding(mesh, P()))
             for s in sizes]

T, N = 8, 8
rng = np.random.default_rng(seed=1234)
ids_full = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(1, T + N)), dtype=jnp.int32)

with mesh:
    t0 = time.time()
    h_full = transformer_body_forward(ids_full, params, swa, comp, cfg)
    logits_full = head_forward(
        h_full, params.head_w, params.final_norm_w,
        params.hc_head_fn, params.hc_head_scale, params.hc_head_base,
        cfg.rms_norm_eps, cfg.hc_eps)
    argmax_full = jnp.argmax(logits_full, axis=-1)
    argmax_full.block_until_ready()
    print(f"eager fresh-prefill argmax: {time.time() - t0:.1f}s")

    @functools.partial(jax.jit, donate_argnums=0)
    def jit_prefill(kv_caches, ids):
        return deepseek_v4_run_with_decode_state(
            kv_caches, ids, params, swa, comp, cfg,
            state_max_seq_len=max_seq, is_decode_step=False,
            start_pos=jnp.int32(0))

    @functools.partial(jax.jit, donate_argnums=0)
    def jit_decode(kv_caches, ids, start_pos):
        return deepseek_v4_run_with_decode_state(
            kv_caches, ids, params, swa, comp, cfg,
            state_max_seq_len=max_seq, is_decode_step=True,
            start_pos=start_pos)

    t0 = time.time()
    kv_caches, _ = jit_prefill(kv_caches, ids_full[:, :T])
    print(f"jit prefill: {time.time() - t0:.1f}s")

    n_bad = 0
    t0 = time.time()
    for step in range(N):
        pos = T + step
        kv_caches, h_step = jit_decode(
            kv_caches, ids_full[:, pos:pos+1], jnp.int32(pos))
        logits_step = head_forward(
            h_step, params.head_w, params.final_norm_w,
            params.hc_head_fn, params.hc_head_scale, params.hc_head_base,
            cfg.rms_norm_eps, cfg.hc_eps)
        argmax_step = int(jnp.argmax(logits_step[0, 0]))
        argmax_ref = int(argmax_full[0, pos])
        match = argmax_step == argmax_ref
        if not match:
            n_bad += 1
        mark = "  " if match else "**"
        print(f"  [sharded jit] {mark} step={step:>2} pos={pos:>2}  "
              f"argmax_step={argmax_step:>5}  argmax_ref={argmax_ref:>5}  "
              f"{'OK' if match else 'MISMATCH'}")
    print(f"jit decode: {time.time() - t0:.1f}s, bad={n_bad}/{N}")

if n_bad == 0:
    print("OK: sharded jit+donate decode matches fresh-prefill argmax")
    sys.exit(0)
print(f"FAIL: {n_bad}/{N} mismatches on sharded mesh")
sys.exit(2)

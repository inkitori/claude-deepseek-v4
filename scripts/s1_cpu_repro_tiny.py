"""Standalone reproducer for S1: JIT+donation decode drift in V4.

Bypasses pytest+conftest+torch. Hand-builds tiny DeepseekV4Config, random
JAX params via make_abstract_transformer_params, runs prefill+N decode
steps under jax.jit(donate_argnums=0), and compares argmax to fresh
prefill argmax.

If `argmax_match` flips to False on step 1+, S1 is reproduced.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import functools
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

from tpu_inference.models.jax.deepseek_v4 import (
    DeepseekV4Config,
    make_abstract_transformer_params,
    make_freqs_cis,
    transformer_body_forward,
    head_forward,
    deepseek_v4_run_with_decode_state,
    v4_layer_packed_sizes_from_cfg,
)


def make_tiny_cfg():
    # Mirror `make_tiny_args` defaults + DeepseekV4Config field-name translation.
    # All three attention flavors are present in compress_ratios.
    return DeepseekV4Config(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=64,
        moe_intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=16,
        q_lora_rank=64,
        o_lora_rank=64,
        o_groups=2,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        num_nextn_predict_layers=1,
        sliding_window=8,
        swiglu_limit=0.0,
        score_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=16.0,
        rope_beta_fast=32,
        rope_beta_slow=1,
        rope_original_seq_len=0,
        max_position_embeddings=256,
        compress_ratios=(0, 0, 4, 128, 4, 0, 0),
        index_n_heads=4,
        index_head_dim=16,
        index_topk=8,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )


def make_random_params(cfg, seed=0):
    params_struct = make_abstract_transformer_params(cfg)
    leaves, treedef = jax.tree_util.tree_flatten(
        params_struct, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct))
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    randomized = [
        (jax.random.normal(k, x.shape, dtype=jnp.float32) * 0.02).astype(x.dtype)
        for k, x in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, randomized)


def run_test(label, jit_donate, T=8, N=12, seed=1234):
    cfg = make_tiny_cfg()
    params = make_random_params(cfg, seed=0)
    max_seq = 64
    swa, comp = make_freqs_cis(cfg, max_seq)
    sizes = v4_layer_packed_sizes_from_cfg(cfg, max_seq, batch_size=1)

    rng = np.random.default_rng(seed=seed)
    ids_full = jnp.asarray(
        rng.integers(0, cfg.vocab_size, size=(1, T + N)), dtype=jnp.int32)

    # Fresh-prefill argmax reference.
    h_full = transformer_body_forward(ids_full, params, swa, comp, cfg)
    logits_full = head_forward(
        h_full, params.head_w, params.final_norm_w,
        params.hc_head_fn, params.hc_head_scale, params.hc_head_base,
        cfg.rms_norm_eps, cfg.hc_eps)
    argmax_full = jnp.argmax(logits_full, axis=-1)

    if jit_donate:
        @functools.partial(jax.jit, donate_argnums=0)
        def run_prefill(kv_caches, ids):
            return deepseek_v4_run_with_decode_state(
                kv_caches, ids, params, swa, comp, cfg,
                state_max_seq_len=max_seq, is_decode_step=False,
                start_pos=jnp.int32(0))

        @functools.partial(jax.jit, donate_argnums=0)
        def run_decode(kv_caches, ids, start_pos):
            return deepseek_v4_run_with_decode_state(
                kv_caches, ids, params, swa, comp, cfg,
                state_max_seq_len=max_seq, is_decode_step=True,
                start_pos=start_pos)
    else:
        def run_prefill(kv_caches, ids):
            return deepseek_v4_run_with_decode_state(
                kv_caches, ids, params, swa, comp, cfg,
                state_max_seq_len=max_seq, is_decode_step=False,
                start_pos=jnp.int32(0))

        def run_decode(kv_caches, ids, start_pos):
            return deepseek_v4_run_with_decode_state(
                kv_caches, ids, params, swa, comp, cfg,
                state_max_seq_len=max_seq, is_decode_step=True,
                start_pos=start_pos)

    kv_caches = [jnp.zeros((s,), dtype=jnp.float32) for s in sizes]
    t0 = time.time()
    kv_caches, _ = run_prefill(kv_caches, ids_full[:, :T])
    print(f"[{label}] prefill: {time.time()-t0:.1f}s")

    n_bad = 0
    t0 = time.time()
    for step in range(N):
        pos = T + step
        kv_caches, h_step = run_decode(
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
        print(f"  [{label}] {mark} step={step:>2} pos={pos:>2}  "
              f"argmax_step={argmax_step:>5}  argmax_ref={argmax_ref:>5}  "
              f"{'OK' if match else 'MISMATCH'}")
    print(f"[{label}] decode: {time.time()-t0:.1f}s, bad={n_bad}/{N}")
    return n_bad


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    if mode in ("eager", "both"):
        print("=== eager (no jit) ===")
        n_eager = run_test("eager", jit_donate=False, T=T, N=N)
    if mode in ("jit", "both"):
        print("=== jit+donate ===")
        n_jit = run_test("jit", jit_donate=True, T=T, N=N)

    print()
    if mode == "both":
        if n_eager == 0 and n_jit > 0:
            print(f"S1 REPRODUCED: eager all match, jit {n_jit}/{N} mismatch")
            sys.exit(2)
        elif n_eager > 0 and n_jit > 0:
            print(f"BOTH mismatch (eager={n_eager}, jit={n_jit})")
            sys.exit(3)
        elif n_eager == 0 and n_jit == 0:
            print("OK: both eager and jit match fresh-prefill argmax")
            sys.exit(0)
    else:
        sys.exit(0)

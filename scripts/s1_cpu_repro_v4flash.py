"""Standalone reproducer for S1 using real V4-Flash dimensions (4 layers, 8 experts).

Mirrors `TestRealConfigDecodeStability::_build_setup(n_layers=4, n_experts=8)`
but without needing the v4_flash/config.json file. Hand-builds the
DeepseekV4Config with V4-Flash dimensions from TINY_CONFIG.md.

Expected: ~3 min on CPU; reproduces S1 (eager OK, jit+donate mismatch).
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


def make_v4_flash_truncated_cfg(n_layers=4, n_experts=8):
    """V4-Flash dimensions per TINY_CONFIG.md, truncated to n_layers and n_experts.
    compress_ratios[:4] from V4-Flash = [0, 0, 4, 128] — covers all three flavors."""
    return DeepseekV4Config(
        vocab_size=129280,
        hidden_size=4096,
        intermediate_size=2048,
        moe_intermediate_size=2048,
        num_hidden_layers=n_layers,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1024,
        o_lora_rank=1024,
        o_groups=8,
        n_routed_experts=n_experts,
        n_shared_experts=1,
        num_experts_per_tok=min(6, n_experts),
        num_hash_layers=3,
        num_nextn_predict_layers=0,
        sliding_window=128,
        swiglu_limit=10.0,
        score_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=16.0,
        rope_beta_fast=32,
        rope_beta_slow=1,
        rope_original_seq_len=0,
        max_position_embeddings=4096,
        compress_ratios=(0, 0, 4, 128)[:n_layers],
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )


def make_random_params(cfg, seed=0):
    params_struct = make_abstract_transformer_params(cfg)
    leaves, treedef = jax.tree_util.tree_flatten(
        params_struct, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct))
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))

    def _rand(k, x):
        # QUANT fp4 leaves: packed-FP4 expert weights are uint8 (full-range bytes
        # -> exercises the whole e2m1 codebook, both signs); e8m0 block scales are
        # float8_e8m0fnu, synthesized small (bytes 118..121 -> 2^-9..2^-6) so the
        # dequanted experts are ~0.02-magnitude (the bf16-baseline regime) and the
        # model stays in a numerically stable, near-tie-free decode trajectory.
        if x.dtype == jnp.uint8:
            return jax.random.randint(k, x.shape, 0, 256, dtype=jnp.uint8)
        if x.dtype == jnp.float8_e8m0fnu:
            b = jax.random.randint(k, x.shape, 118, 122, dtype=jnp.uint8)
            return jax.lax.bitcast_convert_type(b, jnp.float8_e8m0fnu)
        return (jax.random.normal(k, x.shape, dtype=jnp.float32) * 0.02).astype(x.dtype)

    randomized = [_rand(k, x) for k, x in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(treedef, randomized)


def run_test(label, jit_donate, cfg, params, swa, comp, max_seq, T=8, N=12, seed=1234,
             argmax_full=None):
    sizes = v4_layer_packed_sizes_from_cfg(cfg, max_seq, batch_size=1)

    rng = np.random.default_rng(seed=seed)
    ids_full = jnp.asarray(
        rng.integers(0, cfg.vocab_size, size=(1, T + N)), dtype=jnp.int32)

    if argmax_full is None:
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
    return n_bad, argmax_full


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    n_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    print(f"Setup: n_layers={n_layers}, T={T}, N={N}, mode={mode}")
    cfg = make_v4_flash_truncated_cfg(n_layers=n_layers, n_experts=8)
    t0 = time.time()
    params = make_random_params(cfg, seed=0)
    print(f"Params built in {time.time()-t0:.1f}s")

    # state_max_seq // 4 must be >= index_topk; index_topk=512, so >=2048.
    max_seq = 4096
    swa, comp = make_freqs_cis(cfg, max_seq)

    argmax_full = None
    n_eager = 0
    if mode in ("eager", "both"):
        print("=== eager (no jit) ===")
        n_eager, argmax_full = run_test(
            "eager", jit_donate=False, cfg=cfg, params=params, swa=swa, comp=comp,
            max_seq=max_seq, T=T, N=N, argmax_full=argmax_full)
    if mode in ("jit", "both"):
        print("=== jit+donate ===")
        n_jit, _ = run_test(
            "jit", jit_donate=True, cfg=cfg, params=params, swa=swa, comp=comp,
            max_seq=max_seq, T=T, N=N, argmax_full=argmax_full)

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
    elif mode == "jit":
        if n_jit > 0:
            print(f"JIT mismatch: {n_jit}/{N}")
            sys.exit(2)
        sys.exit(0)
    elif mode == "eager":
        if n_eager > 0:
            print(f"EAGER mismatch: {n_eager}/{N}")
            sys.exit(3)
        sys.exit(0)

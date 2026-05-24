"""Isolated parity: parallel compressor_prefill vs incremental
compressor_decode_step (zero-state, all positions), ratio=4, PEAKED weights.
Compares the finalized compressed blocks. If they diverge -> compressor is S1."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import numpy as np, jax, jax.numpy as jnp
import tpu_inference.layers.jax.attention.deepseek_v4_attention as A

def make_freqs(max_seq, rope_head_dim, theta=10000.0):
    half = rope_head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, half) / half))
    ang = np.outer(np.arange(max_seq), inv)              # [max_seq, half]
    return jnp.asarray(np.exp(1j * ang), dtype=jnp.complex64)

def run(scale, ratio=4, S=8, dim=16, head_dim=8, rope_head_dim=4, seed=0):
    coff = 2 if ratio == 4 else 1
    rng = np.random.default_rng(seed)
    def R(*shp): return jnp.asarray(rng.standard_normal(shp) * scale, dtype=jnp.float32)
    params = A.CompressorParams(
        ape=R(ratio, coff * head_dim), wkv=R(coff * head_dim, dim),
        wgate=R(coff * head_dim, dim), norm_w=jnp.asarray(rng.standard_normal(head_dim)*0.1+1.0, jnp.float32),
        head_dim=head_dim, rope_head_dim=rope_head_dim, compress_ratio=ratio,
        norm_eps=1e-6, rotate=True)
    freqs = make_freqs(max_seq=64, rope_head_dim=rope_head_dim)
    x = R(1, S, dim)
    # parallel
    blocks_pre = A.compressor_prefill(x, params, freqs)      # [1, S//ratio, head_dim]
    # incremental
    kv_state, score_state = A.compressor_init_state(1, head_dim, ratio)
    blocks_dec = []
    for pos in range(S):
        kv_state, score_state, kv_comp, did = A.compressor_decode_step(
            x[:, pos:pos+1], pos, params, freqs, kv_state, score_state)
        if bool(did):
            blocks_dec.append(kv_comp)   # [1,1,head_dim]
    blocks_dec = jnp.concatenate(blocks_dec, axis=1) if blocks_dec else jnp.zeros((1,0,head_dim))
    nb = min(blocks_pre.shape[1], blocks_dec.shape[1])
    print(f"scale={scale}: prefill_blocks={blocks_pre.shape[1]} decode_blocks={blocks_dec.shape[1]}")
    for b in range(nb):
        bp = blocks_pre[:, b]; bd = blocks_dec[:, b]
        rel = float(jnp.linalg.norm(bd - bp) / (jnp.linalg.norm(bp) + 1e-9))
        print(f"  block {b}: relErr={rel:.5f}  {'<<< DIVERGE' if rel > 0.05 else ''}")

for s in (0.02, 0.6):
    run(s)

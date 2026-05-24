"""Test the prefill->decode SEED at PEAKED weights: does
_compressor_state_from_prefill(x[:T]) + decode steps produce the SAME compressed
blocks as parallel compressor_prefill over the full sequence? The zero-state
incremental path already matched prefill (relErr 0); the seed is the untested gap."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import numpy as np, jax, jax.numpy as jnp
import tpu_inference.layers.jax.attention.deepseek_v4_attention as A

def make_freqs(max_seq, rhd, theta=10000.0):
    half = rhd // 2
    inv = 1.0/(theta**(np.arange(0,half)/half))
    return jnp.asarray(np.exp(1j*np.outer(np.arange(max_seq), inv)), dtype=jnp.complex64)

def run(scale, T, M, ratio=4, dim=16, head_dim=8, rhd=4, seed=0):
    coff = 2 if ratio==4 else 1
    rng = np.random.default_rng(seed)
    def R(*s): return jnp.asarray(rng.standard_normal(s)*scale, jnp.float32)
    p = A.CompressorParams(ape=R(ratio,coff*head_dim), wkv=R(coff*head_dim,dim),
        wgate=R(coff*head_dim,dim), norm_w=jnp.asarray(rng.standard_normal(head_dim)*0.1+1,jnp.float32),
        head_dim=head_dim, rope_head_dim=rhd, compress_ratio=ratio, norm_eps=1e-6, rotate=True)
    freqs = make_freqs(64, rhd)
    x = R(1, M, dim)
    blocks_pre = A.compressor_prefill(x, p, freqs)                 # parallel, full seq
    # SEED at T, then decode positions T..M-1
    kv_state, score_state = A._compressor_state_from_prefill(x[:, :T], p)
    blocks_dec = {}
    for pos in range(T, M):
        kv_state, score_state, kvc, did = A.compressor_decode_step(
            x[:, pos:pos+1], pos, p, freqs, kv_state, score_state)
        if bool(did):
            b = (pos+1)//ratio - 1     # block index just finalized
            blocks_dec[b] = kvc
    print(f"scale={scale} T={T} M={M}: blocks finalized during decode = {sorted(blocks_dec)}")
    worst = 0.0
    for b, kvc in sorted(blocks_dec.items()):
        bp = blocks_pre[:, b]
        rel = float(jnp.linalg.norm(kvc[:,0] - bp)/(jnp.linalg.norm(bp)+1e-9))
        worst = max(worst, rel)
        print(f"  block {b}: relErr={rel:.5f}  {'<<< SEED DIVERGES' if rel>0.05 else ''}")
    return worst

for scale in (0.02, 0.6):
    # T=9 (remainder 1), M=16 -> boundaries at pos 11,15 finalize blocks 2,3 during decode
    run(scale, T=9, M=16)

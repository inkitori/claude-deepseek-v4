#!/usr/bin/env python3
"""TPU micro-benchmark: the V4 DECODE NON-MoE device cost (roadmap #1, the ~33 ms).

P.6 proved the MoE expert-FFN (~106 ms, 76% of the 138.9 ms device_wait) is at its
LOSSLESS FLOOR. The remaining ~33 ms is everything else per decode step: attention
(43 layers) + the CSA indexer top_k/score (21 layers) + logits + misc. That 33 ms is
currently a RESIDUAL (device_wait 138.9 - MoE 105.8), never directly attributed -- the
only on-disk trace is prefill-dominated + first-exec tainted (HANDOFF), so we measure
the prime suspects HERE, in isolation, on the real 16-chip mesh with SYNTHETIC inputs at
the real V4-Flash decode shapes (NO full-model load, NO vllm) -- the MoE-bench method.

WHAT IS TIMED (all replicated P(), matching the decode _v4_decode_replicate path):
  * sparse_attn: the Mosaic KERNEL (production, `_sparse_attn_kernel_sharded`) vs the
    math-identical pure-JAX REF (`sparse_attn`, the CPU-oracle path) -- at all 3 decode
    KV-cache shapes. If pure-JAX wins on TPU at N=1, swapping is a LOSSLESS lever (the
    same shape of win P.6 found for the MoE: XLA's dense path beat the in-trace kernel).
    Parity is proven (test_kernel_matches_jax_sparse_attn) so the swap can't shift the GATE.
  * indexer top_k: `lax.top_k([1,1,1024], 512)` -- the CSA-layer sort over the static
    compressed buffer (only ratio-4 layers; ratio-128/HCA use a deterministic index).
  * indexer score block: the 64-head score einsum + relu + head-sum feeding that top_k.

DECODE SHAPES (config.json: dim=4096 H=64 head_dim=512 q_lora=1024 index_topk=512
sliding_window=128, MAX_LEN=4096; kv_cache = [1, win+MAX_LEN//ratio, 512]):
  CSA  (ratio 4,   x21 layers): kv=[1,1152,512] topk_idxs=[1,1,640]  (win128 + index_topk512)
  HCA  (ratio 128, x20 layers): kv=[1, 160,512] topk_idxs=[1,1,160]  (win128 + 32 compressed)
  dense(ratio 0,   x2  layers): kv=[1, 128,512] topk_idxs=[1,1,128]  (win128 only)

PER-STEP ROLLUP (what to compare against the ~33 ms residual):
  sparse_attn/step = 21*CSA + 20*HCA + 2*dense ; indexer/step = 21*(topk + score).

Run (TPU FREE; SYNC first -- mh_run runs each host's own clone):
    scripts/full_slice_v4_sync.sh
    MH_TIMEOUT=900 scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_attn_decode.py --distributed
"""
import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from tpu_inference.layers.jax.attention.deepseek_v4_attention import (
    sparse_attn, _sparse_attn_kernel_sharded)

# Real DeepSeek-V4-Flash decode dims (config.json).
DIM = 4096
H = 64               # num_attention_heads
DH = 512             # head_dim
WIN = 128            # sliding_window
INDEX_TOPK = 512     # index_topk
INDEX_H = 64         # index_n_heads
INDEX_DH = 128       # index_head_dim
MAX_LEN = 4096       # the serving buffer (V4_DECODE_TIMERS ran MAX_LEN=4096)
SCALE = float(DH ** -0.5)

# Layer-type counts from config.compress_ratios (43 real layers).
N_CSA, N_HCA, N_DENSE = 21, 20, 2

# Per-shape (label, kv_len, topk_len). kv_len = WIN + MAX_LEN//ratio (or WIN for dense).
SHAPES = [
    ("CSA",   WIN + MAX_LEN // 4,   WIN + INDEX_TOPK),   # 1152, 640
    ("HCA",   WIN + MAX_LEN // 128, WIN + MAX_LEN // 128),  # 160, 160
    ("dense", WIN,                  WIN),                # 128, 128
]
LAYER_COUNT = {"CSA": N_CSA, "HCA": N_HCA, "dense": N_DENSE}


def make_replicated(shape, seed, mesh, dtype=jnp.bfloat16):
    sh = NamedSharding(mesh, P())
    rng = np.random.default_rng(seed)
    base = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    return jax.make_array_from_callback(shape, sh,
                                        lambda idx, _b=base: _b[idx]).astype(dtype)


def make_idx(K, N, seed, mesh):
    """Replicated [1,1,K] int32 topk indices in [0,N), a few -1 sentinels.
    (Timing is K-determined: the kernel/gather process all K slots; validity only
    masks the softmax.)"""
    sh = NamedSharding(mesh, P())
    rng = np.random.default_rng(seed)
    idx = (rng.integers(0, N, size=(1, 1, K))).astype(np.int32)
    idx[0, 0, -3:] = -1  # a few sentinels, like a partially-filled buffer tail
    return jax.make_array_from_callback((1, 1, K), sh, lambda i, _x=idx: _x[i])


def make_sink(seed, mesh):
    sh = NamedSharding(mesh, P())
    rng = np.random.default_rng(seed)
    s = (rng.standard_normal((H,)) * 0.1).astype(np.float32)
    return jax.make_array_from_callback((H,), sh, lambda i, _s=s: _s[i])


def indexer_score_block(qf, kvf, weights):
    """The CSA indexer score compute feeding top_k (deepseek_v4_attention.py:651-653).
    qf [B,1,index_H,index_Dh] fp32, kvf [B, MAX_LEN//4, index_Dh] fp32, weights [B,1,index_H]."""
    index_score = jnp.einsum("bshd,btd->bsht", qf, kvf)
    index_score = jax.nn.relu(index_score) * weights[..., None]
    return index_score.sum(axis=2)  # [B,1,MAX_LEN//4]


def lax_top_k(s, k):
    v, _ = lax.top_k(s, k)
    return v


def measure(f, iters, warmup=8):
    for _ in range(warmup):
        jax.block_until_ready(f())
    total = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        total.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(total), min(total)


def amortized_ms(op, primary, others, iters, R=64):
    """True in-program device-ms PER CALL of `op`, with the single-dispatch +
    block_until_ready overhead amortized over R on-device iterations (a fori_loop,
    so no unroll/CSE). `op(primary_perturbed, *others) -> array`; we perturb
    `primary` by the fp32 carry (x1e-9) so XLA can't hoist the call out of the loop
    (loop-invariant code motion). Returns (median_per_call_ms, min_per_call_ms)."""
    def loop(primary, *others):
        def body(i, acc):
            pp = primary + (acc * 1e-9).astype(primary.dtype)
            o = op(pp, *others)
            return acc + o.astype(jnp.float32).reshape(-1)[0]
        return lax.fori_loop(0, R, body, jnp.float32(0.0))
    f = jax.jit(loop)
    args = (primary, *others)
    m, n = measure(lambda: f(*args), iters)
    return m / R, n / R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--distributed", action="store_true")
    a = ap.parse_args()
    if a.distributed:
        jax.distributed.initialize()
    nd = jax.device_count()
    p0 = (not a.distributed) or jax.process_index() == 0
    devices = np.array(jax.devices()).reshape(nd)
    mesh = jax.sharding.Mesh(devices, ('attn_dp',))

    rows = []  # (label, med, min) for sparse_attn shapes (kernel + jax)
    with jax.set_mesh(mesh):
        # ---- sparse_attn: Mosaic KERNEL vs pure-JAX REF, per decode shape. ----
        q = make_replicated((1, 1, H, DH), 1, mesh)           # [1,1,64,512] bf16
        sink = make_sink(2, mesh)                              # [64] fp32
        kern_times, jax_times = {}, {}          # single-call median (dispatch-inflated)
        akern_times, ajax_times = {}, {}         # amortized device-ms/call (dispatch removed)
        for name, kv_len, topk_len in SHAPES:
            kv = make_replicated((1, kv_len, DH), 100 + kv_len, mesh)   # [1,N,512] bf16
            idx = make_idx(topk_len, kv_len, 200 + topk_len, mesh)      # [1,1,K] int32
            kern = lambda q, kv, sink, idx: _sparse_attn_kernel_sharded(q, kv, sink, idx, SCALE)
            jref = lambda q, kv, sink, idx: sparse_attn(q, kv, sink, idx, SCALE)
            kern_f, jax_f = jax.jit(kern), jax.jit(jref)
            km, kn = measure(lambda: kern_f(q, kv, sink, idx), a.iters)
            jm, jn = measure(lambda: jax_f(q, kv, sink, idx), a.iters)
            ak, _ = amortized_ms(kern, q, (kv, sink, idx), a.iters)
            aj, _ = amortized_ms(jref, q, (kv, sink, idx), a.iters)
            kern_times[name], jax_times[name] = km, jm
            akern_times[name], ajax_times[name] = ak, aj
            # parity (the swap must be GATE-safe): max|kernel - jax|
            d = float(jnp.max(jnp.abs(kern_f(q, kv, sink, idx).astype(jnp.float32)
                                      - jax_f(q, kv, sink, idx).astype(jnp.float32))))
            rows.append((name, kv_len, topk_len, km, kn, jm, jn, d, ak, aj))

        # ---- indexer (CSA only): top_k over the static compressed buffer + score block. ----
        ksz = MAX_LEN // 4  # 1024
        score = make_replicated((1, 1, ksz), 7, mesh, dtype=jnp.float32)
        topk_f = jax.jit(lambda s: lax_top_k(s, INDEX_TOPK))
        tkm, tkn = measure(lambda: topk_f(score), a.iters)
        atk, _ = amortized_ms(lambda s: lax_top_k(s, INDEX_TOPK), score, (), a.iters)

        qf = make_replicated((1, 1, INDEX_H, INDEX_DH), 8, mesh, dtype=jnp.float32)
        kvf = make_replicated((1, ksz, INDEX_DH), 9, mesh, dtype=jnp.float32)
        weights = make_replicated((1, 1, INDEX_H), 10, mesh, dtype=jnp.float32)
        score_f = jax.jit(indexer_score_block)
        scm, scn = measure(lambda: score_f(qf, kvf, weights), a.iters)
        asc, _ = amortized_ms(indexer_score_block, qf, (kvf, weights), a.iters)

    if not p0:
        return

    print("\n=== V4 DECODE NON-MoE microbench (16-chip, N=1, replicated) ===")
    print(f"dims: dim={DIM} H={H} head_dim={DH} win={WIN} index_topk={INDEX_TOPK} MAX_LEN={MAX_LEN}")
    print("single-call med/min is DISPATCH-INFLATED (~0.18 ms host round-trip floor); the")
    print("AMORTIZED column (fori_loop, dispatch removed) is the true in-program device-ms/call.\n")
    print("sparse_attn shape kv_len topk | KERNEL single | JAX single | k/j | AMORT kern | AMORT jax | max|d|")
    for name, kv_len, topk_len, km, kn, jm, jn, d, ak, aj in rows:
        ratio = ak / aj if aj else 0.0
        print(f"  {name:5s}          {kv_len:5d} {topk_len:4d} | "
              f"med {km:5.3f} min {kn:5.3f} | med {jm:5.3f} | {ratio:4.2f}x | "
              f"{ak:6.3f} | {aj:6.3f} | {d:.1e}")

    print(f"\nindexer top_k lax.top_k([1,1,{ksz}],{INDEX_TOPK}): single med {tkm:5.3f} | AMORT {atk:6.3f} ms")
    print(f"indexer score einsum+relu+sum (64h x {ksz})     : single med {scm:5.3f} | AMORT {asc:6.3f} ms")

    # ---- per-step rollups on the AMORTIZED (trustworthy) device-ms. ----
    def rollup(times):
        return sum(LAYER_COUNT[n] * times[n] for n in LAYER_COUNT)
    sa_kern = rollup(akern_times)
    sa_jax = rollup(ajax_times)
    idx_step = N_CSA * (atk + asc)
    print("\n--- PER-STEP rollup, AMORTIZED device-ms (21 CSA + 20 HCA + 2 dense) ---")
    print(f"  sparse_attn KERNEL  : {sa_kern:6.2f} ms/step  "
          f"(CSA {akern_times['CSA']:.3f} x21, HCA {akern_times['HCA']:.3f} x20, dense {akern_times['dense']:.3f} x2)")
    print(f"  sparse_attn JAX-ref : {sa_jax:6.2f} ms/step  (kernel/jax = {sa_kern/sa_jax:.3f}x)")
    print(f"  indexer topk+score  : {idx_step:6.2f} ms/step  (atk {atk:.3f} + asc {asc:.3f}) x21 CSA")
    print(f"  => KERNEL sparse_attn + indexer = {sa_kern + idx_step:6.2f} ms/step of the ~33 ms NON-MoE")
    print(f"     (remainder ~{33 - (sa_kern + idx_step):.0f} ms = Q/KV/O projections + gate + logits + misc)")
    if sa_jax < sa_kern:
        print(f"\n  *** pure-JAX sparse_attn FASTER by {sa_kern - sa_jax:.2f} ms/step => LOSSLESS swap lever. ***")
    else:
        print(f"\n  Mosaic kernel WINS ({sa_kern:.2f} vs jax {sa_jax:.2f} ms/step, {sa_jax/sa_kern:.0f}x) "
              f"=> kernel is optimal; NO swap lever (parity bit-identical).")


if __name__ == "__main__":
    main()

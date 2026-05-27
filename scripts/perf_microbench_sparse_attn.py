#!/usr/bin/env python3
"""TPU micro-benchmark for `sparse_attn` — the V4 decode/prefill bottleneck.

Times the REAL `sparse_attn` (imported from tpu_inference, so edits to it are
measured here) on TPU with SYNTHETIC inputs at realistic decode/prefill shapes.
NO 543 GiB weight load, NO vllm, NO full-model compile — this is the CHEAP inner
loop for the perf campaign (Phase 0.0): ~1 min vs a 25-45 min full smoke.

`sparse_attn` is a purely LOCAL per-device op (no collectives / no sharding
constraint), so each process times its own local chip and the per-device ms is
what matters. The v6e-32 slice CANNOT boot TPU from a lone host, though, so this
must run across all 8 hosts with --distributed (jax.distributed.initialize); the
launcher below handles the fan-out + the mandatory pre-clean.

Realistic shapes (H=64, D=512, scale=1/sqrt(512)); B=1; window=128:
  decode  (M=1):  swa N=K=128 | csa N=K=128+msl/4 | hca N=K=128+msl/128
  prefill (M=S):  swa N=S,K=min(S,128) | csa N=S+S/4,K=min(S,128)+min(512,S/4)
                  | hca N=S+S/128,K=min(S,128)+S/128

Run (TPU must be FREE — `scripts/full_slice_v4_reset.sh` to stop any engine):
    scripts/perf_microbench.sh --all        # pre-cleans hosts, fans out via mh_run
  read process-0's tail for the timing table. NOTE: the script must be SYNCED to
  all 8 hosts first (scripts/full_slice_v4_sync.sh) — mh_run runs each host's own
  clone.

CPU NUMERICS (the correctness gate for any sparse_attn change) is the EXISTING
test, not this script:
    JAX_PLATFORMS=cpu PYTHONPATH=work/tpu-inference:work/vllm \
      work/vllm_env/bin/python3 -m pytest -q \
      work/tpu-inference/tests/models/jax/test_deepseek_v4.py::TestSparseAttnComponent
"""
import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.layers.jax.attention.deepseek_v4_attention import sparse_attn

H, D = 64, 512
SCALE = 1.0 / (D ** 0.5)
WIN = 128

# preset -> function(seq, msl) -> (B, M, K, N)
PRESETS = {
    "decode_swa": lambda s, m: (1, 1, WIN, WIN),
    "decode_csa": lambda s, m: (1, 1, WIN + m // 4, WIN + m // 4),
    "decode_hca": lambda s, m: (1, 1, WIN + m // 128, WIN + m // 128),
    "prefill_swa": lambda s, m: (1, s, min(s, WIN), s),
    "prefill_csa": lambda s, m: (1, s, min(s, WIN) + min(512, s // 4), s + s // 4),
    "prefill_hca": lambda s, m: (1, s, min(s, WIN) + s // 128, s + s // 128),
}


def make_inputs(B, M, K, N, dev, seed=0):
    """Synthetic q/kv/attn_sink/topk_idxs placed on `dev`. ~3% of topk slots are
    the -1 pad sentinel to exercise the mask path (matches real padded prefill)."""
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((B, M, H, D)).astype(np.float32), jnp.bfloat16)
    kv = jnp.asarray(rng.standard_normal((B, N, D)).astype(np.float32), jnp.bfloat16)
    attn_sink = jnp.asarray(rng.standard_normal(H).astype(np.float32))
    idx = rng.integers(0, N, size=(B, M, K)).astype(np.int32)
    idx[rng.random((B, M, K)) < 0.03] = -1
    topk_idxs = jnp.asarray(idx)
    put = lambda x: jax.device_put(x, dev)
    return put(q), put(kv), put(attn_sink), put(topk_idxs)


def bench(fn, args, warmup=5, iters=50):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    return ts


def run_preset(name, seq, msl, dev):
    B, M, K, N = PRESETS[name](seq, msl)
    args = make_inputs(B, M, K, N, dev)
    fn = jax.jit(lambda q, kv, s, idx: sparse_attn(q, kv, s, idx, SCALE))
    ts = bench(fn, args)
    med, mn = statistics.median(ts), min(ts)
    # Gathered tensor kv_gathered[B,M,K,D]: the materialized intermediate. Report
    # its fp32 size (current code casts kv->fp32 at :181 before gather) and the
    # effective bandwidth to move it once. Bytes drop to bf16 (/2) after Phase 0.2.
    gathered_elems = B * M * K * D
    gathered_mb_fp32 = gathered_elems * 4 / 1e6
    bw_fp32 = gathered_elems * 4 / (med / 1e3) / 1e9  # GB/s (read of fp32 gather)
    print(f"{name:13s} B={B} M={M:>5d} K={K:>5d} N={N:>5d} | "
          f"median {med:8.3f} ms  min {mn:8.3f} ms | "
          f"gathered {gathered_mb_fp32:8.1f} MB fp32  ~{bw_fp32:7.1f} GB/s")
    return name, med, mn, gathered_mb_fp32, bw_fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), help="single preset")
    ap.add_argument("--all", action="store_true", help="run all 6 presets")
    ap.add_argument("--seq", type=int, default=2048, help="prefill seq len S")
    ap.add_argument("--msl", type=int, default=4096, help="decode state_max_seq_len")
    ap.add_argument("--distributed", action="store_true",
                    help="jax.distributed.initialize() for the multi-host pod "
                         "(run under scripts/full_slice_v4_mh_run.sh)")
    args = ap.parse_args()

    if args.distributed:
        jax.distributed.initialize()
        dev = jax.local_devices()[0]
        tag = f"[p{jax.process_index()}]"
    else:
        dev = jax.devices()[0]
        tag = ""

    print(f"{tag} backend={jax.default_backend()} dev={dev} "
          f"global_devices={jax.device_count()} | seq={args.seq} msl={args.msl}")
    names = list(PRESETS) if args.all else [args.preset or "decode_csa"]
    for n in names:
        run_preset(n, args.seq, args.msl, dev)


if __name__ == "__main__":
    main()

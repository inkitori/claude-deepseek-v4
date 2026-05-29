#!/usr/bin/env python3
"""TPU micro-benchmark: the V4 decode JAX HOST-DISPATCH cost (roadmap L1).

The 2026-05-29 profile pinned ~129 ms/decode-step on HOST-side JAX dispatch: TWO
jits/step (`run_model` + `run_compute_logits`), each ~59 ms in
`PjitFunction.ParseArguments` == per-call pytree flatten + per-leaf abstract-value
(shape / dtype / sharding) extraction + dispatch-cache-key hash. Audit
(2026-05-29): `run_model` takes the full ~1,492-leaf WEIGHT pytree (`state`) as an
EXPLICIT arg every step, so all ~1,492 sharded leaves are re-flattened + hashed
per token (model_loader.py:353 + tpu_runner.py:881).

This reproduces that cost with a SYNTHETIC sharded pytree (NO weights, NO model, NO
vllm) on the real 4x4 / 16-chip mesh, and SWEEPS the leaf count to confirm the
host cost (a) is real + steady-state and (b) scales ~linearly with leaf COUNT
(per-leaf, NOT per-byte -- JAX 0.9.2). That SIZES the prize of trimming the arg
pytree (roadmap L1b: stack/concat weights into far fewer leaves) BEFORE any
25-45 min smoke. It separates HOST dispatch (non-blocking enqueue ==
ParseArguments) from the device round-trip (block_until_ready). NB: real decode
pays this cost 2x/step (run_model + run_compute_logits) -- fusing them is L1a.

Run (TPU must be FREE -- `scripts/full_slice_v4_reset.sh` first; SYNC to 4 hosts
via `scripts/full_slice_v4_sync.sh` -- mh_run runs each host's own clone):
    scripts/perf_microbench_decode.sh --sweep     # read process-0's tail
    scripts/perf_microbench_decode.sh --n-leaves 1492

CAVEAT (P.4): this builds a FLAT list of leaves -- that is the POST-preflatten floor
(cheap), NOT the real nested 1,492-leaf nnx-State the model dispatched pre-P.4 (whose
per-dispatch flatten was ~20 ms, measured on the real dataclass tree via
make_abstract_transformer_params). nnx-preflatten LANDED (P.4); this bench no longer
measures a live lever. To gauge State-flatten cost, flatten the REAL tree, not a flat list.
"""
import argparse
import statistics
import time

import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

DEFAULT_LEAVES = 1492  # measured V4-Flash run_model `state` leaf count (audit 2026-05-29)
SWEEP = [1, 50, 200, 500, 1000, 1492, 2000, 3000]
AXES = ("data", "attn_dp", "attn_dp_expert", "expert", "model")  # production axis names


def build_state(n, mesh):
    """n tiny SHARDED leaves. Parse cost is per-LEAF, not per-byte (JAX 0.9.2), so
    each leaf is a (16,16) f32 = 1 KiB array; we cycle 3 partition specs to mirror
    the heterogeneous sharding of real weights (shard dim0 / shard dim1 / replicated).
    make_array_from_callback is the multi-controller-safe placement (device_put of
    process-local data to a global sharding triggers a tiled-allgather OOM)."""
    specs = [P("model", None), P(None, "model"), P()]
    shards = [NamedSharding(mesh, s) for s in specs]
    base = np.ones((16, 16), np.float32)
    return [jax.make_array_from_callback(base.shape, shards[i % 3],
                                         lambda idx, _b=base: _b[idx])
            for i in range(n)]


def measure(state, x, iters, warmup=8):
    """Split HOST dispatch (non-blocking enqueue) from the device round-trip. Touch
    leaf 0 so nothing is provably-dead-code-eliminated before parse; ParseArguments
    still flattens + hashes ALL of `state` at the C++ dispatch boundary. Each `f` is
    a fresh jit so the per-k cache is clean. Compute is trivial -> the non-blocking
    call time is ~pure host dispatch (the device drains instantly, no backpressure)."""
    f = jax.jit(lambda st, xx: xx + st[0][0, 0])
    for _ in range(warmup):
        jax.block_until_ready(f(state, x))
    host, total = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = f(state, x)              # async: returns after host dispatch + enqueue
        t1 = time.perf_counter()
        jax.block_until_ready(y)     # drain device before next iter
        t2 = time.perf_counter()
        host.append((t1 - t0) * 1e3)
        total.append((t2 - t0) * 1e3)
    return host, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-leaves", type=int, default=DEFAULT_LEAVES)
    ap.add_argument("--sweep", action="store_true", help="sweep leaf counts")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--distributed", action="store_true",
                    help="jax.distributed.initialize() for the 4-host pod "
                         "(run under scripts/full_slice_v4_mh_run.sh)")
    a = ap.parse_args()

    if a.distributed:
        jax.distributed.initialize()
    nd = jax.device_count()
    p0 = (not a.distributed) or jax.process_index() == 0
    tag = f"[p{jax.process_index()}]" if a.distributed else ""
    devices = np.array(jax.devices()).reshape((1, 1, 1, 1, nd))
    mesh = jax.sharding.Mesh(devices, AXES)
    if p0:
        print(f"{tag} backend={jax.default_backend()} global_devices={nd} "
              f"(real decode pays HOST dispatch 2x/step)")
        print(f"{tag} {'leaves':>6s} | {'HOST dispatch ms':>26s} | {'+device ms':>10s} | {'us/leaf':>8s}")

    with jax.set_mesh(mesh):
        x = build_state(1, mesh)[0]            # one sharded (16,16) input
        counts = SWEEP if a.sweep else [a.n_leaves]
        pool = build_state(max(counts), mesh)  # build once, slice per k
        for k in counts:
            host, total = measure(pool[:k], x, a.iters)
            hmed, hmin = statistics.median(host), min(host)
            tmed = statistics.median(total)
            if p0:
                print(f"{tag} {k:>6d} | med {hmed:8.2f}  min {hmin:8.2f} | "
                      f"{tmed:8.2f} | {hmin / k * 1e3:7.2f}")


if __name__ == "__main__":
    main()

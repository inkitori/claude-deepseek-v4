# claude-deepseek-v4 — agent runbook

You're picking up a TPU-inference effort: get
`vllm serve deepseek-ai/DeepSeek-V4-Flash` running end-to-end on a
**v6e-32 TPU slice** (TP=32, 8 hosts × 4 chips). The model is a
flagship MoE — 256 FP4 experts + MLA attention + dense FP8 — about
543 GiB bf16-expanded.

The single non-negotiable goal is **fast, mathematically correct
inference with the real V4-Flash weights**. Synthetic-fixture tests
are the fast iteration loop; real-weight `vllm serve` is the gate
that defines "done".

## Minimum-delta rule (READ THIS FIRST)

The overall delta from the upstream `tpu-inference` repo before any
DeepSeek-V4 work should be **as small as possible** while keeping
the math correct and the serve fast. Every line you add is a line
the next agent has to read, sync to 8 worker hosts, and reason
about. Treat the diff against upstream as a budget, not free space.

Concrete rules:

1. **Don't add files when an existing one fits.** V4 already lives in
   `models/jax/deepseek_v4.py`, `models/jax/deepseek_v4_loader.py`,
   `layers/jax/attention/deepseek_v4_attention.py`, and
   `layers/jax/moe/deepseek_v4_moe.py`. New helpers go in those
   files unless they're genuinely independent and reusable.
2. **Don't add new test classes for variants of an existing case.**
   Add a parametrized test or fold into the existing class. Pattern
   to avoid: `TestDecodeAttentionParity` + `TestDecodeAttentionParityExtended`
   + `TestDecodeRollingParityLong` (each ~50 lines doing the same
   shape of check). Prior sessions already accreted ~33 test classes
   in `tests/models/jax/test_deepseek_v4.py` (3185 LOC, 5–10× the
   size of any peer model's test file). Consolidate when you touch.
3. **Reuse upstream layers.** Before writing a V4-specific helper,
   check if `layers/jax/{attention,moe,...}/` already has a
   primitive that does what you need (`dense_moe_fwd`, `sparse_attn`,
   `rms_norm`, etc.). The custom `deepseek_v4_attention.py` and
   `deepseek_v4_moe.py` exist because V4's MLA + sqrtsoftplus + hash
   routing genuinely don't fit the generic helpers — but verify
   that's still true for any *new* helper before duplicating.
4. **Don't add files outside the V4 namespace.** Anything that
   touches the runtime (`runner/`, `worker/`, `platforms/`) should
   be a last resort, not a first resort, and needs explicit
   justification in the commit message.
5. **Delete dead code as you go.** If a TODO has been resolved, drop
   it. If a "tier"/"keystone"/"sentinel" comment refers to a
   superseded plan, remove it. Comments rot; code stays.
6. **No re-export shim files.** `tests/models/test_deepseek_v4.py`
   exists only to re-export from `tests/models/jax/test_deepseek_v4.py`
   because some prior autonomous-task spec expected that path. It's
   a candidate for removal once you confirm nothing CI-relevant
   imports it.

When in doubt: the smaller change wins. A revert + minimal patch
beats a refactor + the same fix.

This file is the durable operational knowledge that's not obvious
from the code: cluster layout, the iterate loop, env knobs, orphan
state surfaces, and pitfalls that have already cost real time. Read
it once before doing anything; everything below has been learned
by burning iterations.

## Cluster topology

* Slice: **v6e-32** = 8 hosts × 4 chips × 32 GB HBM = 992 GiB total.
* Head: `10.164.0.41` (also `worker 0` — launch from here).
* Workers: `10.164.0.{22, 35, 36, 39, 45, 18, 30}` (worker 1–7).
* Each host is a **separate VM with its own clone of this repo**.
  There is **no shared filesystem** between hosts. A `git push` from
  the head does not propagate to workers — see "Source sync".
* Ray address: `10.164.0.41:6379`. Bootstrapped from
  `scripts/full_slice_v4_ray_restart.sh`. `ray status` should show
  8 nodes and `0.0/32.0 TPU` when idle.
* Shared venv path: `~/claude-deepseek-v4/work/vllm_env` on every
  host (mirrored once at bootstrap).
* SSH keys:
  * `~/.ssh/google_compute_engine` — cross-host SSH within the slice
    (the GCE-provisioned identity).
  * `~/.ssh/id_ed25519` — for `git push` to GitHub.
* GCS-mounted weights: `~/.cache/huggingface/hub` resolves to the
  staged HF cache layout under `gs://<bucket>/<dir>/` via
  `scripts/mount_gcs.sh`. Required for `vllm serve` (no internet).

## The iterate loop

Every change-then-test cycle is exactly:

```bash
scripts/full_slice_v4_reset.sh        # cluster cleanup; safe to re-run
scripts/full_slice_v4_sync.sh         # rsync source to all 7 worker hosts
scripts/full_slice_v4_smoke.sh        # launch vllm serve; writes pid + log
scripts/full_slice_v4_smoke_check.sh  # validate /v1/completions when ready
```

* **`full_slice_v4_reset.sh`** — handles the four orphan-state
  surfaces (api-server pid, `VLLM::EngineCore` Ray actor children,
  `/tmp/libtpu_lockfile`, leaked Ray placement groups). Does **not**
  restart Ray itself. Cheap; run it before every smoke launch.
* **`full_slice_v4_ray_restart.sh`** — heavier nuke (`ray stop --force`
  + fresh `ray start` on all 8 hosts). Reach for it only when reset
  isn't enough (e.g. `ABORTED: TPU is already in use by process X`
  even after reset).
* **`full_slice_v4_sync.sh`** — **mandatory after any code edit**.
  rsyncs `work/tpu-inference/` and `scripts/` to all 7 worker hosts.
  See pitfall #3.
* **`full_slice_v4_smoke.sh`** — launches `vllm serve` with the V4
  optimization knobs set + forwarded to Ray workers. Writes pid to
  `logs/full-slice-v4-smoke.pid` and a timestamped log to
  `logs/full-slice-v4-smoke-<TS>.log`.
* **`full_slice_v4_smoke_check.sh`** — polls `/v1/models` until
  ready, fires the deterministic "capital of France" completion
  twice, asserts byte-identical responses + that the text contains
  "Paris". Has a self-test at `scripts/test_smoke_check_harness.sh`
  (uses `scripts/_mock_openai_server.py` — no TPU needed).
* **`full_slice_v4_warm_cache.sh`** — runs the smoke + check, then
  cleans up. Use once on a fresh VM (or after a `/tmp` wipe) to
  populate the JAX compile cache; subsequent real launches' first
  curl is sub-minute on cache hit.

Pass criterion: `full_slice_v4_smoke_check.sh` exits 0 with
`PASS: deterministic completion contains 'Paris'`.

## Killing vLLM cleanly

The `vllm serve` process forks a child EngineCore which spawns Ray
actors on each worker. SIGKILL'ing the api-server doesn't reap
them, and they continue to hold TPU + libtpu state. Always use:

```bash
scripts/full_slice_v4_reset.sh
```

That kills the api-server pid, kills `VLLM::EngineCore` on every
host (head + 7 workers, exact `comm` match — never a broad regex),
removes the libtpu lockfile, and frees leaked placement groups.

If that's not enough, escalate to `full_slice_v4_ray_restart.sh`.

## Optimization knobs

Set on the launching shell; the smoke script forwards them to all
Ray workers via `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`. The smoke
launcher echoes the active values at startup so you can confirm.

| env var | default | what it does |
|---|---|---|
| `V4_LOADER_SLICE_AWARE` | `1` | Each host reads only the rows its local devices own (vs full-tensor read on every host). |
| `V4_LOADER_PLACE_WORKERS` | `8` | Threads driving `place_spec_as_jax_sharded` per host. Most per-tensor work releases the GIL (safetensors mmap reads + JAX C calls), so parallelism is real. Set to `1` for single-thread parity testing. |
| `V4_LOADER_PREFETCH_WORKERS` | `0` | Thread-pool prefetch in the non-slice-aware iterator. Empirically didn't help on real V4 (placement is the bottleneck, not dequant). Knob retained for future work. |
| `JAX_COMPILATION_CACHE_DIR` | `/tmp/jax-compile-cache-v4` | Local-disk persistent compile cache. Each host has its own; survives process restarts; lost if the worker host is rebuilt. **Not GCS** — the bucket the venv mounts is shared, do not write cache there without explicit user authorization. |
| `JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES` | `0` | Cache even small modules (default 1 MB skips them). |
| `JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS` | `0` | Cache even fast-to-compile modules. |
| `RAY_CGRAPH_get_timeout` | `3600` | Ray compiled-graph channel timeout. Default 300 trips during the first inference if `jit_run_model` recompiles for an unseen shape (already burned us once at 5m1s). Don't lower. |
| `V4_XLA_FLAGS` | unset | Opt-in custom `XLA_FLAGS` string for one launch. The smoke script does **not** inherit `XLA_FLAGS` from the parent shell (a stale autorunner env once SIGSEGV'd every Ray worker — see pitfall #4). |

## Known bloat / consolidation candidates

These are concrete pieces of accreted size that future cleanup
passes should target. The math + serve work without them; they're
just noise the next reader has to wade through.

* **`tests/models/jax/test_deepseek_v4.py` (3185 LOC, 33 classes).**
  5–10× the size of any peer model's test file
  (`test_qwen2_5_vl.py` is the next biggest at 764). Multiple
  pairs of duplicate-shape classes:
  `TestDecodeAttentionParity` + `TestDecodeAttentionParityExtended`,
  `TestDecodeRollingParity` + `TestDecodeRollingParityLong` +
  `TestDecodeRollingEquivalenceWithPrefill`,
  several FP8/FP4 dequant classes that overlap. Consolidate to a
  single parametrized class per concept. Don't drop coverage; do
  drop scaffolding.
* **`tests/models/test_deepseek_v4.py` (31 LOC, re-export shim).**
  Exists because a prior autonomous-task spec expected that path.
  Verify nothing in CI imports it, then delete.
* **`deepseek_v4.py` has 26 top-level entities vs `deepseek_v3.py`'s
  12.** Some is legitimate (MTP, hash routing, MLA variants), but
  worth scanning for helpers that could use upstream primitives.

Numbers above are snapshots; re-measure with `wc -l` and `grep -c "^class Test"`
when you touch.

## What's been optimized + verified

* **Streaming sharded loader** (no zero-tree OOM, places one tensor
  at a time onto a sharded global `jax.Array`). ✓
* **Slice-aware load**: each host reads only its row range from the
  safetensors mmap. ✓ Parity-verified on tiny fixture.
* **Multi-threaded placement** (`V4_LOADER_PLACE_WORKERS=8`). ✓
  Parity-verified on tiny fixture.
* **safetensors handle cache** (`_safe_open_cache`): eliminates
  per-tensor mmap+header reopen — observed ~6× load speedup
  (23 t/s → 140 t/s on real V4-Flash, ~4 min total load down from
  ~25 min). ✓
* **Vectorized MoE forward**: 256 expert kernels per layer collapsed
  into 3 einsums via stacked `[E, ...]` weights + one_hot routing
  weight. Mathematically identical to the per-expert loop
  (maxabs=0 across 5 seeds on the synthetic fixture). Drops
  `jit_run_model` HLO instruction count by orders of magnitude
  and cold-compile time with it. ✓
* **Persistent JAX compile cache**: wired up local-disk per host;
  populates after first successful `jit_run_model` finishes. ✓

## Pitfalls already learned (don't repeat)

1. **Broad pkill regex hits raylets.** Patterns like
   `pkill -f "EngineCore|RayWorkerWrapper|vllm"` match strings in
   raylet's own command line on remote workers and kill the daemon
   — losing 7/8 nodes. Always use narrow comm-name match:
   `pkill -x VLLM::EngineCore`, or kill by exact pid.

2. **`/tmp/libtpu_lockfile` survives SIGKILL.** A killed EngineCore
   leaves the libtpu lockfile behind. Subsequent inits SIGSEGV on
   a "clean" start because libtpu sees the lockfile and bails. The
   reset script handles this — but if you've killed something
   manually, also remove the lockfile manually before relaunching.

3. **`git push` doesn't sync workers.** Each worker has its own
   clone; only the head sees pushes. We lost a full 30-minute load
   cycle running stale code on 7/8 hosts. **Always run
   `scripts/full_slice_v4_sync.sh` after any code edit** before
   launching the smoke.

4. **Don't add unverified XLA flags.**
   `--xla_tpu_impure_hlo_parallel_compile=true` looked plausible
   (it appears in deepsea_compiler logs as an internal config
   option) but is **not** a recognized XLA flag in this libtpu
   build, and putting it in `XLA_FLAGS` makes every Ray worker
   FATAL on init. Validate any addition with a quick
   `python -c "import jax; jax.devices()"` under a candidate
   `XLA_FLAGS` value before wiring it into the launcher. The smoke
   script ignores the parent shell's `XLA_FLAGS` for this reason
   — use `V4_XLA_FLAGS=...` to opt in.

5. **First inference is slow on a fresh launch.** The first
   `/v1/completions` call triggers compilation of `jit_run_model`
   (the V4 forward pass). The vectorized MoE got the cold compile
   from 30+ min down significantly; expect 5–15 min on a cold
   cache, sub-minute on a warm cache. Don't use a 60s curl
   timeout — the smoke check defaults to 900s.

6. **`--enforce-eager` does not skip XLA compile.** That flag only
   affects vLLM's CUDA-graph-equivalent path. The TPU forward is
   JAX/`tpu-inference` and ALWAYS jit-compiles via XLA.

7. **vLLM's `capture_model` can multiply compile cost.** Without
   `--enforce-eager`, vLLM precompiles many shape buckets up front
   — for V4-Flash that's many × the single-shape compile time.
   `--enforce-eager` (already in the smoke launcher) skips that
   pre-compile and lets the first request pay the single-shape
   compile cost lazily.

## Layout

* `work/tpu-inference/` — JAX V4 implementation. Git subtree of the
  upstream `tpu-inference` repo. The DeepSeek V4 model lives at
  `work/tpu-inference/tpu_inference/models/jax/deepseek_v4*.py`;
  the MoE math at
  `work/tpu-inference/tpu_inference/layers/jax/moe/deepseek_v4_moe.py`;
  attention at
  `work/tpu-inference/tpu_inference/layers/jax/attention/deepseek_v4_attention.py`.
* `work/vllm/` — vLLM source tree. Don't edit upstream files unless
  you've read `work/vllm/AGENTS.md` (it forbids ad-hoc PRs).
* `scripts/` — operational helpers; per-host entry points all start
  with `full_slice_v4_`.
* `logs/` — `.gitignore`d; smoke logs accumulate here.
* `README.md` — fresh-VM bringup (one-shot via `./run.sh`).
* `.env.example` — every env var documented.

## Sanity check on a fresh VM

After `cp .env.example .env` + filling in tokens + `./run.sh`:

```bash
# 1. CPU-only math test (no TPU needed; ~10s)
work/vllm_env/bin/python3 -m pytest \
    work/tpu-inference/tests/models/jax/test_deepseek_v4.py::TestFp8DequantIndependentReference \
    work/tpu-inference/tests/models/jax/test_deepseek_v4.py::TestFp4DequantIndependentReference \
    -x -q

# 2. Smoke harness self-test (no TPU; mocks vllm)
scripts/test_smoke_check_harness.sh

# 3. Cluster: should be all green
scripts/full_slice_v4_reset.sh
ray status   # expect 8 nodes, 0.0/32.0 TPU

# 4. Real smoke
scripts/full_slice_v4_sync.sh
scripts/full_slice_v4_smoke.sh
tail -f logs/full-slice-v4-smoke-*.log

# 5. Validate when ready
scripts/full_slice_v4_smoke_check.sh   # PASS = "Paris"
```

Update this file as you learn more.

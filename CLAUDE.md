# claude-deepseek-v4 — agent runbook

You're picking up a TPU-inference effort: get `vllm serve deepseek-ai/DeepSeek-V4-Flash` running end-to-end on a v6e-32 TPU slice (TP=32). The model is a flagship MoE (256 FP4 experts + MLA attention + dense FP8) — about 543 GB bf16-expanded.

This file documents the **operational** knowledge that's not obvious from the code: cluster layout, the iterate loop, the orphan-state surfaces that bite a relaunch, env knobs, and pitfalls that have already cost real time. Read it once before doing anything; everything below has been learned by burning iterations.

## Cluster topology

* Slice: `v6e-32` = 8 hosts × 4 chips × 32 GB HBM = 992 GiB total.
* Head: `10.164.0.41` (this is also `worker 0` and where you launch from).
* Workers: `10.164.0.{22,35,36,39,45,18,30}` (worker 1–7).
* Each host is a separate VM with its own clone of `~/claude-deepseek-v4`.
  **There is no shared filesystem between hosts.** A `git push` from the head does not propagate to workers. See "Source sync" below.
* Ray address: `10.164.0.41:6379`. Bootstrapped from `scripts/full_slice_v4_ray_restart.sh`. `ray status` should show 8 nodes and `0.0/32.0 TPU` when idle.
* The shared venv path is `~/claude-deepseek-v4/work/vllm_env` on every host (mirrored once at bootstrap).
* SSH between hosts uses `~/.ssh/google_compute_engine` as the key (the GCE-provisioned identity); the regular `id_ed25519` is for GitHub.

## The iterate loop

Every change-then-test cycle is exactly:

```bash
scripts/full_slice_v4_reset.sh        # cluster-state cleanup; safe to re-run
scripts/full_slice_v4_sync.sh         # rsync source to all 7 worker hosts
scripts/full_slice_v4_smoke.sh        # launch vllm serve; writes pid + log
scripts/full_slice_v4_smoke_check.sh  # validate /v1/completions when ready
```

* `full_slice_v4_reset.sh` handles the four orphan-state surfaces (api-server pid, `VLLM::EngineCore` actor children, `/tmp/libtpu_lockfile`, leaked Ray PGs). It does **not** restart Ray itself.
* `full_slice_v4_ray_restart.sh` is the heavier nuke (`ray stop --force` + fresh `ray start` on all 8 hosts). Reach for it only when reset isn't enough.
* `full_slice_v4_sync.sh` is **mandatory after any code edit**. The whole worker pool will silently run stale code otherwise — see "Pitfalls" below.
* `full_slice_v4_smoke.sh` writes a pid to `logs/full-slice-v4-smoke.pid` and a timestamped log to `logs/full-slice-v4-smoke-<TS>.log`. The reset script reads that pid file to kill cleanly.
* `full_slice_v4_smoke_check.sh` polls `/v1/models` until ready, then fires the deterministic "capital of France" completion twice and asserts byte-identicalness + that the text contains "Paris". It has its own self-test at `scripts/test_smoke_check_harness.sh` (uses `scripts/_mock_openai_server.py` — no TPU needed).

## "How do I fully kill vLLM?"

The vllm serve process forks a child EngineCore which spawns Ray actors on each worker. SIGKILL'ing the api-server doesn't reap them, and they continue to hold TPU + libtpu state. Use:

```bash
scripts/full_slice_v4_reset.sh
```

That kills the api-server pid, kills `VLLM::EngineCore` on every host (head + 7 workers, exact `comm` match — never a broad regex), removes the libtpu lockfile, and frees leaked placement groups.

If you ever hit `ABORTED: TPU is already in use by process with pid X` even after reset, escalate to `scripts/full_slice_v4_ray_restart.sh`.

## Optimization knobs (set on the launching shell; forwarded to Ray workers)

| env var | default | what it does |
|---|---|---|
| `V4_LOADER_SLICE_AWARE` | `1` | Each host reads only the rows its local devices own (vs full-tensor read on every host). |
| `V4_LOADER_PLACE_WORKERS` | `8` | Threads driving `place_spec_as_jax_sharded` per host. Most per-tensor work releases the GIL (safetensors mmap reads + JAX C calls), so parallelism is real. Set to `1` for single-thread parity testing. |
| `V4_LOADER_PREFETCH_WORKERS` | `0` | Thread-pool prefetch inside the non-slice-aware iterator. Empirically didn't help — left as a knob for future work. |
| `JAX_COMPILATION_CACHE_DIR` | `/tmp/jax-compile-cache-v4` | Local-disk persistent compile cache. Each host has its own; survives restarts. **Not GCS** — the bucket the venv mounts is shared, do not write cache there without explicit user authorization. |
| `JAX_COMPILATION_CACHE_MIN_*` | `0` | Cache even small / fast-to-compile modules. |
| `RAY_CGRAPH_get_timeout` | `3600` | Ray compiled-graph channel timeout. Default 300 trips the first time `jit_run_model` compiles for an unseen shape. Don't lower. |

The smoke launcher echoes the active values at startup so you can confirm they took effect.

## Pitfalls already learned (don't repeat)

1. **Broad pkill regex hits raylets.** Patterns like `pkill -f "EngineCore|RayWorkerWrapper|vllm"` match strings in raylet's own command line on remote workers and kill the daemon — losing 7/8 nodes. Always use narrow comm-name match: `pkill -x VLLM::EngineCore` or kill by exact pid.

2. **`/tmp/libtpu_lockfile` survives SIGKILL.** A killed EngineCore leaves the libtpu lockfile behind. Subsequent inits SIGSEGV on a "clean" start because libtpu sees the lockfile and bails. The reset script handles this.

3. **`git push` doesn't sync workers.** Each worker has its own clone; only the head sees pushes. We lost a full 30-minute load cycle running stale code on 7/8 hosts. **Always run `scripts/full_slice_v4_sync.sh` after any code edit** before launching the smoke.

4. **Don't add unverified XLA flags.** `--xla_tpu_impure_hlo_parallel_compile=true` looked plausible (it appears in deepsea_compiler logs as an internal config option) but is **not** a recognized XLA flag in this libtpu build, and putting it in `XLA_FLAGS` makes every Ray worker FATAL on init. Validate any addition with a quick `python -c "import jax; jax.devices()"` under a candidate `XLA_FLAGS` value before wiring it into the launcher.

5. **First inference is slow on a fresh launch.** The first `/v1/completions` call triggers compilation of `jit_run_model` (~5-10 min for the 477K-instruction V4 forward pass). Don't use a 60s curl timeout — use 900s. The smoke check defaults to that.

6. **`--enforce-eager` does not skip XLA compile.** That flag only affects vLLM's CUDA-graph-equivalent path. The TPU forward is JAX/`tpu-inference` and ALWAYS jit-compiles via XLA.

7. **vLLM's profile_run uses different shapes than real inference.** Even with persistent cache, the *first launch ever* will compile twice (warmup shape + actual shape). Subsequent launches hit the cache.

## Layout

* `work/tpu-inference/` — git subtree of `tpu-inference` (JAX V4 impl). The DeepSeek V4 model lives at `work/tpu-inference/tpu_inference/models/jax/deepseek_v4*.py`.
* `work/vllm/` — vLLM source tree. Don't edit upstream files unless you've read `work/vllm/AGENTS.md` (it forbids ad-hoc PRs).
* `scripts/` — operational helpers. Per-host entry points all start with `full_slice_v4_`.
* `memory/` — persisted findings from prior sessions: `feedback_ray_cleanup.md`, `v4_deploy_session.md`, `project_v4_status.md`. Skim `MEMORY.md` (the index) when picking up.
* `CODEX_PLAN.md` — handoff plan from a prior Codex session; partially superseded by this file but still useful for the load-time / kv-cache-budget analysis.
* `logs/` — `.gitignore`d; smoke logs accumulate here.

## Sanity check on a fresh VM

After cloning + bootstrapping the venv:

```bash
# 1. CPU tests (no TPU needed; ~2 min)
work/vllm_env/bin/python3 -m pytest \
    work/tpu-inference/tests/models/jax/test_deepseek_v4.py -x -q

# 2. Smoke harness self-test (no TPU needed)
scripts/test_smoke_check_harness.sh

# 3. Cluster: should be all green
scripts/full_slice_v4_reset.sh
ray status   # expect 8 nodes, 0.0/32.0 TPU

# 4. Real smoke
scripts/full_slice_v4_sync.sh
scripts/full_slice_v4_smoke.sh
tail -f logs/full-slice-v4-smoke-*.log
```

Pass criterion: `scripts/full_slice_v4_smoke_check.sh` exits 0 with `PASS: deterministic completion contains 'Paris'`.

## Status / what's been verified

- Streaming sharded loader (no zero-tree OOM): ✓ committed.
- Slice-aware load (per-host row-range read): ✓ committed + parity-verified on tiny fixture + 4-device CPU mesh.
- Multi-threaded placement (V4_LOADER_PLACE_WORKERS=8): ✓ committed + parity-verified.
- safetensors handle cache (`_safe_open_cache`): ✓ committed. Eliminates per-tensor mmap+header reopen — observed ~6× load speedup (23 t/s → 140+ t/s on real V4-Flash, ~4 min total load down from ~25 min).
- Persistent local JAX compile cache: ✓ wired up; populates after the first jit_run_model finishes.
- Smoke-check harness self-test: ✓ 4/4 scenarios pass.
- End-to-end `Application startup complete` + first request: ⚠️ in flight — the load and compile have been verified working; the first request response is still being shaken out under the new optimizations.

Update this file as you learn more.

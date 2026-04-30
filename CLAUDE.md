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

## Current state (READ BEFORE LAUNCHING)

**Tier 8 deploy gate is GREEN as of 2026-04-30 04:22Z.** A cold
`./run.sh serve` against real V4-Flash weights now reaches
`Application startup complete` and answers
`/v1/completions` with deterministic `Paris` for "The capital of
France is" — `scripts/full_slice_v4_smoke_check.sh` exits 0.

End-to-end timing on the verifying run (cache-cold):
  * weight load + inline MoE consolidation: 4 min 49 s
  * `Application startup complete`: ~10 s after load
  * first `/v1/completions` cold compile (two shape buckets,
    prefill + decode, no cache): 49 s + 47 s ≈ **96 s total**
  * deterministic curl response: byte-identical across two repeats

`jit_run_model` HLO size dropped from **~103k instructions → 47k
optimized** (~2.7× smaller). XLA accounting was clean — no
`CompileTimeHbmOom`, no `RuntimeBufferAllocationFailure`.

What landed to unblock it:
  1. **Freqs cap by `max_model_len`**
     (`deepseek_v4.py::_effective_freqs_seq_len`). V4-Flash's
     `max_position_embeddings = 1 048 576` was producing a
     `f32[1M, 32]` freqs_compressed table that XLA pinned as a
     1 GB argument per chip. Capping at `max_model_len=256`
     shrinks it to KB. -1 GB / chip resident.
  2. **Inline MoE consolidation**
     (`deepseek_v4.py::_maybe_consolidate`). As soon as a
     `(layer, wname)` group's 256 expert weights are placed, the
     loader stacks them into a single `[E, inter, dim]` jax.Array
     sharded `P('attn_dp', None, None)` and nulls out the per-leaf
     references. Eliminates the 126 × 128 MiB MoE all-to-all
     buffers that dominated `BACKEND_PASSES` HLO-temp on the
     previous OOM (~16 GB / chip saved). Doing this incrementally
     — not as a post-load pass — is critical: post-load, HBM is
     fragmented across 33 000+ small allocations and the
     consolidate's 256 MB transient OOMs (we burned an iter
     learning that). Doing it as soon as a group is full means we
     only ever hold one group's per-leaf set alongside its stacked
     tensor at a time.
  3. **`MoEParams` carries optional `w1_stacked / w2_stacked /
     w3_stacked` fields**
     (`deepseek_v4_moe.py`). `moe_forward` branches on
     `params.w1_stacked is not None` — when present, reads the
     stacked tensors directly, skipping the per-call
     `jnp.stack(experts[*].wN)` that previously forced the
     all-to-all on every forward × layer × stack. The per-expert
     fallback path stays intact for synthetic-fixture tests.

### What's now possible / what's still loose

Ground-truth latency is sub-100 s cold compile + sub-second
execute; on cache-warm restarts (post-bootstrap) the compile-cache
should let first-curl finish in seconds.

Still loose ends worth tracking:
  * **Activation budget headroom is unmeasured.** We compiled
    cleanly under `max-model-len=256, max-num-seqs=1`. Bumping
    either knob raises HLO temp roughly linearly; we don't yet
    know how far we can push before a new OOM. Iter that lifts
    the cap should re-run the smoke.
  * **Cross-host JAX cache sharing (lane 2 from the original
    plan)** is still untouched. Each host compiles independently;
    a `scripts/full_slice_v4_share_cache.sh` rsync'ing host 0's
    cache after one good launch would 8× brand-new-slice
    first-launch on a freshly bootstrapped slice.
  * **`Involuntary full rematerialization` warnings** still appear
    in the compile log on `f32[128, 16]` reshards (`{devices=[1,32]}
    → {devices=[16,1,2]} last_tile_dim_replicate`). They didn't
    block the compile this iter, but each one wastes activation
    budget; a future iter should pin a `with_sharding_constraint`
    upstream so XLA doesn't have to do the inefficient
    replicate-then-partition.

## Iteration discipline (READ — applies to humans + agents alike)

**Do NOT use `./run.sh serve` as your inner test loop.** Each attempt
is 25–45 min (4 min load + 10–30 min cold compile + curl wait). That
budget is fixed by XLA, not by anything we can shorten in a single
iteration. Prior sessions burned real time treating it as if it
should be fast. Use the fastest validation that catches the bug
class you're working on:

1. **Standalone math scripts** under `/tmp/` (~10–30s) — example
   pattern: `/tmp/test_moe_vectorize.py` validated the vectorized
   MoE math vs the per-expert reference on 5 seeds in ~10s.
2. **Tiny-fixture pytest classes** in
   `tests/models/jax/test_deepseek_v4.py` (~30s–2min on CPU).
3. **`eval_shape` / `lower().compile()` on the real config**
   (~1–3min). Catches sharding bugs + HLO-emit failures (like
   the current HBM OOM!) without paying the runtime compile cost.
   The agent has used `XLA_FLAGS=--xla_force_host_platform_device_count=32`
   + `JAX_PLATFORMS=cpu` to compile against a virtual mesh — that
   pattern works and surfaces all-gather sizes from HLO inspection
   in seconds.
4. **Real `./run.sh serve`** only when 1–3 are green. Budget at
   most 1–2 of these per session.

### Real-smoke phase budgets (don't bail too early!)

When you have to run the real smoke (path #4), each phase has a
*known* duration. Silence during a phase is normal as long as it's
the right kind of silence. Use these to decide if something's
genuinely stuck vs. just paying the cost:

| Phase | Expected duration | What you should see | Bail signal |
|---|---|---|---|
| **vLLM startup + Ray cluster init** | ~30s | `Init mesh \| mesh=Mesh(...)`, `Init kv-cache`, route registration | No log activity for >2 min, OR `Worker exit type: SYSTEM_ERROR`. |
| **Weight load** | ~4 min | `[deepseek_v4] placed N tensors (R/s, ...)` heartbeat every ~7s, then `load_weights_from_dir done` | No heartbeat for >2 min, OR `placed N` count stops growing. |
| **`capture_model` precompile** | ~30s | A handful of small `running hlo passes for N instructions, module: jit_*` lines (`jit__threefry_seed`, `jit__allocate`, `jit_iota`, `jit_unpack_arrays`, etc.), each tiny | Any `RESOURCE_EXHAUSTED` / `CompileTimeHbmOom`. |
| **`Application startup complete`** | fires immediately after capture_model | Single line | If absent >2 min after capture_model finishes. |
| **`jit_run_model` cold compile** | **10–30 min** | One `running hlo passes for ~100k instructions, module: jit_run_model`, then **long silence punctuated by `HLO PostOptimizationPipeline` lines and SPMD warnings**. The silence is normal — XLA's late codegen passes don't emit progress. | Three or more separate `slow_operation_alarm.cc` warnings (each fires after a single pass exceeds 5 min). One alarm = one slow pass; that alone is *not* enough to bail. Also: any `RESOURCE_EXHAUSTED` / `Worker exit`. |
| **First curl returning** | sub-second after compile finishes | `INFO 127.0.0.1:... "POST /v1/completions" 200 OK` and the `[smoke-check] response 1: ...` line | Curl 900s timeout fires, OR the engine crashes mid-execute. |

**Rule of thumb during real smoke:** silence in the `jit_run_model`
phase ≤ ~25 min is *expected*, not stuck. **Don't bail before 25
min unless the iter timeout is closing in.** The 90-min
ITER_TIMEOUT_SEC has plenty of slack for one full smoke + one bail.

**Concurrent work while compile runs:** the compile is going to
take 10–30 min no matter what you do. Spend that time productively
— don't just sit in a Monitor. Good uses of the wait window:

* Sketch the next-lane fix (lane 2 cache-rsync helper, lane 3
  SPMD remat-warning audit) in a `/tmp/` standalone test so it's
  ready to ship the moment the current smoke confirms.
* Audit the `Involuntary full rematerialization` warnings
  accumulating in the smoke log — each one points at a
  resharding inefficiency you can fix on a future iter.
* Consolidate test bloat (CLAUDE.md "Known bloat" list) — test
  edits don't conflict with the running smoke.

**Quick-test rule (still applies for code edits, NOT for smoke):**
if a CPU pytest / `lower().compile()` probe takes >5 min without a
useful signal, kill it and rethink — that *is* stuck.

### Iter-timeout management

`ITER_TIMEOUT_SEC=5400` (90 min). If you're approaching the deadline
without a result:

1. **At T-15 min:** stop launching new long-running steps. Commit
   whatever code change you've made so far (with a "WIP:" prefix
   describing what was tried + what's still unverified) so iter N+1
   can pick up from the same on-disk state.
2. **At T-5 min:** reset the cluster + push the WIP commit. Don't
   risk the iter being killed mid-`./run.sh serve`.

Better to have a checkpointed WIP commit than to lose the diff
when the timeout SIGTERMs the iter.

### Next attack lanes (in rough ROI order)

The original "stack-then-all-gather" symptom is fixed. The remaining
problem is the per-chip memory budget itself. In ROI order:

1. **Audit the 28 GB of "arguments" per chip.** Expected ~17 GB
   (sharded weights). The extra ~11 GB is the bulk of the OOM gap.
   - Add a one-shot dump after `load_weights_from_dir done` that
     iterates the assembled param tree, computes per-leaf
     `numel * dtype_bytes / num_local_devices` (i.e. per-chip
     resident bytes), and prints the top 20 leaves by size. Look for
     anything large with `spec=P()` (fully replicated) or with a
     sharding axis other than `attn_dp` (mesh has size 1 on
     non-`attn_dp` axes, so non-`attn_dp` shardings are also
     effectively replicated).
   - Particular suspects: `gate.weight` (`[E, dim] = [256, 4096]
     fp32 = 4 MB` — small, fine), `tid2eid` (`[vocab, top_k] =
     [129280, 6] int32 = 3 MB` — small, fine), but the per-layer
     `hc_attn_fn` / `hc_ffn_fn` matrices are `[mix_hc, hc_dim] =
     [(2 + hc_mult) * hc_mult, hc_mult * hidden_size]` — if
     `hc_mult` is large these can be 100+ MB each, ×30 layers ×
     2 = several GB if replicated.
   - Embedding `[vocab, dim] = [129280, 4096] bf16 = 1 GB` and head
     of the same size — if either is replicated that's 2 GB.

2. **Verify the lane-1 follow-on (`_shard_e_mid` on einsum
   intermediates) actually helps on TPU.** Already in tree, CPU
   compile-only is green, but the TPU smoke that observed the
   BACKEND_PASSES OOM was *before* this commit — the next smoke
   tests it. If it shrinks the activation budget far enough to fit,
   we're done; if not, fall through to lane 1.

3. **Fix `Involuntary full rematerialization` warnings.** These
   spam the compile log and indicate XLA can't satisfy a sharding
   spec without a full replicate-then-repartition. Each one
   wastes some of the 11.64 GB activation budget.
   Pattern from the run:
   `cannot go from sharding {devices=[1,32]<=[32]} to {devices=[16,1,2]<=[2,16]T(1,0)}`
   for `f32[128,16]` tensors at param indices stepping by 1592
   (likely the per-layer attention top_k partials, given
   `num_experts_per_tok=6` doesn't match 16 — investigate). Add
   `with_sharding_constraint` upstream so XLA doesn't have to
   reshard.

4. **Pre-stack the experts at load time** in `deepseek_v4_loader.py`.
   The current code stacks 256 separate per-leaf jax.Arrays inside
   the JIT every forward call, which forces an all-to-all to honor
   the `_shard_e_first` constraint. Doing the stack at load time —
   so the param tree carries an `[E, inter, dim]` tensor with
   PartitionSpec `(attn_dp, None, None)` from the start — eliminates
   the per-call all-to-all *and* the per-call stack op. Bigger
   change (touches `MoEParams` dataclass + the loader's per-leaf
   placement loop) but eliminates a structural inefficiency.

5. **Verify cross-host JAX cache sharing.** Each of 8 hosts
   compiles its own SPMD slice; `/tmp/jax-compile-cache-v4` is
   per-host. After one successful compile, fingerprint the cache
   files across hosts — if byte-equal, a tiny
   `scripts/full_slice_v4_share_cache.sh` rsyncs host 0's cache
   to workers and brand-new-slice first-launch is 8× faster.
   Uncertain whether SPMD bakes in per-device IDs; verify before
   shipping.

6. **AOT precompile + binary persist.** `jit().lower().compile()`
   serialized + loaded on subsequent launches. Real XLA-versioning
   risk; defer until after lanes 1–5 are exhausted.

Validation after any change:
```bash
JAX_PLATFORMS=cpu work/vllm_env/bin/python3 -m pytest \
    work/tpu-inference/tests/models/jax/test_deepseek_v4.py::TestMoEComponent -x -q
work/vllm_env/bin/python3 /tmp/test_moe_compile.py     # if MoE-related
scripts/full_slice_v4_reset.sh
scripts/full_slice_v4_sync.sh
scripts/full_slice_v4_smoke.sh
scripts/full_slice_v4_smoke_check.sh   # PASS = "Paris"
```

## What's been optimized + verified (load path only)

* **Streaming sharded loader** (no zero-tree OOM). ✓
* **Slice-aware load**: each host reads only its row range. ✓
  Parity-verified on tiny fixture.
* **Multi-threaded placement** (`V4_LOADER_PLACE_WORKERS=8`). ✓
  Parity-verified on tiny fixture.
* **safetensors handle cache** (`_safe_open_cache`): eliminates
  per-tensor mmap+header reopen — observed ~6× load speedup
  (23 t/s → 140 t/s on real V4-Flash, ~4 min total load down from
  ~25 min). ✓
* **Vectorized MoE forward**: math byte-equivalent to the per-expert
  reference loop (maxabs=0 across 5 seeds on synthetic fixture);
  HLO instruction count drops 4.6× (477k → 103k). ✓ correctness.
* **MoE stacked-weight sharding constraint**
  (`_shard_e_first` / `_shard_e_last` / `_shard_e_mid`): forces
  W1/W2/W3 to be E-sharded across `attn_dp`, eliminating the
  original 4 GiB all-gather per stack. Now mostly superseded by
  inline consolidation (constraints are still applied as defense
  in depth on the per-expert fallback path).
* **Inline MoE consolidation at load**
  (`deepseek_v4.py::_maybe_consolidate`): the 256 per-expert
  weights of each `(layer, wname)` group are stacked into a single
  E-sharded `[E, inter, dim]` jax.Array as soon as the 256th is
  placed; per-leaf references are then nulled. Drops the per-call
  all-to-all storm entirely — `jit_run_model` HLO instructions
  47k optimized vs 103k previously. ✓ Smoke green 2026-04-30.
* **Freqs cap by `max_model_len`**: `_effective_freqs_seq_len()`
  uses `vllm_config.model_config.max_model_len` instead of
  `cfg.max_position_embeddings`, shrinking the YaRN freqs table
  from 1 GB / chip to KB. ✓ Smoke green 2026-04-30.
* **Persistent JAX compile cache**: wired; populated by the
  2026-04-30 successful smoke. Subsequent launches on the same
  worker host should hit the cache and skip the ~96 s compile.
  Lane 2 (`scripts/full_slice_v4_share_cache.sh` to rsync host 0's
  cache to other 7 hosts) is the next compile-time win.

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
   (the V4 forward pass, ~103k HLO instructions post-MoE-vectorize).
   Expect 5–15 min on a cold compile cache, ~30–60s on a warm
   cache. Don't use a 60s curl timeout — the smoke check defaults
   to 900s.

   To warm the cache at bootstrap time (one-time +10-15 min cost,
   then every subsequent first-curl is sub-minute), set
   `WARM_CACHE_ON_BOOTSTRAP=1` in `.env` before `./run.sh bootstrap`.
   `scripts/full_slice_v4_warm_cache.sh` is the underlying helper.

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

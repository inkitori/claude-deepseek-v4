# claude-deepseek-v4 — PERFORMANCE runbook

> **Phase = PERFORMANCE. START HERE: `HANDOFF_PERF.md`** (current state + the verified profile
> + THE ROADMAP + the ONE next action). This file holds durable slice ops (how to run,
> validate, pitfalls). S1 (decode *correctness*/determinism) is **CLOSED** — it is now a HARD
> REGRESSION GATE, not the goal (see below). S1 history: `HANDOFF_S1.md` / `CLAUDE.full.md`.
> Per-iteration narrative goes in **commit messages**, not this file.
>
> **One-line status (2026-05-27):** **Phase 1 CLOSED & GATED** — the fused sparse-attn kernel
> (`kernels/sparse_attn/kernel.py`, wired via `_sparse_attn_kernel_sharded`) works: a re-profile shows
> the attention KV gather went 65.8%→**0.2%** of decode (it was THE bottleneck). Decode is now
> **ALL-REDUCE-bound (31.7%)**, then MoE 25% + indexer top_k 23%. **Next = Phase 2: flip
> `pick_partition_spec` to OUTPUT-dim (axis-0) sharding** (NOT axis-1 — the old hypothesis was
> backwards; the corrected diff is CPU-shape-validated + ready for a cold smoke). Decode breakdown +
> the corrected diff + NEXT ACTION in `HANDOFF_PERF.md`.

## Goal

Cut prefill + decode wall-time for `vllm serve deepseek-ai/DeepSeek-V4-Flash` on the v6e-32
TPU slice (TP=32, 8 hosts × 4 chips), driving the `HANDOFF_PERF.md` roadmap top-down. Shrink
the diff vs upstream `tpu-inference` and de-hack as you go. **The bottleneck is the attention
KV gather, NOT the MoE** (a FLOP count said MoE; the profile overturned it — always profile).

## S1 REGRESSION GATE — NON-NEGOTIABLE for every change

Every committed change MUST still pass, verified on a fresh real-V4 engine:
* `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (visible_words ≥ 10,
  max_word_run < 5).
* FIB decode: **correct Fibonacci** (21, 34, 55, 89, 144 — DETERMINISTIC/high-margin) +
  **N=2 md5 `5bf42256` byte-identical across 2 fresh engines** (`/tmp/s1_probe2.py 2`, =
  md5("21,")). ⚠️ Do NOT gate on a long-tail md5 (`s1_probe2.py 20`+): PERF-0.1 found the FIB
  free-form TAIL is NON-deterministic at temp=0 (flips WITHIN one process,
  `e4d45024`↔`26354502`) — pre-existing DECODE-path runtime nondeterminism (distributed
  all-reduce ordering), so old long-tail refs (`b675be27`) were sampling a nondeterministic
  quantity. (Baseline-confirm TODO: HANDOFF_PERF §0.1-DONE.)
* Still passes after 5 unrelated requests.

A numerics-changing fix MAY shift the md5 — then re-establish a NEW reference and confirm it's
identical across 2 fresh engines + correct Fibonacci. The non-negotiable is **identical across
engines + correct Fibonacci**, not the specific hash. The basic "PASS: contains 'Paris'" line
is a **false positive** (`capital of France` can EOS at token 1, so no decode runs) — and
`completion_tokens`/`max_word_run`/`ends_clean` all read healthy on corrupted output. **READ
the actual response text** AND compare 2 fresh engines. (Why this is sacred: the S1 bug was a
per-process uninit-HBM coin flip — coherent-looking output is NOT proof. Detail in `HANDOFF_S1.md`.)

## How to validate (CHEAPEST signal first — this is the heart of the perf phase)

The full smoke is the expensive thing (543 GiB load + 25-45 min cold compile). Reserve it.
Escalate only as far up as the question needs:

1. **CPU numerics (no slice, cheap):** the torch oracle. `PYTHONPATH=work/tpu-inference:work/vllm
   work/vllm_env/bin/python3 scripts/s1_cpu_repro_v4flash.py both` → "OK: both eager and jit
   match" (regression-only). ⚠️ **BROKEN since Phase 1 landed:** the wired Mosaic kernel is
   interpret-only on CPU, so the full-model eager path raises "Only interpret mode is supported on
   CPU backend" — `s1_cpu_repro` no longer runs the full model. Fix = make the `mesh.empty` branch
   of `_sparse_attn_kernel_sharded` call pure-JAX `sparse_attn`. For attention math the parity test
   `tests/models/jax/test_deepseek_v4.py -k sparse_attn` vs `sparse_attn_torch` still works (interpret).
   CPU CANNOT reproduce S1 (no sharding) — proves math, never proves a determinism fix.
2. **TPU MICRO-BENCHMARK (cheap slice, NO 543 GiB load):** jit + time a kernel/op in isolation
   on the real mesh with SYNTHETIC inputs — measures gather ms / bandwidth / kernel speedup in
   ~1 min vs a 25-45 min smoke. **BUILD this (Phase 0.0) if `scripts/perf_*bench*` doesn't
   exist yet** — it unblocks cheap iteration for the whole kernel campaign.
3. **Profiler re-capture (full smoke + profiler):** the structural op-breakdown truth.
   Recipe in `HANDOFF_PERF.md`. Reserve.
4. **Full smoke + the S1 GATE above:** the per-change closure gate; at most 1-2 per session.

## Slice-serving protocol (marginal slice — do this EVERY smoke)

1. Edit code → `scripts/full_slice_v4_sync.sh` (MANDATORY: 8 hosts, each own clone; `git push`
   does NOT sync them). Verify md5 of edited .py matches head==workers — a mismatch causes
   "different launch id" Core-halts (CODE DESYNC, **not** a slice wedge; do NOT reboot — sync
   fixes it).
2. Clear `~/.cache/vllm/xla_cache/*` on all 8 hosts (stale/mixed cache also → launch-id halt).
3. `scripts/full_slice_v4_reset.sh` (stops any engine, cleans lockfiles).
4. `scripts/full_slice_v4_smoke.sh` (backgrounds vllm serve; prints log path). Self-guards:
   flock single-instance (REFUSES if an engine is up/starting — `SMOKE_NO_GUARD=1` escapes) +
   `VLLM_ENGINE_READY_TIMEOUT_S=2400` so a cold compile doesn't die at vllm's 600s default.
   Bigger config: `MAX_LEN=<n> bash scripts/full_slice_v4_smoke.sh`.
5. Wait for `Application startup complete` (~6 min when xla cache warm; cold compile 10-30
   min; first decode request recompiles ~325s unless cache warm). Then probe with curls; fire
   critical probes first (engine can crash on internal NaN after a few requests; the
   `compute_logits` nan_to_num clamp keeps it alive). Init is a coin-flip (intermittent worker
   SYSTEM_ERROR) — just retry. Probe helper: `/tmp/s1_probe2.py N` (FIB decode, md5 + text).

Slice is HEALTHY when code is synced. Keep both guardians alive before TPU work:
`ps -eo pid,cmd | grep -E 'node_guard[i]an|meta_guard[i]an'` (restart node_guardian per the
loop prompt if dead — never `pkill` a pattern your own command line contains).

## Plumbing (read before touching — perf priority order)

* `layers/jax/attention/deepseek_v4_attention.py` — **the bottleneck.** `sparse_attn` (:160,
  gather :186, fp32 cast :181) + call sites (decode :812, prefill :905); the indexer
  (`indexer_prefill` :366 / `indexer_decode_step` :562, the `lax.top_k` `while` loop);
  compressor; seed-from-prefill. Attention is HEALTHY/correct — the work is making it fast.
* `models/jax/deepseek_v4.py` — `deepseek_v4_run_with_decode_state` (decode entry);
  `transformer_body_forward` (:851) vs `transformer_body_init_state_to_buffer` (:854) = the
  DUPLICATE prefill body (Phase 0.1); `block_forward`/`block_decode_step`; `hc_pre`/`hc_post`;
  `compute_logits` (:2042, nan_to_num clamp :2055). Dead code: `_consolidate_moe_after_load`
  (:1776). The w1/w3 host-gather load (~:1492) IS the S1 fix — do not touch.
* `runner/tpu_runner.py::_prepare_inputs_dp` — `_v4_decode_replicate` (:1359): replicated
  decode activation (the S1 fix). Drives the ~17% collective cost (Phase 2) — but DO NOT
  remove; mind pitfall #5.
* `models/jax/deepseek_v4_loader.py::pick_partition_spec` (:497) — weight sharding heuristic;
  flipping contracting→output dim is the Phase-2 all-reduce win.
* `layers/jax/moe/deepseek_v4_moe.py::moe_forward` — dense all-256 decode path (:217) vs
  sharded `gmm_v2` prefill path (:233); the `use_shard_map` gate (:211) is the chat-wedge
  trigger (Phase 4.1). ~10% of decode — SECONDARY despite 97% of FLOPs.
* `models/common/model_loader.py` — `donate_argnums`, V4 `kv_cache_sharding=P()`, registry.
* Kernel templates: `kernels/flash_attention/kernel.py`, `kernels/mla/v2/kernel.py`. Oracle:
  `tests/models/jax/_deepseek_v4_reference/kernel_stubs.py:60` (`sparse_attn_torch`).
  Invariants: `work/tpu-inference/INVARIANTS.md`.

## Pitfalls (these cost real time)

0. **`different launch id` / `Core halted` / `SLICE_FAILURE` BEFORE startup = CODE DESYNC**,
   not a slice wedge. Run `full_slice_v4_sync.sh` + clear xla_cache; do NOT reboot. Keep
   env-var reads consistent across workers — env-gated module-level reads race across ray
   workers → divergent programs → launch-id halt (this is why S1 diagnostics were always-on).
   Infra: old `node` contention is SOLVED (`DenyUsers mark` + two guardians); don't re-fight it.
1. **Shut down only via `scripts/full_slice_v4_reset.sh`** — never broad `pkill -f` (kills
   raylet, loses nodes). Escalate to `full_slice_v4_ray_restart.sh` if reset fails.
2. **`/tmp/libtpu_lockfile` survives SIGKILL** → next init SIGSEGVs. reset.sh handles it.
3. **`--enforce-eager` does NOT skip XLA compile** — the TPU forward is JAX, always jits.
4. **No unverified XLA flags** — smoke.sh ignores parent-shell `XLA_FLAGS`; opt in via
   `V4_XLA_FLAGS` and validate with `python -c "import jax; jax.devices()"` first.
5. **`with_sharding_constraint(activation, P())` that GATHERS a size-1 decode token axis
   Core-halts the slice** (proven ~8×). A wsc on a POST-reduction `[N,dim]` quantity is safe.
   Any sharding change in Phase 2 must avoid gathering the decode token axis (do it at the
   jit input/output boundary or via load-time weight placement, never an in-trace token gather).
6. **A live engine WEDGES on a new request shape** (HTTP500 → connection-refused; process
   alive but stops serving), esp. the chat/longer-prefill path (Phase 4.1). Raw `/completions`
   SAME shape survived 3 fires. When probing: pick ONE shape, fire critical probes first,
   reset+re-smoke if wedged.

## Slice bootstrap

The slice (`v6spoteu721`, v6e-32, zone `europe-west4-a`, project `prm-research`) is already
bootstrapped: venv, GCS mount, ray up. Head = `10.164.0.192`; workers auto-discover via
`scripts/full_slice_v4_discover.sh`. ssh: `ssh enyouki@<ip> -i ~/.ssh/google_compute_engine`.
Weights load auth-free from the GCS mount (`HF_TOKEN` intentionally unset). venv python 3.12.
Fresh-VM setup: `./scripts/full_slice_v4_bootstrap.sh` + `CLAUDE.full.md`.

## ⇒ CONTEXT HANDOFF PROTOCOL — every session MUST follow

Sessions are disposable; the commit log + `HANDOFF_PERF.md` + this file are the only memory
that survives. When context reaches **~100k–200k tokens** (interactive: check `/context`; heed
"context getting long" reminders) AND you are NOT seriously mid-operation (waiting on a smoke
whose result you must read, or a few calls from finishing a committed step) — HAND OFF:

1. **Keep the markdown LEAN** — trim `CLAUDE.md` + `HANDOFF_PERF.md` (and memory files):
   delete superseded narrative, keep only durable ops + the CURRENT state + the next action.
   Every line rsyncs to 8 hosts / loads into every session. Lean > complete.
2. Rewrite `HANDOFF_PERF.md` to the CURRENT state: what you landed + gated, what's in flight
   (log paths, ports, microbench numbers), and the ONE most important next action.
3. `git add -A && git commit && git push`.
4. **HAND OFF THE WINDOW:** interactive in tmux → run `scripts/perf_handoff_window.sh` (opens a
   NEW tmux window with a fresh `claude --dangerously-skip-permissions --effort max`, empty
   context, auto-loads this file + `scripts/perf_loop_prompt.txt`), then **END YOUR TURN**.
   Headless under `scripts/perf_session_loop.sh` → just END YOUR TURN (the wrapper relaunches).
   Stop the whole loop with `touch /tmp/perf_loop_stop`. Don't start anything you can't finish
   + commit before handing off.

## Discipline

Keep the diff against upstream `tpu-inference` minimal — every line rsyncs to 8 hosts. No new
files when an existing one fits (perf scripts are the exception — name them `scripts/perf_*`).
V4 source should read like `qwen3.py` / `deepseek_v3.py`, but the loader/MoE/seed paths are
fused with the S1 fix — do not "make idiomatic" the parts flagged unsafe in `HANDOFF_PERF.md`
§Phase 5. Commit a checkpoint after every validated step.

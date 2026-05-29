# claude-deepseek-v4 — PERFORMANCE runbook (v6e-16)

> **Phase = PERFORMANCE. START HERE: `HANDOFF_PERF.md`** (current state + THE ROADMAP + the GATE +
> the ONE next action). This file holds durable slice ops (how to run, validate, pitfalls). The job:
> make `vllm serve DeepSeek-V4-Flash` **DECODE + PREFILL FAST** on the **v6e-16** slice WITHOUT
> breaking the correctness GATE. **The FIT milestone is DONE** (QUANT campaign): V4-Flash loads + fits
> + serves + decodes correctly + deterministically on v6e-16 with the 256 routed experts kept
> **FP4-compressed** (fed to `gmm_v2` as fp8 codes + per-block `rhs_scale`), **`MAX_SEQS=1` pinned**.
> That is the GIVEN foundation perf builds on — do NOT re-litigate or rebuild it (history:
> `HANDOFF_QUANT.md`). Other prior campaigns: S1 decode determinism (`HANDOFF_S1.md` /
> `CLAUDE.full.md`). Per-iteration narrative goes in **commit messages**.
>
> **One-line status (2026-05-29 — P.7): the ATTENTION-side decode levers are REFUTED — not the cost.**
> Decode wall stays at P.5's **146 ms** (P.7 changed NO production code — a measurement/analysis commit;
> GATE md5 `3069e80b` UNCHANGED). New tool `perf_microbench_attn_decode.py` (16-chip, amortized) proved:
> the Mosaic `sparse_attn_kernel` is OPTIMAL (5× > pure-JAX, parity-identical, launch-bound ~0.008 ms/layer)
> and the CSA indexer `top_k`/score is NEGLIGIBLE (~0.3 ms/step) — together only ~0.6 ms of the ~33 ms
> non-MoE. So BOTH the MoE (~106 ms, P.6 lossless floor) AND the attention kernel/indexer are floor/optimal;
> the **~32 ms balance = Q/KV/O projections + MoE gate + HC-sinkhorn + per-op launch overhead** (decode is
> launch-bound at N=1), and is the ONLY unattributed lossless frontier. **NEXT = attribute that ~32 ms**
> (extend the attn microbench to a full-per-layer block, and/or a clean multi-step profiler). The one risky
> MoE lever left = scaled-fp8-resident (fits, 0.75/layer, but LOSSY → GATE risk). See `HANDOFF_PERF.md`.
> GATE (non-negotiable): FIB `21,34,55,89,144,233,377,610` + N=2 md5 `3069e80b` ×2 fresh engines + `smoke_check` rc=0.

## Goal

Cut prefill + decode wall-time for `vllm serve deepseek-ai/DeepSeek-V4-Flash` on the **v6e-16** slice
(TP=16, 4 hosts × 4 chips, topology 4×4), driving the `HANDOFF_PERF.md` roadmap top-down — without
ever regressing the correctness GATE below. **Decode at N=1 (`MAX_SEQS=1` is pinned) is the primary
target** (now ~146 ms/step / 0.146 s/tok after P.5 lean-dequant; 277 → 220 → 146 — V4_DECODE_TIMERS, TRUSTED).
The per-step SPLIT (P.5, worker timers): device_wait ~139 ms + host-dispatch ~8 ms + aDAG; the MoE
expert-FFN is ~106 ms (76% of device_wait). P.6 showed that ~106 ms is the MoE's LOSSLESS FLOOR (every
in-trace decode kernel loses to XLA's materialize+matmul; bf16-resident experts don't fit). P.7 then
REFUTED the attention-side suspects (sparse_attn kernel OPTIMAL, indexer NEGLIGIBLE, ~0.6 ms together).
NEXT = attribute the ~32 ms non-MoE balance (Q/KV/O projections / MoE gate / HC-sinkhorn / per-op launch
overhead); the one risky MoE lever left is scaled-fp8-resident (lossy — see `HANDOFF_PERF.md`). The model already loads + fits + serves correctly with the FP4 experts kept compressed (QUANT
campaign — DONE; that path is a GIVEN, do not rebuild it). Shrink the diff vs upstream `tpu-inference`
as you go; V4 should read like `qwen3.py` / `deepseek_v3.py`, EXCEPT the loader/MoE/seed paths fused
with the S1 fix (do not "make idiomatic" — see §Phase 5 in `HANDOFF_PERF.md`).

## CORRECTNESS GATE — NON-NEGOTIABLE for every change

This is the **v6e-16 baseline** established by the QUANT campaign (the old v6e-32 + bf16 md5
`5bf42256` is **DEAD** — bf16 can't even load here, and keeping experts FP4 changed the numerics).
Every committed change MUST still pass, verified on a fresh real-V4 engine:
* `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (visible_words ≥ 10,
  max_word_run < 5).
* FIB decode: **correct Fibonacci** (`21, 34, 55, 89, 144, 233, 377, 610` — DETERMINISTIC/high-
  margin) + **N=2 md5 `3069e80b` byte-identical across 2 fresh engines** (`scripts/s1_probe2.py 2`,
  text `' 21'`). ⚠️ Do NOT gate on a long-tail md5 (`s1_probe2.py 20`+): the FIB free-form TAIL is
  NON-deterministic at temp=0 (pre-existing DECODE-path runtime nondeterminism — distributed
  all-reduce ordering), so a long-tail ref samples a nondeterministic quantity.
* Still passes after 5 unrelated requests.

A numerics-changing fix MAY shift the md5 — then re-establish a NEW reference and confirm it's
identical across 2 fresh engines + correct Fibonacci. The non-negotiable is **identical across
engines + correct Fibonacci**, not the specific hash. The basic "PASS: contains 'Paris'" line
is a **false positive** (`capital of France` can EOS at token 1, so no decode runs) — and
`completion_tokens`/`max_word_run`/`ends_clean` all read healthy on corrupted output. **READ
the actual response text** AND compare 2 fresh engines. (Why this is sacred: the S1 bug was a
per-process uninit-HBM coin flip — coherent-looking output is NOT proof. Detail in `HANDOFF_S1.md`.)

## How to validate (CHEAPEST signal first — this is the heart of the perf phase)

The full smoke is the expensive thing (full-model load + 25-45 min cold compile). Reserve it.
Escalate only as far up as the question needs:

1. **CPU numerics (no slice, cheap):** the torch oracle. `PYTHONPATH=work/tpu-inference:work/vllm
   work/vllm_env/bin/python3 scripts/s1_cpu_repro_v4flash.py both` → "OK: both eager and jit
   match" (regression-only). ✅ **RESTORED (90e7c520):** the `mesh.empty` (CPU/interpret) branch of
   `_sparse_attn_kernel_sharded` now calls pure-JAX `sparse_attn` (the Mosaic kernel is interpret-
   only on CPU, which used to crash the full-model eager path). A numerics SHIFT (e.g. a bf16 cast)
   still passes this oracle — eager+jit shift together; it catches breakage/NaN, not md5 drift.
   Attention-math parity tests `tests/models/jax/test_deepseek_v4.py -k sparse_attn` exist but are
   interpret-mode SLOW (>540s on CPU, often times out — not a practical quick check; the e2e oracle
   above is the CPU tier). CPU CANNOT reproduce S1 (no sharding) — proves math, never a determinism fix.
2. **TPU MICRO-BENCHMARK (cheap slice, NO full-model load):** jit + time a kernel/op in isolation
   on the real mesh with SYNTHETIC inputs — measures op ms / bandwidth / kernel speedup in
   ~1 min vs a 25-45 min smoke. `scripts/perf_microbench*` exists (sparse-attn) — **EXTEND it**
   for the op you're chasing (all-reduce on 4×4, `lax.top_k` over `T`, gmm_v2 fp8 vs dense bf16).
3. **Profiler re-capture (full smoke + profiler):** the structural op-breakdown truth.
   Recipe in `HANDOFF_PERF.md`. Reserve. ⚠️ The profiler INFLATES host `ParseArguments` ~100×
   (observer effect — proven P.2) and a 1-step capture is first-exec program-load, NOT steady-state:
   capture ≥20 decode steps + read the 2nd+; device op-% is reliable, absolute host/first-step is not.
   For an un-perturbed host-vs-device split prefer non-profiler worker wall-timers (HANDOFF NEXT ACTION).
4. **Full smoke + the S1 GATE above:** the per-change closure gate; at most 1-2 per session.

## Slice-serving protocol (marginal slice — do this EVERY smoke)

1. Edit code → `scripts/full_slice_v4_sync.sh` (MANDATORY: 4 hosts, each own clone; `git push`
   does NOT sync them). Verify md5 of edited .py matches head==workers — a mismatch causes
   "different launch id" Core-halts (CODE DESYNC, **not** a slice wedge; do NOT reboot — sync
   fixes it).
2. Clear `~/.cache/vllm/xla_cache/*` on all 4 hosts (stale/mixed cache also → launch-id halt).
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

* `layers/jax/attention/deepseek_v4_attention.py` — `sparse_attn` (:173 pure-JAX ref / Mosaic kernel via
  `_sparse_attn_kernel_sharded` :219) + call sites (decode :857, prefill :950); the indexer
  (`indexer_prefill` :366 / `indexer_decode_step` :607, the `lax.top_k` :659 over a STATIC `T`). ⚠️ P.7
  REFUTED both as decode levers: the Mosaic kernel is OPTIMAL (5× > pure-JAX) + the indexer is NEGLIGIBLE
  (~0.3 ms/step) — see DO-NOT-RETRY #15,16. Attention is HEALTHY/correct AND fast; the non-MoE cost is the
  projections+gate+launch overhead, NOT here. compressor; seed-from-prefill.
* `models/jax/deepseek_v4.py` — `deepseek_v4_run_with_decode_state` (decode entry);
  `transformer_body_forward` (:851) vs `transformer_body_init_state_to_buffer` (:854) = the
  DUPLICATE prefill body (Phase 0.1); `block_forward`/`block_decode_step`; `hc_pre`/`hc_post`;
  `compute_logits` (:1992, nan_to_num clamp :2005). Dead code: `_consolidate_moe_after_load`
  (:1776). The w1/w3 host-gather load (~:1492) IS the S1 fix — do not touch.
* `runner/tpu_runner.py::_prepare_inputs_dp` — `_v4_decode_replicate` (:1359): replicated
  decode activation (the S1 fix). Drives the ~17% collective cost (Phase 2) — but DO NOT
  remove; mind pitfall #5.
* `models/jax/deepseek_v4_loader.py::pick_partition_spec` (:497) — weight sharding heuristic;
  flipping contracting→output dim is the Phase-2 all-reduce win.
* `layers/jax/moe/deepseek_v4_moe.py::moe_forward` — dense decode path (:217, LEAN bf16-dequants the
  16 local FP4 experts per step, `_dequant_fp4_experts:37`) vs sharded `gmm_v2` prefill path (:233, FP4
  codes → fp8 + `rhs_scale`, the QUANT fix); the `use_shard_map` gate (:211) is the chat-wedge trigger.
  **This dense path (:300-308) is the dominant decode device cost — ~106 ms, and P.6 proved that is its
  LOSSLESS FLOOR** (2.46 ms/layer): the cost is the bf16-dequant MATERIALIZATION (not the 0.78 matmul), but
  it can't be removed losslessly — every in-trace kernel loses (naive Pallas matvec 11.6, gmm 5.48) and
  bf16-resident experts don't fit (34.3>31.25 GiB). Decode operands already bf16/fp32 (PERF 3.1). Do NOT
  re-attempt a decode MoE kernel (HANDOFF DO-NOT-RETRY #12–14).
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

The slice (`v6spoteu719`, **v6e-16**, topology 4×4 = 16 chips / 4 hosts, zone `europe-west4-a`,
project `prm-research`) is bootstrapped: venv on all 4 hosts, GCS weights mounted, **Ray healthy
(16 TPU, `ray.init` verified)**.
⚠️ **WEIGHTS readable by enyouki on ALL 4 hosts** — run `scripts/full_slice_v4_mount_weights.sh`
once per bringup (idempotent; smoke pre-flight runs it; mounts die on reboot). The bringup mounts the
GCS bucket enyouki-owned ONLY on the head; workers get a root-only mount → enyouki EACCES → the worker
serve proc silently loads `jnp.zeros` (dummy fallback) → engine wedges. Verify `config.json` readable
on .8/.17/.16. Head = `10.164.0.15`; workers (`.8`/`.17`/`.16`) auto-discover via
`scripts/full_slice_v4_discover.sh`. ssh: `ssh enyouki@<ip> -i ~/.ssh/google_compute_engine`.
Weights load auth-free from the GCS mount (`HF_TOKEN` intentionally unset). venv python 3.12.
⚠️ **numpy MUST be `<2.4` (pinned `2.3.5`)** — 2.4.x breaks `import numba`, crashing the smoke's
APIServer before any TPU work (unrelated-looking stack). No pip in the venv (uv):
`~/.local/bin/uv pip install --python work/vllm_env/bin/python3 'numpy==2.3.5'` on ALL 4 hosts
(per-host venvs). Pre-smoke check: `python3 -c "import numba"` per host.
Smoke = TP=16 (`full_slice_v4_smoke.sh`); ray = `full_slice_v4_ray_restart.sh` (verifies 16 TPU).
⚠️ `ray.init` "version mismatch" = mark's rogue ray-2.54.1 `node` container poisoning the GCS
`CLUSTER_METADATA` (NOT a corrupt venv). FIX = keep BOTH guardians alive: `node_guardian` (occupies
the `node` container name) + `meta_guardian` (re-stamps metadata→2.55.1, needs head-IP arg
`10.164.0.15:6379`). Start/restart per the loop prompt + Pitfall #0. Started this bringup.
Fresh-VM setup: `./scripts/full_slice_v4_bootstrap.sh` — needs `uv` + the
`~/.ssh/google_compute_engine` key on all hosts (generate/propagate via
`gcloud compute tpus tpu-vm ssh <node> --zone europe-west4-a --worker=0`).

## ⇒ CONTEXT HANDOFF PROTOCOL — every session MUST follow

Sessions are disposable; the commit log + `HANDOFF_PERF.md` + this file are the only memory
that survives. When context reaches **~100k–200k tokens** (interactive: check `/context`; heed
"context getting long" reminders) AND you are NOT seriously mid-operation (waiting on a smoke
whose result you must read, or a few calls from finishing a committed step) — HAND OFF:

1. **Keep the markdown LEAN** — trim `CLAUDE.md` + `HANDOFF_PERF.md` (and memory files):
   delete superseded narrative, keep only durable ops + the CURRENT state + the next action.
   Every line rsyncs to 4 hosts / loads into every session. Lean > complete.
2. Rewrite `HANDOFF_PERF.md` to the CURRENT state: what you landed + gated, what's in flight
   (log paths, ports, microbench numbers), and the ONE most important next action.
3. `git add -A && git commit && git push`.
4. **HAND OFF THE WINDOW:** the loop runs as an INTERACTIVE chain — run
   `scripts/perf_handoff_window.sh`, then **END YOUR TURN**. It opens a NEW tmux window with a
   fresh `claude --dangerously-skip-permissions --effort max` (empty context, auto-loads this file +
   `scripts/perf_loop_prompt.txt`) that continues the campaign. ⚠️ Do NOT end your turn WITHOUT
   running it expecting a wrapper to relaunch you — the headless `perf_session_loop.sh` wrapper is
   RETIRED, so skipping the script KILLS the loop (no successor). Stop the loop on purpose with
   `touch /tmp/perf_loop_stop` (the script honors it). Don't start anything you can't finish + commit
   before handing off.

## Discipline

Keep the diff against upstream `tpu-inference` minimal — every line rsyncs to 4 hosts. No new
files when an existing one fits (perf scripts are the exception — name them `scripts/perf_*`).
V4 source should read like `qwen3.py` / `deepseek_v3.py`, but the loader/MoE/seed paths are
fused with the S1 fix — do not "make idiomatic" the parts flagged unsafe in `HANDOFF_PERF.md`
§Phase 5. Commit a checkpoint after every validated step.

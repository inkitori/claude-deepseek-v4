# claude-deepseek-v4 — agent runbook

You're picking up a TPU-inference effort: get
`vllm serve deepseek-ai/DeepSeek-V4-Flash` deployable on a
**v6e-32 TPU slice** (TP=32, 8 hosts × 4 chips) at production
quality. The model is a flagship MoE — 256 FP4 experts + MLA-
shaped attention + dense FP8, ~543 GiB bf16-expanded.

The non-negotiable goal is **fast, mathematically correct
inference with the real V4-Flash weights**, served via the
OpenAI-compatible HTTP endpoint to many concurrent users.

**State of the system (read before claiming progress):**
Commit `1f212036` implements S1 Option C — `lax.optimization_barrier`
on each output packed buffer in `deepseek_v4_run_with_decode_state` —
and removes the prior 14-callback-per-layer band-aid (it was
deterministic-breaking AND incomplete; missed the compressor/
indexer state outputs). The fix is **CPU-validated** at tiny
config and V4-Flash-truncated dims under `jit + donate_argnums=0`
but **NOT yet TPU-validated** on real V4-Flash. See S1 below for
the full picture.

**To pick this up cold:**
1. `git log --oneline -5` — current head should be `1f212036` "S1
   fix: optimization_barrier on output packed buffers, remove host
   callbacks".
2. CPU repros that prove the fix doesn't regress:
   * `scripts/s1_cpu_repro_tiny.py` — tiny config, ~30s eager / ~5s jit.
   * `scripts/s1_cpu_repro_v4flash.py` — V4-Flash truncated to 4
     layers with 8 experts (full V4-Flash hidden_size / 64 heads /
     1024 q_lora_rank / 512 index_topk), ~30s param init / ~85s
     jit prefill / ~80s jit decode on CPU.
   * `scripts/s1_cpu_hlo_check.py` — dumps lowered + compiled HLO
     of one decode step, counts `stablehlo.optimization_barrier`
     ops. Lowered HLO should contain 6 barriers (one per layer);
     CPU XLA strips them in compile, TPU XLA should keep them.

   Run via:
   ```
   PYTHONPATH=work/tpu-inference:work/vllm \
     <venv>/bin/python3.11 scripts/s1_cpu_repro_tiny.py both 8 8
   ```
   Both repros should end in "OK: both eager and jit match
   fresh-prefill argmax". Needs `jax==0.9.2`, `numpy`, `torch`
   (CPU); the on-host venv at `work/vllm_env/` works but a
   minimal CPU venv with just those works too.
3. Real-V4 verification requires bootstrap of the slice; see
   "Slice bootstrap" section below. After bootstrap, the iterate
   loop in this file is the validation gate.
4. If output is still degenerate on TPU after this fix, set
   `V4_DECODE_NAN_TRIPWIRE=1` to re-enable callback-based anchoring
   (non-deterministic but suppresses NaN) and gather per-field
   diagnostics. See S1's "Fallback if Option C is insufficient
   on TPU" paragraph.

## Discipline

### 1. Minimum-delta rule
The diff against upstream `tpu-inference` should be **as small
as possible**. Every line you add gets rsync'd to 8 worker hosts
and read by the next agent.

* No new files when an existing one fits. V4 lives in
  `models/jax/deepseek_v4{,_loader}.py`,
  `layers/jax/{attention,moe}/deepseek_v4_*.py`.
* No new test classes for variants of an existing case —
  parametrize.
* Reuse upstream layers (`sparse_attn`, `rms_norm`,
  `megablox/gmm`, `ragged_paged_attention/v3`) before writing
  V4-specific copies.
* Touch `runner/` / `worker/` / `platforms/` only as a last
  resort. The two existing V4 runtime hooks
  (`kv_cache_manager.py::_initialize_kv_cache_deepseek_v4`,
  `tpu_runner.py::_maybe_set_v4_decode_start_pos`) are the
  entire delta and should stay that small.
* Delete dead code as you touch it.

### 2. CLAUDE.md is for durable knowledge — `git log` is for narrative
Per-iter narrative goes in commit messages. CLAUDE.md only
gets updated when something durable changes (new pitfall,
corrected env knob, topology change, backlog reorder). Targets
~300 lines; prune before adding.

### 3. Code style matches upstream
This work targets eventual upstream PR. V4 source files should
read **indistinguishable from `qwen3.py` / `deepseek_v3.py`**.

* Brief Args/Returns docstrings. No multi-paragraph rationale,
  no "previous implementation" history.
* Inline comments only when WHY is non-obvious. No iter-narrative
  ("iter-5h", "Bug A", "hyp-3"), no section banners.
* Trust internal call paths; validate only at the vLLM API
  boundary.
* No backwards-compat shims for never-shipped code.

30-second diff against `qwen3.py` before declaring "done".

## Slice bootstrap

The `.env` and venv aren't checked in; a fresh slice needs:

1. **gcloud auth** must reach this project (`prm-research`). Check
   with `gcloud config get project` + `gcloud compute tpus tpu-vm
   list --zone=<your-zone>`. The slice's hostname will look like
   `t1v-n-<id>-w-<worker_num>`; verify with `hostname`.
2. **Discover worker IPs**:
   ```bash
   curl -s -H "Metadata-Flavor: Google" \
     http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints \
     | tr ',' '\n' | awk -F: '{print $3}'
   ```
   Order in that list is worker 0 → worker N. Worker 0 is the head.
3. **Cross-worker SSH**: `gcloud compute tpus tpu-vm ssh
   <user>@<tpu-name> --zone=<zone> --worker=0 --command=true` once
   to auto-generate `~/.ssh/google_compute_engine` and propagate
   it to all workers. After that, plain `ssh -i
   ~/.ssh/google_compute_engine <user>@<worker_ip>` works.
4. **`.env`** (copy from `.env.example`):
   ```
   HF_TOKEN=hf_...    # only needed for tokenizer/config download fallback
   MOUNT_GCS=1
   GCS_BUCKET=<bucket with V4-Flash hub layout>
   GCS_ONLY_DIR=<path/under/bucket>
   ```
   `mark`'s slice had V4-Flash staged at `gs://personal-mark-eu/vllm/hub/`
   readable by the project's default compute service account — useful
   if you have read access. Otherwise stage it yourself or fall back
   to HF download (slow, 543 GiB).
5. **Override slice IPs**: `scripts/full_slice_v4_bootstrap.sh` and
   `scripts/full_slice_v4_smoke.sh` hard-code `HEAD_IP=10.164.0.41`
   and a specific 7-worker list. Either edit those constants or run
   the bootstrap with `HEAD_IP=<head> WORKERS="<7 space-separated
   IPs>"` env vars. The smoke launcher hard-codes `RAY_ADDRESS`
   too — patch that or export `RAY_ADDRESS=<head_ip>:6379`.
6. From the head (worker 0): `./scripts/setup.sh` (builds local
   venv) then `./scripts/full_slice_v4_bootstrap.sh` (fans setup
   to other 7 workers + starts Ray cluster). First run takes
   ~10-15 min (parallel venv builds).
7. Then the normal iterate loop in this file works.

## Cluster topology

* Slice: **v6e-32** = 8 hosts × 4 chips × 32 GB HBM = 992 GiB.
* Head + worker 0: `10.164.0.41` (launch from here).
* Workers 1-7: `10.164.0.{22,35,36,39,45,18,30}`.
* Each host is its **own VM with its own clone**. **No shared
  filesystem.** `git push` does NOT propagate to workers — see
  pitfall #3.
* Ray address `10.164.0.41:6379`; `ray status` should show
  8 nodes / `0.0/32.0 TPU` when idle.
* Venv: `~/claude-deepseek-v4/work/vllm_env` on every host.
* Weights: GCS-mounted via `scripts/mount_gcs.sh` to
  `~/.cache/huggingface/hub`.

## The iterate loop

```bash
scripts/full_slice_v4_reset.sh        # cluster cleanup
scripts/full_slice_v4_sync.sh         # MANDATORY after any code edit
scripts/full_slice_v4_smoke.sh        # launch vllm serve (background)
scripts/full_slice_v4_smoke_check.sh  # validate
```

`reset.sh` clears 4 orphan-state surfaces (api-server pid,
`VLLM::EngineCore` actors, `/tmp/libtpu_lockfile`, leaked
placement groups). Don't escalate to `ray_restart.sh` unless
reset fails.

`sync.sh` rsyncs `work/tpu-inference/tpu_inference/` and
`scripts/` to all 7 workers. Repo-root markdown is head-only.

`smoke_check.sh` polls `/v1/models`, fires the Paris probe ×2,
asserts byte-equal + contains "Paris". Optional probes via
`*_REQUIRED=1` (see knob table). Self-test:
`scripts/test_smoke_check_harness.sh`.

## Pass criterion

`smoke_check` exits 0 with `PASS: deterministic completion
contains 'Paris'` AND every `_REQUIRED=1` probe passes.
`LONG_GEN_REQUIRED=1` is **default-on** since the basic Paris
gate only validates 1-3 tokens.

## Knobs

Set on the launching shell; smoke.sh forwards to Ray workers
via `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`.

| env var | default | what it does |
|---|---|---|
| `MAX_LEN` | `256` | `--max-model-len`. A1 lifts this. |
| `MAX_SEQS` | `1` | `--max-num-seqs`. S2 lifts this. |
| `V4_LOADER_SLICE_AWARE` | `1` | Each host reads only its row range. |
| `V4_LOADER_PLACE_WORKERS` | `8` | Per-host placement threads. |
| `V4_LOADER_PREFETCH_WORKERS` | `0` | Prefetch threads (no win observed). |
| `VLLM_XLA_CACHE_PATH` | `~/.cache/vllm/xla_cache` | Per-host JAX compile cache. **Cross-host rsync is unsound** (SPMD compiles host-specific binaries). |
| `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES` | `0` | Cache small modules (JAX 0.9 name). |
| `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` | `0` | Cache fast-to-compile modules. |
| `RAY_CGRAPH_get_timeout` | `3600` | Ray compiled-graph timeout. Default 300 trips on first inference. |
| `V4_XLA_FLAGS` | unset | Opt-in custom `XLA_FLAGS` (smoke.sh does NOT inherit parent-shell `XLA_FLAGS`; pitfall #4). |
| `V4_DECODE_NAN_TRIPWIRE` | `0` | When `1`, `_v4_nan_tripwire` prints per-field nan/inf/max_abs reductions for diagnostics. When `0` (default), the function is a no-op — anti-elision is held by `_v4_anchor_output_buffers` (`lax.optimization_barrier` on each output packed buffer), see S1. |

`*_REQUIRED` smoke-check knobs (exit code in parens): `CHAT_REQUIRED` (4),
`REASONING_REQUIRED` (5), `STREAMING_REQUIRED` (6), `SAMPLING_REQUIRED` (7),
`STOP_REQUIRED` (8), `LOGPROBS_REQUIRED` (9), `TOPK_REQUIRED` (10),
`PRESENCE_REQUIRED` (11), `N_REQUIRED` (12), `LONG_GEN_REQUIRED` (13,
default-on). See `scripts/full_slice_v4_smoke_check.sh` head doc for
specifics.

## Production-readiness backlog

Pick the highest-leverage uncompleted item. Items earlier block
later. Per-iter narrative goes in commit messages.

### Tier S — silent correctness bombs (fix before perf)

#### S1. Decode produces degenerate output past pos ~2

**The single most important item.** All other backlog work is
blocked on this. `__call__` routes through
`deepseek_v4_run_with_decode_state`, threading per-layer
`AttentionDecodeState` through `kv_caches`. Tiny-config tests pass;
real-V4 generation falls off the prose manifold within 2-3 tokens.

**Symptom (real V4-Flash on v6e-32, pre-fix):**
```
prompt: "Tell me a short story about a robot exploring Mars:"
max_tokens=64, temperature=0
→ visible text: " 0.0 0.0 0.0 0.0 0.0 0.0 …" (numeric attractor)

prompt: "The capital of France is" (3 runs, byte-identical, temp=0)
→ run 1: " Paris, 2014-2015, 2016-2017"
→ run 2: " Paris, Paris, 巴黎，法国，..."
→ run 3: " Paris, Paris, Paris, Paris, Paris,"
```

The first decoded token is correct; subsequent tokens collapse into
a degenerate attractor (repetition, list, numeric run). Outputs are
also non-deterministic from byte-identical input at temperature=0.

**Don't trust `usage.completion_tokens`, `max_word_run`, or
`ends_clean`** — all read healthy on corrupted output (pinned by
`long_gen_required_invisible` in the harness self-test). Closure
gate is `LONG_GEN_REQUIRED=1`: `visible_words >= 10` AND
`max_word_run < 5`.

**Root cause (audited 2026-05-01):** V4 writes its KV-cache via
pure JAX `at[].set` partial writes on a manually-packed fp32
buffer (sites in `attention_decode_step`,`compressor_decode_step`,
`indexer_decode_step`, and the prefill seed
`_compressor_state_from_prefill`), then re-packs through
`_pack_layer_state`. V3 and Qwen3 work fine under the same
`donate_argnums=2` because their kv_caches are written by Pallas
kernels whose outputs are consumed and returned (opaque to XLA's
alias analysis). V4's pack/unpack indirection gives XLA a long
sequence of pure functional ops where alias analysis can rewrite
the partial writes as in-place updates of the donation slot and
drop ones whose result it can statically prove isn't observed
elsewhere — typically the compressor / indexer state slots, whose
only consumer is the *next* decode call.

Earlier band-aids anchored some fields via host callbacks
(commit `f1ec28f8`, fix #4: ~14 callbacks per attention layer ×
~43 layers ≈ 600/decode_step). Those anchored the inputs and the
SWA post-write, but NOT the compressor/indexer outputs — so
NaN→pad was suppressed (the SWA write was preserved) but the
compressor state still drifted, producing the degenerate
attractors above. The callback count also perturbed SPMD
floating-point reduction order, breaking determinism at
temperature=0.

**Current fix — Option C (output-side `lax.optimization_barrier`):**
`_v4_anchor_output_buffers` in `models/jax/deepseek_v4.py:772`
wraps each output packed buffer in `lax.optimization_barrier`
before they're returned from `deepseek_v4_run_with_decode_state`.
That forces XLA to materialise every `at[].set` and concatenate
upstream of the barrier — including the compressor/indexer fields
the old callback set missed. Host callbacks are removed (deferred
behind `V4_DECODE_NAN_TRIPWIRE=1` for diagnostics), restoring
deterministic float reduction order.

Why a barrier on the *output* and not on the input (as fix #7
tried): `optimization_barrier(b)` on an output forces every upstream
op contributing to `b` to compute the intended value, because the
barrier's output is observed via the JIT return. Fix #7's
input-side `k + optimization_barrier(0.0)` was algebraically
identity (`k + 0 = k`) — XLA's algebraic simplifier could fold
through the constant barrier and proceed to in-place rewrite.

**Validation matrix:**
* CPU tiny-config (7 layers) + JIT + `donate_argnums=0`: passes,
  byte-equal to fresh-prefill argmax across N=8 decode steps.
* CPU V4-Flash-truncated (4 layers, 8 experts, full V4-Flash
  dims) + JIT + `donate_argnums=0`: passes, byte-equal to
  fresh-prefill argmax across N=6 decode steps. (~85s prefill /
  ~80s decode on CPU.)
* CPU lowered HLO contains 6 `stablehlo.optimization_barrier` ops
  (one per layer); compiled HLO retains all 132 dynamic-update-
  slice writes. CPU XLA strips the barriers in compile but the
  writes are kept regardless — CPU never had the elision bug.
* TPU verification (real V4-Flash, v6e-32, 43 layers, 256
  experts): **NOT YET RUN**. Needs `scripts/full_slice_v4_smoke.sh`
  + `scripts/full_slice_v4_smoke_check.sh` with
  `LONG_GEN_REQUIRED=1`. Other code in the project
  (`offload/utils.py`, `layers/common/fused_moe_gmm.py`) uses
  `lax.optimization_barrier` on TPU as the standard anti-
  optimization primitive.

**Fallback if Option C is insufficient on TPU:**
Set `V4_DECODE_NAN_TRIPWIRE=1` to re-enable the per-field host
callbacks AND get visible prints of nan/inf/max_abs per layer.
That re-introduces non-determinism but restores the old NaN-
suppression behaviour and produces a diagnostic trace.

**Other untried paths (if Option C also fails):**
1. Fold `at[].set` writes into a `pl.pallas_call` whose output is
   the new state (matches V3/Qwen3 pattern; merges naturally with
   B1's sparse-attention Pallas kernel).
2. Special-case `donate_argnums` for V4 prefill JIT only (decode
   keeps donation). The prefill path doesn't actually use arg2;
   fix #6 attempted full un-donation and crashed, so partial
   un-donation needs diagnosis of which kernel asserted donation.

**Don't repeat (full traces in `git log`):**
- `ac8d2077` — single per-layer callback in transformer body. Insufficient.
- `98b0a677` — single callback at kv_cache_post_write. NaN returns.
- `14e11136` — callbacks at at_entry × 6 fields. NaN returns.
- `5c9d9213` (rev `c32fe431`) — full un-donate kv_caches for V4. TPU UserFatal.
- `75b92f4b` (rev `9d2f15ec`) — input-side opaque copy. XLA optimized in-place.

**Real-V4 verification:**
* Diagnostics: `V4_DECODE_NAN_TRIPWIRE=1 scripts/full_slice_v4_smoke{,_check}.sh`.
* End-to-end gate (default-on): `LONG_GEN_REQUIRED=1
  scripts/full_slice_v4_smoke_check.sh`.
* Determinism check: run 3+ Paris probes, healthy = byte-equal.

**Plumbing locations** (read before touching):
* `layers/jax/attention/deepseek_v4_attention.py::attention_init_state_from_prefill` — likely first-corruption site.
* `models/jax/deepseek_v4.py::deepseek_v4_run_with_decode_state` — Option C goes here.
* `models/common/model_loader.py:332+` — `donate_argnums=2`, V4 `kv_cache_sharding=P()`.
* `runner/kv_cache_manager.py::_initialize_kv_cache_deepseek_v4`.
* `runner/tpu_runner.py::_maybe_set_v4_decode_start_pos`.

S1 closure unlocks A1, B1, S5.

#### S2. Multi-sequence dispatch is a Python loop

`__call__` runs each active sequence sequentially through
`transformer_body_decode_step_from_buffer`. Need a ragged-batch
jit'd kernel (extend `kernels/ragged_paged_attention/v3` for
top-k + attn_sink + dual-buffer KV, or jit V4's path with
`lax.dynamic_slice` per active seq). Until then `--max-num-seqs=1`.

#### S3. Reasoning + tool parsers — wired, output broken with S1

Smoke launcher passes `--reasoning-parser deepseek_v4`,
`--enable-auto-tool-choice`, `--tool-call-parser deepseek_v4`.
`REASONING_REQUIRED=1` fails today for the same S1 reason.
`TOOLS_REQUIRED=1` probe still TODO — assert `tools=[...]` populates
`tool_calls` not raw DSML in `content`.

Quick parser-registry sanity check (no TPU):
```bash
PYTHONPATH=work/vllm:work/tpu-inference work/vllm_env/bin/python3 -c "
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
ReasoningParserManager.get_reasoning_parser('deepseek_v4')
ToolParserManager.get_tool_parser('deepseek_v4')
print('OK')"
```

#### S4. Chat encoding — RESOLVED upstream

`DeepseekV4Tokenizer` auto-loads, ignores `--chat-template`,
byte-equal to V4-Flash reference encoder. Pinned by
`TestVllmChatTemplateParity`. Regression boundary only.

#### S5. MTP speculative decoding hook is not wired

`runner/speculative_decoding_manager.py` only handles `ngram` and
`eagle3`. Math is ready (`deepseek_v4_mtp_forward` validated on
tiny). Need `DeepseekV4MTPProposer` in
`tpu_inference/spec_decode/jax/` + wire into `execute_draft_model`
+ engine flag `--speculative-config '{"method":"deepseek_v4_mtp",
"num_speculative_tokens":1}'`. 1.5–2× decode throughput once S1
lands.

#### S6. Sampling parameters — probes scaffolded; only validate first ~3 tokens

Per-knob probes (sampling, stop, logprobs, top-k, presence, n>1)
all "pass" but only on the same Paris-shaped prompt that produces
1-3 tokens. Any "S6 done" claim is meaningless until S1.
Per-request `seed` is rejected by vLLM/TPU on non-greedy paths
(HTTP 400) — determinism under sampling not asserted.

#### S7. Streaming — equivalence probe scaffolded, same caveat as S6

`STREAMING_REQUIRED=1` re-fires Paris with `stream=true` and
asserts SSE byte-equality vs non-stream. Doesn't validate
sustained streaming. Latency probe (TTFT/ITL behind
`STREAMING_LATENCY_REQUIRED=1`) still TODO.

#### S8. Sustained-generation gate — `LONG_GEN_REQUIRED=1`

Default-on smoke probe: `max_tokens=64` on
`"Tell me a short story about a robot exploring Mars:"`,
asserts `completion_tokens >= 30`, no 5+-word streaks, trailing
5 chars contain ≥2 alphanumerics. This is the gate that exposes S1.

Future extensions (same backlog item): `MULTI_TURN_REQUIRED=1`
(2-turn chat coherence), `LONG_PROMPT_REQUIRED=1` (2k-token
prompt, separate from C2's needle sweep).

### Tier A — production infra (all blocked on S1)

* **A1.** `MAX_LEN=256, MAX_SEQS=1` hard-coded; lift after S1.
* **A2.** Persistent compile cache is host-local + ephemeral.
  Move to durable mount; one-shot bootstrap warm.
* **A3.** No engine crash recovery. Supervisor + drain on SIGTERM.
* **A4.** No metrics / observability. `--enable-metrics` not set.
* **A5.** No TLS / auth / rate limiting; currently 0.0.0.0:18081 plain.
* **A6.** Single slice — no horizontal scale.
* **A7.** Prefix caching disabled (`--enable-prefix-caching` not
  verified compatible with V4's per-layer state in params tree).
  5-10× win on shared-prefix workloads.
* **A8.** Cancellation propagation unverified — TCP-disconnect
  mid-stream may leak compute.
* **A9.** No `/health` vs `/ready` distinction (K8s would route
  during cold compile).
* **A10.** No server-side request caps (max_tokens, prompt length,
  payload size, n unvalidated).
* **A11.** No worker-host weight-divergence detection. Hash-and-
  compare bf16 weight bytes per host at engine init.

### Tier B — performance

* **B1.** Sparse-attention Pallas kernel (currently fully-
  materialized in `deepseek_v4_attention.py::sparse_attn`).
  2–5× decode latency.
* **B2.** True sparse MoE dispatch via `kernels/megablox/gmm.py`
  (currently vectorized-dense; FLOP cost is `top_k * E` higher).
* **B3.** SPMD remat-warning audit — DONE for `compressor.ape`
  family (126 → 0). Re-audit: `grep "Involuntary full
  rematerialization" logs/full-slice-v4-smoke-*.log`.
* **B4.** AOT compile + binary persist. Defer until B1+B2.

### Tier C — quality gates

* **C1.** lm-eval-harness vs DeepSeek reference (MMLU, HellaSwag,
  GSM8K, HumanEval, MATH). Needs S2. The honest "we serve V4-Flash"
  gate.
* **C2.** Long-context functional (4k → 1M needle-in-haystack).
  Needs A1.
* **C3.** Math regression suite under load.
* **C4.** Tokenizer edge cases — extend `TestVllmChatTemplateParity`.
* **C5.** Refusal/safety preservation.

### Tier D — janitorial

* **D1.** `tests/models/jax/test_deepseek_v4.py` is large (~3500
  LOC, ~30 test classes). Per-class audit needed; decode-state
  classes are the biggest cut after S1.
* **D2.** No log rotation. `logs/full-slice-v4-smoke-*.log`
  accumulates ~1–2 MB per run.

## Iteration discipline

**Don't use the full smoke as your inner test loop** — each
attempt is 25-45 min of waiting. Use the fastest validation that
catches the bug class:

1. Standalone math scripts under `/tmp/` — ~10–30s.
2. Tiny-fixture pytest classes in `test_deepseek_v4.py` — ~30s-2min CPU.
3. `eval_shape` / `lower(...).compile()` on real config under
   virtual mesh — ~1-3 min. Pattern:
   `XLA_FLAGS=--xla_force_host_platform_device_count=32
   JAX_PLATFORMS=cpu`.
4. Real `vllm serve` smoke — at most 1-2 per session.

### Real-smoke phase budgets

| Phase | Expected | Bail signal |
|---|---|---|
| Startup + Ray init | ~30s | No log activity >2 min |
| Weight load | ~4 min | No `[deepseek_v4] placed N tensors` >2 min |
| `capture_model` precompile | ~30s | `RESOURCE_EXHAUSTED` |
| `jit_run_model` cold compile | **10–30 min cold, ~97s warm** | 3+ `slow_operation_alarm.cc`, OR `RESOURCE_EXHAUSTED` |
| First curl | sub-second after compile | timeout, engine crash |

Silence in `jit_run_model` ≤25 min is *expected*, not stuck.

### Iter-timeout management

`ITER_TIMEOUT_SEC=5400` (90 min). At T-15 min stop launching new
long steps + commit a "WIP:" checkpoint. At T-5 min reset cluster
+ push.

## Killing vLLM cleanly

`vllm serve` forks an EngineCore that spawns Ray actors per
worker. SIGKILL on the api-server doesn't reap them; they hold
TPU + libtpu state. Always use `scripts/full_slice_v4_reset.sh`
— it kills by exact `comm` match (never broad regex; pitfall #1).
Escalate to `full_slice_v4_ray_restart.sh` only when reset fails.

## Pitfalls already learned (don't repeat)

1. **Broad `pkill` regex hits raylets.** `pkill -f
   "EngineCore|RayWorkerWrapper|vllm"` matches strings in
   raylet's command line on remote workers and kills the daemon
   (lost 7/8 nodes once). Use exact comm match
   (`pkill -x VLLM::EngineCore`) or pid.

2. **`/tmp/libtpu_lockfile` survives SIGKILL.** Killed EngineCore
   leaves lockfile; subsequent inits SIGSEGV. Reset script handles;
   manual kills need manual lockfile removal.

3. **`git push` doesn't sync workers.** Each worker has its own
   clone. **Always `scripts/full_slice_v4_sync.sh` after any code
   edit.** Only `work/tpu-inference/tpu_inference/` and `scripts/`
   are synced — repo-root markdown is head-only.

4. **Don't add unverified XLA flags.** `--xla_tpu_impure_hlo_
   parallel_compile=true` looked plausible but isn't recognised
   in this libtpu build and SIGSEGVs every Ray worker. Validate
   with `python -c "import jax; jax.devices()"` first. smoke.sh
   ignores parent-shell `XLA_FLAGS`; opt in via `V4_XLA_FLAGS`.

5. **First inference is slow.** Cold `jit_run_model` compile
   = 5-15 min; warm cache ~30-60s. `smoke_check` defaults curl
   to 900s. To warm at bootstrap: `WARM_CACHE_ON_BOOTSTRAP=1` in
   `.env` before `./run.sh bootstrap`.

6. **`--enforce-eager` doesn't skip XLA compile.** That flag only
   affects vLLM's CUDA-graph-equivalent path. The TPU forward is
   JAX/`tpu-inference` and ALWAYS jit-compiles.

7. **vLLM's `capture_model` multiplies compile cost.** Without
   `--enforce-eager`, vLLM precompiles many shape buckets up
   front. smoke.sh uses `--enforce-eager` to skip that.

8. **`JAX_COMPILATION_CACHE_DIR` does nothing under vLLM.**
   `compilation_manager.py:53` overrides to `VLLM_XLA_CACHE_PATH`.
   Verify cache via `ls -la ~/.cache/vllm/xla_cache`.

9. **First chat call OOM-retries.** Chat path lands in 1024-token
   prefill bucket vs 256 for completions; tight HBM triggers
   `TpuLoadedExecutable::ExecutePrepareWithOomRetries`. Adds
   ~30s to first-chat latency; subsequent calls are fast.

10. **Don't trust `usage.completion_tokens` for output validity.**
    The engine reports 64 even when visible text is 1 character.
    Always read response `text` (or `extract_long_gen_metrics`).

## Sanity check on a fresh VM

```bash
# CPU math (~10s, no TPU)
work/vllm_env/bin/python3 -m pytest \
    work/tpu-inference/tests/models/jax/test_deepseek_v4.py::TestFp8DequantIndependentReference \
    work/tpu-inference/tests/models/jax/test_deepseek_v4.py::TestFp4DequantIndependentReference \
    -x -q

# Smoke harness self-test (no TPU)
scripts/test_smoke_check_harness.sh

# Cluster (expect 8 nodes, 0.0/32.0 TPU)
scripts/full_slice_v4_reset.sh && ray status

# Real smoke
scripts/full_slice_v4_sync.sh
scripts/full_slice_v4_smoke.sh
scripts/full_slice_v4_smoke_check.sh
```

## Layout

* `work/tpu-inference/tpu_inference/models/jax/deepseek_v4*.py`,
  `layers/jax/{attention,moe}/deepseek_v4_*.py` — V4 source.
* `work/vllm/` — vLLM upstream (don't edit unless you've read
  `work/vllm/AGENTS.md`).
* `scripts/full_slice_v4_*.sh` — per-host operational helpers.
* `logs/` — `.gitignore`d.
* `prompt.md` — autonomous loop's prompt; read CLAUDE.md first.

Durable docs in `work/tpu-inference/`: `INVARIANTS.md` (math
invariants), `DECISIONS.md` (architectural decisions),
`TINY_CONFIG.md`, `TOLERANCE_LOG.md`, `V3_TO_V4_DIFF.md`.

V4-Flash ships **no Jinja `chat_template`**.
`DeepseekV4Tokenizer` resolves via upstream `encode_messages`,
ignoring any `--chat-template`. Pinned by
`TestVllmChatTemplateParity`. `vllm chat` CLI needs
`--url http://localhost:18081/v1`.

<!-- ===== trimmed from CLAUDE.md by s1_trim_claudemd.sh on 2026-05-24T11:18:14Z ===== -->

## PHASE 2 — S1 REPRODUCED at small scale (2026-05-24)

There is now a **cheap multi-host reproducer.** The single-host small-TPU
loop the prior runbook assumed DOES NOT EXIST: a lone worker can't boot a
v6e-32 (libtpu `CreateTpuSystemState` waits forever for the other 7 hosts);
all TPU work needs all 8 hosts. New tooling (committed):
* `scripts/full_slice_v4_mh_run.sh <script.py> [args]` — fan a script out
  across all 8 hosts as ONE `jax.distributed` job (no-arg metadata
  auto-detect; head may land as any proc index), cleans the libtpu lock,
  per-proc logs in `logs/mh-*`.
* `scripts/s1_mh_repro.py <mode> <T> <N> <n_layers> <action> <n_experts>` —
  truncated V4 (random weights, no 543 GiB load) on the 32-chip mesh with
  production's named axes (layers/common/sharding.py); KV state replicated
  `P()`, kv_caches donated. `mode`∈{replicated,sharded}. `action=repro`
  prints VERDICT; `action=diff` localizes the corrupted packed-state field
  (donate vs non-donate buffer diff). Must build replicated arrays via
  `make_array_from_callback` (device_put-reshard of process-local arrays to
  global `P()` tile-allgathers the embed/head weight 8× and OOMs HBM).
* `scripts/full_slice_v4_node_guardian.sh` — background loop re-stopping the
  redeployed `node` container (it CAME BACK 2026-05-24 — run during TPU work).

**Result — the bug needs SHARDING + donation, not just TPU + donation:**
* `replicated` (all `P()`, 32 chips, no reshard) → **NO_S1** (bad=0/12), as CPU.
* `sharded` (attn_dp=8, production-style: experts+attn parallel, KV `P()`)
  → **S1_REPRODUCED** (bad=1/12).

**DONATION IS EXONERATED — and the decode MATH is correct.** `action=diff`
(donate vs NON-donate, both sharded) → packed buffers **byte-identical** at
every (step,layer,field) (`first=None`) AND identical argmax (bad_d=bad_n=1).
So donation drops NOTHING — overturning the hypothesis the entire prior
effort (barrier `1f212036`, un-donation `5c9d9213`, callbacks) rested on;
that's why they ALL failed. The lone sharded divergence is a **benign
near-tie flip**: the 1/12 mismatch is at step 4 then RECOVERS (steps 5-11
OK), oscillating among the same token-ids the oracle emits (random *0.02
weights ⇒ flat near-tie logits + attn_dp reduction-order noise). It does NOT
compound. So `deepseek_v4_run_with_decode_state` (compressor/indexer/SWA
state threading) is CORRECT under replicated AND sharded, donated or not.

**S1 IS NOT IN THE MODEL DECODE MATH.** Both the CPU repros and this MH repro
test that core and pass. Stop patching the write sites / donation.

**RULED OUT so far (don't re-tread):**
- Donation (diff: byte-identical buffers).
- Decode math: replicated bad=0; sharded bad=1/12 and 4/48 — but mismatches
  are SCATTERED (steps 20/21/32/37 at N=48), never compound, and every one is a
  near-tie (top1-top2 logit gap ~5e-4..4e-3, same as OK steps) ⇒ benign attn_dp
  reduction noise on flat random-weight logits.
- Cross-mode forward (free, from logs): replicated-forward vs sharded-forward
  teacher-forced argmax differ at ONLY the same near-tie step ⇒ forward
  sharding perturbation is benign, not structural.
- Multi-seq prefill branch: single-prompt decode (`query_start_loc=[0,1]`,
  n_active=1) takes the single-seq branch (correct); multi-seq is batch>1 only.
- Single-seq wrapper `deepseek_v4.py:1812-1871`: mirrors the repro, looks right.
- Runner KV threading: `tpu_runner.py:869` captures & stores the returned
  `kv_caches` each step; prefill→decode handoff fine; donation aliasing correct.

**Therefore S1 is NOT reproducible with random weights / the functional path,
and is almost certainly REAL-CONFIG / REAL-WEIGHT specific or a runtime-input
issue** (key point: with confident trained logits, ~1e-3 reduction noise can't
flip argmax, so S1's collapse is a LARGE structural error the repro would have
shown if it were in the decode math — it isn't). **Remaining suspects:**
1. **Real-config-only paths**: real V4 = 61 layers, 256 experts, real
   `compress_ratios` per layer (my repro: 4 layers, 8 experts, ratios
   (0,0,4,128)), real `state_max_seq_len`. A bug in a compress_ratio/layer
   combo or at scale won't show in the 4-layer truncation.
2. **Runtime attention_metadata**: is `seq_lens[0]-1` (start_pos) actually the
   right position each decode step at runtime? Is `state_max_seq_len`
   (`v4_state_max_seq_len_from_vllm_config`) what the buffers were sized for?
3. **Non-determinism source** (S1 is non-det at temp=0; repro is deterministic):
   uninitialized HBM read / non-det collective / cross-request buffer reuse.

**NEXT: an INSTRUMENTED smoke (real weights) — the only thing left that can
show S1.** Good news: `V4_DECODE_NAN_TRIPWIRE=1` is ALREADY plumbed to workers
(smoke.sh copies it) and wired at ~30 sites incl. all 6 state fields
`_at_entry`/`_post_write` per layer, each printing `pos={start_pos}` + nan/inf/
`max_abs`. So enabling it gives the start_pos trajectory (must be P, P+1, P+2…)
AND the per-field magnitude trace with ZERO code change. Run a SHORT gen
(max_tokens~8, S1 collapses in 2-3 tokens) to keep the trace small; read the
actual decode text (attractor vs coherent); + 3× Paris determinism probe.
CAVEAT: the tripwire uses jax.debug.print host callbacks, which prior history
says can perturb SPMD reduction order — if S1 *vanishes* under it, that's itself
a clue (callback-sensitive). Localize: which (layer, field) `max_abs` blows up
at the collapse step; is `pos=` correct each step.

> **INFRA BLOCKER — ROOT CAUSE FOUND + FIXED (2026-05-24):** `vllm`'s
> `ray.init()` dies "Version mismatch: cluster 2.54.1 vs local 2.55.1" at engine
> init (NOT S1). The prior theory (stale `/tmp/ray-vllm` / reinstall the venv)
> was **WRONG**: all 8 venvs are uniformly 2.55.1/commit `237c2455` (verified
> via `compute_version_info()` — the exact fn the check uses — and a single
> `ray-2.55.1.dist-info`); a clean `/tmp` wipe + `ray_restart` did NOT fix it.
> **The real culprit is mark's `node` Docker container** (`vllm/vllm-tpu:nightly`,
> ray **2.54.1**/commit `8768a329`). When it's redeployed it tries to join our
> ray at `:6379`, FAILS its own version check (status `Exited (1)`), but in the
> attempt calls `put_cluster_metadata(overwrite=True)`, **poisoning the GCS
> `CLUSTER_METADATA` key to 2.54.1** — which then makes our own 2.55.1 head log
> the right version (`gcs_server_main.cc:98 ray_version=2.55.1`) yet the stored
> metadata read by every client/worker say 2.54.1. It also blocks our own
> workers from joining (only 2/8 registered until fixed). Confirm the diagnosis:
> `GcsClient(address=head:6379).internal_kv_get(b"CLUSTER_METADATA", namespace=b"cluster")`
> (raw GcsClient bypasses the version gate) → look at `ray_version`/`git_commit`.
> **FIX (what worked):** (1) `sudo docker rm -f node` on all 8 hosts (removes the
> poisoner; `docker update --restart=no` + `stop` is NOT enough — a controller
> redeploys it); (2) either re-stamp the key
> (`gc.internal_kv_put(b"CLUSTER_METADATA", json_with_ray_version_2.55.1, True, namespace=b"cluster")`)
> OR just re-run `scripts/full_slice_v4_ray_restart.sh` now that no poisoner is
> present (head writes 2.55.1 natively). Verify: `ray.init(address='auto')`
> connects, `cluster_resources()['TPU']==32.0`. **Silver lining:** now that our
> cluster is 2.55.1, the 2.54.1 `node` container CANNOT successfully join (it
> Exits 1 on the mismatch), so it can't hold the TPU — residual harm is only the
> metadata poisoning, fixed by keeping `node` removed. KEEP the guardian running
> (`scripts/full_slice_v4_node_guardian.sh`; kill via
> `pkill -f "[f]ull_slice_v4_node_guardian"`) so a redeploy is caught fast; the
> teardown helper is `/tmp/v4_teardown.sh` (run as a FILE to avoid `pkill -f`
> self-match — the pattern string in an inline `pkill -9 -f gcs_server` matches
> the executing shell's own argv and kills it).

(Production V4 runs attn_dp=32: num_kv_heads=1 + bf16 ⇒ TP folds entirely into
attn_dp, model=expert=1, KV `P()`. The repro used attn_dp=8 / 8 experts.)

## PHASE 3 — S1 IS A DECODE-PATH BUG (real-weight smoke, 2026-05-24)

The instrumented smoke (`V4_DECODE_NAN_TRIPWIRE=1`) ran clean on real weights and
**localized S1 to the decode path.** Decisive findings:

* **PREFILL IS HEALTHY.** Pure single-token probes (`max_tokens=1`, ZERO decode
  steps) are all correct: France→`Paris`, Japan→`Tokyo`, hot→`cold`,
  hydrogen→`oxygen`, George→`Washington`, violets→`blue`. So the weights, config,
  embedding, MoE routing, sparse-attention SELECTION and the whole forward are
  correct **in the prefill path**.
* **DECODE COLLAPSES.** With `max_tokens≥2` the output degenerates by token 2-3
  into a repeating attractor (Mars→`' "The first thing that is a good and the
  first thing'`; Paris→`' Paris, 2000, 2000, 2000'`). The collapse appears the
  moment decode steps (which reuse the threaded KV + compressor/indexer/SWA
  decode-state) run instead of recomputing over the full sequence.
* **Tripwire trace is BENIGN at the field level:** finite everywhere (NO nan/inf
  in decode), `pos=` correct each step (11,12,…), all 6 state fields + per-layer
  activations finite and varying with reasonable magnitude. The `-inf` in
  `compressor_score`/`indexer_score` is by-design masking that correctly
  *decreases* as positions fill. So S1 is **wrong FINITE values from attending to
  the wrong context / mis-threaded state**, not a blowup — exactly why the
  nan-tripwire alone never caught it.
* **Non-determinism is a DOWNSTREAM symptom, not the bug.** It survives the
  tripwire host-callbacks (so it's not merely debug-perturbable reduction order)
  and only appears at LATE tokens (first ~6 tokens identical across temp=0 runs,
  then split) — i.e. once decode is already in a flat-logit degenerate regime,
  32-way reduction noise flips near-ties. The PRIMARY bug is the **collapse**.

**This OVERTURNS the PHASE-2 "decode math is correct" conclusion** — that was
proven only on the truncated random-weight repro (4 layers, ratios (0,0,4,128),
8 experts, short seq, attn_dp=8). The real-config decode path (61 layers, real
per-layer `compress_ratio`/`window_size`, real `state_max_seq_len`, attn_dp=32)
IS broken. So S1 is a **real-config-only decode bug**: prime suspects are the
sparse-attention index selection at decode (`get_window_topk_idxs_decode` /
`get_compress_topk_idxs_decode` / indexer `compress_topk` in
`deepseek_v4_attention.py:~750-790`), the SWA ring-buffer wrap
(`kv_cache.at[:, start_pos % win]`) when `win < seq_len`, or a prefill→decode
seed slot-layout mismatch (`attention_init_state_from_prefill` /
`_compressor_state_from_prefill`) that the short-seq/ratio-0 repro never exercised.
Confirm with a teacher-forcing comparison (decode trajectory vs re-prefill every
token) — re-prefill-every-token should generate COHERENTLY since prefill is healthy.

## PHASE 4 — S1 REPRODUCED ON CPU (peaked weights) — fast iteration unlocked

**The runbook's "CPU can never reproduce S1" was WRONG** — it was an artifact of
the repro's `normal*0.02` weights, which make the compressor/indexer internal
softmaxes ~UNIFORM and average out the decode-vs-prefill discrepancy. With
**peaked** weights the bug surfaces on CPU, eager, single-device (NO sharding /
NO reduction noise → a genuine deterministic decode-math discrepancy):

* `scripts/s1_cpu_repro_peaked.py <scale> <n_layers> <T> <N> <seed>` — same
  truncated cfg as `s1_cpu_repro_v4flash.py` but weights `normal*scale`, eager
  only, compares decode argmax vs the fresh-prefill (`transformer_body_forward`)
  reference. `scale=0.02`→bad=0/12 (matches old runbook); **`scale=0.5`→bad=3/12**
  (S1). `scale=1.0`→0 (logits saturate). ~23s/run.
* Structural confirmation (`/tmp/s1_structural_check.py` logic): at scale=0.5,
  worst decode steps have **||h_dec−h_pre||/||h_pre|| = 0.20–0.41** (vs 0.004 at
  scale=0.02) — a LARGE hidden-state divergence, not a near-tie float flip. This
  IS S1's class of bug.

**Localization so far:** the bug is NOT in `deepseek_v4_attention.py` — an
isolated single-layer attention parity test (prefill seed+decode_step vs
reference) stays at relErr ~3e-3 even under peaked weights. The structural error
only appears in the FULL decode path `deepseek_v4_run_with_decode_state`
(`deepseek_v4.py`): `_pack_layer_state`/`_unpack_layer_state`, the MoE decode,
and/or multi-layer state threading. **USE THE CPU REPRO to bisect which
layer/component first diverges** (compare per-layer decode h vs prefill h) — no
smoke needed until final closure.

## PHASE 5 — bug is in the COMPRESSION-LAYER decode INTEGRATION (2026-05-24)

Airtight: decode is broken even on UNAMBIGUOUS factual prompts (no greedy-loop
confound) while prefill is perfect:
* prefill `max_tokens=1`: France→Paris, Japan→Tokyo, hot→cold, hydrogen→oxygen,
  George→Washington, violets→blue — all correct.
* decode `max_tokens≥2`: `"first six primes are 2,3,5,"`→`' 0 0 0 0 0 0'`;
  `"Count from 1 to 20: 1,2,3,"`→`' '`; `"Days: Monday, Tuesday,"`→
  `' Wednesday, 2012-12-19 12:'` (note: `Wednesday` = correct 1st decode token,
  THEN collapses). So decode collapse is real and immediate; prefill is healthy.
  (The earlier "story prompt pure-prefill loops too" was a weak greedy-raw-
  completion artifact, NOT the bug — disregard it.)

**CPU bisection (scripts/s1_cpu_repro_peaked.py, scale=0.5, ~10-23s each):**
* n_layers 1,2 (ratios (0)/(0,0), pure SWA) → bad=0. SWA + MoE decode are FINE.
* n_layers 3 (adds ratio=4 layer) → bad=1; n_layers 4 (adds ratio=128) → bad=3,
  with worst-step hidden-state ||h_dec−h_pre||/||h_pre|| = **0.20–0.41** (vs
  0.004 baseline) = structural, not a near-tie flip.
⇒ **The bug is introduced by the ratio=4 / compression layer's decode.**

**RULED OUT (isolated CPU tests, peaked weights, all relErr≈0.000):**
* Main compressor: `compressor_prefill` vs zero-state incremental
  `compressor_decode_step` → byte-identical (`/tmp/comp_parity.py`).
* Prefill→decode SEED: `_compressor_state_from_prefill`+decode vs prefill →
  byte-identical, incl. in-progress-window remainder (`/tmp/seed_parity.py`).
* pack/unpack: every field's actual shape == `_layer_decode_state_layout` shape
  (`/tmp/s1_shape_check.py`, no mismatch).
* Isolated single-layer attention parity (subagent): relErr ~3e-3 even peaked.
* Window/SWA (n_layers 1,2 clean), donation (PHASE 2), runner KV threading.

**THEREFORE** the error is in the FULL-LAYER decode INTEGRATION that the
component tests don't exercise — i.e. inside `attention_decode_step` for ratio>0
layers: how window-topk ∪ indexer `compress_topk` feed `sparse_attn` over the
combined ring-buffer+compressed `new_kv_cache`, OR the indexer's own
state/score path, OR a freqs (swa vs `compress_rope_theta` `comp`) dispatch
difference between the prefill and decode call sites.

**NEXT (do this first, on CPU, ~seconds):** write the decisive INTEGRATION test
— for the truncated cfg's ratio=4 layer (`params.layers[2].attn`), compare
`attention_prefill(x[:M])[:,P]` vs `attention_init_state_from_prefill(x[:T])` +
`attention_decode_step` stepped to position P, at peaked weights (scale 0.5).
Mind which freqs each call site receives (swa vs comp) — `transformer_body_forward`
is the source of truth for per-layer freqs dispatch. If `y` diverges → bug is in
`attention_decode_step` integration (likely the topk/sparse_attn assembly or the
indexer `compress_topk`, esp. the prefill `K=min(index_topk, S//ratio)` vs decode
`K=index_topk` asymmetry at L412 vs L600); if `y` matches → look at MoE/residual
across the multi-layer stack. Then fix, confirm on `s1_cpu_repro_peaked.py 0.5`
(bad→0) AND `s1_cpu_repro_v4flash.py both 8 12 4` (no regression), THEN one smoke.

## PHASE 6 — the CPU PEAKED repro is a RED HERRING; decode math is fp32-EXACT (2026-05-24)

**PHASE 4/5 are OVERTURNED.** Three new CPU diagnostics (committed) settle it:
* `scripts/s1_cpu_integration_test.py` — isolated SINGLE-LAYER attention decode vs
  prefill, ratio=4 layer, peaked scale 0.5, bf16. **MATCHES** (relErr ~1e-3, bf16
  noise). So the bug is NOT an isolated `attention_decode_step` integration bug —
  directly refutes the PHASE-5 "ratio=4 attention integration is broken" claim.
* `scripts/s1_cpu_layer_bisect.py` — per-layer teacher-forcing (real threaded
  activations, `block_forward` vs `block_init_state_and_forward`+`block_decode_step`).
  Divergence is **SCATTERED across ALL layer types** incl pure-SWA L1 (0.194 @pos14);
  "first diverging layer" VARIES by position (L3@9, L2@11, L1@14). With win=128 and
  positions<128 there is NO ring wrap and ratio=128 has ZERO compressed tokens in
  range, so L3 is effectively pure-SWA too. Pure-SWA L1 diverging CONTRADICTS the
  PHASE-5 "n_layers 1,2 clean" claim (that was argmax-level only; hidden h DOES
  diverge there).
* `scripts/s1_cpu_dtype_disambig.py` — reruns the bisect in pure fp32 vs bf16.
  **DECISIVE: fp32 worst relErr = 0.00026 (0/12 positions); bf16 worst = 0.227
  (5/12).** In fp32 decode is bit-exact equal to prefill.

**CONCLUSION: the peaked-repro divergence is bf16 ROUNDING amplified by RANDOM-weight
near-ties** — non-hash MoE top-k routing flips (`top_k` over near-tie expert scores,
`deepseek_v4_moe.py:98`) and the swiglu `±10` clip boundary (`expert_forward:128-129`),
both random-weight artifacts. The decode ALGORITHM is correct (fp32-exact == prefill,
confirming PHASE-2's sharded byte-identical finding from another angle). **`s1_cpu_repro_peaked.py`
does NOT capture real S1** and `bad>0` there is NOT a valid fix target. Real trained
weights give CONFIDENT logits, so symmetric bf16 rounding-order noise CANNOT produce
the real HARD collapse ("0 0 0 0"). **STOP iterating on the random-weight CPU repro.**

**Therefore real S1 lives in what the standalone repro does NOT exercise:**
1. **The vLLM RUNTIME integration** (PRIME suspect — the repro hand-feeds the correct
   `start_pos`, so a runtime start_pos / attention_metadata / kv-threading bug is
   INVISIBLE to every CPU/MH repro done so far). Audit `_maybe_set_v4_decode_start_pos`
   (`runner/tpu_runner.py`) and `_initialize_kv_cache_deepseek_v4`
   (`runner/kv_cache_manager.py`): is `start_pos` the right absolute position each
   decode step? does prefill→decode hand off the right position? is `state_max_seq_len`
   what the buffers were sized for?
2. **Real config**: 61 layers, 256 experts, full per-layer `compress_ratios` pattern,
   real `state_max_seq_len` / `index_topk` vs seq_len — a combo the 4-layer truncation
   never hits.

**NEXT: audit the runtime start_pos/metadata path (free, no smoke), and probe the live
instrumented smoke on :18081 for the real collapse + start_pos trajectory.** A faithful
repro needs REAL WEIGHTS (random weights are proven inadequate) — consider loading a few
real V4-Flash layers on CPU (~9 GiB/layer) if the runtime audit doesn't pin it.


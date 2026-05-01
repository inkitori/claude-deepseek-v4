# claude-deepseek-v4 — agent runbook

You're picking up a TPU-inference effort: get
`vllm serve deepseek-ai/DeepSeek-V4-Flash` deployable on a
**v6e-32 TPU slice** (TP=32, 8 hosts × 4 chips) at production
quality. The model is a flagship MoE — 256 FP4 experts + MLA-
shaped attention + dense FP8, ~543 GiB bf16-expanded.

The non-negotiable goal is **fast, mathematically correct
inference with the real V4-Flash weights**, served via the
OpenAI-compatible HTTP endpoint to many concurrent users.

**State of the system (read before claiming progress):** the
basic ` Paris` smoke probe passes on a fresh engine. **Anything
past 1 generated token is broken** — the `LONG_GEN_REQUIRED=1`
gate has never passed end-to-end. See S1 below.

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

#### S1. Decode-state generation produces empty/garbage past pos ~2

**The single most important item.** All other backlog work is
blocked on this.

`__call__` always routes through `deepseek_v4_run_with_decode_state`
(threading per-layer `AttentionDecodeState` through
`kv_caches`). Tiny-config tests pass; real-V4 generation
fails the moment a real decode step runs.

**Reproducer (real V4, fresh engine):**
```
prompt: "Tell me a short story about a robot exploring Mars:"
max_tokens=64, temperature=0, seed=0
→ completion_tokens=64, finish_reason=length, visible_words=1
→ visible text: ' This' (1 word; 63 tokens decode to empty)
```

The basic Paris probe (`max_tokens=8` on
`"The capital of France is"`) returns " Paris" — the FIRST decoded
token is correct. But `usage.completion_tokens=8` while visible
text is just " Paris": the trailing 7 decode steps emit pad/control
tokens that decode to empty strings. `LONG_GEN_REQUIRED=1`
(default-on) catches this via the `visible_words >= 10` floor.
**Don't trust `usage.completion_tokens` and don't trust
`max_word_run`/`ends_clean`** — those metrics all read healthy
on the corrupted output (pinned by `long_gen_required_invisible`
in the harness self-test).

**Iter 2026-05-01 (HEAD):** one real NaN source closed; user-visible
bug still red.

`_v4_force_kv_caches_read` was folding `b + opaque_zero * kv` to force
XLA to emit a real read of donated `kv_caches`. Compressor / indexer
`score_state` init slots hold `-inf` (intended; softmax-zeros them).
IEEE: `0.0 * -inf = NaN`. Every decode step poisoned its output buffer
at every -inf slot. CPU localize probe (`/tmp/test_v4_decode_nan_localize.py`):
prefill ratio>0 layers had 256/512 -inf slots (correct); after decode
step 0 those flipped to 256/512 NaN slots; by step 3 NaN reached `h`.

Fix: replace `b + opaque_zero * kv` with `where(opaque_false, b + kv, b)`.
The where reads `kv` (HLO retains it; donation alias marker preserved
per `/tmp/test_v4_force_read_alt_hlo.py`) but `b + (-inf) = -inf`
(finite, never NaN) and runtime selects `b`. Pinned by
`test_run_with_decode_state_does_not_propagate_nan_through_kv_caches`
(13/13 in `TestPackedDecodeStateBuffer`).

**Real-V4 smoke after fix:** basic Paris R1==R2 green; `LONG_GEN_REQUIRED=1`
still red with the same symptom (`completion_tokens=64 visible_words=1`).
`logprobs` at `max_tokens=1` is finite (-1.066 for " Paris"); at
`max_tokens=2+` HTTP 400 "nan". So **decode step 0** is still producing
NaN h on real V4 — separate from the kv_caches contamination my fix
addressed. Tiny CPU does NOT reproduce (orchestrator over 6 layers /
random bf16 weights / 4 decode steps stays NaN-free).

Remaining hypotheses, in order of likelihood:
1. **Quantization-specific NaN at decode step 0.** Real V4 uses FP4
   experts and FP8 attention; tiny config uses bf16 throughout. A
   dequant kernel may produce NaN on a path decode hits but prefill
   doesn't (e.g. single-token dequant tile vs multi-token, or zero
   activation tiles). Probe: read smoke logs for FP8/FP4 dequant warnings,
   or instrument a `jax.debug.print(jnp.any(jnp.isnan(...)))` at the
   block-output boundary inside `block_decode_step`.
2. **SPMD partitioner divergence in `attention_decode_step` or
   `compressor_decode_step`.** Real V4 runs with
   `attention_data_parallelism=32`; tiny CPU is fully replicated.
   `lax.dynamic_update_slice` semantics under sharded `start_pos`,
   or `top_k` over a -inf-padded `index_score`, may differ.
3. **HC head / sigmoid saturation on real-V4 weights.** `head_hc`
   does `sigmoid(mixes * hc_scale + hc_base) + hc_eps`; if `mixes`
   has extreme values, sigmoid grad overflows / underflows. Decode-
   only because decode's `x_step` shape is `[B, 1, hc, D]` vs prefill's
   `[B, T, hc, D]` — accumulation order differs.
4. **SWA wraparound under traced `start_pos`:** off-by-one in
   `lax.dynamic_update_slice` for slot `start_pos % win`.

**Runtime-hook locations** (read before touching plumbing):
* `models/common/model_loader.py` — V4 `kv_cache_sharding=P()`.
* `runner/kv_cache_manager.py::_initialize_kv_cache_deepseek_v4`.
* `runner/tpu_runner.py::_maybe_set_v4_decode_start_pos`.
* `models/jax/deepseek_v4.py::v4_state_max_seq_len_from_vllm_config`.

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

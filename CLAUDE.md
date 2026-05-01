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
| `V4_DECODE_NAN_TRIPWIRE` | `0` | Per-sub-block NaN/Inf logger inside `block_decode_step` (S1 diagnostic). When `1`, every decode emits `[v4nan] L{i} pos={p} {name}: nan=N +inf=N -inf=N` per layer. Read at module import; no-op when off (HLO byte-identical). |
| `V4_WEIGHT_NAN_AUDIT` | `0` | One-shot finiteness audit of the loaded V4 param tree (S1 hyp 1: bad FP4/FP8 dequant on a single layer). When `1`, `load_weights_from_dir` emits `[weight_nan] {path}: nan_any=B inf_any=B` per non-finite leaf + a `[weight_nan_audit] examined=N nan_leaves=N inf_leaves=N` summary right before forward starts. Read on every load; loader code-path unchanged when off. |

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

**Status (2026-05-01 v6): NaN/Inf RULED OUT. The bug is logic-level,
not numerical.** Smoke `20260501T083332Z` (TRIPWIRE=1, all 6
AttentionDecodeState fields probed at decode entry) emitted 13,583
`[v4nan]` lines and EXACTLY ZERO of them have `nan>0`. Every input
field to every decode step at every layer at every position is
finite. kv_cache_at_entry max_abs grows monotonically over decode
positions (5.16 → 5.34 → 5.38 ...), confirming new tokens ARE being
written. Yet LONG_GEN still produces 1 visible word of 64 tokens.

So the next iter must move PAST NaN debugging entirely.

`V4_DECODE_NAN_TRIPWIRE=1` runs:
* 9 decode-step inner probes (`kv_cache_at_entry`, `qr_postnorm`,
  `q_postrsqrt`, `q_postrope`, `kv_postrope`, `kv_cache_post_write`,
  `sparse_attn_o`, `o_post_inv_rope`, `wo_b_y`)
* 5 decode-entry probes for the OTHER state fields
  (`compressor_kv_at_entry`, `compressor_score_at_entry`,
  `indexer_kv_at_entry`, `indexer_score_at_entry`,
  `indexer_kv_cache_at_entry`)
* 3 prefill→buffer probes (`prefill_state_kv_cache`,
  `packed_buffer_post_pack`, `packed_buffer_post_force_read`)
* 8 inner prefill-init probes inside `attention_init_state_from_prefill`

All share the same `V4_DECODE_NAN_TRIPWIRE` gate; HLO unchanged when
off. Pinned by `test_decode_nan_tripwire_when_enabled_runs_clean`.
**Tripwire-on breaks Paris determinism** (callback side effects
prevent some XLA optimizations); use ONLY for NaN localization, not
for end-to-end gate validation.

**What's ruled out (cumulative):**
* Bad weights (`V4_WEIGHT_NAN_AUDIT`).
* rsqrt / sparse_attn / inverse-RoPE numerics.
* `attention_init_state_from_prefill` math (inner probes clean).
* `_pack_layer_state` / `_v4_force_kv_caches_read` (no propagation).
* kv_cache field aliasing (v5 concat fix).
* ANY NaN/Inf anywhere in the decode pipeline (smoke 083332Z).

**What's left — strictly logic-level hypotheses:**
1. **Sampler / lm_head**: model emits finite logits, sampler picks
   pad/EOS/control tokens. Add a probe on `hidden_TD` (output of
   `__call__`) and on the post-`compute_logits` array — measure
   argmax id and top-3 logit values per decode step. The
   `completions text` shows real first token then pad; the logit
   trajectory will show whether the first decode genuinely picks
   the right token via argmax and subsequent ones converge to a
   pad-token-favoring fixed point.
2. **Stale residual stream**: if the per-decode-step h doesn't
   actually advance (the kv_cache write happens but the attention
   READ of new positions is wrong), the residual stream stays
   essentially fixed and logits collapse to a vocabulary-prior
   fixed point that often favors special tokens. Probe `h` (the
   per-layer post-MoE output) at decode pos 5 and pos 11; it should
   be substantively different. If max_abs(h_pos11 - h_pos5) ≈ 0,
   the bug is in attention READS not writes.
3. **start_pos plumbing**: observed positions in smoke 083332Z were
   5,6,7,...11 (correct: prompt has ~5 tokens, max_tokens=8 → 7
   decodes). So start_pos is incrementing. But maybe the WRITE
   slot is wrong: SWA writes go to `start_pos % win`, compressor
   writes at `start_pos // ratio`. If those are off-by-one, attention
   reads stale slots. Add a probe that prints `start_pos`, `start_pos
   % win`, and `(start_pos+1) // ratio` per layer per step.
4. **vLLM-side cache management**: `_initialize_kv_cache_deepseek_v4`
   creates one `kv_caches` list at engine init and never resets per
   request. If two unrelated requests collide on the same buffer
   (which they shouldn't — vLLM's scheduler runs one request at a
   time on V4 due to MAX_SEQS=1), there's no issue. But verify the
   donation chain: after a request completes, what holds the
   `kv_caches[i]` reference, and does the next request's prefill
   actually overwrite it? `runner/kv_cache_manager.py` line 884.

**Real-V4 verification:**
* Localize the bug: TRIPWIRE=1 + add a probe on `h` and on logits.
* End-to-end gate (default-on): `LONG_GEN_REQUIRED=1
  scripts/full_slice_v4_smoke_check.sh`. Pre-fix and post-fix both
  produce `visible_words=1` of 64 tokens. The closed gate is
  `visible_words >= 10`.

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

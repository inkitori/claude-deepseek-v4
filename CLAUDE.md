# claude-deepseek-v4 — agent runbook

You're picking up a TPU-inference effort: get
`vllm serve deepseek-ai/DeepSeek-V4-Flash` deployable on a
**v6e-32 TPU slice** (TP=32, 8 hosts × 4 chips) at production
quality. The model is a flagship MoE — 256 FP4 experts + MLA
attention + dense FP8 — about 543 GiB bf16-expanded.

The smoke gate is GREEN: `/v1/completions` returns
deterministic `Paris` for "The capital of France is", cold
compile ~97s warm-cache curl sub-second. The current work is
converting that demo into something that handles real
concurrent OpenRouter-grade traffic without silent correctness
regressions. The non-negotiable goal is **fast, mathematically
correct inference with the real V4-Flash weights**, served via
the OpenAI-compatible HTTP endpoint to many concurrent users.

## Discipline (READ FIRST)

### 1. Minimum-delta rule

The overall delta from upstream `tpu-inference` should be **as
small as possible** while keeping math correct and serve fast.
Every line you add is a line the next agent has to read and
rsync to 8 worker hosts.

* **Don't add files when an existing one fits.** V4 already
  lives in `models/jax/deepseek_v4.py`,
  `models/jax/deepseek_v4_loader.py`,
  `layers/jax/attention/deepseek_v4_attention.py`,
  `layers/jax/moe/deepseek_v4_moe.py`.
* **Don't add new test classes for variants of an existing
  case.** Parametrize or fold in.
* **Reuse upstream layers** (`dense_moe_fwd`, `sparse_attn`,
  `rms_norm`, `megablox/gmm`, `ragged_paged_attention/v3`)
  before writing V4-specific copies.
* **Touch `runner/` / `worker/` / `platforms/` only as a last
  resort.** The two existing V4 runtime hooks
  (`kv_cache_manager.py`'s `deepseek_v4` model_type override,
  the `deepseek_v4_fp8` quant stub in `tpu_platform.py`) are
  the entire delta and should stay that small.
* **Delete dead code as you go.** Resolved TODOs, superseded
  "tier"/"keystone" comments, doc-stub markdown files
  (PROGRESS / SUMMARY / STATUS were all that pattern; they
  shouldn't reappear).

When in doubt, the smaller change wins.

### 2. CLAUDE.md is for durable knowledge — `git log` is for narrative

**Do NOT append iter-by-iter narrative to this file.** Per-
session decisions, hypothesis-by-hypothesis history, "iter 5h
tried X and reverted" — all that lives in commit messages.
`git log --grep='S1 iter-'` reconstructs the trail.

Update CLAUDE.md only when you've learned something **durable**:
a new pitfall, a corrected env knob, a topology change, a
backlog reordering. If something was true for one iter and
isn't anymore, remove it. The file targets ~500 lines; if it's
ballooning, prune before adding.

### 3. Code style matches upstream — this work targets upstream PR

V4 source must be **indistinguishable in style** from peer models
(`qwen3.py`, `deepseek_v3.py`, `llama3.py`). Minimum-delta covers
quantity; this covers style.

* **Docstrings**: brief Args/Returns. No multi-paragraph rationale,
  no "previous implementation was X" history.
* **Comments**: only when WHY is non-obvious. No iter-narrative
  ("iter-5h fix", "Bug A was…"), no section banners
  (`# ===== gate =====`).
* **Defensive programming**: validate only at the vLLM API
  boundary, not at every helper. Peer models don't.
* **No backwards-compat shims**: V4 is new code; delete dead
  branches.

Before declaring a piece "done", 30-second diff against
`qwen3.py` / `deepseek_v3.py`. If chattier, prune.

## Cluster topology

* Slice: **v6e-32** = 8 hosts × 4 chips × 32 GB HBM = 992 GiB total.
* Head: `10.164.0.41` (also `worker 0` — launch from here).
* Workers: `10.164.0.{22, 35, 36, 39, 45, 18, 30}` (worker 1–7).
* Each host is a **separate VM with its own clone of this
  repo**. There is **no shared filesystem**. A `git push` from
  the head does not propagate to workers — see "Source sync"
  pitfall.
* Ray address: `10.164.0.41:6379`. Bootstrapped from
  `scripts/full_slice_v4_ray_restart.sh`. `ray status` should
  show 8 nodes and `0.0/32.0 TPU` when idle.
* Shared venv path: `~/claude-deepseek-v4/work/vllm_env` on
  every host (mirrored once at bootstrap).
* SSH keys: `~/.ssh/google_compute_engine` (cross-host SSH
  within the slice), `~/.ssh/id_ed25519` (`git push` to GitHub).
* GCS-mounted weights: `~/.cache/huggingface/hub` resolves to
  the staged HF cache layout under `gs://<bucket>/<dir>/` via
  `scripts/mount_gcs.sh`. Required for `vllm serve` (no
  internet).

## The iterate loop

Every change-then-test cycle is exactly:

```bash
scripts/full_slice_v4_reset.sh        # cluster cleanup; safe to re-run
scripts/full_slice_v4_sync.sh         # rsync source to all 7 worker hosts
scripts/full_slice_v4_smoke.sh        # launch vllm serve; writes pid + log
scripts/full_slice_v4_smoke_check.sh  # validate /v1/completions when ready
```

* `reset.sh` — clears 4 orphan-state surfaces (api-server pid,
  `VLLM::EngineCore` Ray actors, `/tmp/libtpu_lockfile`, leaked
  Ray placement groups). Doesn't restart Ray. Run before every
  smoke launch.
* `ray_restart.sh` — heavier nuke (`ray stop --force` + fresh
  `ray start` on all 8 hosts). Use only when reset isn't enough.
* `sync.sh` — **mandatory after any code edit**. Rsyncs only
  `work/tpu-inference/tpu_inference/` and `scripts/` to workers;
  the repo-root markdown files are head-only.
* `smoke.sh` — launches `vllm serve` (background); writes pid +
  timestamped log under `logs/`.
* `smoke_check.sh` — polls `/v1/models` until ready, fires
  deterministic Paris completion ×2, asserts byte-identical +
  contains "Paris". Optional probes gated by `*_REQUIRED=1`
  env knobs (see knob table for exit codes 4–13). Self-test at
  `scripts/test_smoke_check_harness.sh` (28 mock scenarios).
* `warm_cache.sh` — smoke + check + cleanup, used once on a
  fresh VM to populate the JAX compile cache.

Pass criterion: `smoke_check.sh` exits 0 with
`PASS: deterministic completion contains 'Paris'` AND any
`_REQUIRED=1` probes pass.

## Killing vLLM cleanly

`vllm serve` forks an EngineCore that spawns Ray actors on each
worker. SIGKILL'ing the api-server doesn't reap them; they hold
TPU + libtpu state. Always use `scripts/full_slice_v4_reset.sh`
— it kills by exact `comm` match (never a broad regex; see
pitfall #1). Escalate to `full_slice_v4_ray_restart.sh` only
when reset isn't enough.

## Optimization knobs

Set on the launching shell; the smoke script forwards them to
Ray workers via `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`.

| env var | default | what it does |
|---|---|---|
| `MAX_LEN` | `256` | `--max-model-len`. Production needs this lifted (A1); each step up surfaces fresh activation-HBM tightness. |
| `MAX_SEQS` | `1` | `--max-num-seqs`. Production needs this lifted (S2/A1). Bumping without S2 won't help throughput (sequential Python loop). |
| `V4_LOADER_SLICE_AWARE` | `1` | Each host reads only the rows its local devices own. |
| `V4_LOADER_PLACE_WORKERS` | `8` | Placement threads per host. Most per-tensor work releases the GIL. Set to `1` for parity testing. |
| `V4_LOADER_PREFETCH_WORKERS` | `0` | Thread-pool prefetch in non-slice-aware iterator. Empirically not useful; retained for future work. |
| `VLLM_XLA_CACHE_PATH` | `~/.cache/vllm/xla_cache` | Per-host JAX persistent compile cache. tpu_inference's `compilation_manager.py:53` overrides any `JAX_COMPILATION_CACHE_DIR` env var. **Not GCS** — bucket is shared, do not relocate without explicit user authorization. **Cross-host rsync is unsound** (verified: same filename, 8 distinct sha256s — SPMD compiles host-specific binaries). |
| `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES` | `0` | Cache even small modules (JAX 0.9 config name). |
| `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` | `0` | Cache even fast-to-compile modules. |
| `RAY_CGRAPH_get_timeout` | `3600` | Ray compiled-graph channel timeout. Default 300 trips during first inference if `jit_run_model` recompiles. Don't lower. |
| `V4_XLA_FLAGS` | unset | Opt-in custom `XLA_FLAGS` for one launch. The smoke script does **not** inherit `XLA_FLAGS` from the parent shell (see pitfall #4). |
| `V4_DECODE_STATE` | `0` | `1` routes `__call__` through `deepseek_v4_run_with_decode_state` (real packed-state decode); `0` is the prefill-recompute baseline. **Default `0` because `1` produces empty/garbage output past position ~2 on real V4 — see S1.** Set `1` to reproduce / debug the broken path. |
| `CHAT_REQUIRED` | `0` | smoke_check chat probe; exit 4 on missing/empty content. Adds ~30 s for first-chat OOM-retry (pitfall #9). |
| `REASONING_REQUIRED` | `0` | Thinking-mode chat probe; exit 5 on empty `message.reasoning`. Pins S3 runtime. Currently fails under both `V4_DECODE_STATE=0` (prefill-recompute, all-newlines) and `=1` (decode plumbing, empty/garbage past token ~2 — see S1). |
| `STREAMING_REQUIRED` | `0` | SSE byte-equality probe vs non-streaming; exit 6. Pins S7. |
| `SAMPLING_REQUIRED` | `0` | `temperature=0.7, top_p=0.9, frequency_penalty=0.1`; exit 7 on empty/invalid. Pins S6. Determinism not asserted (per-request seed unsupported on non-greedy paths). |
| `STOP_REQUIRED` | `0` | `stop=["Paris"]`; exit 8 if response contains Paris or `finish_reason!=stop`. Pins S6. Baseline emits ` Paris` first, so a working handler MUST intercept. |
| `LOGPROBS_REQUIRED` | `0` | `logprobs=5`; exit 9 if any position has fewer than 5 alternatives. Pins S6. |
| `TOPK_REQUIRED` | `0` | `temperature=0.7, top_k=10`; exit 10 on empty/invalid. Pins S6. |
| `PRESENCE_REQUIRED` | `0` | `temperature=0.7, presence_penalty=0.5`; exit 11 on empty/invalid. Pins S6. |
| `N_REQUIRED` | `0` | `n=2`; exit 12 if fewer than 2 non-empty choices. Pins S6. Under `--max-num-seqs=1` likely sequentializes but cardinality must equal n. |
| `LONG_GEN_REQUIRED` | `1` | `max_tokens=64` long-answer probe; exit 13 if `completion_tokens<30`, any word repeats 5+ times in a row, or trailing 5 chars contain <2 alphanumerics. Pins S8. Logs observed tok/s. Default `1` because the basic Paris probe is too thin a gate. Override via `LONG_GEN_MAX_TOKENS`. |

## Production-readiness backlog

Pick the highest-leverage uncompleted item. Items earlier block
items later. Per-iter narrative belongs in commit messages; the
backlog itself stays brief.

### Tier S — silent correctness bombs (fix before perf)

#### S1. Real decode plumbing — RUNTIME HOOKS SHIPPED, GENERATION IS BROKEN

The runtime plumbing is complete and the JIT-cache structure is
right (one trace fits all decode positions). **But the actual
generation output is empty/garbage past position ~2.** Smoke
launcher reverted to `V4_DECODE_STATE=0` (prefill-recompute
baseline) — that path returns ` Paris` correctly but is also
unvalidated past 2 tokens.

**What we know (probed 2026-04-30 ~22:50Z, real V4 + traced
`start_pos`):**

```
prompt: "Tell me a short story about a robot exploring Mars:"
max_tokens=64, temperature=0, seed=0
→ completion_tokens=64, finish_reason=length
→ visible text: '#'   (1 character)
```

Equivalent reproducers across 4 unrelated prompts at
`max_tokens=8` → text=`''` (empty) for all of them. The same
engine PID returned the canonical ` Paris` text on the basic
probe early in its lifetime, then degraded to `'# '` for the
same Paris probe ~15 minutes later. So generation is broken AND
the engine state corrupts over the lifetime of a vllm serve
process.

**Plumbing that's actually shipped (runtime hooks, math
kernels, JIT cache structure):**
* Decode-step kernels (`attention_decode_step`,
  `compressor_decode_step`, `indexer_decode_step`) thread
  packed `AttentionDecodeState` through `kv_caches[i]` (one
  fp32 packed buffer per layer).
* `start_pos` is traced (constant output shapes via
  `lax.dynamic_slice_in_dim`, `-1` sentinel topk slots,
  constant `K=index_topk`); one JIT trace per shape bucket.
* Tiny-config tests pass (`TestPackedDecodeStateBuffer`,
  `TestTransformerBodyDecodeRoundTrip`,
  `TestPrefillToDecodeStateParity`,
  `test_buffer_decode_jit_cache_hits_across_positions` —
  cache-size delta == 1 over 4 positions).
* `LONG_GEN_REQUIRED=1` gate in smoke_check (default-on)
  catches the empty-output failure mode end-to-end.

**Runtime hook locations (read before touching V4 plumbing):**
* `models/common/model_loader.py` — V4-only
  `kv_cache_sharding=P()` replicated override.
* `runner/kv_cache_manager.py::_initialize_kv_cache_deepseek_v4`
  — V4 packed-buffer allocator.
* `runner/tpu_runner.py::_maybe_set_v4_decode_start_pos` —
  sets `decode_start_pos` meta-gate on `AttentionMetadata`.
* `models/jax/deepseek_v4.py::v4_state_max_seq_len_from_vllm_config`
  — single source of truth for buffer sizing.

**Hypotheses for the empty-output bug** (untested as of writing,
investigate in priority order):
1. Request-state pollution: prior decode state isn't reset
   between requests, leaking into subsequent prefills (same
   shape as the iter-5g..5j Bug B saga; `_v4_force_kv_caches_read`
   was supposed to fix but may not survive traced `start_pos`
   refactor or `--enable-prefix-caching`).
2. Compressor compress-event branching via `lax.cond` doesn't
   produce identical state to the Python-static version on real
   weights (works on tiny config CPU, fails on real V4 / TPU SPMD).
3. SWA wraparound write under traced `start_pos` corrupts
   kv_cache for positions ≥3 (`lax.dynamic_update_slice` index
   computation off-by-one or sharding-dependent).
4. Indexer top-k mask under traced `start_pos` picks wrong
   positions; downstream attention reads corrupted slots.

**Validation discipline going forward:** any "S1 fixed" claim
must produce a `LONG_GEN_REQUIRED=1` PASS on a fresh vllm serve
PLUS the same probe re-fired after 5 unrelated requests (to
catch state-pollution bugs the single-request gate misses).
Don't trust `completion_tokens` from `usage` — the model
reports 64 even when visible text is 1 character.

S1 closure unlocks A1, B1, S5 — but those are blocked on real
generation working, not on more runtime plumbing.

#### S2. Multi-sequence dispatch is a Python loop in eager mode

`__call__` runs each active sequence sequentially through
`transformer_body_forward`. Production needs a ragged-batch
jit'd kernel — either extend
`kernels/ragged_paged_attention/v3` to support V4's top-k +
attn_sink + dual-buffer KV (DECISIONS D2), or jit V4's path
with `lax.dynamic_slice` per active seq. Until S2 lands,
`--max-num-seqs=1` is forced. Independent of S1 but they
multiply.

#### S3. Reasoning + tool parsers — wired, but reasoning output broken with S1

Smoke launcher passes `--reasoning-parser deepseek_v4`,
`--enable-auto-tool-choice`, `--tool-call-parser deepseek_v4`;
both register at startup. `REASONING_REQUIRED=1` currently fails:
both `V4_DECODE_STATE=0` (all-newlines) and `=1` (empty/garbage
past token ~2) produce un-usable reasoning. Hypothesized to fix
itself once S1 generation is actually working.

Tool-call runtime probe (`TOOLS_REQUIRED=1`) still TODO — assert
a request with `tools=[...]` populates `tool_calls` rather than
emitting raw DSML tokens in `content`.

Sanity check parsers are wired (no TPU needed):

```bash
PYTHONPATH=work/vllm:work/tpu-inference work/vllm_env/bin/python3 -c "
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
ReasoningParserManager.get_reasoning_parser('deepseek_v4')
ToolParserManager.get_tool_parser('deepseek_v4')
print('OK')"
```

#### S4. Chat encoding — RESOLVED upstream

vLLM's `DeepseekV4Tokenizer` auto-loads for
`DeepseekV4ForCausalLM` and routes through the upstream
`encode_messages`, byte-equal to the V4-Flash reference
encoder across chat / thinking / tools / tool_calls /
reasoning_effort. The custom tokenizer **ignores any
`--chat-template` arg**. Pinned by
`TestVllmChatTemplateParity`. Kept here as a regression
boundary.

#### S5. MTP speculative decoding hook is not wired

`work/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py`
only handles `ngram` and `eagle3`. Math is ready
(`deepseek_v4_mtp_forward` validated on tiny fixture; MTP
weights load). Need a `DeepseekV4MTPProposer` in
`tpu_inference/spec_decode/jax/` + wire into
`execute_draft_model` + engine flag
`--speculative-config '{"method":"deepseek_v4_mtp","num_speculative_tokens":1}'`.
1.5–2× decode throughput once S1 lands.

#### S6. Sampling parameters — probes scaffolded; only validate first ~3 tokens

Per-knob probes (sampling, stop, logprobs, top-k, presence,
n>1) pass on real `vllm serve` but only on the same Paris-shaped
prompt that produces 1-3 tokens before stopping. They don't
exercise sustained generation, so any "S6 done" claim is
meaningless until S1's empty-output bug is fixed. Determinism
under sampling not asserted (vLLM/TPU rejects per-request `seed`
on non-greedy paths with HTTP 400).

#### S7. Streaming — equivalence probe scaffolded, same caveat as S6

`STREAMING_REQUIRED=1` re-fires the deterministic completion
with `stream=true`, reassembles SSE chunks, asserts byte-equality
vs non-streaming. Passes today on the Paris probe but doesn't
validate sustained streaming. Latency-budget probe (TTFT/ITL,
behind `STREAMING_LATENCY_REQUIRED=1`) still TODO.

#### S8. Sustained-generation probe — minimum check past the Paris gate

The basic smoke validates "deterministic ` Paris`" — 2-3 generated
tokens. The longest verified output across all probes is the
reasoning probe at len=2 ("We"). A degenerate-output bug (token-
repetition loop, premature stop, flat-logits regime kicking in
after token N, MoE expert collapse mid-generation) silently passes
today's gate. Tier S because "outputs garbage on token 6" is a
silent correctness bomb in exactly the shape S3's thinking-mode
bug was.

C1 (lm-eval-harness) and C2 (long-context functional) are the
honest "we serve V4-Flash" gates but defer to Tier C until S2
unblocks tractable wall-clock. **S8 is the *minimum* sustained-
generation check that should run on every smoke.**

Probe (gated `LONG_GEN_REQUIRED=1`, exit 13):
* `/v1/completions`, prompt `"Tell me a short story about a robot
  exploring Mars:"`, `max_tokens=64`, `temperature=0`, `seed=0`.
  Same prefill bucket as the basic completions probe (no extra
  compile).
* Wall-clock from request to response; log
  `observed_tps = completion_tokens / wall_clock_s`.
* Assert: `completion_tokens >= 30`, no word repeated 5+ times in
  a row (token-loop detector), trailing 5 chars of stripped text
  contain ≥2 alphanumerics (rejects `....` endings + trailing
  whitespace).

Future S8 extensions (file under same item):
* `MULTI_TURN_REQUIRED=1`: 2-turn chat, assert turn 2 is coherent
  given turn 1's response, no engine state leak between turns.
* `LONG_PROMPT_REQUIRED=1`: 2k-prompt completion (separate from
  C2's 4k-1M sweep). Smaller, faster gate that catches the
  prompt-bucket-2 path.

### Tier A — production-deployment infra (model is correct but infra isn't)

* **A1.** `MAX_LEN=256, MAX_SEQS=1` is hard-coded. Activation
  HBM scales linearly with both. **Depends on S1**: lifting
  `MAX_LEN` is meaningless without real decode.
* **A2.** Persistent compile cache is host-local and ephemeral
  (most cloud VMs clear `/tmp` on reboot). Move to a verified-
  durable mount; add a one-shot
  `scripts/full_slice_v4_warm_cache.sh` per host on first
  boot. AOT precompile + binary persist (B4) is the next-level
  fix.
* **A3.** No engine crash recovery. Supervisor + drain on
  SIGTERM. Tie supervision to vLLM's `/health`.
* **A4.** No metrics / observability. `--enable-metrics` not
  set; no TTFT, ITL, throughput, queue depth, KV utilization,
  error rate.
* **A5.** No TLS / authentication / rate limiting. Currently
  `0.0.0.0:18081` plain HTTP. For OpenRouter exposure: TLS at
  reverse proxy, per-API-key auth, per-key rate limiting,
  non-root systemd unit.
* **A6.** Single slice — no horizontal scale. Multi-slice
  needs model-aware load balancer with sticky sessions.
* **A7.** Prefix caching disabled. `Prefix cache hit rate:
  0.0%` in every smoke log. vLLM's `--enable-prefix-caching`
  hasn't been verified compatible with V4's dual-buffer KV
  (SWA circular buffer + compressor pool) — INVARIANTS I34
  notes V4's per-layer state lives in the params tree, not
  vllm's `kv_caches`. 5–10× throughput win for OpenRouter-
  style chat with shared system prompts if the integration
  works.
* **A8.** Cancellation propagation unverified. Client
  TCP-disconnects mid-stream may leak compute (JAX/TPU async
  dispatch keeps running the orphaned request). Add an
  end-to-end probe + verify vLLM's request cancellation is
  wired through to the JAX scheduler.
* **A9.** No `/health` vs `/ready` distinction. K8s rolling
  updates would route traffic to a server still in cold
  compile. Split: `/health` = process alive (responds during
  compile); `/ready` = warm-cache populated + first-curl
  succeeded.
* **A10.** No server-side request caps. `max_tokens`, prompt
  length, payload size, tool-call depth, `n` are unvalidated
  beyond vLLM's defaults — runaway / malicious clients can
  starve the queue or trigger fresh-compile storms (per-
  position decode JIT cache miss, see S1 residual).
* **A11.** No worker-host weight-divergence detection. Cache-
  fingerprint finding showed 8 distinct sha256s for "the
  same" compile cache entry — that's *expected* (SPMD compiles
  host-specific binaries). What's *not* checked: bf16 weight
  bytes per host. Silent gcsfuse fault / partial mmap on one
  host = silently wrong output on 1/8 of devices, no alarm.
  Hash-and-compare at engine init.

### Tier B — known performance work

* **B1.** Sparse-attention Pallas kernel.
  `sparse_attn` at
  `layers/jax/attention/deepseek_v4_attention.py:131` is
  fully-materialized (DECISIONS D2). Real Pallas: gather kv
  only for top-k indices in TPU SRAM, fuse sink term into
  softmax denominator, avoid `[B, M, K, D]` gather buffer.
  2–5× decode latency once S1 unlocks the regime.
* **B2.** True sparse MoE dispatch.
  `moe_forward` at `layers/jax/moe/deepseek_v4_moe.py:156` is
  vectorized-dense; FLOP cost is `top_k * E` higher (32× over
  true sparse for `top_k=8, E=256`). Wire
  `kernels/megablox/gmm.py` into V4's MoE.
* **B3.** SPMD remat-warning audit — DONE for `compressor.ape`
  family (126 → 0 via `pick_partition_spec` `_MIN_SHARD_ELEMENTS=8K`
  threshold in `models/jax/deepseek_v4_loader.py`). Re-audit
  after future kernel work via:
  ```
  grep "Involuntary full rematerialization" \
       logs/full-slice-v4-smoke-*.log
  ```
* **B4.** AOT compile + binary persist.
  `jit().lower().compile()` → serialize → load. Per-host
  because of the cache fingerprint finding. Drops cold compile
  from ~97s to ~5s/host. Defer until B1+B2 land.

### Tier C — quality gates (don't claim "we serve V4" without these)

* **C1.** Benchmark vs DeepSeek's reference scores (MMLU,
  HellaSwag, GSM8K, HumanEval, MATH) via `lm-eval-harness`.
  Needs S2 for tractable wall-clock. **The gate that lets you
  claim "we serve V4-Flash" honestly.**
* **C2.** Long-context functional test (4k, 16k, 64k, 256k,
  1M needle-in-haystack). V4-Flash's compressor + indexer are
  unexercised in production. Needs A1.
* **C3.** Math regression suite under load (random sampling,
  long contexts, tool calls, multi-turn).
* **C4.** Tokenizer edge cases (non-ASCII, leading whitespace,
  multilingual code, very-long single tokens). Extend
  `TestVllmChatTemplateParity` rather than writing a new test.
* **C5.** Refusal/safety preservation — small refusal-eval set
  before each S1/B2 change (low-bit MoE quant especially can
  shift safety tuning).

### Tier D — janitorial

* **D1.** `tests/models/jax/test_deepseek_v4.py` is 2904 LOC,
  29 test classes (~4× peer models). Per-class audit needed
  for further consolidation; cheap wins are taken.
* **D2.** No log rotation. `logs/full-slice-v4-smoke-*.log`
  accumulates ~1–2 MB per smoke run on the head host. Add a
  retention policy or pipe through `logrotate`.

## Iteration discipline

**Don't use `./run.sh serve` as your inner test loop** — each
attempt is 25–45 min. Use the fastest validation that catches
the bug class:

1. Standalone math scripts under `/tmp/` (~10–30s).
2. Tiny-fixture pytest classes in
   `tests/models/jax/test_deepseek_v4.py` (~30s–2min CPU).
3. `eval_shape` / `lower().compile()` on real config under
   virtual mesh (~1–3min). Pattern:
   `XLA_FLAGS=--xla_force_host_platform_device_count=32
   JAX_PLATFORMS=cpu`.
4. Real `./run.sh serve` — at most 1–2 per session.

### Real-smoke phase budgets

| Phase | Expected | Bail signal |
|---|---|---|
| Startup + Ray init | ~30s | No log activity >2 min |
| Weight load | ~4 min | No `[deepseek_v4] placed N tensors` heartbeat >2 min |
| `capture_model` precompile | ~30s | `RESOURCE_EXHAUSTED` / OOM |
| `jit_run_model` cold compile | **10–30 min cold, ~97s warm** | 3+ `slow_operation_alarm.cc` warnings, OR `RESOURCE_EXHAUSTED` |
| First curl | sub-second after compile | Curl timeout, engine crash |

Silence in `jit_run_model` ≤25 min is *expected*, not stuck.

### Iter-timeout management

`ITER_TIMEOUT_SEC=5400` (90 min). At T-15 min stop launching
new long steps + commit a "WIP:" checkpoint. At T-5 min reset
the cluster + push.

## What's verified

**On real V4-Flash via `vllm serve`:**
* Basic ` Paris` smoke gate (R1==R2 byte-equal) on a *fresh*
  engine — degrades to garbage after enough requests.
* Sampling / stop / logprobs / top-k / presence / n>1 probes —
  but all only validate the FIRST 1-3 tokens.
* Streaming SSE byte-equal to non-streaming.

**On tiny-config CPU (math correctness only — does NOT translate
to real V4 generation working):**
* Decode kernels match torch reference at 1e-4
  (`TestPackedDecodeStateBuffer`,
  `TestTransformerBodyDecodeRoundTrip`,
  `TestPrefillToDecodeStateParity`,
  `test_buffer_decode_jit_cache_hits_across_positions`).
* MTP forward math validated (S5 runtime not wired).
* Chat encoding byte-equal to V4-Flash reference encoder
  (`TestVllmChatTemplateParity`).

**Infra (no per-token correctness claim):**
* Streaming sharded loader (~4 min, no OOM).
* Slice-aware load + multi-threaded placement.
* Inline MoE consolidation at load (HLO 47k vs 103k prior).
* Freqs cap by effective seq-len (1 GB → KB).
* Persistent JAX compile cache populated under
  `~/.cache/vllm/xla_cache` per host.
* Reasoning + tool parser registry lookup.
* Tiny-tensor replication at load (B3 fix): 126 → 0
  `Involuntary full rematerialization` warnings.

**NOT verified — explicitly broken or untested past 1-3 tokens:**
* Sustained generation (S1 / S8 — output is empty/`#` past
  position ~2 on real V4 with `V4_DECODE_STATE=1`).
* Multi-turn conversations (no probe yet).
* Long context (4k, 16k, 64k+ — C2).
* Real tok/s under sustained generation (last measurement was
  0.53 tok/s but the generation was empty).
* Refusal preservation (C5).
* Math / code reasoning quality vs reference (C1).

## Pitfalls already learned (don't repeat)

1. **Broad pkill regex hits raylets.** Patterns like
   `pkill -f "EngineCore|RayWorkerWrapper|vllm"` match strings
   in raylet's command line on remote workers and kill the
   daemon — losing 7/8 nodes. Always use narrow comm-name
   match: `pkill -x VLLM::EngineCore`, or kill by exact pid.

2. **`/tmp/libtpu_lockfile` survives SIGKILL.** A killed
   EngineCore leaves the lockfile; subsequent inits SIGSEGV
   on a "clean" start. Reset script handles it; manual kills
   need manual lockfile removal.

3. **`git push` doesn't sync workers.** Each worker has its
   own clone. Lost a 30-min load cycle running stale code on
   7/8 hosts. **Always `scripts/full_slice_v4_sync.sh` after
   any code edit.** Note: only `work/tpu-inference/tpu_inference/`
   and `scripts/` are synced — the repo-root markdown files
   (CLAUDE.md, etc.) are head-only.

4. **Don't add unverified XLA flags.**
   `--xla_tpu_impure_hlo_parallel_compile=true` looked
   plausible but is not a recognized flag in this libtpu build
   and SIGSEGV'd every Ray worker. Validate any addition with
   `python -c "import jax; jax.devices()"` first. The smoke
   script ignores parent-shell `XLA_FLAGS` — use
   `V4_XLA_FLAGS=...` to opt in.

5. **First inference is slow on a fresh launch.** Cold
   compile of `jit_run_model` = 5–15 min; warm cache ~30–60s.
   Don't use a 60s curl timeout — smoke check defaults to
   900s. To warm at bootstrap, set `WARM_CACHE_ON_BOOTSTRAP=1`
   in `.env` before `./run.sh bootstrap`.

6. **`--enforce-eager` does not skip XLA compile.** That flag
   only affects vLLM's CUDA-graph-equivalent path. The TPU
   forward is JAX/`tpu-inference` and ALWAYS jit-compiles.

7. **vLLM's `capture_model` can multiply compile cost.**
   Without `--enforce-eager`, vLLM precompiles many shape
   buckets up front. The smoke launcher uses `--enforce-eager`
   to skip that.

8. **`JAX_COMPILATION_CACHE_DIR` does nothing under vLLM.**
   `tpu_inference/runner/compilation_manager.py:53` calls
   `jax.config.update(...)` overriding whatever the launcher
   set. The real cache is at `~/.cache/vllm/xla_cache` (or
   `VLLM_XLA_CACHE_PATH`). Verify cache activity by
   `ls -la ~/.cache/vllm/xla_cache`, not the launcher's
   echoed path.

9. **`/v1/chat/completions` first-call OOM-retry is normal.**
   Chat path lands in 1024-token prefill bucket vs 256 for
   completions; on a tight HBM budget engine sometimes hits
   `RESOURCE_EXHAUSTED` and recovers via
   `TpuLoadedExecutable::ExecutePrepareWithOomRetries`. Adds
   ~30s to first-chat latency; subsequent chat calls are fast.

## Layout

* `work/tpu-inference/tpu_inference/models/jax/deepseek_v4*.py`,
  `layers/jax/{attention,moe}/deepseek_v4_*.py` — V4 source.
* `work/vllm/` — vLLM upstream. Don't edit unless you've read
  `work/vllm/AGENTS.md`.
* `scripts/full_slice_v4_*.sh` — per-host operational helpers.
* `logs/` — `.gitignore`d.
* `prompt.md` — autonomous loop's prompt; read CLAUDE.md first.

Durable docs in `work/tpu-inference/`: `INVARIANTS.md` (math
invariants), `DECISIONS.md`, `BLOCKERS.md` (pointer to backlog),
`TINY_CONFIG.md`, `TOLERANCE_LOG.md`, `V3_TO_V4_DIFF.md`.

## Chat template note

V4-Flash ships **no Jinja `chat_template`**. vLLM's
`DeepseekV4Tokenizer` auto-resolves via upstream `encode_messages`,
ignoring any `--chat-template` arg. Pinned by
`TestVllmChatTemplateParity`. `vllm chat` CLI needs
`--url http://localhost:18081/v1`.

## Sanity check on a fresh VM

```bash
# CPU math (no TPU; ~10s)
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

Update this file as you learn — but updates are for *durable*
operational knowledge. Per-session decisions go in commit
messages.

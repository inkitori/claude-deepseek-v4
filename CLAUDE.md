# claude-deepseek-v4 — agent runbook

You're picking up a TPU-inference effort: get
`vllm serve deepseek-ai/DeepSeek-V4-Flash` deployable on a
**v6e-32 TPU slice** (TP=32, 8 hosts × 4 chips) at production
quality. The model is a flagship MoE — 256 FP4 experts + MLA
attention + dense FP8 — about 543 GiB bf16-expanded.

The Tier-8 deploy gate is GREEN: `/v1/completions` returns
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

### 3. Code style matches upstream — this work targets eventual upstream PR

V4 source files should be **indistinguishable in style** from
their peer models in this repo (`qwen3.py`, `deepseek_v3.py`,
`llama3.py`, `llama4.py`, `gemma4.py`). The minimum-delta rule
covers *quantity*; this rule covers *style*. A reviewer at
`vllm-project/tpu-inference` should not be able to tell which
lines came from an autonomous agent.

* **Docstrings**: brief Args/Returns blocks. No multi-paragraph
  hypothesis-rationale, no "previous implementation was X"
  history (that's a commit-message thing).
* **Comments**: only when the WHY is non-obvious. No
  iter-narrative references ("iter-5h fix", "Tier-8 keystone",
  "Bug A was…"). No section banners (`# ===== gate =====`,
  `# ----- expert -----`). Well-named identifiers do the WHAT
  job already.
* **Defensive programming**: trust internal call paths. Validate
  only at the vLLM API boundary (e.g. `__call__` argument
  shapes), not at every internal helper. Peer models don't add
  belt-and-suspenders asserts; neither should V4.
* **Backwards-compat shims**: none. V4 is new code; if a branch
  is dead, delete it. No `# removed for X` placeholders, no
  re-exports of types nothing imports, no `_unused_var` renames.
* **Naming**: snake_case modules + functions, PascalCase
  classes, verbatim from upstream where applicable
  (`AttentionMetadata`, `compute_q_projection`, etc.).

Before considering a piece of V4 work "done", do a 30-second
diff-side-by-side with `qwen3.py` or `deepseek_v3.py`. If your
file looks chattier, prune. If your function has a 30-line
docstring and qwen3's equivalent has 4 lines, prune.

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

* **`full_slice_v4_reset.sh`** — handles the four orphan-state
  surfaces (api-server pid, `VLLM::EngineCore` Ray actor
  children, `/tmp/libtpu_lockfile`, leaked Ray placement
  groups). Does **not** restart Ray itself. Cheap; run before
  every smoke launch.
* **`full_slice_v4_ray_restart.sh`** — heavier nuke
  (`ray stop --force` + fresh `ray start` on all 8 hosts).
  Reach for it only when reset isn't enough (e.g. `ABORTED:
  TPU is already in use by process X` even after reset).
* **`full_slice_v4_sync.sh`** — **mandatory after any code
  edit**. rsyncs `work/tpu-inference/tpu_inference/` and
  `scripts/` to all 7 worker hosts. (The repo-root markdown
  files and other dirs are NOT synced.)
* **`full_slice_v4_smoke.sh`** — launches `vllm serve` with V4
  optimization knobs forwarded to Ray workers. Writes pid to
  `logs/full-slice-v4-smoke.pid` and a timestamped log to
  `logs/full-slice-v4-smoke-<TS>.log`.
* **`full_slice_v4_smoke_check.sh`** — polls `/v1/models` until
  ready, fires the deterministic "capital of France"
  completion twice, asserts byte-identical responses + that
  the text contains "Paris". Self-test at
  `scripts/test_smoke_check_harness.sh` (24 mock scenarios, no
  TPU). Also fires nine optional probes, each gated by an env
  knob (default off so the smoke stays cheap):
  `CHAT_REQUIRED`, `REASONING_REQUIRED`, `STREAMING_REQUIRED`,
  `SAMPLING_REQUIRED`, `STOP_REQUIRED`, `LOGPROBS_REQUIRED`,
  `TOPK_REQUIRED`, `PRESENCE_REQUIRED`, `N_REQUIRED` — see the
  knobs table for one-line semantics and exit codes 4–12.
* **`full_slice_v4_warm_cache.sh`** — runs the smoke + check,
  then cleans up. Use once on a fresh VM (or after `/tmp`
  wipe) to populate the JAX compile cache; subsequent first-
  curls are sub-minute on cache hit.

Pass criterion: `full_slice_v4_smoke_check.sh` exits 0 with
`PASS: deterministic completion contains 'Paris'`.

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
| `V4_DECODE_STATE` | `0` | S1 gate. `1` flips `__call__` to `deepseek_v4_run_with_decode_state` (orchestrator path threading packed `AttentionDecodeState` through `kv_caches`). Default `0` preserves the green gate. When `1`, every JIT trace into `__call__` logs `(call_idx, T, start_pos, is_decode, state_max_seq_len, kv_caches_count)`. |
| `V4_DECODE_STATE_DIAG` | `0` | Adds `jax.debug.print` of kv_caches[0] sum + nonfinite-count at orchestrator entry/exit. Forwarded to workers. Use only when debugging Bug B (S1). |
| `CHAT_REQUIRED` | `0` | smoke_check chat probe; exit 4 on missing/empty content. Adds ~30 s for first-chat OOM-retry (pitfall #9). |
| `REASONING_REQUIRED` | `0` | Thinking-mode chat probe; exit 5 on empty `message.reasoning`. Pins S3 runtime. Passes under `V4_DECODE_STATE=1` (S1 iter-5j); fails under default `V4_DECODE_STATE=0` (broken-decode prefill-only path produces all-newlines). Pair with `REASONING_MAX_TOKENS=8` to bound per-position decode-compile cost when run under `V4_DECODE_STATE=1`. |
| `STREAMING_REQUIRED` | `0` | SSE byte-equality probe vs non-streaming; exit 6. Pins S7. |
| `SAMPLING_REQUIRED` | `0` | `temperature=0.7, top_p=0.9, frequency_penalty=0.1`; exit 7 on empty/invalid. Pins S6. Determinism not asserted (per-request seed unsupported on non-greedy paths). |
| `STOP_REQUIRED` | `0` | `stop=["Paris"]`; exit 8 if response contains Paris or `finish_reason!=stop`. Pins S6. Baseline emits ` Paris` first, so a working handler MUST intercept. |
| `LOGPROBS_REQUIRED` | `0` | `logprobs=5`; exit 9 if any position has fewer than 5 alternatives. Pins S6. |
| `TOPK_REQUIRED` | `0` | `temperature=0.7, top_k=10`; exit 10 on empty/invalid. Pins S6. |
| `PRESENCE_REQUIRED` | `0` | `temperature=0.7, presence_penalty=0.5`; exit 11 on empty/invalid. Pins S6. |
| `N_REQUIRED` | `0` | `n=2`; exit 12 if fewer than 2 non-empty choices. Pins S6. Under `--max-num-seqs=1` likely sequentializes but cardinality must equal n. |

## Production-readiness backlog

Pick the highest-leverage uncompleted item. Items earlier block
items later. Per-iter narrative belongs in commit messages; the
backlog itself stays brief.

### Tier S — silent correctness bombs (fix before perf)

#### S1. Decode is not real decode — IN FLIGHT

`__call__` in `models/jax/deepseek_v4.py` defaults to
`transformer_body_forward`, which recomputes attention over
prompt+generated context every step. Decode-step kernels
(`attention_decode_step`, `compressor_decode_step`,
`indexer_decode_step`) match the torch reference at 1e-4 on
tiny config but aren't wired in production.

**What's landed (env-gated behind `V4_DECODE_STATE=1`):**
* Closed-form prefill→state init
  (`attention_init_state_from_prefill`); transformer-body
  primitives (`block_init_state_and_forward`,
  `block_decode_step`, `transformer_body_init_state_to_buffer`,
  `transformer_body_decode_step_from_buffer`); pack/unpack of
  `AttentionDecodeState` to a 1D fp32 buffer per layer.
  Pinned by `TestPackedDecodeStateBuffer`,
  `TestTransformerBodyDecodeRoundTrip`,
  `TestPrefillToDecodeStateParity` (37 cases, ≤5e-3 vs torch
  reference on tiny config).
* Orchestrator
  `deepseek_v4_run_with_decode_state(kv_caches, ids, params,
  ..., is_decode_step, start_pos)` branches on
  `is_decode_step`, threads packed state through `kv_caches`.
  Prefill `h` is byte-equal-by-construction to
  `transformer_body_forward` (path-A decoupling +
  `lax.optimization_barrier(h)` to prevent XLA CSE/fusion
  drift on bf16 reductions).
* Runtime hooks: V4-only `kv_cache_sharding=P()` replicated
  override in `model_loader.py`, V4 packed-buffer allocator
  in `kv_cache_manager.py`, `decode_start_pos` meta_field on
  `AttentionMetadata`, `_maybe_set_v4_decode_start_pos` in
  `tpu_runner.py`, module-level
  `v4_state_max_seq_len_from_vllm_config` (single source of
  truth for buffer sizing).
* Decode-detection uses `decode_start_pos > 0` directly (the
  runner pads decode shapes to bucket sizes so a `T==1` check
  never triggers).
* iter-5g `with_sharding_constraint(P())` on packed buffers
  dodges the TPU `[USER]` FATAL on R2's cached prefill
  artifact re-execution.
* iter-5j `_v4_force_kv_caches_read(buffers, kv_caches)`:
  fold `pb + opaque_zero * kv` into each output buffer where
  `opaque_zero = lax.optimization_barrier(jnp.zeros(()))`. The
  compute output equals `pb` byte-for-byte (zero at runtime)
  but XLA must emit a real read of `kv_caches[i]` and a real
  elementwise add. Without this, XLA could elide the
  donation-aliased write when output value is statically
  derivable without reading the input — leaving prior-call
  contents partly bleeding through.

**Bug B FIXED by iter-5j (2026-04-30 ~20:03Z, smoke
`logs/full-slice-v4-smoke-20260430T195032Z.log`):** R1 and R2
both return ` Paris` byte-equal on real V4-Flash with
`V4_DECODE_STATE=1`. Both HTTP 200, no FATAL, no divergence.
The CPU virtual-mesh experiment
(`/tmp/test_v4_iter5j_kv_caches_dep.py`) confirmed the
mechanism: BEFORE iter-5j the HLO entry signature dropped
`kv_caches` inputs entirely (XLA proved unused); AFTER it
keeps them with explicit `input_output_alias={ {i}: (i, ...,
may-alias) }` markers, while compute on CPU stays byte-equal
between zeros and random `kv_caches` inputs (opaque_zero is
0 at runtime).

The decode runtime is now real-V4-correct. **Remaining S1
work**: per-position decode JIT cache misses (each new
`start_pos` triggers a fresh ~50s compile because
`decode_start_pos` is hashed into the cache key — see
optimization-knobs row for `V4_DECODE_STATE`). Refactor
decode kernels to accept traced `start_pos` for one-compile-
fits-all positions (iter-5b "Tactical pick (a)"). Until then,
warm-cache amortization + AOT precompile (A2 / B4) makes
this a perf nit, not a correctness blocker.

S1 unlocks A1 (`max-model-len`), B1 (sparse_attn Pallas
becomes worthwhile), S5 (MTP speculative decoding).

#### S2. Multi-sequence dispatch is a Python loop in eager mode

`__call__` runs each active sequence sequentially through
`transformer_body_forward`. Production needs a ragged-batch
jit'd kernel — either extend
`kernels/ragged_paged_attention/v3` to support V4's top-k +
attn_sink + dual-buffer KV (DECISIONS D2), or jit V4's path
with `lax.dynamic_slice` per active seq. Until S2 lands,
`--max-num-seqs=1` is forced. Independent of S1 but they
multiply.

#### S3. Reasoning parser — FIXED by S1; tool-call runtime probe still TODO

Smoke launcher passes `--reasoning-parser deepseek_v4`,
`--enable-auto-tool-choice`, `--tool-call-parser deepseek_v4`.
vLLM validates parser names at startup (smoke-green = parsers
loaded). V4's reasoning parser is registered upstream as an
alias for `DeepSeekV3ReasoningParser` (standard `<think>...</think>`
extraction).

**Thinking-mode bug FIXED by S1 iter-5j (2026-04-30 ~20:34Z,
smoke `logs/full-slice-v4-smoke-20260430T201602Z.log` +
`logs/smoke-check-iter5k-20260430T202223Z.log`):** under the
green-gate baseline (`V4_DECODE_STATE=0`, prefill-recompute
every step) thinking-mode chat at `temperature=0, seed=0`
returned N newlines (all-whitespace, `finish_reason=length`)
because the prefill-only path kept re-rolling from a
flat-logits regime. With `V4_DECODE_STATE=1` (real decode,
packed `AttentionDecodeState` threaded through `kv_caches`)
the model emits real reasoning tokens. Verification:

```bash
V4_DECODE_STATE=1 scripts/full_slice_v4_smoke.sh
REASONING_REQUIRED=1 REASONING_MAX_TOKENS=8 \
  scripts/full_slice_v4_smoke_check.sh
# expect: "[smoke-check] reasoning (len=N>0): <real tokens>"
#         "[smoke-check] PASS"
```

`REASONING_MAX_TOKENS=8` (vs the default 96) keeps the
per-position decode-compile cost bounded under
`V4_DECODE_STATE=1` until the JIT-cache-miss amortization is
in place; len>0 still discriminates the bug since broken=
all-whitespace.

**Remaining S3 work**: tool-call runtime probe
(`TOOLS_REQUIRED=1`) — assert a request with `tools=[...]`
populates `tool_calls` rather than emitting raw DSML tokens
in `content`. Parser is registered (`deepseekv4_tool_parser.py`
in vLLM upstream) but not exercised end-to-end yet.

Sanity check that parsers are still wired (no TPU needed):

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

#### S6. Sampling parameters — all probes scaffolded + verified end-to-end

Per-knob probes (sampling, stop, logprobs, top-k, presence,
n>1) all pass on real `vllm serve` 2026-04-30. See knob table
above. **Determinism under sampling is not asserted** — vLLM/
TPU's runner rejects per-request `seed` on non-greedy paths
with HTTP 400 ("JAX does not support per-request seed."), so
the sampling probe omits `seed`.

Future S6 work would be combinatorial coverage (logprobs+stop,
topk+presence, etc.) — defer until a production client report
shows a gap.

#### S7. Streaming — equivalence probe verified; latency budget probe TODO

`STREAMING_REQUIRED=1` re-fires the deterministic completion
with `stream=true`, reassembles SSE chunks, asserts
byte-equality vs non-streaming. Verified on real V4 2026-04-30.

Still TODO: latency-budget probe (TTFT / ITL thresholds, gated
behind `STREAMING_LATENCY_REQUIRED=1`). OpenRouter-style
clients default to streaming, so latency-on-streaming is the
production metric users feel.

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

**Do NOT use `./run.sh serve` as your inner test loop.** Each
attempt is 25–45 min (4 min load + 10–30 min cold compile +
curl wait). That budget is fixed by XLA. Prior sessions burned
real time treating it as if it should be fast. Use the fastest
validation that catches the bug class you're working on:

1. **Standalone math scripts** under `/tmp/` (~10–30s).
2. **Tiny-fixture pytest classes** in
   `tests/models/jax/test_deepseek_v4.py` (~30s–2min on CPU).
3. **`eval_shape` / `lower().compile()` on the real config**
   (~1–3min). Catches sharding bugs + HLO-emit failures
   without paying runtime compile cost. Pattern:
   `XLA_FLAGS=--xla_force_host_platform_device_count=32
   JAX_PLATFORMS=cpu`.
4. **Real `./run.sh serve`** only when 1–3 are green. Budget at
   most 1–2 of these per session.

### Real-smoke phase budgets (don't bail too early)

| Phase | Expected duration | Bail signal |
|---|---|---|
| **vLLM startup + Ray cluster init** | ~30s | No log activity for >2 min, OR `Worker exit type: SYSTEM_ERROR`. |
| **Weight load** | ~4 min | No `[deepseek_v4] placed N tensors` heartbeat for >2 min, OR placed count stops growing. |
| **`capture_model` precompile** | ~30s | Any `RESOURCE_EXHAUSTED` / `CompileTimeHbmOom`. |
| **`Application startup complete`** | fires immediately after capture_model | If absent >2 min after capture_model finishes. |
| **`jit_run_model` cold compile** | **10–30 min** cold, **~97s** warm | Three or more `slow_operation_alarm.cc` warnings (one alarm = one slow pass; that alone is *not* enough to bail). Any `RESOURCE_EXHAUSTED` / `Worker exit`. |
| **First curl returning** | sub-second after compile | Curl 900s timeout, OR engine crashes mid-execute. |

Silence in `jit_run_model` ≤25 min is *expected*, not stuck.
**Don't bail before 25 min** unless `ITER_TIMEOUT_SEC` (5400 /
90 min) is closing in. Spend compile time productively —
sketch the next-lane fix in `/tmp/`, audit warning families,
consolidate test bloat (test edits don't conflict with the
running smoke).

### Iter-timeout management

* **At T-15 min:** stop launching new long-running steps.
  Commit whatever code change you've made (with "WIP:" prefix
  describing what was tried + what's still unverified) so iter
  N+1 picks up from on-disk state.
* **At T-5 min:** reset the cluster + push the WIP commit.
  Don't risk the iter being killed mid-`./run.sh serve`.

## What's verified

* Streaming sharded loader (no zero-tree OOM).
* Slice-aware load + multi-threaded placement
  (`V4_LOADER_PLACE_WORKERS=8`); safetensors handle cache
  (~6× load speedup, ~4 min total).
* Vectorized MoE forward (math byte-equal to per-expert
  reference; HLO 4.6× smaller). ✓ correctness, ✗ optimal flops
  (B2).
* Inline MoE consolidation at load (256 per-expert weights of
  each `(layer, wname)` group stacked into single E-sharded
  jax.Array). `jit_run_model` HLO 47k optimized vs 103k prior.
* Freqs cap by `max_model_len` — `_effective_freqs_seq_len`
  uses `vllm_config.model_config.max_model_len` (1 GB/chip →
  KB).
* Persistent JAX compile cache populated under
  `~/.cache/vllm/xla_cache` per host.
* Decode-step kernels (`attention_decode_step`,
  `compressor_decode_step`, `indexer_decode_step`) match torch
  reference at 1e-4. ✓ math, ✓ runtime integration on real V4
  (S1 iter-5j; engine returns deterministic ` Paris`
  byte-equal across R1/R2 with `V4_DECODE_STATE=1`). Default
  `V4_DECODE_STATE=0` keeps green-gate as the smoke-pinned
  fallback until the per-position decode JIT-cache-miss
  amortization is in place.
* Transformer-body decode primitives + packed
  `AttentionDecodeState` buffers + orchestrator with JIT-
  correctness validation. Pinned by 37-test iter-5 suite.
* MTP forward math validated. ✓ math, ✗ runtime (S5).
* Chat encoding byte-equal to V4-Flash reference encoder
  across all S4 scopes (`TestVllmChatTemplateParity`).
* Reasoning + tool parser wiring (registry lookup ✓; runtime
  reasoning probe ✓ under `V4_DECODE_STATE=1` — fixed by
  S1 iter-5j; tool-call runtime probe still TODO, see S3).
* Streaming probe (SSE = non-streaming byte-for-byte).
* Sampling / stop / logprobs / top-k / presence / n>1 probes
  all pass on real V4.
* Tiny-tensor replication at load (B3 fix): 126 → 0
  `Involuntary full rematerialization` warnings.

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

* `work/tpu-inference/` — JAX V4 implementation (subtree of
  upstream `tpu-inference`). V4 model:
  `tpu_inference/models/jax/deepseek_v4*.py`. MoE:
  `layers/jax/moe/deepseek_v4_moe.py`. Attention:
  `layers/jax/attention/deepseek_v4_attention.py`.
* `work/vllm/` — vLLM source. Don't edit upstream files
  unless you've read `work/vllm/AGENTS.md`. Reasoning + tool
  parsers for `deepseek_v4` already exist upstream.
* `scripts/` — operational helpers; per-host entry points all
  start with `full_slice_v4_`.
* `logs/` — `.gitignore`d; smoke + iter logs accumulate here.
* `README.md` — fresh-VM bringup (one-shot via `./run.sh`).
* `.env.example` — every env var documented.
* `prompt.md` — the prompt the autonomous loop hands to
  `claude -p`. Read CLAUDE.md (this file) first.

Durable docs in `work/tpu-inference/`:
* `INVARIANTS.md` — math invariants. Each broken invariant is
  a shipping bug.
* `DECISIONS.md` — durable architectural decisions.
* `BLOCKERS.md` — short pointer at the backlog above.
* `TINY_CONFIG.md`, `TOLERANCE_LOG.md`, `V3_TO_V4_DIFF.md` —
  math reference; don't decay.

## Chat template note

V4-Flash deliberately ships **no Jinja `chat_template`** —
`tokenizer_config.json` omits the field. vLLM's
`DeepseekV4Tokenizer` auto-resolves and routes through
upstream `encode_messages`, ignoring any `--chat-template`
arg. The smoke launcher does not pass one; there's no Jinja
file in the repo. Behavior pinned by
`TestVllmChatTemplateParity`. `vllm chat` CLI needs
`--url http://localhost:18081/v1` (smoke launcher binds 18081).

## Sanity check on a fresh VM

After `cp .env.example .env` + filling in tokens + `./run.sh`:

```bash
# 1. CPU-only math test (no TPU; ~10s)
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

Update this file as you learn — but updates are for *durable*
operational knowledge. Per-session decisions go in commit
messages.

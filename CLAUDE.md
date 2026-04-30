# claude-deepseek-v4 — agent runbook

You're picking up a TPU-inference effort: get
`vllm serve deepseek-ai/DeepSeek-V4-Flash` deployable on a
**v6e-32 TPU slice** (TP=32, 8 hosts × 4 chips) at production
quality. The model is a flagship MoE — 256 FP4 experts + MLA
attention + dense FP8 — about 543 GiB bf16-expanded.

The Tier-8 deploy gate is GREEN (`/v1/completions` returns
deterministic `Paris` for "The capital of France is", cold compile
~97s, warm-cache curl sub-second). The work now is converting that
demo into something that handles real concurrent OpenRouter-grade
traffic without silent correctness regressions. That backlog lives
in **"Production-readiness backlog"** below.

The single non-negotiable goal is **fast, mathematically correct
inference with the real V4-Flash weights**, served via the
OpenAI-compatible HTTP endpoint to many concurrent users.
Synthetic-fixture tests are the fast iteration loop; real-weight
`vllm serve` is the gate that defines "done".

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
   Add a parametrized test or fold into the existing class.
3. **Reuse upstream layers.** Before writing a V4-specific helper,
   check if `layers/jax/{attention,moe,...}/` or `kernels/` already
   has a primitive (`dense_moe_fwd`, `sparse_attn`, `rms_norm`,
   `megablox/gmm`, `ragged_paged_attention/v3`, etc.). The custom
   `deepseek_v4_attention.py` and `deepseek_v4_moe.py` exist because
   V4's MLA + sqrtsoftplus + hash routing + top-k + sink don't fit
   the generic helpers — but verify that's still true for any *new*
   helper before duplicating.
4. **Touch the runtime sparingly.** Anything that touches the
   runtime (`runner/`, `worker/`, `platforms/`) should be a last
   resort, not a first resort. The two existing V4 runtime hooks
   (`kv_cache_manager.py`'s deepseek_v4 model_type override, the
   `deepseek_v4_fp8` quantization stub in `tpu_platform.py`) are
   the entire delta and should stay that small.
5. **Delete dead code as you go.** If a TODO has been resolved,
   drop it. If a "tier"/"keystone"/"sentinel" comment refers to a
   superseded plan, remove it. Comments rot; code stays.
6. **No re-export shim files** and no doc-stub files that just
   point at this runbook (PROGRESS.md / SUMMARY.md / FAILURES.md /
   STATUS.md / STUCK.md / CODEX_PLAN.md / PROD_TOPOLOGY_RISKS.md
   were all that pattern; if they reappear, delete them).

When in doubt: the smaller change wins. A revert + minimal patch
beats a refactor + the same fix.

This file is the durable operational knowledge that's not obvious
from the code: cluster layout, the iterate loop, env knobs, orphan
state surfaces, the prioritized backlog, and pitfalls that have
already cost real time. Read it once before doing anything;
everything below has been learned by burning iterations.

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
  (uses `scripts/_mock_openai_server.py` — 24 scenarios, no TPU
  needed). Also fires nine optional probes, each gated by an env
  knob so the default smoke stays cheap:
  * `/v1/chat/completions` — `CHAT_REQUIRED=1` to fail on
    missing/empty content (exit 4).
  * Thinking-mode chat (`chat_template_kwargs={"thinking":true}`) —
    `REASONING_REQUIRED=1` to fail when `message.reasoning` is empty
    (exit 5). Pins S3's runtime — verifies the
    `--reasoning-parser deepseek_v4` flag actually emits a
    `<think>...</think>` block into the `reasoning` field.
  * `/v1/completions` with `stream=true` — `STREAMING_REQUIRED=1`
    to fail when the reassembled SSE chunks don't byte-equal the
    non-streaming completion (exit 6). Pins S7.
  * `/v1/completions` with `temperature=0.7, top_p=0.9,
    frequency_penalty=0.1` — `SAMPLING_REQUIRED=1` to fail when
    the response text is empty/whitespace-only or the
    `finish_reason` isn't `stop`/`length` (exit 7). Pins backlog
    item S6 — verifies the sampling code path (temperature scaling
    + top-p filter + token penalties) doesn't crash or produce
    garbage on TPU.
  * `/v1/completions` with `stop=["Paris"]` — `STOP_REQUIRED=1`
    to fail when the response still contains `Paris` or
    `finish_reason` isn't `stop` (exit 8). Pins backlog item S6
    broader-matrix — verifies the stop-sequence handler actually
    truncates before the matched token. The deterministic baseline
    emits ` Paris` as its first token, so a working handler MUST
    intercept it.
  * `/v1/completions` with `logprobs=5` — `LOGPROBS_REQUIRED=1`
    to fail when any emitted position has fewer than 5 alternatives
    in `top_logprobs` (exit 9). Pins backlog item S6 broader-matrix
    — verifies the logprobs postprocessing path emits the per-token
    alternative distribution that production clients rely on for
    confidence scoring.
  * `/v1/completions` with `temperature=0.7, top_k=10` —
    `TOPK_REQUIRED=1` to fail when the response text is
    empty/whitespace-only or `finish_reason` isn't `stop`/`length`
    (exit 10). Pins backlog item S6 broader-matrix — verifies the
    top-k filter (rank-bounded candidate set, distinct from top_p's
    mass-bounded set) doesn't crash or zero-out on TPU.
  * `/v1/completions` with `temperature=0.7, presence_penalty=0.5`
    — `PRESENCE_REQUIRED=1` to fail when the response is
    empty/whitespace-only or `finish_reason` is invalid (exit 11).
    Pins backlog item S6 broader-matrix — verifies presence-penalty
    (per-token, distinct from frequency_penalty's per-occurrence
    semantics) is correctly applied on the TPU sampling path.
  * `/v1/completions` with `n=2` — `N_REQUIRED=1` to fail when the
    response has fewer than 2 choices, or any choice has empty text
    (exit 12). Pins backlog item S6 broader-matrix — verifies vLLM's
    multi-completion expansion (one request → n parallel sequences
    sharing the prompt) actually emits n choices on the TPU runner.
    Under `--max-num-seqs=1` likely sequentializes but cardinality
    must still equal `n`.
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
| `MAX_LEN` | `256` | `--max-model-len` passed to vllm serve. Production needs this lifted (see backlog A1); each step up surfaces fresh activation-HBM tightness. |
| `MAX_SEQS` | `1` | `--max-num-seqs` passed to vllm serve. Production needs this lifted (backlog S2/A1). Today S2 is unjit'd, so each extra seq is a sequential Python loop in eager mode — bumping without S2 won't help throughput. |
| `V4_LOADER_SLICE_AWARE` | `1` | Each host reads only the rows its local devices own (vs full-tensor read on every host). |
| `V4_LOADER_PLACE_WORKERS` | `8` | Threads driving `place_spec_as_jax_sharded` per host. Most per-tensor work releases the GIL (safetensors mmap reads + JAX C calls), so parallelism is real. Set to `1` for single-thread parity testing. |
| `V4_LOADER_PREFETCH_WORKERS` | `0` | Thread-pool prefetch in the non-slice-aware iterator. Empirically didn't help on real V4 (placement is the bottleneck, not dequant). Knob retained for future work. |
| `VLLM_XLA_CACHE_PATH` | `~/.cache/vllm/xla_cache` | Per-host JAX persistent compile cache. tpu_inference's `compilation_manager.py:53` calls `jax.config.update("jax_compilation_cache_dir", VLLM_XLA_CACHE_PATH)` — overriding any `JAX_COMPILATION_CACHE_DIR` env var the launcher might set. **Not GCS** — the bucket the venv mounts is shared, do not relocate the cache there without explicit user authorization. **Cross-host rsync is unsound** — SPMD compiles to host-specific binaries even when JAX's cache filename is identical (verified by `scripts/full_slice_v4_cache_fingerprint.sh`: same name, 8 distinct sha256s). |
| `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES` | `0` | Cache even small modules. JAX 0.9 config name. |
| `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` | `0` | Cache even fast-to-compile modules. Default `1.0`s skips small inits (`jit_sample`, etc.). |
| `RAY_CGRAPH_get_timeout` | `3600` | Ray compiled-graph channel timeout. Default 300 trips during the first inference if `jit_run_model` recompiles for an unseen shape (already burned us once at 5m1s). Don't lower. |
| `V4_XLA_FLAGS` | unset | Opt-in custom `XLA_FLAGS` string for one launch. The smoke script does **not** inherit `XLA_FLAGS` from the parent shell (a stale autorunner env once SIGSEGV'd every Ray worker — see pitfall #4). |
| `V4_DECODE_STATE` | `0` | S1 iter-5d/5e gate. `1` flips the model's `__call__` from the green-gate baseline (`transformer_body_forward`, every step recomputes prefill on the full prompt+generated context) to the iter-5e orchestrator (`deepseek_v4_run_with_decode_state` with the runner-tagged `decode_start_pos` driving `is_decode`). Default `0` preserves the green gate. iter-5e fixed the T-padding decode-detection bug (the runner pads decode to bucket sizes, so the previous `(T==1) and (start_pos>0)` check never triggered — empty-text completions resulted), and the decode kernel chain now executes on real V4 weights for the first time (verified 2026-04-30 16:28Z). FIRST `/v1/completions` curl returns 200 OK; SECOND request still triggers a TPU `[USER]` FATAL on the cached prefill kernel — iter-5d hypotheses 1 (kv_caches donation reuse) / 3 (XLA buffer aliasing) remain candidates for iter-5f. When `1`, every JIT trace-time entry to `__call__` logs `(call_idx, T, start_pos, is_decode, state_max_seq_len, kv_caches_count)` so the smoke log correlates compile fingerprints to argument shapes. Forwarded to Ray workers via `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`. |
| `CHAT_REQUIRED` | `0` | Default makes the smoke_check's `/v1/chat/completions` probe informational (HTTP-success best-effort). Set to `1` to make a missing/empty chat response fail the gate (exit 4). The chat path lands in a 1024-token prefill bucket vs 256 for completions and on a tight HBM budget the engine sometimes needs `TpuLoadedExecutable::ExecutePrepareWithOomRetries` to land — usually succeeds but adds ~30s to first-chat latency. |
| `REASONING_REQUIRED` | `0` | Set to `1` to fire a thinking-mode chat (`chat_template_kwargs={"thinking":true}`) with a reasoning-eliciting prompt and assert `message.reasoning` is non-empty (exit 5 on empty). Pins backlog item S3's runtime. Adds ~30s on first-chat-cold-cache (lands in chat path; same OOM-retry caveat as `CHAT_REQUIRED`). |
| `STREAMING_REQUIRED` | `0` | Set to `1` to additionally fire `/v1/completions` with `stream=true`, reassemble the SSE chunks, and assert byte-equality vs the non-streaming output (exit 6 on mismatch / no chunks). Pins backlog item S7. Cheap — same prefill bucket as the existing completions probe. |
| `SAMPLING_REQUIRED` | `0` | Set to `1` to additionally fire `/v1/completions` with `temperature=0.7, top_p=0.9, frequency_penalty=0.1` and assert the response has non-empty text + a valid `finish_reason` (exit 7 on empty / invalid). Pins backlog item S6 — verifies the sampling code path doesn't crash or produce garbage. Cheap — same prefill bucket as the existing completions probe. Determinism under sampling is **not** asserted (vLLM/TPU runner doesn't honour per-request seed for non-greedy paths; CLAUDE.md pitfall context). |
| `STOP_REQUIRED` | `0` | Set to `1` to additionally fire `/v1/completions` with `stop=["Paris"]` and assert (a) the response text does NOT contain `Paris` and (b) `finish_reason="stop"` (exit 8 on either). Pins backlog item S6 broader-matrix — verifies vLLM's stop-sequence handler on the TPU runner truncates before the matched token and reports the right reason. Cheap — same prefill bucket as the existing completions probe. The deterministic baseline emits ` Paris` as its first token at temp=0/seed=0, so a working handler MUST intercept it. |
| `LOGPROBS_REQUIRED` | `0` | Set to `1` to additionally fire `/v1/completions` with `logprobs=5` and assert every emitted position's `top_logprobs` entry has at least 5 alternatives (exit 9 on missing object / dropped alternatives). Pins backlog item S6 broader-matrix — verifies the logprobs postprocessing path on the TPU runner emits the per-token alternative distribution that production clients (confidence scoring, structured-output reranking) rely on. Cheap — same prefill bucket as the existing completions probe. |
| `TOPK_REQUIRED` | `0` | Set to `1` to additionally fire `/v1/completions` with `temperature=0.7, top_k=10` and assert non-empty text + valid `finish_reason` (exit 10). Pins backlog item S6 broader-matrix — verifies the top-k filter (rank-bounded candidate set, distinct from top_p's mass-bounded set) doesn't crash or zero-out the candidate set on the TPU sampling path. Cheap — same prefill bucket as the existing completions probe. |
| `PRESENCE_REQUIRED` | `0` | Set to `1` to additionally fire `/v1/completions` with `temperature=0.7, presence_penalty=0.5` and assert non-empty text + valid `finish_reason` (exit 11). Pins backlog item S6 broader-matrix — verifies presence-penalty (per-token, distinct from frequency_penalty's per-occurrence semantics) is correctly applied on the TPU sampling path. Cheap — same prefill bucket as the existing completions probe. |
| `N_REQUIRED` | `0` | Set to `1` to additionally fire `/v1/completions` with `n=2` and assert the response has at least 2 choices, all non-empty (exit 12 on missing/short choices / empty text). Pins backlog item S6 broader-matrix — verifies vLLM's multi-completion expansion (one request → n parallel sequences sharing the prompt) actually emits n choices on the TPU runner. Under `--max-num-seqs=1` likely sequentializes but cardinality must still equal `n`. Cheap — same prefill bucket as the existing completions probe. |

## Current state (READ BEFORE LAUNCHING)

**Tier 8 deploy gate is GREEN as of 2026-04-30 04:22Z.** Cold
`./run.sh serve` → load weights → cold compile → first
`/v1/completions` returns deterministic `Paris` for "The capital of
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

`/v1/chat/completions` returns 200 OK. vLLM auto-resolves
`tokenizer_mode='deepseek_v4'` for `DeepseekV4ForCausalLM`
(`work/vllm/vllm/config/model.py:578`) and routes the chat path
through `DeepseekV4Tokenizer` (`work/vllm/vllm/tokenizers/deepseek_v4.py`)
whose `apply_chat_template` calls the upstream `encode_messages`
encoder directly — byte-equivalent to the V4-Flash reference
encoder shipped at `<hf-snapshot>/encoding/encoding_dsv4.py`
across chat / thinking / tools / reasoning_effort. Pinned by
`TestVllmChatTemplateParity` in
`tests/models/jax/test_deepseek_v4.py`. The chat probe in
`smoke_check` is informational by default — see backlog S3 for the
deferred runtime assertion.

The smoke is `MAX_LEN=256, MAX_SEQS=1, --enforce-eager`. That's
the demo configuration, not the production configuration. See the
backlog below for what's required to widen it.

## Production-readiness backlog (READ BEFORE PICKING WORK)

This is the prioritized list of what stands between Tier-8-GREEN
demo serving and OpenRouter-grade production. Items earlier in
the list block items later, and items earlier have higher
correctness or throughput leverage. Pick the first uncompleted
item; if you can't make progress, document why in the commit
message and move down.

### Tier S — silent correctness bombs (fix before perf)

These are issues where the smoke goes green but the model isn't
doing what users will actually ask of it. Fix these *first*.

#### S1. Decode is not real decode — every step recomputes prefill on the full prompt+generated context (math + orchestrator + JIT-correctness landed iter-4; `__call__` flip + kv_cache_manager allocator is iter-5's job)

This is the headline correctness/perf bug. `__call__` in
`work/tpu-inference/tpu_inference/models/jax/deepseek_v4.py:1363`
still routes every step (prefill or decode) through
`transformer_body_forward` → `block_forward` → `attention_prefill`.
The decode-step kernels are fully implemented and
correctness-tested (`attention_decode_step` at
`layers/jax/attention/deepseek_v4_attention.py:710`,
`TestDecodeRollingParity` at
`tests/models/jax/test_deepseek_v4.py:1390` — 1e-4 vs torch
reference) but **not yet invoked** by the production `__call__`.

Net effect: every decode step is O((prompt + generated)²) in
attention compute and re-runs the full MoE per step instead of
O(1)/step over a cached state. Throughput at 1k context is
~10–50× worse than it should be; at 100k+ context decode is
non-functional, not just slow. V4-Flash's 1M-token claim is
unreachable on the current path.

**Iter 2026-04-30 (foundational primitive):** the missing
piece that blocked threading state through `__call__` is now
landed. `attention_init_state_from_prefill` in
`layers/jax/attention/deepseek_v4_attention.py` constructs an
`AttentionDecodeState` byte-equivalent to the torch reference's
post-prefill state in **closed form** — i.e., directly from x
without iterating the decode kernel T times.

Why closed-form rather than iterating the decode kernel:
`attention_decode_step` (and the compressor / indexer decode
steps it calls) take `start_pos: int` as a Python static and
use static control flow (`if did_compress`, `kv_state.at[:,
start_pos % win].set(...)`). A `lax.scan` over T positions
would require refactoring the kernels to traced-`start_pos`
form (lots of code changes, heavy regression risk on
correctness). A Python-loop unroll inside jit would copy T
graphs into HLO at compile time — for ~50 layers × T=256, that
explodes compile cost. The closed-form version mirrors torch
reference's `Compressor.forward(start_pos=0)` semantics
directly: sparse writes into the slots that prefill would
populate, leaving the rest at init. Pinned by
`TestPrefillToDecodeStateParity` (26 cases — SWA / CSA / HCA at
T values spanning window/ratio boundaries; both field-level
state parity AND rolling parity where the closed-form state
drives one decode step at start_pos=T and is compared to torch
reference's prefill+1-step output, ≤1e-4 on bf16).

Subtlety surfaced during this work and folded into the helper:
torch's `Compressor.forward` prefill path leaves the "back"
slots of `kv_state` (overlap mode) / the "right" slots
(non-overlap mode) at INIT values, while an
iterative-decode-from-zero path would leave them holding stale
data from prior writes. Both paths produce identical
compression outputs at the next compression boundary because
those slots are fully overwritten before then. The closed-form
helper deliberately matches torch prefill (sparse writes,
leave the rest at init) so the parity test fires the right
comparison.

**Iter 2026-04-30 (transformer-body primitives):** the
prefill→decode-state→step round-trip now closes at the
`transformer_body` level, not just per-layer attention. Three
new helpers in `models/jax/deepseek_v4.py`:

* `block_init_state_and_forward(x, input_ids, params, fc, …)`
  — `block_forward` plus per-layer `AttentionDecodeState`
  capture (calls `attention_init_state_from_prefill` on the
  attention input AFTER hc_pre + rms_norm, before
  attention_prefill). Output `[B, T, hc, D]` is byte-equal to
  `block_forward` on the same input.
* `block_decode_step(x_step, input_ids_step, params, fc,
  prev_state, start_pos)` — `block_forward` with
  `attention_prefill` swapped for `attention_decode_step`.
  Returns `(new_state, [B, 1, hc, D])`.
* `transformer_body_init_state_from_prefill(input_ids, params,
  swa, comp, cfg, state_max_seq_len)` — runs all layers via
  `block_init_state_and_forward`, returns
  `(h: [B, T, hc, D], states: List[AttentionDecodeState])`.
* `transformer_body_decode_step(input_ids_step, params, swa,
  comp, cfg, prev_states, start_pos)` — runs all layers via
  `block_decode_step`, returns `(h: [B, 1, hc, D], new_states)`.

`state_max_seq_len` is exposed as a parameter (not pinned to
`cfg.max_position_embeddings`) so the integrator picks vLLM's
`max_model_len` and avoids V4-Flash's 1M architectural cap
blowing per-layer kv_cache to GiB.

Pinned by `TestTransformerBodyDecodeRoundTrip` in
`tests/models/jax/test_deepseek_v4.py` (4 cases, T ∈ {4, 8, 16,
24}). For each T it runs:

  A) `transformer_body_forward(ids[:, :T+1], …)` → h_full
  B) `transformer_body_init_state_from_prefill(ids[:, :T], …)`
     → h_pref, states; assert maxabs(h_pref - h_full[:, :T])
     ≤ 1e-3 (forward output unchanged)
  C) `transformer_body_decode_step(ids[:, T:T+1], states, …,
     start_pos=T)` → h_step; assert maxabs(h_step -
     h_full[:, T:T+1]) ≤ 5e-3 (decode step matches the T-th
     position of fresh prefill)

This pins the whole chain (embed + HC + attention(prefill →
decode) + MoE + HC × all layers) end-to-end. Tolerance budget
is ≈ 6 layers × 1e-4/layer accumulated bf16 noise; observed
worst across 4 T cases ≪ 5e-3.

**Iter 2026-04-30 (pack/unpack schema):** the missing engine
plumbing primitive — concrete pack/unpack helpers that flatten
`AttentionDecodeState` into a single `jax.Array` per layer — now
landed and tested. Six new helpers in
`models/jax/deepseek_v4.py`:

* `_layer_decode_state_layout(layer_params, cfg_index_head_dim,
  state_max_seq_len, batch_size=1)` — returns the per-field
  layout `((name, shape, dtype), ...)` for one layer's
  `AttentionDecodeState`. Mirrors `attention_init_state_from_prefill`
  field decisions exactly. SWA/CSA/HCA all populate the same 6
  named fields with zero-sized placeholders for unused fields,
  so the schema is uniform across layer types (total element
  count differs).
* `_layer_packed_size(layout)` — total fp32 elements for one
  layer's packed state.
* `_pack_layer_state(state, layout)` — flatten one layer's
  `AttentionDecodeState` into a 1D fp32 array. bf16 fields
  (`kv_cache`, `indexer_kv_cache`) upcast losslessly to fp32;
  fp32 fields are stored directly. Score-state's `-inf` init
  preserved.
* `_unpack_layer_state(packed, layout)` — inverse: slice by
  field offsets, reshape, cast back to natural dtype.
* `transformer_body_layout(params, cfg, state_max_seq_len,
  batch_size=1)` — per-layer layouts for the whole transformer
  body (drives buffer allocation in iter-4).
* `transformer_body_init_state_to_buffer(input_ids, params, ...,
  state_max_seq_len)` — prefill that returns
  `(h, packed_buffers)`. Wraps
  `transformer_body_init_state_from_prefill` + pack.
* `transformer_body_decode_step_from_buffer(input_ids_step,
  params, ..., prev_buffers, start_pos, state_max_seq_len)` —
  one decode step driven by per-layer packed buffers. Wraps
  unpack + `transformer_body_decode_step` + pack.

Pinned by `TestPackedDecodeStateBuffer` in
`tests/models/jax/test_deepseek_v4.py` (4 cases):
  * pack/unpack round-trip is bit-exact across all 6 fields on
    every layer (including `compressor_score_state`'s -inf
    entries — verified via `jnp.array_equal` rather than
    `abs(diff)` because `abs(-inf - (-inf)) == NaN`)
  * one decode step from a packed buffer matches a fresh prefill
    on `ids[:, :T+1]`'s last position at T ∈ {4, 16}
    (≤ 5e-3, same budget as the underlying primitives — pack/
    unpack adds no error)
  * 3 sequential decode steps from a packed buffer (T=12, N=3)
    match the corresponding positions in a full T+N prefill —
    rules out a tail-field truncation bug that single-step would
    miss

Total round-trip overhead: pack/unpack are pure JAX ops
(reshape + cast + concat + dynamic_slice); they add no error
and inline cleanly into surrounding JIT. Per-layer packed size
on a tiny config layer is a few KB; on real V4-Flash with
`MAX_LEN=256, MAX_SEQS=1`: SWA layer ≈ 16K fp32 elements (64
KB), CSA layer ≈ 50K (200 KB), HCA layer ≈ 65K (260 KB). Total
across ~50 layers ≈ 10 MB before sharding, well below any HBM
budget.

**Iter 4 progress (commit pending, 2026-04-30):** the
orchestration layer + JIT-correctness validation now landed.
Two new pieces in `models/jax/deepseek_v4.py`:

* `deepseek_v4_run_with_decode_state(kv_caches, input_ids,
  params, freqs_swa, freqs_comp, cfg, state_max_seq_len,
  is_decode_step, start_pos)` — branches on `is_decode_step` and
  threads packed `AttentionDecodeState` through `kv_caches`.
  Returns `(updated_kv_caches, h)`. The prefill branch wraps
  `transformer_body_init_state_to_buffer` (replacing
  kv_caches's contents wholesale, since prefill starts a fresh
  state). The decode branch wraps
  `transformer_body_decode_step_from_buffer` (reads each
  `kv_caches[i]`, advances by one position, writes the new
  buffer back).

Pinned by two new tests in
`tests/models/jax/test_deepseek_v4.py::TestPackedDecodeStateBuffer`:

* `test_buffer_chain_compiles_under_jit` — wraps the iter-3
  primitives `transformer_body_init_state_to_buffer` /
  `transformer_body_decode_step_from_buffer` in `jax.jit` and
  verifies (a) compile succeeds, (b) outputs match eager within
  ≤5e-3 (bf16-reorder budget). Mid-implementation this test
  caught a real bug: `start_pos` MUST be marked
  `static_argnums` on the jit boundary because
  `attention_decode_step` indexes circular buffers with Python
  ints (`start_pos % win`). Without static_argnums the trace
  raised `IndexError: Slice entries must be static integers`.
  The test now hardcodes the right pattern; iter-5's runtime
  wiring inherits it.
* `test_run_with_decode_state_kv_caches_round_trip` —
  end-to-end: starts with zero kv_caches, runs prefill (T
  tokens), then 3 sequential decode steps (each reading +
  writing kv_caches). Each step's `h` matches the
  corresponding position in a fresh full prefill. Pins the
  iter-5 runtime call pattern: `kv_caches → __call__(prefill) →
  kv_caches' → __call__(decode) → kv_caches'' → ...`.

Total CPU pytest time on tiny config: ~125s for all 6
TestPackedDecodeStateBuffer tests. The pack/unpack ops
inline cleanly into JIT (verified by the existing pure-pack
round-trip test PLUS the new under-jit test); no surprise
all-gathers, no NaN drift on `compressor_score_state`'s -inf
init (the under-jit test masks NaN-from-`-inf-minus-(-inf)`
explicitly because `jnp.abs(-inf - (-inf)) == NaN`).

**Iter 5a (commit pending, 2026-04-30):** the cfg-based
per-layer packed-size helper + parity test landed.
`v4_layer_packed_sizes_from_cfg(cfg, state_max_seq_len,
batch_size=1) → List[int]` in `models/jax/deepseek_v4.py`
returns the fp32-element count of each main-body layer's
packed `AttentionDecodeState`, derived from `cfg` alone (no
need for loaded `params`). Mirrors `_layer_decode_state_layout`
shape decisions exactly. Pinned by
`test_v4_layer_packed_sizes_from_cfg_matches_layout` in
`TestPackedDecodeStateBuffer` (3 cases × all main-body layers
= asserts cfg-derived == params-derived for state_max_seq_len
∈ {64, 32, 16}).

This unblocks iter-5b's runtime allocator: kv_cache_manager
needs to size the fp32 packed buffers BEFORE the V4 model's
weights are loaded — it has `runner.model.config` available
but `runner.model.params_v` may still be abstract at that
point. The cfg-only helper avoids that race.

**Iter 5a — verified-WRONG hint from iter-4 plan:** the iter-4
recommendation to use shape `[1, packed_size, 1]` with the
existing `kv_cache_sharding=P(ATTN_DATA, None, ATTN_HEAD)`
"because size-1 axes become replicated" does NOT compile on
the v6e-32 mesh. Verified on a virtual `8 × 4` CPU mesh:

```
jax._src.sharding.IndivisibleError: Sharding NamedSharding(...
P('attn_dp', None, 'attn_head'), ...) implies that array axis
0 is partitioned 8 times, but the dimension size is 1
```

JAX requires each sharded axis to be evenly divisible by the
mesh axis it's sharded along — there's no implicit "size-1 ⇒
replicate" fallback. iter-5b must use a different approach.

**Iter 5b — corrected scope (S1 still NOT runtime-integrated):**

V4's `kv_caches` is a passthrough today (INVARIANTS I34); the
model's `nnx.Variable`s are read inside JIT but mutations are
lost because `tpu_inference/models/common/model_loader.py:332`
returns `(kv_cache, hidden, aux)` without capturing the mutated
nnx state. Use `kv_caches` as the carrier (the standard pattern
in `llama3.py:338-344`).

Two real blockers to resolve in iter-5b:

  **B1. kv_cache_sharding is hardcoded for non-V4 4D layout.**
  `tpu_inference/models/common/model_loader.py:308-311` defines
  `kv_cache_sharding = NamedSharding(mesh, P(ATTN_DATA, None,
  ATTN_HEAD))`, applied as both input-donation and out_shardings
  for `run_model`. V4's packed buffers are 1D `[packed_size]`
  fp32 — none of the 3D-sharding-spec axes map cleanly. Two
  resolution paths:

    (a) **Detect V4 in `get_flax_model` and override
       `kv_cache_sharding` to `NamedSharding(mesh, P())`
       (replicated).** Smallest change: ~10 lines in
       model_loader. Each chip stores its own copy of the
       packed buffer (~50 KB/layer × ~50 layers × 32 chips
       = ~80 MB total — negligible vs 31 GB/chip budget). All
       chips compute the same buffer update under SPMD (the
       packed state is per-sequence, not sharded), so
       replication is correct, not wasteful in the bad sense.

    (b) **Allocate `[mesh.attn_dp, packed_size,
       mesh.attn_head]` fp32 with `P('attn_dp', None,
       'attn_head')` and treat all shards as identical.**
       Avoids the model_loader change but wastes 32× HBM per
       layer and complicates `__call__` (must pick one shard's
       view). Strictly worse than (a). Skip.

  Pick (a). Verify it compiles by allocating a 1D fp32 array
  with `P()` on a 32-virtual-CPU mesh and inspecting the shard
  count == 32 (replicated).

  **B2. `start_pos` is required as a Python int by every
  decode kernel** (`attention_decode_step` →
  `freqs_cis_full[start_pos:start_pos+1]` Python slice;
  `compressor_decode_step` → `if did_compress:` Python branch
  on `(start_pos+1) % ratio == 0`; `indexer_decode_step` → `K
  = min(params.index_topk, end_pos_div_ratio)` Python compare;
  `get_window_topk_idxs_decode` → Python branches +
  `jnp.arange` with traced bounds). Inside the JIT'd
  `run_model`, `attention_metadata.input_positions` is a
  tracer; `np.asarray(traced_positions)` raises. Two paths:

    (a) **Refactor decode kernels to accept traced
       `start_pos`.** Replace `arr[a:a+1]` →
       `lax.dynamic_slice_in_dim(arr, a, 1, axis=0)`,
       `if did_compress` → `jnp.where(did, computed, original)`
       with always-compute-then-mask, `K = min(...)` →
       always-K via `lax.top_k` + invalid-mask. Substantial
       (~6 sites × careful per-site refactor) but produces
       ONE compile that handles all decode positions. Right
       long-term answer.

    (b) **Pass `start_pos` as a Python static via meta_field
       on AttentionMetadata.** `meta_fields` are part of the
       pytree's auxiliary data, hashed for the JIT cache key.
       Per-position compile (~50–100 s cold per unique
       value); persistent cache amortizes after first run.
       For 8-token smoke fresh: 1 prefill + 8 decode compiles
       ≈ 7–15 min added to first launch. Fast on subsequent
       runs. Touches `tpu_inference/layers/common/attention_metadata.py`
       (1 line: add `decode_start_pos: int = 0` meta_field) +
       `tpu_inference/runner/tpu_runner.py` (~5 lines per
       prepare_inputs path: read CPU mirrors, set the field
       when V4-decode). Other models default to 0 — single
       cache key, no perf impact.

  Tactical pick: **(b) for iter-5b**, defer (a) to a
  future kernel-perf iter (the kernel refactor's correctness
  risk is high enough to merit its own iter, and (b) lands
  the runtime correctness immediately).

**Iter 5b status (2026-04-30): runtime infra LANDED + GREEN; the
final `__call__` flip is GATED — produces NaN logits on real
V4-Flash.**

What landed and is verified green on real `vllm serve`:
  1. **model_loader.py**: V4 detection +
     `kv_cache_sharding = NamedSharding(mesh, P())` override.
     Non-V4 unchanged.
  2. **kv_cache_manager.py**: V4-only `_initialize_kv_cache_deepseek_v4`
     allocates 1D fp32 `[packed_size_i]` buffers per layer, sized
     via `v4_layer_packed_sizes_from_cfg` + new helper
     `v4_state_max_seq_len_from_vllm_config`, sharded P()
     replicated. Real-smoke log line:
     `Init kv-cache (DeepSeek V4) | num_layers=43 |
     state_max_seq_len=8192 | total_packed_bytes=136200192
     (~129.9 MB) | sharding=replicated`. Allocator runs in
     under a second; hbm change negligible.
  3. **attention_metadata.py**: `decode_start_pos: int = 0` as
     a meta_field (Python-static; hashed into JIT cache key).
     Default 0 = prefill / non-V4. Pytree round-trip verified.
  4. **tpu_runner.py**: `_maybe_set_v4_decode_start_pos` helper
     called after `build_attn` in both `_prepare_inputs_dp` and
     `_prepare_inputs_non_dp`. Sets
     `decode_start_pos = seq_lens_cpu[0] - 1` when V4 + the
     request shape is decode (query_len[0]==1, seq_lens[0]>1).
     Cached `runner._is_deepseek_v4` flag avoids per-request
     hf_config lookup.
  5. **module-level `v4_state_max_seq_len_from_vllm_config`**:
     single source of truth so the engine allocator and the
     model __call__ agree on buffer size. Mirrors
     `_effective_freqs_seq_len`'s decision tree.
  6. **scripts/full_slice_v4_smoke_check.sh**: new
     `COMPLETION_MAX_TOK` env knob (defaults 8 unchanged) — under
     iter-5b each new decode position triggers a fresh JIT trace,
     so smoke validation against the orchestrator path benefits
     from a smaller MAX_TOK to fit under `CURL_MAX_TIME`.

What was attempted and **REVERTED at the source** because it
breaks real V4-Flash:
  7. **deepseek_v4.py `__call__` flip** to call
     `deepseek_v4_run_with_decode_state(kv_caches, ids_2d,
     params, ..., is_decode_step, start_pos)` instead of
     `transformer_body_forward`. Tiny-config tests pass byte-
     equal vs `transformer_body_forward` at all positions
     (verified by /tmp/test_v4_orchestrator_padded.py with
     T_padded=64, N_actual=5: maxabs=0, no NaN). Virtual 32-CPU
     mesh `lower().compile()` succeeds with replicated `P()`
     kv_caches (verified by /tmp/test_v4_iter5b_compile.py).
     But on the real v6e-32 + V4-Flash + bf16 + replicated SPMD:

       * `/v1/completions` returns 200 with `text=""`,
         `finish_reason="length"`, `completion_tokens=4` —
         the model emits 4 tokens that all decode to empty
         (likely BOS-id 0 repeated).
       * Adding `logprobs=1` triggers
         `HTTP 400: ValueError: Out of range float values are
         not JSON compliant: nan` — the logits at the prompt's
         last position contain NaN values. The argmax somehow
         lands on an empty-decode token (BOS) instead of " Paris".
       * Reproducer (smoke up):
         ```bash
         curl -s --max-time 60 http://127.0.0.1:18081/v1/completions \
           -H "Content-Type: application/json" \
           -d '{"model":"deepseek-ai/DeepSeek-V4-Flash",
                "prompt":"The capital of France is",
                "max_tokens":4,"temperature":0,"seed":0,
                "logprobs":1}'
         ```

     Root cause UNDIAGNOSED. CPU-tiny tests pass. Suspects:
       * 43-layer bf16 accumulation on real weights might
         exhibit a NaN that doesn't show on tiny-config tests
         (which use 6-layer fp32-on-CPU) — but the OLD path
         through `transformer_body_forward` works, and the NEW
         path is supposed to be byte-equal forward + extra state
         capture. So bf16 drift alone isn't a complete story.
       * `attention_init_state_from_prefill` allocates extra
         intermediates (kv_state, compressor_score_state,
         indexer_kv_state) that may interact with XLA's
         operation reordering under SPMD in a way that changes
         the effective reduction order in `attention_prefill`.
       * The HCA layer's compressor_score_state init at -inf,
         when `state_max_seq_len=8192` (vs T_padded=256), means
         many slots stay at -inf. Some downstream reduction
         (logsumexp?) might hit `exp(-inf) - exp(-inf) = NaN`
         if a row of all -inf is consumed.

     `models/jax/deepseek_v4.py::__call__` is reverted to the
     pre-iter-5b path: `transformer_body_forward(ids_2d, ...)`
     for prefill, `kv_caches` passed through unchanged. The V4
     allocator from #2 still runs (kv_caches are 1D fp32
     buffers carrying no semantic load yet); the runner still
     tags `decode_start_pos` (consumed by no one, but kept so
     the `v4_state_max_seq_len_from_vllm_config` agreement is
     wired end-to-end).

**Iter 5c status (2026-04-30): orchestrator decoupling fix LANDED;
__call__ flip GATED — TPU `[USER]` FATAL on second request after
first request returns 200 OK.**

What landed in iter-5c:
  * `deepseek_v4_run_with_decode_state` prefill branch now
    computes `h` via `transformer_body_forward` (path A — the
    proven baseline) and packed_buffers via
    `transformer_body_init_state_to_buffer` SEPARATELY. The
    orchestrator's own `h` from init_state_to_buffer is
    discarded. XLA CSEs the shared kv/compressor intermediates
    between the two computations so the duplicated surface is
    largely free at compile time. Critically, this guarantees
    the prefill `h` is byte-equal to `transformer_body_forward`
    BY CONSTRUCTION — not subject to XLA's reduction-order
    drift between the forward path and the closed-form
    state-init's intermediate allocations. iter-5b's NaN
    hypothesis (XLA reorders bf16 reductions when the
    state-init's extra fp32 buffers feed the same scheduler
    pass as the forward path) is solved by this decoupling.

  * Pinned by `TestPackedDecodeStateBuffer` (7 cases) +
    `TestTransformerBodyDecodeRoundTrip` (4 cases) +
    `TestPrefillToDecodeStateParity` (26 cases): all pass on
    tiny config CPU pytest unchanged from iter-5b — the
    decoupling preserves the prefill `h` semantics.
    `/tmp/test_v4_iter5c_compile.py` confirms under
    32-virtual-CPU mesh w/ replicated `P()` kv_caches: prefill
    h diff vs path-A = 0.000002 (≪ 5e-3 budget); 32 identical
    shards; all `h` finite.

What was attempted and **REVERTED at the source** because it
crashes the engine on the second request (real V4 `vllm serve`,
2026-04-30 14:40Z):
  * **`__call__` flip to invoke `deepseek_v4_run_with_decode_state`
    on the single-active-seq path** (same flip iter-5b made +
    reverted, now with iter-5c's path-A-h orchestrator).
    First `/v1/completions` (max_tokens=4) succeeds — `POST
    /v1/completions HTTP/1.1 200 OK` at 14:40:04. ≈90 s of
    wall-time = 1 prefill compile (`fingerprint
    2588072356c02f4008a3c8aeec911c6d89821ccbd2b368f645edac6cfe32f938`)
    + 3 decode compiles each fresh per `start_pos` (per-decode-
    position recompile: see iter-5b "Tactical pick (b)").
    Generation throughput logged 0.1 tok/s under cold compile.

    Then the smoke_check fires the second `/v1/completions`
    (byte-equality probe) ≈1 s later. Within ≈1 s of arrival,
    the TPU emits
    `async_driver.cc:779] [/dev/vfio/2 tpu1:pe2:3] vf_id:0 !!!!
    FATAL ERROR !!!! observed errors are: [USER]. Now taking a
    TPU core dump...` — pure async-driver dump, no Python
    traceback, engine actor dies. Subsequent requests 500.

    Repro (after a fresh `vllm serve` with __call__ flipped):
    ```bash
    # First request — succeeds
    curl -s --max-time 1500 \
        http://127.0.0.1:18081/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"deepseek-ai/DeepSeek-V4-Flash",
             "prompt":"The capital of France is",
             "max_tokens":4,"temperature":0,"seed":0}'
    # Second identical request — TPU [USER] FATAL within ~1s
    ```

    Same iter-5c log: `logs/full-slice-v4-smoke-20260430T143050Z.log`.

  Hypotheses for the second-request crash (iter-5d's lane):
    1. **kv_caches state leakage under JIT donation across
       requests.** Request 1's last decode step writes packed
       buffers into kv_caches. Request 2's prefill is supposed
       to OVERWRITE these via the closed-form init's sparse
       writes (slots not touched stay at the closed-form's
       init values: zero or -inf), but under XLA's
       `donate_argnums=2` the donated buffer is reused. If some
       intermediate XLA pass reads-before-write on a slot that
       was -inf or NaN-tainted from request 1, the bf16-cast
       could surface a sentinel that triggers the TPU error.
    2. **decode_start_pos meta_field stale across requests.**
       The runner's `_maybe_set_v4_decode_start_pos` only sets
       the field on the decode-shape branch and DOES NOT reset
       it for prefill. If the same AttentionMetadata instance
       is reused across requests (vs. freshly allocated per
       request), request 2's prefill gets request 1's
       last-decode `start_pos`. Even though `__call__`'s
       `is_decode = (T==1) and (start_pos>0)` would still
       correctly route a prefill (T=256, not 1) to the prefill
       branch, the JIT cache key differs from request 1's
       prefill (whose start_pos was 0) → fresh compile, possibly
       triggering the TPU error.
    3. **Buffer aliasing under iter-5c's doubled prefill.**
       The decoupled prefill computes `h` AND `packed_buffers`
       in the same JIT'd block. Under SPMD with `P()`-replicated
       kv_caches output, XLA may decide to alias intermediate
       buffers between the two sub-computations in a way that
       interacts with the donated input on the second call.

  Recommended iter-5d scope (in order of cheapness):
    1. **Verify hypothesis 2** by adding a debug `jax.debug.print`
       on `decode_start_pos` at __call__ entry. If non-zero on
       request 2's prefill, fix tpu_runner's helper to reset
       the field to 0 on the prefill branch.
    2. **Verify hypothesis 1** by zeroing kv_caches BEFORE
       request 2's prefill computation begins (instrument vLLM
       runtime, or change orchestrator's prefill branch to
       explicitly `jnp.zeros_like(kv_caches[i])` before the
       sparse writes — just as a NaN-mitigation experiment).
    3. **Verify hypothesis 3** by computing `transformer_body_forward`
       FIRST in the orchestrator, materializing `h`, then
       computing the state init in a SEPARATE JIT call.
       (Probably overkill but bounds the search.)

**Iter 5d status (2026-04-30): hypothesis 2 RULED OUT;
flip is now env-gated behind `V4_DECODE_STATE`; default smoke
stays green; future iters can test fixes by toggling the
env var.**

What landed in iter-5d:
  * **Hypothesis 2 ruled out by code reading**:
    `tpu_runner.py:1634` constructs a fresh `AttentionMetadata`
    on every scheduler step (every call to `build_attn`). The
    dataclass field `decode_start_pos: int = 0` defaults to 0
    on each new instance. The runner's
    `_maybe_set_v4_decode_start_pos` helper only mutates the
    field on the decode-shape branch (q0==1 && s0>1); request
    2's prefill (q0=N, fresh AttentionMetadata) inherits the
    default 0, NOT request 1's last-decode value. So
    hypothesis 2 ("stale decode_start_pos carries over from
    request 1") is dead — the field is reset to 0 by virtue
    of fresh allocation, no helper change needed.

  * **`__call__` flip is env-gated behind `V4_DECODE_STATE=1`**.
    Module-level constant
    `V4_DECODE_STATE_ENABLED = os.environ.get("V4_DECODE_STATE",
    "0") == "1"` (read at module import). When `1`, __call__
    routes through `deepseek_v4_run_with_decode_state` (the
    iter-5c orchestrator) and threads packed
    `AttentionDecodeState` through `kv_caches`. When `0`, the
    legacy green-gate path (`transformer_body_forward`, kv_caches
    passes through unchanged) is unchanged. The env var is
    forwarded to Ray workers via `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`
    and echoed at smoke startup.

  * **Diagnostic logging** at __call__ entry when flip is
    active: a Python `logger.info` line per JIT trace records
    `(call_idx, T, start_pos, is_decode, state_max_seq_len,
    kv_caches_count)`. Since `__call__` runs at JIT trace time
    only (the function is wrapped by `@jax.jit run_model` in
    `models/common/model_loader.py:344`), the log fires per
    fresh JIT cache key — i.e. once per unique
    (start_pos, is_decode) combination. A request 2 prefill
    that hits the cached request-1-prefill kernel won't log,
    which itself is signal: if the FATAL still fires on a cache
    hit, the bug is in the COMPILED ARTIFACT (likely
    hypothesis 1 or 3), not in trace-time argument shapes.

  * Pinned by the existing 37-test iter-5 suite
    (`TestPackedDecodeStateBuffer` + `TestPrefillToDecodeStateParity`
    + `TestTransformerBodyDecodeRoundTrip`): all pass on tiny
    config CPU pytest unchanged from iter-5c (172 s wall).
    Default `V4_DECODE_STATE=0` exercises only the legacy
    `transformer_body_forward` path, identical bytewise to
    pre-iter-5d behaviour.

What is NOT yet done in iter-5d:
  * Real-V4 smoke with `V4_DECODE_STATE=1` was **not** run
    this iter — the iter ran out of budget after the env-gate
    + diagnostic landed. The next iter (5e) should:
      1. Set `V4_DECODE_STATE=1` in the smoke launcher (one
         env edit, no code change), run the smoke, capture the
         log.
      2. Read the `[V4_DECODE_STATE] __call__ #N: …` log lines
         at trace boundaries — this gives a list of unique
         (start_pos, is_decode) seen.
      3. Cross-reference against the compile fingerprints in
         the smoke log. If the FATAL fingerprint matches a
         compile from request 1 (cache hit on request 2), the
         bug is in the compiled artifact's runtime behavior on
         a re-execution — pursue hypothesis 1 (donation reuse)
         or 3 (buffer aliasing). If the FATAL fires during a
         FRESH trace from request 2's prefill, that's a new
         JIT cache miss path that didn't exist in iter-5c's
         smoke — pursue from there.
  * Either way, the iter-5d checkpoint makes the next test
    cheap: one env-var flip away from running on real V4.

**Iter 5e status (2026-04-30): T-padding decode-detection bug
DIAGNOSED + FIXED; decode kernel chain executes on real V4 for
the first time; second-request FATAL persists.**

Iter-5e ran the env-gated `V4_DECODE_STATE=1` smoke that iter-5d
deferred and immediately surfaced a NEW root cause: the model's
decode-detection check `is_decode = (T == 1) and (start_pos > 0)`
NEVER triggers on real V4 because the TPU runner pads decode
calls to bucket sizes (T=64 typical for our smoke), so T==1 is
never seen. The diagnostic logs from iter-5d's instrumentation
made the bug visible:

```
__call__ #1: T=256 start_pos=0 is_decode=False ...   (R1 prefill)
__call__ #1: T=64  start_pos=5 is_decode=False ...   (should be decode!)
```

Both R1 and R2 in iter-5e's first smoke were served entirely by
those two compiled kernels — the orchestrator's PREFILL branch
ran for what should have been decode steps, generating logits
on padded zero-input. Both requests returned 200 OK with EMPTY
text — the model's argmax landed on BOS-id (token 0 → empty
string) at the only real position. iter-5b's "text="" with
finish_reason=length" repro is the same bug surfacing — iter-5c's
path-A-h decoupling fixed prefill correctness but the decode
branch was never actually entered.

iter-5e fix (in `models/jax/deepseek_v4.py` `__call__`'s
single-active-seq path under `V4_DECODE_STATE_ENABLED`):

  * Use `decode_start_pos > 0` directly as the decode signal.
    The runner's `_maybe_set_v4_decode_start_pos` already
    encodes the q0==1/s0>1 contract; a non-zero value means
    "this is a decode call regardless of T padding".
  * Slice `ids_for_orchestrator = ids_2d[:, 0:1]` for the
    decode branch — `tpu_runner.py:1475` zeroes the rest of
    the bucket via `input_ids_cpu[total_num_scheduled_tokens:] = 0`,
    so position 0 is always the real query token.
  * Pad the decode-branch `h` (shape `[B, 1, hc, D]`) back to
    `[B, T, hc, D]` with zeros before `head_hc`, so the
    downstream `hidden_TD = h_BSD.reshape(B*T, D)` keeps the
    runner-expected `[T_padded, D]` shape across the JIT
    cache key. The runner samples logits[0] for decode and
    ignores the padded positions (`head_hc` on a zero
    `[B, T-1, hc, D]` slice produces zero output by
    `rms_norm(0) = 0` propagation, no NaN risk).

Verified on real V4-Flash 2026-04-30 16:28Z (3rd smoke,
post-ray-restart, `logs/full-slice-v4-smoke-20260430T162251Z.log`):
the fresh diagnostic shows the decode trace firing for the
FIRST TIME on real weights —
`__call__ #1: T=64 start_pos=5 is_decode=True ...` at 16:28:52,
immediately followed by a fresh `jit_run_model` compile of
30,764 HLO instructions (vs prefill's ~80k — consistent with
the smaller decode kernel). The compile completed
successfully, R1 returned `POST /v1/completions HTTP/1.1 200
OK`, and `smoke_check` advanced to fire R2.

The decode kernel chain (`block_decode_step` →
`attention_decode_step` / `compressor_decode_step` /
`indexer_decode_step`) ran on real V4-Flash bf16 weights
without FATAL, NaN, or kernel-side error — pinning that the
iter-3-iter-5a math (closed-form prefill→state init + per-
position decode round-trip, ≤5e-3 vs torch reference on tiny
config) holds at scale.

**The second-request FATAL persists**: ≈3 s after R2 fired
(16:29:37Z), the TPU emits the same `[USER] FATAL` we saw in
iter-5c's repro — `tpu1:pe2:3` async_driver.cc:779. R2 is
hitting the cached prefill kernel from R1 (no new
`[V4_DECODE_STATE]` log line fires before FATAL = cache hit
on the same compiled artifact that just succeeded for R1).
This is iter-5d hypothesis 1 / 3 territory — the COMPILED
ARTIFACT misbehaves on RE-EXECUTION when its donated
kv_caches input contains data from a prior call. iter-5d's
hypothesis-2 ruling-out (stale meta_field) still holds —
`decode_start_pos=0` resets per-AttentionMetadata
construction.

iter-5e ALSO surfaced a non-deterministic FATAL on R1 in the
2nd smoke (no ray restart between 1st and 2nd) — same
prefill-orchestrator HLO that iter-5e first smoke and iter-5e
3rd smoke both ran without R1 FATAL. After ray restart, R1
was stable. Likely TPU-state leakage from a prior process —
iter-5d's hypothesis 1 / 3 may have a shared root cause.

**Iter 5f status (2026-04-30): hypothesis 1 RULED OUT —
dropping donation REGRESSES R1, not a fix.**

iter-5f tested CLAUDE.md's hypothesis 1 ("the donated kv_caches
buffer reuse is what trips the cached prefill artifact on R2").
Implementation: V4-only branch in
`model_loader.py:347-358` that omits `donate_argnums=2` for
`_is_deepseek_v4` (the same flag that already gates the
`kv_cache_sharding=NamedSharding(mesh, P())` override 25 lines
above). Non-V4 unchanged.

Tested on real V4-Flash + `V4_DECODE_STATE=1` after a fresh
ray-restart (`logs/full-slice-v4-smoke-20260430T172651Z.log`).
Timeline:
  17:32:31Z  Application startup complete (weight load + JIT
             setup, ~5min 40s from launcher)
  17:32:55Z  __call__ #1: T=256 start_pos=0 is_decode=False
             state_max_seq_len=8192 kv_caches=43
             (R1's prefill orchestrator JIT trace; FRESH compile
              because dropping donation changes the cache key)
  17:33:14Z  TPU [USER] FATAL on tpu6:pe2:2 (10.164.0.45)
             during R1's compiled-prefill execution
  17:33:25Z  Session master detects SLICE_FAILURE_SW_INJECT_ERROR;
             cluster shuts down
  smoke_check times out on R1 — R2 never fires

Compare iter-5e (with donation, same orchestrator code):
  R1 succeeded (200 OK, " Paris" returned) on a freshly-compiled
  artifact, then R2 FATAL'd on the CACHED prefill artifact ~3s
  after firing.

Conclusion: dropping donation produces a DIFFERENT compiled
artifact (different JIT cache key) that has its own runtime
bug on FIRST execution. Donation alone is not the root cause;
the "with donation" artifact has a re-execution bug, the
"without donation" artifact has a first-execution bug. Both
manifest as the same TPU [USER] FATAL signature.

iter-5f hyp-1 reverted at the source (commit 571a82f3).

**Notes for iter-5g:**

  * **hyp-1 (drop donation) is dead** — confirmed regression.

  * **hyp-3 (zero kv_caches at orchestrator prefill entry) is
    HARDER than CLAUDE.md's iter-5e write-up suggested.** The
    orchestrator's `packed_buffers` output has NO data
    dependency on input `kv_caches`
    (`transformer_body_init_state_to_buffer` derives everything
    from `input_ids`/`params`/`freqs`), so XLA will DCE any
    `jnp.zeros_like(kv_caches)` we add unless we manufacture an
    artificial dependency that itself changes the HLO. A
    `b + 0 * z` pattern is trivially DCE'd. A `dynamic_update_slice`
    against the input buffer would require restructuring the
    orchestrator. iter-5g should weigh whether hyp-3 in this
    form is worth the surgery vs the diagnostic alternative
    below.

  * **`jax.debug.print` instrumentation (CLAUDE.md iter-5f
    option 3) is the next viable path.** Add explicit reads of
    `kv_caches[0]`'s sum / first-element / nonfinite-count at
    orchestrator entry and exit. XLA can't DCE `jax.debug.print`
    side-effects, so the values WILL be observed at runtime.
    Comparing R1's prints (success) to R2's prints
    (pre-FATAL) should pin whether prior-call data really
    is bleeding through, or whether the bug is in the
    orchestrator's intermediate computations independent of
    input contents. Caveat: adding prints CHANGES THE HLO, so
    a new compile fingerprint fires — a "passes with prints"
    result doesn't necessarily mean "the bug is fixed", it
    might just mean "the prints disturb XLA's scheduling enough
    to dodge the FATAL".

  * **Sharding-constraint angle (new for iter-5g):** the
    orchestrator's `packed_buffers` are produced by
    `_pack_layer_state` (reshape + cast bf16→fp32 + concat).
    Each per-field array's sharding is inherited from
    intermediate state arrays, which inherit from
    `attn_dp`/`attn_head`-sharded params. The JIT's
    `out_shardings=(P(), ...)` for V4 forces a final reshard
    to replicated. With donation, XLA must produce the output
    INTO the donated `P()` buffer; without donation, it
    allocates a fresh `P()` buffer. The reshard implementation
    may differ. Try adding
    `with_sharding_constraint(b, P())` on each `packed_buffer`
    inside `transformer_body_init_state_to_buffer` BEFORE the
    final return, so the reshard happens at a known point in
    the code rather than at the JIT boundary. If R1 + R2 both
    pass, the bug was in the JIT-boundary reshard.

  * **Non-determinism caveat:** iter-5e itself observed a
    non-deterministic R1 FATAL on its 2nd smoke without
    ray-restart (after a 1st smoke that had R1 succeed +
    R2 fail). iter-5f's R1 FATAL was on a fresh ray-restart,
    so it's not the same nondeterminism — but the SLICE_FAILURE
    pattern persisting across requests means a single-FATAL
    observation doesn't prove "this artifact ALWAYS FATALs".
    Re-running the same configuration multiple times after
    ray-restart between each is the safest way to claim a
    reliable result.

iter-5f burned ~12 min of cluster time (one ray-restart + two
smoke launches; the first hit a TPU `Halt is unexpected` error
during weight loading, which is the known TPU-state-leakage
pattern from CLAUDE.md iter-5e — required ray-restart to clear)
but yielded a firm result that hyp-1 is dead.

S1 still unlocks A1 (lift `max-model-len`), B1 (sparse_attn
Pallas becomes worthwhile), and S5 (MTP speculative decoding
becomes meaningful).

#### S2. Multi-sequence dispatch is a Python loop in eager mode

`__call__` at `deepseek_v4.py:1438-1475` runs each active
sequence sequentially through `transformer_body_forward`. The
multi-seq tests at `tests/models/jax/test_deepseek_v4.py:822`
(`TestConcurrentMultiSeqDispatch`) verify per-seq isolation
correctness, not throughput.

Production needs a ragged-batch jit'd kernel. The
ragged-paged-attention v3 kernel at
`work/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py`
exists but doesn't support V4's top-k + attn_sink + dual-buffer
KV layout (see DECISIONS D2). Either extend that kernel or jit
V4's path with `lax.dynamic_slice` per active seq slot padded to
a static bound.

Until S2 lands, `--max-num-seqs=1` in
`scripts/full_slice_v4_smoke.sh` is forced — one user blocks all
others.

S2 can land independently of S1 but they multiply each other:
real concurrency only matters once decode is fast.

#### S3. `--reasoning-parser deepseek_v4` and `--tool-call-parser deepseek_v4` enabled in smoke launcher (launcher + reasoning runtime probe DONE; tool-call runtime probe deferred)

`work/vllm/vllm/reasoning/__init__.py:31-32` registers
`deepseek_v4 → DeepSeekV3ReasoningParser` (handles
`<think>...</think>` block extraction →
`reasoning` field on `ChatMessage`).
`work/vllm/vllm/tool_parsers/deepseekv4_tool_parser.py`
provides `DeepSeekV4ToolParser` (DSML tool tokens → `tool_calls`).
Both are wired into `scripts/full_slice_v4_smoke.sh` via
`--reasoning-parser deepseek_v4`,
`--enable-auto-tool-choice`, `--tool-call-parser deepseek_v4`.
vLLM validates these names at startup and refuses to launch if
they're misregistered, so the smoke gate green = parsers loaded.

**Reasoning runtime probe (scaffolded; currently failing on real V4)**:
`scripts/full_slice_v4_smoke_check.sh` accepts `REASONING_REQUIRED=1`,
which fires a thinking-mode chat
(`chat_template_kwargs={"thinking":true}` + a multiplication prompt)
and asserts the non-whitespace length of `message.reasoning` is > 0
(exit 5 on empty/whitespace-only). Default off so the cheap smoke
gate stays cheap. Mock-server self-test
`scripts/test_smoke_check_harness.sh` covers all three
end states: present-reasoning passes; empty-reasoning fails;
whitespace-only-reasoning fails (the third caught a real bash
gotcha — `$(...)` strips trailing newlines, so a `-z` check passed
on a 96-newline payload; the probe now uses `len(reasoning.strip())`
inside Python). The field on `ChatMessage` is `reasoning`
(vLLM-specific, defined at
`work/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:64`),
not `reasoning_content` — the latter is an *input* field used by
vLLM's chat_utils to round-trip prior assistant turns.

**REAL V4 BUG SURFACED — thinking-mode produces degenerate output**:
On a successful `vllm serve` (Tier-8 GREEN, completions return Paris),
firing the thinking-mode chat probe at `temperature=0, seed=0`
returns HTTP 200 + a `reasoning` field containing **N newlines**
(N == max_tokens, finish_reason=length) and an empty `content`
field — verified at `MAX_LEN=256, max_tokens=96` and
`max_tokens=64`, with both `What is 17 multiplied by 23?` and
`What is 17 * 23?` prompts. Switching to `temperature=0.7` (no
seed; `JAX does not support per-request seed`) produces incoherent
random tokens (`packagepackage`, `[201`, `﻿#include`, ...) —
the logits distribution is approximately uniform, suggesting
attention or MoE state is producing near-zero/garbage activations
specifically in thinking-mode prompts. Adding
`reasoning_effort:"high"` to `chat_template_kwargs` does NOT help
(same N-newline output at temp=0).

This is a Tier-S correctness bomb that the *original* smoke gate
silently missed: regular chat works, completions work, but
thinking-mode is broken. The reasoning runtime probe now catches
it. Possible causes (none investigated yet):
1. Chat-template encoding is correct (pinned by
   `TestVllmChatTemplateParity` byte-equality vs reference) — so the
   *prompt* the model sees is right.
2. The model's behavior after the trailing `<think>` token may be
   sensitive to KV-cache state in a way the prefill-only path
   (S1: every step recomputes prefill) gets wrong — e.g. greedy
   decode at every step keeps re-rolling from the prefill output
   rather than from a continuing decode state, so a flat-logits
   regime persists and the most-likely token is `\n` over and over.
   If true, S1 (real decode) likely fixes thinking-mode too.
3. MoE expert routing for the `<think>`-conditioned activations
   may hit a flat/dead-expert region. The vectorized-MoE math is
   pinned byte-equal vs per-expert reference on tiny fixture, but
   the tiny fixture doesn't include thinking-mode.
4. FP4-quantized expert weights for the thinking-mode path may
   have lost too much precision. A bf16 unquantized run would
   isolate this.

Reproducer (smoke up; expects all-newlines reasoning):
```bash
curl -s --max-time 600 http://127.0.0.1:18081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash",
       "messages":[{"role":"user","content":"What is 17 * 23?"}],
       "max_tokens":96,"temperature":0,"seed":0,
       "chat_template_kwargs":{"thinking":true}}'
```

**Tool-call runtime probe (still TODO)**: an analogous
`TOOLS_REQUIRED=1` probe that sends a request with `tools=[...]`
plus a tool-eliciting user message and asserts
`message.tool_calls` is non-empty. Prompt design is harder than
reasoning (need to land on a prompt where the model reliably
chooses to call a tool at temp=0); follow-up.

Sanity check that the parsers are still wired (no TPU needed):

```bash
PYTHONPATH=work/vllm:work/tpu-inference work/vllm_env/bin/python3 -c "
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
ReasoningParserManager.get_reasoning_parser('deepseek_v4')
ToolParserManager.get_tool_parser('deepseek_v4')
print('OK')"
```

#### S4. Chat encoding — RESOLVED upstream by `DeepseekV4Tokenizer` (kept here as a regression boundary)

vLLM's upstream `DeepseekV4Tokenizer`
(`work/vllm/vllm/tokenizers/deepseek_v4.py`) auto-loads for
`DeepseekV4ForCausalLM` (`work/vllm/vllm/config/model.py:578`
sets `tokenizer_mode='deepseek_v4'`) and its `apply_chat_template`
calls the upstream encoder
`work/vllm/vllm/tokenizers/deepseek_v4_encoding.py::encode_messages`
— byte-identical to the reference encoder shipped with the
V4-Flash snapshot at `<hf-snapshot>/encoding/encoding_dsv4.py`.
The custom tokenizer **ignores any `--chat-template` arg** and
covers all four scopes the original S4 flagged:

* `tools` array — encoded by prepending a system message carrying
  the tools list (matches reference)
* `tool_calls` round-trip in multi-turn assistant turns — handled
  by `merge_tool_messages` + `tool_calls_template` in the encoder
* `thinking` kwarg (boolean; `enable_thinking` accepted as alias)
  → emits `<｜Assistant｜><think>` for the trailing generation
  prompt instead of `</think>`
* `reasoning_effort="max" | "high"` — emits the
  REASONING_EFFORT_MAX preamble at index 0

Pinned by `TestVllmChatTemplateParity` in
`work/tpu-inference/tests/models/jax/test_deepseek_v4.py` —
parametrized over 10 representative cases (chat / thinking /
multi-turn / tools / tool_calls / reasoning_effort). Run as:

```bash
PYTHONPATH=work/vllm:work/tpu-inference work/vllm_env/bin/python3 \
    -m pytest work/tpu-inference/tests/models/jax/test_deepseek_v4.py::TestVllmChatTemplateParity \
    -x -q
```

`latest_reminder` is a non-OpenAI-API role (DeepSeek's internal
quick-instruction guidance) and isn't reachable via the chat
completions endpoint — there's no client surface for it. If/when
some downstream consumer needs it, route via a synthetic
`{"role": "latest_reminder", ...}` message — `encode_messages`
already handles that role.

#### S5. MTP speculative decoding hook is not wired

`work/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py:71-89`
only handles `ngram` and `eagle3` methods. There is no
`deepseek_v4_mtp` proposer. But:

* `deepseek_v4_mtp_forward` at
  `models/jax/deepseek_v4.py:370` exists and is unit-tested
* MTP weights are loaded (`deepseek_v4.py:561` builds
  `mtp_blocks`)
* DECISIONS D4 explicitly notes "vLLM's speculative-decoding
  hook integration is downstream work outside the math-correctness
  goal."

Implementation:
* Add `DeepseekV4MTPProposer` in
  `tpu_inference/spec_decode/jax/` calling
  `deepseek_v4_mtp_forward` to draft `n_mtp_layers=1` extra
  tokens per step.
* Wire into `speculative_decoding_manager.execute_draft_model`.
* Engine plumbing:
  `--speculative-config '{"method":"deepseek_v4_mtp","num_speculative_tokens":1}'`.

1.5–2× decode throughput once S1 lands.

#### S6. Sampling parameters — sampling + stop + logprobs + top_k + presence_penalty + n>1 probes scaffolded; runtime verification per-probe (see entries below)

The path is standard — V4 inherits tpu-inference's
`sampling.py` + `rejection_sampler.py` via `compute_logits` at
`deepseek_v4.py:1528`. The default smoke check fires
`temperature=0, seed=0` only.

**Sampling probe (DONE — verified end-to-end 2026-04-30)**:
`scripts/full_slice_v4_smoke_check.sh` accepts `SAMPLING_REQUIRED=1`,
which fires `/v1/completions` with
`temperature=0.7, top_p=0.9, frequency_penalty=0.1` and asserts the
response has non-empty text plus a valid `finish_reason` (`stop` or
`length`); exit 7 on empty / invalid. Default off so the cheap
smoke stays cheap. Mock-server self-test
`scripts/test_smoke_check_harness.sh` covers all three end states
(`sampling_required_match` / `sampling_required_empty` /
`sampling_required_bad_finish`). The mock differentiates sampling
vs deterministic by inspecting `body.temperature > 0` so a single
test scenario can simulate a working deterministic path **and** a
broken sampling path simultaneously. End-to-end-verified on real
`vllm serve` 2026-04-30 09:57Z: response text=" the\n###!/\n\n\npackage",
`finish_reason="length"`, exit 0.

Determinism under sampling is **not** asserted: vLLM/TPU's runner
doesn't honour per-request seed on non-greedy paths and in fact
**rejects** the request with HTTP 400 ("JAX does not support
per-request seed.") if `seed` is sent alongside `temperature>0`.
The probe omits `seed` for that reason; the deterministic
completions probe still sends `seed=0` because at temperature=0
JAX accepts it (as a no-op). That asymmetry is the same reason
byte-equality across runs would be a false guarantee.

**Stop-sequence probe (DONE — verified end-to-end 2026-04-30)**:
`STOP_REQUIRED=1` fires `/v1/completions` with `stop=["Paris"]` and
asserts (a) the response text does NOT contain `Paris` and
(b) `finish_reason="stop"`; exit 8 on either. The deterministic
baseline (`fire_completion`, no stop) emits ` Paris` as its first
token at temp=0/seed=0, so a working stop-sequence handler MUST
intercept that token, truncate, and report `stop`. Mock harness
covers `stop_required_match` (mock truncates correctly →
exit 0) and `stop_required_leak` (mock with `--stop-honor 0`
ignores the field → exit 8). End-to-end-verified on real
`vllm serve` 2026-04-30 10:16Z: `contains_needle=0 text_len=0
finish_reason=stop`, exit 0. The empty-text outcome is correct:
the model's first generated token IS the stop sequence, so a
correctly-truncated response is empty.

**Logprobs probe (DONE — verified end-to-end 2026-04-30)**:
`LOGPROBS_REQUIRED=1` fires `/v1/completions` with `logprobs=5`
and asserts every emitted position's `top_logprobs` entry has at
least 5 alternatives (exit 9 on missing object / dropped
alternatives). Mock harness covers `logprobs_required_match`
(mock emits 5 alts → exit 0), `logprobs_required_missing`
(mock with `--logprobs-alts 0` omits the object → exit 9), and
`logprobs_required_dropped` (mock with `--logprobs-alts 3` emits
fewer than requested → exit 9). End-to-end-verified on real
`vllm serve` 2026-04-30 10:16Z: `min_alternatives=5 (requested 5)`,
exit 0. Production clients (confidence scoring, reranking,
structured-output likelihood) depend on this path; vLLM's TPU
runner has historically had quirks here.

**Top-k / presence-penalty / n>1 probes (DONE — verified
end-to-end 2026-04-30)**:
`TOPK_REQUIRED=1` (temperature=0.7+top_k=10, exit 10),
`PRESENCE_REQUIRED=1` (temperature=0.7+presence_penalty=0.5,
exit 11), and `N_REQUIRED=1` (n=2; assert >=2 non-empty choices,
exit 12). Top-k gates the candidate set by rank rather than by
top_p's cumulative-prob mass; presence_penalty is a per-token
deduction the first time a token appears (vs frequency_penalty's
linear-in-count); n>1 expands one request into n parallel
sequences sharing the prompt — under `--max-num-seqs=1` this
likely sequentializes but the choices array length must still
equal n. All three follow the same shape as
SAMPLING/STOP/LOGPROBS: one `fire_*` function + bash branch +
mock branch (`--n-cap` for n; `--sampling-text ""` already
covers topk/presence empty-text paths, since both hit the
mock's `temperature>0` branch). Mock harness extended from
18 → 24 scenarios (3 new pass + 3 new fail), all green locally.
End-to-end-verified on real `vllm serve` 2026-04-30 10:36Z:
top-k `text len=1 finish_reason=length`, presence
`text len=3 finish_reason=length`, n=2
`choices=2 non_empty=2 (requested n=2)`. Same launch verified
SAMPLING/STOP/LOGPROBS still pass (3-min weight load + ~6 min
total wall, includes the documented chat-OOM-retry stretch).

**Still TODO under S6** — none of the standard sampling parameters
remain unprobed. Future S6 work would be combinatorial coverage
(e.g. logprobs+stop, topk+presence, n>1+sampling, n+logprobs)
which is unlikely to surface anything that the per-knob probes
miss; defer until a real production client report shows otherwise.

#### S7. Streaming (SSE) — equivalence probe DONE; latency budget probe still TODO

vLLM's framework supports `stream: true` natively;
tpu-inference shouldn't need V4-specific changes.

**Stream-equals-non-stream probe (DONE — verified 2026-04-30)**:
`scripts/full_slice_v4_smoke_check.sh` accepts `STREAMING_REQUIRED=1`,
which re-fires the deterministic completion with `stream=true`,
reassembles the SSE chunks (handles `data: {...}\n\n` lines and the
terminating `data: [DONE]`), and asserts byte-equality vs the
non-streaming output captured earlier in the same run (exit 6 on
mismatch / no chunks). Default off. Mock + harness cover both
match and mismatch paths
(`streaming_required_match` / `streaming_required_mismatch` —
the mismatch scenario uses `--stream-text " Berlin."` so the
non-streaming Paris assertion still passes before the streaming
probe fires). End-to-end-verified on real `vllm serve`: streaming
output reassembled to ` Paris`, byte-equal to non-streaming.

**Still TODO**: a latency-budget probe — assert TTFT < N seconds
and ITL < M ms. Needs threshold tuning against a warm-cache run
to pick numbers that don't false-positive on cold compile. Probably
gate behind `STREAMING_LATENCY_REQUIRED=1` so the equivalence and
latency probes can land independently. OpenRouter-style clients
default to streaming, so latency-on-streaming is the production
metric users actually feel.

### Tier A — production-deployment infra (model is correct but infra isn't)

#### A1. `MAX_LEN=256, MAX_SEQS=1` is hard-coded in the smoke launcher

Activation HBM scales roughly linearly with both. The whole
CSA/HCA + indexer machinery — V4-Flash's value proposition — is
currently unexercised at scale. First experiment: `MAX_LEN=4096
MAX_SEQS=4` and capture the HLO temp profile to see how much
headroom remains. Then push toward 1M context. **Depends on S1**:
without real decode, lifting `MAX_LEN` is meaningless because
attention compute scales O(L²) per step.

#### A2. Persistent compile cache is host-local and ephemeral

`~/.cache/vllm/xla_cache` is on each host's local disk (not
shared, not GCS, cross-host rsync verified unsound — see
optimization-knobs row above). Most cloud VMs clear `/tmp` on
reboot; some clear `~/.cache`. A host swap = 5–10 min cold
penalty.

* Move the cache to a path on a verified-durable mount.
* Add a one-shot bootstrap step that runs
  `scripts/full_slice_v4_warm_cache.sh` per host on first boot
  of a new worker.
* AOT precompile + binary persist
  (`jit().lower().compile()` + serialize) is the next-level
  fix; per-host because of the cache fingerprint finding. Could
  drop cold compile from 97s to ~5s/host.

#### A3. No engine crash recovery

If `VLLM::EngineCore` dies, the api-server is a husk and
`./run.sh stop` is required. CLAUDE.md pitfall #2 documents the
libtpu lockfile orphaning.

* Supervisor (systemd unit, k8s liveness probe, or
  `scripts/supervise.sh`) that runs `full_slice_v4_reset.sh` +
  relaunches on detected EngineCore death.
* Tie supervision to vLLM's `/health`, not just process liveness.
* Drain on SIGTERM: api-server should refuse new requests and
  let in-flight ones finish before exit (today `./run.sh stop`
  SIGKILLs mid-request).

#### A4. No metrics / observability

vLLM's `--enable-metrics` is not currently set. Without it: no
TTFT, no ITL, no throughput, no queue depth, no KV utilization,
no error rate. Pass the flag; scrape into Prometheus + Grafana
or push to whatever observability backend is real. Per-host TPU
utilization needs separate observability (libtpu-side metrics or
`gcloud monitoring`).

#### A5. No TLS / authentication / rate limiting

Currently `0.0.0.0:18081` plain HTTP, no auth. For OpenRouter-
grade exposure: TLS termination at a reverse proxy, per-API-key
auth (vLLM's single-key flag isn't sufficient — multi-tenant
needs LiteLLM or a custom frontend), per-key rate limiting. Run
vLLM as a non-root systemd unit, not from `$HOME`.

#### A6. Single slice — no horizontal scale

One v6e-32 slice ceilings on per-key concurrency at whatever
S2's eventual ragged-batch implementation tops out at. Multi-
slice requires: a model-aware load balancer that sticks
per-conversation sessions to one slice (so KV cache hits hold),
per-slice health monitoring, and shared model-weight storage
(the GCS bucket already is that).

### Tier B — known performance work

#### B1. Sparse-attention Pallas kernel

`sparse_attn` at
`layers/jax/attention/deepseek_v4_attention.py:131` is
fully-materialized `jnp.take_along_axis` + dense einsum +
softmax. DECISIONS D2 documents this as correctness-over-perf.
Real Pallas kernel: gather kv only for top-k indices in TPU
SRAM, fuse the sink term into the softmax denominator, avoid
materializing the `[B, M, K, D]` gather buffer. 2–5× decode
latency improvement once S1 unlocks the regime where this
matters. Multi-week effort.

#### B2. True sparse MoE dispatch

`moe_forward` at
`layers/jax/moe/deepseek_v4_moe.py:156` is "vectorized dense":
every token sees every expert via masked einsum. FLOP cost is
`top_k * E` higher than necessary — for `top_k=8, E=256` that's
32× over true sparse. Wire the existing
`tpu_inference/kernels/megablox/gmm.py` (grouped matmul) into
V4's MoE. Hash-routing layers (INVARIANTS I18) need a different
treatment — `tid2eid` lookup is per-token, but the dispatch
pattern is the same.

#### B3. SPMD `Involuntary full rematerialization` audit — DONE for the `compressor.ape` family (126 → 0)

The 126 `Involuntary full rematerialization` warnings observed
in the green-gate smoke log were all the `compressor.ape` /
`indexer.compressor.ape` family — tiny `f32[128,16]` /
`f32[4,32]` / `f32[4,8]` constant tables that the loader's
sharding heuristic split along `attn_dp` because their largest
dim happened to divide cleanly into 32. Consumers wanted them
replicated (or differently sharded), and XLA reported the
unavoidable reshard at every consume site. The
`_replicate(params.ape)` `with_sharding_constraint` inside
`compressor_prefill` / `compressor_decode_step` was a partial
mitigation but didn't drop the warnings — XLA still has to
reshard from the loaded sharding to the constraint.

Fix: `pick_partition_spec` (`models/jax/deepseek_v4_loader.py`)
now returns `P()` for any tensor below `_MIN_SHARD_ELEMENTS`
(8K elements, ~32 KiB f32). Anything bigger keeps the sharded
path; norm weights (4096 f32 elements at hidden_size=4096) and
similar fall under the threshold and replicate too — the per-
chip HBM cost is negligible (~16 KiB/chip extra per norm × ~80
norms × 32 chips = ~40 MiB total), and the extra reshard noise
is gone. Verified 2026-04-30: real-smoke green with 0 remat
warnings (vs 126 prior) and the same sub-100s cold compile.

To re-audit after future kernel work:

```
grep "Involuntary full rematerialization" \
    logs/full-slice-v4-smoke-*.log
```

#### B4. AOT compile + binary persist

`jit().lower().compile()` → serialize → load. Per-host because
of the cache fingerprint finding (see A2). Could drop cold
compile from 97s to ~5s/host. Defer until B1+B2 land — there's
no point persisting a sub-optimal binary.

### Tier C — quality gates (don't claim "we serve V4" without these)

#### C1. Benchmark vs DeepSeek's reference scores

MMLU, HellaSwag, GSM8K, HumanEval, MATH — match V4-Flash's
published scores within tolerance. Use `lm-eval-harness`. Each
run is hours at production batch sizes; needs S2 (multi-seq
concurrent) to be tractable in wall-clock. **This is the gate
that lets you claim "we serve V4-Flash" honestly.** Without it,
a silent kernel divergence (e.g. unnoticed bf16 vs fp32 cast in
the MoE down-projection) could cost a meaningful fraction of
model quality and the "Paris" smoke wouldn't catch it.

#### C2. Long-context functional test

Even before perf-tuning long context: a single request with
`max-model-len=131072` that asserts coherent output. V4-Flash's
compressor + indexer machinery is currently completely
unexercised in production paths. Suggested: needle-in-a-haystack
at 4k, 16k, 64k, 256k, 1M. Each context size needs the bumped
`MAX_LEN` (A1, which depends on S1).

#### C3. Math regression suite under load

Random sampling, long contexts, tool calls, multi-turn —
verify outputs stay within reference tolerance under
temperature>0 and concurrent load. Catches regressions where
greedy decoding looks fine but sampling has a subtle skew.

#### C4. Tokenizer edge cases

Non-ASCII (Chinese, Arabic, emoji), leading whitespace,
multilingual code blocks, very-long single tokens. V4-Flash's
`tokenizer_config.json` has `add_bos_token=false`; the upstream
`encode_messages` emits BOS itself. `TestVllmChatTemplateParity`
covers role-transition byte-equality vs `encode_messages` on
representative cases — extend it with non-ASCII / leading-
whitespace / very-long-single-token fixtures rather than writing
a new test.

#### C5. Refusal/safety behavior preservation

V4 was tuned for specific refusal patterns. After kernel
rewrites this often regresses (low-bit MoE quant especially can
shift the safety tuning). Run a small refusal-eval set (a few
dozen prompts spanning the model card's tested categories)
before each significant change to S1/B2.

### Tier D — code-hygiene / janitorial

#### D1. Test bloat

`tests/models/jax/test_deepseek_v4.py` is **2904 LOC, 29 test
classes** (~4× the next biggest model's test file). The
strictly-weaker FP8/FP4 smoke classes were folded into their
byte-equal reference siblings (their unique parametric layers
were merged into `TestFp{8,4}DequantIndependentReference`'s
parametrize list). Further consolidation candidates (compile-
shape variants, decode-step parity classes) likely exist but
need a per-class audit; the cheap wins are taken.

Re-measure before claiming further progress: `wc -l` and
`grep -c "^class Test"`.

## Iteration discipline (READ — applies to humans + agents alike)

**Do NOT use `./run.sh serve` as your inner test loop.** Each
attempt is 25–45 min (4 min load + 10–30 min cold compile + curl
wait). That budget is fixed by XLA, not by anything we can
shorten in a single iteration. Prior sessions burned real time
treating it as if it should be fast. Use the fastest validation
that catches the bug class you're working on:

1. **Standalone math scripts** under `/tmp/` (~10–30s) — pattern:
   `/tmp/test_moe_vectorize.py` validated the vectorized MoE
   math vs the per-expert reference on 5 seeds in ~10s.
2. **Tiny-fixture pytest classes** in
   `tests/models/jax/test_deepseek_v4.py` (~30s–2min on CPU).
3. **`eval_shape` / `lower().compile()` on the real config**
   (~1–3min). Catches sharding bugs + HLO-emit failures (like
   the original HBM OOM) without paying the runtime compile
   cost. Pattern:
   `XLA_FLAGS=--xla_force_host_platform_device_count=32
   JAX_PLATFORMS=cpu` to compile against a virtual mesh.
4. **Real `./run.sh serve`** only when 1–3 are green. Budget at
   most 1–2 of these per session.

### Real-smoke phase budgets (don't bail too early!)

When you have to run the real smoke (path #4), each phase has a
*known* duration. Silence during a phase is normal as long as
it's the right kind of silence:

| Phase | Expected duration | What you should see | Bail signal |
|---|---|---|---|
| **vLLM startup + Ray cluster init** | ~30s | `Init mesh \| mesh=Mesh(...)`, `Init kv-cache`, route registration | No log activity for >2 min, OR `Worker exit type: SYSTEM_ERROR`. |
| **Weight load** | ~4 min | `[deepseek_v4] placed N tensors (R/s, ...)` heartbeat every ~7s, then `load_weights_from_dir done` | No heartbeat for >2 min, OR `placed N` count stops growing. |
| **`capture_model` precompile** | ~30s | A handful of small `running hlo passes for N instructions, module: jit_*` lines, each tiny | Any `RESOURCE_EXHAUSTED` / `CompileTimeHbmOom`. |
| **`Application startup complete`** | fires immediately after capture_model | Single line | If absent >2 min after capture_model finishes. |
| **`jit_run_model` cold compile** | **10–30 min** on cold cache; **~97s** on warm cache. | One `running hlo passes for ~100k instructions, module: jit_run_model`, then long silence punctuated by `HLO PostOptimizationPipeline` lines and SPMD warnings. The silence is normal — XLA's late codegen passes don't emit progress. | Three or more separate `slow_operation_alarm.cc` warnings (each fires after a single pass exceeds 5 min). One alarm = one slow pass; that alone is *not* enough to bail. Also: any `RESOURCE_EXHAUSTED` / `Worker exit`. |
| **First curl returning** | sub-second after compile finishes | `INFO 127.0.0.1:... "POST /v1/completions" 200 OK` and the `[smoke-check] response 1: ...` line | Curl 900s timeout fires, OR the engine crashes mid-execute. |

**Rule of thumb during real smoke:** silence in the
`jit_run_model` phase ≤ ~25 min is *expected*, not stuck.
**Don't bail before 25 min unless the iter timeout is closing
in.** The 90-min ITER_TIMEOUT_SEC has plenty of slack for one
full smoke + one bail.

**Concurrent work while compile runs:** the compile is going to
take however long it takes. Spend that time productively —
sketch the next-lane fix in a `/tmp/` standalone test, audit
warning families in the smoke log, consolidate test bloat (test
edits don't conflict with the running smoke). Don't just sit in
a Monitor.

**Quick-test rule (still applies for code edits, NOT for smoke):**
if a CPU pytest / `lower().compile()` probe takes >5 min without
a useful signal, kill it and rethink — that *is* stuck.

### Iter-timeout management

`ITER_TIMEOUT_SEC=5400` (90 min). If you're approaching the
deadline without a result:

1. **At T-15 min:** stop launching new long-running steps. Commit
   whatever code change you've made so far (with a "WIP:" prefix
   describing what was tried + what's still unverified) so iter
   N+1 can pick up from the same on-disk state.
2. **At T-5 min:** reset the cluster + push the WIP commit. Don't
   risk the iter being killed mid-`./run.sh serve`.

Better to have a checkpointed WIP commit than to lose the diff
when the timeout SIGTERMs the iter.

## What's been verified

* **Streaming sharded loader** (no zero-tree OOM). ✓
* **Slice-aware load**: each host reads only its row range. ✓
  Parity-verified on tiny fixture.
* **Multi-threaded placement** (`V4_LOADER_PLACE_WORKERS=8`). ✓
  Parity-verified on tiny fixture.
* **safetensors handle cache** (`_safe_open_cache`): eliminates
  per-tensor mmap+header reopen — observed ~6× load speedup
  (23 t/s → 140 t/s on real V4-Flash, ~4 min total load down
  from ~25 min). ✓
* **Vectorized MoE forward** (math byte-equal to per-expert
  reference on 5 seeds, maxabs=0; HLO instructions 4.6× smaller).
  ✓ correctness, ✗ optimal flops (B2).
* **Inline MoE consolidation at load** (the 256 per-expert
  weights of each `(layer, wname)` group are stacked into a
  single E-sharded `[E, inter, dim]` jax.Array as soon as the
  256th is placed; per-leaf references are then nulled). Drops
  the per-call all-to-all storm —
  `jit_run_model` HLO instructions 47k optimized vs 103k
  previously. ✓
* **MoE stacked-weight sharding constraint**
  (`_shard_e_first` / `_shard_e_last` / `_shard_e_mid`): forces
  W1/W2/W3 to be E-sharded across `attn_dp`, eliminating the
  original 4 GiB all-gather per stack. Mostly superseded by
  inline consolidation (constraints stay as defense in depth on
  the per-expert fallback path). ✓
* **Freqs cap by `max_model_len`**: `_effective_freqs_seq_len()`
  uses `vllm_config.model_config.max_model_len` instead of
  `cfg.max_position_embeddings`, shrinking the YaRN freqs table
  from 1 GB / chip to KB. ✓
* **Persistent JAX compile cache**: wired; populated under
  `~/.cache/vllm/xla_cache` on every host. Subsequent launches
  on the same worker host skip the ~96 s compile. ✓
  (Cross-host sharing is unsound — verified.)
* **Decode-step kernels** (`attention_decode_step`,
  `compressor_decode_step`, `indexer_decode_step`): all match
  the torch reference to 1e-4 across the parametrized
  `TestDecodeAttentionParity` /
  `TestDecodeRollingParity` matrix. ✓ math, ✗ runtime
  integration (S1 — they aren't called by `__call__`).
* **Transformer-body decode primitives**
  (`block_init_state_and_forward`, `block_decode_step`,
  `transformer_body_init_state_from_prefill`,
  `transformer_body_decode_step` in `models/jax/deepseek_v4.py`):
  prefill→init_state→decode_step round-trip matches a
  fresh-prefill on the T+1 sequence within ~5e-3 across
  T ∈ {4,8,16,24} on the 6-layer tiny config (3 SWA + 2 CSA +
  2 HCA pattern). Pinned by `TestTransformerBodyDecodeRoundTrip`.
  ✓ math, ✗ runtime integration (S1 — `__call__` is the
  remaining surgery; see S1's "Iter-4 starts at").
* **Packed decode-state buffers** (`_layer_decode_state_layout`,
  `_pack_layer_state`, `_unpack_layer_state`,
  `transformer_body_init_state_to_buffer`,
  `transformer_body_decode_step_from_buffer` in
  `models/jax/deepseek_v4.py`): one layer's 6-field
  `AttentionDecodeState` flattens losslessly into a single 1D
  fp32 array via `_pack_layer_state`; the inverse via
  `_unpack_layer_state` is bit-exact (verified including
  `compressor_score_state`'s -inf entries via `jnp.array_equal`).
  Stitched into the transformer-body decode round-trip: prefill
  via `init_state_to_buffer` then 3 sequential decode steps via
  `decode_step_from_buffer` reproduce a fresh T+N prefill within
  ≤ 5e-3, the same budget as the non-buffer primitives. Pinned
  by `TestPackedDecodeStateBuffer` (6 cases: round-trip +
  2 single-step + 1 multi-step + 1 under-jit + 1 kv_caches
  round-trip). The pack/unpack ops are pure JAX
  (reshape+cast+concat+dynamic_slice) and inline cleanly into
  surrounding JIT — no extra error sources. ✓ math + schema, ✗
  runtime integration (S1 iter-5 wires these into
  `__call__` via `kv_caches`).
* **Decode-state orchestrator + JIT-correctness validation
  (S1 iter-4)**: `deepseek_v4_run_with_decode_state(kv_caches,
  ids, params, …, is_decode_step, start_pos)` in
  `models/jax/deepseek_v4.py`. Branches on `is_decode_step` and
  threads packed `AttentionDecodeState` through `kv_caches`:
  prefill replaces the buffer wholesale (closed-form
  init from x); decode reads `kv_caches[i]`, advances by one
  position via `transformer_body_decode_step_from_buffer`, and
  writes the new buffer back. Returns `(updated_kv_caches, h)`
  matching the iter-5 `run_model` contract.
  Pinned by `test_run_with_decode_state_kv_caches_round_trip`
  (one prefill on T tokens + N=3 sequential decode steps via
  `kv_caches` threading; each step matches a fresh full prefill
  within ≤5e-3 on tiny config). Also pinned under jit by
  `test_buffer_chain_compiles_under_jit` which surfaced and
  fixed a real bug: `start_pos` must be marked
  `static_argnums` on the jit boundary because
  `attention_decode_step` indexes circular buffers with Python
  ints. Without that, the trace raises `IndexError: Slice
  entries must be static integers`. iter-5's runtime wiring
  inherits the right pattern. ✓ math + JIT-compatibility, ✗
  runtime allocation + `__call__` flip (iter-5).
* **Cfg-derived per-layer packed sizes (S1 iter-5a)**:
  `v4_layer_packed_sizes_from_cfg(cfg, state_max_seq_len,
  batch_size=1) → List[int]` in `models/jax/deepseek_v4.py`
  computes per-layer `AttentionDecodeState` packed-size from
  `cfg` alone (no loaded `params` needed). Mirrors
  `_layer_decode_state_layout` shape decisions exactly. Pinned
  by `test_v4_layer_packed_sizes_from_cfg_matches_layout` —
  asserts cfg-derived sizes ≡ params-derived sizes (via
  `transformer_body_layout` + `_layer_packed_size`) for
  `state_max_seq_len ∈ {64, 32, 16}` on a tiny 6-layer config
  (3 SWA + 2 CSA + 2 HCA pattern). Unblocks iter-5b's
  kv_cache_manager allocator: it can size the V4 fp32 packed
  buffers from `cfg + max_model_len` BEFORE `model.params_v`
  is concretely loaded. Same iter also empirically refuted the
  iter-4 plan's "size-1 axes are replicated" hypothesis: a
  shape `[1, packed_size, 1]` fp32 array sharded `P('attn_dp',
  None, 'attn_head')` fails at allocation with
  `IndivisibleError` on a 32-virtual-CPU mesh — JAX has no
  size-1-axis fallback. iter-5b uses replicated `P()` instead.
  ✓ helper + parity test, ✗ runtime allocation (iter-5b).
* **Orchestrator path-A-h decoupling (S1 iter-5c)**:
  `deepseek_v4_run_with_decode_state`'s prefill branch in
  `models/jax/deepseek_v4.py` now computes `h` via
  `transformer_body_forward` (path A — the green-gate baseline)
  and packed_buffers via `transformer_body_init_state_to_buffer`
  SEPARATELY; the orchestrator's own `h` from
  init_state_to_buffer is discarded. Solves iter-5b's NaN
  hypothesis (XLA reorders bf16 reductions when the closed-form
  state-init's intermediate fp32 buffers feed the same scheduler
  pass as the forward path). The forward `h` is now byte-equal
  to `transformer_body_forward` BY CONSTRUCTION. Pinned by
  `TestPackedDecodeStateBuffer` + `TestTransformerBodyDecodeRoundTrip`
  + `TestPrefillToDecodeStateParity` (37 cases) on tiny config
  CPU pytest. `/tmp/test_v4_iter5c_compile.py` confirms under
  32-virtual-CPU mesh w/ replicated `P()` kv_caches: prefill
  h diff vs path-A = 0.000002 (≪ 5e-3 budget); 32 identical
  shards; all `h` finite. On real V4-Flash + `vllm serve`,
  the FIRST `/v1/completions` returned 200 OK at 14:40:04
  on 2026-04-30, indicating the prefill NaN regression is
  fixed; the SECOND request triggered TPU `[USER]` FATAL —
  see S1 iter-5c status above for the iter-5d hand-off.
  ✓ orchestrator decoupling math + first-request prefill,
  ✗ second-request stability (iter-5d to investigate before
  re-flipping `__call__`).
* **`__call__` flip env-gated + diagnostic logging (S1
  iter-5d)**: `models/jax/deepseek_v4.py` now reads
  `V4_DECODE_STATE` once at module import into
  `V4_DECODE_STATE_ENABLED`. Default `0` keeps the legacy
  green-gate path (`transformer_body_forward`, kv_caches
  passes through unchanged). `1` routes through the iter-5c
  orchestrator (`deepseek_v4_run_with_decode_state`). When
  enabled, every JIT-trace entry to __call__ logs `(call_idx,
  T, start_pos, is_decode, state_max_seq_len, kv_caches_count)`
  — a Python `logger.info` line that fires per fresh JIT
  cache key (so a request-2 prefill that hits the cached
  request-1 prefill kernel produces NO log, which is itself
  signal). Wired through `scripts/full_slice_v4_smoke.sh` via
  `V4_DECODE_STATE` in `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` and
  echoed at smoke startup. Hypothesis 2 ("stale
  decode_start_pos meta_field across requests") was ruled out
  by code reading: `tpu_runner.py:1634` constructs a fresh
  `AttentionMetadata` every scheduler step, so
  `decode_start_pos` is reset to 0 by virtue of fresh
  allocation. Pinned by the existing 37-case iter-5 suite —
  default `V4_DECODE_STATE=0` is bytewise identical to
  iter-5c's behaviour. ✓ env-gating + diagnostic + tests
  green; ✓ real-V4 smoke with `V4_DECODE_STATE=1` ran in
  iter-5e and the diagnostic surfaced the T-padding
  decode-detection bug (see next entry).
* **Decode-detection bug DIAGNOSED + FIXED (S1 iter-5e)**:
  the iter-5d diagnostic logs at real-V4 boot showed
  `__call__ #1: T=64 start_pos=5 is_decode=False` for what
  should have been a decode call — the runner pads decode
  shapes to bucket sizes (T=64 typical), so `(T==1) and
  (start_pos>0)` never triggers. iter-5e replaces the check
  with `is_decode = start_pos > 0` (the runner-provided
  `decode_start_pos` already encodes the q0==1/s0>1 contract
  from `tpu_runner._maybe_set_v4_decode_start_pos`), slices
  `ids_2d[:, 0:1]` for the decode branch (the runner zero-
  pads the rest of the bucket), and pads the decode-branch
  `h` from `[B, 1, hc, D]` to `[B, T, hc, D]` with zeros so
  the downstream `hidden_TD` reshape keeps the runner-
  expected shape. Verified 2026-04-30 16:28Z on real V4-Flash:
  `__call__ #1: T=64 start_pos=5 is_decode=True` fires for
  the first time, fresh `jit_run_model` decode compile
  (30,764 HLO instructions vs prefill's ~80k), R1 returns
  `200 OK` — the decode kernel chain executes end-to-end on
  real bf16 V4 weights. ✓ decode-branch routing + decode
  kernel runs on real V4 (first time ever); ✗ second-request
  stability still blocks (iter-5f hyp-1 ruled out — see next
  entry; hyp-3 / instrumentation deferred to iter-5g).
* **iter-5f hyp-1 RULED OUT (2026-04-30)**: dropping
  `donate_argnums=2` for V4 only (model_loader.py V4 branch)
  was attempted as the cheapest fix for iter-5d's "donated
  buffer reuse" hypothesis. Tested on real V4-Flash with
  `V4_DECODE_STATE=1` and a fresh ray-restart
  (logs/full-slice-v4-smoke-20260430T172651Z.log): R1's
  fresh-compile prefill orchestrator FATAL'd with the same
  TPU `[USER]` signature within ~19s of the trace —
  `tpu6:pe2:2` at 17:33:14Z. Whereas iter-5e's WITH-donation
  R1 had succeeded and only R2 FATAL'd, iter-5f's WITHOUT-
  donation R1 failed outright. Conclusion: dropping donation
  produces a different compiled artifact whose first
  execution has its own bug. Hypothesis 1 dead; revert
  shipped at commit 571a82f3. iter-5g picks up from
  jax.debug.print instrumentation OR the new sharding-
  constraint angle (force `with_sharding_constraint(b, P())`
  on each packed_buffer inside `_pack_layer_state` so the
  reshard-to-replicated happens at a known code point rather
  than at the JIT boundary). See "Iter 5f status" above for
  full notes.
* **MTP forward**: `deepseek_v4_mtp_forward` math validated on
  tiny fixture. ✓ math, ✗ runtime integration (S5).
* **Chat encoding (all scopes)**: vLLM's upstream
  `DeepseekV4Tokenizer` calls `encode_messages` from
  `vllm/tokenizers/deepseek_v4_encoding.py` directly — byte-equal
  to the V4-Flash reference encoder
  (`<hf-snapshot>/encoding/encoding_dsv4.py`) across chat /
  thinking / tools / tool_calls / reasoning_effort. Pinned by
  `TestVllmChatTemplateParity`. ✓ across all four S4 scopes;
  `--chat-template` is unused (the tokenizer ignores it). See S4.
* **Reasoning + tool parsers wired** (`--reasoning-parser deepseek_v4`,
  `--enable-auto-tool-choice --tool-call-parser deepseek_v4` in
  `scripts/full_slice_v4_smoke.sh`). Registry lookup verified via
  the snippet in S3. vLLM validates parser names at startup, so
  smoke-green = parsers loaded. ✓ wiring; ✓ runtime probe scaffolded
  (`REASONING_REQUIRED=1` smoke_check assertion + 10-scenario harness
  self-test); ✗ runtime probe **fails on real V4-Flash** — thinking-mode
  emits N newlines (greedy) or random tokens (sampling), independently
  of prompt or `reasoning_effort`. The probe is doing its job; the
  underlying behavior is a Tier-S correctness bomb (see S3 in the
  backlog for repro + hypotheses).
* **Streaming probe verified end-to-end** (`STREAMING_REQUIRED=1`
  smoke_check + 13-scenario harness self-test + real `vllm serve` run
  on 2026-04-30: reassembled SSE = non-streaming " Paris" byte-for-byte).
  ✓ See S7.
* **Sampling probe verified end-to-end** (`SAMPLING_REQUIRED=1`
  smoke_check + 13-scenario harness self-test + real `vllm serve` run
  on 2026-04-30: temperature=0.7+top_p=0.9+frequency_penalty=0.1 returns
  non-empty text and `finish_reason=length`). ✓ See S6.

  Caught a real bug along the way: vLLM/TPU's runner rejects per-request
  `seed` on non-greedy paths with HTTP 400 ("JAX does not support
  per-request seed."). The original probe sent `seed=0` along with
  `temperature=0.7` and got 400 → smoke_check exit 7. Fix was
  one-line: drop `seed` from `fire_completion_sampling`. The
  deterministic completions probe still sends `seed=0` (greedy
  ignores it) and the harness mock still passes (it doesn't
  inspect `seed`). The pitfall is the deeper "TPU runner doesn't
  support per-request seed" CLAUDE.md note showing up *as a 400*,
  not just as silently-ignored.
* **Stop-sequence + logprobs probes verified end-to-end** (`STOP_REQUIRED=1`
  + `LOGPROBS_REQUIRED=1` smoke_check + 18-scenario harness self-test +
  real `vllm serve` run on 2026-04-30 10:16Z). ✓ See S6.

  Stop probe sends `stop=["Paris"]` against the deterministic prompt
  whose first token is ` Paris` — a working handler must truncate
  before that token and report `finish_reason="stop"`. Real-V4
  outcome: `contains_needle=0 text_len=0 finish_reason=stop`. The
  empty text is correct — the *very* first emitted token is the
  stop sequence, so a correctly-truncated response IS empty.

  Logprobs probe sends `logprobs=5` and asserts every position's
  `top_logprobs` has at least 5 alternatives. Real-V4 outcome:
  `min_alternatives=5 (requested 5)`, exit 0. Production clients
  (confidence scoring, reranking, structured-output likelihood)
  depend on this path.
* **Top-k + presence-penalty + n>1 probes verified end-to-end**
  (`TOPK_REQUIRED=1` + `PRESENCE_REQUIRED=1` + `N_REQUIRED=1`
  smoke_check + 24-scenario harness self-test + real `vllm serve`
  run on 2026-04-30 10:36Z). ✓ See S6.

  Top-k probe sends `temperature=0.7, top_k=10` and asserts
  non-empty text + valid `finish_reason`. Distinct from the existing
  sampling probe (top_p + frequency_penalty) because top-k bounds
  the candidate set by rank rather than by cumulative probability
  mass — different code path on the TPU sampler. Real-V4 outcome:
  `text len=1 finish_reason=length`, exit 0.

  Presence-penalty probe sends `temperature=0.7, presence_penalty=0.5`
  and asserts the same well-formedness. Distinct from
  `frequency_penalty` (the sampling probe's penalty) because
  presence_penalty is a fixed per-token deduction the first time a
  token appears, not a per-occurrence linear penalty — separate
  reduction step in the TPU sampler. Real-V4 outcome:
  `text len=3 finish_reason=length`, exit 0.

  N=2 probe sends `temperature=0.7, n=2` and asserts the response
  has at least 2 non-empty choices. vLLM expands `n>1` into n
  parallel sequences sharing the prompt — under `--max-num-seqs=1`
  these sequentialize, but the choices array must still have length
  n. Real-V4 outcome: `choices=2 non_empty=2 (requested n=2)`,
  exit 0. Confirms the multi-completion expansion path is live on
  the TPU runner even with the constrained seq-dispatch budget;
  S2 (real ragged-batch multi-seq) will improve throughput but
  isn't required for n>1 correctness.

  Same smoke run also re-verified SAMPLING/STOP/LOGPROBS still
  pass (no regression from the broader-matrix wiring). Total wall
  ~6 min including the documented chat-first-call OOM-retry
  stretch (CLAUDE.md pitfall #9); cold compile of `jit_run_model`
  on warm cache was sub-100s.
* **Tiny-tensor replication at load** (B3 fix): `pick_partition_spec`
  in `models/jax/deepseek_v4_loader.py` returns `P()` for shape
  products below `_MIN_SHARD_ELEMENTS=8K` (~32 KiB f32). Eliminates
  the loader-side `attn_dp`-axis sharding for `compressor.ape` /
  `indexer.compressor.ape` / norm weights / `attn_sink` and similar
  tiny constants whose consumers couldn't accept the heuristic
  layout. Drops the green-gate smoke from 126 `Involuntary full
  rematerialization` warnings to 0 with no perf regression
  (cold compile ~75s end-to-end, prefill+decode buckets each at
  ~47k optimized HLO instructions, deterministic " Paris" ✓).
  Verified 2026-04-30 09:28:55Z. See B3.

## Chat template (chat-completions)

V4-Flash deliberately ships **no Jinja `chat_template`** —
`tokenizer_config.json` omits the field and the upstream HF
README points users at the Python encoder at
`<snapshot>/encoding/encoding_dsv4.py`. vLLM handles this
without a Jinja template at all: `tokenizer_mode='deepseek_v4'`
auto-resolves for `DeepseekV4ForCausalLM` and routes through
`DeepseekV4Tokenizer.apply_chat_template`, which calls
upstream's `encode_messages` (a direct port of the model card's
encoder) and **ignores any `--chat-template` arg**. So the smoke
launcher does not pass `--chat-template` — there's no Jinja
file in the repo. Behavior is pinned by
`TestVllmChatTemplateParity` (10 cases covering chat / thinking /
multi-turn / tools / tool_calls / reasoning_effort).

To re-validate against the V4-Flash reference encoder:

```bash
PYTHONPATH=work/vllm:work/tpu-inference work/vllm_env/bin/python3 \
    -m pytest work/tpu-inference/tests/models/jax/test_deepseek_v4.py::TestVllmChatTemplateParity \
    -x -q
```

`vllm chat` CLI needs `--url http://localhost:18081/v1` since
the smoke launcher binds 18081 (not vllm's default 8000).

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
   `/v1/completions` call triggers compilation of `jit_run_model`.
   Expect 5–15 min on a cold compile cache, ~30–60s on a warm
   cache. Don't use a 60s curl timeout — the smoke check defaults
   to 900s.

   To warm the cache at bootstrap time (one-time +10–15 min cost,
   then every subsequent first-curl is sub-minute), set
   `WARM_CACHE_ON_BOOTSTRAP=1` in `.env` before
   `./run.sh bootstrap`. `scripts/full_slice_v4_warm_cache.sh` is
   the underlying helper.

6. **`--enforce-eager` does not skip XLA compile.** That flag only
   affects vLLM's CUDA-graph-equivalent path. The TPU forward is
   JAX/`tpu-inference` and ALWAYS jit-compiles via XLA.

7. **vLLM's `capture_model` can multiply compile cost.** Without
   `--enforce-eager`, vLLM precompiles many shape buckets up
   front. `--enforce-eager` (already in the smoke launcher) skips
   that pre-compile and lets the first request pay the
   single-shape compile cost lazily.

8. **`JAX_COMPILATION_CACHE_DIR` does nothing under vLLM.**
   `tpu_inference/runner/compilation_manager.py:53` calls
   `jax.config.update("jax_compilation_cache_dir",
   vllm_envs.VLLM_XLA_CACHE_PATH)` during engine init,
   *overriding* whatever the launcher set. The real cache always
   lives at `~/.cache/vllm/xla_cache` (or `VLLM_XLA_CACHE_PATH`).
   Verify cache activity by `ls -la ~/.cache/vllm/xla_cache`
   after a smoke, not by the launcher's echoed path.

9. **The `/v1/chat/completions` first-call OOM-retry is normal.**
   The chat path lands in a 1024-token prefill bucket vs 256 for
   completions; on a tight HBM budget the engine sometimes hits
   `RESOURCE_EXHAUSTED: RuntimeProgramAllocationFailure` and
   recovers via `TpuLoadedExecutable::ExecutePrepareWithOomRetries`
   which defragments and retries. Adds ~30s to first-chat
   latency; subsequent chat calls are fast. Not a bug; just
   noisy. The `smoke_check` chat probe is informational by
   default for this reason.

## Layout

* `work/tpu-inference/` — JAX V4 implementation. Git subtree of
  the upstream `tpu-inference` repo. The DeepSeek V4 model lives
  at
  `work/tpu-inference/tpu_inference/models/jax/deepseek_v4*.py`;
  the MoE math at
  `work/tpu-inference/tpu_inference/layers/jax/moe/deepseek_v4_moe.py`;
  attention at
  `work/tpu-inference/tpu_inference/layers/jax/attention/deepseek_v4_attention.py`.
* `work/vllm/` — vLLM source tree. Don't edit upstream files
  unless you've read `work/vllm/AGENTS.md` (it forbids ad-hoc
  PRs). Reasoning + tool parsers for `deepseek_v4` already exist
  upstream and are enabled in the smoke launcher (`--reasoning-parser
  deepseek_v4`, `--tool-call-parser deepseek_v4`).
* `scripts/` — operational helpers; per-host entry points all
  start with `full_slice_v4_`.
* `logs/` — `.gitignore`d; smoke + iter logs accumulate here.
* `README.md` — fresh-VM bringup (one-shot via `./run.sh`).
* `.env.example` — every env var documented.
* `prompt.md` — the prompt the autonomous loop hands to
  `claude -p` each iter. Read CLAUDE.md (this file) first; the
  prompt only points back here.

Durable docs in `work/tpu-inference/`:
* `INVARIANTS.md` — math invariants. Each broken invariant is a
  shipping bug.
* `DECISIONS.md` — durable architectural decisions (not
  per-session).
* `BLOCKERS.md` — short pointer to the production-readiness
  backlog above.
* `TINY_CONFIG.md`, `TOLERANCE_LOG.md`, `V3_TO_V4_DIFF.md` —
  math reference, don't decay.

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

Update this file as you learn more — but updates are for *durable*
operational knowledge. Per-session decisions go in commit
messages.

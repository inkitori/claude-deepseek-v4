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
  (uses `scripts/_mock_openai_server.py` — 13 scenarios, no TPU
  needed). Also fires four optional probes, each gated by an env
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
| `CHAT_REQUIRED` | `0` | Default makes the smoke_check's `/v1/chat/completions` probe informational (HTTP-success best-effort). Set to `1` to make a missing/empty chat response fail the gate (exit 4). The chat path lands in a 1024-token prefill bucket vs 256 for completions and on a tight HBM budget the engine sometimes needs `TpuLoadedExecutable::ExecutePrepareWithOomRetries` to land — usually succeeds but adds ~30s to first-chat latency. |
| `REASONING_REQUIRED` | `0` | Set to `1` to fire a thinking-mode chat (`chat_template_kwargs={"thinking":true}`) with a reasoning-eliciting prompt and assert `message.reasoning` is non-empty (exit 5 on empty). Pins backlog item S3's runtime. Adds ~30s on first-chat-cold-cache (lands in chat path; same OOM-retry caveat as `CHAT_REQUIRED`). |
| `STREAMING_REQUIRED` | `0` | Set to `1` to additionally fire `/v1/completions` with `stream=true`, reassemble the SSE chunks, and assert byte-equality vs the non-streaming output (exit 6 on mismatch / no chunks). Pins backlog item S7. Cheap — same prefill bucket as the existing completions probe. |
| `SAMPLING_REQUIRED` | `0` | Set to `1` to additionally fire `/v1/completions` with `temperature=0.7, top_p=0.9, frequency_penalty=0.1` and assert the response has non-empty text + a valid `finish_reason` (exit 7 on empty / invalid). Pins backlog item S6 — verifies the sampling code path doesn't crash or produce garbage. Cheap — same prefill bucket as the existing completions probe. Determinism under sampling is **not** asserted (vLLM/TPU runner doesn't honour per-request seed for non-greedy paths; CLAUDE.md pitfall context). |
| `STOP_REQUIRED` | `0` | Set to `1` to additionally fire `/v1/completions` with `stop=["Paris"]` and assert (a) the response text does NOT contain `Paris` and (b) `finish_reason="stop"` (exit 8 on either). Pins backlog item S6 broader-matrix — verifies vLLM's stop-sequence handler on the TPU runner truncates before the matched token and reports the right reason. Cheap — same prefill bucket as the existing completions probe. The deterministic baseline emits ` Paris` as its first token at temp=0/seed=0, so a working handler MUST intercept it. |
| `LOGPROBS_REQUIRED` | `0` | Set to `1` to additionally fire `/v1/completions` with `logprobs=5` and assert every emitted position's `top_logprobs` entry has at least 5 alternatives (exit 9 on missing object / dropped alternatives). Pins backlog item S6 broader-matrix — verifies the logprobs postprocessing path on the TPU runner emits the per-token alternative distribution that production clients (confidence scoring, structured-output reranking) rely on. Cheap — same prefill bucket as the existing completions probe. |

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

#### S1. Decode is not real decode — every step recomputes prefill on the full prompt+generated context

This is the headline correctness/perf bug. `__call__` in
`work/tpu-inference/tpu_inference/models/jax/deepseek_v4.py:1363`
always routes to `transformer_body_forward` →
`block_forward` → `attention_prefill`. The decode-step kernels
are fully implemented and correctness-tested
(`attention_decode_step` at
`layers/jax/attention/deepseek_v4_attention.py:710`,
`TestDecodeRollingParity` at
`tests/models/jax/test_deepseek_v4.py:1390` — 1e-4 vs torch
reference) but **never invoked** by the production `__call__`.

The `__call__` docstring at `deepseek_v4.py:1403` admits this:
"vllm in `--enforce-eager` paged-KV-disabled mode passes the full
prompt+generated context on each step, which makes this correct".

Net effect: every decode step is O((prompt + generated)²) in
attention compute and re-runs the full MoE per step instead of
O(1)/step over a cached state. Throughput at 1k context is
~10–50× worse than it should be; at 100k+ context decode is
non-functional, not just slow. V4-Flash's 1M-token claim is
unreachable on the current path.

What to do:
* Thread `AttentionDecodeState`
  (`deepseek_v4_attention.py:659`) through `nnx.Variable` storage
  on the model instance. The state is per-(layer, batch-slot) so
  it lives alongside `params_v` rather than in vllm's
  `kv_caches` list (which is a passthrough placeholder for V4 —
  see INVARIANTS I34).
* In `__call__`, branch on `attention_metadata.input_positions`
  per active sequence: `>0` ⇒ run a decode step using
  `attention_decode_step` per layer, mutating the state in
  place; `==0 .. N-1` ⇒ prefill a fresh state.
* Persist the new state back into `params_v`'s pytree leaves so
  the next call sees it. Verify byte-equivalence vs a
  prefill-only forward over the concatenated prompt+generated
  on at least 2 sequences from `TestConcurrentMultiSeqDispatch`.
* Validate via path #3 (`lower().compile()`) before launching a
  real smoke; the compile-time HBM accounting will tell you if
  the per-batch state allocations push you over 31.25 GiB / chip.

This unlocks A1 (lift `max-model-len`), B1 (sparse_attn Pallas
becomes worthwhile), and S5 (MTP speculative decoding becomes
meaningful).

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

#### S6. Sampling parameters — single-config + stop + logprobs probes DONE; top_k / n>1 / presence_penalty still TODO

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

**Stop-sequence probe (DONE — harness-verified, real-V4 verification deferred)**:
`STOP_REQUIRED=1` fires `/v1/completions` with `stop=["Paris"]` and
asserts (a) the response text does NOT contain `Paris` and
(b) `finish_reason="stop"`; exit 8 on either. The deterministic
baseline (`fire_completion`, no stop) emits ` Paris` as its first
token at temp=0/seed=0, so a working stop-sequence handler MUST
intercept that token, truncate, and report `stop`. Mock harness
covers `stop_required_match` (mock truncates correctly →
exit 0) and `stop_required_leak` (mock with `--stop-honor 0`
ignores the field → exit 8). Real-V4 verification deferred to
the next real-smoke run.

**Logprobs probe (DONE — harness-verified, real-V4 verification deferred)**:
`LOGPROBS_REQUIRED=1` fires `/v1/completions` with `logprobs=5`
and asserts every emitted position's `top_logprobs` entry has at
least 5 alternatives (exit 9 on missing object / dropped
alternatives). Mock harness covers `logprobs_required_match`
(mock emits 5 alts → exit 0), `logprobs_required_missing`
(mock with `--logprobs-alts 0` omits the object → exit 9), and
`logprobs_required_dropped` (mock with `--logprobs-alts 3` emits
fewer than requested → exit 9). Real-V4 verification deferred.
Production clients (confidence scoring, reranking, structured-
output likelihood) depend on this path; vLLM's TPU runner has
historically had quirks here (per-position emission can be
silently dropped under some sampling configs).

**Still TODO** — `top_k` (typically combined with `top_p` to bound
the candidate set), `presence_penalty` (different from
`frequency_penalty` — applied per-token rather than per-occurrence),
`n>1` (interacts with seq dispatch — under today's
`--max-num-seqs=1` this likely sequentializes; informative either
way). Each is a separate gated probe (`TOPK_REQUIRED=1`, etc.)
following the same pattern. Each new probe adds one fire_*
function + one extractor + one mock branch + 1–2 harness
scenarios + one CLAUDE.md entry. Total cost per probe ~50 LOC.

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
* **Stop-sequence + logprobs probes scaffolded** (`STOP_REQUIRED=1`
  + `LOGPROBS_REQUIRED=1` smoke_check + 18-scenario harness self-test).
  ✓ harness; ✗ real-V4 verification (deferred to next real-smoke run).
  Stop probe sends `stop=["Paris"]` against the deterministic prompt
  whose first token is ` Paris` — a working handler MUST truncate
  before that token and report `finish_reason="stop"`. Logprobs probe
  sends `logprobs=5` and asserts every position's `top_logprobs` has
  at least 5 alternatives. Mock covers leak (stop ignored) and dropped/
  missing (logprobs alts fewer than requested or omitted entirely).
  See S6.
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

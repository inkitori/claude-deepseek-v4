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
  (uses `scripts/_mock_openai_server.py` — no TPU needed). Also
  fires a `/v1/chat/completions` probe (informational by default,
  `CHAT_REQUIRED=1` to make missing/empty fail).
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

`/v1/chat/completions` returns 200 OK with the byte-equivalent
chat template applied (`scripts/v4_chat_template.jinja`). The chat
probe in `smoke_check` is informational by default — see backlog
S3/S4 for what's still loose on the chat path.

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

#### S3. `--reasoning-parser deepseek_v4` and `--tool-call-parser deepseek_v4` enabled in smoke launcher (launcher portion DONE; runtime assertion deferred)

`work/vllm/vllm/reasoning/__init__.py:31-32` registers
`deepseek_v4 → DeepSeekV3ReasoningParser` (handles
`<think>...</think>` block extraction →
`reasoning` / `reasoning_content` field).
`work/vllm/vllm/tool_parsers/deepseekv4_tool_parser.py`
provides `DeepSeekV4ToolParser` (DSML tool tokens → `tool_calls`).
Both are wired into `scripts/full_slice_v4_smoke.sh` via
`--reasoning-parser deepseek_v4`,
`--enable-auto-tool-choice`, `--tool-call-parser deepseek_v4`.
vLLM validates these names at startup and refuses to launch if
they're misregistered, so the smoke gate green = parsers loaded.

What's left for S3 (depends on S4): a runtime assertion that a
think-mode-triggering chat request produces a non-empty
`reasoning` field. Today's chat template
(`scripts/v4_chat_template.jinja`) emits `<｜Assistant｜></think>`
unconditionally — i.e. it tells the model "thinking is done,
answer now" — so the model never produces `<think>` blocks
regardless of the parser being wired. Once S4 lands a
thinking-mode-aware template that omits the `</think>` open and
respects `chat_template_kwargs.thinking=True`, add a smoke_check
chat probe that sets `chat_template_kwargs={"thinking": true}`
plus a reasoning-eliciting prompt and asserts the response's
`reasoning` field is non-empty. Same applies to a tool-using
probe — depends on S4's `tools` scope.

Sanity check that the parsers are still wired (no TPU needed):

```bash
PYTHONPATH=work/vllm:work/tpu-inference work/vllm_env/bin/python3 -c "
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
ReasoningParserManager.get_reasoning_parser('deepseek_v4')
ToolParserManager.get_tool_parser('deepseek_v4')
print('OK')"
```

#### S4. Chat template covers chat-mode only — think and tool modes silently produce wrong tokens

`scripts/v4_chat_template.jinja` is byte-identical to
`encode_messages(thinking_mode="chat")` for the user / assistant
/ system subset only. The reference encoder is at
`<hf-cache>/snapshots/<sha>/encoding/encoding_dsv4.py`. Each of
the four missing scopes needs a Jinja translation + a
byte-parity validation test against `encode_messages(...)`:

* `tools` array → tools encoded with the wrong delimiters →
  model ignores tool definitions
* `tool` role / `tool_calls` from prior assistant turns → don't
  round-trip → multi-turn tool-using conversations break
* `thinking_mode="think_high"` / `"think_max"` → must emit
  `<think>` instead of an immediate `</think>` → reasoning is
  currently always suppressed
* `latest_reminder` injection → DeepSeek's quick-instruction
  guidance is missing → quality regression on tasks that depend
  on it

Re-validation pattern is the snippet at the top of CLAUDE.md's
"Chat template" section.

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

#### S6. Sampling parameters are untested under load

The path is standard — V4 inherits tpu-inference's
`sampling.py` + `rejection_sampler.py` via `compute_logits` at
`deepseek_v4.py:1528`. But the only thing the smoke check fires
is `temperature=0, seed=0`. Nothing exercises temperature>0,
top_p, top_k, frequency/presence penalties, stop sequences,
multiple completions (`n>1`), or logprobs.

Add a sampling matrix to `smoke_check` (or a
`tests/test_v4_sampling_e2e.py` against the running smoke)
before claiming production readiness. Each combination has
known quirks under vLLM's TPU runner.

#### S7. Streaming (SSE) is unverified

vLLM's framework supports `stream: true` natively;
tpu-inference shouldn't need V4-specific changes. But the
smoke_check only fires non-streaming requests. Add a streaming
probe: assert TTFT < N seconds, ITL < M ms, and that the
reassembled stream matches the non-streaming output.
OpenRouter-style clients default to streaming.

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

#### B3. SPMD `Involuntary full rematerialization` audit

Each warning in the smoke log is XLA giving up on a sharding
spec and falling back to replicate + re-partition. They lengthen
compile and add slow runtime barriers.

```
grep "Involuntary full rematerialization" \
    logs/full-slice-v4-smoke-*.log
```

Group by sharding pair, fix the worst offenders by adding
`with_sharding_constraint` or `_replicate` calls. The
`compressor.ape` family was already eliminated by a `_replicate`
in `compressor_prefill`/`compressor_decode_step`; same pattern
likely applies to others.

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
multilingual code blocks, very-long single tokens. The V4
tokenizer config's BOS handling
(`v4_chat_template.jinja:11` has `add_bos_token=false` so the
template emits BOS itself) is fragile — verify each role
transition encodes byte-identically to `encode_messages()`.

#### C5. Refusal/safety behavior preservation

V4 was tuned for specific refusal patterns. After kernel
rewrites this often regresses (low-bit MoE quant especially can
shift the safety tuning). Run a small refusal-eval set (a few
dozen prompts spanning the model card's tested categories)
before each significant change to S1/B2.

### Tier D — code-hygiene / janitorial

#### D1. Test bloat

`tests/models/jax/test_deepseek_v4.py` is **2997 LOC, 30 test
classes** (~4× the next biggest model's test file). Concrete
fold-ins:
* `TestRealFp8DequantSmoke` is strictly weaker than
  `TestFp8DequantIndependentReference`. Fold the smoke into the
  reference class as a second test method or drop it.
* `TestRealFp4DequantSmoke` ↔ `TestFp4DequantIndependentReference`
  — same.
* `TestFp8Dequant` (synthetic-fixture full-loader bit-identical)
  and `TestFp8CastByteDomain` (256-byte numpy-vs-torch parity)
  are distinct concepts — keep both.
* `TestFp4CodebookReference` exhaustively enumerates the 16-entry
  codebook — distinct from the byte-equal real-data tests.

Re-measure before claiming progress: `wc -l` and
`grep -c "^class Test"`.

#### D2. Stale comment cleanup

The `__call__` docstring at `deepseek_v4.py:1395-1409` talks
about "Tier 8 first pass" and "BLOCKERS.md B1 followup" —
Tier-8 is green, B1 is archived. Update the docstring to reflect
the *current* contract (still prefill-stateless decode, but say
so as the contract not as a deferred item).

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
* **Chat template (chat-mode subset)**: byte-equivalent to
  `encode_messages(thinking_mode="chat")` on representative
  inputs. ✓ for chat-mode only — see S4 for the missing scopes.
* **Reasoning + tool parsers wired** (`--reasoning-parser deepseek_v4`,
  `--enable-auto-tool-choice --tool-call-parser deepseek_v4` in
  `scripts/full_slice_v4_smoke.sh`). Registry lookup verified via
  the snippet in S3. vLLM validates parser names at startup, so
  smoke-green = parsers loaded. ✓ wiring, ✗ runtime emission test
  (S3 — depends on S4's thinking-mode template).

## Chat template (chat-completions)

V4-Flash deliberately ships **no Jinja `chat_template`** —
`tokenizer_config.json` omits the field and the upstream HF
README points users at the Python encoder at
`<snapshot>/encoding/encoding_dsv4.py`. Without a template, vllm
falls back to a generic format and `/v1/chat/completions` returns
garbage.

`scripts/v4_chat_template.jinja` is the byte-equivalent Jinja
translation of `encode_messages(thinking_mode="chat")` for the
system / user / assistant subset. The smoke launcher passes it
via `--chat-template`; the `smoke_check` runs an informational
chat probe. **Scope is chat-mode only** (no thinking, no tools,
no tool results, no `latest_reminder`, no quick-instruction
tasks) — backlog item S4 covers the missing scopes.

To re-validate the existing scope vs `encode_messages()`:

```python
import sys, os
SNAP = "<hf-cache>/snapshots/<sha>"
sys.path.insert(0, os.path.join(SNAP, "encoding"))
from encoding_dsv4 import encode_messages
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(SNAP, "tokenizer.json"))
tok.add_special_tokens({"bos_token":"<｜begin▁of▁sentence｜>",
                         "eos_token":"<｜end▁of▁sentence｜>"})
tmpl = open("scripts/v4_chat_template.jinja").read()
msgs = [{"role":"user","content":"hi"}]
assert encode_messages(msgs, thinking_mode="chat") == \
    tok.apply_chat_template(msgs, chat_template=tmpl,
                             tokenize=False, add_generation_prompt=True)
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
  upstream; the smoke launcher just doesn't enable them yet (S3).
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

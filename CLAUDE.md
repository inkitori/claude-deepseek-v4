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
HEAD `b3e44530` keeps S1 Option C in place — `lax.optimization_barrier`
on each output packed buffer in `deepseek_v4_run_with_decode_state`
— with the prior 14-callback-per-layer band-aid removed. CPU repros
pass at tiny + V4-Flash-truncated dims. **TPU XLA strips
`optimization_barrier` in compile** (verified empirically:
`s1_cpu_hlo_check.py` with `JAX_PLATFORMS=tpu` shows lowered HLO has
6 barriers, compiled HLO has **0**; the with-anchor vs no-anchor
compiled HLO is byte-identical at 0 barriers / 174 dynamic-update-
slice ops). But TPU jit+donate decode at V4-Flash-truncated dims
*still passes* (8/8 byte-equal to eager fresh-prefill argmax),
which means **donation aliasing alone preserves the writes at this
scale** — the anchor is a JIT no-op, not the safety net it was
believed to be.

Real V4-Flash on v6e-32 has **NOT** been re-verified post-fix at the
correct gate. The one post-fix smoke (mark's contested slice) gave
empty output + `SLICE_FAILURE_SW_INJECT_ERROR`; that was attributed
to the barrier being stripped, but the TPU validation above contradicts
that story — at truncated scale the JIT path is correct without the
anchor. The real failure is likely something else (numerical instability
at real weight values, sharding-specific interaction, or vLLM/Ray
multi-host glue). v6e-16 cannot fit V4-Flash (math: 256 experts × 43
layers × bf16 / 16 chips ≈ 34.6 GB/chip > 31.25 GB available; verified
by SIGSEGV-without-OOM-msg at "placed 2200 tensors, layer 10/43").
v6e-32 is the minimum.

**To pick this up cold:**
1. `git log --oneline -5` — current head should be `b3e44530` "ray_restart.sh:
   parameterize SLICE_SIZE for v6e-16" on branch `s1-v6e-16-bring-up`
   (pushed to origin).
2. CPU + TPU repros confirming the JIT path works at truncated scale:
   * `scripts/s1_cpu_repro_tiny.py` — tiny config, ~30s eager / ~5s jit on CPU.
   * `scripts/s1_cpu_repro_v4flash.py` — V4-Flash truncated to 4 layers
     with 8 experts (full V4-Flash hidden_size / 64 heads / 1024 q_lora_rank
     / 512 index_topk). **CPU**: ~30s param init / ~85s jit prefill / ~80s
     jit decode. **TPU** (single-host 4-chip, `JAX_PLATFORMS=tpu`): ~45s
     param init / ~17s eager prefill / ~77s jit prefill / ~80s jit decode,
     8/8 byte-equal.
   * `scripts/s1_cpu_hlo_check.py` — dumps lowered + compiled HLO of one
     decode step. **CPU**: lowered=6 barriers, compiled=0, 132 dynamic-
     update-slice. **TPU**: lowered=6 barriers, compiled=**0**, 174
     dynamic-update-slice. TPU XLA strips the barrier — but compiled
     HLO still has all the writes via donation aliasing.
   * `scripts/s1_tpu_anchor_compare.py` (TPU-only) — proves the anchor is
     a JIT no-op on TPU. With-anchor vs no-anchor compiled HLO is byte-
     identical at 0 barriers / 174 DUS / 147 copy / 21 scatter / 254
     concatenate.
   * `scripts/s1_tpu_sharded.py` (TPU-only) — runs the V4-Flash-truncated
     decode parity on a sharded mesh (`attn_dp=local_chip_count`,
     `kv_caches` replicated with `P()`). Diagnoses sharding-axis
     interaction with `at[].set` writes.

   Run TPU variant:
   ```
   ssh <head> 'cd ~/claude-deepseek-v4 && \
     TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 \
     TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
     PYTHONPATH=work/vllm:work/tpu-inference JAX_PLATFORMS=tpu \
     work/vllm_env/bin/python3.12 scripts/s1_cpu_repro_v4flash.py both 8 8 4'
   ```
   Both repros end in "OK: both eager and jit match fresh-prefill argmax".
   Needs `jax==0.9.2`, `numpy`, `torch`; the on-host venv at
   `work/vllm_env/` works.
3. Real V4 verification needs a v6e-32 slice — v6e-16 doesn't fit (above).
   See "Slice bootstrap" section below.
4. If real-V4 output is broken on v6e-32, set `V4_DECODE_NAN_TRIPWIRE=1` to
   emit per-field nan/inf/max_abs diagnostics. **Note**: as of HEAD,
   `_v4_nan_tripwire` is a hard early-return when this flag is 0 — the prior
   "silent callback even when disabled" anchoring behaviour is GONE. If
   the v6e-32 failure was actually being masked by those silent callbacks
   (a possibility worth exploring), partially reverting to the pre-fix
   callback emission may be the right move.

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

**Current fix — Option C (output-side `lax.optimization_barrier`),
but it's a JIT no-op on TPU:**
`_v4_anchor_output_buffers` in `models/jax/deepseek_v4.py:772`
wraps each output packed buffer in `lax.optimization_barrier`
before they're returned from `deepseek_v4_run_with_decode_state`.
The intent was to force XLA to materialise every `at[].set` and
concatenate upstream of the barrier. Host callbacks were removed
(deferred behind `V4_DECODE_NAN_TRIPWIRE=1`).

**What TPU validation actually showed (2026-05-23):**
* TPU XLA strips `lax.optimization_barrier` in compile — same as CPU.
  Verified by `s1_cpu_hlo_check.py` with `JAX_PLATFORMS=tpu`:
  lowered HLO has 6 barriers, compiled HLO has **0**.
* With-anchor vs no-anchor compiled HLO is byte-identical at 0
  barriers / 174 dynamic-update-slice / 147 copy / 21 scatter /
  254 concatenate ops. The anchor is a complete JIT no-op on TPU.
* `scripts/s1_cpu_repro_v4flash.py` run via `JAX_PLATFORMS=tpu`
  (single-host, 4 chips, no mesh): **PASS** — 8/8 decode steps
  byte-equal to eager fresh-prefill argmax under jit+donate. The
  V4-Flash-truncated JIT decode is correct on TPU **without any
  effective anchor**.

This means the original S1 hypothesis ("XLA elides at[].set writes
under donation aliasing") is **false at this scale**. Donation
aliasing alone preserves the writes — XLA materialises them as
dynamic-update-slice ops that operate in-place on the donated buffer.
The 14-callback band-aid wasn't fixing an elision bug; whatever it
was masking on real V4-Flash v6e-32 is something else (numerical
instability at real weights, sharding-specific interaction, multi-
host SPMD glue).

**The big unknown:** real V4-Flash on v6e-32 has not been
re-verified with the current head. The one attempt after committing
the fix (mark's contested slice) produced empty completion +
`SLICE_FAILURE_SW_INJECT_ERROR`. That was attributed to the barrier
being stripped, but per the above the barrier is irrelevant —
removing it changes nothing in the compiled HLO. The real cause
needs a fresh v6e-32 run with `V4_DECODE_NAN_TRIPWIRE=1` to localize
the first field that NaNs/Infs.

**Validation matrix:**
* CPU tiny-config (7 layers) + JIT + `donate_argnums=0`: passes.
* CPU V4-Flash-truncated (4 layers, 8 experts) + JIT + donate: passes.
* TPU V4-Flash-truncated (single-host, 4 chips, no mesh) + JIT +
  donate: passes (8/8 byte-equal). Run via
  `JAX_PLATFORMS=tpu scripts/s1_cpu_repro_v4flash.py both 8 8 4`.
* TPU V4-Flash-truncated, single-host with `attn_dp=4` sharded mesh
  + `P()`-replicated kv_caches + JIT + donate: passes (8/8 byte-
  equal). Run via `scripts/s1_tpu_sharded.py`. ~24s eager / ~151s
  jit prefill / ~158s jit decode.
* TPU HLO: anchor stripped in compile, writes preserved via donation
  (174 dynamic-update-slice in compiled HLO, identical with vs without
  anchor). Run via `scripts/s1_tpu_anchor_compare.py`.
* Real V4-Flash on v6e-32 with current head: **NOT YET RE-RUN**.
  v6e-16 cannot fit (34.6 GB/chip > 31.25 GB).

**If real-V4 on v6e-32 fails:** the right first move is
`V4_DECODE_NAN_TRIPWIRE=1` to localize the corruption. Then,
plausible structural fixes:
1. Restore the pre-fix unconditional silent callbacks in
   `_v4_nan_tripwire` (read-only `jax.debug.callback(lambda *_: None,
   ...)`). They may have been doing more than anchoring — host
   round-trips of bf16 values are bit-preserving for NaN/Inf, but
   ordered effects can prevent some XLA fusions whose rounding
   differs. Try this BEFORE concluding the bug is elsewhere.
2. Fold `at[].set` writes into a `pl.pallas_call` whose output is
   the new state (matches V3/Qwen3 pattern; merges with B1's
   sparse-attention Pallas kernel).
3. Audit numerical stability of the compressor / indexer paths
   under real weights — bf16 accumulation can saturate at real
   scale where 0.02-stddev random params don't.

**Don't repeat (full traces in `git log`):**
- `ac8d2077` — single per-layer callback in transformer body. Insufficient.
- `98b0a677` — single callback at kv_cache_post_write. NaN returns.
- `14e11136` — callbacks at at_entry × 6 fields. NaN returns.
- `5c9d9213` (rev `c32fe431`) — full un-donate kv_caches for V4. TPU UserFatal.
- `75b92f4b` (rev `9d2f15ec`) — input-side opaque copy. XLA optimized in-place.
- `1f212036` (current HEAD ancestor) — output-side `optimization_barrier`
  + removal of all silent callbacks. Stripped by TPU XLA in compile;
  v6e-32 smoke produced empty + SLICE_FAILURE. Possibly because the
  removed callbacks were masking a separate (non-elision) bug.

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

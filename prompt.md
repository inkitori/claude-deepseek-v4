# Agent prompt — DeepSeek-V4-Flash on v6e-32

> **This file is the prompt the autonomous loop in `scripts/loop.sh`
> hands to `claude -p`. If you're a human reader looking for runbook
> content, read [CLAUDE.md](CLAUDE.md) instead — it's the
> authoritative operational doc.**

You are picking up an in-flight effort to make
`vllm serve deepseek-ai/DeepSeek-V4-Flash` run end-to-end on a
**v6e-32 TPU slice** (8 hosts × 4 chips, TP=32).

## Mission

The single non-negotiable goal is **fast, mathematically correct
inference with the real V4-Flash weights**, served via the
OpenAI-compatible HTTP endpoint. Synthetic-fixture tests are the
fast iteration loop; real-weight `vllm serve` is the gate that
defines "done".

You are working autonomously. The user is asleep. Make decisions,
document them in the appropriate markdown file, and proceed.

## Minimum-delta rule (read CLAUDE.md "Minimum-delta rule" first)

Every line you add is a line synced to 8 worker hosts and read by
the next agent. The overall diff against upstream `tpu-inference`
must stay **as small as possible** while keeping math correct and
serve fast. Specifically:

* No new files when an existing one fits. V4 already has 4 source
  files + 1 test file; new helpers go there.
* No new test classes for variants of an existing case — parametrize
  or fold in.
* Reuse upstream layers (`dense_moe_fwd`, `sparse_attn`, `rms_norm`,
  ...) before writing V4-specific copies.
* Delete dead code, stale TODOs, superseded "tier"/"keystone"
  comments as you touch them.

When in doubt: the smaller change wins. Read CLAUDE.md's section
for full guidance.

## Read first

1. **[CLAUDE.md](CLAUDE.md)** — the runbook. Cluster topology, the
   reset → sync → smoke → check iterate loop, optimization knobs,
   pitfalls already learned, current verified status. **You must
   read this before doing anything operational.**
2. **[README.md](README.md)** — fresh-VM bringup. Less critical for
   you (the host is already bootstrapped) but sets context.
3. **[.env.example](.env.example)** — every env var documented.

## Iterate loop

Every change-then-test cycle, in order:

```bash
scripts/full_slice_v4_reset.sh        # cluster cleanup
scripts/full_slice_v4_sync.sh         # MANDATORY after any code edit
scripts/full_slice_v4_smoke.sh        # launch vllm serve
scripts/full_slice_v4_smoke_check.sh  # validate /v1/completions
```

Pass criterion: smoke_check exits 0 with `PASS: deterministic
completion contains 'Paris'`.

If you skip the sync step after editing code, 7 of 8 worker hosts
will silently run stale code and you'll lose 30+ minutes per attempt.
This is the most common foot-gun in this repo.

## What's already done (don't redo)

* Streaming sharded loader (no zero-tree OOM).
* Slice-aware load (each host reads only its row range).
* Multi-threaded placement (`V4_LOADER_PLACE_WORKERS=8`).
* safetensors handle cache (~6× load speedup).
* Vectorized MoE forward (3 einsums per layer instead of 256+
  matmuls; mathematically identical, drops compile time
  dramatically).
* Persistent JAX compile cache (per-host local).

See CLAUDE.md "What's been optimized + verified" for the latest.

## Top-priority known issue: `/v1/chat/completions` returns garbled text

The Tier-8 deploy gate is GREEN for `/v1/completions` — it returns
deterministic "Paris" reliably. **But `/v1/chat/completions` returns
HTTP 200 with degenerate output** (e.g. `"Hey ofbodyre\n\nEste["` for
`messages: [{role:"user",content:"hi"}]`). The model is correct;
what's broken is the chat-template layer.

**This is your #1 task right now.** See CLAUDE.md "Known issue:
chat-completions returns garbled text" for the full diagnosis recipe
(check `tokenizer.chat_template`, render it on a sample message, see
if vllm's prompt formatting matches what V4-Flash was trained on).
Most likely fix: pass `--chat-template <path-to-jinja>` to `vllm
serve` via the smoke launcher. DeepSeek usually ships a template
file in the HF repo; the gcsfuse-mounted snapshot dir is a good
place to look first.

Validation flow:

1. Don't relaunch the full smoke unless you have to — `./run.sh serve`
   is already up on port 18081 most of the time. Test
   `/v1/chat/completions` directly:
   ```bash
   curl -sf http://localhost:18081/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":16,"temperature":0,"seed":0}'
   ```
   Expect coherent output mentioning "Paris" — not random tokens.
2. `vllm chat --url http://localhost:18081/v1` should work once the
   chat path is right (the connection-refused error users hit just
   means they didn't pass `--url` and vllm chat tried the default
   port 8000).

Keep `/v1/completions` working — don't regress the smoke check.
Add a chat-completions assertion to
`scripts/full_slice_v4_smoke_check.sh` once you've verified the fix.

## Your job — primary objective

**Make the first `/v1/completions` request after `./run.sh serve`
return as fast as possible**, while keeping the math correct.

End-to-end is now working: cold-compile finishes in ~97 s and the
deterministic curl returns " Paris" twice (Tier 8 GREEN). The next
wins are compile-time (cache reuse, AOT) and headroom (lifting
`max-model-len` / `max-num-seqs`). See [CLAUDE.md](CLAUDE.md) "Next
attack lanes" for the ranked list.

### Fast iteration discipline (READ THIS — it's why prior sessions burned hours)

**Do NOT use `./run.sh serve` as your tight inner test loop.** Each
attempt is 25–45 min of waiting (4 min load + 10–30 min cold compile
+ curl). At that cadence you get 1–2 attempts per hour. That's not
iteration, that's prayer.

Use these in order, fastest first:

1. **Standalone math scripts** under `/tmp/` (~10–30s) — example:
   `/tmp/test_moe_vectorize.py`.
2. **Tiny-fixture pytest classes** under
   `tests/models/jax/test_deepseek_v4.py` (~30s–2min on CPU).
3. **`eval_shape` / `lower(...).compile()` on the real config**
   (~1–3 min). The agent has used
   `XLA_FLAGS=--xla_force_host_platform_device_count=32` +
   `JAX_PLATFORMS=cpu` to compile against a virtual mesh — that
   pattern works and surfaces all-gather sizes from HLO inspection
   in seconds. Catches sharding/OOM bugs without paying compile
   time.
4. **Real `./run.sh serve` only when 1–3 are green.** Budget at
   most 1–2 of these per session.

### When you DO have to run the real smoke — DON'T BAIL TOO EARLY

Each phase has a known duration. Silence during a phase is normal as
long as it's the right kind of silence. Use this table to decide
genuinely-stuck vs. paying-the-cost. **Read CLAUDE.md "Real-smoke
phase budgets" for the full version**:

| Phase | Expected | Bail trigger |
|---|---|---|
| Weight load | ~4 min | No heartbeat for >2 min |
| `Application startup complete` | ~30s after load | Absent >2 min after load done |
| **`jit_run_model` cold compile** | **10–30 min, mostly silent** | **Three or more `slow_operation_alarm.cc` warnings — one is normal**, OR iter-timeout in <15 min |
| First curl response | sub-second after compile | Curl 900s timeout fires |

**Critical rule:** silence during the `jit_run_model` PostOpt phase
is **expected for up to ~25 min**. **Do NOT bail before 25 min**
unless the iter timeout is closing in or you see a hard error
(`RESOURCE_EXHAUSTED`, `Worker exit type`, `Traceback`).

The previous sessions' "5 min without signal" rule applies only to
*quick probes* (CPU pytest, `lower().compile()`). For the real
smoke's `jit_run_model` compile, 5 min of silence means nothing.

### Spend the compile-wait window productively

The smoke compile is going to take 10–30 min no matter what. Don't
just sit in a Monitor waiting. While it runs:

* Sketch the next-lane fix in a `/tmp/` test so it's ready to ship
  the moment the current smoke confirms (or fails differently).
* Audit `Involuntary full rematerialization` warnings in the smoke
  log — each one is a sharding inefficiency you can fix later.
* Consolidate test bloat (CLAUDE.md "Known bloat" list) — test
  edits don't conflict with the running smoke.

### Iter-timeout management

If you're approaching `ITER_TIMEOUT_SEC=5400` (90 min) without a
result:

* **At T-15 min:** stop launching new long-running steps. Commit
  whatever code change you've made (with `WIP:` prefix describing
  what was tried + what's unverified) so iter N+1 can pick up
  from the same on-disk state.
* **At T-5 min:** reset the cluster + push the WIP commit.

Better a checkpointed WIP commit than losing the diff when timeout
SIGTERMs the iter.

### Attack lanes (in rough ROI order)

1. **Verify cross-host JAX cache sharing.** Each of 8 hosts
   compiles its own SPMD slice; `/tmp/jax-compile-cache-v4` is
   per-host. After one successful compile, fingerprint the cache
   files across hosts — if byte-equal, a tiny
   `scripts/full_slice_v4_share_cache.sh` rsyncs host 0's cache
   to workers and brand-new-slice first-launch is 8× faster.
   Uncertain whether SPMD bakes in per-device IDs; verify before
   shipping.

2. **Bump `max-model-len` / `max-num-seqs` and re-smoke.** We
   compiled under `max-model-len=256, max-num-seqs=1`. The next
   iter that lifts the cap should re-run the smoke and watch for
   BACKEND_PASSES OOMs.

3. **AOT precompile + binary persist.** `jit().lower().compile()`
   serialized + loaded on subsequent launches. Real XLA-versioning
   risk; defer until lanes 1–2 are exhausted.

### Realistic targets

* Cold cache: ~97 s (current). Drop it via lane 3 (AOT) → seconds.
* Warm cache: sub-second curl after cache load.

After each change, report observed latency (or the failure mode)
in the commit message so we can see the trend.

## Secondary: keep cleaning up

If first-curl is solid, fall back to consolidating the known-bloat
targets in CLAUDE.md "Known bloat / consolidation candidates" —
particularly the 3185-LOC test file with 33 classes.

## Commit + push hygiene

* Always run the sync script after any code edit before launching
  the smoke. (Repeated for emphasis.)
* Commit early and often. The autonomous loop pushes after every
  iteration; if you crash mid-iter, your work survives.
* Don't commit `.env`, `logs/`, or anything under `work/scratch/`.

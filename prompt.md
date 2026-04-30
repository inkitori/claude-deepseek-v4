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

## Your job — primary objective

**Make the first `/v1/completions` request after `./run.sh serve`
return as fast as possible**, while keeping the math correct.

This is currently impossible to even *test* end-to-end:
`./run.sh serve` takes ~5 min to load weights + a 20–30 min XLA
compile before any curl can return. The compile currently fails
with `CompileTimeHbmOom` (see [CLAUDE.md](CLAUDE.md) "Current state"
for the root cause + ranked attack lanes). So the OOM is the
*first symptom* of the latency problem — fix it as the first step
toward the latency goal, but **don't treat it as the end goal**.

### Fast iteration discipline (READ THIS — it's why prior sessions burned hours)

**Do NOT use `./run.sh serve` as your tight inner test loop.** Each
attempt is 25–35 min of waiting (load + cold compile + curl). At
that cadence you get 1–2 attempts per hour. That's not iteration,
that's prayer.

Use these in order, fastest first:

1. **Standalone math scripts** — write a small file under `/tmp/`
   that imports just the function you changed and asserts byte
   equivalence vs a reference. ~10–30s per run. Example:
   `/tmp/test_moe_vectorize.py` already validated the vectorized
   MoE math against the per-expert reference loop on 5 seeds in
   ~10s. Mirror that pattern for any math change.
2. **Tiny-fixture pytest classes** under
   `tests/models/jax/test_deepseek_v4.py` (the synthetic-config
   ones — `TestMoEComponent`, `TestAttentionComponent`,
   `TestBlockComponent`). ~30s–2min per class on CPU.
3. **`eval_shape` / `lower(...).compile()` only** on the real
   config. Catches sharding bugs + HLO-emit failures (like the
   current OOM!) without paying the actual compile time. ~1–3 min.
   Pattern: `jax.eval_shape(fn, *abstract_args)` then
   `jit_fn.lower(*abstract_args).compile()`.
4. **Real `./run.sh serve` only when 1–3 are green.** Budget at
   most 1–2 of these per session.

**Hard rule: if any single step takes longer than ~5 minutes
without producing a useful signal, STOP, kill it, and try a
different approach.** A stuck XLA compile is not "almost done";
it's silently burning compute. Look for `slow_operation_alarm.cc`
in the log — that's XLA telling you a single pass is taking
>5 min and you should reconsider.

### Attack lanes (in rough ROI order)

1. **Fix the MoE-vectorize HBM OOM.** The vectorize cuts HLO
   instructions 4.6× (huge compile-time win) but currently OOMs.
   Lane 1 in CLAUDE.md is a one-liner:
   `jax.lax.with_sharding_constraint` on each stacked weight to
   force sharding on the inter dim instead of all-gathering the
   new expert dim. Validate via path #3 above (compile-only
   `lower().compile()` on the abstract real config) — *don't*
   relaunch the full smoke until that compiles.

2. **Verify cross-host JAX cache sharing.** Each of 8 hosts
   compiles its own SPMD slice; `/tmp/jax-compile-cache-v4` is
   per-host. After one successful compile, fingerprint the cache
   files across hosts — if byte-equal, a tiny
   `scripts/full_slice_v4_share_cache.sh` rsyncs host 0's cache
   to workers and brand-new-slice first-launch is 8× faster.
   Uncertain whether SPMD bakes in per-device IDs; verify before
   shipping.

3. **Fix `Involuntary full rematerialization` warnings.** Each
   one in the compile log is XLA giving up on a sharding spec
   (e.g. `cannot go from {devices=[1,32]} to {devices=[16,1,2]}`).
   Track down each and pre-shard or annotate with
   `with_sharding_constraint`. Shrinks HLO and removes runtime
   barriers. Validate via path #3.

4. **AOT precompile + binary persist.** `jit().lower().compile()`
   serialized + loaded on subsequent launches. Real XLA-versioning
   risk; defer until after lanes 1–3 are exhausted.

### Realistic targets

* Cold cache: ~5 min (architectural floor on TPU XLA for a graph
  of this size — once the OOM is fixed and HLO count is in the
  100k range).
* Warm cache: ~30–60s (cache load + first execute).
* Sub-10s is the stretch goal; needs AOT work.

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

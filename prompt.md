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

**Fix the `CompileTimeHbmOom` so `./run.sh serve` actually returns
a curl response.** The detailed failure breakdown lives in
[CLAUDE.md](CLAUDE.md) "Current state (READ BEFORE LAUNCHING)" with
4 ranked attack lanes. Start with lane 1 (`with_sharding_constraint`
on the stacked MoE weights) — it's a one-line fix per stack and
should preserve the 4.6× HLO instruction count win.

Once curl returns, the **secondary objective** is reducing first-curl
latency. The graph is currently ~103k HLO instructions; cold compile
takes ~10 min. Realistic targets:

* Cold cache: ~5-10 min (architectural floor on TPU XLA for a graph
  of this size).
* Warm cache: ~30-60s (cache load + first execute).
* Sub-10s is the stretch goal but requires AOT-precompile work.

Attack lanes for first-curl (after the OOM is fixed):

1. **Verify cross-host JAX cache sharing is safe.** Each of the 8
   hosts currently compiles its own slice of the SPMD program;
   `/tmp/jax-compile-cache-v4` is per-host. After a successful
   compile, fingerprint the cache files (sha256 the .pb / .bin
   contents on every host) — if host 0's matches hosts 1-7, write
   `scripts/full_slice_v4_share_cache.sh` to rsync host 0's cache
   to all workers after a bootstrap-time warmup. Brand-new-slice
   first-launch becomes 8× faster on cold. Uncertain: SPMD might
   bake in per-device identifiers that break byte-equality. Verify
   before shipping.

2. **Fix `Involuntary full rematerialization` warnings.** Every one
   in the smoke log is XLA giving up on a sharding spec (e.g.
   `cannot go from {devices=[1,32]} to {devices=[16,1,2]}`). Track
   down which leaf or activation generates each and either pre-shard
   it at load-time or annotate the forward with
   `with_sharding_constraint`. Shrinks HLO and removes runtime
   barriers.

3. **Reduce HLO instruction count further.** Look for Python-level
   loops over heads / layers / hash shards that XLA is unrolling.
   Consolidate where the same memory-safety considerations as the
   MoE vectorize are met (don't create unsharded stacks).

4. **AOT precompile + binary persist.** Use
   `jit().lower().compile()` to produce a serializable XLA
   executable, load on subsequent launches without re-tracing. Real
   work; non-trivial XLA-versioning risk.

After each change, validate with the loop:

```bash
./run.sh serve
./scripts/full_slice_v4_smoke_check.sh   # PASS = "Paris"
```

Report observed first-curl latency (or the failure mode) in the
commit message so we can see the trend.

## Secondary: keep cleaning up

If the OOM is fixed and first-curl is already optimal, fall back to
consolidating the known-bloat targets in CLAUDE.md "Known bloat /
consolidation candidates" — particularly the 3185-LOC test file
with 33 classes.

## Commit + push hygiene

* Always run the sync script after any code edit before launching
  the smoke. (Repeated for emphasis.)
* Commit early and often. The autonomous loop pushes after every
  iteration; if you crash mid-iter, your work survives.
* Don't commit `.env`, `logs/`, or anything under `work/scratch/`.

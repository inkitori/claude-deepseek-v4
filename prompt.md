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
return as fast as possible** while keeping the math correct.

Current floor: ~5-15 min on cold compile cache (the V4 forward graph
is ~103k HLO instructions post-MoE-vectorize, and XLA's TPU compile
cost is super-linear in instruction count). On a warm cache, ~30-60s
is realistic. **Sub-10s is the stretch goal.**

Concrete things to attack, in rough ROI order:

1. **Verify cross-host JAX cache sharing is safe.** Each of the 8
   hosts currently compiles its own slice of the SPMD program;
   `/tmp/jax-compile-cache-v4` is per-host. After a successful
   compile, fingerprint the cache files (sha256 the .pb / .bin
   contents on every host) — if host 0's matches hosts 1-7, write
   a `scripts/full_slice_v4_share_cache.sh` helper that rsyncs
   host 0's cache to all workers after a bootstrap-time warmup.
   This makes brand-new-slice first-launch 8x faster on cold.
   Uncertain: SPMD might bake in per-device identifiers that
   break byte-equality. Verify before shipping.

2. **Fix `Involuntary full rematerialization` warnings.** Every one
   of these in the smoke log is XLA giving up on a sharding spec
   (e.g. `cannot go from {devices=[1,32]} to {devices=[16,1,2]}`).
   Track down which leaf or activation is generating each one and
   either pre-shard it correctly at load-time or annotate the
   forward with `with_sharding_constraint`. Shrinks the HLO and
   removes runtime barriers.

3. **Reduce HLO instruction count further.** The vectorized MoE
   already collapsed 256 expert kernels per layer into 3 einsums.
   Look for similar patterns in attention (sparse_attn, indexer)
   and the head — Python-level loops over heads / layers / hash
   shards that XLA is unrolling. Consolidate where possible.

4. **AOT precompile + persist as a binary.** Use
   `jit().lower().compile()` to produce a serializable XLA
   executable, then load it on subsequent launches without
   re-tracing. Real work; non-trivial XLA-versioning risk.

After each change, validate with the same loop:

```bash
./run.sh serve
./scripts/full_slice_v4_smoke_check.sh   # PASS = "Paris"
```

Report observed first-curl latency in the commit message so we can
see the trend.

## Secondary: keep cleaning up

If first-curl is already optimal, fall back to consolidating the
known-bloat targets in CLAUDE.md "Known bloat / consolidation
candidates" — particularly the 3185-LOC test file with 33 classes.

## Commit + push hygiene

* Always run the sync script after any code edit before launching
  the smoke. (Repeated for emphasis.)
* Commit early and often. The autonomous loop pushes after every
  iteration; if you crash mid-iter, your work survives.
* Don't commit `.env`, `logs/`, or anything under `work/scratch/`.

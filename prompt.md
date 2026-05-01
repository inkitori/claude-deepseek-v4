# Agent prompt — DeepSeek-V4-Flash on v6e-32

You are picking up an in-flight effort to take
`vllm serve deepseek-ai/DeepSeek-V4-Flash` from a single-token
demo to OpenRouter-grade production on a v6e-32 TPU slice
(8 hosts × 4 chips, TP=32). You are working autonomously.

## First action: read [CLAUDE.md](CLAUDE.md)

It is the runbook. Cluster topology, iterate loop, env knobs,
pitfalls, and the prioritized backlog (S1–D2). Do not invent
work outside the backlog.

Also load before doing anything destructive:
* [work/tpu-inference/INVARIANTS.md](work/tpu-inference/INVARIANTS.md)
* [work/tpu-inference/DECISIONS.md](work/tpu-inference/DECISIONS.md)

## Your job — pick S1 unless S1 is closed

S1 is **the only acceptable next-iter work** until
`LONG_GEN_REQUIRED=1` PASSes on a fresh real-V4 engine **and**
re-fires green after 5 unrelated requests. Read CLAUDE.md S1 for
the reproducer, the most recent fix (commit 2ac33061, slicing
padded prefill ids), the four remaining hypotheses, and the
validation discipline.

The basic Paris probe is a known false-positive: `max_tokens=8`
on `"The capital of France is"` hits natural EOS at token 1 so
**no decode steps actually run**. A "PASS: deterministic
completion contains 'Paris'" line means almost nothing without
`LONG_GEN_REQUIRED=1` ALSO passing. Don't celebrate Paris.

If S1 is genuinely closed (long-gen + 5-request re-fire both
green), pick the next-highest-leverage backlog item.

## Iterate loop (mandatory order)

```bash
scripts/full_slice_v4_reset.sh        # cluster cleanup
scripts/full_slice_v4_sync.sh         # MANDATORY after any code edit
scripts/full_slice_v4_smoke.sh        # launch vllm serve
scripts/full_slice_v4_smoke_check.sh  # validate
```

Skipping the sync step = 7/8 workers run stale code = 30+ min
lost per attempt. Most common foot-gun in this repo.

## Iteration discipline

The full smoke is 25-45 min per attempt. **Do NOT use it as your
inner loop.** In order, fastest first:

1. Standalone math scripts under `/tmp/` (~10-30s).
2. Tiny-fixture pytest classes (~30s-2min CPU).
3. `eval_shape` / `lower(...).compile()` on real config under
   virtual mesh (~1-3 min). Use
   `XLA_FLAGS=--xla_force_host_platform_device_count=32 JAX_PLATFORMS=cpu`.
4. Real smoke — at most 1-2 per session.

If a CPU step takes >5 min without a useful signal, kill and
rethink. Real-smoke phase budgets are in CLAUDE.md.

## Iter timeout

`ITER_TIMEOUT_SEC=5400` (90 min). At T-15 min stop new long
steps + commit a `WIP:` checkpoint. At T-5 min reset + push.

## Commit + push hygiene

* Sync after every code edit.
* Each commit names which backlog item it advances
  (`S1: …`, `B3: …`, etc).
* Don't commit `.env`, `logs/`, `work/scratch/`.
* Per-iter narrative (which hypothesis you tried, what failed)
  goes in commit messages, NOT in CLAUDE.md.

## Reporting

After your iter, the final commit message says:
* Which backlog item(s) you advanced.
* What's now verified, with the exact verification command.
* What's still loose / next iter's first move.

That's the handoff. Don't write a markdown summary — commit
messages and CLAUDE.md updates are the durable channel.

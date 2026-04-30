# Blockers (archived)

This file tracked deferred / partially-open items from the prior
overnight sessions (v3–v8). The structural blockers that prevented
`vllm serve` from reaching `/v1/completions` (B1 multi-seq dispatch,
B4 vLLM MLA-classification gate, etc.) are all resolved on the
current main branch; the iterate loop runs end-to-end.

The current real blockers, prioritized:

1. **First-curl latency.** Cold compile takes ~10–15 min on a clean
   `/tmp/jax-compile-cache-v4`. The vectorized MoE dropped HLO
   instruction count 4.6× (477k → 103k) but XLA's TPU compile is
   super-linear, so the absolute time is still long. See `prompt.md`
   "Your job — primary objective" for the four attack lanes.

2. **SPMD `Involuntary full rematerialization` warnings.** Every one
   of these in the smoke log is XLA giving up on a sharding spec
   (e.g. `[1,32] -> [16,1,2]`) and falling back to replicate +
   re-partition. They lengthen compile and add slow runtime barriers.

For day-to-day operational state and "what's been verified", read
[`../../CLAUDE.md`](../../CLAUDE.md). The historical narrative
(B1–B4, W1–W4, Tier 1–8 mythology) is preserved in `git log` if you
need it.

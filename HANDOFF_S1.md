# S1 handoff — DECODE BUG IS FIXED; closing steps remain

Goal: make `vllm serve deepseek-ai/DeepSeek-V4-Flash` produce coherent, deterministic
decode on the v6e-32 slice (bug **S1**). Ops details in `CLAUDE.md`; this is live state.

## STATE (2026-05-25, SESSION 7) — S1 DECODE BUG FIXED ✅

**decode == prefill-everything (FAITHFUL)** — verified this session on a fresh engine
(:18081). Greedy Fibonacci decode and the chained-mt=1 faithful reference are
**BYTE-IDENTICAL** through 14 tokens: `21, 等等。 34, 等等。 55,` — *including* a
0.16-logprob near-tie at the 等等-vs-55 step that BOTH paths resolved identically (⇒ the
two code paths' logits agree to <0.16; conclusive numerical equivalence). Decode
independently correct through term 7 (377: `21, 等等。 34, … 233, 等等。 377`).

**The "term-8 drift" chased for 2 sessions was a MISFRAME.** The model's greedy top-1
after "<number>," is genuinely **"等等"** (Chinese "etc."; logprob −1.25, beating "34" at
−3.52). prefill-everything emits the SAME "等等。" → it is the BASE MODEL's greedy quirk,
faithfully reproduced by decode, **NOT a decode bug**. (Re-confirms Session 5 commit
`3bf3ea9b`; Session 6's comp_full/i_cache cache-bug chase + reverted zeroing was a dead
end. The seed boundary slot `n_safe` is EXONERATED: with config `sliding_window=128`,
`index_topk=512`, ratios `[0,0,4,128,…,0]`, compression is always-on and the drift tracks
decode-step count, not prompt length / a compression boundary.)

**Gate status:**
* ✅ decode==prefill-everything (faithful) — the real done signal.
* ✅ Determinism: 3× byte-identical at temp=0 (`"Paris is the capital of France. Two famous
  landmarks there are"` → `" the Eiffel\nTower and the Notre Dame."`).
* ✅ Survives many requests (~28 served, no crash). Model correct on facts (Paris,
  Washington, five); greedy LOOPS on open-ended prompts = base-model (identical in
  prefill); sampling temp=0.8 non-degenerate (unique 15–25 vs greedy 5).

## NEXT ACTION — formal close (the bug is fixed; this is cleanup + tick the gate)

1. **Remove all S1 diagnostics** (CLAUDE.md requires it on close): `_v4_fp`/`_v4_dir`,
   `[seedfp]`, `[kvchk-pf]`, `[fwd*]`/`[dec*]`/`[pf4]`/`[moeRS]` in
   `models/jax/deepseek_v4.py` + `layers/jax/attention/deepseek_v4_attention.py`.
   (Bonus: without diagnostics decode is FAST — no longer 9–18 s/tok.)
2. **sync → clear xla_cache (code changed) → reset → ONE clean smoke** (cold compile).
3. **Tick the gate on the clean engine:** `/tmp/s1_div_logprobs.py "…Fibonacci…13, " 24 5`
   shows decode==prefill (now fast); 3× Paris determinism byte-identical. READ the text.
   NOTE: do NOT trust `smoke_check.sh` LONG_GEN — its open-ended greedy prompt makes the
   BASE MODEL loop (may fail visible_words≥10) and is the documented red herring; the
   faithful-comparison gate above is the correct one.
4. If clean → record DONE in CLAUDE.md + here, commit+push, **`touch /tmp/s1_loop_stop`**.

## PROBES (this session, in /tmp — reusable)
* `/tmp/s1_div_logprobs.py "PROMPT" N TOPK` — decode-vs-prefill + top-k logprob gap at
  first divergence (distinguishes real bug from benign near-tie). Run UNBUFFERED.
* Faithful chain is slow (~60–180 s/step: full re-prefill × always-on diagnostics) — that
  is why N>20 faithful comparison is painful TODAY; removing diagnostics fixes it.
* Engine NOT left running if you reset at end. Re-smoke per CLAUDE.md (warm xla_cache ⇒
  ~6 min; the clean-code artifact may already be cached). **First request of each new
  prompt SHAPE JIT-compiles (~5 min) — use timeout ≥580 s.**

## Durable lessons (kept)
* **decode==prefill-everything is THE done signal** (not "looks coherent", not
  smoke_check). Greedy looping/quirks (等等, repeated phrases) are the BASE MODEL — decode
  reproduces them faithfully. CPU parity is non-predictive (reverted S6 fix passed CPU,
  regressed slice).
* HARD CONSTRAINT (~8×): `with_sharding_constraint(ACTIVATION, P())` gathering a size-1
  decode token axis Core-halts; a wsc on a POST-reduction `[N,dim]` is safe.
* PRIMARY collapse fix (still live, real): SWA seed `6245ea84` + compressor/indexer STATE
  `90bf85c3` thread the traced `n_real`.

## Recovery / loop
* `different launch id`/`Core halted`/`SLICE_FAILURE` before startup = CODE DESYNC →
  `full_slice_v4_sync.sh` + clear+verify-empty `~/.cache/vllm/xla_cache/*` on all 8 hosts.
  Do NOT reboot. Clean engine → `full_slice_v4_reset.sh`. Escalate: `..._ray_restart.sh`.
* Keep node_guardian + meta_guardian alive before TPU work.
* Loop: `scripts/s1_session_loop.sh` (stop: `touch /tmp/s1_loop_stop`).

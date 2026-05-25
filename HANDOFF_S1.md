# S1 handoff — CLOSED ✅ (decode is coherent + deterministic)

Goal was: make `vllm serve deepseek-ai/DeepSeek-V4-Flash` produce coherent, deterministic
decode on the v6e-32 slice (bug **S1**). **DONE.** The loop is stopped
(`/tmp/s1_loop_stop`). This file is the closing record; ops live in `CLAUDE.md`.

## VERDICT (2026-05-25, SESSION 7)

S1 decode collapse is **FIXED and VERIFIED** on a fresh, diagnostic-free slice engine:

* **Coherent + correct:** greedy decode of the Fibonacci prompt →
  `21, 34, 55, 89, 144, 233, 377, 610` (all correct, no degenerate attractor); factual
  prompt → `the Eiffel Tower and the Notre Dame de Paris`. READ the actual text.
* **Deterministic:** 3× byte-identical at temp=0 on BOTH the Fibonacci and Paris prompts.
* **Faithful to the model:** decode ≈ prefill-everything to within ~0.24 logprob — identical
  through term 7; the only divergence is a **benign 0.236-logprob near-tie at term 8**
  (decode picks the CORRECT 610; the re-prefill path picks 676). Two distinct kernels
  cannot be bit-identical at coin-flip decisions — this is expected, not a bug.
* Survived ~31 requests across two fresh engines this session, no crash.

## What the prior sessions got wrong (so it isn't re-chased)

* The "term-8 drift / repeating 等等 (etc.)" that S5–S6 chased as a decode/cache bug was an
  **ARTIFACT of the always-on `jax.debug.print` diagnostics** (`_v4_fp` casts + forced
  materialization changed XLA fusion / fp reduction order, flipping near-tie argmaxes
  toward 等等). With the diagnostics removed (commit `77c0c7be`), decode produces clean
  correct Fibonacci. Lesson: **always-on heavy debug prints can perturb the numerics they
  measure — remove them and re-measure before trusting a "residual bug."**
* The comp_full/i_cache seed boundary slot `n_safe` was a dead end (S6 zeroing reverted).
  Compression is always-on (`sliding_window=128`, `index_topk=512`); the seed slot is not
  the cause. Exonerated.

## The real fixes that closed S1 (all committed, live)

1. Metadata-replicate decode (`tpu_runner._prepare_inputs_dp` `_v4_decode_replicate`) →
   deterministic decode (P() activations, not ATTN_DATA).
2. PRIMARY collapse: SWA seed `6245ea84` + compressor/indexer STATE `90bf85c3` thread the
   traced `n_real` so the decode seed is built over REAL tokens, not padding.
3. Remove always-on S1 diagnostics `77c0c7be` (was perturbing near-tie numerics + slowing
   decode to 9–18 s/tok).

## Durable lessons (keep)
* **decode-vs-prefill-everything on the slice** is the faithful test — but it agrees only to
  ~0.2 logprob (two kernels), so judge by COHERENCE + DETERMINISM + near-tie gaps, NOT
  byte-identity. CPU parity is non-predictive (CPU passed the reverted S6 fix).
* HARD CONSTRAINT (~8×): `with_sharding_constraint(ACTIVATION, P())` gathering a size-1
  decode token axis Core-halts; a wsc on a POST-reduction `[N,dim]` is safe.
* `_v4_nan_tripwire` (env-gated default-off) + `_v4_anchor_output_buffers`
  (`optimization_barrier` anti-elision anchor for the packed decode-state buffer) are REAL,
  kept. The `nan_to_num` clamp in `compute_logits` is REAL safety, kept.
* Recovery: `different launch id`/`Core halted` before startup = CODE DESYNC →
  `full_slice_v4_sync.sh` + clear+verify-empty `~/.cache/vllm/xla_cache/*` on all 8 hosts
  (ssh `-i ~/.ssh/google_compute_engine enyouki@HOST`). Do NOT reboot. Reset via
  `full_slice_v4_reset.sh`. Keep node_guardian + meta_guardian alive.

## If you must re-verify
Re-smoke per CLAUDE.md (sync → reset → smoke → wait "Application startup complete"), then
`/tmp/s1_div_logprobs.py "…Fibonacci…13, " 24 5` (decode clean Fibonacci) + 3× temp=0
determinism. Engine NOT left running (reset at close).

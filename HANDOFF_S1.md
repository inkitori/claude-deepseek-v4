# S1 handoff — variance LOCALIZED to the MoE ROUTED-EXPERT collective (L0); attention seed is CLEAN

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL (user-confirmed) — see memory [[s1-goal-reliable-coherence]]
DONE = **reliably coherent decode on EVERY fresh engine** (bug is a per-process coin flip). RIGOROUS gate:
**byte-identical (md5) FIB across 2 fresh engines AND coherent chat text.** Model is INSTRUCT — coherence
via `/v1/chat/completions` (system+user). Raw `/v1/completions` FIB = wedge-safe md5 determinism probe only.

## STATE (2026-05-26 SESSION 16) — DECISIVE LOCALIZATION (reverts S14, confirms S13)
cfc65ca4 (attention-seed input mask, S15) was VALIDATED on 2 fresh engines → **gate STILL FAILS**:
* ENG_A: FIB md5 `0a72aece` (3× identical, coherent `21,34,55,89,144,233,377`), chat `46aadc63` (correct first-15 then over-gen ramble).
* ENG_B: FIB md5 `ba467a77` (3× identical, coherent prose), chat `2b0dc29c` (correct `1,1,…,610,987`, clean stop).
* Both coherent + within-engine deterministic, DIFFER cross-engine ⇒ per-process variance PERSISTS.

**ROOT CAUSE (rigorous, noise-filtered).** Mined both engine logs for quantities STABLE-within-each-engine
but DIVERGENT-across (the ONLY valid signal — global `[ckS]` sums include per-process-CONSTANT idle-rank
garbage, so a raw A≠B is not proof). At L0:
* BYTE-IDENTICAL A==B: `seed_x_in`, `seed_kv_postlinear`, `seed_kv_cache`, `blk_attn_out`, `blk_post_attn_x`
  (= MoE input), `moe_perexpw` (gate), `moe_shared` (shared expert).
* FIRST DIVERGENCE: **`moe_routed_y`** (routed-expert collective output) — A=8.685e4 vs B=1.007e5.
⇒ **The ATTENTION SEED IS CLEAN** (S14's "variance upstream in the seed" is REFUTED; cfc65ca4 made the seed
checksums identical = it worked for the seed). The variance enters at the **MoE ROUTED-EXPERT collective**
(the S14 shard_map all_gather/psum, `deepseek_v4_moe.py` ~241-268). Same input + same gate + clean shared
expert, only the routed COLLECTIVE diverges. S13's MoE-L0 localization was RIGHT; S14 shard_map is PARTIAL.
Evidence logs: `logs/full-slice-v4-smoke-20260526T074717Z.log` (A), `...081139Z.log` (B).

## ⚠ CRITICAL CAVEAT — confirm REAL rows before fixing (prior sessions burned smokes on this)
`moe_routed_y` is a GLOBAL sum. For E=256, attn_dp=32 (N%32==0, 8 experts/rank) there is NO padding/idle-
expert slot, so the shard_map has no obvious uninit read (the "E%axis mask" idea is a NO-OP here — dead).
The A≠B could be (a) REAL-row corruption (the bug) OR (b) per-process-constant garbage in PAD/idle ROWS
(rows>=n_real) of the sum (noise). MUST disambiguate with a REAL-ROWS-ONLY checksum.

## NEXT ACTION
1. **Confirm with a REAL-ROWS-ONLY checksum.** In `moe_forward` add a checksum of routed-y over rows < n_real:
   `((jnp.arange(N) < n_real)[:,None] * y_routed).sum()` (real rows are at GLOBAL [0,n_real)). Plumb `n_real`
   into `moe_forward` if absent (it's traced in the seed path; thread from `block_init_state_and_forward`).
   Also checksum MoE input + `moe_shared` the same way as controls. CPU-validate (`s1_cpu_repro both`), sync,
   2 fresh engines, diff the real-rows values.
   * **REAL rows DIVERGE** → routed collective corrupts real rows → fix the collective (see 2).
   * **REAL rows CLEAN** → it's idle/pad-row global-sum noise; moe_routed_y is NOT the bug — re-localize the
     next stable-divergent REAL-rows quantity downstream (lm_head/final-norm token-axis reduction is the one
     UNTESTED-by-location suspect; decode-step is the other).
2. **Fix lead (if real rows diverge):** production `fused_moe_gmm.py` (lines 236-238, `valid_rows_mask` →
   `jnp.where(mask, token_topk_hidden, 0.0)` BEFORE the per-rank sum/psum) makes idle ranks emit zeros into
   the collective; bespoke `deepseek_v4_moe.py` LACKS any pre-psum mask (line ~257). PORT a pad-row mask
   (`jnp.where(arange(N)<n_real, ..., 0)`) before the psum — BUT note: the two impls use DIFFERENT token↔expert
   decompositions, so verify the mask semantics; the audit-agent's specific `(r+1)*NP` mask code was WRONG.
   Alternative: adopt the production `fused_moe_gmm` for V4 outright. LOCAL elementwise only — NO new
   `with_sharding_constraint` gathering a size-1/empty axis (pitfall #5, Core-halts ~8×).
3. Validate: 2 fresh engines FIB md5 identical + coherent chat (READ TEXT) + survives 5 reqs.

## KEEP cfc65ca4 (attn-seed input mask): CONFIRMED it makes the seed deterministic (identical seed checksums
A==B); sound + matches torch/vLLM-GPU reference (seed built only from real positions). Revert only if it
complicates the MoE work.

## DONE gate: md5-identical FIB across TWO fresh engines + coherent chat (READ TEXT). Engine CORE-HALTS on
PARIS shape → FIB-only for completions. Smoke serves max_model_len=256 (chat max_tokens<=256).

## Tools / ops
* Probes: `/tmp/s1_warmup.py` (FIB — RUN FIRST: primes the ~345s cold compile of the completions+logprobs
  path; `s1_fib2.py`'s 180s per-req timeout is TOO SHORT for that cold compile and will look like a hang),
  `/tmp/s1_fib2.py <label>` (FIB×3 md5 — wedge-safe determinism), `/tmp/s1_chat.py <label> [prompt]` (instruct
  coherence). Results → `/tmp/s1_eng{A,B}_results.txt`.
* On-disk `xla_cache` is WARM for current code ⇒ engine startup ~5-6 min; after engine A's warmup primes the
  logprobs+chat compiles, engine B's probes are fast (shared on-disk cache). KEEP the cache (don't clear)
  unless a launch-id halt. Per-engine smoke ≈ startup + 1 warmup compile + probes.
* Checksums via `_v4_checksum` (`deepseek_v4_attention.py:70`); labels seed_*/blk_*/moe_*. GLOBAL = noisy →
  ADD real-rows-only versions for localization; REMOVE ALL diagnostics when S1 closes.
* Slice HEALTHY (many clean smokes S16, no halts). Guardians up (node 497956, meta 4039835). Reset CLEAN 0/32.
  `/tmp/s1_loop_stop` NOT set. Both engines were reset down — slice is idle/clean.

## DEAD (do not retry)
* **cfc65ca4 attn-seed mask ALONE** — does NOT achieve cross-engine determinism (variance is in MoE routed, not
  seed). KEPT as a sound partial.
* **MoE shard_map ALONE (5ca26d66)** — PARTIAL; routed collective still admits per-process variance.
* **E%axis expert-slot mask in the shard_map** — NO-OP for E=256 (256%32==0); not the bug.
* XLA `--xla_tpu_*_collective_matmul_mode` flag — REJECTED by libtpu. fp32 matmul (S12). Option A prefill-
  replicate → NaN/all-BOS. Zeroing MoE output/input/pad rows (S6/S13) — no-op (collective/scratch, not data).
  Zeroing comp_full/i_cache SEED pad-slots (S6) — regressed decode. `wsc(activation,P())` gathering empty/idle/
  size-1 token axis → Core-halts (~8×).

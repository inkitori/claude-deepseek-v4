# S1 handoff — corruptor DECISIVELY localized to routed gmm OWNED rows; testing zero_initialize=False

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL / how to judge (see [[s1-symptom-nondeterminism-not-collapse]])
Bug is NOT collapse. Symptoms: (1) cross-process non-determinism (FIB md5 differs across fresh engines;
deterministic WITHIN a process), (2) quality drift. DONE = byte-identical md5 across 2 fresh engines AND
correct Fibonacci. Model is INSTRUCT. Validate: warmup absorbs ~336s recompile, then compare moe_routed +
FIB md5 across 2 fresh engines.

## STATE (2026-05-26 S23) — DECISIVE localization: corruptor = routed gmm OWNED rows
2-engine diagnostic (eng1 log 162729Z, eng2 log 164055Z; full data /tmp/s1_eng1_s23.txt) with new layer-0
checksums [ckR0]routing [ckR2]owned-g1 [ckR3]owned-g2 in `_routed_local`:
* FIB md5 8765a2cd != 0c468f16 (eng2 errs 377→630, want 610) ⇒ non-det CONFIRMED.
* **[ckR0] routing idx_gsum/w_gsum/x_full BYTE-IDENTICAL ×2** (decode 1.76530e4, prefill 1.024635e6) +
  moe_input + moe_shared identical ⇒ **GATE/ROUTING DETERMINISTIC** (Agent gate-sharding hypothesis REFUTED).
* Only **moe_routed diverges** (85.72/abs1.424 vs 60.05/abs1.300).
* **[ckR2] g1own: gsum PRESERVED ~7 figs, ABSMAX DIVERGES** (r=4 2.22 vs 1.70). g1 INPUT x_sorted is provably
  identical (x_full+routing+argsort all identical) ⇒ the g1 gmm itself injects non-det into a FEW owned rows
  with LARGE values. [ckR3] g2own: gsum+absmax both diverge big (g1 contamination amplified thru silu).
⇒ corruptor is **gmm_v2 reading uninit-HBM into VISITED/owned output rows**, first in g1. Signature
"bulk deterministic + few large non-det rows" = the owned m-block BOUNDARY (partial sublane tile).

## zero_initialize=False — TESTED & DISPROVEN (HEAD keeps it: reference-aligned + correct, but NOT the fix)
HEAD has BOTH routed gmm_v2 calls (deepseek_v4_moe.py:300,318) at zero_initialize=**False** (matches qwen3/v3;
non-owned handled by the token-space owned-mask :331). 2 fresh engines (logs 170733Z + 171953Z):
* eng1 md5=49a2e06e moe_routed=30.07/abs1.485 ; eng2 md5=e63871d0 moe_routed=27.47/abs1.364 ⇒ **STILL NON-DET**.
  (moe_input 247.14 + moe_shared 70.88 identical, as always.) Output stays COHERENT/correct (eng2: 610,987,1597,2584).
⇒ owned-row non-det is **INDEPENDENT of zero_initialize** (not the zero-DMA). KEPT False (reference-aligned, safe,
  output correct), but the boundary-row corruptor persists. Magnitude dropped (True 85/60 → False 30/27).

## NEXT ACTION — the owned-row corruptor is in the gmm compute; all proven-clean INPUTS except WEIGHTS untested
Prioritized (cheapest-decisive first):
1. **[ckW] WEIGHT checksum across 2 engines** — the ONE gmm input never directly checksummed. Add layer-0
   per-rank prints of W13_l + W2g_l gsum/absmax in `_routed_local` (like [ckR2]). If they DIFFER ×2 ⇒ the
   EXPERT-WEIGHT LOADING/sharding has per-process uninit (loader bug, NOT the kernel) ⇒ refocus on
   deepseek_v4_loader.py expert-weight placement. If IDENTICAL ⇒ confirms bug is gmm-internal compute.
2. **Dense-einsum vs gmm A/B** — temporarily replace the two gmm_v2 calls with a masked dense einsum over THIS
   rank's owned experts (owned rows only). 2 engines: if DETERMINISTIC ⇒ bug is gmm-kernel-internal (→ lead 3);
   if STILL non-det ⇒ bug is in the dispatch/all_gather/combine COMMON to both einsum+gmm (history: S18 einsum
   was ALSO corruptor — this would be the big reframe: NOT the matmul).
3. **Pad each rank's owned m-block to a size_lhs_sublane(=8) multiple** — eliminate the partial boundary tile
   (Agent C: the structural diff from qwen3/v3 may be ALIGNMENT, not zero_init/mask). Pad group_sizes/dispatch.
GATE for any fix = moe_routed byte-identical ×2 AND FIB md5 ×2 AND correct Fibonacci; then CLAUDE.md formal gate
×2 + `touch /tmp/s1_loop_stop`. Grep: `bash /tmp/s1_ck.sh <log>`.

## Ops
* ONE engine. Warm cache ⇒ ~5 min ready; FIRST req recompiles ~336s. `python /tmp/s1_warmup.py`(Fib, 40 tok)
  absorbs it + emits diagnostics. Grep new checksums: `bash /tmp/s1_ck.sh <smoke-log>`. Compare moe_routed +
  FIB md5 ACROSS 2 fresh engines (single-engine re-runs only prove within-process det). ssh -i ~/.ssh/google_compute_engine.
* edit→`full_slice_v4_sync.sh`(md5 8/8)→cache: KEEP if .py unchanged / CLEAR if changed→`reset.sh`→`smoke.sh`. Guardians up.
* HYGIENE: ~6 stale parked claude sessions in tmux (handoff-window spawns); harmless, kill old windows.
* Diagnostics IN (cleanup AFTER fix): [ckR0/2/3] in _routed_local (~273,300,318), [ckR]180-182, [ckS]230-233,
  [ckL]~334, [ckD] dsv4 2016-21. Keep token-space owned-mask + nan_to_num.

## DEAD (do not retry): zero_initialize True↔False toggle (both non-det — owned-row bug is independent of it);
partial_out_ref uninit (gm_id==0 guard is per-rank-LOCAL, written-before-read — CLEAN);
gmm_v2:518 tiled_out_ref[last_row] OOB (discarded by the jnp.where exactly when OOB — RED HERRING); GATE/ROUTING
sharding non-det (routing byte-identical ×2 — REFUTED, staged gate-replicate fix is moot); acc_ref (clean);
S22 owned-mask ALONE (necessary, kept, insufficient); attention SEED; psum/all_gather/x_full/weights pad-masking;
wsc(act,P()); prefill-replicate decode meta. Anchors: /tmp/s1_eng1_s23.txt, logs 162729Z + 164055Z + 170733Z.

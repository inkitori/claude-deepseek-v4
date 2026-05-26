# S1 handoff — owned-mask removed the BIG non-owned garbage but owned-row residual REMAINS; localize which gmm

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL / how to judge (see [[s1-symptom-nondeterminism-not-collapse]])
Bug is NOT collapse. Symptoms: (1) cross-process non-determinism (FIB md5 differs across fresh engines;
deterministic WITHIN a process), (2) quality drift. DONE = byte-identical md5 across 2 fresh engines AND
correct Fibonacci. Model is INSTRUCT. Validate: warmup absorbs ~336s recompile, then s1_fib_clean ×2 / 2 engines.

## STATE (2026-05-26 S23) — partial_out_ref REFUTED; diagnostic smoke IN FLIGHT to localize root cause
HEAD has S23 diagnostics (commit "SESSION 23"). Two prior leads UPDATED:
* **partial_out_ref hypothesis REFUTED** (Agent trace): buffer always written-before-read; `gm_id==0` guard is
  per-rank-LOCAL (starts at 0 for each rank's owned groups) so ranks r>0 do NOT read uninit. gmm scratch is CLEAN.
* **NEW lead — gate sharding**: V4 gate weight [256,4096] is sharded P(None,'attn_dp') on the CONTRACTION axis
  (deepseek_v4_loader.py pick_partition_spec picks larger divisible dim=4096) ⇒ router matmul scores=x@w.T
  (deepseek_v4_moe.py:83) forces a cross-process all-reduce. **V3 REPLICATES its gate** (deepseek_v3.py:861
  ed_sharding=P(None,None)) — V4 deviates exactly here. Could flip top-k → divergent ROUTING. Symptom fits:
  moe_routed gsum 55 vs 85 (35% systematic shift) + stable absmax = "different experts selected" > "uninit bleed".
* **STAGED FIX (apply ONLY if [ckR0] confirms routing divergence)**: in deepseek_v4_loader.py:484-489, right after
  the `if total < _MIN_SHARD_ELEMENTS: return P()` block, insert: `if len(shape)==2 and shape[0]==256 and shape[1]==4096: return P()`
  — replicates ONLY the gate (unique shape; experts are [2048,4096]/[4096,2048]). Mirrors V3.

IN FLIGHT: diagnostic-only smoke (NO fix) running — log logs/full-slice-v4-smoke-20260526T162729Z.log. Added layer-0
checksums in _routed_local: [ckR0] idx/w/x gsum (routing det), [ckR2] owned-masked g1, [ckR3] owned-masked g2.
Grep with `bash /tmp/s1_ck.sh <log>`. Need eng1 + eng2 (fresh) → first divergence in chain
routing([ckR0]) → g1own([ckR2]) → g2own([ckR3]) → moe_routed([ckR]) → local([ckL]) PINS the cause:
routing⇒apply staged gate fix; g1/g2⇒gmm kernel owned-row uninit (Agent C lead: sublane-boundary partial tile,
m_end_local%size_lhs_sublane); combine/psum⇒downstream. NOTE [ckR2/3] only meaningful if [ckR0] identical (the
owned-mask uses idx_flat, so routing divergence propagates).

## STATE (2026-05-26 S22) — owned-expert mask = PARTIAL fix (kept), residual is in OWNED gmm rows
HEAD **b022ff10**: `_routed_local` (deepseek_v4_moe.py:304-315) now masks g2 rows whose expert∉[r·EP,(r+1)·EP)
to 0 before the weight-combine/psum (the `valid_rows_mask` production has; the code's own NOTE deferred exactly
this). 2-engine gate (full detail: `logs/s1_s22_disproof.txt`):
* eng1 FIB md5=06bfeeb9 (10/12: ...610,987,1597,**2583,4160**) ≠ eng2 md5=0a72aece (7/12, breaks) ⇒ **STILL NON-DET**.
* `[ckR] L0 moe_routed` eng1=**55.23/abs1.423189878** vs eng2=**85.72/abs1.424025536**: rsum DIFFERS big,
  but rabsmax **CONVERGED & nearly-equal** (pre-fix B=61.34/abs1.709 C=68.08/abs1.459 — absmax also differed).
* moe_input/moe_shared/seed chain = byte-identical ×2 (always clean).
⇒ mask correctly removed the LARGE non-owned garbage (absmax converged); the SMALL residual (sum 55→85, absmax
  stable) is in rows that SURVIVE the mask = the **OWNED-expert rows** ⇒ uninit read INSIDE the gmm for VALID
  owned rows. **KEEP the mask** (correct + necessary; production has it). S21's "routed gmm is corruptor" stands.

## NEXT ACTION — localize WHICH gmm produces the nondet owned rows, THEN fix the kernel scratch
1. Add **owned-masked checksums** inside `_routed_local` to split the two gmms (diagnostics only; .py edit ⇒
   sync + CLEAR cache + cold compile): after `x_full` (gmm INPUT — re-confirm deterministic w/ mask in place),
   after g1 (gate/up, masked to owned in SORTED space), after the g2 owned-mask (`token_hidden`). Print
   per-rank rsum/rabsmax (like the existing `[ckL]` at :307). 2 fresh engines → FIRST divergent quantity localizes it.
2. **Leading hypothesis** (Agent audit): gmm **`partial_out_ref`** (cross-SUBLANE carry scratch — gmm_v2.py:1217
   decl, **:501** `tiled_out_ref[0]+=where(gm_id==0,0,partial_out_ref)`, :515-519 write). It is uninit VMEM, NOT
   touched by `zero_initialize`, and flows into VALID owned rows; guard is only `gm_id==0`. SUSPECT iff num_n>1
   (V4 dim large ⇒ yes) AND `gm_id==0` is the GLOBAL gm-index not this rank's LOCAL first group under
   `group_offset=[r·EP]` — then 31/32 ranks read uninit for their first owned group's boundary row. If g1/g2
   owned output diverges, audit gmm_v2.py:495-521 + the gm_id/group_offset semantics; fix = seed partial_out_ref=0
   on this rank's first owned gm-tile (local), or zero that scratch.
3. gmm **acc_ref** matmul accumulator is PROVABLY CLEAN (don't re-audit). Production `fused_moe_func` swap is NOT
   worth it (medium rewrite; same latent uninit+mask). Gate verdict = `[ckR] moe_routed` byte-identical ×2 AND FIB md5 ×2.

## Ops
* ONE engine. Warm cache ⇒ ~5 min ready; FIRST req recompiles ~336s. `python /tmp/s1_warmup.py`(Fib) absorbs it.
  Verify both engines in one shot: `bash /tmp/s1_verify.sh <lbl> <smoke-log>` (warmup + s1_fib_clean ×2 +
  greps [ckR]/[ckS] L0). Compare [ckR] L0 moe_routed (real rows) NOT [ckS] (per-forward/decode noise).
  ssh -i ~/.ssh/google_compute_engine.
* edit→`full_slice_v4_sync.sh`(md5)→cache: KEEP if .py unchanged / CLEAR if changed→`reset.sh`→`smoke.sh`. Guardians up.
* HYGIENE: ~5 stale parked claude sessions in tmux accumulating (handoff-window spawns); harmless, kill old windows.
* Diagnostics IN (cleanup AFTER fix): [ckR]180-182, [ckS]230-233, [ckL]307-309, [ckD] dsv4 2016-21. Keep row-mask+nan_to_num.

## DEAD (do not retry): S22 owned-mask ALONE (necessary, insufficient — owned-row residual remains); gmm acc_ref
(provably clean); gmm_v2+zero_init alone; production fused_moe_func swap; attention SEED; psum/all_gather/x_full/weights
/y≥n_real pad-masking; wsc(act,P()); prefill-replicate decode meta. Anchors: logs/s1_s22_disproof.txt, s1_eng1_s22fix.txt.

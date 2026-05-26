# S1 handoff — corruptor LOCALIZED to the gmm ROUTED path; seed/input exonerated; next = fix gmm zero_init

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL / how to judge (see [[s1-symptom-nondeterminism-not-collapse]])
Bug is NOT collapse. Symptoms: (1) cross-process non-determinism (FIB md5 differs across fresh engines;
deterministic WITHIN a process), (2) quality drift. DONE = byte-identical md5 across 2 fresh engines AND
correct Fibonacci. Model is INSTRUCT. Validate: warmup absorbs ~336s recompile, then s1_fib_clean ×2 / 2 engines.

## STATE (2026-05-26 S21) — DECISIVE: corruptor = the gmm ROUTED path; input+shared+seed all CLEAN
3 fresh same-code engines (commit 21063d80=HEAD .py) ALL differ: A19=d99ee354, B=39c33b59, C=7488bc96.
Engine C (host .202, first prefill, Fib prompt) vs B anchor, REAL-ROWS [ckR] L0:
* `moe_input` 247.14/abs3.234 = **IDENTICAL** • `moe_shared` 70.88/abs3.713 = **IDENTICAL** •
  `moe_routed` C 68.078/abs1.459 vs B 61.344/abs1.709 = **DIFFERS**. seed_x_in/seed_kv*/blk_attn_out all IDENTICAL.
* abs**max** differs ⇒ genuine element-wise garbage in REAL rows, NOT fp reorder ⇒ uninit-HBM in routed path.
⇒ Corruptor is the **routed-expert gmm_v2 path** (moe_input/shared/seed clean). S20's "upstream/attention-seed"
  lead is REFUTED. S19's `zero_initialize=True` is INSUFFICIENT on TPU. (Re-confirms S18: routed path is it.)

## NEXT ACTION — fix the gmm routed non-determinism; verify with C-vs-new-engine moe_routed match
Audit gave 2 candidates (try cheap first, ONE smoke each, keep cache unless kernel edit):
1. **stable argsort** `deepseek_v4_moe.py:277,281` add `stable=True` (cheap; LOW odds — revert is a permutation
   so no ties; but free). 2. **zero whole gmm out** `kernels/megablox/gmm_v2.py` zero_out_start only zeros
   edges → empty-expert-group rows uninit; force `out_ref[...] = 0`. 3. simplest: **mask routed y to real rows
   (<n_real)** BEFORE the un-sort/psum in moe_forward. Gate verdict = C's moe_routed md5 stays IDENTICAL across
   2 fresh engines AND FIB md5 matches ×2. Anchor: logs/s1_engB_gmm_ckR_anchor.txt + logs/s1_engC_localization.txt.

## Ops
* ONE engine. Warm cache ⇒ ~5 min ready; FIRST req recompiles ~336s. `/tmp/s1_warmup.py`(Fib) absorbs it, then
  `/tmp/s1_fib_clean.py <lbl>`. Compare [ckR] L0 (real rows) NOT [ckS] (unmasked=padding noise). ssh -i ~/.ssh/google_compute_engine.
* edit→`full_slice_v4_sync.sh`(md5)→cache: KEEP if .py unchanged / CLEAR if changed→`reset.sh`→`smoke.sh`. Guardians up.
* HYGIENE: 4 stale parked claude sessions in tmux (122150/130123/141328/144348Z) — harmless but accumulating; kill old windows.
* Diagnostics IN (cleanup AFTER fix): [ckR]180-182,[ckS]230-233,[ckL]308-309,[ckD] dsv4 2016-21. Keep row-mask+nan_to_num.

## DEAD (do not retry): gmm_v2+zero_init alone; attention SEED (clean this session); psum/all_gather/x_full/weights/all pad-masking; wsc(act,P()); prefill-replicate decode meta.

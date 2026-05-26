# S1 handoff — gmm fix DISPROVEN; non-determinism is in the PREFILL forward; next = 2-engine [ckR] localization

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL / how to judge a fix (see memory [[s1-symptom-nondeterminism-not-collapse]])
Bug is **NOT a collapse**. The model emits coherent-LOOKING decode even WITH the bug — coherence ≠ fix. Two
symptoms: (1) **cross-process non-determinism** (FIB md5 differs across 2 fresh engines; deterministic WITHIN a
process), (2) **quality degradation** (FIB terms wrong). DONE = BOTH gone: byte-identical md5 across 2 fresh
engines AND correct Fibonacci vs a prefill reference. Model is INSTRUCT (coherence via /v1/chat system+user).

## STATE (2026-05-26 S20) — gmm fix (21063d80) DISPROVEN; bug is in PREFILL
S19's gmm_v2 + `zero_initialize=True` MoE port did **NOT** fix cross-process non-determinism. Proven on 2 fresh
SAME-CODE engines (moe `deepseek_v4_moe.py` md5 **1814bc99** identical on all 8 hosts — NO desync):
* Engine **A19** (S19): FIB md5 **d99ee354**, text `21, 34, 55, 89, 144, 233, 377, 570…` (NUMBERS), 7/12 correct.
* Engine **B** (S20): FIB md5 **39c33b59**, text `' ... \n#  The Fibonacci sequence begins with the numbers 1 and 1…'`
  (PROSE), **0/12** correct. Within-B deterministic (warmup + fib_clean×2 all = 39c33b59; logprobs artifact ruled out).
* **⇒ 2 fresh engines DIFFER ⇒ symptom #1 PERSISTS.** gmm `zero_init` is insufficient, OR the corruptor is not the MoE.

**NEW localization — the bug is in the PREFILL forward, NOT decode-specific.** B's pure-prefill first token
(`s1_prefill_gen` Phase-1 step0, max_tokens=1) = `' ...'`, identical to B's decode first token; A19's was `21`.
The first generated token IS the prefill-forward argmax → **prefill argmax differs across same-code engines.**
(So the runbook's "first decode token is correct (prefill argmax)" is FALSE for B. Stop framing S1 as decode-only.)

## NEXT ACTION — 2-engine `[ckR] L0` localization (decisive: upstream-of-MoE vs in-MoE)
Engine B is engine #1; anchor saved → `logs/s1_engB_gmm_ckR_anchor.txt`:
  `[ckR] L0 moe_input rsum=2.471417236e+02 | moe_routed=6.134395218e+01 | moe_shared=7.088032532e+01`
Spin ONE fresh SAME-CODE engine (commit 21063d80 — do NOT edit code first), capture its FIRST-forward (prefill)
`[ckR] L0 moe_*` from the smoke log (grep `\[ckR\] L0`), compare to the anchor:
* **moe_input DIFFERS** ⇒ corruptor is UPSTREAM of the MoE (embedding / attention / residual / hyper-connect).
  ⇒ S18's "the MoE expert einsum is the corruptor" was a MISATTRIBUTION. Re-open the search upstream — the
  heavily-logged `[ckS]` seed_kv_cache / seed_c_kv / seed_kv_postlinear checksums make the **attention seed** a
  prime suspect. (Confirm by comparing those `[ckS]` values across the 2 engines too.)
* moe_input IDENTICAL, **moe_routed DIFFERS** ⇒ gmm `zero_initialize` does NOT zero all read rows on TPU ⇒
  dig into gmm_v2 zero_init on TPU (group_offset/tiling) — reverting to einsum won't help (both non-det).
Rigor: the anchor is valid only if code is UNCHANGED (md5-verify hosts first); otherwise spin 2 fresh engines
back-to-back in this session and compare directly.

## Tools / ops
* ONE engine at a time. Warm cache ⇒ ~3 min to ready; the FIRST request then recompiles ~336s — run
  `/tmp/s1_warmup.py` to absorb it, THEN `/tmp/s1_fib_clean.py <label>` (FIB×2 md5, no logprobs, 420s timeout).
  `/tmp/s1_prefill_gen.py "PROMPT" N` = pure-prefill ref (Phase 1) + decode contrast (Phase 2, slow — skip/kill it).
* Slice protocol (CLAUDE.md): edit → `full_slice_v4_sync.sh` (md5-verify; **ssh needs `-i ~/.ssh/google_compute_engine`**)
  → KEEP xla_cache if code unchanged (warm ⇒ ~3 min) / CLEAR it if code changed → `full_slice_v4_reset.sh` → `full_slice_v4_smoke.sh`.
* Guardians up (node 497956, meta 4039835). Hard EngineCore crash dropping a raylet (`<32 TPU` after reset) → `full_slice_v4_ray_restart.sh`.
* Diagnostics still IN (the DONE cleanup — do only AFTER a real fix): SAFE-delete `[ckR]`(moe 180-182),
  `[ckS]`(230-233), `[ckL]`(308-309, INSIDE shard_map = the ~2-3 tok/s slowness), `_ckR` calls (335, 342),
  `[ckD]`(deepseek_v4.py 2016-2021). KEEP row-mask (moe 317-323, real logic) + nan_to_num (deepseek_v4.py 2028, scaffold).

## DEAD (do not retry)
* **gmm_v2 + zero_initialize as THE determinism fix — DISPROVEN this session** (2 same-code engines still differ).
* psum / all_gather / x_full-buffer / weights / ALL pad-row masking / optimization_barrier — exonerated (S17-18).
* `wsc(act, P())` gathering a size-1 / idle axis (Core-halts). prefill-replicate of decode metadata (NaN).
* (NOTE: "decode/seed hunt is clean" is NO LONGER safe — the bug is in prefill, and the attention seed is now a suspect.)

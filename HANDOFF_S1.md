# S1 handoff — gmm KERNEL non-det given IDENTICAL input (dtype refuted); suspect = partial-tile VMEM carry

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL / how to judge (see [[s1-symptom-nondeterminism-not-collapse]])
Bug is NOT collapse. Symptoms: (1) cross-process non-determinism (FIB md5 differs across fresh engines;
deterministic WITHIN a process), (2) quality drift. DONE = byte-identical md5 across 2 fresh engines AND
correct Fibonacci. Model is INSTRUCT. Correct FIB after "13, " = 21,34,55,89,144,233,377,610,987,1597,2584,4181,6765.

## STATE (2026-05-26 S24) — bf16 dtype REFUTED; gmm kernel is non-det given byte-identical input
Tested bf16 gmm lhs/rhs (was fp32) on 2 fresh engines (logs 174930Z=eng1, 180255Z=eng2; snapshots
/tmp/s1_eng1_s24.txt + /tmp/s1_eng2_s24.txt):
* FIB md5 2a3ffbc3 != 4d346071 ⇒ STILL NON-DET. eng1 derails 377→521; eng2 correct to 987 then 1598≈1597.
* **REORDER-IMMUNE aggregates ([ckR], replicated/summed → label-scramble-proof):** moe_input rsum 2.471417236e2
  IDENTICAL ×2; moe_shared 7.088032532e1 IDENTICAL ×2; **moe_routed 7.114e1 vs 2.815e1 DIVERGES**. [ckR0]
  routing(idx/w) + x_full BYTE-IDENTICAL ×2 (both prefill+decode regimes).
* x_full identical + x_sorted = x_full[deterministic argsort] ⇒ gmm INPUT identical ×2 ⇒ **the gmm kernel
  computes this rank's OWNED rows NON-DETERMINISTICALLY across processes from IDENTICAL inputs.**
⇒ dtype (fp32↔bf16) REFUTED; combined with zero_initialize refuted (zeros only NON-visited OUTPUT-HBM rows,
  not visited rows nor VMEM scratch) ⇒ corruptor is the kernel's **partial_out_ref VMEM carry** read for
  OWNED PARTIAL-TILE BOUNDARY rows (only place the carry is consumed). bf16 KEPT (prod-aligned; see below).

## Why V4 and not qwen3/v3 (same gmm_v2 kernel, deterministic there)
Two V4-unique deltas vs prod fused_moe_gmm on TP=32 (which uses tensor_parallel_gmm, group_offset=0, ALL
groups visited, bf16): (1) fp32 lhs/rhs — REFUTED this session; (2) **nonzero group_offset=[r*EP] SUBSET call**
(8 of 256 groups visited, 248 unvisited interleaved). Agent traced gm_id is RELATIVE to first visited group so
the gm_id==0 carry-zero FIRES — yet owned rows still diverge ⇒ the partial-tile carry mechanism itself, on
this rank's owned group BOUNDARIES (non-sublane-aligned group_sizes), is the open suspect.

## NEXT ACTION — eliminate the partial-tile carry (cheapest-decisive first)
1. **SUBLANE-PAD owned group_sizes** to a multiple of size_lhs_sublane (get_sublane_tiling(lhs.dtype),
   gmm_v2:951; =8 for bf16) so NO partial sublane tile exists at any owned group boundary → the carry row is
   never read. Pad x_sorted rows + group_sizes accordingly (route pad rows to a masked dummy slot; they get
   discarded by the existing owned-mask :340). 2 fresh engines: moe_routed byte-identical ×2 ⇒ carry CONFIRMED.
2. **group_offset=0 compaction** (agent sketch, see commit): compact x_sorted to ONLY owned rows, length-EP
   group_sizes, group_offset=[0], scatter back. NB likely SAME visited-group tiling as the subset call ⇒ may
   not change behavior; lead 1 is more targeted. Kernel rhs is [EP,...]; validate group_sizes/rhs len match.
3. If both fail ⇒ patch gmm_v2 carry (init partial_out_ref / write-not-+= the carry row) — but it's the SHARED
   kernel (qwen3/v3 use it); guard by dtype/shape so prod path is untouched.
CLEAN MEASUREMENT: fire `max_tokens=2` as the FIRST request on EACH fresh engine (1 decode step → clean
[ckR1b]/[ckR2] per rank; the 40-tok warmup pollutes per-step prints). Then compare [ckR1b] (owned gmm INPUT)
vs [ckR2] (owned g1 OUTPUT): input identical + output diverges = kernel confirmed. GATE = moe_routed md5 ×2 +
FIB md5 ×2 + correct Fibonacci; then CLAUDE.md formal gate ×2 + `touch /tmp/s1_loop_stop`.

## Ops
* ONE engine. Warm xla cache ⇒ ~5 min ready; FIRST req recompiles ~332s. `python /tmp/s1_warmup.py`(FIB 40tok,
  prints md5+text, absorbs recompile, emits [ckR*]). Grep: `bash /tmp/s1_ck.sh <smoke-log>` (updated for
  [ckR1b] xso-input + [ckR2] sqsum). Compare moe_routed + FIB md5 ACROSS 2 fresh engines. ssh -i ~/.ssh/google_compute_engine.
* edit→`full_slice_v4_sync.sh`(verify md5 8/8)→CLEAR xla_cache 8 hosts if .py changed→`reset.sh`→`smoke.sh`.
  node_guardian UP. meta_guardian is BROKEN (GcsClient WrongClusterID, auth-API drift) but cluster is HEALTHY
  and node_guardian is the real defense — don't re-fight it.
* Diagnostics IN (cleanup AFTER fix): [ckR0] routing+xfull ~278, [ckR1b] xso-owned-INPUT + [ckR2] g1own-OUTPUT
  (+sqsum) ~306-318, [ckR3] g2own ~326, [ckR]180-182, [ckS]230-233, [ckL]~346, [ckD] dsv4. Keep owned-mask :341 + nan_to_num.

## DEAD (do not retry): bf16↔fp32 gmm dtype (both non-det — S24); zero_initialize True↔False (zeros only
non-visited OUTPUT rows); partial_out_ref gm_id==0 init (RELATIVE to first visited group — FIRES); gmm_v2:518
OOB (where-discarded); GATE/ROUTING sharding (routing byte-identical ×2); acc_ref; S22 owned-mask ALONE
(necessary, kept, insufficient); attention SEED; psum/all_gather/x_full/weights pad-masking; EXPERT WEIGHTS
uninit (256/32=8 ep/rank, all rows written, no jnp.empty — statically exonerated); wsc(act,P()); prefill-replicate
decode meta. Anchors: /tmp/s1_eng1_s24.txt, /tmp/s1_eng2_s24.txt, logs 174930Z + 180255Z.

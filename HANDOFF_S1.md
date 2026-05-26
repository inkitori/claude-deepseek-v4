# S1 handoff — bug = MoE routed EINSUM reads uninit HBM (psum + all_gather EXONERATED; isolated S18)

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL (user-confirmed) — see memory [[s1-goal-reliable-coherence]]
DONE = **reliably coherent decode on EVERY fresh engine** (bug is a per-process coin flip). RIGOROUS gate:
**byte-identical (md5) FIB across 2 fresh engines AND coherent chat text.** Model is INSTRUCT — coherence
via `/v1/chat/completions` (system+user). Raw `/v1/completions` FIB = wedge-safe md5 determinism probe.

## STATE (2026-05-26 S18) — EINSUM-vs-psum SETTLED: it is the EINSUM
Bug = per-process uninit-HBM read in the per-rank **expert EINSUM** inside `deepseek_v4_moe.py::moe_forward`
`_routed_local` (shard_map over 'attn_dp', lines ~255-294). Settled by running engine B3 (SAME executable as
A3 — git clean, synced) and comparing the `[ckL]` pre-psum local-sum set:
* `[ckG]` x_full (einsum INPUT, post all_gather): A3==B3 byte-identical (2.141315039e+04, 2.142430469e+04,
  2.144123242e+04, 2.471417236e+02; B3 just dropped 1 print) ⇒ einsum INPUT identical across engines.
* `[ckL]` local (einsum OUTPUT, pre-psum): A3 vs B3 share **0 common values** (23 vs 25 distinct); each has a
  lone ~1e5 outlier that DIFFERS — A3 9.417071875e+04 vs B3 8.498864844e+04 ⇒ einsum OUTPUT diverges with
  identical input ⇒ **the einsum injects per-process uninit.**
* `[ckR]` moe_routed REAL-rows (row<n_real, post-psum): B3 9.151400757e+01 ≠ A3 4.582982254e+01 ⇒ REAL rows
  contaminated. psum is per-ELEMENT ⇒ divergent real-row OUTPUT ⟹ divergent real-row einsum output ⟹ einsum
  hits REAL rows (NOT just pad). B3 FIB md5 26d81071 ≠ A3 bb5adb1b (bug reproduces).
  Refs: `logs/s1_engA4_ckL.txt`, `logs/s1_engB3_ckL.txt` (BOTH corrupted — no clean baseline exists yet).
* EXONERATED: psum as SOURCE (divergence is pre-psum); all_gather ([ckG] identical). Config: E=256,
  attn_dp axis=32, EP=8, E%32=0 ⇒ NO idle ranks (empty-tile theory DEAD).
* CLEAN CONTRAST: the dense path (lines 233-248, decode/CPU) runs the SAME einsum via auto-SPMD E-sharding →
  deterministic; moe_shared (dense, no E-shard) → deterministic. Bug is specific to the MANUAL
  all_gather→per-rank-einsum→psum shard_map (auto-SPMD reasons about padding; the manual path doesn't).

## NEXT ACTION — FIX the einsum uninit read
⚠ ALL pad-row / valid-row masking is REFUTED (real rows are hit; per-element psum can't spread pad→real). Don't retry.
⚠ VALIDATION NEEDS 2 FRESH ENGINES: single-engine `[ckR]` is GLOBALLY all-reduced (same on every process), so it
CANNOT see per-process variance. The bug only shows as cross-engine FIB-md5 / [ckR] differences. Budget 2 smokes/fix.

Source NARROWED (S18 agents, no smoke): the cheap input-side fixes are EXHAUSTED —
  * weight-tile sub-128 padding UNLIKELY: V4 dims d=4096 & inter=2048 are exact 128-multiples (verify vs the real
    V4-Flash config; sourced from TINY_CONFIG/cpu_repro). EP=8 experts/rank, all real.
  * x_full buffer is NOT the source: `optimization_barrier(x_full)` (line 261) already forces clean materialization
    and is REFUTED ⇒ the uninit enters in the matmul OUTPUT/accumulator or is XLA-internal to the manual shard_map
    all_gather→einsum→psum collective (auto-SPMD dense path reasons about extents; the manual path doesn't).

1. PRIMARY FIX (recommended) — port the routed path to the proven `fused_moe_gmm` gmm kernel with
   `zero_initialize=True` (gmm_v2.py:1130 `gmm_v2(lhs[N,K], rhs[E,K,D], group_sizes[G], *, zero_initialize=True,...)`;
   the DMA zero-init of unvisited rows is the determinism lever). NOT the big rewrite previously feared: ~60 new
   lines reusing ~95% of `fused_moe_gmm._process_tokens_locally` (top_k dispatch: gate indices/weights ALREADY
   computed @183 → argsort by expert → group_sizes → gather → gmm_v2 → ragged scatter → weight-combine). PRESERVE
   V4 gate (hash/sigmoid routing, sqrtsoftplus), swiglu_limit, per_expert_weight. Drop in place of the manual
   shard_map block (lines ~250-316). (Dense-all-E via gmm is INFEASIBLE — needs real top_k dispatch.)
2. CHEAP LONG-SHOTS first if you want a 1-edit probe (low confidence, but untried): `optimization_barrier` on the
   einsum OUTPUTS (g/u/o or `local`) — the barrier on the INPUT x_full is refuted, OUTPUTS are untried; and/or
   `preferred_element_type=jnp.float32` on the bf16 DOWN-proj einsum (line 281) (g/u @274,275 already fp32 → no-op there).
3. Then: 2-engine md5 gate + coherent chat (READ TEXT) + survives 5 reqs → REMOVE all diagnostics → final smoke →
   record DONE in HANDOFF+CLAUDE → `touch /tmp/s1_loop_stop`.

## Tools / ops
* ONE engine at a time. Probes: `/tmp/s1_warmup.py` (FIB, run FIRST — absorbs ~348s recompile of a changed program),
  `/tmp/s1_fib2.py <label>` (FIB×3 md5; the 3rd req may hit the bash timeout — 2 identical gens suffice),
  `/tmp/s1_chat.py <label> [prompt]` (coherence). Diag goes to the smoke log.
* Extract [ckL]: `grep -hoE 'local_gsum=[-0-9.e+]+' LOG | sed -E 's/local_gsum=//' | sort -u`; compare to
  logs/s1_engA4_ckL.txt + logs/s1_engB3_ckL.txt. [ckR]: `grep -hoE '\[ckR\] L0 moe_routed: rsum=[-0-9.e+]+' LOG`.
  [ckG]: `grep -hoE 'xfull_gsum=[-0-9.e+]+' LOG`. jax.debug.print DROPS lines under load — re-fire / use distinctive values.
* xla_cache WARM ⇒ startup ~5-6 min; a code edit recompiles only the changed program (~350s, absorbed by warmup).
  KEEP cache unless a launch-id halt. After sync, the script md5-verifies head==workers (key -i ~/.ssh/google_compute_engine, user enyouki).
* Slice HEALTHY (reset CLEAN 0/32 S18, no halts). Guardians up. `/tmp/s1_loop_stop` NOT set.
* HOUSEKEEPING: each handoff spawns a NEW tmux window but the OLD session doesn't self-exit → idle claude windows
  accumulate (3 now: 074026Z, 084922Z, me 122150Z). Idle = harmless (not driving slice, not generating tokens).
  `tmux kill-window -t <id>` the stale ones if they grow.

## DEAD (do not retry)
* psum as the corruptor (divergence is PRE-psum, [ckL]). all_gather (x_full byte-identical, [ckG]).
* x_full BUFFER as the uninit source — `optimization_barrier(x_full)` (line 261) materializes it cleanly & is REFUTED.
* weight-tile sub-128 padding — V4 dims (4096, 2048) are 128-aligned, no sub-tile pad. Dense-all-E via gmm (infeasible).
* ALL pad-row/valid-row masking: input (a3982a2b), output (S13), idle-rank — real rows hit, refuted. optimization_barrier(x_full).
* Decode/seed hunt (re-audited clean). [ckD] probe (reads pad row). E%axis expert mask. XLA collective-matmul flag
  (libtpu-rejected). fp32 matmul of whole path (S12 — but `preferred_element_type` specifically is UNtried). prefill-replicate (NaN).
  `wsc(act,P())` gathering empty/idle/size-1 axis (Core-halts ~8x).

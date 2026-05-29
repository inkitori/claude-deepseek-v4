# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign (v6e-16)

> **Phase = PERFORMANCE.** Make `vllm serve DeepSeek-V4-Flash` DECODE + PREFILL FAST on the v6e-16
> slice WITHOUT breaking the correctness GATE. Durable slice ops + pitfalls: `CLAUDE.md`. The FIT
> milestone (256 routed experts kept FP4-compressed; `MAX_SEQS=1`) is DONE and is a GIVEN — history in
> `HANDOFF_QUANT.md`. S1 determinism history: `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29 — P.13): prefill DISPATCH owned-gather lever REFUTED/BLOCKED (scripts-only ⇒ GATE intact).**
> The dispatch microbench had a DCE bug (the lumped `dispatch_local.sum()` folded the order-invariant gather→reduce
> and DCE'd the argsort, mis-attributing cost). Rewrote it (shard_map + array-returns) for the FIRST trustworthy
> decomposition: **real dispatch = 36/40/64/95 ms/fwd @N=512/1024/2048/4096** (sortFwd+invArg+gathFu+revtFu). The SORT
> (sortFwd+invArg, ~25 ms/fwd @N=4096) is UNAVOIDABLE (production sorts identically); the GATHERS (gathFu+revtFu,
> 19→70 ms/fwd) are the only attackable part, **DOMINATED by the fp32 REVERT gather (revtFu 55 ms/fwd @N=4096)**. The
> roadmap-#1 fix (owned-only `ragged_gather`/`ragged_scatter` = 16× less DMA, mirror prod fused_moe_gmm EP) is **BLOCKED
> on v6e**: SparseCore IS LIVE (`SparseCoreInfo(num_cores=2,num_subcores=16,num_lanes=8)` — V4's "ragged_* fall back"
> comment is STALE/wrong) but the kernels FAIL TO COMPILE — the SC pipeline grid is derived from the DYNAMIC `(end-start)`,
> XLA can't prove it tile(8)-aligned, and they lack `tpu.assume_multiple` (only the v7-gated `gather_reduce` has it);
> `fused_moe_gmm` has NO v6e guard (its EP ragged path is v7-targeted). Unblocking needs an upstream Mosaic patch
> (out-of-scope diff on a shared kernel). Also REFUTED: the 2nd-argsort→O(M)-scatter inverse (scatter ≥ argsort).
> P.13 touched only `scripts/perf_microbench_moe_prefill.py` ⇒ GATE trivially intact (P.11 md5 `3069e80b` is the live
> code). NEXT: roadmap #1 tapped — pivot to roadmap #2 (rhs-prep dual-residency HBM smoke) or the contained bf16-gmm-
> output revert-halving (lossy ~27 ms/fwd @N=4096).

---

## GATE — regression bar, NON-NEGOTIABLE (full mechanics in `CLAUDE.md` §CORRECTNESS GATE)
- **correct Fibonacci** `21,34,55,89,144,233,377,610` + **N=2 md5 `3069e80b` byte-identical ×2 fresh
  engines** (`scripts/s1_probe2.py 2`) + `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` rc=0.
- A numerics-changing fix MAY shift the md5 → re-establish a NEW ref + confirm identical ×2 engines +
  correct Fib. Do NOT gate on the long-tail md5 (`s1_probe2.py 100` → `ab07ecbb` is NON-deterministic at
  temp=0 by design). **`MAX_SEQS=1` is PINNED** (concurrent decode S1-broken).
- P.11 (prefill attention-sharding) is the first production-code change since P.5 — GATED on 2 fresh engines,
  md5 still `3069e80b` (LOSSLESS, per-query independence — CONFIRMED, not just predicted), correct Fib, smoke_
  check rc=0. P.6–P.10 changed only `scripts/perf_*`. The bit-identical md5 across this change proves intact.
- P.12/P.13 changed only `scripts/perf_*` (unpack bench / dispatch decomp) — no production `.py` touched ⇒ no smoke
  needed, the P.11 GATED state (md5 `3069e80b`) is the live model code, trivially intact.

---

## ⇒ NEXT ACTION — roadmap #1 is TAPPED (P.13). Pick: roadmap #2 (rhs-prep dual-residency) or bf16-gmm-output
P.13 REFUTED the dispatch owned-gather lever (the 16× DMA win is BLOCKED on v6e — see one-liner + DO-NOT-RETRY #24).
What's LEFT of the dispatch is small/risky (none is the obvious next step): the fp32 revert (revtFu, 55 ms/fwd @N=4096)
could be HALVED by **bf16 gmm output** (gmm `preferred_element_type=bf16`, keep `acc_dtype=fp32`) so g2 + the revert
are bf16 — but LOSSY (md5 shift; decode's DENSE path ALREADY uses bf16 expert outputs, so plausibly fine) and the S24
note flags the gmm fp32-output path as determinism-sensitive (the fp32 `partial_out_ref` carry was the non-det
ORIGINATOR — so bf16 output may be SAFER, but VERIFY on 2 engines). Other dispatch options are worse: full-range SC
gather (no DMA win, likely also fails to compile), segment-sum revert (scatter-add, S1-risky), upstream kernel patch
(out-of-scope diff).

**Recommended NEXT = roadmap #2 (the biggest remaining prize): rhs-prep dual-residency.** Run an HBM-FEASIBILITY smoke
for keeping the experts BOTH fp4-resident (decode) AND fp8-resident (prefill): pre-unpack the FP4 codes → fp8 at LOAD,
store alongside, and have the PREFILL path read the fp8-resident weights (skip the 194 ms/fwd in-trace `_fp4_rhs_and_scale`
unpack) while DECODE keeps reading fp4 (UNCHANGED ⇒ decode-neutral). Prize = **194 ms/fwd, ALL N** (the single biggest
prefill lever). Risk = **+17.1 GiB resident on top of fp4's 8.57 ⇒ ~27 GiB / ~4 free, HBM-MARGINAL** — so the FIRST
step is a feasibility smoke at small MAX_LEN (does it even load+serve?), then measure the prefill-wall delta. ⚠️ a
loader change (`deepseek_v4_loader.py` / `moe_weights.py`) + tension with the FIT foundation. *Smoke-gated; the md5
should be UNCHANGED (fp8 codes are lossless — e2m1⊂e4m3, CPU-confirmed P.9b).*

  Then: #3 (non-MoE launch floor, hard — op-count↓ re-opens S1), #4 (attention token-sharded OUTPUT, blocked on #2's HBM).

---

## THE ROADMAP (re-ranked P.13 — dispatch owned-gather BLOCKED; rhs-prep dual-residency now leads)
1. ~~**[prefill DISPATCH fuse]**~~ **TAPPED (P.13).** Clean decomp: dispatch = 36/40/64/95 ms/fwd @N=512/1024/2048/4096;
   the SORT (~25 ms/fwd) is unavoidable, the GATHERS (19→70) are attackable but the owned-only 16×-DMA win is BLOCKED
   on v6e (DO-NOT-RETRY #24) and the inverse-scatter is REFUTED (#23). Only scrap left = bf16-gmm-output (lossy,
   ~27 ms/fwd @N=4096; see NEXT ACTION) — small + risky, not a clear lever.
2. **[prefill MoE rhs-prep KILL — dual-residency, HBM gamble]** ← TOP / NEXT ACTION. the 194 ms/fwd FP4→fp8 unpack can't be made
   FASTER (P.12: XLA native convert is the VPU floor — DO-NOT-RETRY #22), only ELIMINATED by keeping fp8-resident-
   for-PREFILL alongside fp4-resident-for-DECODE (decode UNCHANGED, so DECODE-NEUTRAL). Prize HUGE (194 ms/fwd,
   all N) but +17.1 GiB resident on top of fp4's 8.57 ⇒ ~27 GiB resident / ~4 free ⇒ HBM-MARGINAL + a loader
   change + tension w/ the FIT foundation. *L · needs an HBM-feasibility smoke FIRST; likely fits only short prefill.*
3. **[prefill non-MoE LAUNCH floor ~117 ms]** PROJ 42 + NORM 31 + HC 17–23 + GATE 11 + CMP 10 + IDX 5, seq-
   INDEP, launch-bound at tiny per-chip n. Lossless cut = op-count↓ (layer scan / fuse the ~215 per-fwd
   matmuls) but that re-opens S1 (DO-NOT-RETRY #10). *M · hard.*
4. **[prefill attention — token-sharded OUTPUT, the P.11 DEFERRED extra]** leave o token-sharded → shard the
   o-proj too. BLOCKED on #2 (needs the rhs-prep HBM peak down first — DO-NOT-RETRY #21). *S once #2 lands.*
5. **[fp8-resident FOR ALL — REFUTED for normal serving]** P.9b: net-negative (decode +39 ms/step, break-even
   ~6 gen tokens). NOTE: roadmap #2 is the DIFFERENT decode-neutral variant (fp4 RETAINED for decode). *L · parked.*
6. **[decode clean profiler / 5-cleanup]** confirmatory; + Phase-5 diff-shrink: remove `_v4_nan_tripwire`
   (37 sites + def + `smoke.sh:81/116`), edit `.py`+`.sh` TOGETHER (Pitfall #0), KEEP `_linear` clamp +
   `compute_logits` nan_to_num. *S.*

---

## DECODE — CLOSED at its ~lossless floor (P.1–P.8; do NOT re-chase without a new idea)
Decode wall **146 ms/step** (277→220→146; V4_DECODE_TIMERS). device_wait 138.9 = MoE **105.8** (P.6 LOSSLESS
floor, 2.46 ms/layer ×43) + attn 0.6 (P.7) + proj/gate/hc **7.87** (P.8) + **~24 ms per-op LAUNCH overhead**
(N=1 is launch-bound; ~1760 ops/step). The only lossless lever for the ~24 ms is op-count↓ (a layer
`lax.scan`), which re-opens S1 (DO-NOT-RETRY #10). ⇒ a large lossless DECODE win likely doesn't exist; the
prefill pivot (above) is where the EV is. Decode DO-NOT-RETRY items #1,#10–18 below remain in force.

---

## DO-NOT-RETRY (dead ends — do NOT burn a smoke; ★ = P.9)
1. **gmm_v2 for the DENSE N=1 DECODE path — REFUTED (P.5, 0.78×).** gmm's per-group machinery at 1 row/group
   costs more than the einsum. (Prefill is different — gmm at M≫1 rows IS the right path, near floor, P.9 #★.)
2. **The bf16-dequant materialization was the DECODE lever, NOT the matmul** (P.5: dequant 6.15 vs einsum 0.79/layer).
3. **nnx-preflatten — DONE (P.4).** Host-dispatch 56→8 ms/step.
4. **Decode "48% copy/transpose" — that was PREFILL** (the `_routed_local` sort+gathers; CONFIRMED P.9 = dispatch 80 ms/fwd at N=4096). Decode emits none.
5. **Async scheduling** — DISABLED (RayDistributedExecutor forces `async_scheduling=False`).
6. **Collective fusion / `pick_partition_spec` axis flip** — ≤1% on the 4×4 ICI torus.
7. **In-trace FP4→BF16 dequant on the PREFILL/sharded path — `CompileTimeHbmOom`** (Q.11). (fp8 codes is half the bytes and is what the prefill path ALREADY does — #★ below leverages that.)
8. **Native typed `float4_e2m1fn` rhs to a kernel** — `MosaicError` on v6e (needs v7); fp8 codes is the v6e floor.
9. **`MAX_SEQS>1` concurrent decode** — CONFIRMED BROKEN (Q.15). Hard-pinned `=1`.
10. **Un-replicate/reshard the decode activation; `lax.scan` over layers; remove anchor buffers** — all re-open S1 / Pitfall #5.
11. **Remove the `_linear |r|<1e8` clamp or `compute_logits nan_to_num`** — both load-bearing (S1 + Q.15).
12. **A hand-written in-trace DECODE kernel to beat 2.46/layer MoE — REFUTED (P.6).** Naive dense Pallas matvec 11.6 (VECTOR-bound); gmm 5.48. Both lose to XLA materialize+matmul (2.46).
13. **fp8 codes FOR BANDWIDTH at N=1 DECODE — REFUTED (P.6).** einsum-fp8w 0.75 ≈ einsum bf16 0.78: the N=1 matmul is latency/MXU-fill-bound, not byte-bound. (fp8-resident's value is skipping the in-trace dequant/unpack, NOT reading faster — see #★.)
14. **bf16-resident (pre-materialized) experts — does NOT FIT (P.6).** 34.3 GiB > 31.25 budget. (fp8 codes = 17.1 GiB DOES fit — #★.)
15. **Swapping the Mosaic `sparse_attn_kernel` for pure-JAX on decode — REFUTED (P.7).** Kernel 5× faster, launch-bound, parity max|Δ|=0.
16. **The CSA indexer `lax.top_k`/score einsum as a decode lever — REFUTED (P.7).** ~0.27 ms/step; negligible at MAX_LEN=4096.
17. **`compute_logits` head_w vocab sharding — ALREADY column-sharded 16-way (P.7).** No replicated-2.1GB lever.
18. **Q/KV/O projections + MoE gate + HC-sinkhorn are NOT the decode ~30 ms — REFUTED (P.8).** proj 4.13 + gate 0.75 + hc 2.99 = 7.87; the balance is ~24 ms launch overhead.
19. ★ **prefill gmm-CORE as a lever — REFUTED (P.9).** gmm is 0.82–1.24× the dense fp8 MXU floor and only 23–38 ms/fwd; it's MXU-underutilized at 96–1536 rows/rank but near its practical floor. The prefill MoE cost is the rhs-prep (the FP4→fp8 UNPACK), NOT the matmul.
20. ★ **fp8-codes-resident experts FOR ALL (prefill+decode) — REFUTED as a net win (P.9b).** Removes prefill's 225 ms/fwd rhs-prep but the decode dense path then dequants from fp8 (2× expert HBM read) → **decode +39 ms/step (106→145, 1.37×)**; break-even ~6 gen tokens ⇒ net-negative for normal serving (G≫6). Use a DECODE-NEUTRAL prefill cut instead (roadmap #2). fp8 IS lossless (CPU-confirmed); only the economics fail.
21. **prefill attention TOKEN-SHARDED OUTPUT (leave o sharded, shard the o-proj) — CompileTimeHbmOom +1.25G (P.11).** The sharded-kernel COMPUTE win is fine (LANDED, output GATHERED to replicated), but propagating token-sharding past o into the FFN/MoE region let XLA keep ~20 concurrent rhs-prep weight-unpacks live (20× `f8e4m3fn[16,4096,4096]` ≈ 5 GB) → Used 32.50/31.25 GiB. NOT lossy, NOT wrong — purely an HBM-peak/scheduling tip. DEFERRED behind roadmap #2 (rhs-prep peak↓); revisit once the peak drops. (At MAX_LEN=256 with the gathered-output variant, baseline fits — the 6 prior FIT smokes confirm.)
22. **A faster JAX-level FP4→fp8 unpack to beat the in-trace `u8_unpack_e2m1(w).astype(fp8)` — REFUTED (P.12).** XLA's NATIVE `float4_e2m1fn.astype(fp8)` is at the VPU floor (4.5 ms/layer ×43 = 194 ms/fwd). 3 integer bit-math formulations that SKIP the float4_e2m1fn dtype (int32 / uint8 / closed-form, all bit-identical to prod on CPU+device) LOSE at 0.83–0.89×; a 16-entry fp8-LUT GATHER is catastrophic (756× — TPU small-table gather lowers pathologically). Bench: `perf_microbench_fp4_unpack.py`. ⇒ the rhs-prep TIME is removable only via dual-residency (roadmap #2), not a rewrite. (A Pallas integer-unpack kernel would reuse the SAME arithmetic that already lost — not worth it; in-kernel native f4 unpack is v7-only, DO-NOT-RETRY #8.)
23. **2nd-argsort → O(M)-scatter inverse-permutation (`revert_idx = zeros.at[argsort_idx].set(arange)`) — REFUTED (P.13).**
   The scatter (`invScat` 0.19–0.28 ms/layer) is ≥ the 2nd `jnp.argsort` (`invArg` ~0.18, ~constant); both are bit-
   identical inverse permutations but the scatter LOSES on v6e. The revert-index computation is NOT a dispatch lever.
24. **Owned-only SparseCore `ragged_gather`/`ragged_scatter` dispatch (the 16×-DMA win, mirror prod fused_moe_gmm EP) —
   BLOCKED on v6e (P.13).** SparseCore IS LIVE (`SparseCoreInfo(num_cores=2,num_subcores=16,num_lanes=8)` — V4's
   `_routed_local:342` "ragged_* fall back to plain gather" comment is STALE/WRONG: they'd CRASH, not fall back). The
   kernels FAIL TO COMPILE inside a shard_map with a DYNAMIC `[start,end)` range: the SC pipeline grid (`num_blocks` /
   `row_tile_start`) derives from the traced `(end-start)`, and XLA can't statically prove tile(8)-alignment ("Slice
   sizes along tiled dimensions must be aligned to tiles ... use tpu.assume_multiple"). `ragged_gather`/`ragged_scatter`
   have NO `assume_multiple` (only the v7-gated `sc_gather_reduce` in `gather_reduce.py` does), and `fused_moe_gmm.py`
   has NO v6e guard on its dynamic-offset ragged calls (EP path is v7-targeted). Unblocking needs an upstream Mosaic
   patch — out-of-scope diff on a shared kernel. (The owned-gather is provably BIT-IDENTICAL — red-team confirmed gmm
   reads only the owned absolute `[cumsum[r·EP], cumsum[(r+1)·EP])` rows, the matmul is row-independent, and the
   `_owned` jnp.where forces non-owned to 0 — so it's purely a COMPILE-TIME block, not a correctness issue. Revisit if
   libtpu/the kernel gains v6e dynamic-range support.)

---

## VERIFIED FACTS / cheap tiers (don't re-derive)
- ★ **`perf_microbench_moe_prefill.py` (16-chip EP=16, real dims, balanced routing; P.9) — min ms/layer:**
  RHS-PREP (`_fp4_rhs_and_scale`×3 + concat + layout_constraint, the :351-358 in-trace prep) **5.24**,
  SEQ-INDEPENDENT (decomp: unpack-only 5.35, +swapaxes 5.74, +concat/layout 5.24 ⇒ the cost IS the bit-
  unpack). gmm-core 0.53/0.59/0.69/0.87 at N=512/1024/2048/4096 (0.82–1.24× the dense-fp8 floor); dispatch
  0.29/0.43/1.07/1.87 (⚠️ DCE-BUGGY — the lumped `disp.sum()` folded the order-invariant gather→reduce;
  SUPERSEDED by the P.13 decomposition below); collective 0.33/0.45/0.69/1.17. ×43 ⇒ prefill MoE
  **275/288/331/393 ms** (MoE total still ~right; rhs-prep dominates); rhs-prep
  alone = **225 ms/forward** (the lossless lever). Run: `full_slice_v4_sync.sh` then `MH_TIMEOUT=900
  scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_moe_prefill.py --distributed`. ⚠️ gmm_v2 is a
  Mosaic kernel ⇒ MUST be wrapped in `shard_map` (inputs replicated P() ⇒ per-rank cost). Caveat: rhs-prep
  is a standalone UPPER bound (XLA may overlap layer L+1's VPU unpack with layer L's MXU); the load-time
  lever removes it regardless of overlap.
- ★ **`perf_microbench_moe_prefill.py` DISPATCH decomposition (P.13, shard_map + ARRAY-returns ⇒ no DCE) — min
  ms/layer @N=512/1024/2048/4096:** sortFwd (argsort+arange.repeat+token gather) 0.21/0.24/0.31/0.41; invArg (2nd
  argsort = revert_idx) ~0.18 (≈const); gathFu (full bf16 lhs gather) 0.23/0.27/0.27/0.36; revtFu (full **fp32**
  revert gather) 0.22/0.24/0.73/**1.27**. ⇒ real dispatch = sortFwd+invArg+gathFu+revtFu = **36/40/64/95 ms/fwd** (×43);
  UNAVOIDABLE sort floor (sortFwd+invArg) = 17/18/21/25 ms/fwd; attackable GATHERS = 19/19/43/**70 ms/fwd**, dominated
  by the fp32 revert. invScat (scatter inverse) 0.19/0.20/0.22/0.28 ≥ invArg ⇒ NO (DO-NOT-RETRY #23). gathOwn/revtOwn
  (owned-only SparseCore ragged) = **compile-FAIL on v6e** (DO-NOT-RETRY #24). SparseCore probe:
  `SparseCoreInfo(num_cores=2,num_subcores=16,num_lanes=8)` (LIVE). ★ ANTI-DCE: time gathers returning the ARRAY, never
  `.sum()` (sum-of-permutation folds away the gather+sort).
- ★ **`perf_microbench_fp4_unpack.py` (P.12) — FP4→fp8 unpack candidates @real dims, 16-chip EP=16:** prod (XLA
  native `bitcast→float4_e2m1fn→.astype(fp8)`) **4.5 ms/layer** (×43 = 194 ms/fwd); intarith int32 0.89×, uint8
  0.88×, closed-form 0.83×, 16-entry fp8-LUT gather **756× SLOWER** — ALL bit-identical to prod (CPU + on-device).
  ⇒ the unpack is VPU-floored (DO-NOT-RETRY #22); rewriting it can't win. Run: `full_slice_v4_sync.sh` then
  `MH_TIMEOUT=900 scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_fp4_unpack.py --distributed`.
- ★ **`perf_microbench_prefill_nonmoe.py` (16-chip; P.10) — prefill NON-MoE per-chip ms/forward:** dense
  ops seq-parallel at n=N/16 (token-sharded activation per `tpu_runner.py:1431 P(ATTN_DATA)`, replicated
  weights); ATTENTION is the Mosaic kernel REPLICATED at M=N (every chip runs the full seq — the prefill
  all-gather-to-replicated path, `deepseek_v4_attention.py:219-245`). Seq-INDEP floor ≈ **117 ms**: PROJ 42
  (5 matmuls ×43) + NORM ~31 (×4×43, UPPER bd) + HC 17–23 (×2×43, the 19-iter sinkhorn) + GATE 11 + CMP 10
  (×41) + IDX 5 (×21) + LOG 0.3 (col-sharded head, last-token). ATTN (×2 SWA + 21 CSA + 20 HCA): 33/69/197/
  **584** ms at N=512/1024/2048/4096 (CSA dominates). nonMoE 151/189/318/711 vs MoE 275/288/331/393 ⇒ nonMoE
  share **35/40/49/64%**. LEVER (shard attn over token axis — the sharded Mosaic kernel RAN in the bench):
  attn → 9/12/20/**44** ms ⇒ prefill wall **−5.6/−12.1/−27.3/−48.9%**. Run: `full_slice_v4_sync.sh` then
  `MH_TIMEOUT=900 scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_prefill_nonmoe.py --distributed`.
  CAVEATS: NORM ×4 = upper bd; the lever saving is a LOWER bd (it also removes the large q all-gather).
  → **LANDED P.11** (the kernel-compute win; output GATHERED to replicated to dodge the rhs-prep HBM-OOM —
  DO-NOT-RETRY #21). GATED md5 `3069e80b` unchanged ×2 engines. The o-proj-sharding extra is deferred to #1.
- **PREFILL MoE path = `deepseek_v4_moe.py:331-433` `_routed_local`** (use_shard_map when N≥axis): rhs-prep
  (:351-358) → shard_map[all_gather x → argsort by expert → gather `x_sorted[N·top_k,dim]` → 2× `gmm_v2`
  over this rank's EP=16 experts via `group_offset=[r·EP]`, fp8 codes + per-block fp32 rhs_scale → revert/
  owned-mask/combine/psum]. gmm `lhs[M,dim]@rhs[EP,k,n]`, `group_sizes[E]` balanced summing to lhs rows.
- **Decode per-step split (P.5, V4_DECODE_TIMERS):** device_wait 138.9 / wall 145.6 ms; MoE ~106 (76%).
  `perf_microbench_moe_decode.py` (lean-noWSC 2.46 = production /layer; einsum-fp8w 0.75 = the fp8-read
  floor; ★ P.9b `lean-fp8res` 3.38 = fp8-codes-resident dequant, 1.37× REGRESSION from the 2× read) +
  `perf_microbench_attn_decode.py` (`--block`) are the decode tiers. (Full numbers: git P.5–P.8.)
- ★ **fp8-codes losslessness (P.9b, CPU-confirmed):** `u8_unpack_e2m1(w).astype(f8_e4m3).astype(bf16)` ==
  `.astype(bf16)` (max|Δ|=0) — e2m1⊂e4m3. ⚠️ on TPU the same check was 5.2e5 (unresolved; likely the v6e
  float4→f8 cast, DO-NOT-RETRY #8) — resolve before trusting fp8-on-v6e numerics; it doesn't change P.9b timing.
- **HBM:** fp4 experts 8.57 GiB/chip (×43); fp8 codes resident = 17.1 GiB (FITS, ~21 free); bf16-resident
  34.3 (does NOT fit). N=1 decode HBM floor ~5.5 ms/step.
- **CPU torch oracle** `scripts/s1_cpu_repro_v4flash.py both` = Tier-1 math/NaN check (a bit-identical change keeps "OK both match").
- **THE PROFILE re-capture recipe:** profiled smoke (`V4_PROFILER_ARGS=…torch`), `/start_profile` →
  `s1_probe2.py 20` → `/stop_profile`. Parser `scripts/perf_parse_trace.py <trace> --bucket-ops`. Read the
  **2nd+** decode step; discount host `ParseArguments` ~100× (observer effect). The trace ALSO contains a
  prefill region (the prompt) — `perf_parse_trace.py` aggregates the WHOLE trace (no windowing yet; a
  `--decode-step`/`--prefill` window would need ~40 lines, see old fork A / roadmap #5).

# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign (v6e-16)

> **Phase = PERFORMANCE.** Make `vllm serve DeepSeek-V4-Flash` DECODE + PREFILL FAST on the v6e-16
> slice WITHOUT breaking the correctness GATE. Durable slice ops + pitfalls: `CLAUDE.md`. The FIT
> milestone (256 routed experts kept FP4-compressed; `MAX_SEQS=1`) is DONE and is a GIVEN — history in
> `HANDOFF_QUANT.md`. S1 determinism history: `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29 — P.18): prefill o-proj/hc/gate now TOKEN-SHARDED (roadmap #1 LANDED) — GATED ×2 engines, LOSSLESS, decode-neutral.**
> Keep the prefill sparse-attn kernel output TOKEN-SHARDED (`deepseek_v4_attention.py:269`, `P()` → `P(None,ATTN_DATA,None,None)`)
> instead of gathering to replicated. The whole post-attention non-MoE region (o-proj / hc_post / hc_pre / rms_norm / MoE gate)
> contracts only the FEATURE dim, so token-sharding rides straight to the MoE shard_map (in_specs already `P('attn_dp',None)`; its
> all_gather at `deepseek_v4_moe.py:371` is the single re-gather point) ⇒ the non-MoE prefill ops run 1/16-SHARDED instead of
> REPLICATED (the P.10 lever; P.11 landed the kernel-compute half, this extends sharding through o-proj — one fewer all-gather/layer).
> UNBLOCKED by P.17: this was DO-NOT-RETRY #21 (CompileTimeHbmOom +1.25G — XLA hoisted ~20 concurrent rhs-prep unpacks); P.17's
> barrier caps that hoist, so the prefill forward now fits easily: HBM peak **1.51 → 2.95 GiB** (+1.44 from reshard intermediates;
> ~28 GiB headroom). DECODE forward UNCHANGED (**19.62 GiB** — prefill-only change; the decode branch :270 is untouched). GATED:
> md5 `3069e80b` byte-identical ×2 fresh engines + correct Fib `21..610` + smoke_check rc=0 (visible_words=47). ⚠️ CAVEAT: the e2e
> prefill LATENCY win is microbench-PROJECTED (P.10: −5.6..−48.9% @N=512..4096), NOT yet measured e2e — the change adds (pre-existing-
> category) involuntary-remat reshards at the gate/hc boundary (replicate-then-partition; the +1.44 GiB). VERIFY via profiler
> re-capture before claiming the wall-time win. The DECODE forward (19.62 GiB) is now the binding HBM peak (roadmap #2).

---

## GATE — regression bar, NON-NEGOTIABLE (full mechanics in `CLAUDE.md` §CORRECTNESS GATE)
- **correct Fibonacci** `21,34,55,89,144,233,377,610` + **N=2 md5 `3069e80b` byte-identical ×2 fresh
  engines** (`scripts/s1_probe2.py 2`) + `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` rc=0.
- A numerics-changing fix MAY shift the md5 → re-establish a NEW ref + confirm identical ×2 engines +
  correct Fib. Do NOT gate on the long-tail md5 (`s1_probe2.py 100` → `ab07ecbb` is NON-deterministic at
  temp=0 by design). **`MAX_SEQS=1` is PINNED** (concurrent decode S1-broken).
- P.18 added the 1-line prefill o-proj sharding (`deepseek_v4_attention.py:269`, value-identity) + a docstring refresh; GATED on
  **2 fresh engines** (md5 `3069e80b` byte-identical, correct Fib `21..610`, smoke_check rc=0 visible_words=47; prefill HBM 2.95
  GiB / no OOM / decode 19.62 unchanged). Bit-identical md5 across the change ⇒ LOSSLESS.
  (History: P.17 prefill barrier, P.11 attention-sharding, P.5 lean-dequant were prior prod changes, all GATED ×2 engines;
  P.6–P.10/P.12–P.14/P.16 scripts-only; P.15 dual-residency implemented→OOM→fully REVERTED — DO-NOT-RETRY #25.)

---

## ⇒ NEXT ACTION — roadmap #2: decode dense-dequant HBM (microbench FIRST; scripts-only ⇒ GATE-safe to explore)
The DECODE forward is now the binding HBM peak (**19.62 GiB** this run; baseline 11.60 — XLA-scheduler-VARIABLE), from the SAME
unpack-hoisting P.17/P.18 work around for prefill, but on the DENSE decode path's bf16 dequant `_dequant_fp4_experts` (NOT touched
by the prefill barrier — decode takes the dense path, not the shard_map). A decode-side `optimization_barrier` could cap it
LOSSLESSLY, but decode LATENCY is the PRIMARY target ⇒ A/B HBM-vs-latency on a MICROBENCH before prod.

**The scoped change (audited P.18):** dense decode branch of `moe_forward` (`deepseek_v4_moe.py`): the dequant at
**:313-315** (`W1=_dequant_fp4_experts(W1,S1)`, `W2=…`, `W3=…` → bf16) feeds einsums at **:320/:322** (`'nd,eid->nei'` gate/up) +
**:329** (`'nei,edi->ned'` down); incoming activation `flat_x` at **:239** is the decode residual analog. Add the prefill analog
**`W1,W3,W2,flat_x = jax.lax.optimization_barrier((W1,W3,W2,flat_x))`** BEFORE the dequants so XLA can't hoist all 43 layers'
bf16 dequants concurrent. **Microbench:** copy `perf_microbench_moe_prefill_hbm.py` → `perf_microbench_moe_decode_hbm.py`, swap
the prefill `rhs_prep`+`gmm_core` per-layer body for `_dequant_fp4_experts` + the 3 dense einsums at **M=1** (N=1 token), keep the
`compiled.memory_analysis().temp_size` A/B (barrier on/off) + min-latency report. EXPECT: HBM caps to single-digit GiB; latency
cost likely <1% (at M=1 decode is already residual-serialized — little cross-layer overlap to lose; cf. prefill's +3%). Run:
`MH_TIMEOUT=1500 scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_moe_decode_hbm.py --distributed`. If confirmed cheap ⇒
prod barrier (GATE ×2 engines, md5 SHOULD stay `3069e80b`) AND re-opens dual-residency (#5: 26.29 resident + single-digit decode
< 31.25 → fits → would kill the 225 ms/fwd prefill rhs-prep).

**Also pending (confirmatory, lower priority — decode is primary):** P.18's e2e prefill LATENCY win is UNMEASURED (microbench-
projected −5.6..−48.9%). A profiler re-capture (recipe at bottom) would confirm the wall-time delta + quantify the new reshard
cost. Roadmap #7 covers this.

---

## THE ROADMAP (re-ranked P.18 — #1 o-proj sharding LANDED; the decode HBM microbench is now the top lever)
1. **[prefill o-proj TOKEN-SHARDING]** ✅ **LANDED P.18** (GATED ×2 engines, lossless, decode-neutral; prefill HBM 1.51→2.95).
   o stays token-sharded → o-proj/hc/norm/gate run 1/16-sharded to the MoE all_gather. e2e LATENCY win microbench-PROJECTED
   (P.10 −5.6..−48.9% @N=512..4096) but UNMEASURED e2e — confirmatory profiler re-capture pending (NEXT ACTION / #7).
2. **[decode dense-dequant HBM — microbench FIRST]** ⇐ TOP. the DECODE forward is the binding HBM peak (~12–20 GiB,
   scheduler-variable) from hoisted `_dequant_fp4_experts`. A decode-side barrier caps it LOSSLESSLY but may cost decode
   latency (PRIMARY target) — A/B HBM-vs-latency on a decode microbench before prod (SCOPED — see NEXT ACTION). Payoff:
   decode HBM headroom + possibly re-opens dual-residency (#5). *M.*
3. **[prefill non-MoE LAUNCH floor ~117 ms]** PROJ 42 + NORM 31 + HC 17–23 + GATE 11 + CMP 10 + IDX 5, seq-INDEP,
   launch-bound at tiny per-chip n. Lossless cut = op-count↓ (layer scan / fuse the ~215 matmuls) but re-opens S1
   (DO-NOT-RETRY #10). *M · hard.*
4. ~~**[prefill DISPATCH fuse]**~~ **TAPPED (P.13).** dispatch 36/40/64/95 ms/fwd @N=512/1024/2048/4096; the SORT is
   unavoidable, owned-gather BLOCKED on v6e (#24), inverse-scatter REFUTED (#23). Scrap left = bf16-gmm-output (lossy
   ~27 ms/fwd @4096) — small + risky, not a clear lever.
5. ~~**[prefill rhs-prep KILL — dual-residency]**~~ **REFUTED (P.15/P.17, #25).** Resident fits (26.29 GiB) but the
   forward peak (now DECODE ~12–20 GiB) can't coexist with +16 GiB. CONDITIONALLY re-openable IF #2 caps decode HBM to
   single digits. rhs-prep is also stuck on LATENCY (225 ms/fwd; P.12 VPU floor — the barrier caps HBM, not time).
6. **[fp8-resident FOR ALL — REFUTED]** P.9b net-negative (decode +39 ms/step). *parked.*
7. **[decode clean profiler / diff-shrink]** confirmatory + remove `_v4_nan_tripwire` (37 sites + def + `smoke.sh:81/
   116`), edit `.py`+`.sh` TOGETHER (Pitfall #0), KEEP `_linear` clamp + `compute_logits` nan_to_num. *S.*

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
21. **prefill attention TOKEN-SHARDED OUTPUT — was CompileTimeHbmOom +1.25G (P.11), UNBLOCKED by P.17, LANDED P.18 (DONE).** The OOM cause was XLA keeping ~20 concurrent rhs-prep weight-unpacks live (20× `f8e4m3fn[16,4096,4096]` ≈ 5 GB → Used 32.50/31.25 GiB) when token-sharding propagated past o into the FFN/MoE — EXACTLY the hoist the P.17 `optimization_barrier` caps. P.18 kept o token-sharded (`deepseek_v4_attention.py:269`, `P()`→`P(None,ATTN_DATA,None,None)`); prefill HBM 1.51→2.95 GiB (fits easily, ~28 GiB headroom), decode 19.62 unchanged, GATED ×2 engines md5 `3069e80b` (lossless). The new cost: (pre-existing-category) involuntary-remat reshards at the gate/hc boundary (the +1.44 GiB) — e2e latency win UNMEASURED, profiler re-capture pending.
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
25. **Prefill rhs-prep DUAL-RESIDENCY (keep experts fp4-resident for decode + fp8-resident for prefill) — REFUTED (P.15).**
   IMPLEMENTED end-to-end (loader pre-builds the fp8 gmm rhs `W13`/`W2t` post-load; prefill `_routed_local` reads them
   when `params.w13_fp8_stacked is not None`, rebuilding only the cheap e8m0 scale in-trace; registered as MoEParams
   pytree CHILDREN so they thread as jit args (the FIRST attempt closed them over → "non-addressable" RuntimeError);
   `V4_MOE_FP8_RESIDENT=1` env flag forwarded to workers). The model LOADS, the fp8 pre-build runs (44 layers), +16 GiB
   resident (26.29 total) FITS, KV cache (670k tok) allocates, `Application startup complete`. But the FIRST request OOMs:
   `jit_run_model` needs **19.56 GiB** new reservation, only 4.70 reservable (26.29 resident) ⇒ RESOURCE_EXHAUSTED on a
   1-token DECODE. The unified forward's ~19.56 GiB peak can't coexist with +16 GiB resident — at MAX_LEN=256/
   `max_num_batched_tokens=256` (already minimal), so no config saves `26.29+19.56=46 ≫ 31.25`. The P.14 microbench
   (`perf_microbench_dual_residency.py`, KEPT) measured the MoE-gmm transient ONLY (≤0.62 GiB) and missed the
   full-program live-buffer footprint — so resident-fits ≠ forward-fits. Prod edits REVERTED (NOT committed; the recipe
   to re-apply is exactly: (a) MoEParams fields `w13_fp8_stacked`/`w2t_fp8_stacked` + register them as pytree CHILDREN at
   `deepseek_v4.py:_register_pytree(MoEParams,...)`; (b) a post-load pass after `load_weights_from_dir done` that, gated on
   `V4_MOE_FP8_RESIDENT`, unpacks each layer's `w{1,3,2}_stacked`→fp8 `W13`=concat[unpack(w1),unpack(w3)] + `W2t`=unpack(w2)
   and sets them on the moe; (c) `_routed_local` reads them when `params.w13_fp8_stacked is not None`, rebuilding only the
   e8m0 scale in-trace via a `_fp4_scale_only` helper; (d) `V4_MOE_FP8_RESIDENT` exported + added to smoke.sh's
   `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`. THE FIX TO TRY: build (b) so the fp8 ALIASES as a forward input — mirror how
   `w1_stacked` is built (host-gather/`make_array_from_callback` + donation), NOT a standalone `jax.jit`.) ⚠️ ONE open
   question (NEXT ACTION Step 1): is 19.56 GENUINE forward activation (dual truly dead) or the 16 GiB
   fp8 weights NOT aliased as a jit input (salvageable — build them the loader's host-gather way, donate them, re-smoke)?
   (The impl was REVERTED, NOT committed — the 4-component recipe below is the spec to re-apply.)

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
- ★ **P.16/P.17 prefill HBM-peak cap (tier-2 microbench + real-model smoke):** `perf_microbench_moe_prefill_hbm.py` (K-layer
  MoE region, `compiled.memory_analysis().temp_size`) A/Bs the rhs-prep unpack-concurrency barrier — K=4/16/43 noBar
  2.00/7.76/9.14 GiB → bar 0.50/0.50/0.51 GiB (FLAT in K, +7/4/3% latency). REAL model (smoke 210955Z): prefill
  `jit_run_model` **18.34 → 1.51 GiB**; decode 11.60 → **19.62 GiB** (scheduler-VARIABLE; the SAME hoist on the dense
  `_dequant_fp4_experts`, NOT the barrier — decode is the dense path). The 18.34 baseline = smoke 085830Z. Run:
  `MH_TIMEOUT=1500 scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_moe_prefill_hbm.py --distributed`.
  ★ **P.18 (o-proj token-sharding, smoke 214536Z):** prefill `jit_run_model` 1.51 → **2.95 GiB** (+1.44 from gate/hc
  involuntary-remat reshard intermediates — replicate-then-partition), decode **19.62 GiB UNCHANGED** ⇒ the o-proj sharding
  fits with ~28 GiB headroom AND is decode-neutral. (Involuntary-remat warnings are pre-existing: 210955Z baseline 1011 vs
  214536Z 246 — a normal V4-compile artifact, correct, NOT new to P.18.)
- ★ **P.14/P.15 dual-residency HBM (real-model smoke, MAX_LEN=256):** keeping BOTH fp4 (8.57) + fp8-weights (16.13,
  scales kept e8m0) + non-expert (~1.6) = **26.29 GiB/chip RESIDENT** — loads, fp8 pre-build runs (44 layers), KV
  cache (670k tok ≈ MLA is byte-tiny) allocates, startup completes. BUT `jit_run_model` reserves **~19.56 GiB beyond
  resident** ⇒ 26.29+19.56=46 > 31.25 ⇒ OOM (DO-NOT-RETRY #25). KEY NEW FACT: the unified prefill-forward program peak
  is ~19.56 GiB (vs the MoE-gmm transient's 0.62) — the model runs near the 31.25 ceiling, which is ALSO why #4 (token-
  sharded o) is HBM-blocked (#21). The `perf_microbench_dual_residency.py` resident allocation (26.29) is accurate; its
  transient projection is NOT (MoE-only). `GPU_MEM_UTIL`/`--gpu-memory-utilization` doesn't help (program reservation is
  separate from the KV budget; even 0 KV leaves 46>31.25).
- **CPU torch oracle** `scripts/s1_cpu_repro_v4flash.py both` = Tier-1 math/NaN check (a bit-identical change keeps "OK both match").
- **THE PROFILE re-capture recipe:** profiled smoke (`V4_PROFILER_ARGS=…torch`), `/start_profile` →
  `s1_probe2.py 20` → `/stop_profile`. Parser `scripts/perf_parse_trace.py <trace> --bucket-ops`. Read the
  **2nd+** decode step; discount host `ParseArguments` ~100× (observer effect). The trace ALSO contains a
  prefill region (the prompt) — `perf_parse_trace.py` aggregates the WHOLE trace (no windowing yet; a
  `--decode-step`/`--prefill` window would need ~40 lines, see old fork A / roadmap #5).

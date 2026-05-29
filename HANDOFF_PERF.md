# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign (v6e-16)

> **Phase = PERFORMANCE.** Make `vllm serve DeepSeek-V4-Flash` DECODE + PREFILL FAST on the v6e-16
> slice WITHOUT breaking the correctness GATE. Durable slice ops + pitfalls: `CLAUDE.md`. The FIT
> milestone (256 routed experts kept FP4-compressed; `MAX_SEQS=1`) is DONE and is a GIVEN — history in
> `HANDOFF_QUANT.md`. S1 determinism history: `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29 — P.10): PREFILL non-MoE split DONE → a NEW #1 lever found.** New
> `perf_microbench_prefill_nonmoe.py` (16-chip) characterizes the WHOLE non-MoE prefill at the REAL sharding
> (dense ops token/seq-parallel at n=N/16; ATTENTION all-gathered to REPLICATED at M=N — the Mosaic kernel
> can't auto-partition). Two halves: (a) a SEQ-INDEPENDENT non-MoE floor ≈ **117 ms/fwd** (PROJ 42 + NORM 31
> + HC 17–23 + GATE 11 + CMP 10 + IDX 5), LAUNCH-bound at the tiny per-chip token counts (the prefill analog
> of decode's launch floor); and (b) **ATTENTION, which runs REPLICATED at M=N on EVERY chip (16× redundant)
> and DOMINATES long context: 33/69/197/584 ms at N=512/1024/2048/4096.** non-MoE share of the prefill device
> wall = 35→64% as context grows (MoE 275→393). **THE NEW #1 LEVER:** shard the prefill attention over the
> token axis (the kernel docstring's OWN "future optimization"). MICROBENCH-PROVEN feasible + the payoff: attn
> 584→**44 ms** at N=4096, cutting the TOTAL prefill wall by **5.6 / 12.1 / 27.3 / 48.9%** at N=512/1024/2048/
> 4096. DECODE-NEUTRAL (decode is M=1, stays replicated — the S1 fix path) + LOSSLESS (per-query independence).
> Measurement-only ⇒ **GATE md5 `3069e80b` UNCHANGED.**

---

## GATE — regression bar, NON-NEGOTIABLE (full mechanics in `CLAUDE.md` §CORRECTNESS GATE)
- **correct Fibonacci** `21,34,55,89,144,233,377,610` + **N=2 md5 `3069e80b` byte-identical ×2 fresh
  engines** (`scripts/s1_probe2.py 2`) + `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` rc=0.
- A numerics-changing fix MAY shift the md5 → re-establish a NEW ref + confirm identical ×2 engines +
  correct Fib. Do NOT gate on the long-tail md5 (`s1_probe2.py 100` → `ab07ecbb` is NON-deterministic at
  temp=0 by design). **`MAX_SEQS=1` is PINNED** (concurrent decode S1-broken).
- P.6–P.10 changed NO production code (only `scripts/perf_*`) ⇒ md5 still `3069e80b`, no smoke needed. The
  last GATE pass (P.5, bit-identical lean-dequant) stands. The roadmap-#1 attention-sharding change is
  LOSSLESS (per-query independence) ⇒ md5 SHOULD be unchanged — verify with a smoke before claiming it.

---

## ⇒ NEXT ACTION — implement the prefill ATTENTION-SHARDING lever (new #1; cuts the prefill wall up to 48.9%)
The P.10 split found prefill attention runs REPLICATED at M=N on every chip (the Mosaic kernel can't be
auto-partitioned, so prefill all-gathers the activation to replicated — `deepseek_v4_attention.py:219-245`
docstring). At long context this DOMINATES (584 ms/fwd at N=4096, > the whole MoE). The microbench PROVED
the kernel runs correctly sharded over the query/token axis (M), cutting attn 584→44 ms (N=4096) and the
total prefill wall by up to 48.9% — and it's the docstring's OWN intended optimization.

  **Implement a TOKEN-SHARDED prefill attention path.** At the prefill call site (`attention_prefill`
  :950), run the kernel under a shard_map that SHARDS the query axis over `attn_dp` instead of the
  replicated map. Proven-working specs (microbench `attn_kernel_sharded`):
    in_specs=(P(None,'attn_dp',None,None),  # q   [B,M,H,D]  sharded on M
              P(),                           # kv  [B,Nk,D]   REPLICATED (all keys needed)
              P(),                           # attn_sink [H]
              P(None,'attn_dp',None)),       # topk_idxs [B,M,K] sharded on M
    out_specs=P(None,'attn_dp',None,None)    # o  [B,M,H,D]  sharded on M
  ⚠️ DO NOT touch `_sparse_attn_kernel_sharded` itself — DECODE goes through it at M=1 and MUST stay
  REPLICATED (the S1 fix / Pitfall #5; M=1 can't shard 16-way). Add a PREFILL-ONLY sharded variant (or gate
  on M>1). Wire it so q/topk arrive sharded on the prefill token axis and o stays sharded (the o-proj +
  downstream MoE then run on the chip's token slice; the MoE already all_gathers). Also resolve the small
  cross-shard key reach: kv stays REPLICATED (each chip needs all keys), so the only added collective is a
  small kv all-gather — and sharding REMOVES the large q all-gather, so the real win ≥ the measured compute.
  *Risk MED-HIGH: touches the S1-sensitive sharding → REQUIRES a full smoke + the GATE (FIB + N=2 md5
  `3069e80b` ×2 fresh engines + smoke_check). Lossless ⇒ md5 SHOULD be unchanged — verify.* Highest-EV
  prefill change on the board (per-N saved: 24/58/177/540 ms at N=512/1024/2048/4096).

  Then (seq-indep, short-context): the MoE rhs-prep (225 ms/fwd) DECODE-NEUTRAL faster unpack (roadmap #2),
  and the launch-bound non-MoE floor (roadmap #3, hard — op-count↓ re-opens S1).

---

## THE ROADMAP (re-ranked P.10 — prefill: attention-sharding now leads at long context; decode CLOSED)
1. **[prefill ATTENTION-SHARDING]** NEXT ACTION: shard the replicated M=N prefill attention over the token
   axis. Cuts the prefill wall 5.6/12.1/27.3/48.9% at N=512/1024/2048/4096. Microbench-proven feasible,
   decode-neutral, lossless. *MED-HIGH risk (S1-sensitive sharding; needs a smoke). TOP EV.* ← TOP.
2. **[prefill MoE — DECODE-NEUTRAL rhs-prep cut]** the 225 ms/fwd FP4→fp8 UNPACK (~9× its HBM floor ⇒
   compute-bound). The top SEQ-INDEPENDENT cost (dominant at SHORT prefill). Keep fp4-resident (decode
   unchanged); faster in-trace unpack / small Pallas unpack kernel vs the 5.3 ms/layer baseline in
   `perf_microbench_moe_prefill.py`. *M · risk low (tier-2 first; no GATE risk until a real edit).*
3. **[prefill non-MoE LAUNCH floor ~117 ms]** PROJ 42 + NORM 31 + HC 17–23 + GATE 11 + CMP 10 + IDX 5, seq-
   INDEP, launch-bound at tiny per-chip n (decode's launch story, in prefill). Lossless cut = op-count↓
   (layer scan / fuse the ~215 per-fwd matmuls) but that re-opens S1 (DO-NOT-RETRY #10). *M · hard.*
4. **[prefill dispatch]** the MoE sort + two `[N·top_k,dim]` gathers, 80 ms/fwd at N=4096, grows with N.
   Fuse the gather into gmm / ragged scatter. *M.*
5. **[fp8-resident — REFUTED for normal serving]** P.9b: net-negative (decode +39 ms/step, break-even ~6
   gen tokens). Revisit ONLY for prefill-dominated (G<6) workloads. The LOSSY scaled-fp8 variant is worse
   still. *L · parked.*
6. **[decode clean profiler / 5-cleanup]** confirmatory (decode near-floor); + Phase-5 diff-shrink: remove
   `_v4_nan_tripwire` (37 sites + def + `smoke.sh:81/116`), edit `.py`+`.sh` TOGETHER (Pitfall #0), KEEP
   `_linear` clamp + `compute_logits` nan_to_num. *S.*

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

---

## VERIFIED FACTS / cheap tiers (don't re-derive)
- ★ **`perf_microbench_moe_prefill.py` (16-chip EP=16, real dims, balanced routing; P.9) — min ms/layer:**
  RHS-PREP (`_fp4_rhs_and_scale`×3 + concat + layout_constraint, the :351-358 in-trace prep) **5.24**,
  SEQ-INDEPENDENT (decomp: unpack-only 5.35, +swapaxes 5.74, +concat/layout 5.24 ⇒ the cost IS the bit-
  unpack). gmm-core 0.53/0.59/0.69/0.87 at N=512/1024/2048/4096 (0.82–1.24× the dense-fp8 floor); dispatch
  0.29/0.43/1.07/1.87; collective 0.33/0.45/0.69/1.17. ×43 ⇒ prefill MoE **275/288/331/393 ms**; rhs-prep
  alone = **225 ms/forward** (the lossless lever). Run: `full_slice_v4_sync.sh` then `MH_TIMEOUT=900
  scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_moe_prefill.py --distributed`. ⚠️ gmm_v2 is a
  Mosaic kernel ⇒ MUST be wrapped in `shard_map` (inputs replicated P() ⇒ per-rank cost). Caveat: rhs-prep
  is a standalone UPPER bound (XLA may overlap layer L+1's VPU unpack with layer L's MXU); the load-time
  lever removes it regardless of overlap.
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

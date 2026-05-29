# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign (v6e-16)

> **Phase = PERFORMANCE.** Make `vllm serve DeepSeek-V4-Flash` DECODE + PREFILL FAST on the v6e-16
> slice WITHOUT breaking the correctness GATE. Durable slice ops + pitfalls: `CLAUDE.md`. The FIT
> milestone (256 routed experts kept FP4-compressed; `MAX_SEQS=1`) is DONE and is a GIVEN — history in
> `HANDOFF_QUANT.md`. S1 determinism history: `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29 — P.9): PIVOTED to PREFILL (fork B); first prefill MoE characterization
> DONE.** Decode is at its ~lossless floor (146 ms/step, fully attributed P.5–P.8, LAUNCH-bound at N=1 —
> see "DECODE — CLOSED"). Prefill was a blank slate (zero numbers in 8 iters) and is the untapped half of
> the goal. New `perf_microbench_moe_prefill.py` (16-chip, mirrors the real `_routed_local` gmm_v2 path):
> **the prefill MoE is DOMINATED by the in-trace FP4→fp8 rhs-prep = 225 ms/forward, SEQ-INDEPENDENT**
> (decomp: it's the bit-UNPACK ~5.3 ms/layer ×43; swapaxes +0.4, concat ~0). gmm-core is near its dense-fp8
> floor (0.82–1.24×, 23–38 ms/fwd, MXU-underutilized at these per-expert counts — **not** a lever). dispatch
> (the `[N·top_k,dim]` argsort+gathers = the old "48% copy", CONFIRMED) + collective SCALE with N (80+50 ms
> at N=4096). Projected prefill MoE: 275 (N=512) → 393 ms (N=4096). **THE LEVER:** rhs-prep is removable
> LOSSLESSLY by storing the fp8 CODES resident at load (17.1 GiB/chip FITS; e2m1⊂e4m3) — DISTINCT from
> roadmap #4's LOSSY scaled-fp8. Likely a DOUBLE win (re-opens decode's einsum-fp8w floor, P.6 0.75/layer ⇒
> ~106→32 ms) pending decode-bench validation. Measurement-only commit ⇒ **GATE md5 `3069e80b` UNCHANGED.**

---

## GATE — regression bar, NON-NEGOTIABLE (full mechanics in `CLAUDE.md` §CORRECTNESS GATE)
- **correct Fibonacci** `21,34,55,89,144,233,377,610` + **N=2 md5 `3069e80b` byte-identical ×2 fresh
  engines** (`scripts/s1_probe2.py 2`) + `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` rc=0.
- A numerics-changing fix MAY shift the md5 → re-establish a NEW ref + confirm identical ×2 engines +
  correct Fib. Do NOT gate on the long-tail md5 (`s1_probe2.py 100` → `ab07ecbb` is NON-deterministic at
  temp=0 by design). **`MAX_SEQS=1` is PINNED** (concurrent decode S1-broken).
- P.6–P.9 changed NO production code (only `scripts/perf_*`) ⇒ md5 still `3069e80b`, no smoke needed. The
  last GATE pass (P.5, bit-identical lean-dequant) stands. A LOSSLESS fp8-resident change SHOULD keep the
  md5 — verify with a smoke before claiming it.

---

## ⇒ NEXT ACTION — validate the fp8-codes-resident lever CHEAP-FIRST, then implement at load
P.9 found the prefill MoE's dominant cost is the in-trace FP4→fp8 rhs-prep (225 ms/forward, seq-indep),
removable LOSSLESSLY by storing fp8 codes resident at load. Before the risky load-path change, confirm
it's a clean win at the cheapest tier, in this order:

  **(1) DECODE-side check [cheap, tier-2, ~2 min].** fp8-resident means decode's dense N=1 path
  (`moe_forward` :300-330, `_dequant_fp4_experts`→bf16) would dequant from **fp8** (17.1 GiB, 2× the fp4
  HBM read, but NO bit-unpack) instead of fp4. Does that regress or improve decode? Add a `lean-dequant-
  from-fp8` variant to `perf_microbench_moe_decode.py` (lean dequant but codes pre-cast fp8 not packed-fp4)
  and compare to the production fp4 lean-noWSC (2.46 ms/layer). P.6 `einsum-fp8w`=0.75 hints the *direct*
  fp8 read floor is great, but that needs a kernel (gmm loses at N=1, DO-NOT-RETRY #1) — so the realistic
  decode path is fp8→bf16 lean dequant. **If decode is neutral-or-better, fp8-resident is a clean DOUBLE
  win.** *S · risk low.*

  **(2) Implement load-time fp8-codes-resident [the big prefill win].** Pre-unpack+swapaxes+concat the FP4
  codes to fp8 W13/W2t (+ keep e8m0 scales) ONCE at load; `moe_forward`'s prefill path reads the pre-built
  rhs (skip `_fp4_rhs_and_scale` :351-358) → removes ~225 ms/forward. Touches the loader (the w1/w3
  host-gather IS the S1 fix — careful, do NOT alter the all_gather/barrier/owned-mask/psum determinism
  structure). LOSSLESS ⇒ GATE md5 should HOLD — but this is a load-path change, so it NEEDS a full smoke +
  the GATE. *M · risk MED (load path + a required smoke).*

  **(3) Complete the prefill NON-MoE split [parallel, cheap].** MoE is characterized; attention-prefill /
  sinkhorn / projections / gate are NOT — so we don't yet know MoE's share of the FULL prefill wall.
  Extend `perf_microbench_attn_decode.py` to prefill shapes (the `prefill_csa/hca/swa` presets already
  exist in `perf_microbench_sparse_attn.py:48-55`); add the sinkhorn at [S,·]. *M · risk low.*

---

## THE ROADMAP (re-ranked P.9 — PREFILL is now the active half; decode is CLOSED at its floor)
1. **[prefill MoE — the big LOSSLESS lever]** NEXT ACTION (1)+(2): validate then implement fp8-codes-
   resident → removes the 225 ms/forward rhs-prep (prefill MoE 275→~50 / 393→~168 ms) and likely also
   drops decode MoE (~106→32, einsum-fp8w). *M · risk MED.* ← TOP.
2. **[prefill full split]** NEXT ACTION (3): characterize prefill attention/sinkhorn/projections to find
   the next prefill stage after MoE. *M · risk low.*
3. **[prefill dispatch]** The sort + two `[N·top_k,dim]` gathers scale with N (80 ms/fwd at N=4096) — a
   long-context lever (fuse the gather into gmm / ragged dispatch). Secondary. *M.*
4. **[MoE scaled-fp8-resident — LOSSY variant]** roadmap-#4-old: bake `code×scale→fp8` (vs #1's lossless
   codes-only). Only if #1's lossless form doesn't fit or decode needs the *direct* fp8 read. LOSSY (rel
   ~3e-3/layer → likely breaks GATE). *L · risk HIGH — a gamble, superseded by #1's lossless framing.*
5. **[decode clean profiler]** old fork (A): a steady multi-step decode profiler to confirm the ~24 ms
   launch overhead directly + settle the ~700-copy question. Confirmatory only; decode is near-floor. *M.*
6. **[5-cleanup]** Phase 5 diff-shrink — remove `_v4_nan_tripwire` (37 sites + def + `smoke.sh:81/116`).
   Edit `.py` AND `.sh` TOGETHER (Pitfall #0). KEEP `_linear` clamp + `compute_logits` nan_to_num. *S.*

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
- **PREFILL MoE path = `deepseek_v4_moe.py:331-433` `_routed_local`** (use_shard_map when N≥axis): rhs-prep
  (:351-358) → shard_map[all_gather x → argsort by expert → gather `x_sorted[N·top_k,dim]` → 2× `gmm_v2`
  over this rank's EP=16 experts via `group_offset=[r·EP]`, fp8 codes + per-block fp32 rhs_scale → revert/
  owned-mask/combine/psum]. gmm `lhs[M,dim]@rhs[EP,k,n]`, `group_sizes[E]` balanced summing to lhs rows.
- **Decode per-step split (P.5, V4_DECODE_TIMERS):** device_wait 138.9 / wall 145.6 ms; MoE ~106 (76%).
  `perf_microbench_moe_decode.py` (lean-noWSC 2.46 = production /layer; einsum-fp8w 0.75 = the fp8-read
  floor) + `perf_microbench_attn_decode.py` (`--block`) are the decode tiers. (Full numbers: git P.5–P.8.)
- **HBM:** fp4 experts 8.57 GiB/chip (×43); fp8 codes resident = 17.1 GiB (FITS, ~21 free); bf16-resident
  34.3 (does NOT fit). N=1 decode HBM floor ~5.5 ms/step.
- **CPU torch oracle** `scripts/s1_cpu_repro_v4flash.py both` = Tier-1 math/NaN check (a bit-identical change keeps "OK both match").
- **THE PROFILE re-capture recipe:** profiled smoke (`V4_PROFILER_ARGS=…torch`), `/start_profile` →
  `s1_probe2.py 20` → `/stop_profile`. Parser `scripts/perf_parse_trace.py <trace> --bucket-ops`. Read the
  **2nd+** decode step; discount host `ParseArguments` ~100× (observer effect). The trace ALSO contains a
  prefill region (the prompt) — `perf_parse_trace.py` aggregates the WHOLE trace (no windowing yet; a
  `--decode-step`/`--prefill` window would need ~40 lines, see old fork A / roadmap #5).

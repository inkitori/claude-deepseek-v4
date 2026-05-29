# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign (v6e-16)

> **Phase = PERFORMANCE.** Make `vllm serve DeepSeek-V4-Flash` DECODE + PREFILL FAST on the v6e-16
> slice WITHOUT breaking the correctness GATE. Durable slice ops + pitfalls: `CLAUDE.md`. The FIT
> milestone (256 routed experts kept FP4-compressed; `MAX_SEQS=1`) is DONE and is a GIVEN — history in
> `HANDOFF_QUANT.md`. S1 determinism history: `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29 — P.8): the ~30 ms non-MoE balance is ATTRIBUTED — roadmap #1 CLOSED. It is
> mostly LAUNCH OVERHEAD, not the projections.** New `--block` mode on `perf_microbench_attn_decode.py`
> (16-chip, amortized, collectives INCLUDED) measured every named non-MoE suspect at real decode shapes:
> **projections** (wq_a/wq_b/wkv/wo_a/wo_b, with their all-reduce/all-gather) = **4.13 ms/step** (0.096/layer
> ×43); **MoE gate** (matmul+sqrtsoftplus+top_k+one_hot) = **0.75 ms/step** (×40 std-moe); **hc_pre + 19-iter
> sinkhorn** = **2.99 ms/step** (×86, the biggest non-MoE per-layer item). Sum = **7.87 ms/step**; with P.7's
> sparse_attn 0.32 + indexer 0.27 the measured op-groups are only **~8.5 ms** of the ~30 ms. The full split is
> self-consistent: device_wait 138.9 = MoE 105.8 + attn 0.6 + proj/gate/hc 7.87 + **~24 ms per-op launch/
> dispatch/bubble overhead** (✓ sums to 138.5). Corroborated by the on-disk trace: **~1760 XLA ops/step**
> (first-exec-tainted, approximate) ⇒ ~24 ms ÷ 1760 ≈ **13.6 µs/op effective**, ~2.7× the amortized per-op
> floor (~5 µs; a trivial add = 4.74 µs, a tiny mm = 5.37 µs) — exactly the gap real data-dependent ops have
> over a tight fori_loop. **⇒ decode at N=1 is LAUNCH-bound, as premised.** The only lossless lever for the
> ~24 ms is op-count reduction, whose big hammer (a layer-level `lax.scan`) re-opens S1 (DO-NOT-RETRY #10).
> No production code changed ⇒ GATE md5 `3069e80b` UNCHANGED, wall stays 146 ms. **⇒ The decode LOSSLESS
> frontier is exhausted-or-marginal: MoE 106 ms (P.6 floor), non-MoE 30 ms = ~8.5 ms irreducible compute +
> ~24 ms S1-blocked launch overhead. The remaining decode wins are the risky LOSSY MoE (roadmap #4) or a
> clean profiler to hunt removable ops; the campaign goal's PREFILL half is barely explored — see NEXT.**

---

## GATE — regression bar, NON-NEGOTIABLE (full mechanics in `CLAUDE.md` §CORRECTNESS GATE)
- **correct Fibonacci** `21,34,55,89,144,233,377,610` + **N=2 md5 `3069e80b` byte-identical ×2 fresh
  engines** (`scripts/s1_probe2.py 2`) + `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` rc=0.
- A numerics-changing fix MAY shift the md5 → re-establish a NEW ref + confirm identical ×2 engines +
  correct Fib. Do NOT gate on the long-tail md5 (`s1_probe2.py 100` → `ab07ecbb` is NON-deterministic at
  temp=0 by design). **`MAX_SEQS=1` is PINNED** (concurrent decode S1-broken).
- P.6/P.7/P.8 changed NO production code (only `scripts/perf_microbench*` + `scripts/perf_dense_fp8_moe_kernel.py`)
  ⇒ md5 still `3069e80b`, no smoke needed. The last GATE pass (P.5, bit-identical lean-dequant) stands.

---

## ⇒ NEXT ACTION — the decode attribution is DONE; pick the fork (decode lossless frontier is exhausted-or-marginal)
P.8 closed roadmap #1: the ~30 ms non-MoE = ~8.5 ms measured op-groups (proj 4.13 + gate 0.75 + hc 2.99 +
attn 0.6) + **~24 ms per-op launch overhead** (decode is launch-bound at N=1). Every named lossless suspect
is now either at its floor (MoE 106, attn) or measured-small (proj/gate/hc). The ONLY lossless lever for the
~24 ms is op-count reduction; its big hammer — a layer-level `lax.scan` — re-opens S1 (DO-NOT-RETRY #10).
So a *large* lossless decode win likely does NOT exist. Three honest forks, ranked by EV:

  **(A) [recommended] CLEAN multi-step profiler re-capture** — the cheap-ish way to DECIDE if ANY op-count
  lever survives before declaring the decode floor. The ~24 ms is currently INFERRED (residual) +
  corroborated by a first-exec-tainted ~1760-op trace. A steady 2nd+ decode step would (1) confirm ~24 ms
  directly and (2) settle the ~700-"copy" question the tainted trace raised — DO-NOT-RETRY #4 says decode
  emits NO copy/transpose (that was prefill), so if a clean trace shows many copies they may be a removable
  resharding artifact = a real lossless lever; if it shows only irreducible tiny dispatches, the decode
  floor is PROVEN and the campaign moves to (B)/(C). Needs a profiled smoke (`V4_PROFILER_ARGS`, recipe in
  VERIFIED FACTS) + a NEW `--decode-step` windowing flag on `perf_parse_trace.py` (take the 2nd
  `jit_run_model` start in the `XLA Modules` lane, walk `XLA Ops` to the next >2 ms gap). *M · risk low.*

  **(B) PIVOT to PREFILL** — the campaign goal is "prefill + decode" but the roadmap has been 100% decode,
  which is now shown launch-bound/near-floor. Prefill is COMPUTE-bound (gmm_v2 FP4 path, real MXU work) and
  barely profiled here — likely bigger, safer levers than squeezing launch-bound decode. Start: a prefill
  V4_DECODE_TIMERS-style split + the gmm_v2 prefill microbench already in `perf_microbench_moe_decode.py`.
  *L · the untapped half of the goal.*

  **(C) the risky LOSSY MoE — scaled-fp8-resident experts** (roadmap #4) — the only LARGE remaining decode
  win: bake `(fp4_code × e8m0_scale) → fp8 e4m3` at LOAD (fits 17.1 GiB/chip), decode reads fp8 directly
  (einsum-fp8w floor **0.75/layer**, ~106→~32 ms, NO in-trace dequant). But **LOSSY** (rel ~3e-3/matmul,
  compounds ×43 → likely breaks GATE) + touches the S1-fused load path. A deliberate gamble; needs
  re-baselining the md5 + a smoke that may fail. *L · risk HIGH — don't default into it.*

---

## THE ROADMAP (re-ranked P.8 — roadmap #1 "attribute the ~32 ms" is CLOSED; every item clears the GATE)
1. **[clean profiler]** Fork (A) in NEXT ACTION: a steady multi-step decode profiler to confirm the ~24 ms
   launch overhead DIRECTLY and decide if any op-count lever survives (the ~700-"copy" question vs
   DO-NOT-RETRY #4). Needs a profiled smoke + a `--decode-step` windowing flag on `perf_parse_trace.py`.
   Either finds a removable-op lever or PROVES the decode floor. *M · risk low.* ← TOP.
2. **[prefill]** Fork (B): pivot to the untapped PREFILL half of the goal (compute-bound, real MXU levers).
   *L.*
3. **[MoE scaled-fp8-resident]** Fork (C): the only LARGE remaining decode win — bake fp8 experts at load,
   decode reads fp8 (0.75/layer, ~106→32 ms). LOSSY (rel ~3e-3/layer → likely breaks GATE) + load-path.
   *L · risk HIGH — a deliberate gamble, not a default.*
4. **[dtype]** attention `_linear` `deepseek_v4_attention.py:514` bf16-in/fp32-acc (KEEP `|r|<1e8` clamp).
   The projections are only 4.13 ms now (P.8), so this is a SMALL numerics-shifting change — needs a
   re-baselined md5 + smoke, not a free win. Low priority. *S · risk MED (md5 shift).*
5. **[5-cleanup]** Phase 5 diff-shrink — remove `_v4_nan_tripwire` (37 sites + def + `smoke.sh:81/116`).
   Edit `.py` AND `.sh` TOGETHER (Pitfall #0). KEEP `_linear` clamp + `compute_logits` nan_to_num.
   Cosmetic; the documented fallback when levers stall. *S · risk low.*

---

## DO-NOT-RETRY (dead ends — do NOT burn a smoke; ★ = added/updated P.5–6)
1. ★ **gmm_v2 for the DENSE N=1 decode path — REFUTED (P.5 microbench 0.78×).** gmm's per-group machinery
   at 1 row/group costs MORE than the plain einsum, and the dense decode matmul is ALREADY near the
   bf16-read floor (0.79 ms/layer, ~1.6× floor — NOT MXU-starved). The fuse win is NOT "use gmm"; it's
   "stop materializing the bf16 dequant" (P.5 lean+wsc did this in XLA; the kernel route was then tried in
   P.6 and REFUTED — #12).
2. ★ **The bf16-dequant materialization was the lever, NOT the matmul.** Decomposed (P.5): dequant 6.15
   ms/layer (89%) vs einsum 0.79 (11%). "N=1 MXU starvation" was a red herring — the matmul is fine.
3. **nnx-preflatten — DONE (P.4).** Host-dispatch 56→8 ms/step. The residual ~6.7 ms fwd_disp is
   `_prepare_inputs`+embeds+enqueue (not flatten); sub-3%, not worth it.
4. **Decode "48% copy/transpose" device cost — that was PREFILL** (gmm_v2 rhs-prep `swapaxes`). DECODE emits none.
5. **Async scheduling** — DISABLED (RayDistributedExecutor forces `async_scheduling=False`); sync `device_get` is the live block.
6. **Collective fusion / `pick_partition_spec` axis flip / all-reduce** — ~0.3–2 ms/step (≤1%) on the 4×4 ICI torus (re-check under the re-profile, roadmap #1, now MoE is 76%).
7. **In-trace FP4→bf16 dequant on the PREFILL/sharded path** — `CompileTimeHbmOom` (Q.11). (Lever-1's kernel is DECODE-LOCAL, different.)
8. **Native typed `float4_e2m1fn` rhs to a kernel** — `MosaicError` on v6e (needs v7); fp8 codes is the v6e floor.
9. **`MAX_SEQS>1` concurrent decode** — CONFIRMED BROKEN (Q.15). Hard-pinned `=1`.
10. **Un-replicate/reshard the decode activation; `lax.scan` over layers; remove anchor buffers** — all re-open S1 / Pitfall #5.
11. **Remove the `_linear |r|<1e8` clamp or `compute_logits nan_to_num`** — both load-bearing (S1 + Q.15).
12. ★ **A hand-written in-trace decode kernel to beat the 2.46/layer MoE — REFUTED (P.6).** Naive dense
    Pallas matvec (`scripts/perf_dense_fp8_moe_kernel.py`) lowers + is CORRECT (CPU 4.1e-7) but is **11.6
    ms/layer** = 4.7× SLOWER: it contracts on the VECTOR unit, not the MXU (interpret-mode can't catch
    this). gmm_v2 (MXU + in-kernel per-block dequant) is 5.48. BOTH lose to XLA's materialize+matmul
    (2.46). An MXU-based dense kernel is conceivable but a long shot vs XLA's tuned N=1 dot + the per-block
    scale accumulation overhead. The kernel scratch file is kept as evidence / an MXU-rewrite seed.
13. ★ **fp8 codes FOR BANDWIDTH at N=1 — REFUTED (P.6).** einsum-fp8w (fp8 weights) 0.75 ≈ einsum-only
    (bf16) 0.78 /layer: the N=1 matmul is NOT weight-bandwidth-bound (the 0.78 floor is 1.6× the bf16-read
    HBM floor, so it's latency/MXU-fill-bound, not byte-bound). Reading 1-byte fp8 instead of 2-byte bf16
    saves nothing. (fp8-resident still has VALUE — but to skip the in-trace dequant, not to read faster.)
14. ★ **bf16-resident (pre-materialized) experts — does NOT FIT (P.6).** 8.57 GiB/chip fp4 × 4 = 34.3
    GiB > the 31.25 GiB budget. This is WHY QUANT keeps experts FP4 — do not re-litigate. (scaled-fp8
    resident = 17.1 GiB DOES fit but is lossy; roadmap #4.)
15. ★ **Swapping the Mosaic `sparse_attn_kernel` for pure-JAX `sparse_attn` on decode — REFUTED (P.7).**
    The kernel is **5× FASTER** even amortized (CSA 0.008 vs jax 0.051 ms/layer; HCA/dense 0.007 vs 0.015)
    and launch-bound (CONSTANT ~0.008 ms across kv_len 128→1152, i.e. the N=1 attention compute is ~nil).
    Parity bit-identical (max|Δ|=0). The kernel is optimal; do NOT revisit. (`perf_microbench_attn_decode.py`.)
16. ★ **The CSA indexer `lax.top_k` / score einsum as a decode lever — REFUTED (P.7).** `top_k([1,1,1024],
    512)`=0.009 ms, score einsum=0.004 ms (×21 CSA layers = 0.27 ms/step). The "indexer top_k scales with
    ctx" worry is real but the absolute cost is negligible at MAX_LEN=4096. Reducing `index_topk` is also
    NOT lossless (changes which KV the sparse attn sees). Drop it from the suspect list.
17. ★ **compute_logits head_w vocab sharding — ALREADY DONE (P.7).** `head_w[129280,4096]` fp32 is already
    column-sharded `P('attn_dp', None)` 16-way (`pick_partition_spec` → largest dim; mesh attn_dp=16); each
    chip reads only ~0.13 GB and computes its 1/16 logit slice with ZERO all-reduce/all-gather on the head.
    No replicated-2.1GB lever to capture — it's captured. (loader :469-520, model_loader :344/:398.)
18. ★ **The Q/KV/O projections + MoE gate + HC-sinkhorn are NOT the ~30 ms — REFUTED as the cost (P.8).**
    `--block` microbench (collectives included): proj 4.13 + gate 0.75 + hc-sinkhorn 2.99 = **7.87 ms/step**
    (+ P.7 attn 0.6 ⇒ ~8.5 ms). The ~24 ms balance is per-op LAUNCH overhead (per-op floor ~5 µs; ~1760
    ops/step). The only lossless lever (op-count↓ via a layer `lax.scan`) re-opens S1 (#10). Do NOT re-chase
    the projections as a decode lever; the dtype tweak (roadmap #4) is a small numerics-shift, not a free win.

---

## VERIFIED FACTS / cheap tiers (don't re-derive)
- **Decode per-step split (P.5 — V4_DECODE_TIMERS, 96 steady steps, MAX_SEQS=1 MAX_LEN=4096):**
  device_wait **138.9** / wall **145.6 ms** (was 207.9 / 216.3 at P.4). Of device_wait, MoE expert-FFN
  ≈ 106 ms (76%); the **~33 ms balance** is NON-MoE — P.7 showed sparse_attn+indexer is only ~0.6 ms of
  it, so the ~32 ms is projections + MoE gate + HC-sinkhorn + per-op launch overhead (unattributed; #1).
- **`perf_microbench_attn_decode.py` (16-chip, N=1, replicated; P.7) — AMORTIZED device-ms/call** (fori_loop
  R=64, dispatch removed): sparse_attn KERNEL CSA **0.008** HCA 0.007 dense 0.006 (→0.32 ms/step ×layers);
  pure-JAX sparse_attn CSA 0.051 (kernel 5× faster, parity max|Δ|=0); indexer `top_k([1,1,1024],512)`
  **0.009**, score einsum **0.004** (→0.27 ms/step ×21 CSA). ⚠️ the SINGLE-call med is ~0.18 ms
  dispatch-inflated — at N=1 ALWAYS amortize. Run: sync then `MH_TIMEOUT=900 …mh_run.sh
  scripts/perf_microbench_attn_decode.py --distributed`.
- **`perf_microbench_attn_decode.py --block` (16-chip, N=1, x replicated; P.8) — AMORTIZED ms/call, proj
  outputs forced replicated so each matmul's all-reduce/all-gather is INCLUDED:** wq_a 0.016 wq_b 0.022
  wkv 0.016 wo_a 0.022 wo_b 0.020 (SUM 0.096/layer → **4.13 ms/step ×43**); MoE gate (matmul+sqrtsoftplus+
  top_k+one_hot) 0.019 → **0.75 ms/step ×40**; hc_pre+19-iter sinkhorn 0.035 → **2.99 ms/step ×86**.
  TOTAL **7.87 ms/step**. Per-op launch floor: trivial add 4.74 µs, tiny mm 5.37 µs. ⇒ the ~24 ms balance
  is launch overhead (DO-NOT-RETRY #18). Run: `…mh_run.sh scripts/perf_microbench_attn_decode.py
  --distributed --block`. (wo_a is byte-modeled as a dense [8192,4096]; grouping doesn't change its bytes.)
- **`perf_microbench_moe_decode.py` (16-chip, N=1, real dims dim=4096 inter=2048 E=256/16 local, top_k=6) —
  ms/layer (P.6 re-run):** baseline **3.70** / lean **3.69** / lean-noWSC **2.46** (production) / gmm **5.48** /
  KERNEL (drafted dense fp8 in-reg dequant) **11.60** / dequant-only **5.15** / einsum-only **0.78** (matmul
  floor on resident bf16) / einsum-fp8w **0.75** (resident fp8 wts, bf16 act) / einsum-fp8 **0.75**. LEAN
  bit-identical to baseline (max|Δ|=0). lean-noWSC 2.46 == full-model V4DT/layer ⇒ FAITHFUL. The KERNEL +
  fp8w/fp8 variants are TIMING probes (kernel down=shared-x proxy; correctness = the kernel's own CPU
  oracle). Run: `scripts/full_slice_v4_sync.sh` then `MH_TIMEOUT=900 scripts/full_slice_v4_mh_run.sh
  scripts/perf_microbench_moe_decode.py --distributed`.
- **`scripts/perf_dense_fp8_moe_kernel.py`** — drafted dense FP8-code in-register-dequant Pallas matvec +
  its CPU oracle (`python3 scripts/perf_dense_fp8_moe_kernel.py`, interpret, 4.1e-7 rel err). Lowers on
  v6e (3D out [E,1,out] satisfies the Pallas last-2-dims-div-(8,128) rule) but is VECTOR-unit-bound (11.6,
  DO-NOT-RETRY #12). Seed for an MXU rewrite if anyone revisits; not on the live path.
- **`V4_DECODE_TIMERS` harvest:** smoke with `V4_DECODE_TIMERS=1`; `T0=$(date +%s)`; `s1_probe2.py 100`;
  `find /tmp/ray-vllm/ -type f -newermt "@$T0" | xargs grep -h '\[V4DT\]'`. decode = `ntok=32`. Drop
  first ~3 + `wall>500` (per-shape recompile) outliers; median. Brackets in `tpu_runner.py` (822-937 disp,
  1111-1114 the one `device_get` block).
- **CPU torch oracle** `scripts/s1_cpu_repro_v4flash.py both` = Tier-1 math/NaN check (truncated config,
  4 layers/8 experts; exercises the dense FP4 path). A bit-identical change keeps "OK both match".
- **HBM floor for N=1 decode ≈ 5.5 ms/step** (9.0 GiB/chip resident ÷ 1638 GiB/s); fp8-codes floor ~2×
  that, bf16-read floor ~4×. The dense matmul is near the bf16-read floor — the lever is the dequant.
- **THE PROFILE re-capture recipe:** profiled smoke (`V4_PROFILER_ARGS=…torch`), `/start_profile` →
  `s1_probe2.py 20` → `/stop_profile`. Parser `scripts/perf_parse_trace.py <trace> --bucket-ops`. Read
  the **2nd+** decode step; device timing is HW-accurate, discount host `ParseArguments` ~100× (observer effect).

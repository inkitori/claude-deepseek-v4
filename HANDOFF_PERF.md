# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign (v6e-16)

> **Phase = PERFORMANCE.** Make `vllm serve DeepSeek-V4-Flash` DECODE + PREFILL FAST on the v6e-16
> slice WITHOUT breaking the correctness GATE. Durable slice ops + pitfalls: `CLAUDE.md`. The FIT
> milestone (256 routed experts kept FP4-compressed; `MAX_SEQS=1`) is DONE and is a GIVEN — history in
> `HANDOFF_QUANT.md`. S1 determinism history: `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29 — P.7): the ATTENTION-SIDE decode levers are REFUTED — they are NOT the
> cost.** New tool `scripts/perf_microbench_attn_decode.py` (faithful, 16-chip, amortized) measured the
> prime non-MoE suspects in isolation at real decode shapes. Findings: **(1)** the Mosaic
> `sparse_attn_kernel` is OPTIMAL — **5× faster than the math-identical pure-JAX `sparse_attn`** even
> amortized (parity bit-identical, max|Δ|=0) AND launch-bound (~**0.008 ms/layer** true device, CONSTANT
> across kv_len 128→1152) ⇒ the pure-JAX swap is DEAD (DO-NOT-RETRY #15). **(2)** the CSA indexer is
> NEGLIGIBLE — `lax.top_k([1,1,1024],512)`=**0.009 ms**, score einsum=**0.004 ms** ⇒ the roadmap-flagged
> "indexer top_k lever" is REFUTED (#16). **(3)** sparse_attn + indexer together = **~0.6 ms** of the
> ~33 ms non-MoE; the **~32 ms balance = Q/KV/O projections + MoE GATE + HC-sinkhorn + the many small
> per-layer ops** — i.e. decode at N=1 is **on-device LAUNCH-bound** (~700 tiny ops × 43 layers), exactly
> the campaign premise. **METHOD NOTE:** a single jitted call has a ~0.18 ms host dispatch+sync floor that
> swamps these small ops (the naive rollup read 9 ms; amortizing R=64 in a `fori_loop` gave the true 0.3 ms)
> — always amortize at N=1. No production code changed ⇒ GATE md5 `3069e80b` UNCHANGED, wall stays 146 ms.
> **⇒ Both the MoE (106 ms, P.6) and the attention kernel/indexer are floor/optimal; the ONLY unattributed
> lossless frontier is the ~32 ms projections+gate+launch-overhead — roadmap #1.**

---

## GATE — regression bar, NON-NEGOTIABLE (full mechanics in `CLAUDE.md` §CORRECTNESS GATE)
- **correct Fibonacci** `21,34,55,89,144,233,377,610` + **N=2 md5 `3069e80b` byte-identical ×2 fresh
  engines** (`scripts/s1_probe2.py 2`) + `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` rc=0.
- A numerics-changing fix MAY shift the md5 → re-establish a NEW ref + confirm identical ×2 engines +
  correct Fib. Do NOT gate on the long-tail md5 (`s1_probe2.py 100` → `ab07ecbb` is NON-deterministic at
  temp=0 by design). **`MAX_SEQS=1` is PINNED** (concurrent decode S1-broken).
- P.6 changed NO production code (only `scripts/perf_microbench*` + a new `scripts/perf_dense_fp8_moe_kernel.py`)
  ⇒ md5 still `3069e80b`, no smoke needed. The last GATE pass (P.5, bit-identical lean-dequant) stands.

---

## ⇒ NEXT ACTION — attribute the ~32 ms balance (projections + gate + launch overhead)
P.7 closed the attention-side suspects: `sparse_attn_kernel` is optimal (DO-NOT-RETRY #15) and the indexer
is negligible (#16) — together only ~0.6 ms. So the ~32 ms balance of the non-MoE is **everything else
per decode step**: the Q/KV/O projections (`_linear` ×43: `wq_a[1024,4096]`, `wq_b[32768,1024]`,
`wkv[512,4096]`, `wo_a` grouped, `wo_b[4096,8192]`), the MoE GATE/router (NOT in the 105.8 ms MoE
microbench — that took pre-routed `pew`), the HC-sinkhorn (`hc_split_sinkhorn`, 20 iters ×43), the
rms_norms / splice_rope / kv_cache `.at[].set()` writes, and the per-op on-device launch overhead.
First-principles, none of those is compute/bandwidth-heavy (projections sharded read <1 ms total, logits
already vocab-sharded ~0.15 ms — see DO-NOT-RETRY #17), so the ~32 ms is most likely **on-device launch
overhead** (decode is launch-bound at N=1). **NEXT = attribute it**, two cheap tiers:
  (a) EXTEND `perf_microbench_attn_decode.py` to time a full per-layer block (all projections + gate +
      hc) amortized, ×43 — if it sums to ~32 ms the cost is those ops; if it's tiny, the cost is the
      inter-op launch overhead a single repeated op can't capture (→ op-count is the lever).
  (b) a CLEAN multi-step profiler re-capture (the only on-disk trace has just 1 first-exec step). Window a
      steady 2nd+ step with a NEW `--decode-step` flag on `perf_parse_trace.py` (recipe: read the
      `XLA Modules` lane, take the 2nd `jit_run_model` event's start, walk `XLA Ops` until a >2 ms gap).
If it IS launch overhead, the lossless lever = op-count reduction (fusion / a layer `lax.scan`), but
`lax.scan` over layers re-opens S1 (DO-NOT-RETRY #10) — so a lossless win may not exist and the campaign
then pivots to the risky MoE lever below or declares the decode floor.

The ONE remaining MoE lever, documented but HIGH-RISK (deliberate, don't default into it):
**scaled-fp8-resident experts** — bake `(fp4_code × e8m0_scale) → fp8 e4m3` at LOAD time (fits: 17.1
GiB/chip), decode reads fp8 directly (einsum-fp8w floor **0.75 /layer**, ~106→~32 ms, NO in-trace dequant).
But **LOSSY** (rel ~3e-3/matmul, compounds ×43 → likely breaks GATE) AND touches the S1-fused load path.
Only if decode latency is paramount and you accept re-baselining the md5 + a smoke that may fail. *L · risk HIGH.*

---

## THE ROADMAP (re-ranked P.7 — every item clears the GATE)
1. **[attribute the ~32 ms]** The NEXT ACTION (above): attention kernel + indexer are CLOSED (~0.6 ms);
   attribute the ~32 ms balance = projections + MoE gate + HC-sinkhorn + per-op launch overhead. Cheap
   (extend `perf_microbench_attn_decode.py` to a full-per-layer amortized block, and/or a clean multi-step
   profiler + a `--decode-step` windowing flag). Determines whether ANY lossless decode lever remains
   or it's S1-blocked launch overhead. *S.* ← TOP.
2. **[dtype]** attention `_linear` `deepseek_v4_attention.py:514` bf16-in/fp32-acc (KEEP `|r|<1e8` clamp).
   The projections ARE in the implicated ~32 ms now — but this is numerics-SHIFTING (not lossless), so it
   needs a re-baselined md5 + smoke, not a free win. *S · risk MED (md5 shift).*
3. **[5-cleanup]** Phase 5 diff-shrink — remove `_v4_nan_tripwire` (37 sites + def + `smoke.sh:81/116`).
   Edit `.py` AND `.sh` TOGETHER (Pitfall #0). KEEP `_linear` clamp + `compute_logits` nan_to_num.
   Cosmetic; the documented fallback when levers stall. *S · risk low.*
4. **[MoE scaled-fp8-resident]** The ONLY sub-2.46 MoE lever (see NEXT ACTION): bake fp8 experts at load,
   decode reads fp8 (0.75/layer, ~106→32 ms). LOSSY (rel ~3e-3/layer → likely breaks GATE) + load-path.
   *L · risk HIGH — a deliberate gamble, not a default.*

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
  scripts/perf_microbench_attn_decode.py --distributed`. EXTEND it next for the projections+gate (roadmap #1).
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

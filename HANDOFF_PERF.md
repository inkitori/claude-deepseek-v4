# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign (v6e-16)

> **Phase = PERFORMANCE.** Make `vllm serve DeepSeek-V4-Flash` DECODE + PREFILL FAST on the v6e-16
> slice WITHOUT breaking the correctness GATE. Durable slice ops + pitfalls: `CLAUDE.md`. The FIT
> milestone (256 routed experts kept FP4-compressed; `MAX_SEQS=1`) is DONE and is a GIVEN — history in
> `HANDOFF_QUANT.md`. S1 determinism history: `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29 — P.5): lean-dequant + wsc-drop LANDED + GATED → decode wall 216→146 ms
> (−33%).** A microbench (`scripts/perf_microbench_moe_decode.py`, real dims on the live 16-chip mesh)
> both CONFIRMED the bottleneck and REFUTED the old roadmap's gmm-fuse. At N=1 the dense MoE expert-FFN
> is **~100% of decode device compute** (baseline 4.28 ms/layer × 43 = 184 ms ≈ the 183 ms budget); the
> matmul itself is only **0.79 ms/layer** (near the bf16-read floor — NOT MXU-starved); **89% of the
> cost is the FP4→bf16 dequant materialization.** Routing it through gmm_v2 LOSES (0.78×: per-group
> overhead at 1 row/group swamps the fp8-streaming win). The WIN, both **BIT-IDENTICAL** (fp4 codes ×
> pow2 e8m0 scale are exact in bf16; `with_sharding_constraint` is value-preserving):
> **(a)** lean bf16 dequant — broadcast-multiply over the 32-block instead of `jnp.repeat` (kills the
> full-size fp32 scale temp); **(b)** drop the intermediate `_shard_e_first` on the dequanted weight (a
> dequant→matmul FUSION BARRIER; XLA now fuses the unpack into the operand load). Together **1.74× on the
> MoE path** (`deepseek_v4_moe.py:52-61,300-308`). Measured (V4_DECODE_TIMERS, 96 steady steps):
> **device_wait 207.9→138.9 ms, wall 216.3→145.6 ms.** GATE: md5 `3069e80b` UNCHANGED ×2 fresh engines,
> correct FIB `21,34,55,89,144,233,377,610`, smoke_check rc=0 (Mars 47 words, max_word_run=1, ends_clean).

---

## GATE — regression bar, NON-NEGOTIABLE (full mechanics in `CLAUDE.md` §CORRECTNESS GATE)
- **correct Fibonacci** `21,34,55,89,144,233,377,610` + **N=2 md5 `3069e80b` byte-identical ×2 fresh
  engines** (`scripts/s1_probe2.py 2`) + `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` rc=0.
- A numerics-changing fix MAY shift the md5 → re-establish a NEW ref + confirm identical ×2 engines +
  correct Fib. Do NOT gate on the long-tail md5 (`s1_probe2.py 100` → `ab07ecbb` is NON-deterministic at
  temp=0 by design). **`MAX_SEQS=1` is PINNED** (concurrent decode S1-broken).
- P.5 lean-dequant is numerics-PRESERVING (bit-identical) ⇒ md5 unchanged `3069e80b`, re-confirmed ×2
  fresh engines + correct FIB + smoke_check rc=0.

---

## ⇒ NEXT ACTION — the MoE is STILL the dominant device cost; close toward the 34 ms matmul-floor
After P.5 the MoE expert-FFN is **~106 ms = 76% of the 139 ms device_wait** (= 2.46 ms/layer; the
full-model V4DT matches the microbench's `lean-noWSC` 2.47 EXACTLY → the microbench is FAITHFUL for the
MoE device cost — iterate there, NOT in 25-min smokes). The matmul-only floor is **0.79 ms/layer
(~34 ms/step)**, so ~72 ms/step of un-fused bf16 dequant materialization REMAINS. XLA fuses MORE after
P.5 (4.28→2.47) but NOT fully — the TPU `dot` still materializes its operand (a Pallas/XLA fact, not a
missed flag). To approach the floor the dequant must move INSIDE the matmul kernel. Two levers (bigger
lifts; prototype in `perf_microbench_moe_decode.py` first — add a variant fn + time it):
  1. **Custom dense N=1 Pallas matmul** that reads fp8 codes + e8m0 scale and dequants IN-REGISTER (the
     gmm fuse WITHOUT gmm's per-group overhead — gmm itself is refuted, see DO-NOT-RETRY). Ceiling
     ~106→~40 ms. *M–L · risk MED (new kernel; fp8 may shift md5 → re-baseline).*
  2. **Top-k expert selection** — the dense path computes ALL 16 local experts then masks; only ~0.375
     are selected/chip (top_k=6 over 256). Far bigger ceiling but re-opens S1 / Pitfall #5 (decode-path
     token gather Core-halts). *L · risk HIGH.*
Cheaper parallel step: a **fresh decode-only device breakdown** of the new 139 ms device_wait — with MoE
now 76% (was ~100%) the other ~33 ms (attention / indexer `top_k` / collectives) is newly worth
attributing; the items dismissed as <1% when MoE was 100% deserve a re-rank.

---

## THE ROADMAP (re-ranked P.5 — every item clears the GATE)
1. **[MoE → matmul-floor]** The NEXT ACTION above: in-register dequant Pallas kernel (lever 1) and/or
   top-k selection (lever 2). The single biggest remaining lever (~72 ms/step of un-fused dequant). *M–L+.*
2. **[re-profile]** Decode-only device breakdown of the new 139 ms device_wait; re-rank attention /
   indexer / collectives now that MoE is 76% not ~100%. Cheap (microbench tiers + 1 profiler). *S.*
3. **[dtype]** attention `_linear` `deepseek_v4_attention.py:514` bf16-in/fp32-acc (KEEP `|r|<1e8` clamp).
   Small; ROI pending #2. *S.*
4. **[5-cleanup]** Phase 5 diff-shrink — remove `_v4_nan_tripwire` (37 sites + def + `smoke.sh:81/116`).
   Edit `.py` AND `.sh` TOGETHER (Pitfall #0). KEEP `_linear` clamp + `compute_logits` nan_to_num.
   Cosmetic; the documented fallback when hard levers stall. *S · risk low.*

---

## DO-NOT-RETRY (dead ends — do NOT burn a smoke; ★ = added/updated P.5)
1. ★ **gmm_v2 for the DENSE N=1 decode path — REFUTED (P.5 microbench 0.78×).** gmm's per-group machinery
   at 1 row/group costs MORE than the plain einsum, and the dense decode matmul is ALREADY near the
   bf16-read floor (0.79 ms/layer, ~1.6× floor — NOT MXU-starved). The fuse win is NOT "use gmm"; it's
   "stop materializing the bf16 dequant" (P.5 lean+wsc did this in XLA; the kernel route is lever 1).
2. ★ **The bf16-dequant materialization was the lever, NOT the matmul.** Decomposed (P.5): dequant 6.15
   ms/layer (89%) vs einsum 0.79 (11%). "N=1 MXU starvation" was a red herring — the matmul is fine.
3. **nnx-preflatten — DONE (P.4).** Host-dispatch 56→8 ms/step. The residual ~6.7 ms fwd_disp is
   `_prepare_inputs`+embeds+enqueue (not flatten); sub-3%, not worth it.
4. **Decode "48% copy/transpose" device cost — that was PREFILL** (gmm_v2 rhs-prep `swapaxes`). DECODE emits none.
5. **Async scheduling** — DISABLED (RayDistributedExecutor forces `async_scheduling=False`); sync `device_get` is the live block.
6. **Collective fusion / `pick_partition_spec` axis flip / all-reduce** — ~0.3–2 ms/step (≤1%) on the 4×4 ICI torus (re-check under roadmap #2 now MoE is 76%).
7. **In-trace FP4→bf16 dequant on the PREFILL/sharded path** — `CompileTimeHbmOom` (Q.11). (Lever-1's kernel is DECODE-LOCAL, different.)
8. **Native typed `float4_e2m1fn` rhs to a kernel** — `MosaicError` on v6e (needs v7); fp8 codes is the v6e floor.
9. **`MAX_SEQS>1` concurrent decode** — CONFIRMED BROKEN (Q.15). Hard-pinned `=1`.
10. **Un-replicate/reshard the decode activation; `lax.scan` over layers; remove anchor buffers** — all re-open S1 / Pitfall #5.
11. **Remove the `_linear |r|<1e8` clamp or `compute_logits nan_to_num`** — both load-bearing (S1 + Q.15).

---

## VERIFIED FACTS / cheap tiers (don't re-derive)
- **Decode per-step split (P.5 — V4_DECODE_TIMERS, 96 steady steps, MAX_SEQS=1 MAX_LEN=4096):**
  device_wait **138.9** / wall **145.6 ms** (was 207.9 / 216.3 at P.4). Of device_wait, MoE expert-FFN
  ≈ 106 ms (76%), matmul-floor ≈ 34 ms, other (attn/logits/gate/collectives) ≈ 33 ms.
- **`perf_microbench_moe_decode.py` (16-chip, N=1, real dims dim=4096 inter=2048 E=256/16 local, top_k=6):**
  baseline **4.28** / lean **3.70** / lean-noWSC **2.47** / gmm **5.48** / dequant-only **6.15** /
  einsum-only **0.79** ms/layer. LEAN bit-identical to baseline (max|Δ|=0). lean-noWSC 2.47 ==
  full-model 2.46/layer ⇒ the microbench is FAITHFUL for the MoE device cost. Run: `scripts/full_slice_v4_sync.sh`
  then `MH_TIMEOUT=900 scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_moe_decode.py --distributed`.
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

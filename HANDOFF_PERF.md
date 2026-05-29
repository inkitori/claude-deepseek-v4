# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign (v6e-16)

> **Phase = PERFORMANCE.** Make `vllm serve DeepSeek-V4-Flash` DECODE + PREFILL FAST on the v6e-16
> slice WITHOUT breaking the correctness GATE. Durable slice ops + pitfalls: `CLAUDE.md`. The FIT
> milestone (256 routed experts kept FP4-compressed; `MAX_SEQS=1`) is DONE and is a GIVEN — history in
> `HANDOFF_QUANT.md`. S1 determinism history: `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29 — P.3): decode per-step FAITHFULLY decomposed** (non-profiler worker
> timers, the new env-gated `V4_DECODE_TIMERS=1`, committed). The **277 ms/step decode wall splits:
> device_wait ~183 ms (66%) + host-dispatch ~56 ms (20%) + Ray-aDAG round-trip/scheduler ~38 ms
> (14%)**. This **OVERTURNS P.2's "host-dispatch is a ≤4% phantom"**: host dispatch is a REAL ~20% —
> the two jit dispatches (`run_model` + `run_compute_logits`) each re-walk the ~1,492-leaf nnx-`State`
> weight pytree (~28 ms each; the P.2 microbench under-estimated). BUT **device execution (66%) is the
> dominant bucket** and is **UN-ATTRIBUTED on the decode path** (P.1's "48% copy/transpose" is the
> PREFILL gmm_v2 rhs-prep transpose — the DECODE dequant emits NO transpose; code-audit confirmed).
> GATE re-confirmed (md5 `3069e80b`, FIB correct, smoke_check rc=0). Engine down; slice clean; guardians alive.

---

## GATE — regression bar, NON-NEGOTIABLE (full mechanics in `CLAUDE.md` §CORRECTNESS GATE)
- **correct Fibonacci** `21,34,55,89,144,233,377,610` + **N=2 md5 `3069e80b` byte-identical ×2 fresh
  engines** (`scripts/s1_probe2.py 2`) + `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` rc=0.
- A numerics-changing fix MAY shift the md5 → re-establish a NEW ref + confirm identical ×2 engines +
  correct Fib. Do NOT gate on the long-tail md5. **`MAX_SEQS=1` is PINNED** (concurrent decode S1-broken).
- P.3 instrumentation is host-only (40+/0− additions in `tpu_runner.py`, off by default) ⇒ md5 unchanged
  (`3069e80b`), confirmed ×2 fresh engines + correct FIB + rc=0.

---

## ⇒ NEXT ACTION — implement nnx-preflatten (the MEASURED ~20% host-dispatch lever, low-risk)
Host-dispatch is now DECOMPOSED and actionable WITHOUT further measurement: ~56 ms/step (`fwd_disp`
30.7 + `logits_disp` 25.2) is the cost of the TWO jit dispatches, each re-flattening the ~1,492-leaf
nnx-`State` weight pytree. Pass pre-flattened `state` leaves + a cached `treedef` (plain tuple) to
`run_model`/`run_compute_logits` and `tree_unflatten` INSIDE the jit (`model_loader.py:351-366`;
`nnx.merge` stays — free on cache-hit). Expected recovery: the State-flatten + pytree-walk portion of
the 56 ms (~10–40 ms; PARTIAL — the pjit C++ dispatch over 1,492 leaves remains). Low md5 risk (no
numerics change). Validate cheap→full: smoke + `V4_DECODE_TIMERS=1`, confirm `fwd_disp`/`logits_disp`
drop + GATE holds. This is roadmap #1.

(IN PARALLEL / AFTER, to unlock the 183 ms device_wait — the 66% bucket: get a **DECODE-ONLY device-op
breakdown** (multi-step profiler, read the 2nd+ decode step — recipe in §THE P.1 PROFILE). Likely
HBM-bound weight streaming at N=1 (every weight read once per token) ⇒ near-fundamental/HARD; CONFIRM
before attempting the risky gather-only-selected-experts MoE, which re-opens S1 / Pitfall #5.)

---

## THE ROADMAP (re-ranked P.3 — every item clears the GATE)
1. **[nnx-preflatten] Host-dispatch trim — MEASURED ~56 ms (20%)** (NEXT ACTION). Kills the per-dispatch
   nnx-`State` flatten across the 2 jit dispatches. `model_loader.py:351-366`. Low risk, low md5 risk. *S.*
2. **[PROFILE] Decode-only device-op breakdown** to attribute the 183 ms device_wait (66%). Multi-step
   profiler, read the 2nd+ decode step. Tells you if the device side is HBM-bound (hard) or has a fixable
   stall. No lever on the device bucket is justified until this lands. *M.*
3. **[L2-multistep] Multi-step on-device decode** — generate K tokens/dispatch; amortizes host-dispatch
   (56) + round-trip (38) = ~94 ms/step over K tokens (caps ~34% as K→∞; does NOT cut the 183 ms
   device_wait PER TOKEN). Big change, high risk (decode loop + S1 replicate boundary + sampling feedback).
   *L · risk HIGH.*
4. **[dtype] bf16-in / fp32-accumulate** — NOTE the decode MoE is ALREADY bf16/fp32 (PERF 3.1,
   `deepseek_v4_moe.py:304-306`). Remaining site: attention `_linear` `deepseek_v4_attention.py:514` (KEEP
   the `|r|<1e8` clamp; exact c78ecb96 pattern). ROI pending the #2 device breakdown. *S · md5 may shift.*
5. **[5-cleanup] Phase 5 diff-shrink — remove `_v4_nan_tripwire`** (37 sites + def + `smoke.sh:81/116`).
   Edit `.py` AND `.sh` TOGETHER (Pitfall #0). KEEP the `_linear` clamp + `compute_logits` nan_to_num.
   Cosmetic, zero perf; the documented fallback when hard levers stall. *S · risk low.*

---

## DO-NOT-RETRY (dead ends — do NOT burn a smoke; ★ = updated/added P.3)
1. ★ **"[L1] host dispatch is a ≤4% phantom" — CORRECTED: it's ~20% (56 ms), MEASURED.** The P.2
   microbench under-estimated (real nnx-`State` flatten on the 1,492-leaf model is ~28 ms/dispatch, not
   ≤12). The lever is **nnx-preflatten (roadmap #1)**, NOT fusing `compute_logits` into `run_model` —
   that saves only ONE of two dispatches AND forces an md5 re-baseline. Still pursue host dispatch, just
   via preflatten.
2. ★ **Decode "48% copy/transpose" device cost — that was PREFILL.** It's the gmm_v2 rhs-prep transpose
   (`deepseek_v4_moe.py:82`, `swapaxes`). The DECODE dequant `_dequant_fp4_experts` (`:37-56`) emits NO
   transpose (bitcast→broadcast→multiply→convert = `convert_bitcast_fusion`). Do NOT chase a decode transpose.
3. **Async scheduling** — DISABLED for this run (RayDistributedExecutor doesn't support it → vllm forces
   `async_scheduling=False`); the sync `device_get` (`tpu_runner.py:1112`) is the live block. The ~38 ms
   round-trip is the aDAG `get()` cost; only L2-multistep amortizes it (P.1's "2% async win" prize is tiny).
4. **Collective fusion / `pick_partition_spec` axis flip / all-reduce levers** — collectives are tiny
   8–13 µs ops on the 4×4 single ICI torus (topology fact). Device-op trace put all-reduce at ~1.3%.
5. **In-trace FP4→bf16 dequant on the PREFILL/sharded path** — `CompileTimeHbmOom` (Q.11). Decode dequant is LOCAL-only.
6. **Native typed `float4_e2m1fn` rhs to `gmm_v2`** — `MosaicError` on v6e (needs v7). fp8-codes is the answer.
7. **`MAX_SEQS>1` concurrent decode** — CONFIRMED BROKEN (Q.15): silent garbage on 3/4 reqs. Hard-pinned `=1`.
8. **Un-replicate / reshard the decode activation** (move `_v4_decode_replicate`) — re-opens S1 + Pitfall #5.
9. **`lax.scan` over decode layers / removing `_v4_anchor_output_buffers`** — layers already fuse into ONE
   jit; the `optimization_barrier` IS the S1 write-elision fix.
10. **Removing the `_linear |r|<1e8` clamp or `compute_logits nan_to_num`** — both load-bearing (S1 + Q.15).
11. **Indexer `top_k`** — DEFERRED at MAX_LEN≤4096 (small device share); approx_max_k flips FIB. Re-rank
    only if the decode-only device breakdown (#2) shows T-scaling.

---

## VERIFIED FACTS / cheap tiers (don't re-derive)
- **Decode per-step split (P.3 — non-profiler worker timers, `V4_DECODE_TIMERS=1`, MAX_SEQS=1
  MAX_LEN=4096):** device_wait **182.8 ms (66%)** / host-dispatch **56.05 ms (20%** = fwd_disp 30.7 +
  logits_disp 25.15 + samp_disp 0.2) / post 0.1; **worker wall 239 ms**. Driver 277 ms − worker 239 ms =
  **~38 ms aDAG round-trip + scheduler** (outside the worker's `_execute_model` timers). Medians over 90
  steady-state steps; all 4 TP workers identical (<1 ms spread).
- **`V4_DECODE_TIMERS` harvest recipe:** smoke with `V4_DECODE_TIMERS=1` → `grep -rh '\[V4DT\]'
  /tmp/ray-vllm/session_*/logs/`. ⚠️ The printed `ntok` = PADDED batch dim, NOT logical tokens:
  **decode = `ntok=32`** (1 tok padded to the 32 bucket); prefill = `ntok=128/256/512`. Drop the first
  ~3 + any `wall>500 ms` (per-shape recompile) outliers. The timer is host-only/off-by-default (gated;
  forwarded to workers via `tpu_platform.additional_env_vars` + `smoke.sh` `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`).
- **Decode MoE shape (`deepseek_v4_moe.py:296-318`, the `not use_shard_map` dense path):** dense einsum
  over all E experts (16 local/chip), FP4→bf16 dequant of ALL local experts (`:300-303`), masked by
  `per_expert_weight` post-hoc (`:315`). ALREADY dtype-optimized (PERF 3.1, bf16 operands/fp32 accumulate).
  top_k = `num_experts_per_tok` (config). Gathering only selected experts re-opens S1/Pitfall #5.
- **The decode wall is real:** non-profiled two-point fit (N=20 5.18 s, N=40 10.72 s) = **0.277 s/tok
  (277 ms/step)**, MAX_LEN=4096, MAX_SEQS=1. TRUSTED.
- **Two jit dispatches/step** (`run_model` `model_loader.py:353` + `run_compute_logits` `:385`); both take
  the full ~1,492-leaf weight `state` as an EXPLICIT arg (the preflatten target). `compute_logits` is at
  `deepseek_v4.py:1992` (rms_norm → fp32 `@ head_w.T` → nan_to_num `:2005`).
- **CPU torch oracle** `scripts/s1_cpu_repro_v4flash.py both` = Tier-1 math/NaN check (no md5/determinism).
- **THE P.1 PROFILE (re-capture recipe ONLY — its decode SPLIT + the prefill-conflated "48% copy/transpose"
  are NOT decode-representative):** profiled smoke (`V4_PROFILER_ARGS=…torch`), `/start_profile` →
  `s1_probe2.py 20` → `/stop_profile`. Parser `scripts/perf_parse_trace.py <trace> --bucket-ops`. ⚠️ For
  a DECODE-only breakdown you MUST capture ≥20 decode steps + read the **2nd+** step (the P.1 trace caught
  only 1 prefill + 1 FIRST decode — the device-op cap); device timing IS hardware-accurate under the
  profiler, but discount host `ParseArguments` ~100× (observer effect, proven P.2).
- **TPU MICROBENCHES are 4-host-correct** (`perf_microbench*.sh`). `perf_microbench_decode.py` (host-dispatch
  bench) + `perf_microbench_sparse_attn.py`. Clean-pattern matches `.py` ONLY (the `.sh` shares the prefix).

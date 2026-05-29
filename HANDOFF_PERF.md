# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign (v6e-16)

> **Phase = PERFORMANCE.** Make `vllm serve DeepSeek-V4-Flash` DECODE + PREFILL FAST on the v6e-16
> slice WITHOUT breaking the correctness GATE. Durable slice ops + pitfalls: `CLAUDE.md`. The FIT
> milestone (256 routed experts kept FP4-compressed; `MAX_SEQS=1`) is DONE and is a GIVEN — history in
> `HANDOFF_QUANT.md`. S1 determinism history: `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29 — P.2):** ⚠️ **The "decode bottleneck DECOMPOSED" profile (P.1) was an
> ARTIFACT — the decode bottleneck is NOT yet decomposed.** Re-examined this session: the P.1 trace
> captured only ONE decode step (the FIRST after prefill) under an ACTIVE profiler. The headline
> "~129 ms/step JAX host dispatch (2× ~59 ms `ParseArguments`)" is **profiler OBSERVER EFFECT** — the
> profiler bypasses the C++ pjit fast-path and re-walks the ~1,492-leaf arg pytree per dispatch. A new
> UN-profiled microbench (`scripts/perf_microbench_decode.py`) dispatches that same 1,492-leaf sharded
> pytree in **0.5 ms**, and the nnx-`State` container adds only 5–24× (≤~12 ms); ×2 dispatches ⇒
> real steady-state host dispatch is **≤~24 ms = ≤9% of the 277 ms/step (likely <4%)**. ⇒ **L1
> (fuse/trim host dispatch) is DEMOTED to a ≤4% micro-bank — it was RANK 1 on a phantom.** The P.1
> device numbers (38 ms "compute", 208/466 ms spans) were FIRST-execution program-load, not steady-
> state. **What IS solid: 277 ms/step decode wall (non-profiled two-point fit).** So ≥253 ms/step
> (≥91%) is device-execution + Ray-aDAG round-trip, **UNDECOMPOSED**. Engine down; slice clean; guardians alive.

---

## GATE — regression bar, NON-NEGOTIABLE (full mechanics in `CLAUDE.md` §CORRECTNESS GATE)
- **correct Fibonacci** `21,34,55,89,144,233,377,610` + **N=2 md5 `3069e80b` byte-identical ×2 fresh
  engines** (`scripts/s1_probe2.py 2`) + `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` rc=0.
- A numerics-changing fix MAY shift the md5 → re-establish a NEW ref + confirm identical ×2 engines +
  correct Fib. Do NOT gate on the long-tail md5. **`MAX_SEQS=1` is PINNED** (concurrent decode S1-broken).
- This session changed NO model/serving code (only new analysis scripts + docs) ⇒ GATE preserved by
  construction; last gate-pass was P.1 (N=2 md5 `3069e80b`, FIB correct). Re-confirm on the next smoke.

---

## ⇒ NEXT ACTION — FAITHFULLY decompose the ≥253 ms/step steady-state decode (the real bottleneck)
P.1 mis-measured this. Get the REAL steady-state host-vs-device-vs-roundtrip split — TWO cheap-ish ways:
1. **Non-profiler worker wall-timers (preferred; un-perturbed, no observer effect).** Add env-gated
   (`V4_DECODE_TIMERS=1`) `time.perf_counter()` reads in the WORKER's decode path
   `runner/tpu_runner.py::_execute_model` — stamps at: enter, after `model_fn` (:880, async return =
   host dispatch), after `_select_from_array_fn` (:918), after `compute_logits_fn` (:920), after the
   sampling `device_get` (:940+ — the one blocking point). Print per-step for steps ~5–15; read the
   WORKER ray log. `t(after compute_logits) − t(enter)` = host-dispatch sum (expect ≤~24 ms ⇒ confirms
   L1 dead); `t(after sample) − t(after compute_logits)` = device + round-trip (expect ~250 ms). Tiny,
   GATE-irrelevant edit (timestamps only, no numerics change). ONE smoke also re-confirms the GATE for free.
2. **Multi-step profiler re-capture.** P.1's recipe captured only 1 step (device-op cap). Fix it to
   record ≥20 decode steps and read the **2nd+** step (device timing IS hardware-accurate even under the
   profiler; DISCOUNT the host `ParseArguments` ~100× per the microbench). Recipe in §THE P.1 PROFILE.
3. THEN rank the real lever: **L2 multi-step on-device decode** (if round-trip/dispatch-bound) vs
   on-device compute/HBM work (if device-compute-bound). Do NOT pick a lever before this split is real.

---

## THE ROADMAP (re-ranked P.2 — every item clears the GATE)
1. **[PROFILE] Faithfully decompose the ≥253 ms/step steady-state decode** (NEXT ACTION). The campaign's
   actual bottleneck lives here and is currently UNKNOWN. No host/device lever is justified until this lands. *M.*
2. **[L2-multistep] Multi-step on-device decode** — generate K tokens per dispatch (on-device sampling +
   KV-update loop) so the per-step device round-trip is paid once per K tokens. Still the leading
   STRUCTURAL lever IF the ≥253 ms is round-trip/dispatch-bound — **re-confirm with the profile first.**
   Big change, high risk (decode loop + S1 replicate boundary + sampling feedback). *L · risk HIGH.*
3. **[dtype] bf16-in / fp32-accumulate** at 2 sites: shared expert `deepseek_v4_moe.py:186-187` +
   attention `_linear` `deepseek_v4_attention.py:514` (KEEP the `|r|<1e8` clamp; exact c78ecb96 pattern).
   ROI pending the faithful device profile. *S · risk low-med · md5 may shift.*
4. **[nnx-preflatten] ≤4% micro-bank, NO md5 risk:** pass pre-flattened `state` leaves + a cached
   `treedef` (plain tuple) to `run_model`/`run_compute_logits`, `tree_unflatten` inside the jit
   (`model_loader.py:351-366`; `nnx.merge` stays — it's free on cache-hit). Kills the nnx-`State`
   per-call flatten (5–24× a plain list, ~2–12 ms/step). Verified-cheap by the nnx agent; low priority. *S.*
5. **[5-cleanup] Phase 5 diff-shrink — remove `_v4_nan_tripwire`** (37 sites + def + `smoke.sh:81/116`).
   Edit `.py` AND `.sh` TOGETHER (Pitfall #0). KEEP the `_linear` clamp + `compute_logits` nan_to_num.
   Cosmetic, zero perf; the documented fallback when hard levers stall. *S · risk low.*

---

## DO-NOT-RETRY (dead ends — do NOT burn a smoke; ★ = killed/added this session P.2)
1. ★ **[L1] Fuse `compute_logits` into `run_model` / trim the arg-pytree leaf count to cut host
   dispatch — PHANTOM.** The "~129 ms host dispatch" was profiler observer effect on a single first-step
   trace. Un-profiled dispatch of the 1,492-leaf sharded `state` = **0.5 ms** (`perf_microbench_decode.py`);
   nnx-`State` adds ≤24× → ≤~12 ms; ×2 ≤~24 ms = ≤9% of wall (likely <4%). The fusion is structurally
   feasible (no host blocker; audit confirmed) and the nnx-preflatten trim is a clean ≤4% bank (roadmap
   #4) — but neither is the lever. Do NOT chase host dispatch as a major win.
2. **Async scheduling** — P.1 measured ≈2% (driver 98% blocked in aDAG `get()`); the prize is tiny and it
   perturbs all-reduce ordering. (Caveat: derived from the single-first-step capture; re-confirm IF the
   faithful profile shows large overlappable host slack — unlikely.)
3. **Collective fusion / `pick_partition_spec` axis flip / all-reduce levers** — collectives are tiny
   8–13 µs ops on the 4×4 single ICI torus (topology fact, not just the trace). P.1's "2.2 ms/step"
   absolute is from the single-first-step capture; the topology argument stands regardless.
4. **In-trace FP4→bf16 dequant on the PREFILL/sharded path** — `CompileTimeHbmOom` (Q.11). Decode dequant is LOCAL-only.
5. **Native typed `float4_e2m1fn` rhs to `gmm_v2`** — `MosaicError` on v6e (needs v7). fp8-codes is the answer.
6. **`MAX_SEQS>1` concurrent decode** — CONFIRMED BROKEN (Q.15): silent garbage on 3/4 reqs. Hard-pinned `=1`.
7. **Un-replicate / reshard the decode activation** (move `_v4_decode_replicate`) — re-opens S1 + Pitfall #5.
8. **`lax.scan` over decode layers / removing `_v4_anchor_output_buffers`** — layers already fuse into ONE
   jit; the `optimization_barrier` IS the S1 write-elision fix.
9. **Removing the `_linear |r|<1e8` clamp or `compute_logits nan_to_num`** — both load-bearing (S1 + Q.15).
10. **Indexer `top_k`** — DEFERRED at MAX_LEN≤4096 (small device share); approx_max_k flips FIB. Re-rank
    only if a faithful longer-ctx trace shows T-scaling.

---

## VERIFIED FACTS / cheap tiers (don't re-derive)
- **The decode wall is real:** non-profiled two-point fit (N=20 5.18 s, N=40 10.72 s) = **0.277 s/tok
  (277 ms/step), MAX_LEN=4096, MAX_SEQS=1**. TRUSTED (no profiler). The per-step SPLIT is NOT known.
- **Host dispatch is small:** `perf_microbench_decode.py --sweep` (4-host, real 16-chip mesh) = **0.5 ms
  for 1,492 sharded leaves, ~0.3 µs/leaf LINEAR**. nnx-`State` flatten adds 5–24× (nnx agent, CPU). So
  un-profiled per-dispatch host cost ≤~12 ms; ×2 ≤~24 ms. Profiler inflates this to ~59 ms (observer effect).
- **Two jit dispatches/step** (`run_model` `model_loader.py:353` + `run_compute_logits` `:385`); both take
  the full ~1,492-leaf weight `state` as an EXPLICIT arg. Decode flow is fully on-device:
  `run_model → hidden → _select_from_array_fn gather (tpu_runner.py:918) → compute_logits → logits`. Fusing
  them has NO host blocker (audit) but saves only one ≤12 ms dispatch — not worth the md5 re-baseline.
- **`compute_logits` is at `deepseek_v4.py:1992`** (rms_norm `final_norm_w` → fp32 `@ head_w.T` →
  nan_to_num :2005; head_w genuinely fp32). CLAUDE.md's `:2042/:2055` was stale.
- **CPU torch oracle** `scripts/s1_cpu_repro_v4flash.py both` = Tier-1 math/NaN check (no md5/determinism).
- **THE P.1 PROFILE (kept for the re-capture recipe ONLY — its decode SPLIT is an artifact):** profiled
  smoke (`V4_PROFILER_ARGS=…torch`), `/start_profile` → `s1_probe2.py 20` → `/stop_profile`. Trace:
  `/home/enyouki/v4_traces/plugins/profile/2026_05_29_09_57_06/t1v-n-8f95c921-w-0.trace.json.gz` (worker;
  captured only 1 prefill + 1 FIRST decode — the device-op cap). Parser:
  `scripts/perf_parse_trace.py <trace> --bucket-ops`. ⚠️ MUST capture ≥20 decode steps + read the 2nd+
  step for steady-state; the profiler inflates host `ParseArguments` ~100× (discount it). Device-op
  RELATIVE breakdown (probably stable across steps): copy/transpose 48% (`convert_bitcast_fusion` =
  FP4→fp8 expert dequant), gather 15%, matmul 13%, sparse_attn-Mosaic 5%, top_k 5%, all-reduce 1.3%.
- **TPU MICROBENCHES are 4-host-correct** (`perf_microbench*.sh`; the "8 hosts" prose is stale, launch is
  dynamic). `perf_microbench_decode.sh` clean-pattern matches `.py` ONLY (the `.sh` shares the prefix —
  a bare pattern SIGKILLs the launcher). `perf_microbench_sparse_attn.py` = the sparse-attn op bench.

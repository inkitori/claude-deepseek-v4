# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign (v6e-16)

> **Phase = PERFORMANCE.** Make `vllm serve DeepSeek-V4-Flash` DECODE + PREFILL FAST on the v6e-16
> slice WITHOUT breaking the correctness GATE. Durable slice ops + pitfalls: `CLAUDE.md`. The FIT
> milestone (256 routed experts kept FP4-compressed; `MAX_SEQS=1`) is DONE and is a GIVEN — history in
> `HANDOFF_QUANT.md`. S1 determinism history: `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29 — P.4): nnx-preflatten LANDED + GATED.** The two hot decode jits
> (`run_model` + `run_compute_logits`) no longer re-walk the 1,492-leaf nnx-`State` on every dispatch:
> the weight `state` is flattened ONCE at load, the bare leaf list is passed to the jits, and the
> `State` is rebuilt inside the trace via a cached `treedef` (`model_loader.py:353-367,380-441`).
> Measured (V4_DECODE_TIMERS, 97 steady steps): **host-dispatch 56.05 → 8.2 ms/step** (fwd_disp
> 30.7→6.7, logits_disp 25.15→1.2 — the ~20 ms/dispatch `State`-flatten is GONE, exactly as predicted).
> **Worker decode wall 239 → 216.3 ms/step (−22.7, −9.5%)**; the end-to-end 2-pt driver fit is
> **277 → 220 ms/step (−20.6%, stable ×3)**. Only ~half the 48 ms host saving converts to wall:
> **device_wait 183 → 208 ms** because host dispatch was partly OVERLAPPED with device exec (the host
> now reaches `device_get` ~48 ms sooner, so less device work hides under it — device COMPUTE is
> unchanged). GATE re-confirmed: md5 `3069e80b` ×2 fresh engines, correct FIB, smoke_check rc=0.

---

## GATE — regression bar, NON-NEGOTIABLE (full mechanics in `CLAUDE.md` §CORRECTNESS GATE)
- **correct Fibonacci** `21,34,55,89,144,233,377,610` + **N=2 md5 `3069e80b` byte-identical ×2 fresh
  engines** (`scripts/s1_probe2.py 2`) + `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` rc=0.
- A numerics-changing fix MAY shift the md5 → re-establish a NEW ref + confirm identical ×2 engines +
  correct Fib. Do NOT gate on the long-tail md5. **`MAX_SEQS=1` is PINNED** (concurrent decode S1-broken).
- P.4 preflatten is numerics-PRESERVING (round-trip is lossless: nnx.State/Variable are registered
  pytrees, leaves are bare arrays, metadata lives in the treedef) ⇒ md5 unchanged (`3069e80b`),
  re-confirmed ×2 fresh engines + correct FIB + smoke_check rc=0.

---

## ⇒ NEXT ACTION — DECODE-ONLY device-op breakdown of the ~208 ms device_wait (now ~96% of the wall)
Host dispatch is now trimmed; **device execution is the whole game.** Get a decode-only device-op
breakdown (multi-step profiler, read the **2nd+** decode step — recipe in §THE P.1 PROFILE) to CONFIRM
the attribution below before touching the MoE path. First-principles analysis (P.4, two agents) already
narrows it sharply — the device bucket is **NOT the fundamental HBM floor** and is **NOT** compute /
collectives / launch / indexer-top_k:
- **HBM floor (FP4 read once, dequant fused into the matmul) ≈ 5.5 ms/step** (per-chip resident weights
  ≈ 9.0 GiB: 8.57 experts-FP4 + 0.45 dense; v6e HBM 1638 GiB/s). The measured ~183 ms device compute is
  **~33× that floor** — there is large headroom.
- **The dominant device cost is the dense MoE decode path** (`deepseek_v4_moe.py:296-318`): it bf16-
  **dequants ALL 16 local FP4 experts IN-TRACE every token** (`_dequant_fp4_experts` :37-56) and does a
  dense einsum over the full E axis, masking post-hoc. Materializing the bf16 operands writes+re-reads
  ~78 GiB/chip/token ≈ **49 ms+ ideal** (a ~10× amplifier over the 5.5 ms FP4-once floor), made worse by
  N=1 matrix-VECTOR MXU starvation (~1/128 tile utilization) reading those bf16 weights at low effective
  BW. **This is the lever.**
- **NOT worth chasing** (P.4 first-principles, ≤1% each): collectives (~0.3–2 ms on the 4×4 ICI torus),
  indexer `lax.top_k` (µs-scale — it's over T_index = MAX_LEN/4 = 1024 at 4096, NOT raw T; scales
  ∝MAX_LEN so a lever only far beyond 4096), per-layer launch (43 layers fuse into ONE jit), raw FLOP
  (~0.04 ms ideal — N=1 is memory-bound, not compute-bound).

**LEVERS (rank after the profiler confirms):** (a) **fuse FP4→bf16 dequant INTO the gmm/einsum** — never
materialize the bf16 operand; cuts the ~10× HBM amplifier toward the 5.5 ms floor. Likely S1-SAFE (no
token-axis gather/reshard — it's a local dequant change). (b) gather only the top-k (6) selected experts
vs all 16 local — a further ~2.6× but **re-opens S1 / Pitfall #5** (decode-path gather). Try (a) first.

---

## THE ROADMAP (re-ranked P.4 — every item clears the GATE)
1. **[device-breakdown → MoE-dequant-fuse]** The NEXT ACTION above. Profile to confirm the MoE-dequant
   attribution of the ~208 ms device_wait, then fuse the FP4→bf16 dequant into the matmul (don't
   materialize bf16). The single biggest remaining lever (66%→~96% of the wall). *M–L · md5 may shift
   (re-baseline if so).*
2. **[dtype] bf16-in / fp32-accumulate** — decode MoE is ALREADY bf16/fp32 (PERF 3.1). Remaining site:
   attention `_linear` `deepseek_v4_attention.py:514` (KEEP the `|r|<1e8` clamp). Small; ROI pending #1. *S.*
3. **[5-cleanup] Phase 5 diff-shrink — remove `_v4_nan_tripwire`** (37 sites + def + `smoke.sh:81/116`).
   Edit `.py` AND `.sh` TOGETHER (Pitfall #0). KEEP the `_linear` clamp + `compute_logits` nan_to_num.
   Cosmetic, zero perf; the documented fallback when hard levers stall. *S · risk low.*
4. **[L2-multistep] Multi-step on-device decode** — DEMOTED: with host-dispatch now 8 ms (was 56), the
   prize it amortizes (host + ~aDAG) is much smaller; it does NOT cut the ~183 ms device compute/token.
   Big change, high risk (S1 replicate boundary). Reconsider only if #1 stalls. *L · risk HIGH.*

---

## DO-NOT-RETRY (dead ends — do NOT burn a smoke; ★ = updated/added P.4)
1. ★ **nnx-preflatten — DONE (P.4).** Host-dispatch 56→8 ms/step, GATED. Do not re-attempt; the
   `State`-flatten per dispatch is eliminated. Further host-dispatch trim (the residual ~6.7 ms fwd_disp
   is `_prepare_inputs`+embeds+enqueue, NOT flatten) is sub-3% and not worth it.
2. ★ **`scripts/perf_microbench_decode.py` UNDER-models the dispatch cost** — it benchmarks a FLAT leaf
   list (= the POST-preflatten floor, ~cheap), NOT the nested 1,492-leaf dataclass `State`. The real
   per-dispatch flatten is ~20 ms (measured on the real structure via `make_abstract_transformer_params`
   + `jax.tree_util.tree_flatten`, data-independent — see VERIFIED FACTS). This is why P.2's microbench
   wrongly "falsified" the host lever. For State-flatten cost, measure the REAL tree, not a flat list.
3. **Decode "48% copy/transpose" device cost — that was PREFILL** (gmm_v2 rhs-prep `swapaxes`
   `deepseek_v4_moe.py:82`). DECODE dequant emits NO transpose.
4. **Async scheduling** — DISABLED (RayDistributedExecutor forces `async_scheduling=False`); the sync
   `device_get` (`tpu_runner.py:1112`) is the live block.
5. **Collective fusion / `pick_partition_spec` axis flip / all-reduce levers** — collectives are ~0.3–2
   ms/step total (≤1%) on the 4×4 single ICI torus (P.4 first-principles + device trace ~1.3%).
6. **In-trace FP4→bf16 dequant on the PREFILL/sharded path** — `CompileTimeHbmOom` (Q.11). (NOTE: the
   roadmap-#1 dequant-FUSE is a DECODE-LOCAL change, different from this prefill OOM.)
7. **Native typed `float4_e2m1fn` rhs to `gmm_v2`** — `MosaicError` on v6e (needs v7). fp8-codes is the answer.
8. **`MAX_SEQS>1` concurrent decode** — CONFIRMED BROKEN (Q.15): silent garbage on 3/4 reqs. Hard-pinned `=1`.
9. **Un-replicate / reshard the decode activation** (move `_v4_decode_replicate`) — re-opens S1 + Pitfall #5.
10. **`lax.scan` over decode layers / removing `_v4_anchor_output_buffers`** — layers already fuse into ONE
    jit; the `optimization_barrier` IS the S1 write-elision fix.
11. **Removing the `_linear |r|<1e8` clamp or `compute_logits nan_to_num`** — both load-bearing (S1 + Q.15).
12. **Indexer `top_k`** — µs-scale at MAX_LEN≤4096 (top_k over MAX_LEN/4=1024). Re-rank only if a
    decode-only device breakdown shows T-scaling (it grows ∝MAX_LEN).

---

## VERIFIED FACTS / cheap tiers (don't re-derive)
- **Decode per-step split (P.4 — V4_DECODE_TIMERS, 97 steady steps, MAX_SEQS=1 MAX_LEN=4096):**
  fwd_disp **6.7** / logits_disp **1.2** / samp_disp **0.2** / device_wait **207.9** / post 0.1;
  **worker wall 216.3 ms** (was 239). host-dispatch **8.2 ms** (was 56.05). 2-pt driver fit **220 ms/step**
  (N=20 4.72 s, N=40 9.12 s; was 277). device COMPUTE unchanged (~183 ms); device_wait grew +25 because
  exposed host latency no longer overlaps. All 4 TP workers identical.
- **`V4_DECODE_TIMERS` harvest:** smoke with `V4_DECODE_TIMERS=1`; harvest ONLY fresh lines (the ray
  session persists across smokes — stale lines lurk): `T0=$(date +%s)`; fire a long FIB (`s1_probe2.py
  100`); `find /tmp/ray-vllm/ -type f -newermt "@$T0" | xargs grep -h '\[V4DT\]'`. Line format:
  `ntok=.. fwd_disp=.. logits_disp=.. samp_disp=.. device_wait=.. post=.. wall=..` (ms). **decode =
  `ntok=32`** (1 tok padded). Drop first ~3 + `wall>500` (per-shape recompile) outliers; take medians.
  Brackets (`tpu_runner.py`): fwd_disp 822–903 = `_prepare_inputs`+embeds+**`model_fn` dispatch**;
  logits_disp 903–937 = `_select`+**`compute_logits_fn` dispatch**; device_wait 1111–1114 = the ONE
  `jax.device_get` block; no block_until_ready inside the disp brackets.
- **Real nnx.State flatten cost (off-slice, faithful):** `make_abstract_transformer_params(cfg)` (no GCS,
  no slice; cfg from the HF `config.json`) builds the structurally-real 1,492-leaf dataclass tree;
  `jax.tree_util.tree_flatten` of it ≈ **20 ms on CPU** (data-INDEPENDENT — so it's the per-dispatch cost
  the worker paid). FP4 experts are plain array dataclass fields (NO qwix/QArray), 1 leaf each. A SHALLOW
  synthetic nnx model under-measures this (~1 ms) — must use the real dataclass nesting.
- **HBM floor for N=1 decode ≈ 5.5 ms** (9.0 GiB/chip resident weights ÷ 1638 GiB/s v6e). The dense MoE
  in-trace bf16 dequant of all 16 local experts amplifies HBM traffic ~10× (~78 GiB/chip/token ≈ 49 ms+)
  — the dominant ~183 ms device cost. See NEXT ACTION.
- **CPU torch oracle** `scripts/s1_cpu_repro_v4flash.py both` = Tier-1 math/NaN check; it loads a
  TRUNCATED config (4 layers/8 experts) via `make_abstract_transformer_params`, NOT nnx — does NOT
  exercise the `model_loader.py` dispatch path.
- **THE P.1 PROFILE (re-capture recipe ONLY):** profiled smoke (`V4_PROFILER_ARGS=…torch`),
  `/start_profile` → `s1_probe2.py 20` → `/stop_profile`. Parser `scripts/perf_parse_trace.py <trace>
  --bucket-ops`. For a DECODE-only breakdown capture ≥20 decode steps + read the **2nd+** step; device
  timing IS hardware-accurate under the profiler, but discount host `ParseArguments` ~100× (observer effect).

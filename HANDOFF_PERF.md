# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign (v6e-16)

> **Phase = PERFORMANCE.** Make `vllm serve DeepSeek-V4-Flash` DECODE + PREFILL FAST on the v6e-16
> slice WITHOUT breaking the correctness GATE. Durable slice ops + pitfalls: `CLAUDE.md`. The FIT
> milestone (256 routed experts kept FP4-compressed; `MAX_SEQS=1`) is DONE and is a GIVEN — history in
> `HANDOFF_QUANT.md`. S1 determinism history: `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29):** 🔬 **PROFILE DONE — decode bottleneck fully DECOMPOSED on v6e-16,
> and it OVERTURNS the old roadmap.** Decode = **0.277 s/tok (3.6 tok/s)** — the old "0.43 tok/s /
> 2.3 s/tok" baseline was STALE (v6e-32). Measured per-step (~277 ms): **~129 ms JAX host dispatch**
> (TWO jit dispatches/step — `run_model` + `run_compute_logits` — each re-parsing a big arg pytree:
> `PjitFunction.ParseArguments` ≈59 ms each, HOST CPU, serial on the critical path) **+ ~147 ms
> blocking `device_get`** (of which only **38 ms is real device compute**; the other ~109–169 ms is
> async-dispatch / module-open round-trip latency). Device decode is **DENSE** (38 ms, 99% busy, zero
> internal stall). ⇒ **async-scheduling ≈2% (DEMOTED), collective-fusion DEAD (2.2 ms/step), on-device
> compute opts <7% (device already dense).** NEW LEVERS: **(1) fuse `compute_logits` into `run_model`
> + trim the `run_model` jit arg pytree** (~47% of wall, HOST-side, lower S1 risk) = **NEW RANK 1**;
> **(2) multi-step on-device decode** (~53%, big change). Engine reset; slice clean; guardians alive.

---

## GATE — regression bar, NON-NEGOTIABLE (full mechanics in `CLAUDE.md` §CORRECTNESS GATE)
- **correct Fibonacci** `21,34,55,89,144,233,377,610` + **N=2 md5 `3069e80b` byte-identical ×2 fresh
  engines** (`scripts/s1_probe2.py 2`) + `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` rc=0.
- A numerics-changing fix MAY shift the md5 → re-establish a NEW ref + confirm identical ×2 engines +
  correct Fib. Do NOT gate on the long-tail md5. **`MAX_SEQS=1` is PINNED** (concurrent decode S1-broken).
- ✅ **Re-confirmed this session on a fresh engine** (no code change): N=2 md5 `3069e80b`, FIB correct
  to `…610, 987, …, 32951280099` at N=200. So the profile below was taken on a healthy, gate-passing engine.

---

## ⇒ NEXT ACTION — confirm + size the ~129 ms JAX-dispatch cost (cheap; no smoke)
The ~129 ms `ParseArguments` is **47% of wall** and HOST-side (doesn't touch device numerics → lower
S1 risk than lever 2). Confirm it's steady-state (not a first-step/profiler artifact — the agent
flagged this; our NON-profiled wall of 0.277 s/tok says the 277 ms is real, but the *split* needs one
confirmation) and identify the exact fix, WITHOUT a full smoke:
1. **Code audit (no slice):** in `models/jax/deepseek_v4.py` + `model_loader.py:364` (`run_model` build),
   find (a) WHY there are 2 dispatches/step — is `compute_logits` (`:2042`) a separate jitted call from
   `deepseek_v4_run_with_decode_state`, and can it be fused into the one `run_model` program? (b) WHY
   `ParseArguments` is ~59 ms/call — how big is the `run_model` arg pytree (are all weights / KV / state
   passed as explicit traced args every step vs closed-over)? A huge flat arg list = per-call pytree
   flatten + dispatch-cache-key hashing = the 59 ms. Trimming/closing-over args, or `donate`, cuts it.
3. **Microbench (rank-4 enabler; real mesh, synthetic inputs, NO full load, ~1 min):** build
   `scripts/perf_microbench_decode.py` (clone `perf_microbench_sparse_attn.py`; ⚠️ fix its STALE v6e-32
   "8 hosts" fan-out → 4 hosts first). Dispatch a decode-shaped `run_model` and time **host-dispatch
   (ParseArguments) vs `block_until_ready` (device round-trip) vs op-compute** separately. Confirms the
   ~129 ms host-dispatch is steady-state and sizes lever 1 vs lever 2 before any `.py` edit + smoke.
4. THEN implement lever 1 (fuse `compute_logits` into `run_model`; trim args) → CPU oracle → 1 smoke ×2
   (md5 may shift if fusion reorders fp ops → re-baseline). Bank ~one dispatch (~63 ms) + reduced parse.

---

## THE PROFILE — measured 2026-05-29 (v6e-16, warm cache, MAX_LEN=4096, MAX_SEQS=1)
**How:** profiled smoke (`V4_PROFILER_ARGS=…torch`), warm + FIB spot-check, then a timed decode-only
window (`/start_profile` → `s1_probe2.py 20` → `/stop_profile`). Wall confirmed NON-profiled with a
two-point fit: N=20 5.18 s, N=40 10.72 s ⇒ **marginal 0.277 s/tok**. Traces (head-local, may be cleaned):
`/home/enyouki/v4_traces/plugins/profile/2026_05_29_09_57_06/t1v-n-8f95c921-w-0.trace.json.gz` (worker
device+host; captured 1 prefill + 1 decode — the device-op cap, NOT 20). Parser:
`scripts/perf_parse_trace.py <trace> --bucket-ops` ⇒ `total on-device op time` is a **cross-chip SUM**
(÷ `device lanes parsed` count = 4 → per-chip; watch stderr for `[fallback]`). py-spy on the driver
(sudo; `ptrace_scope=1` needs sudo) worked; **py-spy on a WORKER FAILS** (falls progressively behind
unwinding JAX/XLA+aDAG native stacks → no output) — use the worker's torch host-lane or add timers instead.

**Per decode step (~277 ms), worker `ray::RayWorkerW` host thread, serial:**
| segment | ms | class | lever |
|---|---|---|---|
| `PjitFunction(run_model)` dispatch incl. ParseArguments ~59 | ~66 | HOST CPU | **L1** |
| `PjitFunction(run_compute_logits)` dispatch incl. ParseArguments ~58 | ~63 | HOST CPU | **L1** |
| blocking `device_get`: 38 ms dense compute + ~169 ms module-open-but-device-idle round-trip | ~147 | DEVICE round-trip | **L2** |
| `_prepare_inputs_dp` host→device puts ~1.8 · aDAG channel write/read ~3 · collectives 2.2 | ~7 | negligible | — |

**Device-op breakdown (per chip, % of the dense 38 ms decode + 467 ms prefill window):** copy/transpose
48% (`convert_bitcast_fusion` = FP4→fp8 expert dequant), gather 15%, matmul 13%, sparse_attn-Mosaic 5%,
top_k 5%, **all-reduce 1.3% (2.2 ms — collectives are NEGLIGIBLE on the 4×4 single ICI torus)**.
Driver EngineCore (pid found via `ps grep EngineCore`): MainThread **98% blocked in the Ray aDAG `get()`**
(`shared_memory_channel.read → get_objects`), only ~2% (~5 ms/step) overlappable host work.

---

## THE ROADMAP (re-ranked by the measurement above; every item clears the GATE)
1. **[L1-dispatch] Cut the ~129 ms/step JAX host dispatch** — fuse `compute_logits` into the `run_model`
   jit (1 dispatch not 2 ⇒ −~63 ms) AND shrink `run_model`'s arg pytree so `ParseArguments` (~59 ms)
   stops re-flattening/hashing a huge explicit arg list each step. **~47% of wall, HOST-side (no device
   numerics change ⇒ lower S1 risk).** Gate via NEXT ACTION (audit→microbench→smoke ×2). *M · risk med.*
2. **[L2-multistep] Multi-step on-device decode** — generate K tokens per dispatch (on-device sampling +
   KV update loop) so the ~147 ms device round-trip (only 38 ms is compute) is paid once per K tokens,
   not per token. Targets the other ~53%; floor ≈ 38 ms/tok ⇒ potential multi-× win. **Big change, high
   risk** (touches the decode loop + S1 replicate boundary + sampling feedback). *L · risk HIGH.*
3. **[dtype] bf16-in / fp32-accumulate** at 2 sites (NOT 3 — see below): shared expert
   `deepseek_v4_moe.py:186-187` + attention `_linear` `deepseek_v4_attention.py:514` (KEEP the `|r|<1e8`
   clamp; exact c78ecb96 pattern, drafted+verified this session). `compute_logits:2002` is **NOT
   emulated** (head_w is genuinely fp32) → skip it. **Profile shows device is dense 38 ms ⇒ wall ROI
   <7% until L1/L2 land** — a safe micro-bank, low priority now. *S · risk low-med · md5 may shift.*
4. **[5-cleanup] Phase 5 diff-shrink — remove `_v4_nan_tripwire`** (37 sites + def + `smoke.sh:81/116`).
   Edit `.py` AND `.sh` TOGETHER (Pitfall #0). KEEP the `_linear` clamp + `compute_logits` nan_to_num.
   Cosmetic, zero perf; the documented fallback when hard levers stall. *S · risk low.*

---

## DO-NOT-RETRY (dead ends — do NOT burn a smoke; ★ = newly killed BY MEASUREMENT this session)
1. ★ **Async scheduling (the old "rank-2 10–100× lever") — ≈2%, NOT WORTH IT.** Measured: the EngineCore
   driver is already **98% blocked in the aDAG `get()`** with only ~5 ms/step of overlappable host work,
   so pipelining step k+1's dispatch behind step k's device wait reclaims ~5 ms of 277 ms. (Mechanism
   was real — `RayDistributedExecutor.supports_async_scheduling()` returns False; 1-line override +
   `--async-scheduling` enables it — but the prize is tiny and it perturbs all-reduce ordering ⇒ re-gate.)
2. ★ **Collective fusion / `pick_partition_spec` axis flip / any all-reduce lever — DEAD.** Measured
   collectives = **2.2 ms/step (1.3% device, ~0.8% wall)**, tiny 8–13 µs ops on the 4×4 single ICI torus.
3. ★ **On-device collective-stall or HBM-streaming-gap hunt — RULED OUT.** The decode device program is
   **98.9% busy, dense, zero internal gap >7 µs**. The ~169 ms tail is module-open-but-device-IDLE
   (async round-trip), not an on-device stall.
4. **In-trace FP4→bf16 dequant on the PREFILL/sharded path** — `CompileTimeHbmOom` (Q.11). Decode dequant
   is LOCAL-only; prefill uses fp8 codes.
5. **Native typed `float4_e2m1fn` rhs to `gmm_v2`** — `MosaicError` on v6e (needs v7). fp8-codes is the answer.
6. **`MAX_SEQS>1` concurrent decode** — CONFIRMED BROKEN (Q.15): silent garbage on 3/4 reqs. Hard-pinned `=1`.
7. **Un-replicate / reshard the decode activation** (move `_v4_decode_replicate`) — re-opens S1 + Pitfall #5.
8. **`lax.scan` over decode layers / removing `_v4_anchor_output_buffers`** — layers already fuse into ONE
   jit; scan cuts compile-time only. The `optimization_barrier` IS the S1 write-elision fix.
9. **Removing the `_linear |r|<1e8` clamp or `compute_logits nan_to_num`** — both load-bearing (S1 + Q.15).
10. **Indexer `top_k` (approx_max_k / valid-prefix)** — DEFERRED at MAX_LEN≤4096 (~5 ms/5% device, tiny);
    approx_max_k flips FIB (gathers sparse-attn KV). Re-rank only if a longer-ctx trace shows T-scaling.

---

## VERIFIED FACTS / cheap tiers (don't re-derive)
- **The wall is real, not a profiler artifact:** NON-profiled two-point fit = 0.277 s/tok. The 277 ms
  splits ~129 ms host-dispatch (serial) + ~147 ms device_get (38 ms compute + ~109 ms round-trip).
- **Two jit dispatches per decode step** (`run_model` + `run_compute_logits`), NOT one — fusing them is L1.
- **Prefill is ~467 ms device-compute/chip** (dense), 12× the decode-compute — a separate later target.
- **CPU torch oracle** `scripts/s1_cpu_repro_v4flash.py both` = Tier-1 math/NaN check (can't see md5/determinism).
  **Drafted+verified dtype edits** for L1-adjacent site 3 are in commit msg / agent notes — re-derive if needed.
- Before reusing any multi-host microbench, fix the stale v6e-32 "8 hosts" launcher in
  `perf_microbench_sparse_attn.py` / `perf_microbench.sh` → 4 hosts.

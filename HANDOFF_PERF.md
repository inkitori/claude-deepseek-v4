# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign (v6e-16)

> **Phase = PERFORMANCE.** Make `vllm serve DeepSeek-V4-Flash` DECODE + PREFILL FAST on the v6e-16
> slice WITHOUT breaking the correctness GATE. Durable slice ops + pitfalls: `CLAUDE.md`. The FIT
> milestone (loading the 256 routed experts FP4-compressed; `MAX_SEQS=1`) is DONE and is a GIVEN —
> history in `HANDOFF_QUANT.md`. Other history: `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29):** 🏁 **PERF reopened on v6e-16.** Decode ≈ **0.43 tok/s** (~2.3 s/token).
> ⚠️ **The bottleneck is UNRESOLVED and the OLD profile is DEAD** — it was captured on v6e-32 / TP=32 /
> PRE-Phase-3.1 / bf16 experts; **every % anywhere below is from that dead config.** Leading hypothesis
> from a 16-agent static analysis (2026-05-29) = **HOST / DISPATCH overhead, NOT compute or collectives**:
> ~2.3 s/token wall vs a ~7.5 ms/token HBM-stream floor and a ~50–300 ms device-busy ceiling ⇒ a 10–40×
> gap that can only be non-device time (Python/PJRT enqueue + Ray compiled-DAG broadcast + blocking
> per-step `get`, with async scheduling force-OFF under Ray). **BUT this is UNVERIFIED on hardware.**
> **THE ONE NEXT ACTION = re-capture a v6e-16 decode-only profiler trace (zero code, zero S1 risk) and
> read device-busy-ms vs wall** — that single measurement decides the whole ranking. **Do NOT land a
> `.py`-edit smoke before it.**

---

## GATE — regression bar, NON-NEGOTIABLE (full mechanics in `CLAUDE.md` §CORRECTNESS GATE)
- **correct Fibonacci** `21,34,55,89,144,233,377,610` + **N=2 md5 `3069e80b` byte-identical ×2 fresh
  engines** (`scripts/s1_probe2.py 2`) + `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` rc=0.
- A numerics-changing fix MAY shift the md5 → re-establish a NEW ref + confirm identical ×2 engines +
  correct Fib. Do NOT gate on the long-tail md5 (`s1_probe2.py 20`+, nondeterministic at temp=0).
- **`MAX_SEQS=1` is PINNED** (concurrent decode is S1-broken, Q.15). READ the decode text — "contains
  Paris" is a false positive (can EOS at tok 1).

---

## ⇒ NEXT ACTION — re-capture the v6e-16 decode profile (rank 1; decides everything)
Cheapest, zero-S1-risk, ranking-deciding. It is a **serve flag, not a `.py` edit** → no sync, no
cache clear, no S1 exposure. (No engine is currently serving — `curl :18081` is refused; the reset is
clean-slate hygiene for a stale pidfile + `/tmp/s1_smoke_launch.lock`.)

1. `scripts/full_slice_v4_reset.sh`   2. `mkdir -p /home/enyouki/v4_traces`
3. Launch with the profiler (hook wired at `full_slice_v4_smoke.sh:144`; KEEP `MAX_SEQS=1`; xla_cache is
   warm ×4 hosts so ~6 min, but tolerate a one-time ~325s first-decode recompile):
   ```bash
   V4_PROFILER_ARGS='--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/home/enyouki/v4_traces' \
     MAX_LEN=4096 bash scripts/full_slice_v4_smoke.sh
   ```
4. After `Application startup complete`: `python3 scripts/s1_probe2.py 2` (warm + FIB spot-check).
5. **Decode-only window, and TIME IT** (the parser does NOT emit wall — you must hand-measure it):
   ```bash
   t0=$(date +%s.%N); curl -sX POST :18081/start_profile
   python3 scripts/s1_probe2.py 20
   curl -sX POST :18081/stop_profile; t1=$(date +%s.%N)   # wall_per_tok = (t1-t0)/20
   ```
6. Parse the head-worker trace (`/home/enyouki/v4_traces/plugins/profile/<ts>/<host>.trace.json.gz`,
   ~750 MB unzipped): `work/vllm_env/bin/python3 scripts/perf_parse_trace.py <trace> --bucket-ops`.
7. **CRITICAL READ:** compare the parser's total **on-device op time (device-busy ms/step)** against the
   hand-measured **wall_per_tok**:
   - **device-busy ≪ wall** → **host/Ray-dispatch-bound CONFIRMED** → next loop = **rank 2 (async
     scheduling)**, size the prize from the host gap.
   - **device-busy ≈ wall** → **on-device** → re-rank by the top op bucket (collectives vs MoE-dequant
     vs top_k) and pursue rank 3/5/6 accordingly.
   - Confirm decode steps actually landed inside the 1M-XLA-op trace cap (the decode-only window handles this).

---

## THE ROADMAP (ranked; almost everything below rank 1 is GATED on that profile)
Cheapest-tier-first; every item clears the GATE. Reserve full smokes (≤1–2/session, COLD = 25–45 min).

1. **[0-profile] Re-capture the v6e-16 decode-only trace** — see NEXT ACTION. *S · no S1 risk · do first.*
2. **[1-dispatch] Enable async scheduling** so step k+1 host/Ray dispatch overlaps step k device compute.
   `ray_executor.py:105` (`max_concurrent_batches`), `:461-467` (compiled-DAG execute + blocking get);
   `vllm/config/vllm.py:830` (Ray ⇒ async OFF); dormant runner path `tpu_runner.py:1049-1090`.
   **GATED on rank-1 showing wall ≫ device-busy.** If host-overhead holds this is the ONLY 10–100× lever.
   *M · risk med-high* (changes token feedback timing → re-prove N=2 md5 ×2; the big form = leaving Ray for
   `MultiprocExecutor` = infra risk on this Ray+guardian slice). Does NOT touch `_v4_decode_replicate`.
3. **[2-dtype] bf16-in / fp32-accumulate** (`preferred_element_type=fp32`) for the remaining fp32-emulated
   matmuls: shared expert (`deepseek_v4_moe.py:186-187`, runs every step via `:422`), attention `_linear`
   (`deepseek_v4_attention.py:516`), `compute_logits` (`deepseek_v4.py:2002`). EXACT Phase-3.1 pattern
   (`c78ecb96`, md5 was unchanged). **Profile-INDEPENDENT — the safest concrete win to bank.** KEEP the
   `_linear |r|<1e8` clamp. CPU oracle → 1 smoke ×2 (md5 may shift → re-baseline). *S · risk low-med.*
4. **[0-enabler] Build `scripts/perf_microbench_decode.py`** (clone `perf_microbench_sparse_attn.py`):
   op#1 psum/all_gather over the real `attn_dp=16` shard at decode payloads (launch- vs bandwidth-bound);
   op#2 `lax.top_k` vs `lax.approx_max_k` swept T∈{1024,4096,16384,131072}; op#3 dense-bf16-einsum-after-
   dequant vs replicated top-6 `gmm_v2` at N∈{1,6,16} + a bf16-vs-gmm max-abs-diff assert. Real mesh,
   synthetic inputs, ~1 min/op, NO full load. Unblocks cheap A/B for ranks 5–8. *M · no S1 risk · build early.*
5. **[3-moe] Fuse FP4→bf16 dequant INTO the decode MoE einsum** so experts stream as FP4 from HBM (~4×
   less expert traffic; stored 192 vs materialized 768 MiB/layer/chip). `deepseek_v4_moe.py:300-303`
   (`_shard_e_first(_dequant_fp4_experts(...))` — the wsc sits between dequant and einsums `:308-316`).
   **GATED on rank-1 (device-busy material) + rank-4 op#3 HLO showing the wsc forces a bf16 temp.** Drop/
   relocate the likely-redundant `_shard_e_first`. *M · risk med* — wsc is on the EXPERT axis (NOT the
   token axis ⇒ Pitfall #5 N/A), but it can change the load-bearing post-einsum all-reduce layout; validate ×2.
6. **[3-collective] Fuse same-contraction down-projs → one matmul → one all-reduce** (load-time weight
   concat; compressor `wkv`+`wgate` share contracting H=4096). `deepseek_v4_attention.py:562-563`, `:800-869`.
   Cuts collective LAUNCH COUNT (~41 logical AR/step on the compressor), **bit-exact (no md5 shift)**.
   **GATED on rank-1 (collectives #1) + rank-4 op#1 (launch-overhead, not bandwidth).** *M · risk low.*
7. **[4-indexer-longctx] Indexer `top_k`** — per-bucket decode JIT (exact, big refactor) OR `approx_max_k`
   (small, gateable, APPROXIMATE). `deepseek_v4_attention.py:659` (`lax.top_k(...,512)` over static
   `T=state_max_seq_len//4`, ×21 layers; jit-blocker = traced seqlen under the monolithic `run_model`).
   **DEFER at MAX_LEN≤4096** (T_CSA≈1024, ~8 valid slots → win negligible); re-rank only if a LONGER-ctx
   trace shows T-scaling dominates. *L · risk HIGH for approx_max_k* — its indices directly gather sparse-attn
   KV (`:198-205`) → can flip FIB; cross-engine determinism unproven → mandatory smoke ×2, abandon if it drifts.
8. **[3-moe] Replicated top-6 `gmm_v2` decode-MoE branch** (replace dense-all-16 + per-step dequant with
   routed gmm over local experts as fp8 codes; reuse `_fp4_rhs_and_scale`). **LARGELY SUBSUMED by rank 5 →
   DEFER.** MXU is negligible at N=1 so the FLOP win is marginal. Keep activation replicated, combine only
   on the post-reduction `[1,dim]` output (Pitfall #5 N/A by construction). *L · risk med · md5 re-baseline.*
9. **[5-cleanup] Phase 5 diff-shrink — remove `_v4_nan_tripwire`** (37 sites + def + module env-read +
   `smoke.sh:81/116`). Cosmetic, ZERO perf delta, near-zero risk (no-op when off, default `'0'`). Edit the
   `.py` AND `.sh` TOGETHER (Pitfall #0: a per-worker module-level env divergence → launch-id halt). **KEEP**
   the `_linear` clamp + `compute_logits` `nan_to_num` (both load-bearing). **The documented fallback when
   hard levers stall**, not a perf lever; still costs a smoke to gate. *S · risk low.*

---

## DO-NOT-RETRY (dead ends — do NOT spend a 25–45 min smoke re-testing these)
1. **Phase-2 `pick_partition_spec` axis-0 (contracting→output) flip** — killed by 3-angle analysis
   (`607629da`): down-projs feed RMSNorm/RoPE/residual needing the full vector ⇒ the flip yields an
   EQUAL-payload all-gather (no win), decode is launch-COUNT-bound at N=1, and the o-path REGRESSES. Weaker
   on the 4×4 single torus. **DO NOT SMOKE.**
2. **In-trace FP4→bf16 dequant on the PREFILL/sharded path** — `CompileTimeHbmOom` (37.32 GiB temp, Q.11
   `bff3eaf4`). This is why decode dequant is LOCAL-only and prefill uses fp8 codes.
3. **Native typed `float4_e2m1fn` rhs to `gmm_v2`** (the "proper" fp4 MXU path) — `MosaicError` on v6e; fp4
   MXU needs TPU v7 (Q.12 `542195d4`). The fp8-codes workaround (Q.13) is the answer.
4. **`MAX_SEQS>1` concurrent decode** — CONFIRMED BROKEN (Q.15 `fc03c1f9`): silent garbage on 3/4 requests.
   Hard-pinned `MAX_SEQS=1`; concurrent-decode determinism is a SEPARATE non-perf workstream.
5. **Un-replicate / in-trace reshard the decode activation** (move the `_v4_decode_replicate` boundary,
   reduce-scatter instead of all-reduce) — re-opens the S1 uninit-HBM corruption AND risks Pitfall #5. The
   replicate boundary is load-bearing.
6. **"Free" valid-prefix-only / arange short-circuit on indexer `top_k`** — REFUTED (`259b57a4`): bit-identical
   math but jit-INVALID (traced seqlen, `dynamic_slice` needs a static size). A static scan-cap W<T silently
   truncates context = a correctness regression.
7. **`lax.scan` over the 43 decode layers / removing `_v4_anchor_output_buffers`** — layers already fuse into
   ONE jitted program (`run_model`, `model_loader.py:364`), so scan cuts compile time only, not wall-clock;
   the `optimization_barrier` IS the S1 write-elision fix. High S1 risk, zero N=1 payoff.
8. **Removing the `_linear |r|<1e8` clamp or `compute_logits nan_to_num`** — both load-bearing (clamp = S1
   seed sanitizer; nan_to_num DID fire in Q.15). Zero perf upside. Leave verbatim.
9. **Re-attacking the sparse-attn gather kernel** — CLOSED (`ace9d576`/`c2557e26`): already 65.8%→0.2%; the
   ~41× isolated microbench gave only ~1.25× e2e. (⚠️ `perf_microbench_sparse_attn.py` hardcodes the STALE
   v6e-32 8-host fan-out — fix that before reusing it.)
10. **`kv_cache_sharding` flip P()→attn_dp** — one-time placement (no recurring collective), B=1 has no clean
    attn_dp dim, residency is fine; resharding is the documented layout-mismatch failure. No win, high risk.
11. **Indexer "keep fp32 end-to-end"** — WRONG: `indexer_kv_cache` is deliberately bf16 (`deepseek_v4.py:583`)
    to save HBM; the `:650` upcast is genuine, not a lossy no-op being undone. No bit-exact win exists.

---

## VERIFIED STRUCTURAL FACTS (don't re-derive these)
- **Decode = ONE monolithic fused jit** (`run_model`, `model_loader.py:364`, `donate_argnums=2`, seqlen
  excluded from `static_argnums`, 43-layer loop trace-UNROLLED) ⇒ NOT per-layer launch-bound; **`lax.scan`
  is a non-lever** for wall-clock (see DO-NOT-RETRY #7).
- **Async scheduling is force-OFF under Ray** (`RayExecutor` doesn't override `supports_async_scheduling`;
  `vllm/config/vllm.py:830` ⇒ `max_concurrent_batches=1`) — this is the rank-2 lever.
- **N=1 floors:** HBM-stream ≈ 7.5 ms/token (≈12.25 GB active weights/chip ÷ 1.64 TB/s); MXU ≈ 0.04–0.6 ms;
  ⇒ device-busy ≤ ~50–300 ms incl. collectives. **~250 logical all-reduces/step**, structural to the S1
  replicate boundary; cheaper per-op on the 4×4 single ICI torus (no DCN, ring 16) than the dead v6e-32 profile.

## ENABLERS / cheap tiers (use before burning a smoke)
- **rank-4 microbench** (above) sizes ranks 5–8 in ~1 min/op. **HLO inspection** of `_dequant_fp4_experts`
  + the 3 einsums decides rank 5 with no smoke. **CPU torch oracle** `scripts/s1_cpu_repro_v4flash.py both`
  is the Tier-1 math/NaN check for any numerics lever (it CANNOT see md5 drift/determinism — never replaces
  the 2-engine smoke). Before reusing any multi-host microbench, fix the stale v6e-32 "8 hosts" launcher
  assumption in `perf_microbench_sparse_attn.py` / `perf_microbench.sh` → 4 hosts.

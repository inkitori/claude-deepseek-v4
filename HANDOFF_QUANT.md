# Handoff — DeepSeek-V4-Flash QUANTIZED-WEIGHT-LOADING campaign (v6e-16)

> **Phase = QUANT / FIT-ON-v6e-16.** Make `vllm serve DeepSeek-V4-Flash` LOAD AND SERVE CORRECTLY on
> **v6e-16** by **NOT dequantizing the FP4 experts to bf16**. Durable slice ops + pitfalls: `CLAUDE.md`.
> Prior campaigns (history): `HANDOFF_PERF.md`, `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29):** 🎯 **GATE PASSED + HARDENED + QUANT AXIS EXHAUSTED.** V4-Flash LOADS +
> FITS + SERVES + DECODES CORRECTLY + DETERMINISTICALLY on v6e-16 (MAX_SEQS=1) with the FP4 experts kept
> compressed — core goal MET (Q.13: FP4 experts → `gmm_v2` as **fp8 codes + per-block rhs_scale**; fp4→fp8
> cast LOSSLESS; v6e Mosaic CANNOT compile the native fp4 unpack), hardened at full context `MAX_LEN=4096`
> (Q.14). **Q.15 (this session): tested the LAST open config — MAX_SEQS>1 (concurrent multi-request decode)
> → CONFIRMED BROKEN** (S1-class uninit-HBM corruption: 3 of 4 concurrent FIB requests return GARBAGE,
> SILENTLY — HTTP 200, no crash). **NOT a quant regression and NOT a quant fix** — it is the S1 fix's known
> scope limit (`_v4_decode_replicate` covers num_reqs==1 only). **MAX_SEQS=1 is the validated production
> config (and was already the pinned constraint).** Every quant-axis config is now gated-working or
> characterized-and-scoped-out ⇒ **the quant axis is DONE. Recommend `touch /tmp/quant_loop_stop`**;
> remaining work (PERF ~0.43 tok/s; concurrent-decode determinism) is SEPARATE non-quant loops. See §NEXT.

---

## GATE — PASSED (v6e-16 baseline; re-confirmed at MAX_LEN=4096 in Q.14)
The old v6e-32 md5 `5bf42256` is DEAD. The NEW v6e-16 baseline + full gate, confirmed at BOTH
`MAX_LEN=256` (Q.13) and `MAX_LEN=4096` (Q.14) — md5s are config-INVARIANT (decode path unchanged):
- **Correct Fibonacci**: `s1_probe2.py 24` → `' 21, 34, 55, 89, 144, 233, 377, 610,'` (md5 `9e91bf45`).
- **N=2 md5 `3069e80b`** (text `' 21'`), **byte-identical across 2 FRESH engines** (the S1-grade
  determinism bar — coherent output alone is NOT proof; intra-engine repeats don't catch the per-process
  uninit-HBM coin flip — you MUST start a 2nd fresh engine). Probe: `python3 scripts/s1_probe2.py N`.
- **`LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0** (64-tok decode: visible_words=30,
  max_word_run=1, ends_clean=1).
- **Q.14 BONUS** (large-N prefill stress): `python3 scripts/quant_longprompt_probe.py 32` → 445-token
  prefill, coherent on-topic summary, byte-identical ×2 fires (md5 `835d3094`).
Re-run this gate after ANY numerics-affecting change; a shift means re-establish + re-confirm ×2 engines.

---

## THE PROBLEM — both HBM hurdles CLEARED ✅
V4-Flash ships natively quantized: dense=FP8, 256 routed experts=**FP4 (MXFP4: codebook ≡
`jnp.float4_e2m1fn`, e8m0 block scale, block 32)**. On-disk routed `experts.{0-255}.{w1,w2,w3}.weight`
= I8 (2 FP4/byte along IN), `.scale` = F8_E8M0; H=4096, I(moe_inter)=2048, 256 experts, 43 layers + 1 MTP.

| stage | scheme | HBM/chip | status |
|---|---|---:|---|
| LOAD | bf16 dequant-at-load (old) | ~542 GiB total | ❌ OOM (dead) |
| LOAD | **FP4 experts kept compressed (Strategy C)** | 9.75 / 31.25 | ✅ cluster-wide |
| FORWARD | FP4→bf16 dequant IN-TRACE (Q.11) | 37.32 temp | ❌ CompileTimeHbmOom (fixed) |
| FORWARD | FP4 as TYPED float4 → gmm_v2 (Q.12) | fits | ❌ v6e Mosaic can't unpack f4E2M1FN (fixed) |
| FORWARD | **FP4 codes → FP8 + rhs_scale → gmm_v2 (Q.13)** | fits | ✅ **compiles + correct + deterministic** |

---

## WHAT LANDED (committed)
**LOAD = Strategy C** (history; do NOT re-litigate): loader keeps the 256 routed experts FP4-compressed
(u8 packed weight + u8 e8m0 scale leaves; host-gathered ⇒ zero reshard collective). Scheckne saga +
divergent-load all RESOLVED (`fb54237b`, `0d8d57fe`, `83a18839`, `ca016156`, … `bff3eaf4`).

**FORWARD = fed FP4 to gmm_v2 (this session).** All in `layers/jax/moe/deepseek_v4_moe.py`:
- `542195d4` (Q.12): new `_fp4_rhs_and_scale` + the two prefill `gmm_v2` calls take `rhs_scale`; dense
  (CPU+decode) path dequants to bf16 LOCALLY. Killed the prefill bf16-dequant OOM. But a TYPED
  `float4_e2m1fn` rhs then hit `MosaicError: Unsupported type 'vector<...xf4E2M1FN>'` (v6e fp4 unpack
  unsupported; fp4 MXU needs TPU v7).
- `0d24d6af` (Q.13, the working fix): `_fp4_rhs_and_scale` casts the fp4 codes to **fp8 e4m3** (e2m1
  codebook ⊂ e4m3 ⇒ lossless) → fp8 rhs `[E,in,out]` + fp32 rhs_scale `[E,in/32,1,out]`, built OUTSIDE
  the shard_map (rank-local; E=axis0 stays sharded) with `with_layout_constraint(Layout((0,1,2)))`
  (mirrors `process_weights/moe_weights.py:256-271`). fp8 is full-byte ⇒ `should_bitcast=False` ⇒ the
  unpack op is never emitted; gmm runs fp8(lhs auto-quant)×fp8(rhs)+per-block scale = native MXFP4.
  The S1-fix `_routed_local` body (all_gather/barrier/owned-mask/psum) is UNTOUCHED.

**HARDENING (Q.14, this session, NO code change):** re-gated at `MAX_LEN=4096` (full model context, 16×
the prior 256), keeping `MAX_SEQS=1`. All gate elements green ×2 fresh engines (FIB correct, N=2 md5
`3069e80b` cross-process identical, smoke_check rc=0) PLUS a 445-token long-prefill probe (coherent,
deterministic) that exercises the Q.13 fp8-gmm path at large N. Residency 10.19 GiB/chip (~21 GiB
headroom) — confirmed the FP4-compressed serving holds far below the prior config. Corrected the HBM
premise (see §NEXT). Logs: `full-slice-v4-smoke-20260529T082830Z.log` (eng#1) / `…084305Z.log` (eng#2).

CPU validation (cheap tier, all green): `scripts/quant_gmm_fp4_orient_check.py` (orientation BYTE-EXACT
vs `_dequant_fp4_experts` in gmm's [g,k,n] layout) + `quant_fp4_dequant_check.py` + `s1_cpu_repro_v4flash.py both`.

---

## <a name="NEXT"></a>⚠️ NEXT ACTION — quant axis EXHAUSTED; recommend stopping the loop
Every config is now either gated-working (MAX_SEQS=1, full context) or characterized-and-scoped-out
(MAX_SEQS>1, below). The blocking goal (load+fit+serve+correct on v6e-16) is COMPLETE + GATED + HARDENED.

**Q.15 FINDING — MAX_SEQS>1 concurrent decode is BROKEN (last open question, now closed).** Tested on a
fresh MAX_SEQS=4 / MAX_LEN=256 engine via `scripts/quant_concurrent_probe.py 4 32`: 4 IDENTICAL FIB
requests fired concurrently genuinely CO-decoded (`Running: 4 reqs` in-log; all returned at 68.3 s vs
7.9 s single). Result: **only 1 of 4 correct; the other 3 returned GARBAGE** (`' etc'`, `','`, `','`) —
SILENTLY (HTTP 200, finish_reason='length', NO crash/NaN log; the `compute_logits` nan_to_num clamp masks
it). Single request correct + byte-stable (N=2 md5 `3069e80b`; N=32 md5 `34660b8b`) BEFORE and AFTER the
batch — **no state pollution, no wedge**. Mechanism = the exact S1 uninit-HBM corruption: `num_reqs>1` ⇒
`_v4_decode_replicate` OFF (it gates on `num_reqs==1`, `tpu_runner.py:1385`) ⇒ activation ATTN_DATA-
sharded across attn_dp=16; but N(=4) < 16 ⇒ MoE takes the **dense einsum** path (gmm needs N≥16,
`deepseek_v4_moe.py:290`), which was built for a REPLICATED activation — a token-sharded N<16 activation
feeds its implicit collective-matmul that reads uninitialised HBM for un-owned tokens. **This is the S1
fix's deliberate scope limit, NOT a quant bug** — and `--max-num-seqs=1` was already the pinned
production constraint for exactly this reason.
⚠️ **PRODUCTION FOOTGUN:** serving with `--max-num-seqs>1` SILENTLY corrupts concurrent requests (no
error surfaced). If MAX_SEQS>1 is ever enabled, add a LOUD startup guard first.

**If concurrent decode is wanted (a SEPARATE S1/determinism workstream, NOT quant):** make the small-N
(N<16) decode path S1-safe under num_reqs>1 — either (a) widen `_v4_decode_replicate` to replicate the
decode activation for num_reqs>1 too (N>1 is NOT the size-1 token-axis gather Pitfall #5 warns Core-halts,
so plausibly safe — but it touches the load-bearing S1 path; validate hard), or (b) force/pad decode to
the gmm path (gmm zero-inits un-owned rows = the determinism lever) — but gmm-for-N=1 collides with the
S1 replication (token-axis gather → Core-halt ×8). Either needs ≥2 cold smokes + the 2-engine determinism
gate + re-confirming the single-request gate still passes. RISKY; out of quant scope. Validate any fix
with `quant_concurrent_probe.py` (all-4-correct = fixed).

**PERF (the other separate loop, see `HANDOFF_PERF.md`):** decode `observed_tps≈0.43`. Cheap entry = a
decode-step microbench. The GATE must still hold after any perf change.

**HBM premise (durable):** `31.25 GiB` is the per-chip BUDGET, not residency. Live residency ~9.75–10.2
GiB/chip ⇒ ~21 GiB headroom. The decode dense-branch bf16 dequant is WEIGHT-ONLY + N-INDEPENDENT (only
the 16 LOCAL E-sharded experts, ~0.75 GiB/chip/layer) ⇒ cannot OOM at any realistic config.

---

## VALIDATION TIERS (cheapest first) — unchanged, but note gmm_v2 is TPU-only
1. **CPU (~40s, no slice):** `JAX_PLATFORMS=cpu PYTHONPATH=work/tpu-inference:work/vllm
   work/vllm_env/bin/python3 scripts/quant_gmm_fp4_orient_check.py` (orientation/shape — gmm_v2 CANNOT
   run on CPU, it calls `get_tpu_info()` at trace time; the orient check replicates gmm's math to byte-
   verify the (rhs, rhs_scale) layout) + `quant_fp4_dequant_check.py` + `s1_cpu_repro_v4flash.py both`.
2. **Full smoke + GATE.** Startup ~65s (cache warm); first-request prefill compile cold ~3-5 min, warm
   seconds. `VLLM_ENGINE_READY_TIMEOUT_S=2400`. Reserve to ≤1-2/session. Single-request gate probe:
   `scripts/s1_probe2.py N`. Concurrent multi-request decode: `MAX_SEQS=4 bash …smoke.sh` then
   `scripts/quant_concurrent_probe.py K N` — verdict ALL_CONCURRENT_CORRECT (KNOWN BROKEN as of Q.15;
   use to validate any future concurrent-decode fix).

---

## SLICE STATE (09:xxZ) — VERIFY, don't trust (all ephemeral)
- **Q.15 MAX_SEQS=4 engine reset at handoff** (clean; 16.0 TPU free, lockfiles cleared). Cache now WARM
  for MAX_LEN=256 (MAX_SEQS=1 AND =4) + MAX_LEN=4096 (MAX_SEQS=1). Q.15 log:
  `logs/full-slice-v4-smoke-20260529T085830Z.log`.
- Guardians alive (`node_guardian` 700703 / `meta_guardian` 700702). Weights mounted all 4 hosts.
- **NO model code changed** (`deepseek_v4_moe.py` md5 `9e2738f0…`) ⇒ no sync/cache-clear needed. Q.15
  added `scripts/quant_concurrent_probe.py` (head-only probe, runs vs localhost:18081 — no sync needed).

---

## INFRA STATUS / v6e-16 (unchanged — durable)
- Slice `v6spoteu719`, zone `europe-west4-a`, project `prm-research`. **v6e-16, topology 4×4**, 16
  chips, 4 hosts: head `10.164.0.15` + workers `10.164.0.8 / .17 / .16`. Ray healthy. TP=16.
- ⚠️ **WEIGHTS readable by enyouki on ALL 4 hosts**: `scripts/full_slice_v4_mount_weights.sh` once per
  bringup (smoke pre-flight runs it; dies on reboot). ⚠️ **numpy `<2.4` (2.3.5)** or numba import breaks
  the APIServer. ⚠️ Ray "version mismatch" ⇒ keep BOTH guardians alive (meta needs `10.164.0.15:6379`).
- After ANY code edit: `full_slice_v4_sync.sh` + clear `~/.cache/vllm/xla_cache/*` on all 4 + verify md5
  head==workers. Shut down ONLY via `full_slice_v4_reset.sh`.

# Handoff — DeepSeek-V4-Flash QUANTIZED-WEIGHT-LOADING campaign (v6e-16)

> **Phase = QUANT / FIT-ON-v6e-16.** Make `vllm serve DeepSeek-V4-Flash` LOAD AND SERVE CORRECTLY on
> **v6e-16** by **NOT dequantizing the FP4 experts to bf16**. Durable slice ops + pitfalls: `CLAUDE.md`.
> Prior campaigns (history): `HANDOFF_PERF.md`, `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29):** 🎯 **MILESTONE Q.13 — GATE PASSED. V4-Flash LOADS + FITS + SERVES +
> DECODES CORRECTLY + DETERMINISTICALLY on v6e-16 with the FP4 experts kept compressed.** The campaign's
> core goal is MET. The prefill forward COMPILES (no OOM, no Mosaic error) and the FIB decode is exact
> (`21, 34, 55, 89, 144, 233, 377, 610`). FP4 experts are fed to `gmm_v2` as **fp8 codes + per-block
> rhs_scale** (fp4→fp8 cast is LOSSLESS; v6e Mosaic CANNOT compile the native fp4 kernel unpack).
> **NEXT = pick up PERF (decode is ~0.43 tok/s) or HARDENING (decode-path fp8, larger configs)** — the
> fit-and-serve goal no longer blocks. See §NEXT.

---

## GATE — PASSED (v6e-16 baseline established this session)
The old v6e-32 md5 `5bf42256` is DEAD. The NEW v6e-16 baseline + the full gate (all ✅ this session):
- **Correct Fibonacci**: `s1_probe2.py 24` → `' 21, 34, 55, 89, 144, 233, 377, 610,'` (exact).
- **N=2 md5 `3069e80b`** (text `' 21'`), **byte-identical across 2 FRESH engines** (the S1-grade
  determinism bar — coherent output alone is NOT proof). Probe: `python3 scripts/s1_probe2.py N`
  (FIB prompt, temp=0 seed=0, port 18081).
- **`LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0** (64-token decode:
  visible_words=30, max_word_run=1, ends_clean=1).
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

CPU validation (cheap tier, all green): `scripts/quant_gmm_fp4_orient_check.py` (NEW; orientation
BYTE-EXACT vs `_dequant_fp4_experts` in gmm's [g,k,n] layout) + `quant_fp4_dequant_check.py` +
`s1_cpu_repro_v4flash.py both`.

---

## <a name="NEXT"></a>⚠️ NEXT ACTION — fit-and-serve is DONE; choose the next campaign axis
The blocking goal (load+fit+serve+correct on v6e-16) is COMPLETE + GATED. Pick up, in priority order:

1. **PERF (likely the real next campaign — see `HANDOFF_PERF.md`).** The gated smoke measured
   `observed_tps=0.43` (64 tokens in ~150 s) — functional but very slow. Suspects: `--enforce-eager`,
   the fp8 gmm path, decode dense all-256 einsum, cold-ish runtime. A microbench of the decode step is
   the cheap entry point. NOTE the gate must still hold after any perf change.
2. **HARDEN the decode/dense path.** `moe_forward`'s DENSE branch (CPU + replicated decode) STILL
   dequants the local experts to bf16 in-trace (`deepseek_v4_moe.py` ~:295). It did NOT OOM at the
   gated config (MAX_LEN=256, MAX_SEQS=1, N=1 decode), but at larger `--max-num-seqs` / longer context
   the bf16 decode temp could grow. If it OOMs, convert the decode path to the same fp8-gmm scheme
   (`_fp4_rhs_and_scale` + a decode-shaped gmm_v2 / quantized einsum).
3. **Larger configs.** Gate passed at MAX_LEN=256/MAX_SEQS=1 only. Validate higher before claiming prod.

If continuing the quant axis isn't fruitful, the loop's stated job (make it LOAD AND SERVE CORRECTLY)
is satisfied — consider whether the operator wants perf next (then `touch /tmp/quant_loop_stop` and move
to the perf loop) or hardening.

---

## VALIDATION TIERS (cheapest first) — unchanged, but note gmm_v2 is TPU-only
1. **CPU (~40s, no slice):** `JAX_PLATFORMS=cpu PYTHONPATH=work/tpu-inference:work/vllm
   work/vllm_env/bin/python3 scripts/quant_gmm_fp4_orient_check.py` (orientation/shape — gmm_v2 CANNOT
   run on CPU, it calls `get_tpu_info()` at trace time; the orient check replicates gmm's math to byte-
   verify the (rhs, rhs_scale) layout) + `quant_fp4_dequant_check.py` + `s1_cpu_repro_v4flash.py both`.
2. **Full smoke + GATE.** Startup ~65s (cache warm); first-request prefill compile cold ~3-5 min, warm
   seconds. `VLLM_ENGINE_READY_TIMEOUT_S=2400`. Reserve to ≤1-2/session.

---

## SLICE STATE (08:1xZ) — VERIFY, don't trust (all ephemeral)
- Engine #2 was UP (warm cache) for the determinism probe; **reset at handoff** (clean state; cache
  WARM ⇒ next smoke startup ~65s + cached prefill compile). Latest gated smoke log:
  `logs/full-slice-v4-smoke-20260529T080950Z.log`.
- Guardians alive (`node_guardian` 700703 / `meta_guardian` 700702). Weights mounted all 4 hosts.
- Code synced + md5 identical head==workers (`deepseek_v4_moe.py` md5 `9e2738f0…`). xla_cache WARM.

---

## INFRA STATUS / v6e-16 (unchanged — durable)
- Slice `v6spoteu719`, zone `europe-west4-a`, project `prm-research`. **v6e-16, topology 4×4**, 16
  chips, 4 hosts: head `10.164.0.15` + workers `10.164.0.8 / .17 / .16`. Ray healthy. TP=16.
- ⚠️ **WEIGHTS readable by enyouki on ALL 4 hosts**: `scripts/full_slice_v4_mount_weights.sh` once per
  bringup (smoke pre-flight runs it; dies on reboot). ⚠️ **numpy `<2.4` (2.3.5)** or numba import breaks
  the APIServer. ⚠️ Ray "version mismatch" ⇒ keep BOTH guardians alive (meta needs `10.164.0.15:6379`).
- After ANY code edit: `full_slice_v4_sync.sh` + clear `~/.cache/vllm/xla_cache/*` on all 4 + verify md5
  head==workers. Shut down ONLY via `full_slice_v4_reset.sh`.

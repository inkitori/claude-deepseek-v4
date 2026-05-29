# Handoff — DeepSeek-V4-Flash QUANTIZED-WEIGHT-LOADING campaign (v6e-16)

> **Phase = QUANT / FIT-ON-v6e-16.** Make `vllm serve deepseek-ai/DeepSeek-V4-Flash`
> LOAD AND SERVE CORRECTLY on **v6e-16** by **NOT dequantizing the FP4 experts to bf16 at load**.
> Durable slice ops + pitfalls: `CLAUDE.md`. Prior campaigns (history): `HANDOFF_PERF.md`,
> `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29):** **Q.2 (the LOADER) LANDED + CPU-GATED + COMMITTED (`26318abf`).**
> The loader now emits FP4 experts COMPRESSED (packed uint8 weight + e8m0 scale leaves) instead of
> dequantizing to bf16. **First real smoke (Q.5) RAN and the FIT LOOKS GOOD** (loads to 1400
> tensors, **zero OOM**) — but hits a **deterministic TPU `scheckne` core-halt at the first
> layer-0 consolidation `device_put`** (NOT my loader logic, which CPU-passes; NOT OOM; NOT
> ordering). **NEXT = diagnose the divergent collective (HLO dump) / try uint8 scales.** Details below.

---

## THE PROBLEM (one table)
V4-Flash ships natively quantized: dense=FP8, 256 routed experts=**FP4 (=MXFP4, codebook
`[0,.5,1,1.5,2,3,4,6]` ≡ `jnp.float4_e2m1fn`, e8m0 block scale, block 32)**. On-disk (confirmed by
reading the real ckpt): routed `layers.{L}.ffn.experts.{0-255}.{w1,w2,w3}.weight` = **I8** packed +
`.scale` = **F8_E8M0**; shared `ffn.shared_experts` = **F8_E4M3** (FP8, NOT fp4); 43 layers,
hidden=4096, moe_inter=2048. Old loader dequantized everything → ~542 GiB > 512 GiB HBM → OOM.

| scheme | HBM | fits v6e-16 (512 GiB)? |
|---|---:|---|
| bf16 (old load path) | ~542 GiB | **NO** (OOM) |
| **fp4-experts-kept + dense bf16 (Strategy C)** | **~155 GiB** | **yes, ~350 GiB free** |

---

## WHAT LANDED — Q.1 + consumer (prior) + Q.2 LOADER (`26318abf`, this session)
Strategy C, all CPU-gated. Per-expert routed leaves: weight `uint8 [out,in/2]`, scale
`e8m0 [out,in/32]`; consolidated E-sharded (`P('attn_dp',None,None)`) into `w*_stacked` +
`w*_scale_stacked`; `moe_forward` dequants uint8+e8m0→bf16 in-trace (`_dequant_fp4_experts`).
- **Q.2 loader edits** (`deepseek_v4_loader.py` + `deepseek_v4.py`): `read_dequant_slice` fp4→raw
  uint8 (no dequant) + new `kind=="e8m0"` (raw scale); `iter_v4_safetensors_specs` yields the e8m0
  scale as its own spec; `map_hf_name_to_jax_path` `.scale`→`_scale` leaf (was dropped as
  `<scale>`); consolidation regex widened `(w[123](?:_scale)?)` → scales stack like w2;
  `_torch_to_numpy_preserve` bitcasts torch e8m0→ml_dtypes (not a numeric cast); `_kind_of`
  `"experts"`→`".experts."` (excludes FP8 shared_experts).
- **Sharding note (matters for the bug below):** uint8 packing makes w1/w3 square `[2048,2048]` →
  `pick_partition_spec` strict-`>` tie-break picks **axis-0**, so w1/w3 flip from bf16's axis-1
  (host-gather) to **axis-0 → device_put reshard** (joining w2). Net: ALL routed weight+scale leaves
  now consolidate via `device_put` collectives; the bf16 w1/w3 host-gather path is now dormant.
- **CPU GATES ALL PASS:** `scripts/quant_loader_fp4_check.py` (NEW, on `tiny_v4_quant` fixture: 168
  experts emit uint8+e8m0, route to wN/wN_scale, dequant byte-identical to groundtruth, fp8 still
  bf16); `quant_fp4_dequant_check.py` max|Δ|=0; CPU oracle eager==jit bad=0/12.

---

## ⚠️ Q.5 FIRST SMOKE — RAN, FIT OK, but DETERMINISTIC TPU CORE-HALT (the blocker)
Three smokes (logs `logs/full-slice-v4-smoke-20260529T01{4014,4437,5204}Z.log`). Every one:
- **Loads cleanly to "placed 1400 tensors" (layer 0), placing BOTH .weight and .scale leaves, NO
  OOM, no Python/XLA/dtype error** — the loader works and **the FP4-compressed model is FITTING**.
- Then a **`Core halted unexpectedly … scheckne` at the SAME pc `TensorCoreSequencer:1:0xba`**
  across MULTIPLE cores (tpu17 pe2/pe4, tpu21) + `different launch id`. Engine init fails.
- **Coincides with `jit_broadcast_in_dim` compiles** → it's the **first layer-0 consolidation
  `device_put`** (a reshard collective), not the model forward.

**Ruled out:** (a) **multi-thread ordering** — `V4_LOADER_PLACE_WORKERS=1` (deterministic
cross-worker drain order) crashes IDENTICALLY at 1400. (b) **flaky hardware** — same assertion pc
across many cores ⇒ deterministic compiled-program assertion, not one bad core. (c) **OOM**. (d)
**loader logic** — all CPU gates pass.
**Diagnosis:** a launch-group **collective-consistency assertion** (`scheckne`) fires ⇒ workers
diverge on the consolidation `device_put` collective. The reshard PATTERN is proven (bf16 w2 did
exactly axis-0→E-shard `device_put`), so the new variable is **dtype**: scales are **e8m0** (the only
new dtype in a consolidation collective; w1/w3/w2 are uint8, same byte-reshard as proven bf16 w2).

---

## ROADMAP / NEXT ACTIONS (do in order; smokes crash ~90s in at 1400 tensors → cheap to iterate)
1. **HLO-DUMP DIFF (the error's own prescribed diagnostic, DECISIVE).** Re-smoke with
   `V4_XLA_FLAGS=--xla_dump_to=/tmp/hlo_dump` (opt in via `V4_XLA_FLAGS`, validate per CLAUDE.md
   pitfall #4), collect per-worker `before_optimizations.txt`, diff across the 4 hosts. If they
   differ → that op is the divergence (almost certainly the consolidation/scale path) → fixes it.
2. **e8m0-reshard hypothesis → store scales as plain uint8** (consumer `e8m0_to_fp32` wants uint8
   anyway; gpt_oss uses uint8 scales). Edits: `deepseek_v4.py:1032-1034` scale leaves `e8m0`→`u8`;
   `read_dequant_slice` `kind=="e8m0"` → `return w.view(torch.uint8)`. **WRINKLE:** the CPU oracle
   `make_random_params` (`s1_cpu_repro_v4flash.py:87-91`) and `quant_loader_fp4_check.py` use the
   **e8m0 dtype as a weight-vs-scale TYPE TAG** to synthesize SMALL stable scale bytes (118-121) vs
   full-range weight bytes — switch them to disambiguate by leaf NAME/shape (`*_scale`), else uint8
   scales get full-range bytes → NaN. Then re-run CPU gates + smoke.
3. **Bisect the 3 changes** if 1-2 inconclusive: temporarily skip scale consolidation (does it crash
   on weights alone?) vs revert w1/w3 to axis-1 host-gather. Isolates scales vs the axis-0 flip.
4. After it loads+serves: **Q.5 GATE** below (correct Fibonacci + md5; establish v6e-16 baseline).

---

## <a name="GATE"></a>GATE (non-negotiable) — for v6e-16
Old v6e-32 md5 `5bf42256` is DEAD (bf16 OOMs; fp4 changes numerics). Bar:
- `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (visible_words ≥ 10, max_run < 5).
- FIB decode: **correct Fibonacci (21,34,55,89,144)** + **N=2 md5 byte-identical across 2 fresh
  engines** (`python3 /tmp/s1_probe2.py 2`). Establish the NEW baseline hash once; confirm identical
  ×2 engines. READ the actual decode text ("contains Paris" is a false positive). Do NOT gate on a
  long-tail md5 (nondeterministic at temp=0). First milestone "loads, fits, serves, correct +
  deterministic" IS the first gate-pass.

---

## VALIDATION TIERS (cheapest first)
1. **CPU loader fp4 check** (NEW, no slice, ~40s): `JAX_PLATFORMS=cpu PYTHONPATH=work/tpu-inference:work/vllm
   work/vllm_env/bin/python3 scripts/quant_loader_fp4_check.py` → "OK …". Needs the
   `work/scratch/tiny_v4_quant`+`tiny_v4_groundtruth` fixtures (rebuild with
   `V4_REAL_FLASH=<gcs snapshot dir> V4_SCRATCH_DIR=work/scratch …python3 scripts/make_tiny_v4_checkpoint.py`;
   gcs snapshot = `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/<hash>`).
1b. **CPU oracle** (`scripts/s1_cpu_repro_v4flash.py both`) + **fp4 dequant check**
   (`scripts/quant_fp4_dequant_check.py`). All three currently PASS.
2. **TPU microbench** (`scripts/perf_microbench*`).
3. **Full smoke + GATE** — RESERVE. Crashes ~90s in right now (cheap to iterate while debugging).

---

## INFRA STATUS / v6e-16
- Slice `v6spoteu719`, zone `europe-west4-a`, project `prm-research`. **v6e-16, topology 4×4**, 16
  chips, 4 hosts: head `10.164.0.15` + workers `10.164.0.8 / .17 / .16`. Ray healthy (16 TPU).
  Smoke = TP=16. Per-host venvs (NOT shared).
- ⚠️ **numpy MUST be `<2.4` (pinned `2.3.5`)** — 2.4.x breaks `import numba` (needs ≤2.3) and the
  vllm-serve import chain crashes the APIServer **before any TPU work** with an unrelated-looking
  stack. Something installed 2.4.6 (2026-05-28); fixed this session on all 4 hosts. venv has **no
  pip** (uv): `~/.local/bin/uv pip install --python work/vllm_env/bin/python3 'numpy==2.3.5'` per
  host. Quick check before a smoke: `python3 -c "import numba"` on each host.
- ⚠️ Ray "version mismatch" = mark's rogue ray container; **FIX = keep BOTH guardians alive**
  (`ps -eo pid,cmd | grep -E 'node_guard[i]an|meta_guard[i]an'`). Restart per the loop prompt
  (meta_guardian needs `10.164.0.15:6379`). Ray (re)start: `scripts/full_slice_v4_ray_restart.sh`.
- After ANY code edit: `full_slice_v4_sync.sh` + clear `~/.cache/vllm/xla_cache/*` on all 4 hosts +
  verify md5 head==workers (mismatch → launch-id halt). Shut down ONLY via `full_slice_v4_reset.sh`.

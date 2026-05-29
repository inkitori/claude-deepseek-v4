# Handoff — DeepSeek-V4-Flash QUANTIZED-WEIGHT-LOADING campaign (v6e-16)

> **Phase = QUANT / FIT-ON-v6e-16.** Make `vllm serve deepseek-ai/DeepSeek-V4-Flash`
> LOAD AND SERVE CORRECTLY on the **v6e-16** slice by **NOT dequantizing the native FP4
> experts to full bf16 at load**. Durable slice ops + pitfalls: `CLAUDE.md`. Prior campaigns
> (history): `HANDOFF_PERF.md`, `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29):** **Q.1 (param tree = FP4 leaves) + the MoE consumer
> (dequant-in-trace) are DONE and CPU-GATED.** The 256 routed experts are now declared in the
> param tree as packed FP4 (uint8 `[E,out,in/2]`) + e8m0 scale (`float8_e8m0fnu [E,out,in/32]`),
> and `moe_forward` dequants them to bf16 in-trace per layer. Two cheap gates pass: the dequant
> is **bit-identical** to the trusted host torch dequant (`scripts/quant_fp4_dequant_check.py`,
> max|Δ|=0 over the full codebook+e8m0 range) and the CPU oracle is **clean** (eager==jit==prefill
> 0/12). **NEXT = Q.2: the LOADER** — emit the two FP4 leaves instead of dequantizing (real load
> is intentionally inert until then). Then Q.5 = first real smoke = first gate-pass.

---

## THE PROBLEM (one table)
V4-Flash ships natively quantized: dense=FP8, 256 routed experts=**FP4 (=MXFP4, codebook
`[0,.5,1,1.5,2,3,4,6]` ≡ `jnp.float4_e2m1fn`, e8m0 block scale, block 32)**. The loader
**dequantizes everything to bf16** → ~542 GiB > 512 GiB HBM → OOM.

| scheme | HBM | fits v6e-16 (512 GiB)? |
|---|---:|---|
| bf16 (old load path) | ~542 GiB | **NO** (OOM) |
| **fp4-experts-kept + dense bf16 (Strategy C)** | **~155 GiB** | **yes, ~350 GiB free** |

Experts = 92–95% of footprint (137 GiB fp4 → 516 GiB bf16). KV negligible. **Keep experts FP4 ⇒ fits.**

---

## THE PLAN — Strategy C (DONE on the device side; loader remains)
Keep the routed experts compressed; dense stays bf16. **Representation (LOCKED, CPU-validated):**
- **Weight leaf**: packed `uint8` `[E, out, in/2]` (2 e2m1/byte, low nibble first along IN).
- **Scale leaf**: `float8_e8m0fnu` `[E, out, in/MXFP4_BLOCK_SIZE]` (one e8m0 per 32 IN elems).
- **Sharded on E (axis-0, `attn_dp`)** — never splits a packed byte or a scale block, and keeps
  the **host-side `np.stack` consolidation (the S1 fix) working** (numpy holds uint8; it cannot
  hold float4 — this is why we store uint8, not `float4_e2m1fn`).
- **`moe_forward` dequants uint8+e8m0 → bf16 in-trace** (`_dequant_fp4_experts`, reuses shared
  `u8_unpack_e2m1`/`e8m0_to_fp32`), feeding bf16 into BOTH paths (dense einsum + gmm_v2 shard_map)
  **byte-identically to the old bf16 path → S1 fix preserved exactly.** The dequant is per-LAYER
  transient (~one layer's local experts ≈ 0.8 GiB/chip); stored fp4 ≈ 8.6 + scale ≈ 2 GiB/chip ≪ 32.
- Shared/dense expert stays **bf16** (separate template; tiny; `expert_forward` untouched).
- **Feeding FP4 straight into `gmm_v2` via its `rhs_scale=[G,K/32,1,N]` arg (unpack-in-VMEM, no
  bf16 transient) is a FUTURE PERF optimization — NOT needed for fit/correctness.** Contract is
  mapped (gmm_v2 wants logically-UNPACKED `float4_e2m1fn[G,K,N]` + fp32/e8m0 `rhs_scale`; touches
  the `_routed_local` shard_map `in_specs` — the S1-fused path, so do it carefully + TPU-validated).

---

## WHAT LANDED THIS SESSION (commit: see git log) — Q.1 + consumer, CPU-GATED
- `models/jax/deepseek_v4.py`: `make_abstract_moe_params` splits **routed** (FP4: uint8 w* +
  e8m0 w*_scale) vs **shared** (bf16) templates; pytree registration adds `w*_scale` /
  `w*_scale_stacked`; imports `MXFP4_BLOCK_SIZE`.
- `layers/jax/moe/deepseek_v4_moe.py`: `ExpertParams`/`MoEParams` gain optional `w*_scale[_stacked]`
  leaves; new `_dequant_fp4_experts`; `moe_forward` binds scales + dequants uint8→bf16 (guarded by
  `W1.dtype==uint8`). **Downstream unchanged.**
- `scripts/s1_cpu_repro_v4flash.py`: `make_random_params` synthesizes valid fp4 leaves (full-range
  weight bytes; small e8m0 scales 2^-9..2^-6 so the model stays in the stable bf16-baseline regime).
- `scripts/quant_fp4_dequant_check.py` (NEW): standalone bit-faithful cross-check.

---

## ROADMAP (drive top-down; cheapest validation tier first)
1. ~~**Q.1** — param tree: experts = fp4 + e8m0 leaves.~~ **DONE + CPU-GATED.**
2. ~~**Q.3/Q.4** — MoE consumer: dequant-in-trace (both paths), S1-preserving.~~ **DONE + CPU-GATED.**
   (rhs_scale-into-gmm = future perf, not on the path to smoke.)
3. **Q.2 — LOADER (THE NEXT BIG PIECE; TPU-only validation).** For `kind=="fp4"` experts, STOP
   dequantizing — emit the packed `uint8` weight leaf + the `e8m0` scale leaf. The abstract tree
   already declares both (Q.1). Five coordinated edits in `models/jax/deepseek_v4_loader.py` +
   `deepseek_v4.py` (verified by the loader-mapping audit):
   - `read_dequant_slice`(:847)/`place_spec_as_jax_sharded`(:918): for fp4, return the **raw packed
     bytes** (uint8, no `dequant_fp4_to_bf16`) + a sibling **scale** array. Axis-0 (out) slicing is
     already byte/block-safe (loader:862 weight, :884 scale) — fp8's block-align dance is NOT needed.
   - **Stop DROPPING `.scale`**: today `map_hf_name_to_jax_path` + the `<scale>` early-returns
     (deepseek_v4.py ~:379, :1613, :1639) discard scales. Route `...wN.scale` → a real `...wN_scale`
     JAX path so it reaches `_assign`.
   - `_maybe_consolidate`(deepseek_v4.py:1497)/`_is_stash_leaf`(:1492): also `np.stack` the per-expert
     scales into `w*_scale_stacked`, sharded `P('attn_dp',None,None)` (same E-shard as the weights).
   - **Shared expert**: confirm its on-disk dtype. If it's FP8 dense (likely) keep the bf16 dequant
     (Strategy C leaves dense bf16); the loader's `_kind_of` (:803) must NOT classify it fp4. If the
     name contains "experts" it may misfire — check.
   - The `target_dtype` cast (loader:993) becomes a bitcast-equivalent (uint8→uint8); ensure no
     numeric `astype` of already-unpacked values.
4. **Q.5 — FIRST REAL SMOKE = first gate-pass** (after `full_slice_v4_sync.sh` + clear xla_cache on
   all 4 hosts). Does it LOAD (fit!), serve, **correct Fibonacci + md5 identical ×2 fresh engines**?
   Establish the NEW v6e-16 md5 baseline here.

---

## <a name="GATE"></a>GATE (non-negotiable) — REDEFINED for v6e-16
The old v6e-32 md5 `5bf42256` is **DEAD** (bf16 OOMs; fp4 changes numerics). Bar:
- `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (visible_words ≥ 10, max_run < 5).
- FIB decode: **correct Fibonacci (21,34,55,89,144)** + **N=2 md5 byte-identical across 2 fresh
  engines** (`python3 /tmp/s1_probe2.py 2`). The fp4 path sets a **NEW** baseline hash — establish
  once, confirm identical ×2 engines. Non-negotiable = **identical ×2 + correct Fibonacci**.
- READ the actual decode text ("contains Paris" is a false positive — can EOS at tok 1).
- Do NOT gate on a long-tail md5 (`s1_probe2.py 20`+) — nondeterministic at temp=0 (decode all-reduce).
- **First milestone "it loads, fits, serves, correct + deterministic" IS the first gate-pass.**

---

## VALIDATION TIERS (cheapest first)
1. **CPU torch oracle** (no slice): `PYTHONPATH=work/tpu-inference:work/vllm work/vllm_env/bin/python3
   scripts/s1_cpu_repro_v4flash.py both` → "OK: both eager and jit match". Proves shapes/math/no-NaN
   + jit-safe; a quant numerics SHIFT still passes. Exercises the DENSE moe path (gmm is TPU-only).
1b. **FP4 dequant cross-check** (no slice, seconds): `…/python3 scripts/quant_fp4_dequant_check.py`
   → "OK …byte-identical". Proves `_dequant_fp4_experts` == host `dequant_fp4_to_bf16` bit-for-bit.
2. **TPU microbench** (`scripts/perf_microbench*`): a kernel/op on the real mesh, no 543 GiB load.
3. **Full smoke + the GATE** — RESERVE (≤1–2/session; cold compile 10–30 min). The fit/correctness gate.

---

## NEXT ACTION (for the session reading this)
Do **Q.2 — the loader** (above). It's the last piece before the model can actually LOAD with experts
kept FP4. The abstract tree + consumer are already in place and CPU-gated, so Q.2 has a fixed target:
produce a `uint8 [E,out,in/2]` weight leaf + a `float8_e8m0fnu [E,out,in/32]` scale leaf per routed
expert (E-sharded), and stop dropping `.scale`. Fan out subagents on the 5 edit sites (mapped above);
the loader can only be validated by a real smoke (Q.5), so land Q.2 + Q.5 in one focused session.
After ANY edit: `full_slice_v4_sync.sh` + clear `~/.cache/vllm/xla_cache/*` on all 4 hosts BEFORE smoke.

⚠️ **Production load is intentionally INERT until Q.2** (tree says uint8; the loader still dequants to
bf16 → dtype mismatch). This regresses nothing (bf16 already OOMs) and the CPU oracle is the gate for
Q.1/consumer. Do NOT smoke before Q.2 lands.

---

## INFRA STATUS / v6e-16 (so the next session doesn't re-derive)
- Slice `v6spoteu719`, zone `europe-west4-a`, project `prm-research`. **v6e-16, topology 4×4**,
  16 chips, 4 hosts: head `10.164.0.15` + workers `10.164.0.8 / .17 / .16`. venvs + GCS weights
  mounted; Ray healthy (16 TPU). Smoke = TP=16 (`full_slice_v4_smoke.sh`, self-guards single-instance).
- ⚠️ **Ray "version mismatch" = mark's rogue ray-2.54.1 `node` container poisoning GCS CLUSTER_METADATA**
  — NOT a corrupt venv. **FIX = keep BOTH guardians alive during ALL TPU work:**
  `ps -eo pid,cmd | grep -E 'node_guard[i]an|meta_guard[i]an'`. Restart if dead:
  `INTERVAL=3 setsid bash scripts/full_slice_v4_node_guardian.sh >logs/node_guardian.log 2>&1 </dev/null &`
  and `setsid work/vllm_env/bin/python scripts/full_slice_v4_meta_guardian.py 10.164.0.15:6379 >logs/meta_guardian.log 2>&1 </dev/null &`
  (⚠️ ALWAYS pass `10.164.0.15:6379` — meta_guardian's default IP is the stale v6e-32 head).
- Ray (re)start: `scripts/full_slice_v4_ray_restart.sh` (verifies 16 TPU). Bootstrap: `full_slice_v4_bootstrap.sh`.

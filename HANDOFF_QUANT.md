# Handoff — DeepSeek-V4-Flash QUANTIZED-WEIGHT-LOADING campaign (v6e-16)

> **Phase = QUANT / FIT-ON-v6e-16.** The job: make `vllm serve deepseek-ai/DeepSeek-V4-Flash`
> LOAD AND SERVE CORRECTLY on the **v6e-16** slice by **NOT dequantizing the model's native
> quantized weights to full bf16 at load time**. Durable slice ops + pitfalls: `CLAUDE.md`.
> Prior campaigns (history, not the goal): `HANDOFF_PERF.md` (perf), `HANDOFF_S1.md` (decode
> determinism). This doc is the loop's memory — current state, the math, the roadmap, the ONE
> next action.
>
> **One-line status (2026-05-28):** **Infra fully bootstrapped & verified on `v6spoteu719`
> (v6e-16, 4×4, 16 chips, 4 hosts).** venv on all 4 hosts, GCS weights mounted, vllm+tpu_inference
> import clean, **Ray cluster healthy (16 TPU, `ray.init` OK, [4,4,4,4]/host)**. v6e-16 topology
> wired into `full_slice_v4_smoke.sh` (TP=16, bounds 2,2,1, place_workers=4) + `full_slice_v4_ray_restart.sh`
> (16-TPU verify). **No real-V4 smoke run yet** — the current bf16 load path is expected to OOM
> (see math). **NEXT = implement the loader change so the fp4 experts stay compressed in HBM
> (Strategy C), CPU-oracle it, then the first real smoke = the first gate-pass.**

---

## THE PROBLEM (the whole campaign in one table)

DeepSeek-V4-Flash ships **natively quantized**: dense linears = **FP8** (e4m3, e8m0 block scale,
128×128); the 256 routed experts = **FP4** (`expert_dtype=fp4`, e8m0 block scale, block 32) —
and V4's FP4 codebook `[0,0.5,1,1.5,2,3,4,6]` is **bit-identical to `jnp.float4_e2m1fn` (MXFP4)**.
The loader (`deepseek_v4_loader.py`) currently **dequantizes EVERYTHING to bf16 at load**. That
blows the checkpoint up from compressed to full bf16:

| scheme | weights in HBM | fits v6e-16 (512 GiB)? |
|---|---:|---|
| **bf16 (current load path)** | **~542 GiB** | **NO — exceeds 512 by ~30 GiB before any KV/activations/XLA scratch** |
| fp8-everywhere (experts fp4→fp8) | ~272 GiB | yes (~240 GiB free) |
| **fp4-experts-kept + dense bf16 (Strategy C)** | **~155 GiB** | **yes, ~350 GiB free** |
| native (fp8 dense + fp4 experts) | ~149 GiB | yes (~363 GiB free) |

- **v6e-16 = 16 chips × 32 GiB = 512 GiB HBM.** bf16 542 GiB > 512 → OOM on every chip
  (542/16 ≈ 34 GiB/chip > 32). This is exactly why V4 ran on v6e-32 (1 TiB) but not here.
- **Routed experts are 92–95% of the footprint** (137 GiB fp4 → 516 GiB bf16). Everything else
  (attn, embed, lm_head, shared expert, MTP) sums to ~11.6 GiB native / ~26 GiB bf16.
- KV cache is **negligible** (MLA-compressed: <0.2 GiB at max-model-len ≤ 2048, max-num-seqs=1).
- So the ONLY thing that must stay compressed to fit is **the experts**. Keep them FP4 and the
  model fits trivially. (Footprint numbers: measured from all 46 safetensors shard headers; the
  on-disk native ckpt is **148.65 GiB** — the `gsutil du` "297 GiB" double-counts blobs/+snapshots.)

---

## THE PLAN — Strategy C (keep FP4 experts compressed; dense stays bf16)

Smallest diff, biggest win. The experts are ~all the footprint; V4's FP4 is MXFP4; and the MoE
matmul kernel **already supports** quantized weights. Reuse, don't build:

- **`kernels/megablox/gmm_v2.py::gmm_v2`** (the grouped matmul V4's MoE prefill already calls)
  accepts **`rhs_scale`** for blockwise-quantized weights, on-the-fly LHS quant, and unpacks
  sub-byte-packed RHS (fp4/int4) in-kernel. `common.py:51` lists `jnp.float4_e2m1fn` as supported.
- **`kernels/quantized_matmul/tuned_block_sizes.py`** ships tuned tiles for
  `('float8_e4m3fn','float4_e2m1fn')` → fp8-act × fp4-weight is a first-class TPU path.
- **Closest blueprint:** `layers/vllm/quantization/mxfp4.py` + `models/jax/gpt_oss.py::_load_mxfp4`
  — GPT-OSS already loads packed-fp4 + e8m0-block-scale experts as `jnp.float4_e2m1fn`. STUDY THIS.
- Dense FP8 (if ever kept): `layers/jax/quantization/fp8.py::Fp8BlockwiseLinearMethod`.

Alternatives **(B) fp8-everywhere** and **(A) full-native** buy HBM headroom we don't need at
max-num-seqs=1, at the cost of a much larger, more correctness-risky diff. Do **C** first.

---

## PLUMBING — the change sites (verified)

**Loader (`models/jax/deepseek_v4_loader.py`) — the production real-weight path is**
`iter_v4_safetensors_specs` (:754, metadata only) → `place_spec_as_jax_sharded` (:918) →
`read_dequant_slice` (:847) → **`dequant_weight` (:226, THE bf16 choke point)**:
- `dequant_fp8_to_bf16` :115, `dequant_fp4_to_bf16` :142 → both return **bf16**.
- `place_spec_as_jax_sharded` casts host np to `target_dtype` (the abstract param tree's leaf
  dtype) at **:993**. Today every leaf is bf16/fp32 → forces bf16.
- **Change:** for `kind=="fp4"` (experts), STOP dequantizing — emit a `jnp.float4_e2m1fn` weight
  leaf + a separate e8m0/fp32 `rhs_scale` leaf. fp4 scale is `[out, in/block]`, row-aligned, so
  `read_dequant_slice`/`place_spec_as_jax_sharded` slice cleanly on axis-0 (mind the 2-fp4/byte
  packing — don't split a byte). This requires the **abstract param tree** (model definition) to
  declare expert leaves as fp4 + scale, not bf16 — i.e. touches the MoE param structure, not just
  the loader.

**Consumer (`layers/jax/moe/deepseek_v4_moe.py`):**
- **Prefill = sharded `gmm_v2`** (:234–:324; calls at :276/:285) — currently passes NO `rhs_scale`.
  Threading `rhs_scale=` here is the key, ~small change (reshape scale to `[G,num_blocks,1,n]`).
- **Decode = dense-all-256 `jnp.einsum`** (:217–:233; gate/up :223/:225, down :231) — can't take a
  packed fp4 array. Either (a) dequant the stacked W to bf16 INSIDE the trace at decode (weights
  still stored fp4 in HBM; the dequant is transient VMEM, not HBM — low-risk start), or (b) route
  decode through `gmm_v2` too. **(a) first.**
- ⚠️ The **S1 fix** lives in the MoE `use_shard_map`/`_routed_local` path (owned-expert mask /
  zero-init) — preserve it EXACTLY through any rhs_scale change.

**Attention / dense linears** (`deepseek_v4_attention.py` `_linear` :492; `deepseek_v4.py` MTP/lm_head):
every weight is `.astype(fp32)` at use — they never depend on bf16 *specifically*. Strategy C
leaves these bf16, so no change needed; only relevant if you later extend to fp8 dense (Strategy A).

---

## ROADMAP (drive top-down; each step clears the GATE; cheapest validation tier first)

1. **Q.0 — confirm/quantify the baseline (optional, ≤1 smoke).** Current bf16 path should OOM
   during weight load. A smoke confirms the v6e-16 serving path (ray/mesh/sharding/loader) is wired
   and shows exactly where HBM runs out. SKIP if you'd rather spend the smoke budget on the fix.
2. **Q.1 — model param tree: declare expert weights as fp4 + e8m0 scale leaves** (MoE definition).
   CPU-oracle the shapes/dtypes (`scripts/s1_cpu_repro_v4flash.py both` must still construct).
3. **Q.2 — loader: emit fp4 weight + scale for `kind=="fp4"`** instead of dequantizing
   (`dequant_weight`/`read_dequant_slice`/`place_spec_as_jax_sharded` target_dtype). CPU-oracle.
4. **Q.3 — MoE prefill: thread `rhs_scale` into the two `gmm_v2` calls.** CPU-oracle vs the torch
   reference (the kernel is interpret-mode on CPU but proves math/no-NaN).
5. **Q.4 — MoE decode: dequant-in-trace (a) or gmm route (b).** CPU-oracle.
6. **Q.5 — FIRST REAL SMOKE = first gate-pass:** model LOADS (fits!), serves, **correct Fibonacci
   + md5 identical ×2 fresh engines**. Establish the NEW v6e-16 md5 baseline here.
7. **Q.6 (only if more headroom needed) — extend to fp8 dense** (Strategy A); otherwise stop.

---

## <a name="GATE"></a>GATE (non-negotiable for every committed change) — REDEFINED for v6e-16

The S1 gate's md5 `5bf42256` was established on **v6e-32 + bf16** and **cannot be reproduced here**
(bf16 OOMs on v6e-16, and keeping experts fp4 changes the numerics anyway). So the bar is:
- `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (visible_words ≥ 10, max_run < 5).
- FIB decode: **correct Fibonacci (21, 34, 55, 89, 144 — deterministic/high-margin)** +
  **N=2 md5 byte-identical across 2 fresh engines** (`python3 /tmp/s1_probe2.py 2`). The fp4 path
  WILL set a **NEW** baseline hash — establish it once and confirm identical ×2 fresh engines.
  Non-negotiable = **identical ×2 engines + correct Fibonacci**, NOT a specific legacy hash.
- READ the actual decode text ("contains Paris" is a known false positive — can EOS at tok 1).
- ⚠️ Do NOT gate on a long-tail md5 (`s1_probe2.py 20`+): it's nondeterministic at temp=0
  (pre-existing decode all-reduce-ordering residual).
- **Because bf16 currently can't even load on v6e-16, the first milestone — "it loads, fits,
  serves, correct + deterministic" — IS the first real gate-pass**, not a regression check.

---

## VALIDATION TIERS (cheapest first — see CLAUDE.md "How to validate")
1. **CPU torch oracle** (no slice): `PYTHONPATH=work/tpu-inference:work/vllm work/vllm_env/bin/python3
   scripts/s1_cpu_repro_v4flash.py both` → "OK". Proves math/no-NaN; a quant numerics SHIFT still
   passes (eager+jit shift together). Cannot reproduce determinism (no sharding).
2. **TPU microbench** (`scripts/perf_microbench*`): time a kernel/op on the real mesh, no 543 GiB load.
3. **Full smoke + the GATE** — RESERVE (≤1–2/session; cold compile 10–30 min). The fit/correctness gate.

---

## NEXT ACTION (for the session reading this)
Implement **Strategy C** top-down from the roadmap: **Q.1 → Q.2 → Q.3 → Q.4 on the CPU oracle**
(cheap), then **Q.5 = the first real smoke** (does it FIT + serve + correct Fibonacci + deterministic
md5 ×2 engines?). Study `models/jax/gpt_oss.py::_load_mxfp4` + `kernels/megablox/gmm_v2.py` rhs_scale
first — they're the blueprint. Fan out subagents AGGRESSIVELY for the audit/draft while the slice is
idle; serialize slice access. Hand off when context grows (`scripts/quant_handoff_window.sh`).

---

## INFRA STATUS / v6e-16 bringup notes (so the next session doesn't re-derive)
- Slice `v6spoteu719`, zone `europe-west4-a`, project `prm-research`. **v6e-16, topology 4×4**,
  16 chips, 4 hosts: head `10.164.0.15` + workers `10.164.0.8 / .17 / .16` (auto-discovered).
- **Bootstrapped this session:** `uv` installed on all 4 hosts (head `~/.local/bin`, workers
  `/usr/local/bin`); `~/.ssh/google_compute_engine` generated + propagated to all workers via
  `gcloud compute tpus tpu-vm ssh`; venvs built (`setup.sh` fan-out); GCS weights mounted
  (`gs://personal-mark-eu/vllm/hub` → `~/.cache/huggingface/hub`); `.env` written (tokens omitted —
  weights are auth-free, `claude` uses `~/.claude` creds).
- ⚠️ **Ray-version-corruption fix (DONE, don't re-fight):** the bootstrap's `uv` install left an
  internally-inconsistent ray (Python 2.55.1 metadata but a stale 2.54.1 artifact → `ray.init`
  "version mismatch"). FIX = `uv pip install --reinstall --no-cache 'ray[default,data]==2.55.1'`
  on every host. Already applied; cluster verified `ray.init OK, 16 TPU, [4,4,4,4]`.
- To (re)start ray: `scripts/full_slice_v4_ray_restart.sh` (now verifies `0.0/16.0 TPU`).
- To smoke: `bash scripts/full_slice_v4_smoke.sh` (TP=16; self-guards single-instance). After any
  `.py` edit: `scripts/full_slice_v4_sync.sh` + clear `~/.cache/vllm/xla_cache/*` on all 4 hosts.
- Guardians: keep `node_guardian`/`meta_guardian` alive before TPU work (CLAUDE.md).

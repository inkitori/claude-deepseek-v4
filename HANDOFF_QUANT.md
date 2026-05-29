# Handoff — DeepSeek-V4-Flash QUANTIZED-WEIGHT-LOADING campaign (v6e-16)

> **Phase = QUANT / FIT-ON-v6e-16.** Make `vllm serve deepseek-ai/DeepSeek-V4-Flash` LOAD AND SERVE
> CORRECTLY on **v6e-16** by **NOT dequantizing the FP4 experts to bf16**. Durable slice ops +
> pitfalls: `CLAUDE.md`. Prior campaigns (history): `HANDOFF_PERF.md`, `HANDOFF_S1.md`. This doc =
> the loop's memory.
>
> **One-line status (2026-05-29):** 🎯 **MILESTONE — V4-Flash LOADS + FITS + SERVES cluster-wide.**
> The Q.10 smoke (`logs/full-slice-v4-smoke-20260529T065043Z.log`) is the FIRST to load REAL weights on
> ALL 4 hosts (every rank `Init model | hbm=[(9.75,31.25)×16]`, `placed 68800`, no dummy zeros) → KV
> cache → forward compile → `Application startup complete`. The whole load/fit/serve-bringup chain WORKS
> (Strategy C fit confirmed slice-wide, not just head). **Remaining blocker = the FORWARD COMPILE:** the
> first `/v1/completions` → HTTP 500 → **`CompileTimeHbmOom`** — the prefill `jit(run_model)` needs
> **37.32 GiB HBM temp vs 31.25 GiB/chip (over by 6.07 GiB)** because the MoE still **dequantizes the FP4
> experts to bf16 IN-TRACE** (`deepseek_v4_moe.py:243-246`), materializing full bf16 `[EP,out,in]` expert
> buffers. 37.30 GiB temp > a full 32 GiB chip ⇒ **NO config/memory-fraction escape.** **NEXT = feed FP4
> directly to `gmm_v2` via `rhs_scale` (skip the bf16 dequant)** — the roadmap core, now proven MANDATORY.
> Exact plan in §NEXT ACTION.

---

## THE PROBLEM — two HBM hurdles (LOAD solved ✅, FORWARD is the live blocker ⛔)
V4-Flash ships natively quantized: dense=FP8, 256 routed experts=**FP4 (MXFP4: codebook ≡
`jnp.float4_e2m1fn`, e8m0 block scale, block 32)**. On-disk routed `experts.{0-255}.{w1,w2,w3}.weight`
= I8 (2 FP4/byte along IN), `.scale` = F8_E8M0; H(hidden)=4096, I(moe_inter)=2048, 256 experts, 43
layers + 1 MTP.

| stage | scheme | HBM/chip | fits? |
|---|---|---:|---|
| LOAD | bf16 dequant-at-load (old) | ~542 GiB total | ❌ OOM |
| LOAD | **FP4 experts kept compressed (Strategy C)** | **9.75 / 31.25** | ✅ **confirmed cluster-wide** |
| FORWARD | FP4 → bf16 dequant IN-TRACE in MoE (current) | **37.32 temp** | ❌ > 32 GiB chip |
| FORWARD | **FP4 fed straight to `gmm_v2` (rhs_scale)** | target ≪ | ⛔ the fix → next |

---

## WHAT LANDED (committed)
**LOAD = Strategy C — DONE + confirmed serving.** Loader keeps the 256 routed experts FP4-compressed
(u8 packed weight + u8 e8m0 scale leaves; host-gathered ⇒ zero reshard collective): `26e4023d`,
`26318abf`, `580b1f83`, `6eb5241f`, `e1b434f8`. Post-load scheckne saga RESOLVED (RoPE freqs→host
`fb54237b`; RNG→host `0d8d57fe` Q.8; eager `create_jit_model` `83a18839` Q.9). Divergent worker
weight-load RESOLVED (`ca016156` Q.10: enyouki gcsfuse mount on workers via
`scripts/full_slice_v4_mount_weights.sh`). **Validated this session** by the `…065043Z` smoke: real
weights on all 4 hosts → `Application startup complete`. The bringup/load/fit/serve chain is proven —
do NOT re-litigate it.

**FORWARD blocker root-caused (this session, no fix yet).** First real `/v1/completions` → HTTP500 →
engine shutdown. Clean log+HLO (no py-spy needed): `JaxRuntimeError: RESOURCE_EXHAUSTED:
CompileTimeHbmOom` in the PREFILL compile of `jit(run_model)` (`tpu_runner.py:880 _execute_model →
model_fn`), 37.32G/31.25G, +6.07G. Dominant temps: MoE `bf16[16,2048,4096]`×8 + `bf16[16,4096,2048]`×5
(16 = experts-per-chip, 2048/4096 = I/H) under `jit(run_model)/shard_map/transpose`; HLO shows
`bf16[…]=bitcast(convert_element_type(u8…))` = the in-trace FP4→bf16 expert dequant. (Secondary:
`f32[16,4096,64,32]` attention broadcast_in_dim temps.)

---

## ⚠️ NEXT ACTION — feed FP4 straight to gmm_v2 (kill the in-trace bf16 dequant)
**One change, mapped to exact lines (all under `work/tpu-inference/tpu_inference/`). MIRROR the
CANONICAL in-repo pattern `layers/jax/moe/utils.py:205-232 gmm_fn` — it already does FP4-rhs gmm_v2.**
(`layers/common/fused_moe_gmm.py:101` = thin wrapper. gpt_oss `_load_mxfp4` = unpack-helper blueprint
only, NOT a gmm_v2 caller.)

1. **Delete/bypass the dequant** — `layers/jax/moe/deepseek_v4_moe.py:243-246`
   (`if W1.dtype==uint8: W1=_shard_e_first(_dequant_fp4_experts(W1,S1))`, +W2,W3). Keep W*/S* as u8.
   `_dequant_fp4_experts` (:36-55) = `u8_unpack_e2m1(w)*repeat(e8m0_to_fp32(scale),32)→bf16` = the blow-up.
2. **Convert the 2 prefill gmm_v2 calls in `_routed_local`** (both currently bf16 RHS + NO rhs_scale):
   - :324-329  `W13_l = concat([W1_l.T(0,2,1), W3_l.T(0,2,1)], axis=2).astype(dtype)`;
     `g1 = gmm_v2(x_sorted, W13_l, group_sizes, group_offset=…, zero_initialize=False, preferred_element_type=fp32)`.
   - :335-338  `W2g_l = W2_l.T(0,2,1).astype(dtype)`; `g2 = gmm_v2(h, W2g_l, …)`.
3. **gmm_v2 quantized-RHS interface** (`kernels/megablox/gmm_v2.py:1130`
   `gmm_v2(lhs, rhs, group_sizes, rhs_scale=None, rhs_bias=None, group_offset=None, *,…)`):
   - `rhs` = **typed `jnp.float4_e2m1fn`** array, logical `[group, k=IN, n=OUT]`. Build via
     `u8_unpack_e2m1(w_u8)` **without** the `.astype(float32)` (= `bitcast_convert_type(u8, float4_e2m1fn)`
     + reshape doubling IN), then transpose to `[EP, in, out]`. gmm sees 4-bit (`itemsize_bits<8`,
     `should_bitcast`) and packs to uint32 internally — **do NOT pass packed u8.**
   - `rhs_scale` = **plain fp32** (`e8m0_to_fp32(scale)`), shape `[group, num_blocks, 1, n]`,
     `num_blocks=k/32`; insert the middle-`1` with `jnp.expand_dims(scale_f32, 2)` (the utils.py move).
   - `lhs` = bf16; leave `maybe_quantize_lhs=True` (gmm auto-quantizes lhs→e4m3 for the fp4×fp8 MXU).
4. **Layout/orientation (Agent C):** stored leaves `[E,out,in]` (w1/w3 out=I=2048 in=H=4096; w2
   out=H=4096 in=I=2048), scale `[E,out,in/32]` blocked on IN, sharded on E(axis0)/attn_dp. Need rhs
   `[EP, in, out]` (transpose) + scale `[EP, in/32, 1, out]`. Build W13 by concatenating the two
   typed-fp4 arrays along OUT(n) and their scales along n; w2 separate. Replicate the `.T(0,2,1)` the
   bf16 code already applies.
5. **DECODE path** `:273-284` is a dense bf16 einsum over all-256 experts (also post-dequant). OOM hit
   PREFILL first; fix prefill, re-smoke. If decode then OOMs, convert it too (gmm_v2 / quantized einsum).

**S1 cautions — do NOT break (Agent A):** the `use_shard_map` prefill branch IS the S1 fix; the
`all_gather`+`optimization_barrier` (:301-302), `_owned`-mask `jnp.where` (:351-352) before `psum`
(:355), and `_v4_decode_replicate` (tpu_runner.py) are load-bearing. The RHS-dtype change is orthogonal
to that STRUCTURE, but the per-block rhs_scale math (block=32, K-orientation) must match
`_dequant_fp4_experts` exactly. **Numerics WILL shift vs bf16 → re-establish the GATE md5 baseline fresh**
(bar = correct Fibonacci + identical ×2 engines, not a specific hash).

**Validate cheap→expensive:** (1) `s1_cpu_repro_v4flash.py both` + `quant_fp4_dequant_check.py` — BUT
gmm_v2 is Pallas/Mosaic; first confirm it runs in CPU-interpret, else the oracle only catches shape/NaN
around it. (2) a tiny synthetic jit to shape-check the new gmm_v2 args before a smoke. (3) smoke + GATE.
After the edit: `full_slice_v4_sync.sh` + clear `~/.cache/vllm/xla_cache/*` on all 4 + verify md5, then
smoke. **Expect 1-2 smoke iterations to nail the orientation/scale shape.**

**SLICE STATE (07:08Z):** reset, 16/16 TPU free; guardians alive; weights mounted all 4 hosts; xla_cache
cleared; code synced (model_loader.py md5 `791b1137…`). All ephemeral — VERIFY, don't trust.

---

## <a name="GATE"></a>GATE (non-negotiable) — for v6e-16
Old v6e-32 md5 `5bf42256` is DEAD. The FIRST gate-pass is pending (needs the forward to compile first).
- `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (port 18081; visible_words ≥ 10,
  max_word_run < 5; forces a real 64-token coherent decode — NOT the "Paris" false positive).
- FIB decode: **correct Fibonacci (21,34,55,89,144)** in a longer decode + **N=2 md5 byte-identical
  across 2 fresh engines** via `python3 scripts/s1_probe2.py N` (now committed to the repo; prompt
  "Here is the start of the Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13,", temp=0 seed=0, port 18081,
  auto-discovers model id). Establish the NEW baseline hash once; confirm identical ×2 engines. READ the
  actual text. Do NOT gate on a long-tail md5 (nondeterministic at temp=0).

---

## VALIDATION TIERS (cheapest first)
1. **CPU loader fp4 check** (~40s): `JAX_PLATFORMS=cpu PYTHONPATH=work/tpu-inference:work/vllm
   work/vllm_env/bin/python3 scripts/quant_loader_fp4_check.py` + CPU oracle `s1_cpu_repro_v4flash.py
   both` + `quant_fp4_dequant_check.py`. NOTE: CPU has mesh=None ⇒ host-gather/forward-compile/multihost
   load NOT exercised; runner-only changes aren't either. gmm_v2 is a Pallas/Mosaic kernel — may be
   interpret-only on CPU.
2. **TPU microbench** (`scripts/perf_microbench*`).
3. **Full smoke + GATE.** Load ~40s/host; cold forward compile 10-30 min. `VLLM_ENGINE_READY_TIMEOUT_S=2400`.

HLO-diff helper (for a forward *scheckne* — NOT this OOM; durable for future):
```
for ip in 8 17 16; do rsync -az --include='*.before_optimizations.txt' --exclude='*' \
  -e "ssh -i ~/.ssh/google_compute_engine" enyouki@10.164.0.$ip:/tmp/hlo_dump/ /tmp/hlo_cmp/h$ip/; done
# per host: ls .../*.before_optimizations.txt | sed -E 's#.*/module_[0-9]+\.##;s#\.cl_[0-9]+.*##' | sort|uniq -c
# ⚠️ COMPARE WORKER-to-WORKER (h8/h17/h16), NOT h15 (mixes driver + co-located rank-0 worker).
```

---

## INFRA STATUS / v6e-16
- Slice `v6spoteu719`, zone `europe-west4-a`, project `prm-research`. **v6e-16, topology 4×4**, 16
  chips, 4 hosts: head `10.164.0.15` + workers `10.164.0.8 / .17 / .16`. Ray healthy (16 TPU). TP=16.
- ⚠️ **WEIGHTS MUST be readable by enyouki on ALL 4 hosts** (Q.10). Bringup mounts the bucket enyouki-
  owned only on the HEAD; workers get root-only → EACCES → silent dummy-zeros load → wedge. **Run
  `scripts/full_slice_v4_mount_weights.sh` once per bringup** (idempotent; the smoke pre-flight runs it;
  mounts do NOT survive reboot). Verify: `config.json` readable on .8/.17/.16.
- ⚠️ **numpy MUST be `<2.4` (pinned `2.3.5`)** — 2.4.x breaks `import numba`, crashes the APIServer.
  `~/.local/bin/uv pip install --python work/vllm_env/bin/python3 'numpy==2.3.5'` per host.
- ⚠️ Ray "version mismatch" = mark's rogue ray container; **FIX = keep BOTH guardians alive**
  (`ps -eo pid,cmd | grep -E 'node_guard[i]an|meta_guard[i]an'`; restart per the loop prompt,
  meta_guardian needs `10.164.0.15:6379`). Ray (re)start: `scripts/full_slice_v4_ray_restart.sh`.
- After ANY code edit: `full_slice_v4_sync.sh` (rsync to 4 hosts; `git push` does NOT sync them) +
  clear `~/.cache/vllm/xla_cache/*` on all 4 + verify md5 head==workers. Shut down ONLY via
  `full_slice_v4_reset.sh`. HLO dump propagates to workers via the ray env (XLA_FLAGS).

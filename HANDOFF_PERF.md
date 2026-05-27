# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign

> **Phase = PERFORMANCE.** S1 (decode *correctness*/determinism) is CLOSED — that is now a
> HARD REGRESSION GATE, not the goal (see §S1-GATE). The job: make prefill + decode FAST on
> the v6e-32 slice without breaking determinism. This doc is the loop's memory — current
> state, the roadmap, the ONE next action. Durable slice ops: `CLAUDE.md`. S1 history:
> `HANDOFF_S1.md` / `CLAUDE.full.md`.
>
> **One-line status (2026-05-27, P0):** Profile VERIFIED (traces re-parsed independently);
> roadmap below is fresh from a 14-agent investigation. **Nothing started yet.**
> **NEXT ACTION → Phase 0.0: build the TPU micro-benchmark harness (§NEXT), then Phase 0.1.**

---

## The verified profile (independently re-parsed from the real traces)

Bottleneck = the **sparse-attention KV gather**, `jnp.take_along_axis` at
`work/tpu-inference/tpu_inference/layers/jax/attention/deepseek_v4_attention.py:186` (inside
`sparse_attn`). Pure memory movement (`model_flops=0`) at **~0.02–0.05 % of HBM bandwidth**.

| | Decode (121.9 ms/tok) | Prefill (120.6 s for a short prompt) |
|---|---|---|
| `sparse_attn` gather (`:186`) | **65.8 %** | **99.0 %** |
| all-reduce (2176 ops) | 13.0 % | <0.5 % |
| MoE einsum | 10.6 % | <0.5 % |
| indexer top-k `while` (`lax.top_k`) | 9.3 % | <0.5 % |
| other collectives | 4.5 % | <0.5 % |

- Decode gathers are **latency-bound** (tiny outputs; 43 separate op launches/step) → win =
  collapse per-layer op-launch overhead by fusing. Prefill gathers are **bandwidth-bound**
  (805 MB fp32 tensors; the `:181` whole-KV fp32 cast doubles traffic) → win = kill the
  materialization + fp32 cast.
- The old "0.31 tok/s" headline was prefill-dominated; steady-state decode is ~8 tok/s.

**🔑 Major finding the original handoff missed:** the serve-time prefill runs the ENTIRE
transformer body TWICE — `transformer_body_forward` (`deepseek_v4.py:851`, produces `h`) +
`transformer_body_init_state_to_buffer` (`:854`, produces a BYTE-IDENTICAL `h` that is
discarded, plus the decode state). XLA does not CSE across the boundary. Corroborated by the
trace: 84 gathers ≈ (21 CSA + 20 HCA) × 2 passes. **Killing one pass ≈ halves prefill
(+ prefill MoE + collectives) for free, no kernel.** Highest ROI item in the campaign.

**We are architecturally faithful to the GPU reference.** Component-by-component vs the
PyTorch oracle (`tests/models/jax/_deepseek_v4_reference/`) and vLLM-GPU
(`work/vllm/.../models/deepseek_v4.py`): MLA, RoPE/YaRN, compressor, indexer, sink-softmax
(I14), hyper-connections, sqrtsoftplus+bias+hash routing, shared experts — ALL match. This
is pure perf + cleanup; there is no correctness gap to chase. The GPU does the two hot paths
with **FlashMLA sparse fused kernels (gather-in-kernel) + fp8 paged KV** and **sparse grouped
GEMM MoE**; we materialize the gather and run dense all-256 MoE.

---

## THE ROADMAP (ordered by ROI × safety — drive top-down)

Every committed change MUST clear the §S1-GATE. Validate on the CHEAPEST tier that can
answer the question first (see `CLAUDE.md` "How to validate"). Reserve full smokes.

### Phase 0 — quick wins, no kernel (do first)
- **0.0 Build the TPU micro-benchmark harness** (enabling work — do this FIRST, it unblocks
  cheap iteration for everything else). A script that jits + times `sparse_attn` (and later
  the new kernel) on the real slice mesh with SYNTHETIC q/kv/topk_idxs/attn_sink at realistic
  shapes — **NO 543 GiB weight load, no full-model compile.** Lets you measure gather
  ms/bandwidth and kernel speedups in ~1 min instead of a 25–45 min full smoke. Also wire a
  CPU numerics check vs `sparse_attn_torch`. Commit it as `scripts/perf_*`.
- **0.1 Eliminate the duplicate prefill body** — take `h` from Pass B, drop Pass A.
  ~2× prefill (+ halves prefill MoE/collectives). Preserve the `state_init_ids`/n_real vs
  padded-`input_ids` split (`deepseek_v4.py:835-840`). Refs: `:851` (Pass A) vs `:854`.
- **0.2 bf16 gather** — drop `kvf = kv.astype(fp32)` at `:181`, gather bf16, upcast inside
  the einsums (`:189`,`:199`). KV is already bf16; oracle starts from bf16 → numerically
  neutral. ~2× prefill / ~1.45× decode on the bottleneck op. Validate: existing CPU test
  `tests/models/jax/test_deepseek_v4.py -k sparse_attn`.
- **0.3 Delete dead code:** `_consolidate_moe_after_load` (`deepseek_v4.py:1776-1830`, zero
  callers, *and* it's the buggy device-reshard path S1 removed) + `_QUANT_SUFFIXES` (`:1269`).

### Phase 1 — the fused sparse-attention kernel (the main prize: 50–100× prefill, 2–3× decode)
Replace the materialized gather with a tiled Pallas kernel doing gather + online-softmax +
sink, never materializing `[B,M,K,D]`. ONE kernel serves both call sites (decode `:812` M=1,
prefill `:905` M=S) — prefill/decode differences are entirely in `topk_idxs` (the kernel just
honors the `-1` mask). **Base decision (agents converged):** flash_attention's online-softmax
+ sink loop × mla/v2's single-shared-KV / q-head-collapse einsum × an index-driven VMEM
gather modeled on mla/v2 `_fetch_bkv`. **Skip `sparse_core/`** (gen-7-gated, v6e is gen-6,
falls back to plain gather) and `ragged_paged_attention/v3` (non-MLA/paged). No drop-in
exists. See §KERNEL for the full contract. Validate vs `sparse_attn_torch` on CPU + the
microbench (0.0), THEN the §S1-GATE.

### Phase 2 — collectives (~17 % decode)
The 2176 all-reduces = replicated decode activation (`_v4_decode_replicate`, the S1 fix —
DO NOT remove) × `attn_dp`-sharded weights on their CONTRACTING dim. **Highest-leverage/
lowest-risk:** flip `pick_partition_spec` (`deepseek_v4_loader.py:497`) so weights shard on
their OUTPUT dim (`wq_a`, `wkv`, shared-expert `w1`/`w3`, compressor/indexer
`wkv`/`wgate`/`weights_proj`) + replicate the tiny gate → eliminates ~6–8 of ~10
all-reduces/layer. It's a load-time placement (no in-jit `wsc`) → cannot trip pitfall #5.

### Phase 3 — MoE + indexer top-k (secondary; profile says ~10 %/~9 %, NOT the 97 % FLOPs implied)
- 3.1 Drop the fp32 cast in the dense decode MoE (`deepseek_v4_moe.py:220-222`). Low risk.
- 3.2 Indexer `lax.top_k` → `while` (9.3 %): sorts the full buffer (`state_max_seq_len//4`)
  even when few slots valid. `approx_max_k` (fast, but APPROXIMATE → determinism risk, re-gate)
  or bound the sort length. Only the 21 CSA layers run it.
- 3.3 Sparse top-6 MoE dispatch — N=1 can't shard over attn_dp=32; only safe route is a
  replicated top-6 `gmm_v2`. Measure before committing; payoff is uncertain.

### Phase 4 — serving correctness (blocks real serving, not perf-internal)
- 4.1 **Long/multi-turn chat wedge (HIGH).** Root cause = MoE `use_shard_map` gate
  (`deepseek_v4_moe.py:211`) flipping True at a larger N bucket, entering `_routed_local`
  shard_map (`:317`) for the first time; the `concatenate` (`:272`) is just the trace site.
  Will be RE-TRIGGERED by the Phase-1 recompile → handle together. Capture full traceback,
  make MoE path selection shape-stable / warm the larger buckets.
- 4.2 Tiny smoke config (256 ctx/1 seq) is config-only; raise `MAX_LEN`/`max-num-seqs` to
  reproduce 4.1 and serve real context.
- 4.3 `seed`+sampling → 400 is an upstream limitation (`tpu_platform.py:358`); low priority.

### Phase 5 — de-hack / shrink the diff (AFTER the kernel lands)
Diff vs upstream is already lean (10 files rsync; vllm pristine). Then: remove
`_v4_nan_tripwire` (~41 sites; keep it until the kernel is validated — it's the numerics
tool); audit the two non-reference clamps (`_linear` `|r|<1e8→0` at
`deepseek_v4_attention.py:470`, applied inconsistently; `compute_logits` `nan_to_num` at
`deepseek_v4.py:2055`) — instrument whether `nan_to_num` ever fires before removing. Trim
S1-narrative comments. DO NOT attempt the "read like qwen3" reuse rewrites (V4-specific
and/or fused with the S1 fix — all flagged unsafe).

---

## NEXT ACTION (for the session reading this)
1. Read this doc + `CLAUDE.md` (slice ops, validation tiers, pitfalls).
2. **Build Phase-0.0 (the TPU microbench harness)** — it unblocks cheap iteration for the
   whole campaign and is itself a clean committable step. Then drive **0.1** (the free ~2×
   prefill) and **0.2** (bf16 gather) — both gate-cheap.
3. Commit + push after each validated step. Hand off when context grows (see CLAUDE.md).

---

## <a name="S1-GATE"></a>S1 REGRESSION GATE (non-negotiable for every change)
- `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (visible_words ≥ 10,
  max_word_run < 5).
- FIB decode md5 **byte-identical across 2 fresh engines** + **correct Fibonacci** (21, 34,
  55, 89, 144) vs the prefill-everything reference. Current reference md5 = `5bf42256`. A
  numerics-changing kernel MAY shift the md5 → re-establish a NEW reference and confirm it's
  identical across 2 fresh engines + correct Fibonacci. The non-negotiable is **identical
  across engines + correct Fibonacci**, not the specific hash.
- READ the actual decode text — "contains Paris" is a known false positive (can EOS at tok 1).
- Probe: `python3 /tmp/s1_probe2.py N` (FIB decode, prints md5 + text; N = max_tokens).

---

## <a name="KERNEL"></a>Kernel contract (Phase 1 — for the implementer)
`sparse_attn(q[B,M,H,D] bf16, kv[B,N,D] bf16, attn_sink[H] fp32, topk_idxs[B,M,K] int32
(-1=ignore), softmax_scale: float) -> out[B,M,H,D] bf16`. SINGLE KV head shared across all
H=64 q heads (gather once per (b,m)-tile, reuse across H). D=512, scale=1/sqrt(512).
- **Math (preserve exactly; oracle `kernel_stubs.py:60`, invariant I14):** logits =
  (q·kv_gathered)*scale; mask `-1` slots out; running max INCLUDES the per-head sink
  (`m=max(max_k valid_logit, attn_sink[h])`); `m=0` if non-finite (all-masked-row guard);
  denom = Σ_k exp(logit−m)[valid] + exp(attn_sink[h]−m); out = Σ_k softmax·kv_gathered / denom.
  Sink adds to the DENOMINATOR only (no sink value vector).
- **fp32 accumulation REQUIRED** (bf16-throughout unsafe; K up to 640). Read bf16, accumulate
  matmuls + softmax in fp32, cast out to bf16. Preserve `max(idx,0)` clamp; fixed K-reduction
  order; no uninit-HBM reads (deterministic).
- **K/N per layer flavor** (compress_ratio 0/4/128): decode K = 128 (SWA) / 640 (CSA) /
  128+msl/128 (HCA); prefill K = min(S,128) + (min(index_topk,S/4) CSA | S/128 HCA). Static
  shape params.
- **Pitfall #5:** no `with_sharding_constraint` that gathers the size-1 decode token axis.
- **Templates:** `kernels/flash_attention/kernel.py:82` (softmax+sink loop),
  `kernels/mla/v2/kernel.py` (`:335`/`:401` online-softmax, `:1119` pipeline, shared-KV
  einsum), oracle `tests/models/jax/_deepseek_v4_reference/kernel_stubs.py:60`.

---

## Reproducing the profile (durable recipe)
Add `${V4_PROFILER_ARGS:-} \` to the `vllm serve` line in `scripts/full_slice_v4_smoke.sh`
(before `> "$LOG"`), then:
```bash
mkdir -p /home/enyouki/v4_traces   # head + all 7 workers
V4_PROFILER_ARGS="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/home/enyouki/v4_traces" \
  bash scripts/full_slice_v4_smoke.sh
```
After ready: warm (`python3 /tmp/s1_probe2.py 2`), then `curl -sX POST :18081/start_profile`
→ `python3 /tmp/s1_probe2.py 20` → `curl -sX POST :18081/stop_profile`. Traces land
PER-WORKER at `/home/enyouki/v4_traces/plugins/profile/<ts>/<host>.trace.json.gz` (~750 MB
unzipped — stream with ijson; group "XLA Ops" on the `/device:TPU:0` process by `name`, sum
`dur`; `Decimal`→float). **Caveat:** the 1M XLA-op cap captured only 1 prefill + 1 decode
step — fine for the structural breakdown; for the decode-vs-context cost curve, profile a
decode-only window at a longer context.

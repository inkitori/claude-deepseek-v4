# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign

> **Phase = PERFORMANCE.** S1 (decode correctness/determinism) is CLOSED — now a HARD
> REGRESSION GATE, not the goal (§S1-GATE). Job: make prefill + decode FAST on the v6e-32
> slice without breaking determinism. This doc is the loop's memory — current state, the
> roadmap, the ONE next action. Durable slice ops: `CLAUDE.md`. S1 history: `HANDOFF_S1.md`.
>
> **One-line status (2026-05-27):**
> - **Phase 1 CLOSED & GATED** — fused sparse-attn kernel (`kernels/sparse_attn/kernel.py`, wired
>   via `_sparse_attn_kernel_sharded`): attn KV gather 65.8%→**0.2%** of decode.
> - **Phase 1.E (90e7c520): CPU numerics oracle RESTORED** — the `_sparse_attn_kernel_sharded`
>   `mesh.empty` branch now calls pure-JAX `sparse_attn` (the Mosaic kernel is interpret-only on
>   CPU). TPU-NEUTRAL (real-slice `mesh.empty` is False → TPU program byte-identical → S1 safe,
>   no smoke needed). `s1_cpu_repro_v4flash.py both` passes again — **Tier-1 validation is BACK.**
> - **Phase 2 (pick_partition_spec axis-0 flip) INVESTIGATED & DEPRIORITIZED** — a 3-angle analysis
>   (this session) shows it's a marginal/lateral move, possibly net-NEGATIVE, NOT the clean
>   all-reduce→all-gather win the old doc assumed. **DO NOT smoke it as specified.** (§Phase 2.)
> - **NEW top item = Phase 3.1 (drop MoE decode fp32 cast — small, tractable, SURE win), then
>   Phase 3.2 (indexer top_k — biggest lever esp. at long ctx, but needs design).**

---

## The verified profile (decode, post-kernel re-profile)

Phase 1's fused kernel killed the attention KV gather (`take_along_axis`, was 65.8% of decode →
0.2%). Decode is now collective + MoE + indexer bound. **Decode breakdown** (clean decode-only
step, `2026_05_27_08_46_30` trace, tiny ~30-tok FIB ctx; `scripts/perf_parse_trace.py --bucket-ops`):

| decode op (one step, ~50 ms busy) | % | roadmap |
|---|---|---|
| all-reduce (+psum) (2176/step) | **31.7 %** | Phase 2 — **DEPRIORITIZED** (marginal; §Phase 2) |
| MoE dense `multiply_reduce_fusion` | 25.2 % | **Phase 3.1** (fp32 cast — tractable) + 3.3 |
| indexer top_k `lax.top_k` | 23.3 % | **Phase 3.2** (biggest long-ctx lever; needs design) |
| all-gather | 7.1 % | decode residual (big all-gather is prefill-only) |
| collective-permute | 4.9 % | Phase 2 |
| `sparse_attn_kernel` (fused attn) | 0.2 % | ✅ Phase 1 done |

⚠️ **Capture caveats:** tiny ctx (~30-tok FIB) ⇒ MoE share INFLATED (falls at long ctx). The
indexer top_k buffer is STATIC (`T = state_max_seq_len//4`; **T=2048 in the smoke**, T=131072 at
16k ctx) so top_k cost is context-robust WITHIN a config but SCALES with max-model-len (could
DOMINATE prod decode). all-reduce is context-robust. Only ~2 decode steps captured but they agree.

**We are architecturally faithful to the GPU reference** (MLA, RoPE/YaRN, compressor, indexer,
sink-softmax I14, hyper-connections, routing, shared experts ALL match the PyTorch oracle +
vLLM-GPU). Pure perf + cleanup; no correctness gap. The GPU uses FlashMLA sparse fused kernels +
fp8 paged KV + sparse grouped-GEMM MoE; we have the fused gather kernel + dense all-256 MoE.

---

## THE ROADMAP (re-ranked by ROI × safety × tractability — drive top-down)

Every committed change MUST clear §S1-GATE. **Validate on the CHEAPEST tier first — the CPU oracle
is BACK (Tier 1).** Reserve full smokes (≤1-2/session; COLD = 25-45 min on a `.py` change).

### ✅ DONE (history)
- **0.0** TPU microbench harness `scripts/perf_microbench.sh` (synthetic sparse_attn, no weights).
- **0.1** (`d22df61a`) Killed the duplicate prefill body (~halved the prefill body).
- **0.2/0.3** (`2839a684`/`56abe232`) bf16 sparse_attn gather (kills whole-KV fp32 copy) + deleted
  dead `_consolidate_moe_after_load`.
- **1.A–1.D** Fused Pallas sparse-attn kernel authored, wired, S1-gated, re-profiled (gather→0.2%).
- **1.E** (`90e7c520`) CPU oracle restored (above).

### Phase 2 — DEPRIORITIZED: `pick_partition_spec` axis-0 flip is marginal/lateral
The old doc assumed flipping attn weights from contracting-dim (axis-1) to OUTPUT-dim (axis-0)
sharding would swap the 31.7% decode all-reduce for a ~2× cheaper all-gather. **3-angle analysis
says NO:**
1. **Code (consumer trace):** every all-reduce-producing matmul (`wq_a`, `wkv`, `wo_b`, compressor/
   indexer down-projs) feeds a RMSNorm/RoPE/residual that needs the FULL feature vector immediately
   → axis-0 just turns each all-reduce into an EQUAL-payload all-gather; no chain-through. The
   all-reduce is structural to the replicated-activation S1 fix (`_v4_decode_replicate`), not the
   weight axis. (Sharding axis is `attn_dp`=32, not `model`=1; `wq_b`/`wo_a`/`idx.wq_b`/`w2` are
   ALREADY axis-0.)
2. **Cost regime:** decode is LATENCY/launch-bound (N=1, 2176 tiny collective LAUNCHES). The flip
   doesn't change launch COUNT, only bytes-per-launch — the term that does NOT dominate at N=1.
3. **Concrete regression:** the o-path has `wo_a` ALREADY axis-0 (out=8192) feeding `wo_b`. The
   global flip makes `wo_b` axis-0 too → it needs its full 8192 input replicated → all-gather(8192)
   + all-gather(4096) replacing one all-reduce(4096) = MORE launches + MORE bytes.

**Verdict: do NOT smoke the axis-0 flip.** The diff (largest-dim→first-dim in
`deepseek_v4_loader.py:508-519`) is in this file's git history (`c7123c1e`) if ever revisited. The
REAL lever for the 31.7% all-reduce is cutting collective LAUNCH COUNT (fuse collectives, or rethink
the `_v4_decode_replicate` strategy) — both hard/research; deferred behind the tractable Phase-3 wins.

### Phase 3.1 — drop the MoE decode fp32 cast (NEW #1: tractable, low-risk, small but SURE)
`deepseek_v4_moe.py:220-224` casts `flat_x`/`W1`/`W3` to fp32 before the two dense-decode gate-up
einsums (the prefill `gmm_v2` path already runs bf16-in/fp32-accumulate since S24). Edit:
```python
gate_NEi = _shard_e_mid(jnp.einsum('nd,eid->nei', flat_x, W1))   # drop x_fp32/W1_fp32
up_NEi   = _shard_e_mid(jnp.einsum('nd,eid->nei', flat_x, W3))   # drop W3_fp32
```
Halves W1+W3 HBM streaming on the two biggest MoE einsums (bandwidth). **NOT bit-identical** (bf16
matmul vs fp32 = ULP shift, like 0.2) → **md5 re-baseline**: expect correct Fib + a NEW N=2 md5
identical ×2 engines. Risk LOW (prefill proves the math; SwiGLU inter_dim=2048 won't overflow bf16;
`swiglu_limit` clamp backstops). Validate: CPU oracle (no breakage) → smoke + S1 gate (new md5).
Expected ~2.5-4% decode at FIB ctx, less at long ctx. (Shared-expert `expert_forward:127-128` also
casts fp32 — separate, leave for now.)

### Phase 3.2 — indexer top_k (biggest lever, esp. long-ctx — but NEEDS DESIGN)
`indexer_decode_step` (`deepseek_v4_attention.py:~645`) runs `lax.top_k(index_score, K=512)` over a
STATIC buffer `T = state_max_seq_len//4` (**T=2048 in smoke** [state_max_seq_len = max_num_batched_tokens×dp = 256×32 = 8192]; **T=131072 at 16k ctx**), ×21 unrolled CSA layers/step. Cost = XLA TopK
scanning the full T even when few slots valid → 23.3% at tiny ctx; would DOMINATE at prod ctx (T
scales 64×).
- ❌ **arange short-circuit REFUTED:** only bit-identical when T==K=512 (full sort = identity); T==K
  needs state_max_seq_len=2048 (≤64 tok/rank) — never in practice (smoke T=2048≠512). The top_k
  genuinely selects top-512-BY-SCORE of T candidates; arange returns the first 512 by POSITION = wrong.
- **Real options (evaluate next):** (a) `jax.lax.approx_max_k` — deterministic per (input,shape) so
  MAY pass S1 (re-baseline md5), but APPROXIMATE selection → must verify FIB QUALITY doesn't degrade;
  (b) bound the sort to a static WINDOW ≤ current position (vllm buckets ctx lengths → top_k over a
  bucket-sized `dynamic_slice` not the full T; jit-feasibility TBD); (c) leave it. Numerics-changing
  → smoke + S1 re-gate. HIGH value, MEDIUM-HARD.

### Phase 3.3 / 4 / 5 (later)
- **3.3** Sparse top-6 MoE dispatch — N=1 can't shard over attn_dp=32; only a replicated top-6
  `gmm_v2` is safe. Measure before committing; payoff uncertain.
- **4.1** Long/multi-turn chat WEDGE (HIGH for real serving) — MoE `use_shard_map` gate
  (`deepseek_v4_moe.py:211`) flips True at a larger N bucket → first entry to `_routed_local`
  shard_map; the Phase-1 recompile may re-trigger it. Make MoE path selection shape-stable / warm
  the larger buckets. **4.2** raise smoke `MAX_LEN`/`max-num-seqs` to reproduce 4.1.
- **5** De-hack/shrink the diff (AFTER perf): remove `_v4_nan_tripwire` (~41 sites; the kernel is now
  validated so it's removable), audit the two clamps (`_linear` `|r|<1e8→0`; `compute_logits`
  `nan_to_num` — instrument whether it ever fires), trim S1-narrative comments. Do NOT "make
  idiomatic" the loader/MoE/seed paths fused with the S1 fix.

---

## NEXT ACTION (for the session reading this)
**Phase 3.1 — drop the MoE decode fp32 cast.** Tractable next win; the CPU oracle is BACK to validate
it cheaply.
1. Apply the 3.1 edit (`deepseek_v4_moe.py:220-224`, above).
2. CPU oracle: `PYTHONPATH=work/tpu-inference:work/vllm work/vllm_env/bin/python3
   scripts/s1_cpu_repro_v4flash.py both` → expect "OK: both eager and jit match" (catches shape/jit
   breakage/NaN; the numerics shift is fine — eager+jit shift together, so this won't flag md5 drift).
3. `scripts/full_slice_v4_sync.sh` + clear `~/.cache/vllm/xla_cache/*` on all 8 hosts (a `.py`
   changed → COLD compile). Then smoke + S1 gate: **correct Fib + a NEW N=2 md5 identical ×2 fresh
   engines** (the bf16 shift WILL change the hash → re-baseline, exactly like 0.2). READ the FIB text.
   Re-profile optional (confirm MoE % fell).
4. Commit + push. Then tackle Phase 3.2 design (approx_max_k FIB-quality eval, OR a bucketed-window
   sort).
5. **Do NOT smoke the Phase-2 axis-0 flip** — it's deprioritized (§Phase 2).
6. Hand off when context grows (CLAUDE.md "CONTEXT HANDOFF PROTOCOL").

---

## <a name="S1-GATE"></a>S1 REGRESSION GATE (non-negotiable for every change)
- `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (visible_words ≥ 10,
  max_word_run < 5).
- FIB decode: **correct Fibonacci** (21, 34, 55, 89, 144 — DETERMINISTIC) + **N=2 md5 `5bf42256`
  byte-identical across 2 fresh engines** (`s1_probe2.py 2`, = md5("21,")). ⚠️ The long-tail md5
  (`s1_probe2.py 20`+) is NON-deterministic at temp=0 (pre-existing decode nondeterminism — the
  distributed all-reduce-ordering residual) — do NOT gate on it. A numerics-changing fix (e.g. 3.1)
  may shift even the N=2 md5 → re-establish + confirm identical ×2 engines + correct Fibonacci.
  Non-negotiable = **identical ×2 engines (at N=2) + correct Fibonacci**, not a specific hash.
- READ the actual decode text — "contains Paris" is a known false positive (can EOS at tok 1).
- Probe: `python3 /tmp/s1_probe2.py N` (FIB decode, prints md5 + text; N = max_tokens).

---

## Reproducing the profile (durable recipe)
Add `${V4_PROFILER_ARGS:-} \` to the `vllm serve` line in `scripts/full_slice_v4_smoke.sh` (before
`> "$LOG"`), then:
```bash
mkdir -p /home/enyouki/v4_traces   # head + all 7 workers
V4_PROFILER_ARGS="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/home/enyouki/v4_traces" \
  bash scripts/full_slice_v4_smoke.sh
```
After ready: warm (`s1_probe2.py 2`), then `curl -sX POST :18081/start_profile` → `s1_probe2.py 20`
→ `curl -sX POST :18081/stop_profile`. Traces land PER-WORKER at
`/home/enyouki/v4_traces/plugins/profile/<ts>/<host>.trace.json.gz` (~750 MB unzipped — stream with
ijson; group "XLA Ops" on `/device:TPU:0` by `name`, sum `dur`). Parser: `scripts/perf_parse_trace.py
--bucket-ops`. **Caveat:** the 1M XLA-op cap captures ~1 prefill + 1 decode step; profile a
decode-only window at LONGER ctx for the decode-vs-context cost curve (esp. for Phase 3.2 — top_k
scales with T).

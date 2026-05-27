# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign

> **Phase = PERFORMANCE.** S1 (decode *correctness*/determinism) is CLOSED — that is now a
> HARD REGRESSION GATE, not the goal (see §S1-GATE). The job: make prefill + decode FAST on
> the v6e-32 slice without breaking determinism. This doc is the loop's memory — current
> state, the roadmap, the ONE next action. Durable slice ops: `CLAUDE.md`. S1 history:
> `HANDOFF_S1.md` / `CLAUDE.full.md`.
>
> **One-line status (2026-05-27, P1 CLOSED — re-profiled):** Phase 0.* + **Phase 1 CLOSED & GATED.**
> Fused kernel `kernels/sparse_attn/kernel.py::sparse_attn_kernel` WIRED into both call sites via the
> fully-REPLICATED `_sparse_attn_kernel_sharded` shard_map (a Mosaic custom-call can't be
> SPMD-auto-partitioned; in/out specs all `P()`, `check_vma=False`, `mesh.empty`→direct for CPU/interpret).
> S1-GATED: FIB N=2 md5 `5bf42256` ×2 engines + correct Fib + smoke rc=0. **Re-profile this session
> (clean decode-only trace) CONFIRMS the kernel worked: sparse-attn gather 65.8%→0.2% of decode — attention
> is NO LONGER a decode bottleneck.** Isolated ~41× gave only ~1.25× e2e because the gather was one part and
> the onehot has NO bandwidth win at short ctx (reads same bytes as take_along; fusion/launch win only). **Decode
> is now ALL-REDUCE-bound (31.7%)** — breakdown below. NEXT = **Phase 2: flip `pick_partition_spec` to OUTPUT-dim
> (axis-0) sharding** (the doc's old "axis-1" was BACKWARDS — corrected in §Phase 2). Tools landed: parser
> `scripts/perf_parse_trace.py` (`--bucket-ops`); profiler wired via `V4_PROFILER_ARGS` in smoke.sh.

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

**POST-KERNEL decode breakdown (this session, clean decode-only step isolated by ts from the
`2026_05_27_08_46_30` trace; `scripts/perf_parse_trace.py --bucket-ops`):** attention gather is GONE.

| decode op (one step, ~50 ms busy) | % | roadmap |
|---|---|---|
| **all-reduce (+psum)** (2176/step) | **31.7 %** | **Phase 2 (next)** |
| MoE dense `multiply_reduce_fusion` | 25.2 % | Phase 3 (riskier — chat-wedge gate) |
| **indexer top_k `lax.top_k` `while` loop** | **23.3 %** | Phase 3.2 (high-conf, the loop not the op) |
| all-gather | 7.1 % | (decode residual; the BIG all-gather is PREFILL-only, 57 ms — see §prefill-map) |
| collective-permute | 4.9 % | Phase 2 |
| **`sparse_attn_kernel` (fused attn)** | **0.2 %** | ✅ Phase 1 done — negligible now |
| non-attn gather (MoE/compressor/seed) | ~0 % decode | — |

⚠️ **Capture caveats:** tiny ctx (~30-tok FIB) ⇒ MoE share inflated / attention deflated vs long ctx;
all-reduce + top_k are context-ROBUST, MoE 25% will fall at longer ctx. Only ~2 decode steps captured
(profiler window closed ~603 ms) but the 2 steps agree. `multiply_reduce_fusion`→MoE is med-high conf
(XLA fused the label; could include some norm/softmax reduces).

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
- **0.0 TPU microbench harness — ✅ DONE.** `scripts/perf_microbench.sh --all` (launcher:
  pre-cleans every host, fans `perf_microbench_sparse_attn.py` across 8 hosts via mh_run,
  `--distributed`). Times the REAL `sparse_attn` at decode/prefill shapes (3 layer flavors
  swa/csa/hca) with synthetic inputs — NO weight load. Read **process-0's** tail for the
  table. CPU numerics gate = `scripts/perf_microbench.sh --cpu-check` (existing pytest).
  Baseline measured: **decode_csa 5.44 ms** to move 2.4 MB ⇒ ~0.4 GB/s (~0.025% HBM bw) —
  reproduces the profile. ⚠️ **Slice ops:** a lone host CAN'T boot the v6e-32 TPU → MUST run
  multi-host. `jax.distributed.initialize()` is a coin-flip + a failed run leaves JAX procs
  stuck (ignore SIGTERM) that poison the next init — the launcher SIGKILLs `[p]erf_microbench`
  + clears lockfiles first, but on a hang **retry** (clean is baked in). MUST sync the script
  to all 8 hosts first (mh_run runs each host's clone).
- **0.2 bf16 gather — ✅ CLOSED (committed `2839a684`, gated 2026-05-27).** Killed `kvf =
  kv.astype(fp32)` at `:181`; gather now reads bf16 `kv` (`deepseek_v4_attention.py:189`) and
  `.astype(fp32)` upcasts the *gathered result* before both fp32 einsums. Math is bit-identical
  (CPU 2/2) — **but on TPU it shifted the FIB md5 `5bf42256`→`b675be27`** (deterministic ULP
  change: moving the upcast after the gather makes XLA pick a different MXU matmul accumulation
  order). Gated: md5 `b675be27` byte-identical ×2 fresh engines + correct Fibonacci +
  smoke_check rc=0 (visible_words=45). Reference rebaselined. The numerics win (kill the
  whole-KV fp32 copy ⇒ ~½ prefill gather traffic) is the point; ULP shift accepted.
  ↪ *Not yet quantified on the microbench (cheap follow-up; decode_csa baseline = 5.44 ms;
  decode is latency-bound so expect the win mostly on the prefill shapes).*
- **0.3 Delete dead code — ✅ CLOSED (in `2839a684`).** Removed `_consolidate_moe_after_load`
  (zero callers) + `_QUANT_SUFFIXES`. Behavior-neutral (dead code can't move numerics); gated
  in the same smoke as 0.2.
- **0.1 Eliminate the duplicate prefill body — ✅ DONE & COMMITTED (2026-05-27).** Prefill now
  runs ONE body: `deepseek_v4_run_with_decode_state` returns Pass B's `h`
  (`transformer_body_init_state_to_buffer`), dropped the standalone Pass-A
  `transformer_body_forward`; deleted dead `state_init_ids` param + call-site slicing (net −31
  lines, decode path UNTOUCHED). ~halves the ~120 s prefill body. Gate + the tail-nondet finding
  in §0.1-DONE.

### Phase 1 — the fused sparse-attention kernel (the main prize) — ✅ WIRED + S1-GATED
`kernels/sparse_attn/kernel.py::sparse_attn_kernel` (math LOCKED, CPU parity 4/4, committed
`f170a55d`/`ace9d576`): one Pallas program per (b,m), gather K rows via ONE-HOT MATMUL
(bit-identical to `take_along_axis`; a dynamic per-row `pl.ds` load does NOT lower — E2003), fp32
single-pass softmax + per-head sink + `-1` mask, bf16 in/out, single shared KV across H=64.
**Wired via `_sparse_attn_kernel_sharded` (:209; both call sites: decode `:844`, prefill `:937`)** — a
fully-REPLICATED `shard_map` (in/out all `P()`, `check_vma=False`, `mesh.empty`→direct). Needed
because a Mosaic custom-call can't be SPMD-auto-partitioned in the sharded jit; the microbench
MISSED this (it runs the kernel in isolation, not inside the model mesh). GATE PASSED (md5
`5bf42256` ×2 engines + correct Fib + smoke rc=0). **⚠️ ~41× was the ISOLATED gather op; end-to-end
decode only ≈1.25× (≈10 vs ~8 tok/s)** — see NEXT ACTION (the remaining Phase-1 work is realizing
the win, not the kernel).
- Onehot READS ALL N → for large-decode-msl, switch gather to the DMA idiom (kernel docstring +
  §KERNEL). Prefill currently all-gathers to replicated (shard_map `P()`) → shard the map over the
  prefill token axis to avoid the all-gather + 32× redundant prefill compute (perf follow-up).

### Phase 2 — collectives: decode all-reduce (31.7 % — the re-profiled #1 decode cost)
The 2176 all-reduces/step = replicated decode activation (`_v4_decode_replicate`, the S1 fix — DO
NOT remove) × `attn_dp`-sharded weights on their CONTRACTING dim. **Highest-leverage/lowest-risk:**
flip `pick_partition_spec` (`deepseek_v4_loader.py:468-519`) so attention weights shard on their
OUTPUT dim. Load-time placement (`make_array_from_callback`, `:560-569`; no in-jit `wsc`) → cannot
trip pitfall #5.
- **CORRECTED mechanism (the old doc had the axes BACKWARDS):** weights are stored `[out, in]` and
  consumed as `x @ W.T` (`_linear`, `deepseek_v4_attention.py:489/:502`), so **axis 0 = OUTPUT, axis
  1 = CONTRACTING**. Current code picks the LARGEST divisible dim; the wide attn weights have
  `in=4096 > out`, so it picks **axis 1 = CONTRACTING ⇒ partial sums ⇒ all-reduce**. Fix = prefer
  **axis 0 (output)** = the FIRST divisible dim, so the matmul output is already sharded (a downstream
  all-gather replaces the all-reduce). NOT "axis 1".
- **The diff** (`:508-518`, replace the largest-dim loop):
  ```python
  best_dim = -1
  for i, d in enumerate(shape):
      if d % chosen_size == 0 and d >= chosen_size:
          best_dim = i; break   # axis 0 (output) preferred; later dim only if axis 0 can't carry it
  if best_dim == -1:
      return P()                # replicate (e.g. hc_*_fn out=24, indivisible by 32)
  spec = [None] * len(shape); spec[best_dim] = chosen_axis
  return P(*spec)
  ```
  Flips wq_a/wkv/wo_b + compressor/indexer projections to axis-0; leaves already-axis-0 weights
  (wq_b/wo_a/idx.wq_b/expert.w2) unchanged; auto-replicates `hc_*_fn` (out=24). No weight has a
  size-1 output dim ⇒ no decode-token hazard.
- **Validate (CPU, before the cold smoke):** `scripts/s1_cpu_repro_v4flash.py both` ("OK … match") +
  `pytest tests/models/jax/test_deepseek_v4.py -k "shard or per_device_budget"` (forces cpu+32). Proves
  math+shapes only; the collective-count drop + S1 are TPU-only (full gate). [CPU-validation status: TBD]
- **LIMITS (don't over-claim):** the 25 % MoE (`multiply_reduce_fusion`) is governed by `_shard_e_mid`
  in `deepseek_v4_moe.py`, NOT this — so the win is the ATTENTION all-reduces only; `gate.weight`
  `[256,4096]` is sharded today (1M elems > `_MIN_SHARD_ELEMENTS`, NOT replicated); re-profile after
  to confirm the % actually fell (the all-reduce→all-gather trade isn't guaranteed net-positive).

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
Phase 1 is CLOSED (re-profiled — kernel works, attention gather 0.2% of decode). **Decode is now
ALL-REDUCE-bound (31.7%). NEXT = Phase 2: flip `pick_partition_spec` to OUTPUT-dim (axis-0) sharding.**
1. **CPU-validate the corrected diff FIRST** (cheapest tier): the §Phase 2 diff prefers the FIRST
   divisible dim (axis 0 = output) instead of the largest. Run the CPU oracle + shard tests (cmds in
   §Phase 2); confirm numerics unchanged + every shape divides. [worktree CPU-validation status: see §Phase 2]
2. **Verify the collective trade is a WIN before the cold smoke if you can:** the flip turns each decode
   all-reduce into a downstream all-gather (a replicated activation needs the full feature vector after an
   output-sharded matmul). On TPU all-gather is usually cheaper than all-reduce for equal bytes, but the
   MAGNITUDE IS UNVERIFIED — if a cheap multi-host HLO op-count check exists (s1_mh_repro-style), count
   all-reduce vs all-gather on the real mesh (no 543 GiB load) first.
3. **Then the smoke** (COLD: a `.py` change invalidates the xla_cache → 25-45 min compile) + the full S1
   gate. A numerics-orientation change MAY shift the md5 → rebaseline ×2 engines + correct Fib.
4. KNOWN LIMITS of the flip (audited): MoE collectives are governed by `_shard_e_mid` (NOT
   `pick_partition_spec`) so the win is concentrated in the ATTENTION chain, not the 25% MoE; `gate.weight`
   `[256,4096]` is NOT auto-replicated by `_MIN_SHARD_ELEMENTS` (1M elems — handoff was wrong); the change
   touches ALL leaves (embeddings/lm_head/MTP) so confirm no non-attn regression via the budget test.
5. After Phase 2: indexer top_k `while` loop (23.3%, Phase 3.2) is the next high-confidence target;
   prefill all-gather (57 ms) = shard the prefill map over the token axis (only bites at large prefill).
6. Commit + push after each validated step. Hand off when context grows (see CLAUDE.md).

## <a name="0.1-DONE"></a>Phase 0.1 — DONE (history) + smoke confounds the WIRING session will hit
0.1 (`d22df61a`): prefill runs ONE body (`deepseek_v4_run_with_decode_state` returns Pass B's `h`
from `transformer_body_init_state_to_buffer`; dropped the dup Pass-A `transformer_body_forward` +
dead `state_init_ids`). Decode provably byte-identical (same seed args + decode jit untouched).
The temp=0 FIB free-form-TAIL nondeterminism finding is folded into §S1-GATE (gate on N=2
`5bf42256`, NOT a long-tail md5 — the tail flips `e4d45024`↔`26354502` within one process;
pre-existing decode nondeterminism, = the Phase-2 collective-ordering residual).

**⚠️ Smoke ops for the wiring session (Phase 1 needs the full smoke):** (a) `smoke_check` can OOM
(256M HLO temp) compiling its shapes ON TOP of resident FIB-probe shapes on the memory-tight
slice → `EngineDeadError` + a ray-channel crash that can drop a node (recover via
`scripts/full_slice_v4_ray_restart.sh`). Run `smoke_check` FIRST on a clean engine if you need
rc=0; it uses the chat endpoint (Phase-4.1 wedge-prone). (b) After a `ray_restart`, `node_guardian`
can proliferate (~18 instances) — benign (idempotent 'node' occupation); do NOT `pkill
node_guardian` (the loop-prompt claude argv self-matches) — kill by PID only if problematic.

---

## <a name="S1-GATE"></a>S1 REGRESSION GATE (non-negotiable for every change)
- `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (visible_words ≥ 10,
  max_word_run < 5).
- FIB decode: **correct Fibonacci** (21, 34, 55, 89, 144 — DETERMINISTIC) + **N=2 md5
  `5bf42256` byte-identical across 2 fresh engines** (`s1_probe2.py 2`, = md5("21,")). ⚠️ The
  long-tail md5 (`s1_probe2.py 20`+) is NON-deterministic at temp=0 (pre-existing decode
  nondeterminism — §0.1-DONE) — do NOT gate on it; old refs (`b675be27`) sampled a
  nondeterministic tail. A numerics-changing kernel may shift even the N=2 deterministic md5 →
  re-establish + confirm identical ×2 engines + correct Fibonacci. Non-negotiable = **identical
  ×2 engines (at N=2) + correct Fibonacci**, not a specific long-tail hash.
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
- **Fork plan (agent-derived):** FORK `flash_attention/kernel.py` (its row-independent online
  softmax + per-head sink fits per-query rowmax/denom; reject mla/v2 — its single shared-KV
  einsum materializes `[B,M,K,D]`). Reuse grid `(batch,heads,q_seq,·)` + the m/l accumulator
  init/update/exp-correction (`~:294-330`); set the kv-block grid axis to a dummy and move the
  K gather into the body — gather KV rows by `topk_idxs` via per-index `pl.dslice(idx*D, D)` DMA
  (pattern à la `ragged_*` per-row dynamic slice; Mosaic has no native scatter-gather), masking
  `-1`. Sink: add `exp(sink[h]-m)` into the running max + denom only, NO sink value in the
  V-accum. RISK: dynamic K-gather loop width vs VMEM on v6e gen-6 — prefetch all K rows once
  into VMEM then reuse.

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

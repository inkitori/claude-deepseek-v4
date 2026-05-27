# Handoff — DeepSeek-V4-Flash decode/prefill PERFORMANCE (next phase after S1)

> **Status as of 2026-05-27 (S29):** S1 (decode *correctness*) is **CLOSED** — decode is
> coherent and cross-process deterministic. This document is about the **next** problem:
> the model is **slow**, and a profile pinned exactly why. Read this end-to-end before
> touching code. Durable slice ops live in `CLAUDE.md`; S1 history in `HANDOFF_S1.md`.

---

## 0. TL;DR (read this first)

- **The bottleneck is NOT the MoE.** It is the **sparse-attention KV gather** —
  `jnp.take_along_axis(kvf, idx_expanded, axis=1)` at
  `work/tpu-inference/tpu_inference/layers/jax/attention/deepseek_v4_attention.py:186`
  (inside `sparse_attn`). It runs at **~0.1 % of HBM bandwidth**.
- It is **99 % of prefill** time and **62 % of a decode step**.
- A static FLOP count said "97 % MoE, fix with sparse dispatch" — **the profile overturned
  that.** MoE expert compute is only ~10 % of a decode step. Profile before optimizing.
- **Real decode is ~120 ms/token (~8 tok/s)** — far faster than the "0.31 tok/s" headline,
  which was **dominated by the one-time prefill catastrophe** (~120 s for a *short* prompt).
- **The fix:** `sparse_attn` is a *plain-JAX reference* for a fused kernel that was never
  wired in (its own docstring says so). Replace the materialized gather + fp32 cast with a
  fused/tiled sparse-attention kernel. Expected: **~50–100× on prefill, ~2–3× on decode.**
- **Hard constraint:** do not regress S1. Any change must still pass the determinism gate
  (FIB decode md5 byte-identical across 2 fresh engines + correct Fibonacci + smoke_check
  rc=0). See §5.

---

## 1. What is already solved (don't re-litigate)

**S1 = decode non-determinism + quality drift. CLOSED (commit `2d8ca139`).**
- Root cause: the routed **W1/W3 stacked-weight LOAD** — a device-side consolidation
  reshard read uninitialised HBM, baking per-process garbage into the expert weights ⇒
  two fresh engines produced different decode at temp=0.
- Fix (commit `5a3ed435`): rebuild the w1/w3 stacked tensor from the full per-expert host
  numpy via `jax.make_array_from_callback` (host→device, **no** device reshard, no uninit
  read). w2 was already clean.
- S1 debug instrumentation then removed (commit `2d3e0c45`, 165 deletions). Kept: the
  env-gated `_v4_nan_tripwire` (no-op unless `V4_DECODE_NAN_TRIPWIRE=1`) and the
  `compute_logits` `nan_to_num` clamp (both functional, not diagnostics).
- Verified ×2 fresh engines on the cleaned build: FIB decode md5=`5bf42256` byte-identical,
  correct Fibonacci (21, 34, 55, 89, 144), `smoke_check` rc=0.

The decode path **works and is correct**. The job now is to make it **fast** without
breaking that.

---

## 2. The profile (how it was measured)

Captured 2026-05-27 with vLLM's profiler (jax.profiler under the hood on this TPU backend):

1. Launch `vllm serve` with profiler flags (see §7 for the exact recipe):
   `--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/home/enyouki/v4_traces`
2. After the engine is ready, **warm one request** (pays any decode recompile), then bracket
   a steady-state decode window: `curl -X POST .../start_profile` → run a decode request →
   `curl -X POST .../stop_profile`.
3. Traces are written **per TPU worker** (NOT on the head) at
   `/home/enyouki/v4_traces/plugins/profile/<ts>/<host>.trace.json.gz` (+ `.xplane.pb`).
   The `.trace.json.gz` is a standard Chrome trace (`pid` whose process_name =
   `/device:TPU:0`; thread "XLA Ops" = per-op durations; thread "XLA Modules" = per-step
   program executions). Workers are SPMD-symmetric — analyze any one.

**Caveat:** the profiler caps at 1,000,000 XLA-Op events, which here captured **1 prefill +
1 decode step** — enough for the structural per-op breakdown (the program is identical every
step), but per-step numbers come from a single decode step.

---

## 3. The findings (full detail)

### 3a. Per-decode-step breakdown (~120.5 ms/token; TPU ~100 % busy, ~0 % idle)
| % of step | op (XLA) | what it is |
|---|---|---|
| **61.9 %** | `gather_custom_fusion` (~1–2 per layer × 43) | **`sparse_attn` KV gather** (`take_along_axis`) |
| 11.9 % | `all-reduce` (2176 ops) | collective (replicated activation + 32-way expert shard) |
| 10.0 % | `multiply_reduce_fusion` (708) | **MoE expert einsum** (the dense path) |
| 8.5 % | `while` (~1 per layer) | sparse-attn top-k loop (indexer) |
| 4.5 % | collective-permute / all-gather / all-to-all | more collectives |

So per decode step: **~62 % attention gather, ~17 % collectives, ~10 % MoE, ~8 % top-k loop.**

### 3b. Prefill is the real catastrophe (~120 s for a short prompt)
- **99 %** of prefill is the same `sparse_attn` gather: **84 gathers of ~805 MB each
  (`f32[201M]`) at ~2.13 s each = ~119 s.** Effective bandwidth ≈ **1.5 GB/s ≈ 0.1 % of HBM
  BW**. `model_flops = 0` — it is a pure memory-movement op, zero compute.
- This is why the smoke's **"0.31 tok/s" headline is misleading**: it amortizes the one-time
  prefill. Steady-state decode is ~8 tok/s. As context grows, decode gathers also grow
  (the gather scales with KV length), so later decode steps slow down too.

### 3c. Why the FLOP estimate was wrong
A FLOP count (correct as a FLOP count) said decode is **97 % routed-MoE**: the dense
`moe_forward` path computes **all 256 experts** when only top_k=6 are used (42.7× FLOP
over-compute). But those expert matmuls fold into cheap, well-tiled fusions and are only
~10 % of wall time. The attention gather is ~2 % of FLOPs but 62–99 % of wall time because
it runs at 0.1 % bandwidth. **Wall-clock ≠ FLOPs. Always profile.**

---

## 4. Root cause in code

`work/tpu-inference/tpu_inference/layers/jax/attention/deepseek_v4_attention.py`, function
`sparse_attn` (def at **:160**, body **:165–200**):

```python
qf = q.astype(jnp.float32)
kvf = kv.astype(jnp.float32)                       # :181  casts the WHOLE KV to fp32
safe_idx = jnp.maximum(topk_idxs, 0).astype(jnp.int32)
idx_expanded = jnp.broadcast_to(safe_idx.reshape(B, M*K, 1), (B, M*K, D))
kv_gathered = jnp.take_along_axis(kvf, idx_expanded, axis=1)   # :186  THE BOTTLENECK
kv_gathered = kv_gathered.reshape(B, M, K, D)      # materializes [B, M, K, D] fp32
logits = jnp.einsum("bmhd,bmkd->bmhk", qf, kv_gathered) * softmax_scale   # :189
... sink-softmax ...
out = jnp.einsum("bmhk,bmkd->bmhd", p, kv_gathered)            # :199
```

Two problems: (1) it casts the entire KV cache to fp32 (doubles memory traffic), and
(2) it **materializes** the full `[B, M, K, D]` gather instead of fusing gather+compute in
tiles. The docstring (**:170–171**) explicitly says this is a reference for a fused
`sparse_attn_kernel` — which was **never ported to this repo** (see §6).

The math to preserve (per docstring + `INVARIANTS.md` I14): top-k masked attention with a
learnable per-head **sink** term added to the softmax denominator
(`sum_exp += exp(attn_sink - m_max)`), `-1` entries in `topk_idxs` masked out.

---

## 5. Hard constraints — DO NOT break these

1. **Cross-process determinism (S1).** Any attention/kernel change MUST still pass the gate:
   - `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0
     (visible_words ≥ 10, max_word_run < 5).
   - FIB decode md5 **byte-identical across 2 fresh engines** (`python3 /tmp/s1_probe2.py 2`
     → expect `5bf42256` on the *current* build; a correctness-preserving kernel that changes
     numerics slightly may shift the md5 — then re-establish a NEW reference and confirm it's
     identical across 2 fresh engines + correct Fibonacci. The non-negotiable is **identical
     across engines** + **correct Fibonacci**, not the specific hash).
   - Read the actual decode text — "contains Paris" is a known false positive.
2. **No gather of the size-1 decode token axis to replicated** (CLAUDE.md pitfall #5): a
   `with_sharding_constraint` that gathers the decode token axis Core-halts the slice
   (proven ~8×). A wsc on a post-reduction `[N, dim]` quantity is safe.
3. **Validate numerics on CPU first** against the oracle (cheap, no slice): the PyTorch
   reference `sparse_attn_torch` at
   `work/tpu-inference/tests/models/jax/_deepseek_v4_reference/kernel_stubs.py:60` is the
   numerical oracle for `sparse_attn`. CPU **cannot** reproduce S1 (no sharding) but CAN
   verify a kernel matches the reference math.
4. **Keep the diff minimal** — every changed line rsyncs to 8 hosts.

---

## 6. The fix — concrete starting points (scouted)

**There is no drop-in kernel.** `/mnt/scratch/v4_pro/inference/kernel.py` (the docstring
reference) **does not exist** on this machine. The rewrite authors a new Pallas/Mosaic kernel.

**Interface to preserve** — `sparse_attn(q[B,M,H,D], kv[B,N,D], attn_sink[H] fp32,
topk_idxs[B,M,K] int32 (-1 = ignore), softmax_scale: float) -> [B,M,H,D]`. Call sites:
- **Decode** — `deepseek_v4_attention.py:812` (M=1; `kv` = full `new_kv_cache [B,N,D]`).
- **Prefill** — `deepseek_v4_attention.py:905` (M=S; `kv` = `kv_full`, base ⊕ compressed).

**`topk_idxs` producer (the indexer)** — consumes these, so the kernel can take them as-is:
- Prefill: `indexer_prefill` (`:366`), `lax.top_k` (`:418`), `-1` mask (`:422–423`).
- Decode: `indexer_decode_step` (`:562`), `lax.top_k` (`:614`), mask (`:617`).
- `IndexerParams.index_topk` (the K) near `:361`.

**Reusable kernel templates in-repo** (best → adapt these, don't start from scratch):
- `work/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py:14` — "TPU-Friendly MLA Ragged
  Paged Attention" (DeepSeek-style MLA; **closest architectural match**).
- `work/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py:82` — base
  online-softmax flash kernel; model the sink/top-k variant on this loop.
- `work/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py:1568`
  (entry `ragged_paged_attention`) — gather-free paged-KV flash attention.
- `work/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather.py:237`
  (`ragged_gather`) and `.../gather_reduce.py:72` (`sc_gather_reduce`) — SparseCore
  index-driven gather (+reduce); directly relevant to eliminating/fusing the `take_along_axis`.
- Numerical oracle: `tests/models/jax/_deepseek_v4_reference/kernel_stubs.py:60`
  (`sparse_attn_torch`). Invariant: `work/tpu-inference/INVARIANTS.md` I14 (sink softmax).

**Suggested approach:** author a Pallas sparse-attention kernel that takes `(q, kv,
topk_idxs, attn_sink, scale)` and does tiled gather+online-softmax+sink without materializing
`[B,M,K,D]` or casting the whole KV to fp32. Start from `flash_attention/kernel.py` for the
softmax+sink loop and borrow the index-gather tiling from `sparse_core/ragged_gather.py` (or
`mla/v2` if its paged structure fits). Validate vs `sparse_attn_torch` on CPU, then run the
§5 determinism gate on the slice.

**Secondary targets (after the gather):**
2. Collectives (~17 %/decode step): the per-layer all-reduces from the replicated decode
   activation + 32-way expert sharding. Revisit the decode sharding (`_v4_decode_replicate`
   in `runner/tpu_runner.py`) — but mind pitfall #5 (no token-axis gather).
3. MoE sparse top-k dispatch (~10 %/decode step): route the decode token to only its top_6
   experts instead of the dense all-256 einsum (`moe_forward`, `use_shard_map=False` path,
   `deepseek_v4_moe.py:~217–232`). Lower priority than the FLOP count implied. N=1 is the
   hard case (can't shard over attn_dp=32); the prefill `gmm_v2` path is the model to follow.

---

## 7. How to operate the slice (essentials; full detail in CLAUDE.md)

- **Slice:** v6e-32, 8 hosts × 4 chips, TP=32. Head = `10.164.0.192`; workers `.194 .202
  .204 .193 .198 .195 .200`. ssh: `ssh enyouki@<ip> -i ~/.ssh/google_compute_engine`.
- **Keep guardians alive** before TPU work: `ps -eo pid,cmd | grep -E
  'node_guard[i]an|meta_guard[i]an'` (restart node_guardian per CLAUDE.md if dead — never
  `pkill` a pattern your own command line contains).
- **After any code edit:** `scripts/full_slice_v4_sync.sh` (rsyncs to 8 hosts; `git push`
  does NOT). Verify md5 matches head==workers (mismatch ⇒ launch-id Core-halt, not a wedge).
  If a `.py` changed, **clear `~/.cache/vllm/xla_cache/*` on all 8 hosts**.
- **Cycle:** `scripts/full_slice_v4_reset.sh` → `scripts/full_slice_v4_smoke.sh` (backgrounds
  vllm serve, prints log path). Wait for `Application startup complete` (~6 min warm, 10–30
  min cold). First decode request recompiles (~325 s) unless the xla_cache is warm.
- **Probe helpers (in /tmp):** `s1_probe2.py N` (FIB decode, prints md5 + text; N = max_tokens).
- **One engine at a time** (smoke.sh self-guards with flock). The engine **wedges on a new
  request shape** — see §8.

### Reproducing the profile
The smoke script does **not** carry profiler flags by default (a temporary passthrough hook
was added and reverted). To re-enable, add one line to `scripts/full_slice_v4_smoke.sh` in the
`vllm serve` invocation (before `> "$LOG"`):
```bash
    ${V4_PROFILER_ARGS:-} \
```
then launch with:
```bash
mkdir -p /home/enyouki/v4_traces   # on head AND all 7 workers (rank-0 can be any host)
V4_PROFILER_ARGS="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/home/enyouki/v4_traces" \
  bash scripts/full_slice_v4_smoke.sh
```
After ready: warm (`python3 /tmp/s1_probe2.py 2`), then
`curl -sX POST :18081/start_profile` → `python3 /tmp/s1_probe2.py 20` →
`curl -sX POST :18081/stop_profile`. Collect `*.trace.json.gz` from a worker (e.g. `.194`),
parse the Chrome trace (group "XLA Ops" events on the `/device:TPU:0` process by `name`, sum
`dur`). 750 MB unzipped — stream it (ijson) or load on a high-RAM host.

---

## 8. Known open issues (NOT S1, NOT yet fixed)

- **Long/multi-turn chat wedges the engine.** A `/v1/chat/completions` with a 6-message
  context returned **HTTP 500 → `EngineDeadError`** at `jit(run_model)/shard_map/concatenate`
  and stopped serving (process stays alive). This is CLAUDE.md pitfall #6, now pinned to a
  `shard_map/concatenate` shape fault on the chat/longer-prefill path. Compounded by the smoke
  config's tiny `--max-model-len 256 --max-num-seqs 1`. Real long-chat serving needs (a) a
  larger `max-model-len` and (b) fixing that shape fault. The `vllm chat` REPL is
  `work/vllm_env/bin/vllm chat --url http://localhost:18081/v1 --model deepseek-ai/DeepSeek-V4-Flash`
  but it will trip this on the 2nd–3rd turn.
- **Smoke config is minimal**, not a serving config (256 ctx, 1 seq, `--enforce-eager`).

---

## 9. First moves for the fresh session

1. Read this doc + skim `CLAUDE.md` (slice ops, pitfalls) and `HANDOFF_S1.md` (why
   determinism is sacred).
2. Confirm the bottleneck yourself if desired: reproduce the profile (§7) — should see
   `gather_custom_fusion` dominate.
3. Open `deepseek_v4_attention.py:160` (`sparse_attn`) + the templates in §6. Decide kernel
   strategy. Prototype the kernel, validate vs `sparse_attn_torch` on CPU.
4. Wire it at both call sites (`:812` decode, `:905` prefill), sync, clear cache, smoke, and
   run the §5 determinism gate before/after. Commit per validated step.
5. Then collectives, then MoE sparse dispatch.

Good luck — the correctness war is won; this is the speed campaign.

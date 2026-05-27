# Handoff — DeepSeek-V4-Flash PERFORMANCE campaign

> **Phase = PERFORMANCE.** S1 (decode correctness/determinism) is CLOSED — now a HARD
> REGRESSION GATE, not the goal (§S1-GATE). Job: make prefill + decode FAST on the v6e-32
> slice without breaking determinism. This doc is the loop's memory — current state, the
> roadmap, the ONE next action. Durable slice ops: `CLAUDE.md`. S1 history: `HANDOFF_S1.md`.
>
> **One-line status (2026-05-27):**
> - **Phase 1 CLOSED & GATED** — fused sparse-attn kernel (`kernels/sparse_attn/kernel.py`):
>   attn KV gather 65.8%→**0.2%** of decode. CPU oracle restored (1.E, `90e7c520`).
> - **Phase 3.1 DONE & GATED (`c78ecb96`, this session)** — MoE dense-decode gate/up einsums
>   now bf16-in/**fp32-accumulate** (`preferred_element_type=fp32`) instead of fp32 casts:
>   halves W1/W3 HBM streaming + native bf16 MXU, bit-faithful to the prefill gmm_v2 path.
>   S1 gate clean, **md5 unchanged (5bf42256)** — the bf16 ULP shift didn't flip the argmax.
> - **Phase 2 (pick_partition_spec axis-0 flip) DEPRIORITIZED** — marginal/lateral, possibly
>   net-negative (decode is launch-bound at N=1; the flip changes bytes/launch, not launch
>   count). 3-angle analysis in git (`607629da`). DO NOT smoke it.
> - **The cheap perf wins are now exhausted.** Remaining decode levers (all-reduce 31.7%,
>   indexer top_k 23.3%) are all HARD/design. **Next = re-profile post-3.1, then Phase 3.2.**

---

## The verified profile (decode) — ⚠️ PRE-3.1, re-capture next
Phase 1's kernel killed the attn KV gather (was 65.8% → 0.2%). Breakdown from the
`2026_05_27_08_46_30` trace (tiny ~30-tok FIB ctx; `scripts/perf_parse_trace.py --bucket-ops`),
**captured BEFORE Phase 3.1** — 3.1 should have cut the MoE row, re-profile to confirm:

| decode op (one step, ~50 ms busy) | % | roadmap |
|---|---|---|
| all-reduce (+psum) (2176/step) | **31.7 %** | Phase 2 — DEPRIORITIZED (launch-bound) |
| MoE dense `multiply_reduce_fusion` | 25.2 % | **Phase 3.1 DONE** (cut the fp32 cast) + 3.3 |
| indexer top_k `lax.top_k` | 23.3 % | **Phase 3.2** (biggest long-ctx lever; jit-blocked, §3.2) |
| all-gather | 7.1 % | decode residual |
| collective-permute | 4.9 % | Phase 2 |
| `sparse_attn_kernel` (fused attn) | 0.2 % | ✅ Phase 1 done |

⚠️ **Caveats:** tiny ctx (~30-tok FIB) ⇒ MoE share INFLATED (falls at long ctx); the indexer
top_k buffer is STATIC `T = state_max_seq_len//4` (**T=2048 in the smoke** [max_num_batched_tokens×dp
= 256×32 = 8192 → /4]; T=131072 at 16k ctx) so top_k SCALES with max-model-len (could DOMINATE
prod decode). all-reduce is context-robust. **We are architecturally faithful to the GPU
reference** (MLA/RoPE/YaRN/compressor/indexer/sink-softmax/HC/routing all match the PyTorch
oracle + vLLM-GPU). Pure perf + cleanup; no correctness gap.

---

## THE ROADMAP (drive top-down; every change clears §S1-GATE; cheapest tier first)
Reserve full smokes (≤1-2/session; COLD = 25-45 min on a `.py` change). CPU oracle = Tier 1.

### ✅ DONE (history)
- **0.0** microbench `scripts/perf_microbench.sh`. **0.1** (`d22df61a`) killed duplicate prefill body.
- **0.2/0.3** (`2839a684`/`56abe232`) bf16 sparse_attn gather + deleted dead `_consolidate_moe_after_load`.
- **1.A–1.E** fused Pallas sparse-attn kernel authored/wired/gated/re-profiled (gather→0.2%); CPU oracle restored.
- **3.1** (`c78ecb96`) MoE dense-decode gate/up → bf16-in/fp32-accumulate (`preferred_element_type=fp32`).

### Phase 3.2 — indexer top_k (BIGGEST long-ctx lever, but JIT-BLOCKED — needs a real decision)
`indexer_decode_step` (`deepseek_v4_attention.py:~644-652`) runs `lax.top_k(index_score, K=512)` over
the STATIC buffer `T = state_max_seq_len//4`, ×21 CSA layers/step. Cost scans full T even when few
slots valid → 23.3% at tiny ctx, would DOMINATE at prod ctx (T scales 64×).
**Key findings (this session, 2-agent audit):**
- The padding TAIL is already `-inf`-masked before top_k (`:644-646`), so a valid-prefix-only scan
  would be **bit-identical** (no md5 re-baseline) — *if it were jit-valid*. It is NOT cleanly:
- **The blocker:** seqlen is **traced** (`deepseek_v4.py:1826`, `start_pos = seq_lens[0]-1`) under a
  **MONOLITHIC decode JIT** — vLLM does NOT bucket decode seqlen. `lax.dynamic_slice` needs a STATIC
  slice size, so you cannot slice `[0:valid_len]` exactly. No static per-request length < T exists.
- **Options (all non-trivial):** (a) **static scan-cap config** W<T — only exact if W ≥ every real
  ctx (= T at prod, no win); caps usable context = a product tradeoff, not free. (b) **per-bucket
  decode JIT** (each bucket gets a smaller static T) — EXACT + real win, but a big refactor (vLLM
  decode is monolithic today). (c) **`lax.approx_max_k`** — small code change, gateable, but
  APPROXIMATE selection (the top_k idx directly gather KV for sparse attn `:188-205`, so missed
  high-score slots warp attention → FIB-quality risk) AND cross-engine determinism UNCONFIRMED (only
  a smoke ×2 can prove it — costs the session's smoke budget). (d) leave it. Blocked/two-level top_k
  reads all T → marginal on TPU.
- **Verdict:** not a quick safe win. Best gateable EXPERIMENT = (c) approx_max_k (smoke ×2: check FIB
  correct + md5 identical ×2 engines; if non-deterministic → abandon). "Right" fix = (b), big.

### Phase 3.3 / 4 / 5 (later)
- **3.3** Sparse top-6 MoE dispatch — N=1 can't shard over attn_dp=32; only a replicated top-6
  `gmm_v2` is safe. Measure before committing; payoff uncertain.
- **4.1** Long/multi-turn chat WEDGE (HIGH for real serving) — MoE `use_shard_map` gate
  (`deepseek_v4_moe.py:211`) flips True at a larger N bucket → first entry to `_routed_local`
  shard_map. Make MoE path selection shape-stable / warm larger buckets. **4.2** raise smoke
  `MAX_LEN`/`max-num-seqs` to reproduce.
- **5** De-hack/shrink the diff (READY, tractable, low-risk diff-shrink now the kernel is validated):
  remove `_v4_nan_tripwire` (~41 sites), audit the two clamps (`_linear` `|r|<1e8→0`;
  `compute_logits` `nan_to_num` — instrument whether they ever fire), trim S1-narrative comments. Do
  NOT "make idiomatic" the loader/MoE/seed paths fused with the S1 fix.

---

## NEXT ACTION (for the session reading this)
**Re-profile decode post-3.1** (the table above is PRE-3.1; confirm the MoE row fell and re-rank the
levers — cheap: 1 smoke + profiler, recipe below). THEN act on whatever is now #1:
- If **indexer top_k** is the clear top lever → Phase 3.2. It is NOT a quick win (§3.2 jit-blocker).
  The one gateable experiment is **`lax.approx_max_k`** (swap `lax.top_k`→`approx_max_k`, CPU oracle,
  then smoke ×2: must be **correct Fib + md5 identical ×2 fresh engines**; if approx is
  cross-engine-nondeterministic it FAILS the gate → abandon and document). The exact fix (per-bucket
  decode JIT) is a big refactor — scope it only if approx fails.
- If the hard perf levers stall → pivot to **Phase 5 nan_tripwire removal** (ready diff-shrink, the
  stated secondary campaign goal; kernel is now validated so it's removable). Possibly a tiny perf win.
- **Do NOT** smoke the Phase-2 axis-0 flip (deprioritized, §status).
- Hand off when context grows (CLAUDE.md "CONTEXT HANDOFF PROTOCOL").

---

## <a name="S1-GATE"></a>S1 REGRESSION GATE (non-negotiable for every change)
- `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (visible_words ≥ 10,
  max_word_run < 5).
- FIB decode: **correct Fibonacci** (21, 34, 55, 89, 144 — DETERMINISTIC) + **N=2 md5 `5bf42256`
  byte-identical across 2 fresh engines** (`s1_probe2.py 2`, = md5("21,")). ⚠️ The long-tail md5
  (`s1_probe2.py 20`+, e.g. `b675be27`) is NON-deterministic at temp=0 (pre-existing decode
  all-reduce-ordering residual) — do NOT gate on it. A numerics-changing fix MAY shift even the N=2
  md5 → re-establish + confirm identical ×2 engines + correct Fibonacci. Non-negotiable = **identical
  ×2 engines (N=2) + correct Fibonacci**, not a specific hash.
- READ the actual decode text — "contains Paris" is a known false positive (can EOS at tok 1).
- Probe: `python3 /tmp/s1_probe2.py N` (FIB decode, prints md5 + text; N = max_tokens).
- **2-engine recipe (cheap):** engine 1 = cold smoke (cache cleared after a `.py` edit); reset; engine
  2 = WARM re-smoke (do NOT re-clear cache → ~6 min startup + cached decode). Both probes must match.

---

## Reproducing the profile (durable recipe)
Add `${V4_PROFILER_ARGS:-} \` to the `vllm serve` line in `scripts/full_slice_v4_smoke.sh` (before
`> "$LOG"`), then:
```bash
mkdir -p /home/enyouki/v4_traces
V4_PROFILER_ARGS="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/home/enyouki/v4_traces" \
  bash scripts/full_slice_v4_smoke.sh
```
After ready: warm (`s1_probe2.py 2`), then `curl -sX POST :18081/start_profile` → `s1_probe2.py 20`
→ `curl -sX POST :18081/stop_profile`. Traces land PER-WORKER at
`/home/enyouki/v4_traces/plugins/profile/<ts>/<host>.trace.json.gz` (~750 MB unzipped — stream with
ijson; group "XLA Ops" on `/device:TPU:0` by `name`, sum `dur`). Parser: `scripts/perf_parse_trace.py
--bucket-ops`. **Caveat:** the 1M XLA-op cap captures ~1 prefill + 1 decode step; for the
decode-vs-context curve (esp. Phase 3.2 — top_k scales with T) profile a decode-only window at LONGER ctx.

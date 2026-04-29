# Finish DeepSeek V4 in tpu-inference (v2 — autonomous overnight task)

## Mission
You are working autonomously overnight. The user is asleep — do not wait for
them. Make decisions, document them in markdown files at the repo root, and
proceed.

A previous overnight run on this host produced a mathematically-correct
**prefill-only** JAX implementation of DeepSeek V4 (V4-Flash + V4-Pro). It
lives on branch `deepseek-v4` in `/workspace/work/tpu-inference/`. Your job
is to finish the work so that, when run with real V4 weights via `vllm
serve deepseek-ai/DeepSeek-V4-Flash`, inference would actually succeed.

The single non-negotiable goal remains MATHEMATICAL CORRECTNESS combined
with INTEGRATION COMPLETENESS: outputs must be numerically equivalent to
the DeepSeek `inference/model.py` reference at every code path, AND the
model must compose with vLLM's serving stack (paged-KV, scheduler, OpenAI
handler) end-to-end on synthetic tiny-V4 fixtures.

Performance is irrelevant. A correct, slow implementation is a complete
success.

## Resume mandate

Branch `deepseek-v4` HEAD has 41 passing tests covering Tier 1 (component
unit tests vs PyTorch reference), Tier 2 (small-config logits parity for
prefill), Tier 3 (compile-only on real V4-Flash + V4-Pro configs across
v4-8 + simulated v6e-32 meshes), and Tier 4 (HF-name → JAX-param-tree
mapping over all 69,187 V4-Flash entries). **Do NOT redo this work.** If
you change a function under `tpu_inference/layers/jax/attention/deepseek_v4_attention.py`
or `tpu_inference/layers/jax/moe/deepseek_v4_moe.py`, you must rerun the
existing suite and report any regression in `REGRESSIONS.md`.

Before any new work, in this order:
  1. Read SUMMARY.md (esp. §4 "highest-confidence remaining correctness
     risks" and §8 "what was deliberately not done").
  2. Read PROGRESS.md last 50 lines, BLOCKERS.md, STUCK.md, DECISIONS.md.
  3. Read `/workspace/logs/tpu-preflight.log`. The first line is JSON of
     the form `{"ok": true|false, "n_tpu": N, ...}`. If `ok:false`, real-TPU
     tiers (5, 6) are skipped (not failed) — record the error in
     TPU_PREFLIGHT.md, do not try to edit `run.sh` (you can't — it's
     host-side and read-only from your perspective). If `ok:true`, you
     have 4 v4 chips for tiers 5 and 6.
  4. `git log --oneline -30` to see committed state.
  5. Run the existing test suite to confirm what's actually passing right
     now: `pytest tests/models/test_deepseek_v4.py -v`. The markdown files
     may be slightly stale relative to the code.
  6. Append a `RESUMED at <timestamp>` line to PROGRESS.md with a one-line
     summary of where you're picking up.
  7. Then continue from W1 (or the first W-item not done in PROGRESS.md).

Do NOT restart from Phase 0. Do NOT re-derive Tier 1/2 prefill math.

## Resumability rules (same as v1)

Your session may be killed mid-work due to usage limits, the per-iteration
timeout (90 min), or the host restarting. Assume this will happen.

  - Commit to git after every passing test, with descriptive messages
    that reference the work item and tier (e.g. `W1: decode parity at
    start_pos=8 matches reference`).
  - Update PROGRESS.md every 10–15 minutes. The last line of PROGRESS.md
    must always answer "if I died right now, what would the next session
    need to do first?"
  - Update STATUS.md atomically at the end of every meaningful step. See
    "STATUS.md mandate" below — the host loop tails this file.
  - Never hold critical state only in memory or in your scratch reasoning.
    If it matters, it goes in a markdown file or a commit message.
  - Before starting any task that takes >20 minutes, write a one-line
    plan to PROGRESS.md first.

## Work items (dependency-ordered)

### W1 — Decode path in JAX (gates Tier 5)

The current `Compressor`, `Indexer`, and `Attention` classes in
`tpu_inference/layers/jax/attention/deepseek_v4_attention.py` are
prefill-only. They write nothing to a KV cache and assume `start_pos=0`.

Port the decode-time logic from
`tests/models/jax/_deepseek_v4_reference/model.py` (the same reference
you matched for prefill in the prior session). The reference's
`Compressor.forward(x, start_pos)` maintains `kv_state` and `score_state`
buffers; `Indexer.forward(x, qr, start_pos, offset)` does the same plus a
sliding-window topk.

Implement these as **functional** JAX functions:

```python
def compressor_decode_step(state_in, x_step, params) -> (state_out, y_step)
def indexer_decode_step(state_in, x_step, qr_step, params) -> (state_out, scores)
def attention_decode_step(kv_cache, x_step, start_pos, ...) -> (kv_cache, y_step)
```

State is a pytree of arrays. Threading is the caller's responsibility —
do NOT hide state in module attributes.

Tier 2 expansion (must pass at the existing bf16 tolerances):

  - `test_decode_parity_vs_reference` covering start_pos ∈ {1, 8, 9
    (window-wrap boundary), 64, 256}.
  - 32-step rolling-decode test: state-after-32-decodes ≡ state-after-
    prefill-of-the-same-prefix (allow `atol=2e-2` for accumulation drift,
    document in TOLERANCE_LOG.md).
  - Decode parity at every compress_ratios value (0=SWA, 4=CSA, 128=HCA).

### W2 — Paged-KV via ragged_paged_attention (gates Tier 5)

DECISIONS.md D5 in the prior session deferred `ragged_paged_attention`.
**This run reverses that decision.** vllm serve depends on paged-KV.

Replace the dense materialized attention in `Attention.__call__` with
`tpu_inference.kernels.ragged_paged_attention.v3` for SWA layers. For CSA
and HCA layers, the existing `sparse_attn` Python implementation must be
lowered to a Pallas equivalent (or, if Pallas authoring is infeasible
within this run, document in BLOCKERS.md and use the existing Python
implementation with a TODO — but Tier 5 still needs to work).

Mirror the V3 pattern at `tpu_inference/models/jax/deepseek_v3.py:1322` —
that file already calls into ragged_paged_attention with the right
metadata. Read it end-to-end before writing.

### W3 — DeepseekV4ForCausalLM.__call__ (gates Tier 5)

`tpu_inference/models/jax/deepseek_v4.py:808` currently raises
`NotImplementedError`. Replace it with a real implementation that mirrors
V3's calling convention at `deepseek_v3.py:1383`:

```python
def __call__(self, kv_caches: List[jax.Array], input_ids, positions,
             attn_metadata, ...) -> (kv_caches, x, [])
```

The class must be a real `nnx.Module` (or whatever V3 uses) with
parameters loaded via the model loader from W4.

### W4 — FP4/FP8 weight loader (gates Tier 7)

Real V4 stores experts in FP4 (e2m1fn) and dense layers in FP8 (e4m3fn)
with `weight_block_size=[128,128]` and `*.scale` companions in
`float8_e8m0fnu`. The current loader is name-only — it does not
dequantize.

The canonical recipe is at `/mnt/scratch/v4_pro/inference/convert.py`
(specifically `cast_e2m1fn_to_e4m3fn` and the `FP4_TABLE` codebook) and
`/mnt/scratch/v4_pro/inference/kernel.py`. Use these as ground truth, NOT
guesses about quantization conventions.

Extend `tpu_inference/models/common/model_loader.py` (or a v4-specific
sibling) to:

  1. Detect `quantization_config.quant_method == "fp8"` →
     for tensors with a sibling `*.scale` of dtype `float8_e8m0fnu`,
     dequant via `bf16 = e4m3.float() * (2 ** (e8m0.view(int8).int() -
     127))`, applied per `weight_block_size` block.
  2. Detect `expert_dtype == "fp4"` → for expert weight tensors stored
     as `int8` (or `float4_e2m1fn_x2` packed), unpack via FP4_TABLE
     lookup (low nibble first, then high nibble), then apply the
     ue8m0-block-scale recipe.
  3. Tier 4b (NEW): round-trip the staged real
     `/mnt/scratch/v4_flash/model-00001-of-00046.safetensors` shard
     through your loader. Compare a fixed `(i,j)` of `embed.weight`
     (which is bf16, no scale) against `safetensors.torch.load_file`
     direct read. They must be byte-equal. This validates the bf16 path
     end-to-end against real bytes; the FP4/FP8 path is validated
     synthetically by Tier 7.

## New tiers (additive to v1's 1–4)

### Tier 5 — vLLM serve smoke test

Three pre-staged synthetic fixtures live at:
  - `/mnt/scratch/tiny_v4_bf16/` — tiny config, all weights bf16.
  - `/mnt/scratch/tiny_v4_quant/` — same config, FP8 dense + FP4 expert
    weights with `*.scale` companions, byte-layout matching real V4.
  - `/mnt/scratch/tiny_v4_groundtruth/` — bf16 weights produced by
    dequantizing `tiny_v4_quant` with the canonical recipe.

If those directories don't exist, the user did not run the host-side
fixture script. Write to BLOCKERS.md (`T5-fixtures-missing`) and skip
Tier 5; do NOT generate the fixtures yourself.

Tier 5 procedure:

```bash
# Background server
JAX_PLATFORMS=tpu vllm serve /mnt/scratch/tiny_v4_bf16 \
    --tensor-parallel-size 4 --max-model-len 256 \
    --max-num-seqs 2 --port 18080 --seed 0 \
    --trust-remote-code --dtype bfloat16 &
SERVE_PID=$!

# Wait for /v1/models 200 (timeout 120s)
for i in $(seq 1 60); do
    curl -sf http://localhost:18080/v1/models >/dev/null && break
    sleep 2
done

# Send two identical, deterministic-with-seed completions
RESP1=$(curl -s http://localhost:18080/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"tiny_v4_bf16","prompt":"abc","max_tokens":8,"temperature":0,"seed":0}')
RESP2=$(curl -s http://localhost:18080/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"tiny_v4_bf16","prompt":"abc","max_tokens":8,"temperature":0,"seed":0}')

kill $SERVE_PID
```

Assertions:
  - HTTP 200 on both /v1/models and both /v1/completions.
  - Both responses have a non-empty `choices[0].text`.
  - The two responses are byte-equal (deterministic with fixed seed).

On failure: capture stderr from vllm, kill the server, write traceback +
hypothesis to BLOCKERS.md under `T5-`, move to a non-dependent task. Do
not sit idle.

### Tier 6 — Real-TPU compile + tiny forward

Same `eval_shape` / `jit().lower().compile()` shape as Tier 3, but with
`JAX_PLATFORMS=tpu`. First assert `len(jax.devices('tpu')) == 4` — if
not, this tier skips (not fails) per the pre-flight result.

Run a forward pass with the tiny config (loaded from `tiny_v4_bf16`)
and compare logits to the Tier-2 CPU-reference output at `atol=0.2`
(register T6 in TOLERANCE_LOG.md — bf16 MXU vs CPU emulation drift).

### Tier 7 — FP4/FP8 dequant equivalence

Load `tiny_v4_quant` and `tiny_v4_groundtruth`, run forward on identical
seeded `input_ids`, compare logits at `atol=0.1` (register T7 in
TOLERANCE_LOG.md — quantization-noise budget).

The point of comparing against `tiny_v4_groundtruth` (and not
`tiny_v4_bf16`) is to isolate loader-dequant correctness from
quantization-arithmetic correctness. `tiny_v4_groundtruth` was
pre-dequantized using the same recipe DeepSeek's `convert.py` uses, so
any disagreement is the loader's fault.

## Acceptance gates

A run is "complete" only when Tiers 1–7 all green within a single
`pytest` report (modulo Tier 5/6 skipping if pre-flight `ok:false`).
Tier dependencies:
  - Tier 5 depends on W1, W2, W3 + pre-flight ok + fixtures present.
  - Tier 6 depends on pre-flight ok.
  - Tier 7 depends on W4 + fixtures present.

Same Rules of Engagement as v1: 3 substantive fix attempts → BLOCKERS.md
→ move to non-dependent work. Never sit idle. Never fake correctness.

## STATUS.md mandate

Rewrite `tpu-inference/STATUS.md` atomically (write to STATUS.md.new then
`mv`) at the end of every meaningful step. The host loop greps it for
`^TPU`, `^Latest`, `^Tier`, `^W[1-4]` lines. Required structure:

```
# DeepSeek V4 v2 status
TPU preflight: ok / not_ok (<reason>)
Latest passing tier: T<N>  (or: still on T<N>)
Tier 1: <pass>/<total>
Tier 2: <pass>/<total>
Tier 3: <pass>/<total>
Tier 4: <pass>/<total>
Tier 5: <pass>/<total>  (or: skipped — fixtures missing / no TPU)
Tier 6: <pass>/<total>  (or: skipped — no TPU)
Tier 7: <pass>/<total>
W1 decode:    todo|wip|done|blocked
W2 paged-kv:  todo|wip|done|blocked
W3 __call__:  todo|wip|done|blocked
W4 dequant:   todo|wip|done|blocked

If killed now, next session must: <one line>
```

## Rules of engagement (same as v1, recapped)

1. Decide and document. Never wait for the user. At every fork, pick the
   path most likely to preserve correctness, then write the decision and
   rationale to DECISIONS.md.
2. Commit after every passing test. Tag with work item and tier.
3. Fail loudly to a log, never halt. On failure: write traceback +
   hypothesis to FAILURES.md, try up to 3 substantive fixes. If still
   failing, mark it in BLOCKERS.md and switch to an independent task. Do
   not sit idle.
4. Re-read before re-writing. Resist the urge to rewrite shared
   infrastructure. Prefer additive changes over modifications to v1's
   passing code.
5. Maintain INVARIANTS.md — every assumption you've validated. When
   something breaks, check whether an invariant was violated.
6. Never fake correctness. Loosening tolerance without an evidence-backed
   TOLERANCE_LOG.md entry is forbidden.
7. **No real-weight forward passes.** The temptation will arise. Resist
   it. The synthetic fixtures + Tier 4b real-bf16 spot-check + Tiers 5/6/7
   give you everything a real-weight run would, without the OOM and the
   wasted hours.
8. DeepSeek's `inference/model.py` is ground truth. If your
   implementation disagrees with the reference at
   `tests/models/jax/_deepseek_v4_reference/`, you are wrong until proven
   otherwise.
9. **Don't self-kill.** `pkill -f <pattern>` matches against full command
   lines, *including the shell argv of the bash you used to launch
   pkill*. So `pkill -f "vllm serve"` or `pkill -f "vllm.*serve"` will
   match your own shell — `pkill` won't kill itself but it will SIGTERM
   its parent shell, which propagates up and kills your tool call (exit
   143). This bug has cost real iterations. Instead, track PIDs:
   `vllm serve ... & SERVE_PID=$!` then `kill $SERVE_PID; wait $SERVE_PID
   2>/dev/null`. If you must use pattern-based killing, narrow to a
   pattern that can't appear in your own argv (e.g. `pgrep -f
   '^/.*python.*vllm.entrypoints'`).

## When stuck

If stuck on one problem >45 min:
  - Write problem, attempts, hypotheses to STUCK.md.
  - Try both: (a) simplify further (single layer, single head, single
    token); (b) print everything (shapes, dtypes, intermediate
    activations) on both sides and diff.
  - After another 30 min still stuck: mark BLOCKED, isolate behind
    `pytest.mark.xfail(reason=...)`, move to a non-dependent task.

Non-dependent fallback work (always available even if W1–W4 are stuck):
  - Tier 4b real-shard round-trip (no W-deps).
  - More Tier 1 tests for components that already pass.
  - Documentation polish in SUMMARY.md, PROD_TOPOLOGY_RISKS.md.
  - Increase decode parity coverage (more start_pos values, longer
    rolling tests).

## Markdown files (existing — do not delete; update or append)

PROGRESS.md, DECISIONS.md, V3_TO_V4_DIFF.md, TINY_CONFIG.md,
INVARIANTS.md, TOLERANCE_LOG.md, FAILURES.md, BLOCKERS.md, STUCK.md,
PROD_TOPOLOGY_RISKS.md, SUMMARY.md.

New in v2: STATUS.md, REGRESSIONS.md (only if you cause one),
TPU_PREFLIGHT.md (only if pre-flight fails — diagnostic for the user).

## Success criteria the user will check on waking

1. `cd /workspace/work/tpu-inference && JAX_PLATFORMS=tpu pytest
   tests/models/test_deepseek_v4.py -v` is green from a clean checkout
   of branch `deepseek-v4`. Tier 5/6 may show `skipped` if pre-flight
   failed; that's acceptable provided TPU_PREFLIGHT.md exists.
2. STATUS.md reflects the latest passing tier, with W1–W4 all `done` or
   the BLOCKERS.md entries explaining why not.
3. SUMMARY.md is updated with: what's new since v1, what's now verified,
   what's still residual risk, what the user should look at first.
4. No regressions to v1's 41 tests.
5. `vllm serve /mnt/scratch/tiny_v4_bf16` would work — Tier 5's curl
   round-trip is the proof.
6. `git log --oneline` shows commits tagged `W1:`, `W2:`, `W3:`, `W4:`,
   `T5:`, `T6:`, `T7:` corresponding to each milestone.

If you finish early, do NOT add features. Tighten Tier 5/6/7 tolerances,
add more decode parity points, polish SUMMARY.md. The user explicitly
traded performance for correctness — honor that.

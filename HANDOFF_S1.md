# S1 handoff — DECISIVE: MoE ROUTED-EXPERT COLLECTIVE corrupts REAL rows (per-process). Now FIX it.

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL (user-confirmed) — see memory [[s1-goal-reliable-coherence]]
DONE = **reliably coherent decode on EVERY fresh engine** (bug is a per-process coin flip). RIGOROUS gate:
**byte-identical (md5) FIB across 2 fresh engines AND coherent chat text.** Model is INSTRUCT — coherence
via `/v1/chat/completions` (system+user). Raw `/v1/completions` FIB = wedge-safe md5 determinism probe.

## STATE (2026-05-26 SESSION 17) — ROOT CAUSE NAILED via REAL-ROWS checksum [ckR]
S16 left a caveat: `moe_routed_y` GLOBAL sum differed A≠B, but that could be idle/pad-row noise. S17 added a
**real-rows-only** checksum `[ckR]` (rows < n_real, masking the per-process-constant pad garbage) in
`moe_forward` and ran 2 fresh engines (ENG_A FIB md5 `f3362d36`, ENG_B `aa9eb2ed` — DIFFER ⇒ bug reproduced;
first FIB token "21" agrees but its logprob already differs A `-0.00322` vs B `-0.09206`). [ckR] @ L0, BOTH
prompts (FIB + chat):
* `moe_input` real-rows: **A==B identical** (FIB 2.471e2, chat 2.428e2).
* `moe_shared` real-rows: **A==B identical** (FIB 7.088e1, chat 1.301e2).
* `moe_routed` real-rows: **A≠B DIFFERS** (FIB A 5.566e1 / B 5.214e1; chat A 3.142e1 / B 4.885e1).
⇒ **Same MoE input, clean dense shared-expert path, but the ROUTED-EXPERT COLLECTIVE produces different REAL-
token-row output per process.** NOT pad-row noise — it is REAL-row corruption = THE BUG. Localized to the
shard_map all_gather/einsum/psum in `deepseek_v4_moe.py::moe_forward` `_routed_local` (~L246-275). The S16/S14
"variance in seed" and S17's own "maybe decode" are both REFUTED — it's the PREFILL MoE routed collective.
Evidence logs: `logs/full-slice-v4-smoke-20260526T090613Z.log` (A), `...092907Z.log` (B). Per-engine
diag saved: `/tmp/s1_eng{A,B}_ckR.txt`.

## MECHANISM (hypothesis) + why prior fixes failed
flat_x PAD rows (global rows >= n_real) are uninit HBM (per-process garbage). The shard_map all_gathers x to
full [N,dim] on every rank, then `einsum('nd,eid->nei', x_full, W1)`. Pad-row garbage in x_full appears to
contaminate REAL rows through the routed compute (likely MXU tile cross-contamination from huge/denormal pad
values, OR the collective lowering) — the dense shared path (no collective) is CLEAN on the same input, which
is the tell. S13 output-row masking (zero y rows>=n_real AFTER) was REFUTED; S14 shard_map ALONE was PARTIAL.
⇒ Must kill the garbage on the INPUT side, BEFORE the collective/einsum.

## NEXT ACTION — FIX (input-side pad mask before the routed collective)
In `moe_forward` (`layers/jax/moe/deepseek_v4_moe.py`), zero flat_x PAD rows (global rows >= n_real) BEFORE the
shard_map (so x_l, hence x_full, has deterministic 0 in pad rows; SwiGLU(0)→0 so o pad rows = 0, psum clean,
real rows unaffected by garbage). Concretely, when `use_shard_map and n_real is not None`:
`flat_x_routed = jnp.where((jnp.arange(N) < n_real)[:,None], flat_x, 0)` and feed `flat_x_routed` into the
shard_map (keep the dense path + shared-expert path on the ORIGINAL flat_x — they're already clean & this
preserves bit-for-bit dense behavior). Reference template: production `fused_moe_gmm.py` masks
`token_topk_hidden` with `valid_rows_mask` BEFORE the per-rank sum/psum (it makes idle ranks emit zeros).
NO new `with_sharding_constraint` gathering a size-1/empty axis (pitfall #5). Then:
1. CPU-validate: `PYTHONPATH=work/tpu-inference:work/vllm work/vllm_env/bin/python3 scripts/s1_cpu_repro_v4flash.py both` → "OK: both match".
2. sync (`scripts/full_slice_v4_sync.sh`) + md5-verify (key `-i ~/.ssh/google_compute_engine`, user enyouki).
3. 2 fresh engines (warmup→fib2→chat each). PASS = `[ckR] moe_routed` real-rows A==B identical AND FIB md5
   A==B identical AND coherent chat (READ TEXT) AND survives 5 reqs. Compare via `/tmp/s1_ckdiff.py` or the
   uniq-count grep used in S17 (FIB-prompt [ckR] = the 4×-count value).
If real rows STILL differ after input masking → the contamination is in the collective op itself; fall back to
adopting production `fused_moe_gmm` for V4 routed experts (bigger change).

## KNOWN INSTRUMENTATION BUG (fix if you reuse [ckD])
`[ckD]` in `compute_logits` reads `_lg[-1]` = dp_rank-31 PAD row (T=32 = 1 row/dp_rank; real token is row 0
on an idle engine). So [ckD] argmax/top1/top2 AND the full-batch lsum are GARBAGE (31 of 32 rows are idle-rank
uninit). To salvage: real row = `jnp.argmax(jnp.max(_lg,axis=-1))` (auto-detect the non-degenerate row), or
move the fingerprint after `sample()` (tpu_runner.py:957). [ckR] is unaffected (it's real-rows-masked).

## KEEP cfc65ca4 (attn-seed input mask) + 5ca26d66 (shard_map): sound partials. The decode path (attention +
logits/sample tail) was re-audited CLEAN (no idle-rank uninit read affecting the real row) — do NOT re-hunt it.
Compressor/indexer-seed real-rows checksum patch is staged at `/tmp/s1_seedckR_patch.md` if ever needed (NOT —
[ckR] already localized to MoE routed; this is a dead lead now).

## DONE gate: md5-identical FIB across TWO fresh engines + coherent chat (READ TEXT). Engine CORE-HALTS on
PARIS shape → FIB-only for completions. Smoke serves max_model_len=256 (chat max_tokens<=256).

## Tools / ops
* Probes (RUN IN ORDER, one engine at a time): `/tmp/s1_warmup.py` (FIB, absorbs the ~348s cold compile of
  changed programs — MUST run first; s1_fib2's 180s timeout is too short for cold compile), `/tmp/s1_fib2.py
  <label>` (FIB×3 md5 determinism), `/tmp/s1_chat.py <label> [prompt]` (instruct coherence). Results →
  `/tmp/s1_eng{A,B}_results.txt`. `/tmp/s1_ckdiff.py LOG_A LOG_B` diffs diagnostics.
* xla_cache is WARM for current code (incl S17 [ckR]/[ckD] programs) ⇒ startup ~5-6 min. A NEW code edit
  recompiles only the changed programs (warmup absorbs it). KEEP the cache unless a launch-id halt.
* Checksums via `[ckR]` (real-rows, prefill L0, the trustworthy one) and `[ckS]` (GLOBAL, noisy). REMOVE ALL
  diagnostics ([ckR]/[ckS]/[ckD]/[fwd*]/[dec*]) when S1 closes.
* Slice HEALTHY (4 clean smokes S17, no halts). Guardians up. Reset CLEAN 0/32. `/tmp/s1_loop_stop` NOT set.

## DEAD (do not retry)
* Decode-path hunt (attention/KV + logits/select/sample tail) — re-audited CLEAN; real token row isolated.
* Seed hunt (attn seed clean per cfc65ca4; compressor/indexer was a candidate but [ckR] localized to MoE).
* MoE OUTPUT-row mask (zero y rows>=n_real AFTER, S13) — REFUTED. shard_map ALONE (5ca26d66) — PARTIAL.
* E%axis expert-slot mask (256%32==0, no-op). XLA collective-matmul flag (libtpu-rejected). fp32 matmul (S12).
  Option-A prefill-replicate (NaN/all-BOS). `wsc(activation,P())` gathering empty/idle/size-1 axis (Core-halts).

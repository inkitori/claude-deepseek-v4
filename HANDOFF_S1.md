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

## ⚠ INPUT-MASK FIX (a3982a2b) = INSUFFICIENT — proven on 2 fresh engines (S17, 2nd smoke pair)
Added input-side pad mask (zero flat_x + per_expert_weight rows >= n_real BEFORE the shard_map). Re-ran 2 fresh
engines: `moe_input` real-rows A'==B' identical (mask works — pad inputs ARE zeroed), `moe_shared` real-rows
A'==B' identical, but **`moe_routed` real-rows STILL DIFFER (A'=5.656e1 / B'=5.136e1) and FIB md5 STILL DIFFER
(1b95d044 / fdadf4b4)**. ⇒ With pad-row INPUTS provably zeroed, the collective STILL injects per-process
variance into REAL rows. The uninit-HBM read is in the **COLLECTIVE OP itself (all_gather and/or psum),
INDEPENDENT of input values** — NOT fixable by masking inputs. (A 10% divergence is far too big for FP
non-associativity; it's a gross uninit read.) Mask KEPT (sound: pad rows -> deterministic 0; may be needed
alongside the real fix). The `fused_moe_gmm` valid_rows-mask "minimal port" is REDUNDANT with this (our local
pad rows are already 0) — DEAD. Units of the mask verified correct (hc collapsed before MoE; n_real=seq_lens[0]
= positions = N; matches cfc65ca4's attn-seed mask).

## NEXT ACTION — isolate the collective op + cheap fix shots (in priority order)
The corruptor is inside `_routed_local`: `all_gather(x_l,tiled=True)` -> einsum -> `psum`. Real rows clean in
== out is violated. Try, cheapest first (each = CPU-validate, sync+md5, 2 fresh engines, PASS = `[ckR]
moe_routed` real-rows A==B AND FIB md5 A==B AND coherent chat, READ TEXT):
1. **optimization_barrier shot + isolation diagnostic (ONE smoke pair).** After `x_full = all_gather(...)`, add
   `x_full = jax.lax.optimization_barrier(x_full)` (breaks any XLA AllGather+Dot collective-matmul fusion that
   reads uninit HBM — S14's original suspicion; the explicit shard_map may not stop the *fusion*). ALSO add a
   global checksum of x_full INSIDE _routed_local at L0 (`jax.debug.print("[ckG] xfull_gsum={s:.9e}",
   s=jnp.sum(x_full.astype(jnp.float32)))`, fires 32x/rank, all identical) so the SAME smoke tells you: barrier
   fixed it (moe_routed A==B)? else is x_full A!=B (all_gather corrupts) or x_full A==B but moe_routed A!=B
   (psum/einsum corrupts)? [ckG]: pad rows already 0 so gsum == real-rows sum.
2. If x_full DIFFERS (all_gather is the corruptor): the tiled all_gather reads uninit on idle shards. Try
   `reduce_scatter`-free reforms, or replicate x via `with_sharding_constraint` to P() OUTSIDE the shard_map
   (CAUTION pitfall #5 — only safe on a POST-reduction [N,dim], and N here is the full padded token axis, NOT a
   size-1 decode axis, so a wsc to gather x to replicated may be OK — but VERIFY it doesn't Core-halt).
3. If x_full CLEAN but moe_routed DIFFERS (psum is the corruptor): the psum reads uninit. Try masking `local`
   rows>=n_real to 0 before psum (thread n_real into the shard_map) — though if x_full pad=0 this is already 0;
   alternatively replace `psum`+`dynamic_slice` with `psum_scatter`/`reduce_scatter`.
4. LAST RESORT: full `fused_moe_gmm` adoption (gmm grouped-matmul + ragged gather/scatter, zero_init kernel,
   bounded to valid rows so it NEVER touches uninit). BIG: V4 is dense (per_expert_weight[N,E], sqrtsoftplus,
   hash-routing) vs gmm's top_k-sparse (argsort/group_sizes/[E,2*inter] layout). Agent gap-analysis: needs a
   rewrite, not a direct call.

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

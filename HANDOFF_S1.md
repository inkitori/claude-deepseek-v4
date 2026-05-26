# S1 handoff — bug = MoE routed shard_map COLLECTIVE; narrowed to the einsum-or-psum (NOT the all_gather)

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL (user-confirmed) — see memory [[s1-goal-reliable-coherence]]
DONE = **reliably coherent decode on EVERY fresh engine** (bug is a per-process coin flip). RIGOROUS gate:
**byte-identical (md5) FIB across 2 fresh engines AND coherent chat text.** Model is INSTRUCT — coherence
via `/v1/chat/completions` (system+user). Raw `/v1/completions` FIB = wedge-safe md5 determinism probe.

## STATE (2026-05-26 SESSION 17) — DECISIVE localization via real-rows checksum [ckR]; 2 fixes refuted
Bug is per-process uninit-HBM in the **MoE routed-expert collective** (`deepseek_v4_moe.py::moe_forward`
`_routed_local` shard_map over 'attn_dp': `all_gather(x_l,tiled)` → expert einsums → `psum` → slice). Proven by
`[ckR]` (real-rows-only checksum, rows<n_real, masks pad garbage) on 8 fresh engines (4 pairs):
* `moe_input` + `moe_shared` real-rows: ALWAYS byte-identical A==B (clean input; dense shared-expert path,
  no collective, is clean).
* `moe_routed` real-rows: ALWAYS DIFFER per-process (>2x, e.g. 65.87 vs 28.03 — gross uninit, not FP noise).
* FIB md5 differs every engine (f3362d36, aa9eb2ed, 1b95d044, fdadf4b4, c769c732, 929659cd). Coherence is a
  per-process coin flip (some engines clean chat, some ramble/degenerate) — symptomatic of the same variance.

REFUTED THIS SESSION (all KEPT in tree — sound/harmless, none fix it):
1. **Input-side pad mask (a3982a2b):** zero flat_x + per_expert_weight rows>=n_real BEFORE the shard_map.
   moe_input confirms inputs ARE zeroed, but moe_routed STILL differs ⇒ uninit read is in the COLLECTIVE OP,
   not input values.
2. **optimization_barrier(x_full) (e1dcf39e):** break a possible XLA AllGather+Dot collective-matmul fusion.
   moe_routed STILL differs ⇒ not a fusion issue.
3. **[ckG] x_full checksum (e1dcf39e):** x_full (all_gather OUTPUT) is BYTE-IDENTICAL across processes
   (identical 7-value set A''==B'', `/tmp/ckg_{A,B}.txt`) ⇒ **the all_gather is NOT the corruptor.** Since
   x_full is deterministic but moe_routed (downstream) is not, the per-process variance ENTERS in the
   **expert einsum or the psum** inside `_routed_local`.

Decode path (attn/KV + logits/select/sample tail) + the attention seed were re-audited CLEAN (2 agents); the
[ckD] decode-logit probe is BUGGED (reads dp_rank-31 pad row) — ignore it. Do NOT re-hunt decode/seed.

## NEXT ACTION
1. **Isolate einsum vs psum (ONE diagnostic smoke pair, CHEAP/low-risk).** In `_routed_local`, after
   `local = o.astype(fp32).sum(axis=1)`, add (GLOBAL sum is valid here BECAUSE [ckG] proved x_full — the
   einsum input — is identical across processes; so any per-process diff in `local` is the einsum's fault):
   `if layer_idx==0: jax.debug.print("[ckL] r={r} local_gsum={s:.9e}", r=r, s=jnp.sum(local))`.
   CPU-syntax-check, sync+md5, 2 fresh engines, compare the `[ckL]` value SET (per rank `r`) A vs B (use
   `comm` like /tmp/ckg_*.txt). **local SET differs A≠B → the EXPERT EINSUM injects per-process uninit**
   (its output buffer or a matmul accumulator); **local SET identical but moe_routed differs → the PSUM** is
   the corruptor.
2. **Fix by culprit:**
   * EINSUM: try forcing fp32 accumulation / a clean output (the einsum on the all_gathered full [N,dim] with
     per-rank experts may leave/READ uninit; compare to the DENSE shared einsum which is clean — what differs
     is the shard_map manual context + [N,EP,inter] shape). Consider zero-init via `jnp.zeros`+add, or
     `optimization_barrier(o)`.
   * PSUM: a uninit-reading all-reduce; try `jax.lax.psum` replaced with an explicit reduce, or
     `psum_scatter`/`reduce_scatter`, or mask `local` rows>=n_real to 0 before psum (thread n_real in as a
     replicated `P()` shard_map operand — also enables an OUTPUT-side x_full mask if needed).
3. **LAST RESORT — adopt production `fused_moe_gmm`** (gmm grouped-matmul + ragged gather/scatter, zero_init
   kernel, bounded to valid rows → never touches uninit; the proven-deterministic path qwen3/v3 use). BIG:
   V4 is DENSE (per_expert_weight[N,E], sqrtsoftplus, hash-routing) vs gmm top_k-sparse (argsort/group_sizes/
   [E,2*inter] layout) — needs a rewrite, not a direct call. Agent gap-analysis was done (see git log S17).
4. Validate: 2 fresh engines → `[ckR] moe_routed` real-rows A==B AND FIB md5 A==B AND coherent chat (READ
   TEXT) AND survives 5 reqs (warmup+fib2×3+chat = 5). Then REMOVE all diagnostics, final smoke, record DONE,
   `touch /tmp/s1_loop_stop`.

## Tools / ops
* Probes (one engine at a time): `/tmp/s1_warmup.py` (FIB, absorbs the ~348s recompile of any changed program
  — MUST run first), `/tmp/s1_fib2.py <label>` (FIB×3 md5), `/tmp/s1_chat.py <label> [prompt]` (coherence).
* Extract per-engine diag: `grep -hoE '\[ckR\] L0 moe_routed: rsum=[-0-9.e+]+' LOG | sort|uniq -c|sort -rn`
  (FIB-prompt value = the 4×-count; chat = 1×). `/tmp/s1_ckdiff.py LOG_A LOG_B` also exists.
* xla_cache WARM ⇒ startup ~5-6 min; a code edit recompiles only changed programs (warmup absorbs ~350s).
  KEEP cache unless a launch-id halt. md5-verify after sync: key `-i ~/.ssh/google_compute_engine`, user enyouki.
* Diagnostics in tree: `[ckR]`/`[ckS]` (moe_forward), `[ckG]` (x_full, in shard_map), `[ckD]` (compute_logits,
  BUGGED-ignore), `[fwd*]`/`[dec*]`. REMOVE ALL when S1 closes.
* Slice HEALTHY (8 clean smokes S17, no halts). Guardians up. Reset CLEAN 0/32. `/tmp/s1_loop_stop` NOT set.

## DEAD (do not retry)
* all_gather as the corruptor — [ckG] proved x_full deterministic across processes.
* Input-side pad mask ALONE / optimization_barrier ALONE — both insufficient (KEPT, harmless).
* fused_moe_gmm valid_rows "minimal port" — redundant (our local pad rows already 0 via input mask).
* Decode/seed hunt (re-audited clean). [ckD] probe (reads pad row). S13 output-row mask. E%axis expert mask.
  XLA collective-matmul flag (libtpu-rejected). fp32 matmul (S12). prefill-replicate (NaN). `wsc(act,P())`
  gathering empty/idle/size-1 axis (Core-halts ~8x).

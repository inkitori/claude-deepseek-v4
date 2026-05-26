# S1 handoff — MoE shard_map shipped (partial); per-process variance is UPSTREAM in the attention SEED

Goal: coherent, RELIABLE decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash` on the v6e-32 slice.
Ops in `CLAUDE.md`; this is live state.

## GOAL (user-confirmed, S14)  — see memory [[s1-goal-reliable-coherence]]
DONE = **reliably coherent decode on EVERY fresh engine** (bug is a per-process coin flip). Validate
with the RIGOROUS gate: **byte-identical (md5) FIB across 2 fresh engines AND read the text for
coherence** (within a process the TPU is byte-deterministic, so a COMPLETE fix ⇒ identical across
processes; coherence alone passes by luck). The model is INSTRUCT — assess coherence via
`/v1/chat/completions` (system+user); the raw `/v1/completions` FIB loop is an off-distribution
ARTIFACT, use it only as the wedge-safe determinism (md5) probe.

## STATE (2026-05-26, SESSION 14)

**MoE shard_map fix SHIPPED but is at most PARTIAL — gate STILL FAILS.** Commit `5ca26d66`: rewrote
`moe_forward`'s routed-expert section as explicit `jax.shard_map` over 'attn_dp' (all_gather x +
per_expert_weight → local E/axis einsums → `jax.lax.psum` → dynamic_slice back; static gate
`use_shard_map = N%attn_dp==0 and N>=attn_dp` so PREFILL uses it, replicated decode N=1 + CPU use the
dense path). CPU-validated: shard_map == dense (MAXDIFF=0.0), `s1_cpu_repro both` match. **2-engine
slice test:**
* ENG_A: FIB md5 `b9876039` (×4 identical), chat "0,1,1,2,…,233,377" **correct**.
* ENG_B: FIB md5 `b5659d9c` (×3 identical), chat "1,1,2,…,233,377,**620**"  (**620 wrong**, =610).
* Both DECODE coherently (no hard "Fibs Fibs" collapse) but **DIFFER cross-engine ⇒ per-process
  variance PERSISTS ⇒ gate FAILS. MoE was NOT the (sole) source.**

**WHERE THE VARIANCE IS NOW (refined):** it's present at the **1st decode token's logprob** (ENG_A
gap 2.57 vs ENG_B gap 4.31, both argmax '21') ⇒ the **prefill-built SEED differs per-process**,
upstream of decode. The shard_map keeps MoE real rows clean *given clean input*, but the MoE input
real rows arrive **already contaminated from the ATTENTION SEED build**. ⇒ **NEXT TARGET = the
attention seed** (`attention_init_state_from_prefill`: seed KV / SWA / compressor / indexer reading
idle attn_dp-rank uninit HBM), NOT the MoE.

**METHODOLOGY FIX:** the `[ckS]` GLOBAL sums are **NOISY** — they include idle-rank uninit garbage
that varies request-to-request WITHIN one engine (ENG_A blk_moe_out swung 9.41e4↔9.73e4 while the
decode OUTPUT stayed byte-identical `b9876039`). So S13's "blk_moe_out differs cross-engine ⇒ MoE"
localization was **partly measuring noise**. Trust the decode **OUTPUT md5**; for localization use
**REAL-ROWS-ONLY** checksums (rows < n_real), not global sums.

## NEXT ACTION
1. Add **real-rows-only** (rows < n_real) checksums in the SEED path — esp. inside
   `attention_init_state_from_prefill` (seed_kv/SWA/compressor/indexer) and `block_init_state_and_
   forward` (slice real rows BEFORE the global sum). n_real is already traced/plumbed
   (deepseek_v4.py:1865, seed path). The real-row LAYOUT is the catch: for a single request all real
   tokens sit in ONE attn_dp rank's slice (rows [k·N/32, k·N/32+n_real)), not [0,n_real) — confirm
   the offset (which dp_rank) before slicing. (See [[s1-experiment-confounds]], runner
   `_prepare_inputs_dp` token_offset.)
2. Run 2 fresh engines, diff the real-rows checksums → the FIRST real-rows-divergent quantity is the
   true entry point. Fix it analogously (shard_map / kill the idle-rank read in that collective; NO
   size-1 token-axis wsc — pitfall #5).
3. Re-validate: 2 fresh engines md5-identical FIB + coherent chat (no math errors) + survives 5 reqs.
4. KEEP the MoE shard_map fix (sound, CPU-validated, removes the implicit collective-matmul; plausibly
   fixes the MoE source). Only revert if it complicates the attention-seed work.

## DONE gate: md5-identical FIB across TWO fresh engines + coherent chat (READ TEXT). Engine
CORE-HALTS on PARIS shape — FIB-only for completions. Smoke serves **max_model_len=256** (chat
max_tokens<=256).

## Tools / ops
* Probes: `/tmp/s1_warmup.py` (FIB, timeout 2400 absorbs cold compile), `/tmp/s1_fib2.py <label>`
  (FIB×3, md5+fib_terms — DETERMINISM probe), `/tmp/s1_chat.py <label> [prompt]` (instruct COHERENCE
  probe, max_tokens=200). Results saved `/tmp/s1_eng{A,B}_results.txt`.
* `[ckS]` (NOISY global sums) LIVE in block_init_state_and_forward + moe internals +
  attention_init_state_from_prefill; `_v4_checksum` at deepseek_v4_attention.py:70.
  **REPLACE with real-rows-only versions for localization; REMOVE ALL when S1 closes.**
* Slice HEALTHY (many clean smokes S14, no halts). Guardians up (node 497956, meta 4039835).
  Reset CLEAN 0/32. `/tmp/s1_loop_stop` NOT set. Cold compile of the shard_map ~330s on 1st request.

## DEAD fixes (do NOT retry)
* **MoE shard_map alone (S14, 5ca26d66)** — does NOT achieve cross-engine determinism (variance is
  upstream in the attention seed). KEPT as a sound partial fix.
* **XLA flag `--xla_tpu_*_collective_matmul_mode=none/post_spmd` (S14)** — REJECTED by this libtpu
  ("Unknown flags in XLA_FLAGS", engine crashes at init).
* MoE output-row / input-pos / pad-replicate masks (S6/S13) — no-ops (collective/idle-rank, not data
  values). Option A prefill-replicate → NaN/all-BOS (metadata layout mismatch). fp32 matmul (S12).
  `wsc(activation,P())` gathering empty/idle or size-1 token axis → Core-halts (~8×).

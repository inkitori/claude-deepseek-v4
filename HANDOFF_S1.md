# S1 handoff — DECISIVE: garbage enters at MoE LAYER-0; output-row mask REFUTED; next = fix the collective

Goal: coherent, **deterministic** decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash` on the
v6e-32 slice. Ops in `CLAUDE.md`; this is live state.

## STATE (2026-05-26, SESSION 13)

**1. DECISIVE LOCALIZATION — garbage enters at `moe_forward` LAYER 0.** Added `[ckS]` checksums in
the REAL seed path (`block_init_state_and_forward` — NOT `block_forward`; the S12 handoff named the
wrong fn) + ran 2 fresh engines (ENG1 fib-md5 `790d3884`, ENG2 `cdbeb0b8`). Cross-engine L0 diff:
* `seed_x_in` 1.861629395e+03 (BOTH), `blk_attn_out` 1.294913125e+05 (BOTH), `blk_post_attn_x`
  4.099475781e+04 (BOTH) — **BYTE-IDENTICAL**.
* `blk_moe_out` 9.972828125e+04 vs 9.896228125e+04 — **DIFFERS**.
⇒ attention + both hyperconnections (hc_pre/hc_post) are **EXONERATED**; per-process uninit-HBM
garbage FIRST enters in **`moe_forward` at L0**. Confirmed chain: L0 seed_kv identical; L1 seed_kv +
L1 seed_x_in differ (contaminated by L0 MoE). Mechanism (runner comment tpu_runner.py:1359-1374 +
no-op post-mortem 83f74395): the dense einsums force `flat_x`'s token axis to all-gather to
REPLICATED (via `_shard_e_mid`); idle attn_dp **RANKS** read uninit HBM; the expert all-reduce
(`out_NEd.sum(axis=1)` over the attn_dp-sharded E) folds it into the residual → prefill SEED.
NB E=256, attn_dp=32 → 8 experts/rank, **expert axis divides cleanly (no idle EXPERT ranks)** — the
idle ranks are on the TOKEN sharding, surfaced by the all-gather.

**2. FIX REFUTED — zeroing MoE output `y` rows >= n_real (post-gather replicated mask).** Threaded
traced `n_real` into the seed-path moe_forward, masked `y[rows>=n_real]=0` before return. ENG3: FIB
still COLLAPSES — incoherent ("21, ... Fibonacci Fibs Fibs Fibs"), and the 2 within-engine FIBs even
differed (md5 `4d610def` vs `743ee97d`). `blk_moe_out` dropped to ~1.3e2 (mask DID zero idle rows)
but coherence did NOT return ⇒ garbage is **not confined to pad-row VALUES**; it corrupts the REAL
rows (<n_real) via the all-gather/all-reduce collective itself (same class as the seed `_linear`
no-op 83f74395). Mask body REVERTED; `n_real` left plumbed into moe_forward for the next fix.

## NEXT ACTION — fix the COLLECTIVE (not output values).
Two candidates (pick after the diagnostic below):
* (A) **shard_map the expert reduction with an idle-RANK mask**: build `is_live` per attn_dp rank
  (rank owns real tokens?) from metadata, `jnp.where(is_live, out, 0.0)` then `jax.lax.psum` — mirror
  `layers/common/fused_moe_gmm.py:230-245` (the qwen3/v3 template; `valid_rows_mask` + psum). NO
  size-1 token-axis gather (pitfall #5).
* (B) **avoid the token-axis all-gather**: restructure moe_forward to keep `flat_x` token-sharded
  through the experts (only reduce over E), so no idle-rank token slots are ever gathered.
DIAGNOSTIC FIRST (cheap, 1 engine): add a per-row checksum of moe output at REAL rows only (rows
<n_real) and confirm whether real rows differ cross-engine — if yes, (A)/(B) needed; pin N (padded
prefill len; log shows seq_len=8192/256 — inconclusive). Also VERIFY the within-engine per-request
nondeterminism (ENG3 FIB1≠FIB2) is real vs a padding artifact — if real, fixes can be validated with
**FIB×2 in ONE engine** (much cheaper than 2 engines). Then re-verify: FIB coherent + byte-identical.

## DONE gate (unchanged): FIB coherent through 1597 (READ TEXT) + byte-identical across TWO fresh
engines + survives 5 reqs. Engine CORE-HALTS on PARIS shape — FIB-only when probing.

## Tools / ops
* `[ckS]` instrumentation LIVE in `block_init_state_and_forward` (blk_attn_out/blk_post_attn_x/
  blk_moe_out/blk_L_out gated layer<2) + per-shard-noisy moe internals (moe_perexpw/routed_y/shared,
  gated layer==0, **confounded — printed per-shard not global; ignore or fix to global**) +
  `attention_init_state_from_prefill` seed checksums. **REMOVE ALL `[ckS]` when S1 closes.**
* `_v4_checksum(name,x,layer_idx)` global sum in deepseek_v4_attention.py:70.
* `/tmp/s1_warmup.py` (1 FIB, absorbs cold compile). `/tmp/s1_ckdiff.py A B`. ENG1 vals saved
  `/tmp/s1_eng1_cks.txt`. n_real origin deepseek_v4.py:1865 (traced seq_lens[0]); plumbed seed-path
  only (h-path block_forward stays n_real=None → token-1 argmax untouched).
* Slice HEALTHY (5 clean smokes this session, no halts). Guardians up (node 497956). Reset CLEAN 0/32.
* Cold compile DEFERRED to 1st request (startup ~6min, then warmup ~220-330s). `/tmp/s1_loop_stop` NOT set.

## DEAD fixes (do NOT retry)
* **MoE output-row mask `y[rows>=n_real]=0` (S13)** — REFUTED, decode still collapses (garbage in real
  rows / collective, not pad values).
* fp32 matmul highest (S12); attention KV-SEED fixes (S12, L0 seed byte-identical); :775 seed
  all-reduce (S11 benign); pad/replicate seed token axis (S6/c731f592 no-op — masks SHARDED input pre-reshard).
* `wsc(activation,P())` gathering empty/idle or size-1 token axis → Core-halts (~8×).
* mask matmul INPUT x by position<n_real → no-op (garbage on idle RANKS not pad positions).

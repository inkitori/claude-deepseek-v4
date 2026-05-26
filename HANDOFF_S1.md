# S1 handoff — RUNTIME per-process nondeterminism CONFIRMED (uninit HBM); :775 + cold-compile both refuted

Goal: coherent, **deterministic** decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash` on the
v6e-32 slice. Ops in `CLAUDE.md`; this is live state.

## STATE (2026-05-26, SESSION 11) — DECISIVE: it is RUNTIME per-process, not cold-compile

**Clean experiment (never run before): 2 sequential engines from the SAME warm xla_cache (no code
change, no cache clear) → engine#2 reuses engine#1's IDENTICAL compiled executable.** Both warm
starts (275s / 315s). FIB×3 temp=0 seed=0 (`/tmp/s1_fib2.py`):
* ENG1: `21, 34, 55, 89, 144,`+whitespace | md5 32c0a09d | first-tok '21' lp **−0.05688** | 5 terms
* ENG2: `21,34,55,89,144,233,377,410,410,410…` | md5 30cb9775 | lp **−0.01035** | 7 terms→410-loop
* Each engine BYTE-IDENTICAL within itself (×3); **DIFFERENT across the two processes.**

⇒ **Same executable, different output per process = RUNTIME PER-PROCESS nondeterminism.** This
RULES OUT cold-compile reduction-order variance (executable is identical) AND confirms the only
remaining class: **uninitialized/recycled HBM read fresh per process** (fixed within a process →
byte-identical ×3; differs across processes; survives a fixed compiled schedule). The team's
idle-rank-uninit-HBM intuition was RIGHT; the LOCATION (:775 all-reduce) was WRONG (it's a benign
all-gather — S11 agents). Manifests as a tiny per-process logit perturbation that derails Fibonacci
at a process-dependent term (5 / 7 / full like S9) — NOT a step-1 collapse (first ~7 decode steps
are correct), so the seed is mostly-right but slightly contaminated.

Also nailed this session: greedy is RNG-free + seed=0 deterministic (NOT sampling); MoE expert-sum
has NO idle expert-ranks (8/rank) and routed experts are healthy ([moeRS]); the `[decL]/[fwdL]`
diagnostics were REMOVED from source (CLAUDE.md "always-on" is STALE); `_v4_nan_tripwire` is
env-gated (`V4_DECODE_NAN_TRIPWIRE`, default 0) AND only catches GROSS garbage (the `_linear`
`|r|<1e8` clamp suppresses it pre-tripwire; the FINITE residual is the real culprit and is unlogged).

## NEXT ACTIONS — ranked. Reuse the SAME 2-warm-engine FIB×3 A/B (`/tmp/s1_fib2.py ENG1`/`ENG2`,
compare md5 + READ TEXT) as the pass/fail test for ANY attempt: each engine ~5min warm start.
1. **(FASTEST gate shot — try FIRST) force full-fp32 matmuls via CODE, not env.** A tiny per-process
   perturbation only flips the argmax because bf16 matmul error (~1e-2) ≈ the derail-step logit gap;
   full fp32 may widen the correct-token margin enough to stay coherent AND deterministic even with
   the garbage present (band-aid, but the gate only needs coherent+deterministic). Use the CODE route
   — `jax.config.update("jax_default_matmul_precision", "highest")` at model init (synced to all 8
   hosts ⇒ NO env-propagation launch-id race; do NOT use the `JAX_DEFAULT_MATMUL_PRECISION` env var,
   it risks pitfall #0). Watch for fp32 HBM/OOM. sync → cold smoke → 2-engine FIB×3.
2. **(THE FIX if #1 fails) — but LOCALIZE before editing.** S11 agents argued the seed P()-combine is
   a benign all-gather, yet garbage demonstrably enters per-process SOMEWHERE — so don't blind-edit.
   First add a custom diagnostic (norm/checksum of the seed `kv_cache` + the post-`_linear` `kv` +
   the decode-step kv read), sync, cold smoke, run 2 engines, compare checksums → find the FIRST
   buffer that DIFFERS across processes. THEN zero idle-rank contributions at that spot: `shard_map`
   the producing op over ATTN_DATA, each rank emits `jnp.where(is_live, real, 0.0)` + `jax.lax.psum`
   to replicate (idle ranks → deterministic 0, NO token-axis gather that Core-halts). `is_live` from
   ATTN_DATA-sharded `seq_lens` (0 on idle ranks, tpu_runner.py:1502); precedents
   `_select_from_array_fn` tpu_runner.py:1152, `moe_gmm_local` valid-rows mask fused_moe_gmm.py:115.
   Likely target: deepseek_v4_attention.py `attention_init_state_from_prefill` (:1066) / `_linear` (:457).
   (`V4_DECODE_NAN_TRIPWIRE=1` only catches GROSS garbage — the clamp suppresses it — so it WON'T
   localize the finite residual; you must add a checksum print.)

## DONE gate (unchanged): FIB coherent through 1597 (READ TEXT) + byte-identical across TWO fresh
engines + survives 5 reqs. NB engine CORE-HALTS on the PARIS shape — fire FIB-only when probing.

## DEAD fixes (do NOT retry)
* `wsc(activation, P())` gathering empty/idle shards → Core-halts (~8×).
* mask matmul INPUT x / `position<n_real` on kv/x → no-op (garbage on idle RANKS not pad positions).
* zero compressor/indexer SEED-OUTPUT pad-slots → regressed (S6).
* ANY :775-seed-replicate fix (shard_map seed / is_live on packed buffer) → benign all-gather (S11).
* MoE `out_NEd.sum` mask → no idle expert-ranks (S11). Cold-compile determinism flags as the SOLE
  fix → executable is identical across the differing engines, so flags alone can't be the whole story
  (they may still help #1 by widening margins).

## Tools / ops
* `/tmp/s1_fib2.py <label>` (FIB-only ×3, engine-labelled, prints md5+fib_terms+first-tok lp). Reuse it.
* engine#1 log `logs/...T023908Z.log`, engine#2 `...T025001Z.log` (no internal diagnostics — prints removed).
* Reset CLEAN (0/32). Guardians up (3×node + meta). Slice HEALTHY (2 clean warm smokes this session).
* Warm-cache A/B caveat: small modules may recompile per-process (default thresholds); set
  `VLLM_XLA_CHECK_RECOMPILATION=1` to force-cache everything for a perfectly clean A/B.

## Durable lessons
* `wsc(ACTIVATION,P())` gathering empty/idle shards Core-halts. decode-vs-prefill ON SLICE is the test.
* `different launch id`/`Core halted` BEFORE startup = CODE DESYNC → sync + clear xla_cache. Don't reboot.
* Engine core-halts on the PARIS request shape — FIB-only when probing for the gate.
* (`/tmp/s1_loop_stop` NOT set.)

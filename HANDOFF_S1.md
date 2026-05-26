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

## NEXT ACTIONS — ranked (the bug is now well-localized: kill per-process uninit-HBM in the seed)
1. **(CHEAP, no code, try FIRST) `JAX_DEFAULT_MATMUL_PRECISION=highest`** (± determinism flags). A
   tiny perturbation only flips the argmax because bf16 matmul error (~1e-2) is the same order as
   the derail-step logit gap. Full-fp32 matmuls may widen the correct-token margin enough to stay
   coherent AND deterministic, orthogonal to the garbage. MUST propagate to all 8 ray workers
   (env-gated divergence → launch-id halt — see pitfall #0); confirm smoke.sh/ray_env propagates it
   BEFORE smoking. Cold smoke (cache differs) → 2-engine FIB×3 → want byte-identical + coherent.
   Optional add via `V4_XLA_FLAGS` (validate `python -c "import jax; jax.devices()"` first):
   `--xla_tpu_enable_latency_hiding_scheduler=false --xla_enable_async_all_reduce=false`.
2. **(THE FIX if #1 fails) zero idle-rank contributions in the prefill SEED build, structurally.**
   The reverted `wsc(x,P())` all-gather Core-halts; instead `shard_map` the seed-build activation
   over ATTN_DATA: each rank emits `jnp.where(is_live, real, 0.0)` then `jax.lax.psum` to replicate
   → idle ranks contribute deterministic ZERO, no token-axis gather. `is_live` from the
   ATTN_DATA-sharded `seq_lens` (0 on idle ranks, tpu_runner.py:1502); shard_map precedent
   `_select_from_array_fn` tpu_runner.py:1152, `moe_gmm_local` valid-rows mask fused_moe_gmm.py:115.
   Target file: deepseek_v4_attention.py `attention_init_state_from_prefill` (:1066) / `_linear` (:457).
   Cold smoke + 2-engine verify (byte-identical across engines AND coherent through 1597).
3. (diagnostic only) `V4_DECODE_NAN_TRIPWIRE=1` localizes GROSS garbage but NOT the finite residual;
   a custom seed-norm/checksum printed + compared across 2 processes would directly show the
   per-process-varying buffer. Use only if #1/#2 both miss.

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

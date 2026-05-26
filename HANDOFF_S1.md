# S1 handoff — SESSION 11 IN PROGRESS: S10 mechanism REFUTED; running the clean runtime-vs-compile test

Goal: coherent, **deterministic** decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash` on the
v6e-32 slice. Ops in `CLAUDE.md`; this is live state.

## STATE (2026-05-26, SESSION 11) — supersedes S10's pinned ":775 all-reduce-SUM" mechanism

S11 fan-out (6 read-only agents + firsthand code reads + log archaeology) **REFUTED the S10
mechanism** and re-grounded the bug:

* **S10's ":775 all-reduce-SUM into the replicated seed" is WRONG** (3 independent agents).
  The packed seed buffer is sharded on an UNCONTRACTED batch axis → `P(ATTN_DATA)→P()` lowers to
  an **all-GATHER (concat of empty idle-B-shards = nothing)**, NOT an all-reduce-SUM. The code
  itself says so (deepseek_v4_attention.py:1102). ⇒ every :775 / is_live / shard_map-the-seed fix
  is aimed at a BENIGN collective → that's why they're all dead. **Do not pursue :775.**
* **The MoE `out_NEd.sum(axis=1)` (deepseek_v4_moe.py:219) is NOT the culprit** either: E=256 /
  attn_dp=32 = **8 experts/rank, ZERO idle expert-ranks**; the einsum all-gathers real tokens over
  N. `[moeRS]` logs confirm routed experts are ALIVE & healthy (routed_mean 0.04–0.25, pew_sum
  ~1.5) — contradicting CLAUDE.md's "routed experts dead in decode" (stale claim).
* **Greedy decode is RNG-free + seed-deterministic** (seed=0, no time/PID; sampling.py:86-90,
  tpu_runner.py:315-320). Cross-engine variance is NOT sampling RNG.
* **The "~1e37 idle-rank seed garbage" is a HYPOTHESIS, not logged** — recent (May-26) smoke logs
  contain NO tripwire value, NO seed `_linear` diag, NO NaN. The `_linear` `|r|<1e8` clamp
  (deepseek_v4_attention.py:457-471) SUPPRESSES it from the tripwire, so it's unfalsifiable from
  logs. The S10 "derail" smoke `T015522Z` actually ended in a hard TPU **core-halt** (tpu5:pe2:0,
  host .204) on the PARIS probe — the derail+crash are confounded.
* Leading-but-UNCONFIRMED candidate for any real residual: attention seed `_linear` idle-rank
  uninit HBM (the clamp comment claims it; the finite sub-1e8 part would survive the clamp). NOT
  proven.

## IN FLIGHT — the clean experiment 10 sessions never ran (runtime-per-process vs cold-compile)
Every prior cross-engine comparison cleared xla_cache ⇒ a fresh COLD compile ⇒ "different engine"
was confounded with "different compilation". **Running 2 sequential engines from the SAME warm
xla_cache (NO code change, NO cache clear):** engine#2 reuses engine#1's compiled executable, so
any Fib-output difference is **runtime-per-process**; sameness rules runtime out → it was
cold-compile reduction-order variance.
* engine#1 smoke log: `logs/full-slice-v4-smoke-20260526T023908Z.log` (launched this session).
* probe: `/tmp/s1_fib2.py <label>` — FIB×3 temp=0 seed=0, prints md5 + fib_terms + first-tok lp.
  AVOID Paris (it core-halts the engine — fire FIB only).
* DECISION TREE when results land:
  - engine1 md5 == engine2 md5 AND both Fib coherent (…610,987,1597) → **runtime ruled out**; the
    S8/S9/S10 cross-engine diffs were cold-compile variance → apply determinism flags (below),
    re-verify TWICE; likely near-DONE.
  - engine1 != engine2 → runtime per-process confirmed → idle-rank seed `_linear` is the target;
    implement a per-rank deterministic-zero (shard_map+psum over ATTN_DATA, is_live from the
    ATTN_DATA-sharded `seq_lens` which is 0 on idle ranks — tpu_runner.py:1502; precedent
    `_select_from_array_fn` tpu_runner.py:1152). NOT a :775 fix.
  - both incoherent but identical → deterministic-but-wrong decode = a real bug, not nondet.

## Determinism flags (ready; via V4_XLA_FLAGS — smoke.sh ignores parent XLA_FLAGS; validate with
`python -c "import jax; jax.devices()"` on a worker first, unknown flags hard-fail libtpu):
`V4_XLA_FLAGS="--xla_tpu_enable_latency_hiding_scheduler=false --xla_enable_async_all_reduce=false"`
plus env `JAX_DEFAULT_MATMUL_PRECISION=highest`. (Caveat: `VLLM_XLA_CHECK_RECOMPILATION=1` forces
caching ALL modules so a warm A/B has zero per-process recompiles.)

## DEAD fixes (do NOT retry — proven S10 + history)
* `with_sharding_constraint(activation, P())` anywhere → Core-halts via empty shards (~8×).
* mask the matmul INPUT x → no-op (83f74395). explicit `position<n_real` mask on kv/x → no-op.
* zero compressor/indexer SEED-OUTPUT pad-slots → regressed (S6, b026d9ff).
* ANY :775-seed-replicate fix (shard_map seed / is_live on packed buffer) → aimed at a benign
  all-gather (S11). MoE `out_NEd.sum` mask → no idle expert-ranks to fix (S11).

## Tools / ops
* `/tmp/s1_fib2.py` (Fib-only ×3, engine-labelled), `/tmp/s1_gate.py` (full gate, but Paris crashes
  the engine), `/tmp/s1_confirm.py`. Probe at :18081 raw /completions, FIB first, ONE shape.
* Reset CLEAN (0/32). Guardians up (node+meta). Cache mtime 00:28 (warm) — do NOT clear unless code changes.

## Durable lessons
* `wsc(ACTIVATION,P())` gathering empty/idle shards Core-halts (NOT a size-1-only rule).
* `different launch id`/`Core halted` BEFORE startup = CODE DESYNC → sync + clear xla_cache. Don't reboot.
* decode-vs-prefill-everything ON THE SLICE is the faithful test; CPU parity non-predictive.
* Engine core-halts on the PARIS request shape — fire FIB-only when probing for the gate.
* (`/tmp/s1_loop_stop` NOT set.)

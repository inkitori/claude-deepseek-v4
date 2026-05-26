# S1 handoff — gate FAILS via CROSS-ENGINE idle-rank seed contamination (mechanism pinned)

Goal: coherent, **deterministic** decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash` on the
v6e-32 slice. Ops in `CLAUDE.md`; this is live state.

## STATE (2026-05-26, SESSION 10) — supersedes S9's "input-specific jitter" framing

**The success gate FAILS on a fresh engine.** Smoke `...015522Z`, HEAD 98b25474 (NO code change
since S9), probe `/tmp/s1_gate.py`:
* FIB temp=0/seed=0 ×3 → **byte-identical WITHIN the engine** (md5 0781ab9b) but the TEXT is WRONG:
  `21,34,55,89,144,233,377,` **`410,611,632,656,656,656,656`** — correct through 377 (term 7) then
  DERAILS (410≠610) and collapses to a 656-loop. First-tok '21' logprob **−0.02520** (gap 5.5).
* S9 on the SAME code/prompt/seed: correct through `610,987,1597`, logprob **−0.19027** (gap 3.6).
* ⇒ **CROSS-ENGINE NONDETERMINISM**: identical input, different engine → different logits
  (−0.025 vs −0.190) → different trajectory, severe enough to **break Fibonacci coherence on a
  "bad-seed" engine.** Within-engine deterministic; ACROSS fresh engines nondeterministic. The gate
  (coherent + deterministic, verified TWICE on fresh engines) cannot pass while a fresh engine can
  draw a bad seed → incoherent looping.

S9 was wrong twice: (1) "Fib exact / input-specific" — Fib is NOT immune, it was a lucky-seed engine;
the jitter is PERVASIVE, derailment depends on the random per-engine seed. (2) "refutes the collective"
— it's not benign reorder; see mechanism. (Peaked-vs-flat masks the jitter in a logprob but not in the
multi-step trajectory.)

## MECHANISM (pinned S10: 8 read-only agents + firsthand reads + [[s1-validated-facts]])
Idle attn_dp ranks read **uninitialized/recycled HBM** in the PREFILL seed build:
* Prefill activation is `P(ATTN_DATA)` and **ATTN_DATA binds the leading B axis (size 1)** (tpu_runner
  :1380; ids reshaped [1,T] deepseek_v4.py:1824). So **1 rank holds the whole [1,T,dim]; ~31 ranks hold
  EMPTY B-shards.** Idle ranks materialize FULL-shape output buffers (seed output spec is replicated
  `P()`) = uninitialized HBM = finite garbage (V4_DECODE_NAN_TRIPWIRE: L1 kv-matmul ~1e37, varies
  run-to-run; the `_linear` `|r|<1e8` clamp catches NaN/big but NOT finite sub-1e8).
* That garbage is **all-reduce-SUMmed into the replicated P() seed** at `with_sharding_constraint(b,P())`
  (`_v4_constrain_packed_replicated`, deepseek_v4.py:775, called :859) — GSPMD sums all shards (can't
  prove n_real liveness statically). Contaminated-AND-read = the SWA `kv_cache[:,:win]` seed.
* Garbage is on idle **RANKS**, NOT pad **positions** ⇒ any live-rank position/input mask MISSES it.

## DEAD fixes (do NOT retry — proven S10 + history)
* `with_sharding_constraint(activation, P())` anywhere → Core-halts via empty shards (f1598b82, 7500c742).
* mask the matmul INPUT x → no-op (garbage is output-driven, not input; 83f74395).
* explicit `position<n_real` mask on `kv`/`x` → **no-op, WRONG AXIS** (garbage on a different RANK, not a
  pad position; confirmed S10 by fix-design agent).
* zero compressor/indexer SEED-OUTPUT pad-slots → regressed (S6, b026d9ff; boundary slot read pre-overwrite).

## NEXT ACTION — force IDLE-RANK contributions to deterministic ZERO before the :775 collapse (no gather)
Two tractable directions (respect every dead-fix constraint; each needs a COLD-compile smoke):
1. **(preferred) `shard_map` the seed build** `transformer_body_init_state_to_buffer` (deepseek_v4.py:853)
   with `in_specs=P(ATTN_DATA), out_specs=P()` so each rank's LOCAL program runs on its own B-slice and
   empty-B-shard ranks provably emit ZEROS (not recycled HBM), replacing the GSPMD all-reduce-SUM at :775.
   No token-axis gather → no Core-halt. RISK: shard_map empty-shard semantics may still materialize garbage.
2. **(alt) per-rank `is_live` mask on the PACKED POST-REDUCTION buffer** before the wsc at :775: multiply
   the `[N,dim]` packed seed by `is_live` (1 on the rank the request landed on, 0 on idle ranks). A
   post-reduction `[N,dim]` op = SAFE (pitfall #5). OPEN: source the per-rank is_live (n_real>0 /
   num_scheduled_tokens_per_dp_rank>0) — may need threading a per-rank array into the jit.
FIRST STEP: 1-2 read-only agents to (a) confirm the EXACT sharding of the packed buffer at :775 + whether
shard_map zeros idle shards, (b) locate/derive the per-rank is_live signal. THEN implement → sync (clear
xla_cache → COLD 10-30 min) → smoke → verify Fib COHERENT (correct through 1597) AND byte-identical on
TWO fresh engines.

## Tools / ops
* Probe `/tmp/s1_gate.py` (FIB×3 critical-first + PARIS×3 + 5 misc; reads text, fib_terms, logprobs).
  `/tmp/s1_confirm.py` (Mars×6 + history) also present.
* **ENGINE CRASHES on PARIS** (new shape): 015522Z died with a TPU register-dump (proto_based_error_
  collector) + "Shutting down" when Paris followed the Fib×3 fires. Fire FIB first; ONE shape/smoke;
  the gate's Paris/robustness clauses likely need a separate smoke or a less-fragile prompt set.
* Slice reset CLEAN (0/32, no failures); guardians up. xla_cache WARM now, but the FIX is a code change
  ⇒ next smoke is COLD (sync + clear xla_cache on 8 hosts, 10-30 min).

## Durable lessons
* HARD (~8×): `wsc(ACTIVATION, P())` gathering empty/idle shards Core-halts (NOT a size-1-only rule).
* `different launch id`/`Core halted` BEFORE startup = CODE DESYNC → sync + clear xla_cache. Don't reboot.
* decode-vs-prefill-everything ON THE SLICE is the faithful test; CPU parity non-predictive.
* (`/tmp/s1_loop_stop` NOT set — gate fails, real fixable work remains.)

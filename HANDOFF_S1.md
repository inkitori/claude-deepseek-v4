# S1 handoff — decode COLLAPSE fixed; residual = same-engine temp=0 NONDETERMINISM (now confirmed, fixable)

Goal: coherent, **deterministic** decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash` on the
v6e-32 slice. Ops in `CLAUDE.md`; this is live state.

## STATE (2026-05-26, SESSION 8)

**S1 token-2 decode COLLAPSE = FIXED** (reconfirmed across 3 fresh engines + 2 prior sessions:
greedy Fibonacci `21..610`, Paris coherent, decode faithful to prefill-everything). The
"open-ended looping" is the MODEL on flat/OOD prompts (prefill-everything loops identically);
the success gate `smoke_check.sh` is BROKEN as a loop detector (max_word_run=1 on an obvious
PHRASE loop → false-PASS). Chat WITH a system prompt → correct coherent content. See memory
[[s1-decode-faithful-verified]] + [[s1-gate-false-positive-phrase-loops]].

**NEW THIS SESSION — same-engine temp=0 NONDETERMINISM CONFIRMED (the residual; FIXABLE).**
Filled the gap the peer couldn't (engines kept dying). Mars flat prompt, raw `/v1/completions`,
temp=0/seed=0, **3 fires on ONE live engine → 2 unique outputs**: runs 0&1 first-tok `' they'`
**but logprob differs −1.888 vs −1.905 for the SAME text**; run 2 flipped to a different leader
`' one'`. **Logits jitter run-to-run even when argmax is stable** ⇒ NOT inherent FP-reduction-order
(a fixed compiled program is bit-reproducible) ⇒ a REAL nondet source: uninit-HBM / idle-rank /
nondet collective — plausibly a RESIDUAL of the S1 padding/uninit-KV family (pad-position KV blocks
not zeroed → flat-prompt attention reads varying garbage; sharp prompts' signal dominates → Paris/
Fibonacci stay deterministic). So the COLLAPSE is fixed but the **"deterministic decode" goal is
still violated on flat prompts** — and now it looks fixable, not inherent.

**OPS HARDENED (committed, synced, validated live):** `smoke.sh` now has (1) a flock(9) single-
instance GUARD — concurrent loop sessions were racing into 2 engines that fought for the 32-chip
PG (happened twice in 15min, wedging the slice); the backgrounded serve inherits fd 9 so the lock
is held engine-lifetime; a 2nd/concurrent smoke is REFUSED. (2) `VLLM_ENGINE_READY_TIMEOUT_S=2400`
— a COLD compile (cache cleared) is 10-30min and blew vllm's 600s default → APIServer "Timed out
waiting for engine core" → serve died (this killed 2 smokes this session). Prior "clean" smokes
only worked on a WARM cache. Both fixed.

## NEXT ACTION (engine reset, slice CLEAN 0/32, guardians up)
Two leads; (1) is slice-free + cleaner, do it first:
1. **TORCH-REFERENCE (slice-free, decides looping=model-vs-serving):** run the flat Mars prompt
   through `tests/models/jax/_deepseek_v4_reference/model.py`. Loops too → model is repetition-prone
   (accept + mitigate w/ sampling). Coherent where slice loops → serving degrades = fixable bug.
2. **LOCALIZE the nondet source (the new residual):** same-engine flat-prompt temp=0 is now proven
   non-bit-reproducible. Suspect KV-cache pad-block zeroing / a nondet reduction. Re-smoke (sync→
   reset→smoke→wait "Application startup complete", ~6min WARM / 10-30min COLD), then fire ONE fixed
   shape via raw `/completions` (NOT chat — chat-path resharding wedges the engine; a NEW request
   shape also wedges it: probe ONE shape only). Verify a fix → same-engine flat temp=0 byte-stable.
   Tools: `/tmp/s1_det.py` (same-engine determinism), `/tmp/s1_gate_supp.py` (control/gate/faith/sys).

## Real fixes (committed, live)
* COLLAPSE: SWA seed `6245ea84` + compressor/indexer STATE `90bf85c3` thread traced `n_real`;
  metadata-replicate decode `_v4_decode_replicate`; diagnostics removed `77c0c7be`.
* OPS this session: `smoke.sh` guard + ENGINE_READY_TIMEOUT (see git log "S1 ops:").

## Durable lessons / recovery
* **MULTI-SESSION:** >1 loop session can be active (a peer wrote memory; a human-driven session was
  referenced). The new guard stops smoke-collisions, but reset.sh has NO guard — a peer could reset
  your engine mid-probe. Check `ps -eo pid,cmd|grep 'bin/vllm serve'` is YOURS before probing.
* **Engine fragility:** dies/wedges (HTTP500→connection-refused, process stays alive) when a request
  triggers a new-shape XLA compile (esp. chat-path resharding). Raw `/completions` SAME shape survived
  3 fires. Probe one shape; reset + re-smoke if wedged.
* decode-vs-prefill-everything ON THE SLICE is the faithful test (judge coherence + determinism +
  near-tie gaps, NOT byte-identity; CPU parity non-predictive).
* HARD (~8×): `with_sharding_constraint(ACTIVATION, P())` gathering a size-1 decode token axis Core-halts.
* `different launch id`/`Core halted` before startup = CODE DESYNC → `full_slice_v4_sync.sh` + clear
  `~/.cache/vllm/xla_cache/*` on all 8 hosts. Don't reboot. Reset: `full_slice_v4_reset.sh`. Keep
  node_guardian + meta_guardian alive. (`/tmp/s1_loop_stop` NOT set — real fixable work remains.)

# S1 handoff — decode COLLAPSE fixed; residual = INPUT-SPECIFIC uninit-HBM logit jitter

Goal: coherent, **deterministic** decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash` on the
v6e-32 slice. Ops in `CLAUDE.md`; this is live state.

## STATE (2026-05-26, SESSION 9)

COLLAPSE = FIXED (reconfirmed: greedy Fibonacci `21,34,55,89,144,233,377,610,987,1597` correct).
Residual = run-to-run **logit JITTER** at temp=0, now sharply characterized on a fresh engine
(probe `/tmp/s1_confirm.py`, smoke log `...010801Z`):

* **NONDETERMINISM CONFIRMED, n=6:** Mars flat prompt temp=0/seed=0 → **3 unique/6**. mars0-3 byte-
  identical coherent text but chosen-tok logprob jitters −2.03/−1.63/−1.80/−1.61; mars4,mars5 DIVERGED
  into loops. ⇒ the jitter sometimes flips the argmax and pushes decode into a loop attractor — the
  "open-ended looping" is **partly a SYMPTOM of the jitter**, not purely the model.
* **INPUT-SPECIFIC, not pervasive:** Fibonacci is **EXACT twice** — text AND logprob (−0.19027, gap 3.6)
  byte-reproduced. ⇒ jitter hits Mars but NOT Fib. This **REFUTES the "nondeterministic attn_dp collective"
  verdict** (2 code-audit agents claimed `out_NEd.sum(axis=1)` all-reduce reorders run-to-run): a
  reordering all-reduce runs every layer for Fib too and would jitter Fib's logprob — it's exact to 5 dp.
* **~0.4 nats** jitter is far too large for FP-reduction-order noise (bf16 ≈1e-3) ⇒ an **UNINITIALIZED-HBM
  read**, not a benign reorder. mars0 (−2.03) is a cold-buffer warmup outlier vs warm −1.6..−1.8.
* **NOT stale-KV reuse:** history test mars-A==mars-C (the intervening Fib did NOT determine the result);
  code audit confirms decode KV is rebuilt wholesale per request (`jnp.zeros` once + `concatenate` of
  fully-defined arrays) and SWA/compressed/indexer decode masking is EXACT. So it's a **per-request uninit
  read in the FORWARD**, input-dependent.
* Ruled out this session: torch-reference cross-check (INFEASIBLE on host — random-weights-only toy config,
  fp8/fp4 weights it can't consume, ~554GB > 391GB RAM); prefix-caching artifact (APC explicit `--no-...`).

LEAD (sharpened; matches memory [[s1-prefill-padding-contamination]]): the prefill compressor/indexer are
NOT `n_real`-aware → they process PAD positions → pad-contaminated `comp_full`/`i_cache` seed, hidden by the
`nan_to_num` / `|r|<1e8` clamp; input-LENGTH-specific (Mars's length contaminates its selected logits, Fib's
doesn't). NB: naive zeroing of `comp_full`/`i_cache` SEED pad-slots was DISPROVEN (regressed decode, S6) —
the fix must thread the traced `n_real` so pad positions are NEVER processed, not post-hoc zeroed.

## NEXT ACTION (slice CLEAN 0/32, engine reset, guardians up)
LOCALIZE the uninit buffer, then fix its `n_real`-awareness:
1. Re-add a MINIMAL decode-step checksum diagnostic — `jax.debug.print` of sum/std of attn-out vs moe-out
   vs compressor-seed at decode step 1 (pattern = the just-removed `[fwdS]`/`[decS]` prints, commit
   `77c0c7be`). sync → reset → smoke (~6min WARM) → fire Mars ×2 (`python3 /tmp/s1_confirm.py`, or trim it
   to mars-only) → see which checksum VARIES run-to-run. Pinpoints attn vs MoE vs compressor.
2. Make THAT buffer `n_real`-aware so prefill compressor/indexer skip pad positions. Verify on a fresh
   engine: same-engine Mars temp=0 ×6 BYTE-IDENTICAL **and** Fib still exact.

## Tools / ops
* Probe: `/tmp/s1_confirm.py` (Mars×6 nondet + history discriminator + Paris + Fib, logprobs, critical-first,
  guarded). `/tmp/s1_det.py`, `/tmp/s1_gate_supp.py` also present.
* **ENGINE FRAGILITY (reconfirmed):** died (HTTP 000) after ~9 fires when Paris (a NEW prompt shape) fired.
  Fire critical probes FIRST; prefer ONE prompt shape per smoke; reset+re-smoke if 500/000.
* No code changed this session → xla_cache is WARM; do NOT clear it (keeps smoke ~6min not 10-30min).
* **MULTI-SESSION:** a peer session (PID 3199849, tmux claude-5:5) STOOD DOWN and is idle at a prompt —
  harmless. Verify `ps|grep 'bin/vllm serve'` is yours before probing.

## Real fixes (committed, live)
* COLLAPSE: SWA seed `6245ea84` + compressor/indexer STATE `90bf85c3` thread traced `n_real`;
  metadata-replicate decode `_v4_decode_replicate`; diagnostics removed `77c0c7be`.
* OPS: `smoke.sh` flock guard + `VLLM_ENGINE_READY_TIMEOUT_S=2400` (S8).

## Durable lessons
* HARD (~8×): `with_sharding_constraint(ACTIVATION, P())` gathering a size-1 decode token axis Core-halts.
* `different launch id`/`Core halted` before startup = CODE DESYNC → `full_slice_v4_sync.sh` + clear
  `~/.cache/vllm/xla_cache/*` (only when code CHANGED). Don't reboot. Reset: `full_slice_v4_reset.sh`.
* decode-vs-prefill-everything ON THE SLICE is the faithful test; CPU parity non-predictive.
* (`/tmp/s1_loop_stop` NOT set — real fixable work remains: the jitter fix.)

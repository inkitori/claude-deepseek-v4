# S1 handoff — fresh session, pick up here

Goal: make `vllm serve deepseek-ai/DeepSeek-V4-Flash` produce coherent, deterministic
decode on the v6e-32 slice (bug **S1**). Bypass perms; use the TPU; commit+push
checkpoints; never wait. Operational details (slice-serving protocol, validation,
pitfalls) are in `CLAUDE.md` — this file is the live debugging state.

## ⇒⇒⇒ SESSION 4 UPDATE (2026-05-25) — READ THIS FIRST (corrects SESSION 3)

**The SESSION-3 ref-vs-decode comparison was CONFOUNDED by a diagnostic bug: the
prefill "reference" measured a PAD token (id 0), NOT tok1.** So the headline
"57x dead routed MoE / 4x embed attenuation" ratios are NOT valid evidence — they
compared {pad token id-0 in prefill} against {real tok1 id-1602 in decode}.

Root of the confound (source-confirmed): prefill right-pads `input_ids` with 0
(`tpu_runner.py:1487` `input_ids_cpu[total_num_scheduled_tokens:] = 0`), and the
diagnostic `_v4_lp(z)` / `[moeRS]` took `z[:, -1]` / `[-1]` = the LAST BUFFER slot
= a PAD (id 0), not the real last token. Decode is unpadded (S=1) so its `[decS]/
[decL]/[moeRS]` correctly read tok1 — only the PREFILL side was wrong.

Evidence (cheap, no smoke; from the live 094300Z engine log + tokenizer + embed_w):
* Both the ref prefill (`PROMPT+"21"`, real last tok 1602) AND the test prefill
  (`PROMPT`, real last tok 223 `" "`) printed the IDENTICAL `[fwdS] L-1 embed
  max=0.629 mean=0.135`. Different prompts, same value ⇒ it's reading a constant =
  embed of id 0, not the prompt's real last token.
* `embed_w[0]` (BOS/pad) has mean|.|=0.13497, max|.|=0.62891 — byte-matches the
  "reference" embed. `embed_w[1602]` ("21") has mean|.|=0.03382, max|.|=0.14160 —
  byte-matches the DECODE `[decS]` embed (0.0338 / 0.1416).
* Tokenizer: `tokenize(PROMPT+"21") == tokenize(PROMPT) + [1602]` (clean boundary),
  and decode's embed == true `embed_w[1602]` ⇒ decode DOES process the right token.

CONSEQUENCES:
* **EMBED IS EXONERATED.** The decode embed is the EXACT correct `embed_w[1602]` —
  there is NO embed attenuation, magnitude or directional. (Concern-1b resolved.)
* **The "dead MoE" localization needed a same-token re-test — done below: DISPROVEN.**
* **Still solid:** decode collapses to a deterministic attractor; prefill-everything
  is coherent; CPU passes. The MECHANISM of the decode collapse is RE-OPENED.

FIX APPLIED THIS SESSION (diagnostics only): `_v4_lp(z, input_ids)` and `[moeRS]`
now index the LAST NON-PAD position (`input_ids != 0`), so prefill measures the
SAME token (1602 @ pos 29) as decode. Decode call sites unchanged (S=1).

### CORRECTED-SMOKE RESULT (smoke 113324Z) — MoE THEORY DISPROVEN; bug is the decode KV/attention path

With the corrected same-token reference (prefill `PROMPT+"21"` @ "21" vs decode "21"):
* **MoE MATCHES.** `[moeRS]` L0: ref_routed 0.0788 vs dec_routed 0.0779 (**ratio 0.99**),
  pew_sum IDENTICAL (1.501 == 1.501). L1/L2 likewise. ⇒ routed experts are NOT dead,
  routing weights NOT collapsed. The entire "57x dead MoE" was the pad confound. **Stop
  looking at the MoE.** (For the real token "21", routed≈0.08 < shared≈0.7 in BOTH paths
  — routed-dominates was a property of the id-0 pad token, not a universal truth.)
* **The divergence is a SMOOTH COMPOUNDING error from L0, NOT attenuation.** blk-mean
  ref→dec ratio: L0 1.01 (match) → L3 1.21 → L9 1.71 → L12 2.07 → **L17 4.34 (peak)** →
  reconverges to ~1.1 by L40 (ref grows to meet decode). Decode L0 *attn* already 1.35x
  ref (0.027→0.036). Magnitudes stay within ~2-4x everywhere ⇒ the fatal error is
  **DIRECTIONAL** (magnitude fingerprints can't localize it; need cosine/direction).
* **Logit-level (decisive):** ref predicts "," at logprob −0.0 (prob ~1.0, confident,
  correct Fibonacci). The decode step's distribution is FLAT/high-entropy: "." −1.82,
  "\n\n" −2.2, " " −2.36, "," −2.41 (correct token demoted to rank 4, max prob ~0.16).
  Long gen = "21. The number of the number of the number of…" (degenerate attractor).
  ⇒ decode hidden state is directionally corrupted → flat logits → attractor.
* token1 "21" is confident/correct (prefill argmax, logprob −0.01). Only the DECODE
  STEP collapses (as always).

⇒ **S1 is back in the decode-state / KV-seed / decode-step-attention path** (where
SESSION-2 was before SESSION-3's debunked MoE pivot). The error enters at L0 decode
attention (reads the seeded KV) and compounds. "prefill-everything coherent" + "CPU
passes" + "MoE matches" all point here. OPEN: is it (a) the SEED KV values written by
prefill being subtly wrong (finite≠correct — `[dech]` only checks NaN/max, not values),
or (b) the decode-step attention MATH (`attention_decode_step`) computing wrong over a
correct seed? Magnitude diags can't split these — need a DIRECTION-sensitive diagnostic
(cosine of decode vs prefill per-layer/per-component) or to inspect deepseek_v4_attention.py.
NOTE the SESSION-2 seed fixes all FAILED (NaN/halt) — see HARD CONSTRAINT; don't gather
the activation to replicated.

IN FLIGHT (cheap to resume — no new smoke needed yet): the smoke-113324Z engine is STILL
SERVING on :18081 with the PAD-CORRECTED diagnostics (HEAD code). Probe it directly:
`/tmp/s1_seedstep_probe.py logs/full-slice-v4-smoke-20260525T113324Z.log "<prompt>" "<tok1>"`
(full ref-21-vs-decode-21 table → /tmp/s1_seedstep_full.txt), `/tmp/s1_moe_probe.py` ([moeRS]
routed/shared/pew_sum + embed/attnout/moeout), `/tmp/s1_logits_probe.py` (top-8 + long gen).
NEXT ACTION: add a DIRECTION-sensitive signal (the bug is directional, not magnitude — mag
diags top out within 2-4x) and/or read `layers/jax/attention/deepseek_v4_attention.py`
(`attention_decode_step` vs `attention_prefill`, `attention_init_state_from_prefill`) to
decide SEED-KV-values-wrong vs decode-attention-MATH-wrong. Engine crashes on internal NaN
after a few requests — fire critical probes first; re-smoke via the CLAUDE.md protocol if down.

## SESSION 3 (2026-05-25) — SUPERSEDED by SESSION 4 (its dead-MoE conclusion was a pad confound)

Durable facts that survive from S3 (the rest — "dead MoE / embed attenuated / residual
never accumulates / 57x" — was the pad-token artifact; IGNORE it):
* Collapse is DETERMINISTIC at temp=0 (the metadata-replicate decode fix in
  `tpu_runner._prepare_inputs_dp` — V4-only, places single-seq decode metadata
  REPLICATED P() — suppressed the old non-determinism). token1 correct (prefill argmax);
  the FIRST decode step is wrong.
* Always-on, race-proof diagnostics exist (env-gated module reads race across ray workers
  → launch-id halt, so they're hardcoded): `[fwdL]/[decL]` per-layer last-pos, `[fwdS]/
  [decS]` embed/attnout/moeout L0-2, `[moeRS]` routed/shared/pew_sum — ALL now PAD-CORRECTED
  (index the last non-pad token). Probe `/tmp/s1_seedstep_probe.py LOG PROMPT TOK1`;
  jax.debug.print drops/reorders under volume → re-fire. Remove all S1 diag prints at close.
* Slice is HEALTHY when code is synced (md5 head==workers).

## FIXES TRIED → none fix the collapse (do NOT repeat)

Seed-era attempts (all before the MoE root cause was found; kept for the HARD CONSTRAINT):
1. `with_sharding_constraint(x,P())` in seed (`attention_init_state_from_prefill`) → Core-halt.
2. Runtime-replicate prefill input_ids ("Option A") → forward reshards x back, NaN + halt.
3. replicate `wkv` / 4. zero-pad seed token axis / 5. pin-output `wsc(kv,P(ATTN_DATA))` → NaN.
6. `_linear` clamp (zeros non-finite/`|.|>=1e8`) → finite but still collapses. **KEPT** (fixes
   only the NaN overlay).
7. pad + `_replicate(x)` (pad-then-gather) → Core-halts.
8. output `optimization_barrier` (`1f212036`) → XLA elides it, collapses.
9. (SESSION 3) force routed MoE sum replicated → NO-OP (sum already reduced; see above).

**HARD CONSTRAINT (proven ~8x):** any `with_sharding_constraint(ACTIVATION, P())` that
GATHERS a size-1 decode token axis (ATTN_DATA→P()) Core-halts (SLICE_FAILURE_SW_INJECT_ERROR).
A wsc on a POST-reduction `[N,dim]` quantity is safe. Don't gather the activation to replicated.

## Recovery / loop

* **Launch-id halt / SLICE_FAILURE before startup = CODE DESYNC** → `full_slice_v4_sync.sh` +
  clear `~/.cache/vllm/xla_cache/*` on all 8 hosts; do NOT reboot. (Only reboot the 7 WORKERS
  — never the head — if a genuine wedge persists after sync: wait SSH → remount GCS each
  `cd ~/claude-deepseek-v4 && set -a && source .env && set +a && ./scripts/mount_gcs.sh` →
  `full_slice_v4_ray_restart.sh`.) Clean engine (no halt) → just `full_slice_v4_reset.sh`.
* Infra `node` contention is SOLVED (`DenyUsers mark` on all 8 hosts; 2 guardians run). Don't refight.
* **Loop:** `scripts/s1_session_loop.sh` (stop: `touch /tmp/s1_loop_stop`). Per-session prompt
  `scripts/s1_loop_prompt.txt`.

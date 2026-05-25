# S1 handoff — decode COLLAPSE fixed; open-ended LOOPING is a NEW open question

Goal: coherent, deterministic decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash` on the
v6e-32 slice. Ops in `CLAUDE.md`; this is live state. **Read the nuance — earlier "S1
CLOSED" was an overclaim.**

## STATE (2026-05-25, SESSION 7)

**The ORIGINAL S1 decode collapse IS fixed** (precise bug: "first token correct, then a
token-2 degenerate/numeric attractor, every prompt"). Verified on a fresh diagnostic-free
engine:
* Structured (Fibonacci) greedy decode → `21,34,55,89,144,233,377,610` correct; 3×
  byte-identical (deterministic); tracks prefill-everything (term-7 exact, term-8 benign
  0.24-logprob near-tie).
* Short factual CHAT (native `/v1/chat/completions`, WITH a system prompt) → correct +
  coherent + proper EOS: "The capital of France is Paris", apples "3−1=2", sky-is-blue.

**NEW OPEN ISSUE (discovered this session) — OPEN-ENDED generation LOOPS.** "Teach me about
topology" gives a coherent opening then phrase-repetition:
* temp=0.6 +sys: "Topology is a branch of mathematics… It is a kind of geometry." ×24.
* temp=0 +sys: "…I will teach you about topology. I will teach you about" (loops).
* `vllm chat` (NO system prompt + its higher default temp) → outright garbage ("澳大利亚").
* `repetition_penalty=1.3` did NOT fix it (looped the system prompt instead).

**Is the looping the S1 DECODE bug or the MODEL? — CONFIRMED: the MODEL (decode is FAITHFUL).**
At the loop point, prefilling decode's own prefix and continuing at temp=0 gives
`' I will teach you about topology. I will teach you about topology. I will'` — i.e.
PREFILL-EVERYTHING (the model's true forward, which does NOT use the decode-state path) loops
IDENTICALLY to decode. So the open-ended looping is the MODEL (neural-text-degeneration on
flat-distribution open-ended greedy/low-temp), NOT the sharding decode bug. **S1 (the decode
collapse) stays FIXED; decode is faithful.** The looping is a model/decoding-params limitation,
OUT OF S1's scope: rep_pen=1.3 didn't help; try min_p / higher temp / longer max_tokens, or
accept that the Flash variant loops on open-ended greedy. (Short + structured prompts decode
clean; this only hits flat open-ended generation.)

**REFINEMENT (S7) — temp=0 decode is RUN-TO-RUN NONDETERMINISTIC on FLAT/open-ended prompts
(NOT degradation, NOT the collapse).** Firing the SAME no-system topology prompt at temp=0
**14× on ONE engine** gave ~3 outputs CYCLING non-monotonically (variant A "…stretching and" /
variant B "…but not necessarily under" / variant C "I'm doing great! Let me teach you…" / back
to A). So it is NOT engine-lifetime degradation (cycles, doesn't worsen) and NOT the token-2
collapse — it's **run-to-run nondeterminism**: flat model distribution → near-tie argmaxes →
flipped by 32-way sharded-TPU FP-reduction-order variation across runs (OR a real
nondeterminism source — e.g. idle-rank / uninitialized HBM, cf. `CLAUDE.full.md` PHASES 7-9 —
NOT distinguished). **CONFIDENT prompts (Fibonacci, Paris) ARE deterministic (3× byte-identical)**
— so the earlier "decode is deterministic at temp=0" holds ONLY for sharp distributions, not
flat ones. This explains `澳大利亚` + the looping (unlucky draws, amplified by high temp / no
system prompt). A GOOD system prompt gives correct coherent answers (verified: "Topology is a
branch of mathematics that studies the properties of spaces under continuous transformations…
homeomorphisms, and continuity…"). **OPEN LEAD (the real remaining question):** is this flat-prompt
temp=0 nondeterminism *inherent* sharded-FP-reduction variation (mostly unavoidable on TPU) or a
*fixable* bug (idle-rank/uninitialized memory)? To probe: fire the same flat prompt at temp=0
with `max_num_seqs` filled (all ranks busy) vs idle, and/or check if a deterministic-reduction
XLA flag removes it. The S1 token-2 collapse remains FIXED regardless.

## NEXT ACTION
1. Re-run the decode-vs-prefill verdict above on the live engine (:18081). Read it.
2. If MODEL: try better decoding (min_p / higher temp / longer max_tokens — serve caps
   `max_total_tokens=256`, raise `--max-model-len`). Accept that Flash may just loop on
   open-ended greedy. Document; S1 (sharding decode) stays fixed.
3. If DECODE: reopen — localize with decode-vs-prefill; the flat-distribution / longer-gen
   regime is what Fibonacci didn't exercise.

## Model usage (verified this session)
* INSTRUCT/reasoning model (post-trained SFT+RL; has `<think>`/`reasoning_content`), NOT
  base. Chat works NATIVELY (NO `--chat-template`): serve already passes
  `--reasoning-parser deepseek_v4` and a custom `DeepseekV4Tokenizer` applies
  `encoding/encoding_dsv4.py`. **ALWAYS pass a system message** or short queries misbehave.
* Engine LIVE on :18081 (left up). `vllm chat --url http://localhost:18081/v1 --model
  deepseek-ai/DeepSeek-V4-Flash`.

## Real fixes (committed, live)
* metadata-replicate decode (`_v4_decode_replicate` in `tpu_runner._prepare_inputs_dp`) → determinism.
* SWA seed + compressor/indexer STATE threading traced `n_real` (`6245ea84`/`90bf85c3`) → kills the token-2 collapse.
* removed always-on diagnostics (`77c0c7be`) — they perturbed near-tie numerics.

## Durable lessons / recovery
* decode-vs-prefill-everything ON THE SLICE is the faithful test (two kernels agree to ~0.2
  logprob — judge by coherence + determinism + near-tie gaps, NOT byte-identity). CPU parity
  non-predictive.
* HARD CONSTRAINT (~8×): `with_sharding_constraint(ACTIVATION, P())` gathering a size-1
  decode token axis Core-halts; a wsc on a POST-reduction `[N,dim]` is safe.
* `different launch id`/`Core halted` before startup = CODE DESYNC → `full_slice_v4_sync.sh`
  + clear+verify-empty `~/.cache/vllm/xla_cache/*` on all 8 hosts (ssh `-i
  ~/.ssh/google_compute_engine enyouki@HOST`). Don't reboot. Reset: `full_slice_v4_reset.sh`.
  Keep node_guardian + meta_guardian alive. Loop currently STOPPED (`/tmp/s1_loop_stop`).

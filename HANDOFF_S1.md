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

**Is the looping the S1 DECODE bug or the MODEL? — CONFIRMATION PENDING (do this first).**
* Decode-vs-prefill-everything at temp=0 on the topology chat prompt: DECODE loops;
  PREFILL-EVERYTHING half was **in flight** (slow engine) when this was written — RE-RUN it
  (cheap on the live engine). Script pattern: `encode_messages([sys,user],"chat")` via
  `encoding/encoding_dsv4.py`, then decode N=28 temp=0 vs chained mt=1 temp=0; compare.
* **Strong hypothesis: it's the MODEL** (classic neural-text-degeneration on flat-
  distribution open-ended prompts), NOT the sharding decode bug — because (a) short/
  structured prompts are clean, (b) original S1 was an IMMEDIATE token-2 tight attractor;
  this is a coherent multi-sentence start then phrase-loop (qualitatively different), (c)
  V4-Flash is the small/fast variant.
* IF prefill-everything ALSO loops → confirmed MODEL: decode is faithful, S1-decode stays
  fixed, looping is a model/decoding-params limitation (out of S1's scope). IF decode ≠
  prefill → REOPEN S1: the bug survives on open-ended/longer gen that Fibonacci didn't hit.

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

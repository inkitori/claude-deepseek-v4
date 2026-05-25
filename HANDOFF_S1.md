# S1 handoff — fresh session, pick up here

Goal: make `vllm serve deepseek-ai/DeepSeek-V4-Flash` produce coherent, deterministic
decode on the v6e-32 slice (bug **S1**). Bypass perms; use the TPU; commit+push
checkpoints; never wait. Operational details are in `CLAUDE.md`; this is live state.

## ⇒⇒⇒ SESSION 5 (2026-05-25) — ROOT CAUSE FOUND + FIX IN VALIDATION

**S1 = the decode KV seed is built over the PADDED prefill activation (T=1024 ≫
win=128), so the SWA window (and compressor state) get seeded with PAD-token kv
instead of the real prompt's kv.** Decode attends over pad-token kv → directional
garbage → attractor. It's a **PADDING bug, not sharding** (reproduces on CPU with a
padded input; the exact-length CPU parity test missed it). The GPU/torch **reference
always windows over the real seqlen, never pads** — confirms the fix direction.

DECISIVE EVIDENCE (direction-sensitive `[seedfp]`/`[kvchk-pf]` fingerprints added
this session; same-tensor pre/post-scatter compare in one pass — NOT a confoundable
cross-token/cross-pass ratio like SESSION-3's debunked MoE):
* Smoke 123908Z (pre-fix): L0/L1 seed `kv` (pre-scatter) MATCHES prefill truth, but
  `swa` (post-scatter) = `[10.7,11.07,11.37]` (slowly varying = pad id-0 at rope
  positions). T=1024 printed; roll branch read `kv[896:1024]` = all pad.
* Smoke 131221Z (SWA fix only): `[seedfp]` L0/L1 `swa` FLIPPED to **MATCH** truth
  (`[-13.45,…]`,`[22.03,…]`). ⇒ SWA seed fix verified ON THE SLICE; dynamic gather
  did NOT halt (the take_along_axis over the sharded token axis is fine — the HARD
  CONSTRAINT is specifically about wsc(activation,P()) gathering a size-1 axis).

FIXES (committed `6245ea84` SWA, `90bf85c3` compressor; CPU-validated):
* Thread the **traced** real length `n_real = seq_lens[0]` (correct per-request; the
  old static `state_init_ids`/`L_real` slice is a TRACE-TIME branch baked at warmup
  ⇒ never fires) through run_with_decode_state → transformer_body_init_state_* →
  block_init_state_and_forward → attention_init_state_from_prefill →
  `_swa_kv_cache_from_prefill` (+ `_compressor_state_from_prefill`). `n_real=None`
  preserves exact-length static path (CPU/parity tests still pass).
* SWA: branchless `swa[i]=kv[i+((n_real-1-i)//win)*win]` masked `i<n_real`.
* Compressor/indexer state: cutoff/remainder from n_real via dynamic_slice + masking.
* CPU unit tests (PASS): `/tmp/s1_swa_fix_test.py`, `/tmp/s1_compressor_fix_test.py`
  (traced-over-padded == static-over-exact). `s1_cpu_repro both` OK (no regression).

BOTH-FIXES SMOKE 132821Z RESULT (verified on slice, READ the text):
* `[seedfp]` L0/L1 swa flipped DIVERGE→**MATCH** (seed fix engaged; no halt).
* Decode NO LONGER COLLAPSES: Fibonacci → "21, 34, 55, 89, 144, 233, 377, …" (7 CORRECT
  terms; was immediate garbage). logprobs sharp+correct: 21=0.89, 34=0.76, 55=0.99,
  89=0.87, 144=0.999 — model is confident/correct, NOT a flat-logit regime. Paris
  deterministic (3× byte-identical); chat endpoint gives clean "Paris".
  ⇒ **PRIMARY S1 collapse is FIXED.**

⇒⇒ KEY REFRAME (decode-vs-prefill-everything token compare, `/tmp/s1_prefill_vs_decode.py`):
For prose "Once upon a time, there was a" N=10, decode and PREFILL-EVERYTHING are
**BYTE-IDENTICAL** (' time, and the time was a time, and') ⇒ **decode is FAITHFUL** to
the no-decode-cache path on sharp tokens. So the GROSS GREEDY LOOPING is the **MODEL**
(both paths loop identically), NOT a decode bug. The original S1 (decode diverging from
prefill into its OWN attractor) is FIXED. BUT note: even prefill-everything loops on
open-ended prose — so the model's open-ended greedy output is degenerate via BOTH paths
(likely base-model + greedy + 256 ctx; the model IS correct on structured prompts:
Fibonacci 7 terms sharp, "Paris" correct, chat→clean "Paris"). OPEN: confirm model
quality vs a deeper forward bug via chat+sampling / the GPU reference.

⚠️ SEPARATE small RESIDUAL decode bug (only SOFT tokens, far into gen): decode drifts at
Fibonacci term 8 ("…377, 666") while FAITHFUL prefill ("…377, ") predicts **610** (argmax
0.43; 666=#2 at 0.14). On SHARP tokens the small error doesn't flip argmax (decode==prefill
at N=10), so it only bites soft tokens after ~5 compressions. NOT the cause of the looping.

IMPLICATION for "done": the success-gate's "coherent greedy long-gen" may be UNACHIEVABLE
because the MODEL loops at greedy on open-ended prompts. The right decode-correctness
criterion is **decode == prefill-everything** (faithfulness) — now true except the soft-
token residual. Fix the residual (comp_full/i_cache below), then decode==prefill always.

LEADING HYPOTHESIS for the residual (source-confirmed unfixed): the compressed CACHES
`comp_full` and `i_cache` (attention_init_state_from_prefill ~1167/1193) =
`compressor_prefill(x_padded)[:extra]` — built over the PADDED x, NOT n_real. So the
boundary window straddling n_real (e.g. window 7 = tokens [28,32) = 1 real + 3 pad) is
pad-polluted; decode reads more compressed slots as it generates → small accumulating
error → flips the term-8 soft argmax. (I fixed the STATE c_kv/c_sc/i_kv/i_sc with
n_real, but NOT these caches.) Decode-step math itself is likely fine (decode runs
replicated == CPU; matched faithful on the first 14 tokens).

NEXT ACTION:
1. Fix `comp_full` + `i_cache` to respect n_real: mask/zero compressed windows whose
   source tokens are ≥ n_real (only keep fully-real windows; decode overwrites the rest
   via compressor_decode_step as it generates). CPU-validate like the state fix
   (traced-over-padded vs static-over-exact) then smoke.
2. RE-RUN the drift test: prefill("…377, ")→610 AND decode(Fibonacci)→ must now also
   give 610 (decode == faithful). Then the GATE twice.
3. If drift persists after the cache fix → compare decode vs prefill-everything
   token-by-token (`/tmp/s1_prefill_vs_decode.py`) to find the first divergent step,
   and check the GPU reference `work/vllm/vllm/model_executor/layers/deepseek_v4_attention.py`.

NOTES: decode is SLOW (~18s/tok) — that's the ALWAYS-ON diagnostics' per-layer
jax.debug.print host callbacks, NOT the fix; perf returns when diagnostics are removed
at close. Engine still degrades after several requests (empty completions) — fire
critical probes first. The SWA-only-smoke reference-'' was engine flakiness (132821Z
prefill argmax is fine).

REFERENCE IMPLEMENTATIONS (cross-check resources — use these to avoid false root causes):
* torch ref: `work/tpu-inference/tests/models/jax/_deepseek_v4_reference/model.py`
  (windows SWA over exact seqlen; compressor cutoff=seqlen%ratio).
* **vLLM GPU impl: `work/vllm/vllm/model_executor/layers/deepseek_v4_attention.py`**
  (`DeepseekV4SWACache`) — a second reference for decode/seed behavior.

DIAGNOSTICS still in code (REMOVE at S1 close): `_v4_fp`/`_v4_dir`, `[seedfp]`,
`[kvchk-pf]`, `[fwdSd]`, `[decSd]`. Helpers `/tmp/s1_seed_analyze.py`,
`/tmp/s1_swa_fix_test.py`, `/tmp/s1_compressor_fix_test.py`.

## Durable lessons (kept; the rest of SESSION 3/4 narrative was the pad confound)

* **HARD CONSTRAINT (proven ~8x):** `with_sharding_constraint(ACTIVATION, P())` that
  GATHERS a size-1 decode token axis Core-halts. A wsc on a POST-reduction `[N,dim]`
  quantity is safe. (NB: a plain `take_along_axis`/`dynamic_slice` gather over the
  sharded token axis — as in the SESSION-5 seed fix — does NOT halt; it's the
  wsc-to-replicated of the degenerate axis that does.)
* Collapse is DETERMINISTIC at temp=0 (metadata-replicate decode fix in
  `tpu_runner._prepare_inputs_dp`). token1 (prefill argmax) is correct; decode collapses.
* Diagnostics must be ALWAYS-ON (no env gate) — env-gated module reads race across ray
  workers → launch-id halt.

## Recovery / loop

* **Launch-id halt / SLICE_FAILURE before startup = CODE DESYNC** → `full_slice_v4_sync.sh`
  + clear `~/.cache/vllm/xla_cache/*` on all 8 hosts; do NOT reboot. Clean engine → just
  `full_slice_v4_reset.sh`. Escalate: `full_slice_v4_ray_restart.sh`.
* Infra `node` contention SOLVED (`DenyUsers mark`; 2 guardians). Keep node_guardian +
  meta_guardian alive before TPU work.
* **Loop:** `scripts/s1_session_loop.sh` (stop: `touch /tmp/s1_loop_stop`).

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

IN FLIGHT: **both-fixes smoke `logs/full-slice-v4-smoke-20260525T132821Z.log`** (:18081,
pid in log) compiling now. NEXT ACTION when up:
1. Fire a Fibonacci decode FIRST and READ THE TEXT (critical probe; engine degrades
   after a few requests): `/tmp/s1_seed_analyze.py LOG "The list of Fibonacci numbers
   is: 1, 1, 2, 3, 5, 8, 13, " "21"` → expect `[seedfp]` L0/L1 MATCH (re-confirm) AND
   a COHERENT continuation ("21, 34, 55,…"), not the SWA-only-smoke garbage ('#').
2. If coherent → run the GATE TWICE: `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh`
   (visible_words≥10, max_word_run<5) + 3 byte-identical Paris probes + survives 5 reqs.
3. If STILL collapses → seed is now correct (proven) so the remaining bug is elsewhere
   (decode-step math? compressed CACHE comp_full boundary window? logits?). Add a
   direction probe at the first divergent layer; compare vs the GPU reference.

WATCH / unresolved skepticism (don't declare S1 closed until the gate passes TWICE on
a fresh engine, reading actual text):
* SWA-only smoke 131221Z REFERENCE (prefill mt=1) returned '' (was ',' pre-fix). My
  seed fix must NOT affect the prefill argmax (CPU repro proves argmax unchanged,
  bad=0/12), so this is almost certainly ENGINE FLAKINESS (it returned empty
  completions after a few reqs). RE-VERIFY on 132821Z: the reference should give a
  coherent token; if it gives '' again, investigate a forward-path interaction.
* Engine returns empty completions / degrades after ~2-3 requests (known: NaN clamp
  keeps it alive but degenerate). With the seed fixed there may be fewer NaNs — watch.

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

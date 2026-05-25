# S1 handoff — fresh session, pick up here

Goal: make `vllm serve deepseek-ai/DeepSeek-V4-Flash` produce coherent, deterministic
decode on the v6e-32 slice (bug **S1**). Bypass perms; use the TPU; commit+push
checkpoints; never wait. Ops details are in `CLAUDE.md`; this is live state.

## STATE (2026-05-25, SESSION 6)

**PRIMARY collapse is FIXED** (commits `6245ea84` SWA seed + `90bf85c3` compressor/indexer
STATE — both thread the traced `n_real`). Decode no longer collapses: greedy Fibonacci
("…1,1,2,3,5,8,13, ") decodes CORRECTLY through **term 7** → `21, 34, 55, 89, 144, 233,
377` (early tokens sharp, conf 0.9-0.99). A **residual drift remains at term 8**: decode
diverges there from the faithful path.

**comp_full/i_cache zeroing was TRIED → REVERTED this session** (`ab948e3d`, reverted by
`b026d9ff`). Hypothesis: the two COMPRESSED CACHES decode reads — `comp_full`
(=`kv_cache[:, win:]`) and `i_cache` (=`indexer_kv_cache`) in
`attention_init_state_from_prefill` — are built via `compressor_prefill(x_PADDED)` and so
hold pad-token kv in slots `[n_safe:]` (`n_safe = n_real//ratio`); fix = zero those slots.
It was CPU-validated (made padded-traced == exact-static == torch-ref, which leaves
unwritten slots ZERO) but **REGRESSED decode on the slice**:

| | term7 (after "…144, 233, ") |
|---|---|
| faithful prefill-everything | **377** (conf 0.76) |
| PRE-fix decode | 377 (correct; drift only at term 8) |
| POST-zeroing decode | **144** (LOOPS 144,233,144…; conf 0.45) |

⇒ **KEY FINDING: decode READS comp_full/i_cache slot `n_safe` BEFORE the decode_step
overwrites it.** A ZERO there corrupts attention MORE than the pad value did → the
"overwrite-before-read is safe" argument is FALSE on the sharded slice. **CPU parity did
NOT predict slice behavior** ("CPU can't reproduce S1"). For any decode fix, trust only
**decode-vs-prefill-everything ON THE SLICE**, never CPU parity, never "looks coherent".

## NEXT ACTION — the residual at term 8 (do NOT re-try plain zeroing)

The residual is NOT pad-zeroing. The boundary slot `n_safe` (window straddling `n_real`)
IS read by decode before being overwritten, so it needs its CORRECT compressed value at
seed time — not zero, not pad. Two leads (pick one, prove on the slice):
1. **Seed slot `n_safe` from the n_real-aware compressor STATE.** `c_kv/c_sc` (and indexer
   `i_kv/i_sc`) already hold the in-progress partial window `[cutoff, cutoff+remainder)`
   correctly (the SESSION-5 state fix). Compute that boundary window's compressed kv and
   write it into `comp_full[:, n_safe]` / `i_cache[:, n_safe]` at seed time. Leave slots
   `> n_safe` as whatever decode reads only AFTER overwriting (check #2).
2. **Check the JAX decode-step read/write ORDER vs torch ref**
   (`tests/models/jax/_deepseek_v4_reference/model.py`): torch writes slot
   `start_pos//ratio` THEN reads `[0,(start_pos+1)//ratio)`. If the JAX decode-step reads
   the boundary slot before writing it, fix the order so the seed value never matters.
VERIFY: on a fresh smoke, decode-vs-prefill-everything at N>20 — decode term7 must stay
**377** AND term8 must become **610** (= faithful). Localize with `/tmp/s1_prefill_vs_decode.py`.

## SLICE-PROBE OPS (learned this session — saves you hours)

* The engine is **not** fragile from "degradation" as the old handoff implied. The
  apparent "wedge / empty completions" was the **first-inference-per-input-SHAPE JIT
  compile (~5 min)** exceeding short curl timeouts. **Use `--max-time`/timeout ≥ 580s for
  the FIRST request of each new prompt shape**; same-shape requests after that are fast.
  The always-on diagnostics slow decode to ~9-18s/tok (a 64-tok gen ≈ exceeds 900s — so
  the scripted `LONG_GEN` gate can't complete until diagnostics are removed).
* Faithful reference = **prefill-everything** (chained `max_tokens=1`; does NOT use the
  decode seed). DECODE BUG ⟺ decode ≠ prefill-everything. Helpers:
  `/tmp/s1_prefill_vs_decode.py "PROMPT" N` (per-req timeout already 900),
  `/tmp/s1_faithful_terms.py` (single-term prefills). CPU repro of the (reverted) seed
  diff: `/tmp/s1_compfull_fix_test.py`.
* **No live engine is left running** (reset at session end). Re-smoke per CLAUDE.md (warm
  xla_cache ⇒ ~6 min): edit → sync → reset → smoke → wait "Application startup complete".
  Don't clear xla_cache unless you changed code (clearing forces a cold compile).

## ⚠️ The scripted success gate has a prompt problem
`scripts/full_slice_v4_smoke_check.sh` `LONG_GEN` uses an OPEN-ENDED prompt ("Tell me a
short story about a robot exploring Mars:") at greedy temp=0 — exactly the red herring
CLAUDE.md warns about (a base model loops at greedy regardless of decode correctness).
Plus at ~9-18s/tok a 64-tok decode exceeds the 900s curl cap. Once decode is correct,
either (a) remove diagnostics so decode is fast AND swap the gate prompt to a
discriminating one (strictly-increasing Fibonacci), or (b) define done as
decode==prefill-everything at N>20 on a discriminating prompt + 3× Paris determinism.

## Durable lessons (kept)
* **HARD CONSTRAINT (proven ~8x):** `with_sharding_constraint(ACTIVATION, P())` that
  GATHERS a size-1 decode token axis Core-halts. A wsc on a POST-reduction `[N,dim]`
  quantity is safe. A plain `take_along_axis`/`dynamic_slice` gather over the sharded
  token axis (as in the SESSION-5 seed fix) does NOT halt.
* Collapse is DETERMINISTIC at temp=0 (metadata-replicate decode fix in
  `tpu_runner._prepare_inputs_dp`). token1 (prefill argmax) is correct.
* Model is SOUND: chat (instruct) + sampling give coherent output; gross greedy looping on
  open-ended prompts is the MODEL, not a decode bug.
* Diagnostics must be ALWAYS-ON (no env gate) — env-gated module reads race across ray
  workers → launch-id halt. (REMOVE all S1 diagnostics at close: `_v4_fp`/`_v4_dir`,
  `[seedfp]`, `[kvchk-pf]`, `[fwd*]`/`[dec*]`/`[pf4]`/`[moeRS]`.)

## Recovery / loop
* **`different launch id` / `Core halted` / `SLICE_FAILURE` before startup = CODE DESYNC
  at the COMPILED level** → `full_slice_v4_sync.sh` + clear `~/.cache/vllm/xla_cache/*` on
  all 8 hosts AND **verify each cache dir is actually empty** (a host that silently kept
  stale cache caused a halt this session; re-clearing+verifying fixed it). Do NOT reboot.
  Clean engine → just `full_slice_v4_reset.sh`. Escalate: `full_slice_v4_ray_restart.sh`.
* Keep node_guardian + meta_guardian alive before TPU work.
* **Loop:** `scripts/s1_session_loop.sh` (stop: `touch /tmp/s1_loop_stop`).

## References (cross-check to avoid false root causes)
* torch ref: `work/tpu-inference/tests/models/jax/_deepseek_v4_reference/model.py`
  (compressed/indexer caches over real seqlen; cutoff=seqlen%ratio; unwritten slots=0).
* vLLM GPU: `work/vllm/vllm/model_executor/layers/deepseek_v4_attention.py`
  (`DeepseekV4SWACache`; paged compressed cache written at real slot_mapping only).

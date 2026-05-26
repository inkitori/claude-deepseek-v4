# S1 handoff — gmm_v2 routed path RUNS on TPU; needs 2-engine determinism + quality validation

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL / how to judge a fix — READ THIS (see memory [[s1-symptom-nondeterminism-not-collapse]])
The bug is **NOT a collapse**. The model emits coherent-LOOKING decode (increasing Fibonacci-ish) **even WITH
the bug** — coherence is NOT proof of a fix. Two real symptoms: (1) **cross-process non-determinism** (md5
differs across 2 fresh engines; deterministic WITHIN a process), (2) **slight quality degradation** (FIB drifts
from correct, e.g. 570≠610). DONE = BOTH gone: **byte-identical md5 across 2 fresh engines AND correct Fibonacci
vs the prefill-everything reference.** Model is INSTRUCT (coherence via `/v1/chat/completions` system+user).

## STATE (2026-05-26 S19) — THE FIX IS IN AND RUNNING (not yet validated)
Replaced the bespoke dense per-rank expert EINSUM in `deepseek_v4_moe.py::moe_forward` (the S18-isolated
uninit-HBM matmul-scratch reader) with the production **gmm_v2 grouped-matmul + zero_initialize=True**
(DMA-zeroes unvisited output rows = the determinism lever). Surgical: only the shard_map branch changed;
loader + dense (CPU/decode) path untouched; weights transposed to gmm rhs layout LOCALLY inside the shard_map.
Dispatch mirrors fused_moe_gmm.py (sort (token,slot) by global expert → group_sizes[E]+group_offset=r*EP →
gmm gate+up (fp32) → manual clamped-swiglu → gmm down (bf16) → revert + weight-combine + psum), plain-JAX gather.

* **Committed 21063d80, synced + md5-verified (1814bc99) on all 8 hosts.** Git HEAD == running code.
* CPU-validated BEFORE smoke: dispatch math == dense einsum (rel 1.3e-7, `/tmp/s1_gmm_dispatch_test.py`);
  `s1_cpu_repro both` PASSES (no dense-path regression). gmm_v2 group_offset semantics + fp32-input support
  confirmed by reading the kernel.
* **TPU smoke: gmm path RUNS, no crash.** Engine A19 (started 13:48, now likely reset) FIB md5 **d99ee354**,
  text `21, 34, 55, 89, 144, 233, 377, 570, 987, 1584, 2581, 2584, 258`, leading_correct **7/12**,
  within-process deterministic (2/2 same md5 — EXPECTED, NOT the test).
* (S19 also burned a smoke: the first attempt crashed on a `group_offset` POSITIONAL-arg bug → IndexError →
  took down a raylet → needed `full_slice_v4_ray_restart.sh` (recovered 32/32). Fixed by passing `group_offset=`.)

## NEXT ACTION — VALIDATE (the fix is unproven; do NOT trust coherent output)
1. **DETERMINISM (decisive):** reset + smoke a FRESH engine, warmup (`/tmp/s1_warmup.py`, absorbs ~335s recompile),
   then `/tmp/s1_fib_clean.py B1` (NO logprobs, 420s timeout — `s1_fib2.py`'s logprobs=5 wedged/slowed it).
   Compare its md5 to **d99ee354**. MATCH ⇒ cross-process determinism FIXED (the core S1 bug). DIFFER ⇒ gmm fix
   insufficient, diagnose. (For rigor, can do 2 fresh engines rather than trust the recorded anchor.)
2. **QUALITY:** get the prefill-everything reference for the FIB prompt (`/tmp/s1_prefill_gen.py` — chained
   max_tokens=1; SLOW at this engine speed, do only ~10 tokens to see if it predicts 610 correctly vs decode's
   570). If reference also gives ~7/12, then 7/12 is the model baseline (no degradation). If reference is more
   correct, decode quality is still degraded ⇒ fix incomplete.
3. If determinism holds AND quality matches reference: **REMOVE all [ckS]/[ckL]/[ckR] diagnostics** (also a DONE
   requirement; may fix the slowness too) → final 2-engine smoke (md5-identical + coherent chat, READ TEXT,
   survive 5 reqs) → record DONE in HANDOFF+CLAUDE → `touch /tmp/s1_loop_stop`.

⚠ **SLOWNESS** (~2-3 tok/s, ~2-3 min/request): likely the fp32 g1 gmm (fp32 MXU is ~3-6x) + the `[ckL]`
jax.debug.print inside the shard_map (per-shard sync). Removing diagnostics (step 3) and/or casting g1 to bf16
(`x_sorted`/`W13_l`→bf16, keep preferred_element_type=fp32) should help if perf matters — but validate determinism FIRST.

## Tools / ops
* ONE engine at a time. Probes (localhost:18081): `/tmp/s1_warmup.py` (FIB, run FIRST — absorbs recompile),
  `/tmp/s1_fib_clean.py <label>` (FIB×2 md5, no logprobs, 420s timeout — USE THIS not s1_fib2.py),
  `/tmp/s1_chat.py <label> [prompt]` (coherence). Engine is slow → use long timeouts.
* Slice protocol in CLAUDE.md: edit → `full_slice_v4_sync.sh` (md5-verify) → keep xla_cache (warm⇒~6min start)
  → `full_slice_v4_reset.sh` → `full_slice_v4_smoke.sh`. Guardians up (node 497956, meta 4039835).
* If a hard EngineCore crash takes down a raylet (`<32 TPU` after reset): `full_slice_v4_ray_restart.sh`.

## DEAD (do not retry)
* The S18 einsum-isolation hunt is DONE — superseded by the gmm port (this fix). Don't re-isolate.
* psum / all_gather / x_full-buffer / weights / ALL pad-row masking / optimization_barrier — all exonerated (S17-18).
* Decode/seed hunt (clean). `wsc(act,P())` gathering size-1/idle axis (Core-halts). prefill-replicate (NaN).

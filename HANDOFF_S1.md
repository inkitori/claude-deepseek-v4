# S1 handoff — ✅ CLOSED (fix verified + diagnostics removed, ×2 fresh engines)

S1 is DONE. `vllm serve deepseek-ai/DeepSeek-V4-Flash` produces coherent, RELIABLE
(cross-process deterministic) decode on the v6e-32 slice. The self-perpetuating S1
loop is STOPPED (`/tmp/s1_loop_stop`). Durable slice ops live in `CLAUDE.md`.

## ✅ S1 CLOSED (2026-05-27 S29)

ROOT CAUSE (S26): decode cross-process non-determinism = the routed **W1/W3
stacked-weight LOAD**. The device-side consolidation reshard (jnp.stack of the
axis-1 leaves → device_put to the expert-sharded layout) read UNINITIALISED HBM,
baking per-process garbage into W1/W3 → engines diverged at temp=0. W2 (axis-0
leaf) was byte-clean.

FIX (commit 5a3ed435, S28): eliminate the device-side reshard for w1/w3 WITHOUT
changing leaf sharding. `place_spec_as_jax_sharded(return_host_np=True)` hands back
the FULL per-expert host numpy; `_maybe_consolidate` for w1/w3 does `np.stack` of the
256 host numpys + `jax.make_array_from_callback` straight into the expert-sharded
layout (host→device, NO device reshard, NO uninit read). w2 stays on the clean
device path. (S27 v1 "force axis-0 leaves" FAULTED layer-0 consolidation → reverted.)

HYGIENE (commit 2d3e0c45, S29): removed all ALWAYS-ON S1 debug instrumentation
(165 deletions across 3 files):
* moe.py: `_ckR` helper+calls; all 13 `jax.debug.print` ([ckS]×3, the [ckSPLIT]
  `_ck_all` shard_map COLLECTIVE on layer-0 decode + prints, [ckEY], and the
  sharded-prefill [ckR0]/[ckR1b]/[ckR2]/[ckR3]/[ckL]).
* deepseek_v4.py: [ckD] decode-logit print; 4 `_v4_checksum` calls + dropped from import.
* attention.py: 4 `_v4_checksum` seed calls + the `_v4_checksum` def.
* KEPT (intentional): env-gated `_v4_nan_tripwire` (no-op unless V4_DECODE_NAN_TRIPWIRE=1;
  zero runtime cost, defensive) + the compute_logits `nan_to_num` clamp (functional).

GATE — re-verified on the diagnostics-removed build (2d3e0c45):
* CPU repro `s1_cpu_repro_v4flash.py both` => "OK: both eager and jit match", bad=0/12.
* 2 independent diff audits => SAFE (deletions-only; both MoE paths + nan_tripwire +
  clamp intact; no dangling refs).
* Engine #1: FIB md5=**5bf42256** ×3 fires (within-process det); correct Fibonacci
  (21, 34, 55, 89, 144); smoke_check rc=0 (Paris deterministic ×2, LONG_GEN
  visible_words=45 max_word_run=2 ends_clean=1; survived 7+ requests).
* Engine #2 (fresh process): FIB md5=**5bf42256** — BYTE-IDENTICAL to engine #1.
  [×2 CROSS-PROCESS DETERMINISM GATE MET]

## Nothing left to do for S1.
The loop is stopped. If decode work resumes on this slice: durable serving protocol +
pitfalls are in `CLAUDE.md`; S1 narrative history is in the commit log + `CLAUDE.full.md`.
(Decode throughput is ~0.31 tps — a separate PERF item, NOT an S1 correctness concern.)

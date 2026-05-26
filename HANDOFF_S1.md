# S1 handoff — S26 VERDICT STANDS (W1/W3 weight LOAD); S27 fix v1 REFUTED (faults the load)

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL / how to judge (see [[s1-symptom-nondeterminism-not-collapse]])
Bug is NOT collapse. Symptoms: (1) cross-process non-determinism (decode output differs across fresh engines;
deterministic WITHIN a process), (2) quality drift. DONE = byte-identical FIB md5 across 2 fresh engines AND
correct Fibonacci. Model INSTRUCT. Correct FIB after "13, " = 21,34,55,89,144,233,377,610,987,1597...

## STATE (2026-05-26 S27) — root cause STANDS, but the obvious fix FAULTS the load
ROOT CAUSE (S26, unrefuted): decode corruptor = the routed **W1 & W3 stacked-weight LOAD**. [ckSPLIT] (moe.py
~262) checksummed routed W1/W2/W3 on 2 fresh engines: **W1 & W3 diverge ×2** (gsum ~20%, absmax value-flips),
**W2 byte-IDENTICAL**. Mechanism (3 agents hardened it): `pick_partition_spec` (loader:508) shards w1/w3
[inter=2048,dim=4096] on axis-1 (largest dim) but w2 [4096,2048] on axis-0; at consolidation
`device_put(jnp.stack(leaves), P('attn_dp',None,None))` (deepseek_v4.py:1535) the w1/w3 axis-1→expert reshard
reads uninit HBM, w2's axis-0→expert reshard is byte-clean.

S27 FIX v1 = **REFUTED** (commit 446950f3, now reverted): added `prefer_axis0` to `pick_partition_spec` +
`place_spec_as_jax_sharded`, set it for expert leaves in `_do_place_spec` → force w1/w3 leaves to shard axis-0
like w2 (CPU unit-test PASSED: w1/w3→P('attn_dp',None), w2 idempotent, non-experts untouched). On TPU it
**deterministically faults layer-0 expert consolidation**: 3 fresh smokes ALL died at exactly **600 tensors**
(`layers.0.ffn.experts...`) with worker SIGKILL → `SLICE_FAILURE_SW_INJECT_ERROR`/`ActorDiedError`, **NO Python
traceback** (= device/libtpu fault, not a code exception). DECISIVE control: reverted (original) code placed
**6800 tensors** in ~90s (past the death zone) ⇒ **slice infra is HEALTHY; v1 is the culprit.**

WHY v1 faults is UNKNOWN: w2 already shards axis-0 and consolidates fine via the SAME axis→expert reshard;
only the SHAPE differs (w1/w3 axis-0 = 64 rows/shard, 4096 cols; w2 = 128 rows/shard, 2048 cols). The axis-0
leaf change for w1/w3 (now slice-aware read + axis-1→axis-0 stack reshard) hits a libtpu fault for this shape.
Dequant scale-slicing was audited SAFE (axis-agnostic) — so the fault is in the device-side reshard/stack, not
the host dequant. Logs: smoke 215410Z (v1 fail), 220051Z (revert pass-death-zone).

## NEXT ACTION — v2: eliminate the device-side consolidation reshard (do NOT repeat v1)
Repo is back on WORKING baseline (S26 code, [ckSPLIT] intact, synced to 8 hosts). Slice idle & healthy. Try, in
order, each gated by the S26 [ckSPLIT] 2-engine test:
1. **v2a (recommended, most robust): host-gather consolidation.** In `_maybe_consolidate` (deepseek_v4.py:1535),
   replace `jax.device_put(jnp.stack(weights), e_spec)` with: gather each leaf to host (`np.asarray`/
   `jax.device_get`), `np.stack` on host, then `jax.make_array_from_callback(shape, e_spec_sharding, cb)` (cb
   slices the expert-axis rows for each device). NO device-side reshard ⇒ no uninit read AND no fault. Hosts have
   708GB; transient ~4GiB/group safe. RISK: device→host→device per group (~180 groups) may be slow — measure.
2. **v2b: investigate the fault first.** Re-apply v1 on ONE smoke and read the dead-worker libtpu/HLO logs at the
   600-tensor death (`/tmp/ray-vllm/session_latest/logs/worker-*.err` + gcs/raylet logs) to see WHY the axis-0
   reshard faults — that may reveal a smaller fix (e.g. axis-0 only at consolidation, not at leaf placement).
3. **v2c: place directly into the stacked [256,X,Y] e_spec-sharded buffer** (no per-leaf stack+reshard at all) —
   bigger loader change; consider only if v2a is too slow.
GATE (validate, TWICE on fresh engines): `python3 /tmp/s1_collect.sh <log> /tmp/s1_engN_s28.txt` ×2 engines,
then `bash /tmp/s1_gate.sh <eng1> <eng2>` ⇒ **[ckSPLIT] W1 & W3 gsum+absmax byte-IDENTICAL ×2** + FIB md5
identical ×2 + correct Fibonacci + CLAUDE.md formal gate ×2 → `touch /tmp/s1_loop_stop`.

## Ops (jax.debug.print LOSSY — SCALAR prints; re-fire + sort -u). Helpers in /tmp: s1_probe2.py, s1_collect.sh, s1_gate.sh, s1_cke.sh
* ONE engine. Cache warm ⇒ ~350s ready; cold (after .py change + cache clear) ⇒ 10-30 min. FIRST decode req
  recompiles ~325s. Probe: `python3 /tmp/s1_probe2.py 2` (max_tokens=2 = 1 decode step, output '21,').
* edit→`full_slice_v4_sync.sh`→verify md5 8/8 as **enyouki@**→CLEAR xla_cache 8 hosts if .py changed→`reset.sh`→
  `smoke.sh`→wait `Application startup complete`. IPs: head .192; workers .194 .202 .204 .193 .198 .195 .200.
* MONITORING a smoke: watch logs/<log> for `Application startup complete` (ready) vs
  `SLICE_FAILURE|ActorDiedError|Engine core initialization failed` (fail). To tell a CODE fault from infra: the
  DRIVER (EngineCore) log only shows the symptom — read the WORKER logs `/tmp/ray-vllm/session_latest/logs/
  worker-*.err` for `[deepseek_v4] placed N tensors` progress + any real traceback. A worker that dies with NO
  Python traceback (SIGKILL/EOF) = device/libtpu fault or infra; a worker that gets PAST ~768 tensors (layer-0
  experts) means the expert-load path works. Filter by mtime to find THIS smoke's worker logs.
* INFRA was healthy this session (8/8 nodes, 32 TPU, 708GB RAM/host, GCS up since 13:41 = same session S26
  used). meta_guardian spews `WrongClusterID` (stale connection, BENIGN — fresh ray.init sees 32 TPU fine);
  ignore it. node_guardian UP (pid 497956). There were also **3 orphaned `claude` loop sessions** (pids
  2605015/3031059/3284020) — flock serializes the slice so harmless, but they burn budget; consider whether the
  loop wrapper is double-spawning.
* PITFALL THIS SESSION: a cold smoke can hit `SLICE_FAILURE` during SPMD partition at ~60s (smoke 1) — that one
  WAS a transient (retry), distinct from v1's deterministic 600-tensor load fault. Don't conflate the two.

## DEAD (do not retry): all-reduce (S25); input/prefill cascade (S26 byte-identical ×2); SUBLANE-PAD/gmm-kernel
for DECODE; bf16↔fp32 gmm dtype; zero_initialize; partial_out_ref; gmm_v2:518 OOB; GATE/ROUTING sharding;
wsc(act,P()) token axis; the dense einsum MATMUL itself (W2 clean ⇒ matmul faithful). **NEW: S27 v1 — forcing
expert leaves to shard axis-0 via `pick_partition_spec(prefer_axis0)` — FAULTS layer-0 consolidation (libtpu,
deterministic at 600 tensors). The fix must NOT change w1/w3 LEAF sharding; fix the consolidation reshard itself.**

# S1 handoff — VERDICT: decode corruptor = the routed W1/W3 weight LOAD (consolidation reshard bakes in uninit HBM)

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL / how to judge (see [[s1-symptom-nondeterminism-not-collapse]])
Bug is NOT collapse. Symptoms: (1) cross-process non-determinism (decode output differs across fresh engines;
deterministic WITHIN a process), (2) quality drift. DONE = byte-identical FIB md5 across 2 fresh engines AND
correct Fibonacci. Model INSTRUCT. Correct FIB after "13, " = 21,34,55,89,144,233,377,610,987,1597...

## STATE (2026-05-26 S26) — VERDICT: the routed W1 & W3 stacked weights LOAD non-deterministically
S26 added [ckSPLIT] (moe.py ~250): ONE read-only shard_map prints per-rank-LOCAL [gsum,sqsum,absmax] (gather+
local-sum, reorder-immune, NO all-reduce) of routed weights W1/W2/W3 + every dense-einsum stage. Ran on 2 FRESH
engines, decode step (FIB max_tokens=2, both md5 5bf42256, correct '21'):
| quantity (decode L0) | eng1 gsum | eng2 gsum | eng1 absmax | eng2 absmax | verdict |
|----------------------|-----------|-----------|-------------|-------------|---------|
| W1 (w1_stacked)      | 4192.49   | **5159.84** | 0.250     | **0.1875**  | **DIFFERS** |
| W2 (w2_stacked)      | 143.6895  | 143.6895  | 0.250       | 0.250       | IDENTICAL |
| W3 (w3_stacked)      | 5260.76   | **4293.41** | 0.1875    | **0.250**   | **DIFFERS** |
| gate=x@W1, up=x@W3, h, out | (all DIFFER, downstream of W1/W3) | | | | DIFFERS |
sqsum barely moves (5th sig fig: a few garbage bf16 elems) but gsum ~20% + absmax value-FLIPS ⇒ **W1 & W3 carry
per-process uninit/garbage elements; W2 is byte-identical.** [ckSPLIT] reads the CONSOLIDATED E-sharded weights
with a MATCHING in_spec (P('attn_dp',None,None)) ⇒ NO reshard in the probe ⇒ the garbage is BAKED INTO
w1_stacked/w3_stacked at load, not a probe artifact. Refines (does not contradict) S25: the local einsum diverges
*because its weight inputs W1/W3 diverge*. S25 never measured the weights (only deduced clean). Data:
/tmp/s1_eng1_s26.txt, /tmp/s1_eng2_s26.txt. Logs 204316Z (eng1) 210710Z (eng2).

## MECHANISM (agent-traced, strong) — pick_partition_spec largest-dim fork
`deepseek_v4_loader.py:508-519 pick_partition_spec` shards each weight on its LARGEST axis-divisible dim:
per-expert w1/w3 [inter=2048, dim=4096] → axis 1 (4096) sharded; w2 [dim=4096, inter=2048] → axis 0 (4096)
sharded. So at consolidation `deepseek_v4.py:1535 jax.device_put(jnp.stack(weights), e_spec=P('attn_dp',None,None))`,
w1/w3 leaves (inner-axis-sharded → post-stack axis-2-sharded) need a DIFFERENT reshard to E-major than w2
(axis-0 leaf). The w1/w3 reshard path reads uninit HBM on TPU (CPU reshard values verified bit-exact ⇒ it's the
known uninit-HBM read, NOT a math bug). Refuted: fused gate_up_proj (checkpoint has separate w1/w2/w3 keys).

## NEXT ACTION — fix the W1/W3 consolidation so it reshards WITHOUT reading uninit, then validate
Make w1/w3 load through the SAME clean reshard path as w2. Candidates (try in order, each needs the 2-engine gate):
1. At consolidation (`deepseek_v4.py:1535`): reshard each leaf to replicated `P()` BEFORE `jnp.stack`, then
   device_put to e_spec — uniform shard-from-replicated for all 3. WATCH the memory contract (:1734, leaves+stack
   coexist); leaves are fp4 ~4MB×256 — replicating may spike per-chip HBM. Lowest-conceptual-risk if memory holds.
2. In `pick_partition_spec` (:508): make EXPERT leaves (w1/w2/w3) shard on a CONSISTENT axis (e.g. axis 0) so all
   three take w2's clean path. Surgical risk: function is GENERAL — guard to expert leaves only, don't regress
   attention/other weights (a wrong global change → launch-id desync or perf regression).
3. Force a clean materialization of the stacked tensor (e.g. block_until_ready already there; try an explicit
   zero-init copy or optimization_barrier on the device_put result) — weakest, uninit-reshard may survive it.
GATE (validate the fix, TWICE on fresh engines): rerun `/tmp/s1_probe2.py 2` ×2 engines, `bash /tmp/s1_cke.sh LOG`
⇒ **[ckSPLIT] W1 & W3 gsum+absmax BYTE-IDENTICAL ×2** (currently the only failing quantities). THEN decode FIB md5
identical ×2 + correct Fibonacci (21,34,55,89,144,233,377,610,987,1597) + CLAUDE.md formal gate ×2 →
`touch /tmp/s1_loop_stop`. Keep [ckSPLIT] in until confirmed; remove all diagnostics when S1 closes.

## Ops (jax.debug.print LOSSY — SCALAR single-line prints; re-fire + sort -u to beat drops)
* ONE engine. Cache warm ⇒ ~350s ready; FIRST decode req recompiles ~325s, re-fires ~120s. Probe:
  `python3 /tmp/s1_probe2.py 2` (max_tokens=2 = 1 clean decode step, output '21,'). Extract: `bash /tmp/s1_cke.sh <log>`
  (now greps [ckSPLIT] too). Re-fire ×3 to collect dropped lines (decode is within-process deterministic).
* edit→`full_slice_v4_sync.sh`→verify md5 8/8 as **enyouki@** (NOT mark)→CLEAR xla_cache 8 hosts if .py changed→
  `reset.sh`→`smoke.sh`→wait `Application startup complete`. ssh -i ~/.ssh/google_compute_engine enyouki@<ip>;
  full_slice_v4_discover.sh for IPs. node_guardian UP (pid 497956); meta_guardian running (cluster HEALTHY).
* SECOND-ENGINE GOTCHA: after engine-1 populates xla_cache, a warm engine-2 start hit a "different launch id"
  Core-halt; fix = clear xla_cache 8 hosts + reset + re-smoke (engine 2 cold). Always cold-compile both engines.
* PITFALL HIT THIS SESSION: `pkill -f s1_probe2.py` SELF-MATCHED my own shell (cmdline contained the string) →
  killed the reset. Never pkill a pattern your own command line contains; reset.sh stops probes anyway.
* CPU-validate a new shard_map cheaply: JAX_PLATFORMS=cpu + XLA_FLAGS=--xla_force_host_platform_device_count=32
  (see /tmp/s1_cksplit_validate.py) BEFORE the TPU smoke — catches spec bugs without the cold compile.

## DEAD (do not retry): all-reduce (per-rank local already diverges, S25); input/prefill cascade (moe_shared+
routing byte-identical ×2); SUBLANE-PAD / gmm-kernel fix for DECODE (gmm prefill-only); bf16↔fp32 gmm dtype;
zero_initialize; partial_out_ref; gmm_v2:518 OOB; GATE/ROUTING sharding; wsc(act,P()) token axis; the dense
einsum MATMUL itself (gate/up/out diverge ONLY because W1/W3 inputs diverge — W2 clean ⇒ matmul faithful). The
"matmul output buffer uninit / pad N→8" lead is now SECONDARY (weights are the proven cause; fix them first).

# S1 handoff — PIVOT: decode = DENSE path, not gmm; routed output diverges from IDENTICAL input

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL / how to judge (see [[s1-symptom-nondeterminism-not-collapse]])
Bug is NOT collapse. Symptoms: (1) cross-process non-determinism (FIB md5 differs across fresh engines;
deterministic WITHIN a process), (2) quality drift. DONE = byte-identical md5 across 2 fresh engines AND
correct Fibonacci. Model is INSTRUCT. Correct FIB after "13, " = 21,34,55,89,144,233,377,610,987,1597,2584,4181,6765.

## STATE (2026-05-26 S25) — MAJOR PIVOT: S24's gmm finding was PREFILL-ONLY; decode uses the DENSE path
Decode (N=1, replicated by `_v4_decode_replicate`) takes `use_shard_map=False` (deepseek_v4_moe.py:224,
`N>=axis` false for N=1<32) ⇒ the **DENSE einsum path (moe.py:238-250)**, which NEVER calls gmm_v2. gmm runs
ONLY in PREFILL. So S24's "gmm kernel non-det on owned rows" is a real but **PREFILL** finding; SUBLANE-PAD is
**DEAD for the decode symptom**.
* **Mined S24 server logs (174930Z/180255Z) for DECODE-step [ckS] across 2 engines:** `moe_perexpw`
  (routing) BYTE-IDENTICAL ×2 every step; `moe_shared` (=f(flat_x), same input) BYTE-IDENTICAL ×2 every step;
  **`moe_routed_y` DIVERGES ×2 EVERY decode step** (e.g. decode-1 7.78 vs 2.18). ⇒ decode MoE INPUT + routing
  are CLEAN; the dense routed path (lines 238-250) computes DIVERGENT routed output from BYTE-IDENTICAL input.
* **Weights EXONERATED** (agent + safetensors index): w1/2/3_stacked are jnp.stack of fully-loaded experts,
  E=256/32=8 exact (no pad), block_until_ready before release. Not the corruptor.
* **CPU runs the SAME dense einsum deterministically** ⇒ the only TPU-added factor is E-sharding over attn_dp
  + the `out_NEd.sum(axis=1)` XLA-inferred CROSS-RANK all-reduce. Prime suspect = that reduction/its buffers
  (the einsum compute itself is least likely; CPU-clean). Structurally same reduction as prefill psum :360.

## IN FLIGHT — [ckEtot]/[ckEY] disambiguator smoke (engine 1 RUNNING)
Added to dense path (moe.py ~250, decode-active `layer_idx==0`, read-only, y unchanged). NOTE: per-rank
debug.print is too LOSSY (S25 first run: only 2/32 ranks survived; jax.debug.print drops 32 competing lines),
so the hardened version returns per-rank stats from the shard_map and prints SCALAR totals:
* **[ckEtot]** `pre_gsum/pre_sqsum/pre_absmax` = pre-reduce TOTALS via shard_map(out_specs=P('attn_dp',None))
  returning each rank's LOCAL out_NEd [gsum,sqsum,absmax] (NO collective), gathered (wsc P(), tiny/safe) +
  LOCAL-summed. `pre_sqsum` is reorder-immune + per-rank-sensitive.
* **[ckEY]** `y_gsum/...` = post-all-reduce y (the suspect path; == [ckS] moe_routed_y decode value).
Synced 8/8 (md5 8111447b..., ckEtot=3 each); xla_cache cleared; engine 1 log:
`logs/full-slice-v4-smoke-20260526T190818Z.log` (pid 2758672, COLD compile ~10-30 min). (Prior 184944Z run
used the lossy per-rank [ckE] — superseded; that run did confirm decode generates correct '21' + moe_routed_y
decode=60.05.) Grep: `bash /tmp/s1_cke.sh <log>`. Probe: `python /tmp/s1_probe2.py 2` (max_tokens=2).

## NEXT ACTION
1. Wait for engine-1 `Application startup complete`. Fire `python /tmp/s1_probe2.py 2` FIRST (1 clean decode
   step). `bash /tmp/s1_cke.sh <log>` → record [ckEtot] (decode line) + [ckEY] + FIB-ish text.
2. reset + smoke ENGINE 2, same. Compare across engines:
   * **[ckEtot] pre_sqsum byte-identical ×2 but [ckEY]/moe_routed_y DIFFERS ×2** ⇒ corruptor is the attn_dp
     ALL-REDUCE (every rank's LOCAL out_NEd is clean).
   * **[ckEtot] pre_sqsum DIFFERS ×2** ⇒ corruptor is the PER-RANK LOCAL einsum (XLA matmul scratch /
     _shard_e_mid reshard).
   (Within-engine: pre_gsum should ~= [ckEY] y_gsum; a big gap is itself all-reduce evidence.)
3. Fix follows the verdict (next session): all-reduce ⇒ force a deterministic explicit psum / clean the
   reduce-scatter+all-gather buffer; local ⇒ chase the einsum output/reshard uninit. GATE = decode moe_routed_y
   md5 ×2 + FIB md5 ×2 + correct Fibonacci, then CLAUDE.md formal gate ×2 + `touch /tmp/s1_loop_stop`.

## Ops
* ONE engine. Warm xla cache ⇒ ~5 min ready; cleared now ⇒ COLD ~10-30 min. FIRST req recompiles. Fire
  `max_tokens=2` FIRST (the 40-tok warmup pollutes per-step prints). `python /tmp/s1_warmup.py` = FIB 40tok
  md5+text. `bash /tmp/s1_ck.sh <log>` greps checksums. ssh user = **enyouki@** (NOT mark — DenyUsers).
  ssh -i ~/.ssh/google_compute_engine. Discover IPs: scripts/full_slice_v4_discover.sh.
* edit→`full_slice_v4_sync.sh`(verify md5 8/8 as enyouki@)→CLEAR xla_cache 8 hosts if .py changed→`reset.sh`→`smoke.sh`.
  node_guardian UP (pid 497956). meta_guardian BROKEN but cluster HEALTHY — don't re-fight.
* Diagnostics IN (cleanup AFTER fix): [ckE]/[ckEY] dense path ~250 (NEW S25), [ckR0/1b/2/3] prefill shard_map
  branch, [ckR] real-rows :172-182, [ckS] :230/:382/:388, [ckL] :358, [ckD] decode-logit deepseek_v4.py:2017.
  [ckS] fires BOTH paths (layer_idx==0); [ckR*]/_ckR fire PREFILL-only (n_real is not None).

## DEAD (do not retry): SUBLANE-PAD / any gmm-kernel fix for DECODE (gmm is prefill-only); bf16↔fp32 gmm dtype
(both non-det, prefill); zero_initialize; partial_out_ref carry; gmm_v2:518 OOB; GATE/ROUTING sharding (routing
byte-identical ×2 in decode too); EXPERT WEIGHTS uninit (E-stacked, fully written, safetensors-verified);
wsc(act,P()); prefill-replicate decode meta. Anchors: logs 174930Z+180255Z (S24, decode [ckS] divergence);
this session's engine logs from 184944Z onward.

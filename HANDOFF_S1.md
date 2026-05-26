# S1 handoff — VERDICT: decode corruptor = per-rank LOCAL dense einsum (all-reduce + input EXONERATED)

Goal: coherent, RELIABLE (cross-process deterministic) decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash`
on v6e-32. Ops in `CLAUDE.md`; this is live state.

## GOAL / how to judge (see [[s1-symptom-nondeterminism-not-collapse]])
Bug is NOT collapse. Symptoms: (1) cross-process non-determinism (decode routed output differs across fresh
engines; deterministic WITHIN a process), (2) quality drift. DONE = byte-identical FIB md5 across 2 fresh
engines AND correct Fibonacci. Model INSTRUCT. Correct FIB after "13, " = 21,34,55,89,144,233,377,610,987,1597...

## STATE (2026-05-26 S25) — VERDICT: the PER-RANK LOCAL dense einsum is the decode corruptor
Decode takes the DENSE einsum path (moe.py:238-250; N=1 replicated ⇒ use_shard_map False; gmm is PREFILL-only,
so S24's gmm lead is DEAD for decode). 2 fresh engines, max_tokens=2 decode step, identical code (md5 8111447b):
| quantity (decode, layer 0)        | engine-1 (190818Z) | engine-2 (192051Z) | verdict   |
|-----------------------------------|--------------------|--------------------|-----------|
| moe_shared (=f(input))            | -33.92131042       | -33.92131042       | IDENTICAL |
| moe_perexpw (routing)             |  1.499023438       |  1.499023438       | IDENTICAL |
| [ckEtot] pre_sqsum (per-rank LOCAL out_NEd, gather path — NOT all-reduce) | 34.21117020 | 34.47301102 | **DIFFERS** |
| [ckEtot] pre_gsum                 |  1.467573643       |  3.376025200       | **DIFFERS** |
| [ckEY] y / moe_routed_y (post-reduce) | 1.467573643    |  3.376025200       | DIFFERS (downstream) |
⇒ **the per-rank LOCAL out_NEd diverges from BYTE-IDENTICAL input.** pre_sqsum is reorder-immune & computed via
gather+local-sum (NOT the suspect all-reduce), yet differs ×2 ⇒ corruptor is UPSTREAM of any cross-rank reduce,
in the local compute (einsum chain `x@W1/W3 → silu·w → @W2`, lines 238-249). **ALL-REDUCE EXONERATED** (faithfully
reduces already-divergent locals). **INPUT/cascade EXONERATED** (moe_shared+routing byte-identical ×2).
Within-process: decode is DETERMINISTIC (engine-2 decode constant across 10 re-fires; only prefill [ckS] globals
vary per-request = known PAD-ROW noise, N=1 decode has none). Both engines generate correct '21' (md5 5bf42256).

## NEXT ACTION — sub-localize WITHIN the local compute (lines 238-249)
The dense path is "bit-for-bit unchanged original" + runs DETERMINISTICALLY on CPU ⇒ the TPU-specific factor
(E-sharding / wsc reshard / MXU M=1 matmul scratch / routed-weight load) reads per-process-constant uninit. Test:
1. **Checksum routed weights W1/W2/W3 cross-engine** (decode-active, layer 0): a SINGLE-LINE scalar sum/sqsum of
   each stacked routed weight (gathered, like [ckEtot]). If a weight sum DIFFERS ×2 ⇒ routed E-stacked load is
   uninit (shared expert is clean ⇒ moe_shared identical, but routed E-stacking is a different tensor). If
   IDENTICAL ⇒ weights clean, the matmul/reshard reads uninit.
2. **Split the einsum chain**: add scalar [ckEtot]-style per-rank-local checksums of gate_NEi/up_NEi (post x@W1/W3)
   and h_NEi, vs out_NEd (post @W2). Which stage first diverges ×2 pins the einsum (vs the _shard_e_mid wsc).
3. Likely fix: weights ⇒ fix routed load/stack zero-init; matmul ⇒ uninit MXU/accumulator scratch on M=1 decode;
   reshard ⇒ the _shard_e_mid with_sharding_constraint. GATE = decode moe_routed_y md5 ×2 + FIB md5 ×2 + correct
   Fibonacci, then CLAUDE.md formal gate ×2 + `touch /tmp/s1_loop_stop`.

## Ops (CRITICAL: jax.debug.print is LOSSY — use SCALAR single-line prints, never 32 per-rank lines)
* ONE engine. Warm xla cache ⇒ ~5 min ready (cache now warm from S25). FIRST req recompiles ~324s. Probe:
  `python /tmp/s1_probe2.py 2` (max_tokens=2 = 1 clean decode step; output '21,'). Grep: `bash /tmp/s1_cke.sh <log>`.
  Decode prints DROP under volume — RE-FIRE the same probe several times and `sort -u` to collect (decode is
  deterministic within-process, so re-fires give the same value; only need it to survive a drop once).
* edit→`full_slice_v4_sync.sh` (verify md5 8/8 as **enyouki@**, NOT mark)→CLEAR xla_cache 8 hosts if .py changed→
  `reset.sh`→`smoke.sh`→wait `Application startup complete`. ssh -i ~/.ssh/google_compute_engine.
  scripts/full_slice_v4_discover.sh for IPs. node_guardian UP (pid 497956); meta_guardian BROKEN, cluster HEALTHY.
* Diagnostics IN (cleanup AFTER fix): [ckEtot]/[ckEY] dense path ~250 (S25, decode-active, scalar); [ckS]
  moe_perexpw/moe_routed_y/moe_shared (:230/:382/:388, BOTH paths layer 0); prefill-only [ckR0/1b/2/3]/[ckR]/[ckL]
  in shard_map branch; [ckD] decode-logit deepseek_v4.py:2017.

## DEAD (do not retry): the attn_dp ALL-REDUCE (exonerated — per-rank local already diverges); input/prefill
CASCADE into decode (moe_shared+routing byte-identical ×2 in decode); SUBLANE-PAD / any gmm-kernel fix for DECODE
(gmm is prefill-only); bf16↔fp32 gmm dtype; zero_initialize; partial_out_ref carry; gmm_v2:518 OOB; GATE/ROUTING
sharding; EXPERT WEIGHTS uninit per S24 static analysis (BUT routed E-stacked weights NOT yet cross-engine
checksummed — that's NEXT step 1, don't assume clean); wsc(act,P()) on token axis. Anchors: /tmp/s1_eng1_s25.txt;
logs 190818Z (eng1) + 192051Z (eng2).

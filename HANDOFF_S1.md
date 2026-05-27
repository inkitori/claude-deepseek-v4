# S1 handoff — ✅ FIXED (v2a host-gather consolidation), gate passed ×2 fresh engines

Goal (MET): coherent, RELIABLE (cross-process deterministic) decode for
`vllm serve deepseek-ai/DeepSeek-V4-Flash` on v6e-32. Ops in `CLAUDE.md`.

## ✅ S1 FIXED (2026-05-27 S28)
ROOT CAUSE (S26): decode corruptor = routed **W1/W3 stacked-weight LOAD** — the device-side
consolidation reshard (`device_put(jnp.stack(axis-1-leaves), P('attn_dp',None,None))`,
deepseek_v4.py:1535) read uninit HBM (per-process garbage → cross-engine non-determinism).
W2 (axis-0 leaf) was byte-clean. S27 v1 (force axis-0 leaves) FAULTED → reverted.

FIX v2a (commit 5a3ed435): eliminate the device-side reshard for w1/w3 WITHOUT changing
leaf sharding. `place_spec_as_jax_sharded(return_host_np=True)` hands back the FULL host
numpy it already builds in the non-axis-0 full-read branch; the drain thread stashes it
per expert leaf; `_maybe_consolidate` for w1/w3 does `np.stack` of the 256 host numpys +
`jax.make_array_from_callback` straight into the expert-sharded layout (the loader's
most-exercised primitive — host→device, NO device reshard, NO uninit read). w2 stays on
the clean device path.

GATE — verified on 2 FRESH engines (logs 222628Z, 232312Z):
* Load: `host_gather_groups=88`, placed=35020, 0 faults, past v1's 600-tensor death zone.
* **[ckSPLIT] gsum+absmax byte-IDENTICAL ×2** (W1=9.748932e+03 W3=-2.956816e+02 absmax
  W1=0.25 W3=0.1875 on BOTH) — the S26 W1/W3 divergence is GONE.
* **FIB decode md5 = 5bf42256 IDENTICAL ×2**; correct Fibonacci (21,34,55,89,144).
* CLAUDE.md formal gate (smoke_check rc=0): Paris deterministic ×2, LONG_GEN
  visible_words=32 max_word_run=1 ends_clean=1; engine survived 4+ requests.
CPU value-equiv (host-gather == device-reshard byte-identical): /tmp/s1_consol_equiv.py.

## REMAINING (hygiene, NOT part of the gate) — then `touch /tmp/s1_loop_stop`
Remove the S1 debug instrumentation (CLAUDE.md discipline: "remove diagnostics when S1
closes"). All PURE DEBUG (no real-compute dependency) EXCEPT preserve
`y = out_NEd.astype(fp32).sum(axis=1)` (moe.py ~293, REAL). Inventory:
* moe.py: `_ckR` helper+calls (~172-183, 431, 438); `[ckS]` prints (229-233, 427-430,
  433-437); `_ck_all`+shard_map+`[ckSPLIT]`/`[ckEY]` (262-296, KEEP line 293); sharded-
  prefill `[ckR0/ckR1b/ckR2/ckR3/ckL]` (320-330, 353-369, 379-386, 403-405).
* deepseek_v4.py: `_v4_checksum` calls (328,331,341,344); `_v4_nan_tripwire` calls
  (362,368,372,374,381,384,386,515,736,741) + the helper defs if unused. **Do NOT touch
  the compute_logits nan_to_num clamp — that's functional, not a diagnostic.**
* Also consider dropping the now-redundant w1/w3 device-leaf placement (host-gather
  rebuilds from host stash; the sharded leaf is placed then None'd — wasted device
  transfer). Optional perf, leave if risky.
Then: sync → cold smoke → confirm FIB md5 == 5bf42256 + engine healthy + coherent →
commit + `touch /tmp/s1_loop_stop`. (Decode is slow ~0.3 tps; not an S1 concern but the
diagnostic removal should help the layer-0 hot path.)

## Ops (unchanged; jax.debug.print LOSSY — re-fire + sort -u). Helpers in /tmp: s1_probe2.py, s1_collect.sh, s1_gate.sh, s1_cke.sh
* ONE engine, flock-serialized. Warm cache ⇒ ~6 min ready (290s load); FIRST decode req
  recompiles ~325s. Probe: `python3 /tmp/s1_probe2.py 2` (1 decode step, '21,').
* edit→`full_slice_v4_sync.sh`→verify md5 8/8 (key `-i ~/.ssh/google_compute_engine`,
  user `enyouki@`)→CLEAR xla_cache 8 hosts if .py changed→`reset.sh`→`smoke.sh`→wait
  `Application startup complete`. IPs: head .192; workers .194 .202 .204 .193 .198 .195 .200.
* Gate replay: `bash /tmp/s1_collect.sh <log> /tmp/s1_engN.txt` ×2 engines →
  `bash /tmp/s1_gate.sh <e1> <e2>`. node_guardian pid 497956, meta_guardian 2480054 (its
  WrongClusterID spew is BENIGN). Orphaned `claude` loop pids may linger (flock-harmless).

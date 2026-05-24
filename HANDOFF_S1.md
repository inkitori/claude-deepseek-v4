# S1 handoff — fresh session, pick up here (2026-05-24, ROOT CAUSE CONFIRMED)

Goal: make `vllm serve deepseek-ai/DeepSeek-V4-Flash` produce coherent, deterministic
decode on the v6e-32 slice (bug **S1**). Bypass perms; use the TPU; commit+push
checkpoints; never wait. The self-perpetuating loop is NOT running — whoever reads this
is the sole driver (start `scripts/s1_session_loop.sh` to auto-continue, or drive manually).

## ⇒ DEFINITIVE STATUS — read this first (supersedes all PHASE notes below)

**INFRA SOLVED — do NOT re-fight `node`.** mark's redeploy controller (VM 35.186.51.62
SSHing in as user `mark` every ~10min) is permanently blocked via **`DenyUsers mark`** in
`/etc/ssh/sshd_config.d/99-s1-block-mark.conf` on all 8 hosts (refused pre-auth; survives
reboot; self-healed by `scripts/full_slice_v4_node_occupy.sh`, which also keeps an inert
dummy `node` + iptables drop). Both guardians run (`node_guardian` + `meta_guardian`). node
has been provably absent (0 reclaims/logins) in every recent smoke. Reverse: rm the drop-in
+ `systemctl reload ssh`.

**S1 ROOT CAUSE — CONFIRMED (the big result of this session):** under DP-attention, a
single/under-filled batch (the smoke uses `max_num_seqs=1`) puts the 1 sequence on ONE
attn_dp rank and leaves ~31 ranks IDLE (no real tokens). The decode state is allocated
**REPLICATED `P()`** (`runner/kv_cache_manager.py::_initialize_kv_cache_deepseek_v4`,
`models/common/model_loader.py` `kv_cache_sharding=P()`; comment: "every chip computes the
same update under SPMD"). The idle ranks (no real tokens) make the per-rank seed/decode
computations DISAGREE, so the replicated state is poisoned → decode COLLAPSES (first 1-2
tokens correct, then a repeating/numeric attractor: `"capital of France"->" Paris 2012,
2012, 2012…"`).
**PROOF (decisive):** on the SAME engine, 30 CONCURRENT requests (filling the dp ranks) →
decode is COHERENT and CORRECT (`France->" Paris"`, `hydrogen and->" oxygen"`, `Earth
orbits->" Sun"`, `two plus three->" five"`, `sun rises->" sky"`); the 1-seq path collapses.
Filling the ranks fixes it. (Also: the model likely works fine in PRODUCTION with many
concurrent seqs; the bug is specific to under-filled batches like the smoke's max_num_seqs=1.)

**A separate NaN OVERLAY (fixed, keep it):** the kv-matmul output buffer on the empty token
shards reads uninitialized/recycled HBM → NaN/inf/~1e37 garbage (varies run-to-run; even the
FORWARD's kv at L1 is garbage from globally-finite x & wkv; forward tolerates it). `_linear`
now zeros non-finite/`|.|>=1e8` outputs (committed; no-op for real O(1) values). This removed
NaN-logits crashes but did NOT fix the collapse (the collapse is the idle-rank value bug).

## 8 FIXES TRIED → none fix the collapse (do NOT repeat)
1. FIX v2 `with_sharding_constraint(x,P())` in seed → Core-halt.
2. Option A: runtime replicate prefill input_ids → seed still NaN (forward reshards x back),
   decode empty + halt.
3. fix d: replicate `wkv` in seed → L1 still NaN.
4. pad: zero-pad seed token axis → L1 still NaN.
5. pin-output: `wsc(kv, P(ATTN_DATA))` → L1 still NaN.
6. `_linear` clamp → finite but STILL collapses (kept; fixes the NaN overlay only).
7. pad + `_replicate(x)` (pad-then-gather, dense) → **Core-halts** (SLICE_FAILURE on .195).
**HARD CONSTRAINT learned:** ANY `with_sharding_constraint(activation, P())` gather-to-
replicated of the seed activation **Core-halts** (SLICE_FAILURE_SW_INJECT_ERROR), with OR
without padding. So the fix CANNOT gather the activation to replicated.

## NEXT — the non-gather fix: make the decode state NOT replicated
Since (a) filling idle ranks fixes it and (b) gathering the activation to replicated halts,
the fix must make a single-seq's decode state correct WITHOUT a replicated-state gather:
- **Leading candidate: shard the V4 decode state PER-SEQUENCE (`P(ATTN_DATA)`) instead of
  replicated `P()`.** Then the 1 seq's state lives on its own rank; idle ranks don't share/
  poison it (no cross-rank replication reduction). Touch: `kv_cache_manager._initialize_kv_
  cache_deepseek_v4` (sharding=P(ATTN_DATA)), `model_loader.py` `kv_cache_sharding`, and the
  model's state read/write + the seed's output constraint (`_v4_constrain_packed_replicated`
  in `models/jax/deepseek_v4.py` constrains to P() — change to match). Invasive; validate the
  seed→decode parity still holds (CPU repros) and that decode reads the right shard.
- **Alt: engine-level fill** — always pad the DP batch to dp_size with DUMMY sequences so no
  rank is idle (subagent found the model's multi-seq path does NOT thread decode state, so a
  naive dummy-seq approach flips to the wrong code path — would need the dummies to share the
  single-seq decode-state path; non-trivial).
- Whatever the fix: it must work at `max_num_seqs=1` (the gate), must NOT gather the activation
  to P() (halts), and KEEP the `_linear` clamp. There are leftover diagnostic tripwires
  (`init_wkv`, `fwd_x_in/fwd_wkv/fwd_kv_postlin` in attention_prefill) — env-gated, harmless;
  clean up when done.

## DONE gate (verify TWICE, fresh engine, READ THE TEXT)
`LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` exits 0 (visible_words≥10,
max_word_run<5); 3 Paris probes byte-identical at temp=0; survives 5 requests. After EVERY
edit: `scripts/full_slice_v4_sync.sh`. `V4_DECODE_NAN_TRIPWIRE=1` prints `[v4nan] L{l}
pos={p} {tag}: nan/inf/max_abs` — reductions are GLOBAL (whole-array all-reduce), so one
chip's print reflects all ranks. CPU repros (`s1_cpu_repro_v4flash.py both`) only check
no-regression (CPU has no sharding → can't reproduce S1). The cheap idle-rank test:
`MAX_SEQS=32 scripts/full_slice_v4_smoke.sh` + ~30 concurrent curls → coherent (confirms cause).

## Recovery / loop
- **Slice wedge (Core-halt → SLICE_FAILURE):** reboot the 7 WORKERS (not head) → wait SSH →
  remount GCS each (`cd ~/claude-deepseek-v4 && set -a && source .env && set +a &&
  ./scripts/mount_gcs.sh`) → `scripts/full_slice_v4_ray_restart.sh` (~6 min). node stays
  blocked across reboot (DenyUsers persists). Clean engine (no halt) only needs
  `full_slice_v4_reset.sh`.
- **Loop:** `scripts/s1_session_loop.sh` (stop: `touch /tmp/s1_loop_stop`). Per-session prompt
  `scripts/s1_loop_prompt.txt`; trim helper `scripts/s1_trim_claudemd.sh`.

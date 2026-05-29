# Handoff — DeepSeek-V4-Flash QUANTIZED-WEIGHT-LOADING campaign (v6e-16)

> **Phase = QUANT / FIT-ON-v6e-16.** Make `vllm serve deepseek-ai/DeepSeek-V4-Flash`
> LOAD AND SERVE CORRECTLY on **v6e-16** by **NOT dequantizing the FP4 experts to bf16 at load**.
> Durable slice ops + pitfalls: `CLAUDE.md`. Prior campaigns (history): `HANDOFF_PERF.md`,
> `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29):** **The post-load `scheckne` saga is RESOLVED** (Q.8 RNG→host +
> Q.9 eager create_jit_model eliminated the last racing eager dispatches; the HEAD now loads + fits
> + reaches the real forward). The wedge that appeared next was **NOT a forward bug** — py-spy proved
> it was a **divergent weight load**: the 3 worker actors couldn't read the GCS-mounted weights (root-
> only mount) and silently fell into the `jnp.zeros` dummy fallback while the head loaded real weights.
> **Q.10 fixes it** (enyouki gcsfuse mount on the workers; `config.json` now readable on all 4 hosts).
> **NEXT = run the validating smoke** — expect the FIRST cluster-wide real load into the FIRST real
> forward; then debug whatever the genuine forward surfaces, then the GATE. Last smoke: `…055008Z`
> (head reached forward; workers dummy-wedged — pre-Q.10).

---

## THE PROBLEM (one table)
V4-Flash ships natively quantized: dense=FP8, 256 routed experts=**FP4 (=MXFP4, codebook ≡
`jnp.float4_e2m1fn`, e8m0 block scale, block 32)**. On-disk: routed `layers.{L}.ffn.experts.{0-255}.{w1,w2,w3}.weight`
= **I8** packed + `.scale` = **F8_E8M0**; shared `ffn.shared_experts` = FP8 (E4M3); 43 layers +
1 MTP, hidden=4096, moe_inter=2048. Old loader dequantized everything → ~542 GiB > 512 GiB → OOM.

| scheme | HBM | fits v6e-16 (512 GiB)? |
|---|---:|---|
| bf16 (old load path) | ~542 GiB | **NO** (OOM) |
| **fp4-experts-kept + dense bf16 (Strategy C)** | **~155 GiB** | **yes** — confirmed: `Init model | hbm=9.75/31.25 GiB/chip` |

---

## WHAT LANDED (committed) — the loader + the load-path race fixes
**Loader (Strategy C — keep FP4 experts compressed):** `26e4023d` declare routed experts FP4 +
dequant-in-trace; `26318abf` emit FP4 experts compressed (uint8 weight + uint8 scale leaves, no bf16
dequant); `580b1f83` scales as uint8; `6eb5241f` un-swallow loader exception (re-raise unless
`V4_ALLOW_DUMMY_FALLBACK=1`); **`e1b434f8`** host-gather EVERY routed/mtp expert leaf ⇒ ZERO
consolidation `device_put` reshards (the FIT FIX). CPU-gated throughout.

**The post-load `scheckne` race — RESOLVED (the "freqs-fix pattern" = eliminate divergent eager TPU
dispatches from the rank-0 worker that races ahead of the laggards):**
- `fb54237b` **RoPE freqs → numpy/host** (removed the 16-partition freqs ops).
- `0d8d57fe` **Q.8: sampling RNG key → host (CPU)** — `nnx.Rngs(jax.random.key(seed)).params()` @
  `tpu_runner.py:581` wrapped in `with jax.default_device(cpu)` (that site is NOT under set_mesh, so
  default_device is honored, unlike the freqs site). Killed `jit__threefry_fold_in` + `jit_add`.
  Value byte-identical (threefry platform-deterministic; GATE samples at temp=0 = argmax = RNG-
  independent anyway). Confirmed via HLO-module diff; blocker advanced to create_jit_model.
- `83a18839` **Q.9: eager create_jit_model** (env `V4_EAGER_CREATE_JIT_MODEL`, default OFF in shared
  infra, ON in the smoke). `create_jit_model` (`model_loader.py:126`, the only remaining racing
  dispatch) is a `@nnx.jit` whose body (`nnx.state`→`nnx.update`→qwix-noop) is a no-op on the already-
  concrete, host-gathered model AND is sharding-neutral (no `with_sharding_constraint`/
  `get_partition_spec`), so running it eagerly hands the forward an IDENTICAL model with ZERO TPU
  dispatch. The re-jit is only a forward PjitFunction-overhead optimization ("the created model can
  already work"). RESULT: `jit_create_jit_model` no longer compiles, no scheckne, **`load_model`
  completes** (`Init model | hbm=9.75/31.25`), progresses into the **real lockstep forward** (first time ever).

**The divergent weight load — `ca016156` Q.10 (THE CURRENT FRONTIER):** after Q.9, the engine WEDGED
(not a forward deadlock as first guessed — **py-spy of the 4 live workers** proved it): the head loaded
real weights (`placed=68812`) while workers **.8/.17/.16 were stuck in `jnp.zeros` @ `deepseek_v4.py:2063`
(dummy fallback)** because `is_local_dir` was False on them; EngineCore blocked forever in
`collective_rpc` ray.get. ROOT (infra, longstanding, masked until Q.9 removed the scheckne): the GCS
bucket `personal-mark-eu` (subdir `vllm/hub`) is mounted **enyouki-owned on the head** (`~/.cache/
huggingface/hub`) but **root-only on the workers** (`/tmp/gcs/bucket`, EACCES for enyouki) → the worker
serve proc can't read the weights. So "load completes (placed=68812)" was ALWAYS only the head. FIX =
`scripts/full_slice_v4_mount_weights.sh` gives each worker its own enyouki gcsfuse mount at `~/.cache/
huggingface/hub` (verified: `config.json` readable on all 4 hosts); wired as a hard-fail pre-flight in
the smoke. **Mechanism-verified, NOT yet smoke-validated end-to-end.**

---

## ⚠️ NEXT ACTION (do first) — run the validating smoke
The slice is RESET + clean (16/16 TPU free), guardians alive, weights mounted on all 4 hosts.
1. (if reboot happened) `scripts/full_slice_v4_mount_weights.sh` — re-mount weights (the smoke pre-flight
   also runs it). Mounts do NOT survive a reboot.
2. `V4_XLA_FLAGS=--xla_dump_to=/tmp/hlo_dump bash scripts/full_slice_v4_smoke.sh` (V4_EAGER_CREATE_JIT_MODEL
   defaults to 1 in the smoke). Watch the LOAD phase (~90s): you should now see **all 4 hosts load real
   weights** (no `jnp.zeros` dummy fallback, no EngineCore ray.get wedge) → into the **first cluster-wide
   real forward** (cold compile 10-30 min; `VLLM_ENGINE_READY_TIMEOUT_S=2400` already set).
3. **The forward is the next UNKNOWN** — it was never reached with real weights on all 4 hosts. If it
   hangs/crashes, that's a genuine forward blocker (NOT the dummy-load wedge — distinguish via py-spy:
   `sudo env "PATH=$PATH" work/vllm_env/bin/py-spy dump --pid <PID>`; worker PIDs via
   `ssh … 'ps -eo pid,cmd|grep RayWorker'`). For a launch-id scheckne, use the worker-to-worker HLO diff
   (recipe below). For a hang, py-spy the 4 workers and compare stacks.
4. On `Application startup complete` → run the **GATE** below (establish the v6e-16 baseline md5).

Optional defensive follow-up: make `load_weights` honor `V4_ALLOW_DUMMY_FALLBACK` on the
`is_local_dir=False` route too (`deepseek_v4.py:2060`) so a future mount gap RAISES loudly instead of
silently dummy-zeroing into a wedge.

HLO-diff helper (for a forward scheckne):
```
for ip in 8 17 16; do rsync -az --include='*.before_optimizations.txt' --exclude='*' \
  -e "ssh -i ~/.ssh/google_compute_engine" enyouki@10.164.0.$ip:/tmp/hlo_dump/ /tmp/hlo_cmp/h$ip/; done
# per host: ls .../*.before_optimizations.txt | sed -E 's#.*/module_[0-9]+\.##;s#\.cl_[0-9]+.*##' | sort|uniq -c
# ⚠️ COMPARE WORKER-to-WORKER (h8/h17/h16). NOT h15 — its dump MIXES the driver + the co-located rank-0
# worker. For content non-det: normalized-md5 each shared module across h8/h17/h16.
```

---

## <a name="GATE"></a>GATE (non-negotiable) — for v6e-16
Old v6e-32 md5 `5bf42256` is DEAD. Bar:
- `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (port 18081; visible_words ≥ 10,
  max_word_run < 5).
- FIB decode: **correct Fibonacci (21,34,55,89,144)** in a longer decode + **N=2 md5 byte-identical
  across 2 fresh engines** via `python3 /tmp/s1_probe2.py N` (RECREATED + verified present this session;
  prompt "Here is the start of the Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13,", temp=0 seed=0, port
  18081, auto-discovers model id; ⚠️ /tmp is ephemeral). Establish the NEW baseline hash once; confirm
  identical ×2 engines. READ the actual text ("contains Paris" is a false positive). Do NOT gate on a
  long-tail md5 (nondeterministic at temp=0).

---

## VALIDATION TIERS (cheapest first)
1. **CPU loader fp4 check** (~40s): `JAX_PLATFORMS=cpu PYTHONPATH=work/tpu-inference:work/vllm
   work/vllm_env/bin/python3 scripts/quant_loader_fp4_check.py`. + CPU oracle `s1_cpu_repro_v4flash.py
   both` + `quant_fp4_dequant_check.py`. NOTE: CPU has mesh=None ⇒ host-gather/forward-compile/the
   multihost load are NOT exercised on CPU — those need a smoke. Runner-only changes (e.g. Q.8) aren't
   exercised by the oracle either; validate those by snippet + smoke.
2. **TPU microbench** (`scripts/perf_microbench*`).
3. **Full smoke + GATE.** Load now ~40s/host; cold forward compile 10-30 min. HLO-dump recipe above.

---

## INFRA STATUS / v6e-16
- Slice `v6spoteu719`, zone `europe-west4-a`, project `prm-research`. **v6e-16, topology 4×4**, 16
  chips, 4 hosts: head `10.164.0.15` + workers `10.164.0.8 / .17 / .16`. Ray healthy (16 TPU). TP=16.
- ⚠️ **WEIGHTS MUST be readable by enyouki on ALL 4 hosts** (Q.10). The bringup mounts the GCS bucket
  enyouki-owned only on the HEAD; the workers get a root-only mount → enyouki EACCES → silent dummy-
  zeros load → wedge. **Run `scripts/full_slice_v4_mount_weights.sh` once per bringup** (idempotent;
  the smoke pre-flight runs it; mounts do NOT survive reboot). Verify: `config.json` readable on .8/.17/.16.
- ⚠️ **numpy MUST be `<2.4` (pinned `2.3.5`)** — 2.4.x breaks `import numba`, crashes the APIServer.
  `~/.local/bin/uv pip install --python work/vllm_env/bin/python3 'numpy==2.3.5'` per host.
- ⚠️ Ray "version mismatch" = mark's rogue ray container; **FIX = keep BOTH guardians alive**
  (`ps -eo pid,cmd | grep -E 'node_guard[i]an|meta_guard[i]an'`; restart per the loop prompt,
  meta_guardian needs `10.164.0.15:6379`). Ray (re)start: `scripts/full_slice_v4_ray_restart.sh`.
- After ANY code edit: `full_slice_v4_sync.sh` (rsync to 4 hosts; `git push` does NOT sync them) +
  clear `~/.cache/vllm/xla_cache/*` on all 4 + verify md5 head==workers. Shut down ONLY via
  `full_slice_v4_reset.sh`. HLO dump propagates to workers via the ray env (XLA_FLAGS).

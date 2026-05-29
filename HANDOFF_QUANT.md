# Handoff — DeepSeek-V4-Flash QUANTIZED-WEIGHT-LOADING campaign (v6e-16)

> **Phase = QUANT / FIT-ON-v6e-16.** Make `vllm serve deepseek-ai/DeepSeek-V4-Flash`
> LOAD AND SERVE CORRECTLY on **v6e-16** by **NOT dequantizing the FP4 experts to bf16 at load**.
> Durable slice ops + pitfalls: `CLAUDE.md`. Prior campaigns (history): `HANDOFF_PERF.md`,
> `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29):** **LOADS + FITS; post-load launch-id `scheckne` persists. BOTH
> barrier approaches RULED OUT this session — PIVOT to the FREQS-FIX pattern: eliminate the divergent
> eager TPU dispatches, don't try to barrier them.** Race (proven): the rank-0 worker (co-located with
> the EngineCore driver on head .15, lowest RPC latency) finishes the collective-free load FIRST and
> races into post-load eager programs while the 3 remote workers still place weights → launch-id
> mismatch → `scheckne`. **LANDED precedent `fb54237b`:** RoPE freqs precompute → NUMPY (host) removed
> the 16-partition `jit_iota/outer/exp` from the racing rank; load progressed (3→113 modules) to the
> NEXT eager dispatch. **THIS SESSION (all reverted to baseline, CPU-clean):**
> (1) `sync_global_devices` barrier → BECOMES the divergent 16-partition `identity_fn` collective itself
> (crashes AT the barrier, earlier). (2) host-side `wait_at_barrier` → SILENT NO-OP:
> `jax._src.distributed.global_state.client is None` here (no `jax.distributed.initialize()`; the TPU
> handshake is libtpu-env-var-based) — instrumentation logged `coord=NONE`; `jax.process_count()` still
> returns 4 so the guard passed misleadingly. ⇒ NO coord-service barrier exists. (3) PROVEN: the 4
> workers are BYTE-IDENTICAL (no XLA non-determinism); the divergent post-load programs are
> `jit_create_jit_model` + `jit__threefry_fold_in` + `jit_add`, ALL host-side (0 collectives), run in
> every worker. **NEXT = apply the freqs-fix pattern to the RNG** (`nnx.Rngs(jax.random.key(seed)).params()`
> @ `tpu_runner.py:581` → host/numpy) then create_jit_model; see ROADMAP. Last smokes:
> `…045525Z` (HLO dump → worker-diff), `…051036Z` (barrier instrumented → coord=NONE). Crash ~60-90s (cheap).

---

## THE PROBLEM (one table)
V4-Flash ships natively quantized: dense=FP8, 256 routed experts=**FP4 (=MXFP4, codebook ≡
`jnp.float4_e2m1fn`, e8m0 block scale, block 32)**. On-disk: routed `layers.{L}.ffn.experts.{0-255}.{w1,w2,w3}.weight`
= **I8** packed + `.scale` = **F8_E8M0**; shared `ffn.shared_experts` = FP8 (E4M3); 43 layers +
1 MTP, hidden=4096, moe_inter=2048. Old loader dequantized everything → ~542 GiB > 512 GiB → OOM.

| scheme | HBM | fits v6e-16 (512 GiB)? |
|---|---:|---|
| bf16 (old load path) | ~542 GiB | **NO** (OOM) |
| **fp4-experts-kept + dense bf16 (Strategy C)** | **~155 GiB** | **yes** — and now LOADS (68812 tensors) |

---

## WHAT LANDED (committed, CPU-gated) — Strategy C loader + the FIT FIX
- `26e4023d` Q.1 + MoE consumer: routed experts declared FP4, dequant-in-trace (`_dequant_fp4_experts`).
- `26318abf` Q.2 loader: emit FP4 experts COMPRESSED (uint8 weight + scale leaves), no bf16 dequant.
- `580b1f83` scales stored as **uint8** (not on-device `float8_e8m0fnu`) — REFUTED the e8m0-halt theory
  (crash was byte-identical with uint8). Kept: it's correct (matches gpt_oss/qwix/the consumer).
- `6eb5241f` un-swallow loader exception (`deepseek_v4.py:2026` re-raises unless
  `V4_ALLOW_DUMMY_FALLBACK=1`) — REFUTED the swallowed-exception theory (re-raise never fired ⇒ no
  exception). Correctness win: never silently serve zero-weighted garbage. + host-gather w1/w3/scales
  (left w2 on device_put) — STILL crashed, proving even ONE consolidation device_put diverges here.
- **`e1b434f8` THE FIT FIX:** host-gather **EVERY** routed/mtp expert leaf (w1/w2/w3 + all scales) ⇒
  ZERO consolidation `device_put` reshard collectives. Edits in `deepseek_v4.py`: `_is_stash_leaf`
  (`return m is not None`) + `use_host_gather` (`all(k in _expert_host_np ...)`) match all routed
  leaves; `deepseek_v4_loader.py:981` `place_spec_as_jax_sharded` full-reads + returns host_np when
  `return_host_np=True` even for axis-0 leaves (so host-gather has the data).
  - **WHY it was needed:** uint8 packing makes w1/w3 SQUARE `[2048,2048]` → `pick_partition_spec`
    strict-`>` tie-break (`deepseek_v4_loader.py:~513`) picks **axis-0** → slice-aware path → host_np
    was None → host-gather DORMANT → `device_put` reshard. Those reshards are cross-host collectives;
    on the v6e-16 4×4 topology they desync the SPMD launch group (HLO diff proved: HEAD launched a
    real-data consolidation stack while the 3 WORKERS launched whole-tree scalar-fills). bf16's lone
    w2 device_put worked on v6e-32 but NOT here — only ZERO consolidation collectives passes. (This is
    the same axis-0 consolidation S1's S27 tried + reverted — see `HANDOFF_S1.md`.)
  - host-gather = `make_array_from_callback` from full per-expert host numpy; collective-free + byte-
    clean (no uninit-HBM reshard), so S1-safe. Load is now fully collective-free → completes.

---

- **`fb54237b` FIX #1 (RoPE freqs):** HLO-diff (head h15 vs workers h8/17/16) proved the post-load
  scheckne was the eager freqs precompute: under `set_mesh(mesh)` `precompute_freqs_cis`'s jnp
  arange/outer/exp compiled `num_partitions=16` (`jit_iota/outer/exp/power`, `T(1024)` tiling) and the
  driver launched them while workers still ran `broadcast_in_dim` weight fills. `jax.default_device(cpu)`
  was a NO-OP (set_mesh overrides it — dump byte-identical, still 16-partition). FIX = `precompute_freqs_cis`
  now NUMPY (host; also honors the float64 complex-exp TPU truncates); `make_freqs_cis` returns the numpy
  UNCOMMITTED (device_put-cpu is rejected by create_jit_model's jit mesh: "Received incompatible devices").
  CPU oracle OK ×2; smoke: freqs ops GONE from head, load → real TPU compile (head 3→113 modules).

---

## ⚠️ CURRENT BLOCKER — post-load `scheckne` (barrier approach RULED OUT; pivot to freqs-fix pattern)
On the collective-free host-gather load that LOADS + FITS: load completes (`placed=68812`, ~40s) then
`scheckne` at `TensorCoreSequencer` (tpu17) ~60-90s in. Now well-characterized (this session's agents +
worker-to-worker HLO diff + barrier instrumentation):
- **Race:** the rank-0 worker (co-located with the EngineCore driver on head .15; lowest RPC latency)
  finishes the collective-free, HLO-free `make_array_from_callback` load FIRST and races into the
  post-load eager programs while the 3 remote workers (.8/.17/.16) still place weights → fast rank
  launches a program the laggards don't have at the same launch slot → launch-id mismatch.
- **Workers are BYTE-IDENTICAL** (worker-to-worker diff: .8≡.17≡.16, every module's normalized md5
  matches) → NO XLA compiler non-determinism. ⚠️ **Diff WORKER-to-WORKER (.8/.17/.16), NOT
  head-vs-worker:** head .15's `/tmp/hlo_dump` MIXES the EngineCore driver (few modules) + the
  co-located rank-0 worker (full set) → the OLD head-vs-worker diffs were apples-to-oranges
  (they mistook the driver's small dump for "head-only divergent modules").
- **Divergent post-load programs** (on the racing rank): `jit_create_jit_model` (the `@nnx.jit` @
  `model_loader.py:126`, called `:274` inside get_model) + `jit__threefry_fold_in` + `jit_add` (the RNG
  `nnx.Rngs(jax.random.key(seed)).params()` @ `tpu_runner.py:581`). ALL host-side (**0 collectives, 0
  all-reduce** — confirmed in the HLO): model-state layout + seed-only RNG. They run in ALL 4 workers
  (identical), just at different wall-clock times. (create_jit_model + RNG run in the worker actors,
  NOT the driver — RayDistributedExecutor dispatches `load_model` via `collective_rpc` to 4 actors.)

### Why BARRIERS are ruled out (both tried + REVERTED this session)
- **Device-collective `sync_global_devices`** (`jax.experimental.multihost_utils`): compiles to a
  16-partition `jit__identity_fn` collective that ITSELF needs lockstep launch → it BECOMES the new
  divergent module (the fast rank launches it while laggards place weights) → crashes AT the barrier,
  EARLIER than baseline. HLO-diff confirmed `identity_fn` as the divergent module_0012.
- **Host-side `wait_at_barrier`** (`jax._src.distributed.global_state.client.wait_at_barrier(id, ms)` —
  the coord-service barrier jax's OWN checkpoint mgr uses, signature verified): is a **SILENT NO-OP
  here** — the client is `None` because this libtpu-env-var-based Ray setup never calls
  `jax.distributed.initialize()`, so there is no coordination service. Instrumentation logged
  `rank=0 ... PASSED ... coord=NONE`. (`jax.process_count()` returns 4 from the TPU topology, so a
  `process_count()>1` guard passes MISLEADINGLY.) ⇒ No host-side barrier exists without first standing
  up a coordination service. Verified-correct call (for reuse if a coord svc is set up later):
  `from jax._src import distributed; distributed.global_state.client.wait_at_barrier("id", 600000)`.

### ROADMAP / NEXT ACTIONS (do in order) — eliminate divergent dispatches, the proven lever
1. **Apply the FREQS-FIX PATTERN to the RNG** (proven; `fb54237b` did exactly this for freqs).
   `nnx.Rngs(jax.random.key(self.model_config.seed)).params()` @ `tpu_runner.py:581` compiles
   `jit__threefry_fold_in`+`jit_add` on-device. The key is seed-only/host-deterministic → compute it on
   HOST (numpy / `jax.default_device(cpu)`) and feed the result to `device_array(...)` (the `:582`
   device_array is already host-side make_array_from_callback under Ray). Removes 2 of the 3 divergent
   dispatches from the racing rank. Validate: CPU oracle (no-op for this, confirms no break) + smoke.
   Each removal historically PROGRESSES to the next eager dispatch (freqs: 3→113 modules).
2. **Then create_jit_model** (`@nnx.jit` @ `model_loader.py:126`, donate_argnums=(0,)): host-side
   model-state realization (1492 sharding annots, 0 collectives). If removing the RNG alone doesn't get
   past it, investigate making it NOT dispatch a divergent 16-partition program (run eagerly / outside
   `set_mesh`, or guarantee lockstep launch). Shared infra — keep the diff minimal.
3. **(Deprioritized) a WORKING barrier**, only if 1-2 don't suffice: (a) call `jax.distributed.initialize()`
   early to populate `global_state.client` (RISK: conflict with the libtpu handshake / already-init
   devices), or (b) a non-jax rendezvous (Ray `collective_rpc` barrier, or shared-FS barrier). Barriers
   have NOT helped — try dispatch-elimination first.
4. After it gets past create_jit_model: repeat the WORKER-to-WORKER HLO-diff for the next eager dispatch;
   then the first real forward collective + cold compile (10-30 min, `VLLM_ENGINE_READY_TIMEOUT_S=2400`
   already set) → **GATE** below (establish the v6e-16 baseline md5).

OPEN QUESTION (does NOT block step 1): is the TPU launch-id collective-gated (only cross-core collectives
must match) or per-program (every dispatch counts)? Evidence is mixed, BUT the freqs fix EMPIRICALLY
worked by removing host-side eager dispatches → "eliminate the divergent dispatch from the racing rank"
is the working lever regardless of the exact mechanism.

HLO-diff helper (proven this session):
```
for ip in 8 17 16; do rsync -az --include='*.before_optimizations.txt' --exclude='*' \
  -e "ssh -i ~/.ssh/google_compute_engine" enyouki@10.164.0.$ip:/tmp/hlo_dump/ /tmp/hlo_cmp/h$ip/; done
# then per host: ls .../*.before_optimizations.txt | sed -E 's#.*/module_[0-9]+\.##;s#\.cl_[0-9]+.*##' | sort|uniq -c
# ⚠️ COMPARE WORKER-to-WORKER: h8 vs h17 vs h16 (3 pure workers). DO NOT use h15 — its /tmp/hlo_dump
# MIXES the EngineCore driver (few modules) + the co-located rank-0 worker, so h15-vs-workers is
# apples-to-oranges. For content non-determinism: normalized-md5 each shared module across h8/h17/h16.
```

---

## <a name="GATE"></a>GATE (non-negotiable) — for v6e-16
Old v6e-32 md5 `5bf42256` is DEAD. Bar:
- `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (port 18081; visible_words ≥ 10,
  max_word_run < 5). (smoke_check uses prompt 'The capital of France is', temp=0.)
- FIB decode: **correct Fibonacci (21,34,55,89,144)** in a longer decode + **N=2 md5 byte-identical
  across 2 fresh engines** via `python3 /tmp/s1_probe2.py N` (RECREATED this session — prompt "Here is
  the start of the Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13,", temp=0 seed=0, auto-discovers model id;
  ⚠️ /tmp is ephemeral — the file may be gone next session, the prompt above reproduces it). Establish
  the NEW baseline hash once; confirm identical ×2 engines. READ the actual text ("contains Paris" is a
  false positive). Do NOT gate on a long-tail md5 (nondeterministic at temp=0).

---

## VALIDATION TIERS (cheapest first)
1. **CPU loader fp4 check** (~40s, no slice): `JAX_PLATFORMS=cpu PYTHONPATH=work/tpu-inference:work/vllm
   work/vllm_env/bin/python3 scripts/quant_loader_fp4_check.py` → "OK …". (Needs `work/scratch/tiny_v4_quant`
   + `tiny_v4_groundtruth` fixtures.) **All CPU gates currently PASS.**
1b. **CPU oracle** `scripts/s1_cpu_repro_v4flash.py both` (eager==jit, bad=0/12) + **fp4 dequant**
   `scripts/quant_fp4_dequant_check.py` (max|Δ|=0). NOTE: CPU has mesh=None ⇒ consolidation/host-gather
   is SKIPPED, so CPU CANNOT validate the host-gather or forward-compile paths — those need a smoke.
2. **TPU microbench** (`scripts/perf_microbench*`).
3. **Full smoke + GATE** — load completes (`placed=68812`, ~40s) then hits the create_jit_model/RNG
   `scheckne` ~90s in. Cheap to iterate the blocker (crashes ~90s). HLO-dump recipe + cross-host diff
   (helper above) is the tool: clear xla_cache+/tmp/hlo_dump on all 4, smoke with
   `V4_XLA_FLAGS=--xla_dump_to=/tmp/hlo_dump`, then per-host opname-multiset diff WORKER-to-WORKER
   (h8/h17/h16 — NOT h15, see helper caveat below) → divergent module = the culprit eager program.

---

## INFRA STATUS / v6e-16
- Slice `v6spoteu719`, zone `europe-west4-a`, project `prm-research`. **v6e-16, topology 4×4**, 16
  chips, 4 hosts: head `10.164.0.15` + workers `10.164.0.8 / .17 / .16`. Ray healthy (16 TPU). TP=16.
- ⚠️ **numpy MUST be `<2.4` (pinned `2.3.5`)** — 2.4.x breaks `import numba`, crashes the APIServer
  before any TPU work. venv has no pip (uv): `~/.local/bin/uv pip install --python work/vllm_env/bin/python3
  'numpy==2.3.5'` per host. Pre-smoke: `python3 -c "import numba"` per host.
- ⚠️ Ray "version mismatch" = mark's rogue ray container; **FIX = keep BOTH guardians alive**
  (`ps -eo pid,cmd | grep -E 'node_guard[i]an|meta_guard[i]an'`; restart per the loop prompt,
  meta_guardian needs `10.164.0.15:6379`). Ray (re)start: `scripts/full_slice_v4_ray_restart.sh`.
- After ANY code edit: `full_slice_v4_sync.sh` (rsync to 4 hosts; `git push` does NOT sync them) +
  clear `~/.cache/vllm/xla_cache/*` on all 4 + verify md5 head==workers (mismatch → launch-id halt).
  Shut down ONLY via `full_slice_v4_reset.sh`. HLO dump propagates to workers via the ray env (XLA_FLAGS).

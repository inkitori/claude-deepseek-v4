# Handoff — DeepSeek-V4-Flash QUANTIZED-WEIGHT-LOADING campaign (v6e-16)

> **Phase = QUANT / FIT-ON-v6e-16.** Make `vllm serve deepseek-ai/DeepSeek-V4-Flash`
> LOAD AND SERVE CORRECTLY on **v6e-16** by **NOT dequantizing the FP4 experts to bf16 at load**.
> Durable slice ops + pitfalls: `CLAUDE.md`. Prior campaigns (history): `HANDOFF_PERF.md`,
> `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29):** **LOADS + FITS; now grinding through a CHAIN of post-load
> launch-id `scheckne` races, fixing them one HLO-diff at a time.** Root pattern (HLO-diff proven):
> during init the DRIVER (head) finishes its work and races ahead to the next eager-TPU program
> (compiled `num_partitions=16`) while the 3 worker actors are still running `broadcast_in_dim`
> weight placement → head launches a different program than workers at the same launch slot →
> `scheckne` (TensorCoreSequencer). **FIXED #1 (`fb54237b`): RoPE freqs precompute** — was eager jnp
> under `set_mesh(mesh)` → 16-partition `jit_iota/outer/exp`; now NUMPY (host), returned uncommitted
> (NOT device_put-cpu — create_jit_model's jit mesh rejects a CPU-committed array). Cleared the freqs
> scheckne + a device-mismatch; load now progresses into real TPU compile (head 3→113 modules).
> **CURRENT BLOCKER: the SAME race one step deeper** — head-only divergent modules are now
> `jit_create_jit_model` + `jit__threefry_fold_in` + `jit_add` (the RNG `nnx.Rngs(...).params()` at
> `tpu_runner.py:581` + create_jit_model). RNG data is seed-only/host-identical → PURE timing race,
> no barrier exists. **NEXT = insert a `sync_global_devices` barrier BEFORE create_jit_model** (see
> ROADMAP). Last smoke `logs/full-slice-v4-smoke-20260529T040718Z.log`; each smoke crashes ~90s (cheap).

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

## ⚠️ CURRENT BLOCKER — the SAME driver-races-workers scheckne, one step deeper
Smoke `040718Z`: load completes (`placed=68812`), progresses into TPU compile, then `Core halted …
scheckne` at `TensorCoreSequencer` (tpu17). HLO-diff: workers (identical to each other) show only
weight-fill `broadcast_in_dim`+convert+`threefry_seed`; **HEAD-only divergent modules = `jit_create_jit_model`
+ `jit__threefry_fold_in` + `jit_add`**. Agent-confirmed root cause:
- `jit__threefry_fold_in`+`jit_add` = `nnx.Rngs(jax.random.key(seed)).params()` at **`tpu_runner.py:581`**
  (`fold_in(key,count)` + `count+=1`), run right after `get_model` returns, NO active mesh.
- `jit_create_jit_model` = the `@nnx.jit` at `model_loader.py:126`, called at `:274` INSIDE `get_model`.
- RNG is **seed-only (=0), host-identical** → NOT host-divergent data; it's a **pure TIMING race** + there
  is **NO cross-host barrier** anywhere in load/init (`sync_global_devices`/`multihost_utils` = 0 repo hits).
  The driver (in-engine process, no Ray-RPC latency) finishes load first and launches these 16-partition
  programs while the 3 worker actors still drain `broadcast_in_dim`.

**ROADMAP / NEXT ACTIONS (do in order):**
1. **Insert a cross-host barrier BEFORE `create_jit_model`** so all 4 hosts finish load before any
   launches a post-load 16-partition program. Use `from jax.experimental import multihost_utils` +
   `multihost_utils.sync_global_devices("v4_post_load")`. ⚠️ **Placement matters:** `create_jit_model`
   runs at `model_loader.py:274` INSIDE `get_model` — BEFORE `tpu_runner.py:581` — and workers never
   compiled it (they died first), so it IS part of the race. A barrier at :581 (the agent's first
   suggestion) is TOO LATE. Put it **right before `create_jit_model` (`model_loader.py` :274)** OR, to
   stay V4-focused, at the **end of V4's `load_weights`** (deepseek_v4.py; model_loader calls
   `model.load_weights(rng)` :273 then `create_jit_model` :274 — so end-of-load_weights == pre-jit).
   One barrier there should serialize create_jit_model + the :581 fold_in/add in one shot (they run
   lockstep once the hosts are synced), just as the numpy fix cleared all the freqs ops at once.
2. **Validate:** CPU oracle (`s1_cpu_repro both`) is a single-host no-op for the barrier but confirms no
   import/break; then smoke WITH `V4_XLA_FLAGS=--xla_dump_to=/tmp/hlo_dump`. Past ~120s with no
   scheckne = barrier worked → into the long cold forward compile (10-30 min) → watch for
   `Application startup complete`.
3. **If the barrier is insufficient / a new scheckne appears:** re-run the HLO-diff (helper below) to
   find the next head-only divergent module and repeat. Fallback for the RNG specifically: take
   `nnx.Rngs(...).params()` (tpu_runner:581) fold_in/add off TPU (host/CPU) like the freqs fix.
   ⚠️ Small risk the barrier collective itself diverges if hosts reach it at wildly different slots —
   low (it's the intended rendezvous), but if so, move it earlier (before load_weights too).
4. After it serves: **GATE** below (establish the v6e-16 baseline md5). Likely a long cold compile the
   first time it gets past init — budget for it (`VLLM_ENGINE_READY_TIMEOUT_S=2400` already set).

HLO-diff helper (proven this session):
```
for ip in 8 17 16; do rsync -az --include='*.before_optimizations.txt' --exclude='*' \
  -e "ssh -i ~/.ssh/google_compute_engine" enyouki@10.164.0.$ip:/tmp/hlo_dump/ /tmp/hlo_cmp/h$ip/; done
# head (15) is local: cp /tmp/hlo_dump/*.before_optimizations.txt /tmp/hlo_cmp/h15/
# then per host: ls .../*.before_optimizations.txt | sed -E 's#.*/module_[0-9]+\.##;s#\.cl_[0-9]+.*##' | sort|uniq -c
# divergent count/signature between h15 and h8/h17/h16 = the culprit op.
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
   `V4_XLA_FLAGS=--xla_dump_to=/tmp/hlo_dump`, then per-host `namelist` opname-multiset diff
   (head h15 vs workers) → head-only module names = the divergent eager program.

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

# claude-deepseek-v4 — S1 runbook (blank-slate)

> Trimmed to **S1 only** and deliberately de-anchored: the root cause
> is treated as **open**. Full history/backlog is in `CLAUDE.full.md`
> — read it only if you need the rest of the project.

## Goal

Make `vllm serve deepseek-ai/DeepSeek-V4-Flash` produce coherent,
deterministic decode output on the **v6e-32 TPU slice** (TP=32,
8 hosts × 4 chips). Right now decode collapses into a degenerate
attractor after the first token or two. That bug is S1. Fixing it is
the whole job.

## The bug (precise)

Decode's **first token is correct**, then output falls off the prose
manifold within 2-3 tokens into a repeating/numeric attractor. It is
also **non-deterministic** from byte-identical input at temperature=0.

```
"Tell me a short story about a robot exploring Mars:"  (max_tokens=64, temp=0)
  → " 0.0 0.0 0.0 0.0 0.0 …"

"The capital of France is"  (3 runs, temp=0, NOT byte-identical)
  → " Paris, 2014-2015, 2016-2017"
  → " Paris, Paris, 巴黎，法国，..."
  → " Paris, Paris, Paris, Paris, Paris,"
```

Tiny-config and CPU tests pass — this only reproduces on the real
model on TPU.

## Success gate (the only definition of done)

Verified **twice** on a fresh real-V4 engine:
* `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` exits 0
  — `visible_words >= 10` AND `max_word_run < 5`.
* 3 Paris probes at temp=0 are **byte-identical** (determinism).
* Still passes after 5 unrelated requests.

The basic "PASS: … contains 'Paris'" line is a **false positive** —
`"capital of France"` can hit EOS at token 1, so no decode steps run.
Don't trust it, and don't trust `usage.completion_tokens`,
`max_word_run`, or `ends_clean` — all read healthy on corrupted
output. Read the actual response text.

## First action — test the untested fix

There is an **untested candidate fix already in the tree** (commit
`1f212036`, parent of current HEAD `96aa41d1`):
`_v4_anchor_output_buffers` in
`models/jax/deepseek_v4.py:772` wraps each output packed buffer in
`lax.optimization_barrier` before `deepseek_v4_run_with_decode_state`
returns. It is CPU-validated but **never run on TPU**.

So the cheapest possible win is to just run the TPU smoke as-is and
see if it already passes:

```bash
scripts/full_slice_v4_reset.sh        # cluster cleanup
scripts/full_slice_v4_sync.sh         # MANDATORY after any code edit
scripts/full_slice_v4_smoke.sh        # launch vllm serve (background)
LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh
```

If it passes the success gate → done. If not → the root cause is
**open** (see below); this fix becomes another tried-and-failed data
point, and you diagnose from scratch.

## If it fails — root cause is OPEN

Do not assume any prior theory is correct. **Diagnose before
patching.** Set `V4_DECODE_NAN_TRIPWIRE=1` on the smoke to print
per-field nan/inf/max_abs reductions per layer and find *which*
state field actually drifts and *when*, before changing code.

**Leading hypothesis (UNCONFIRMED — challenge it):** V4 writes its
KV-cache via pure JAX `at[].set` partial writes on a manually-packed
fp32 buffer (sites in `attention_decode_step`,
`compressor_decode_step`, `indexer_decode_step`, and prefill seed
`_compressor_state_from_prefill`), then re-packs via
`_pack_layer_state`. Under `donate_argnums=2`, XLA alias analysis
*might* rewrite these as in-place donation-slot updates and drop the
ones whose only consumer is the next decode call (compressor/indexer
state). V3/Qwen3 avoid this because their KV writes go through Pallas
kernels (opaque to alias analysis). This is a guess that has produced
no working fix yet — verify or discard it with the tripwire before
building on it.

**Already tried → result (don't burn a TPU smoke re-treading these):**
- `1f212036` — output-side `optimization_barrier` (Option C). **Untested on TPU — that's step 1 above.**
- `75b92f4b` — input-side opaque copy (`k + barrier(0.0)`). XLA folded the identity and rewrote in-place anyway.
- `5c9d9213` — fully un-donate kv_caches for V4. TPU UserFatal.
- `14e11136` — host callbacks at entry × 6 fields. NaN returns.
- `98b0a677` — single callback at kv_cache_post_write. NaN returns.
- `ac8d2077` — single per-layer callback in transformer body. Insufficient.

Net of the above: callbacks suppress NaN but don't stop the drift and
break determinism; input-side and un-donation tricks don't survive
XLA. Untried directions if the tripwire confirms a dropped write:
fold the state writes into a `pl.pallas_call` whose output *is* the
new state (matches V3/Qwen3); or partial un-donation for prefill JIT
only.

## How to validate (fastest signal first)

The full smoke is 25-45 min — **do not use it as your inner loop.**
Vet each hypothesis cheaply, then spend at most 1-2 smokes per session.

1. **CPU repros (~seconds-minutes, no TPU):**
   ```
   PYTHONPATH=work/tpu-inference:work/vllm \
     work/vllm_env/bin/python3.11 scripts/s1_cpu_repro_tiny.py both 8 8
   ```
   * `scripts/s1_cpu_repro_tiny.py` — tiny config (~5s jit).
   * `scripts/s1_cpu_repro_v4flash.py` — V4-Flash truncated to 4
     layers/8 experts, full real dims (~85s prefill / ~80s decode).
   * `scripts/s1_cpu_hlo_check.py` — counts barriers + checks all
     dynamic-update-slice writes survive in compiled HLO.

   Both repros should end "OK: both eager and jit match fresh-prefill
   argmax". **Caveat: CPU already passes — it can't reproduce the
   bug.** Use CPU only to rule out regressions / inspect HLO, not as
   proof a fix works.
2. Tiny-fixture pytest classes in `tests/models/jax/test_deepseek_v4.py` (~30s-2min).
3. `eval_shape` / `lower(...).compile()` on real config under virtual
   mesh: `XLA_FLAGS=--xla_force_host_platform_device_count=32 JAX_PLATFORMS=cpu`.
4. Real TPU smoke — the only test that actually reproduces S1.

**Smoke phase budgets (silence is expected, don't bail early):**
weight load ~4 min; `jit_run_model` cold compile **10-30 min** (warm
~97s); first curl sub-second after compile. Bail only on 3+
`slow_operation_alarm.cc`, `RESOURCE_EXHAUSTED`, or >2 min with zero
log activity during load.

## Plumbing (read before touching)

* `models/jax/deepseek_v4.py::deepseek_v4_run_with_decode_state` — decode entry; the candidate barrier is here.
* `layers/jax/attention/deepseek_v4_attention.py::attention_init_state_from_prefill` — suspected first-corruption site.
* `layers/jax/attention/deepseek_v4_attention.py` — `attention_decode_step`, `compressor_decode_step`, `indexer_decode_step`, `_compressor_state_from_prefill`, `_pack_layer_state` (the `at[].set` write sites).
* `models/common/model_loader.py:332+` — `donate_argnums=2`, V4 `kv_cache_sharding=P()`.
* `runner/kv_cache_manager.py::_initialize_kv_cache_deepseek_v4`, `runner/tpu_runner.py::_maybe_set_v4_decode_start_pos` — the only two V4 runtime hooks; keep them that small.

## Pitfalls (these cost real time)

1. **`scripts/full_slice_v4_sync.sh` after EVERY code edit** — each of
   the 8 hosts has its own clone; `git push` does NOT sync them. Skip
   it and 7/8 workers run stale code (30+ min lost). Syncs
   `work/tpu-inference/tpu_inference/` + `scripts/` only.
2. **Shut down only via `scripts/full_slice_v4_reset.sh`** — never a
   broad `pkill -f` (it matches raylet command lines and kills the
   daemon, losing nodes). Escalate to `full_slice_v4_ray_restart.sh`
   only if reset fails.
3. **`/tmp/libtpu_lockfile` survives SIGKILL** → next init SIGSEGVs.
   reset.sh handles it.
4. **First inference is slow** (5-15 min cold compile); smoke_check
   curl defaults to 900s. Not a hang.
5. **No unverified XLA flags** — smoke.sh ignores parent-shell
   `XLA_FLAGS`; opt in via `V4_XLA_FLAGS`, and validate any flag with
   `python -c "import jax; jax.devices()"` first.
6. **`--enforce-eager` does NOT skip XLA compile** — the TPU forward
   is JAX and always jit-compiles.

## Slice bootstrap — DO THIS FIRST (fresh VM, not yet bootstrapped)

This is a **freshly-provisioned slice**. Current state:
* **Head** (this host, worker 0): repo cloned, `work/{tpu-inference,vllm}`
  present, `.env` created. **No venv yet** (`work/vllm_env` missing).
* **Workers 1-7**: not set up (no repo/venv). **Ray is not running.**

**IPs are auto-discovered from GCP metadata — nothing is hardcoded.**
Inspect with:
```
scripts/full_slice_v4_discover.sh        # slice=… HEAD_IP=… WORKERS=…
```
This slice: `v6spoteu721`, v6e-32, zone `europe-west4-a`, project
`prm-research`. Head `10.164.0.192` (worker 0); workers
`10.164.0.{194,202,204,193,198,195,200}`. Every `full_slice_v4_*.sh`
defaults `HEAD_IP`/`WORKERS` to this discovery, so the scripts work on
any slice without edits (override by exporting `HEAD_IP`/`WORKERS`).

Bring the cluster up from the head:
```
./scripts/full_slice_v4_bootstrap.sh     # setup.sh + fan-out venv + GCS mount + ray (~10-15 min)
ray status                               # expect 8 nodes / 0.0/32.0 TPU when idle
```
**Prereqs are already satisfied on this slice** (done during setup):
`uv` is installed on the head + all 7 workers (symlinked into
`/usr/local/bin` so `setup.sh`'s `command -v uv` passes), `git`+`gcsfuse`
present everywhere, and SSH head→workers works via
`~/.ssh/google_compute_engine` (generated + propagated by `gcloud compute
tpus tpu-vm ssh v6spoteu721 --zone=europe-west4-a`). So `bootstrap.sh`
should run clean. Weights load **auth-free** from the GCS mount
(`personal-mark-eu/vllm/hub`); `HF_TOKEN` is intentionally unset. NB:
bootstrap's worker-setup loop is **serial** (~5 min/worker → ~30-40 min
total); fine to let it run, or parallelize the loop to speed it up. Full
detail in `CLAUDE.full.md`.

## Discipline

Keep the diff against upstream `tpu-inference` minimal — every line
gets rsync'd to 8 hosts. No new files when an existing one fits. V4
source should read indistinguishable from `qwen3.py` /
`deepseek_v3.py`. Per-iteration narrative goes in **commit messages**,
not this file. Commit a checkpoint after every validated step so a
fresh context can resume.

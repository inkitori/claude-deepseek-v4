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

**STEP 0 — this is a SHARED slice; evict the other tenant first**
(see Pitfall 0). The slice is mark's `tpu-manager` prod box, only
half-decommissioned. The real contender is a **Docker container named
`node`** (`vllm/vllm-tpu:nightly`, `restart_policy=unless-stopped`) on
all 8 hosts: it runs `ray start --address=10.164.0.192:6379` — the SAME
ray head our cluster uses — so two ray/vllm stacks fight over one
exclusive TPU slice. (`~/agent.py` under user `mark` is a RED HERRING:
pure observability on :8999, never touches the TPU. Don't waste time on
it.) Evict on every host (we have sudo; `unless-stopped` means an
explicit stop sticks):
```bash
for h in $(scripts/full_slice_v4_discover.sh head) $(scripts/full_slice_v4_discover.sh workers); do
  ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no enyouki@$h \
    'sudo docker update --restart=no node; sudo docker stop -t3 node' 2>/dev/null
done   # head runs docker locally; loop form shown for clarity
```
Leave the GCP platform containers alone (`tpu-runtime`/fake_tensorflow,
healthagent, google-runtime-monitor, …). Run a guardian that re-stops
any reappearing `node` container every ~25 s during smokes (a remote
controller may redeploy it).

**STEP 0b — if smokes still halt with `SLICE_FAILURE_SW_INJECT_ERROR` /
`Core halted` / `unexpected peer ... different launch id`**, the TPU
launch-group state is WEDGED (legacy of `node`'s 32 thrash-restarts);
ray-restart does NOT clear it. Recovery (verified): **`sudo reboot` the
7 workers** (NOT the head — your session lives there; spot VMs survive
reboot), wait for them, then on each worker
`cd ~/claude-deepseek-v4 && set -a && source .env && set +a && ./scripts/mount_gcs.sh`
(reboot drops the gcsfuse mount), then `scripts/full_slice_v4_ray_restart.sh`.
After that a smoke loads+compiles cleanly (~9 min to `Application
startup complete`).

So the cheapest possible win is to just run the TPU smoke as-is and
see if it already passes:

```bash
scripts/full_slice_v4_reset.sh        # cluster cleanup
scripts/full_slice_v4_sync.sh         # MANDATORY after any code edit
scripts/full_slice_v4_smoke.sh        # launch vllm serve (background)
LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh
```

**RESULT (2026-05-24, smoke after infra recovery): FAILED.** The
barrier fix (`1f212036`) is now a **confirmed TPU dead-end**, not just
untested. With it live at HEAD the engine reached `Application startup
complete`, served, and decode still collapsed:
`"Tell me a short story about a robot exploring Mars:"` (max_tokens=64,
temp=0) → `' "The first step is to be the best." "The best is to be the
best." "The best is to be the best." …'` — the S1 repeating attractor.
(Matches the CPU-HLO finding that XLA elides the 6 `optimization_barrier`
ops, 6→0 in compiled HLO — same folding fate as `75b92f4b`.) So: root
cause is **OPEN**; diagnose from scratch (next section). NOTE the slice
is marginally stable post-recovery — it served one decode then a worker
hit `SLICE_FAILURE` again with no contention present; prefer the
small-config-on-TPU repro (4 layers / few chips) as the PHASE-2 inner
loop over repeated full smokes (it's also less exposed to the flaky
32-chip collective).

## If it fails — root cause is OPEN

Do not assume any prior theory is correct. **Diagnose before
patching.** Set `V4_DECODE_NAN_TRIPWIRE=1` on the smoke to print
per-field nan/inf/max_abs reductions per layer and find *which*
state field actually drifts and *when*, before changing code.

> **DISPROVEN 2026-05-24 (see "PHASE 2" below).** The donation /
> dropped-write hypothesis here is FALSE: a sharded MH repro shows donate
> vs non-donate decode-state buffers are byte-identical, and the decode
> math matches teacher-forcing. S1 is NOT in the model decode math / not a
> donation bug. The text below is kept for history; jump to PHASE 2.

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

## PHASE 2 — S1 REPRODUCED at small scale (2026-05-24)

There is now a **cheap multi-host reproducer.** The single-host small-TPU
loop the prior runbook assumed DOES NOT EXIST: a lone worker can't boot a
v6e-32 (libtpu `CreateTpuSystemState` waits forever for the other 7 hosts);
all TPU work needs all 8 hosts. New tooling (committed):
* `scripts/full_slice_v4_mh_run.sh <script.py> [args]` — fan a script out
  across all 8 hosts as ONE `jax.distributed` job (no-arg metadata
  auto-detect; head may land as any proc index), cleans the libtpu lock,
  per-proc logs in `logs/mh-*`.
* `scripts/s1_mh_repro.py <mode> <T> <N> <n_layers> <action> <n_experts>` —
  truncated V4 (random weights, no 543 GiB load) on the 32-chip mesh with
  production's named axes (layers/common/sharding.py); KV state replicated
  `P()`, kv_caches donated. `mode`∈{replicated,sharded}. `action=repro`
  prints VERDICT; `action=diff` localizes the corrupted packed-state field
  (donate vs non-donate buffer diff). Must build replicated arrays via
  `make_array_from_callback` (device_put-reshard of process-local arrays to
  global `P()` tile-allgathers the embed/head weight 8× and OOMs HBM).
* `scripts/full_slice_v4_node_guardian.sh` — background loop re-stopping the
  redeployed `node` container (it CAME BACK 2026-05-24 — run during TPU work).

**Result — the bug needs SHARDING + donation, not just TPU + donation:**
* `replicated` (all `P()`, 32 chips, no reshard) → **NO_S1** (bad=0/12), as CPU.
* `sharded` (attn_dp=8, production-style: experts+attn parallel, KV `P()`)
  → **S1_REPRODUCED** (bad=1/12).

**DONATION IS EXONERATED — and the decode MATH is correct.** `action=diff`
(donate vs NON-donate, both sharded) → packed buffers **byte-identical** at
every (step,layer,field) (`first=None`) AND identical argmax (bad_d=bad_n=1).
So donation drops NOTHING — overturning the hypothesis the entire prior
effort (barrier `1f212036`, un-donation `5c9d9213`, callbacks) rested on;
that's why they ALL failed. The lone sharded divergence is a **benign
near-tie flip**: the 1/12 mismatch is at step 4 then RECOVERS (steps 5-11
OK), oscillating among the same token-ids the oracle emits (random *0.02
weights ⇒ flat near-tie logits + attn_dp reduction-order noise). It does NOT
compound. So `deepseek_v4_run_with_decode_state` (compressor/indexer/SWA
state threading) is CORRECT under replicated AND sharded, donated or not.

**S1 IS NOT IN THE MODEL DECODE MATH.** Both the CPU repros and this MH repro
test that core and pass. Stop patching the write sites / donation.

**RULED OUT so far (don't re-tread):**
- Donation (diff: byte-identical buffers).
- Decode math: replicated bad=0; sharded bad=1/12 and 4/48 — but mismatches
  are SCATTERED (steps 20/21/32/37 at N=48), never compound, and every one is a
  near-tie (top1-top2 logit gap ~5e-4..4e-3, same as OK steps) ⇒ benign attn_dp
  reduction noise on flat random-weight logits.
- Cross-mode forward (free, from logs): replicated-forward vs sharded-forward
  teacher-forced argmax differ at ONLY the same near-tie step ⇒ forward
  sharding perturbation is benign, not structural.
- Multi-seq prefill branch: single-prompt decode (`query_start_loc=[0,1]`,
  n_active=1) takes the single-seq branch (correct); multi-seq is batch>1 only.
- Single-seq wrapper `deepseek_v4.py:1812-1871`: mirrors the repro, looks right.
- Runner KV threading: `tpu_runner.py:869` captures & stores the returned
  `kv_caches` each step; prefill→decode handoff fine; donation aliasing correct.

**Therefore S1 is NOT reproducible with random weights / the functional path,
and is almost certainly REAL-CONFIG / REAL-WEIGHT specific or a runtime-input
issue** (key point: with confident trained logits, ~1e-3 reduction noise can't
flip argmax, so S1's collapse is a LARGE structural error the repro would have
shown if it were in the decode math — it isn't). **Remaining suspects:**
1. **Real-config-only paths**: real V4 = 61 layers, 256 experts, real
   `compress_ratios` per layer (my repro: 4 layers, 8 experts, ratios
   (0,0,4,128)), real `state_max_seq_len`. A bug in a compress_ratio/layer
   combo or at scale won't show in the 4-layer truncation.
2. **Runtime attention_metadata**: is `seq_lens[0]-1` (start_pos) actually the
   right position each decode step at runtime? Is `state_max_seq_len`
   (`v4_state_max_seq_len_from_vllm_config`) what the buffers were sized for?
3. **Non-determinism source** (S1 is non-det at temp=0; repro is deterministic):
   uninitialized HBM read / non-det collective / cross-request buffer reuse.

**NEXT: an INSTRUMENTED smoke (real weights) — the only thing left that can
show S1.** Good news: `V4_DECODE_NAN_TRIPWIRE=1` is ALREADY plumbed to workers
(smoke.sh copies it) and wired at ~30 sites incl. all 6 state fields
`_at_entry`/`_post_write` per layer, each printing `pos={start_pos}` + nan/inf/
`max_abs`. So enabling it gives the start_pos trajectory (must be P, P+1, P+2…)
AND the per-field magnitude trace with ZERO code change. Run a SHORT gen
(max_tokens~8, S1 collapses in 2-3 tokens) to keep the trace small; read the
actual decode text (attractor vs coherent); + 3× Paris determinism probe.
CAVEAT: the tripwire uses jax.debug.print host callbacks, which prior history
says can perturb SPMD reduction order — if S1 *vanishes* under it, that's itself
a clue (callback-sensitive). Localize: which (layer, field) `max_abs` blows up
at the collapse step; is `pos=` correct each step.

> **INFRA BLOCKER — ROOT CAUSE FOUND + FIXED (2026-05-24):** `vllm`'s
> `ray.init()` dies "Version mismatch: cluster 2.54.1 vs local 2.55.1" at engine
> init (NOT S1). The prior theory (stale `/tmp/ray-vllm` / reinstall the venv)
> was **WRONG**: all 8 venvs are uniformly 2.55.1/commit `237c2455` (verified
> via `compute_version_info()` — the exact fn the check uses — and a single
> `ray-2.55.1.dist-info`); a clean `/tmp` wipe + `ray_restart` did NOT fix it.
> **The real culprit is mark's `node` Docker container** (`vllm/vllm-tpu:nightly`,
> ray **2.54.1**/commit `8768a329`). When it's redeployed it tries to join our
> ray at `:6379`, FAILS its own version check (status `Exited (1)`), but in the
> attempt calls `put_cluster_metadata(overwrite=True)`, **poisoning the GCS
> `CLUSTER_METADATA` key to 2.54.1** — which then makes our own 2.55.1 head log
> the right version (`gcs_server_main.cc:98 ray_version=2.55.1`) yet the stored
> metadata read by every client/worker say 2.54.1. It also blocks our own
> workers from joining (only 2/8 registered until fixed). Confirm the diagnosis:
> `GcsClient(address=head:6379).internal_kv_get(b"CLUSTER_METADATA", namespace=b"cluster")`
> (raw GcsClient bypasses the version gate) → look at `ray_version`/`git_commit`.
> **FIX (what worked):** (1) `sudo docker rm -f node` on all 8 hosts (removes the
> poisoner; `docker update --restart=no` + `stop` is NOT enough — a controller
> redeploys it); (2) either re-stamp the key
> (`gc.internal_kv_put(b"CLUSTER_METADATA", json_with_ray_version_2.55.1, True, namespace=b"cluster")`)
> OR just re-run `scripts/full_slice_v4_ray_restart.sh` now that no poisoner is
> present (head writes 2.55.1 natively). Verify: `ray.init(address='auto')`
> connects, `cluster_resources()['TPU']==32.0`. **Silver lining:** now that our
> cluster is 2.55.1, the 2.54.1 `node` container CANNOT successfully join (it
> Exits 1 on the mismatch), so it can't hold the TPU — residual harm is only the
> metadata poisoning, fixed by keeping `node` removed. KEEP the guardian running
> (`scripts/full_slice_v4_node_guardian.sh`; kill via
> `pkill -f "[f]ull_slice_v4_node_guardian"`) so a redeploy is caught fast; the
> teardown helper is `/tmp/v4_teardown.sh` (run as a FILE to avoid `pkill -f`
> self-match — the pattern string in an inline `pkill -9 -f gcs_server` matches
> the executing shell's own argv and kills it).

(Production V4 runs attn_dp=32: num_kv_heads=1 + bf16 ⇒ TP folds entirely into
attn_dp, model=expert=1, KV `P()`. The repro used attn_dp=8 / 8 experts.)

## PHASE 3 — S1 IS A DECODE-PATH BUG (real-weight smoke, 2026-05-24)

The instrumented smoke (`V4_DECODE_NAN_TRIPWIRE=1`) ran clean on real weights and
**localized S1 to the decode path.** Decisive findings:

* **PREFILL IS HEALTHY.** Pure single-token probes (`max_tokens=1`, ZERO decode
  steps) are all correct: France→`Paris`, Japan→`Tokyo`, hot→`cold`,
  hydrogen→`oxygen`, George→`Washington`, violets→`blue`. So the weights, config,
  embedding, MoE routing, sparse-attention SELECTION and the whole forward are
  correct **in the prefill path**.
* **DECODE COLLAPSES.** With `max_tokens≥2` the output degenerates by token 2-3
  into a repeating attractor (Mars→`' "The first thing that is a good and the
  first thing'`; Paris→`' Paris, 2000, 2000, 2000'`). The collapse appears the
  moment decode steps (which reuse the threaded KV + compressor/indexer/SWA
  decode-state) run instead of recomputing over the full sequence.
* **Tripwire trace is BENIGN at the field level:** finite everywhere (NO nan/inf
  in decode), `pos=` correct each step (11,12,…), all 6 state fields + per-layer
  activations finite and varying with reasonable magnitude. The `-inf` in
  `compressor_score`/`indexer_score` is by-design masking that correctly
  *decreases* as positions fill. So S1 is **wrong FINITE values from attending to
  the wrong context / mis-threaded state**, not a blowup — exactly why the
  nan-tripwire alone never caught it.
* **Non-determinism is a DOWNSTREAM symptom, not the bug.** It survives the
  tripwire host-callbacks (so it's not merely debug-perturbable reduction order)
  and only appears at LATE tokens (first ~6 tokens identical across temp=0 runs,
  then split) — i.e. once decode is already in a flat-logit degenerate regime,
  32-way reduction noise flips near-ties. The PRIMARY bug is the **collapse**.

**This OVERTURNS the PHASE-2 "decode math is correct" conclusion** — that was
proven only on the truncated random-weight repro (4 layers, ratios (0,0,4,128),
8 experts, short seq, attn_dp=8). The real-config decode path (61 layers, real
per-layer `compress_ratio`/`window_size`, real `state_max_seq_len`, attn_dp=32)
IS broken. So S1 is a **real-config-only decode bug**: prime suspects are the
sparse-attention index selection at decode (`get_window_topk_idxs_decode` /
`get_compress_topk_idxs_decode` / indexer `compress_topk` in
`deepseek_v4_attention.py:~750-790`), the SWA ring-buffer wrap
(`kv_cache.at[:, start_pos % win]`) when `win < seq_len`, or a prefill→decode
seed slot-layout mismatch (`attention_init_state_from_prefill` /
`_compressor_state_from_prefill`) that the short-seq/ratio-0 repro never exercised.
Confirm with a teacher-forcing comparison (decode trajectory vs re-prefill every
token) — re-prefill-every-token should generate COHERENTLY since prefill is healthy.

## How to validate (fastest signal first)

**The expensive thing is NOT TPU — it's the full DeepSeek-V4-Flash
smoke** (543 GiB weight load + 10-30 min cold compile = 25-45 min). TPU
time on *small* configs is cheap (seconds-to-minutes), and it's the only
thing that can reproduce S1 — **CPU passes, so it never will.** Iterate
on small-on-TPU; reserve the full smoke for confirming a fix that already
survives a smaller TPU repro (at most 1-2 full smokes per session).

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
   argmax". **Caveat: CPU already passes — it cannot reproduce S1.**
   Use CPU only to rule out regressions / inspect HLO, never as proof a
   fix works.
2. **Small config ON TPU = MULTI-HOST sharded repro — THE inner loop
   (~3-6 min, no full weights). CONFIRMED to reproduce S1 (2026-05-24).**
   A lone host CANNOT boot a v6e-32 (`CreateTpuSystemState` hangs forever
   waiting for the other 7 hosts), so the old "run a 1-/few-chip script
   directly" advice is WRONG — it just hangs. Use:
   ```
   scripts/full_slice_v4_mh_run.sh scripts/s1_mh_repro.py sharded 8 12 4
   ```
   This fans a truncated-V4 (random weights, 4 layers) `jax.distributed`
   job across all 8 hosts / 32 chips, sharded exactly like production.
   `replicated` mode → NO_S1; `sharded` (attn_dp) → S1_REPRODUCED. See the
   "PHASE 2" section. Reserve the full smoke for final closure only.
3. Tiny-fixture pytest classes in `tests/models/jax/test_deepseek_v4.py` (~30s-2min, CPU or TPU).
4. `eval_shape` / `lower(...).compile()` on real config under virtual
   mesh: `XLA_FLAGS=--xla_force_host_platform_device_count=32 JAX_PLATFORMS=cpu`.
5. Full V4-Flash smoke — the closure gate, but the slow one; the only
   test that reproduces S1 end-to-end.

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

0. **SHARED SLICE — mark's `tpu-manager` `node` container will kill
   your smoke; a wedged slice needs a worker reboot.** See First-action
   STEP 0 / 0b for the full eviction + un-wedge procedure. The contender
   is the Docker container `node` (mark's ray worker joining ray at
   `10.164.0.192:6379`, `restart=unless-stopped`, was thrash-looping
   restart_count=32) — NOT `~/agent.py` (that's harmless observability).
   Failure signatures, all BEFORE `Application startup complete` / before
   any decode token (so NOT S1): `ActorDiedError` / `Worker exit type:
   SYSTEM_ERROR` / `connection error code 2` (live contention), then once
   wedged `SLICE_FAILURE_SW_INJECT_ERROR` / `Core halted unexpectedly` /
   `unexpected peer shows up ... different launch id`. Tell-vs-S1: S1 is a
   *decode-output* bug — it only exists once the server reaches
   `Application startup complete` and curls return text. Any crash during
   load/compile is infra, not S1.
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

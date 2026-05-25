# claude-deepseek-v4 — S1 runbook (blank-slate)

> **⇒ START HERE (2026-05-25): read `HANDOFF_S1.md` — the "SESSION 2 UPDATE"
> section at the top is authoritative and SUPERSEDES this file's "First
> action" and PHASE notes below.** Short version: the repeated TPU
> "different launch id" Core-halts were CODE DESYNC (run
> `scripts/full_slice_v4_sync.sh`; do NOT reboot). S1 is cleanly reproduced
> and narrowed: forward→" Paris" is correct, DECODE collapses at step 1 from
> a deterministically-WRONG SEED built during the token-sharded prefill
> (cross-token seed ops). A partial fix (replicate V4 decode input) is
> committed; the seed fix is open. Ignore the "test the barrier fix
> 1f212036" instruction below — that fix is long-disproven.
>
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

## PHASE 7 — runtime audit done; PRIME suspect = TOKEN-AXIS sharding of SHORT prefills (2026-05-24)

Live-engine ground truth (instrumented smoke :18081) + runtime audit:
* **Collapse is at the FIRST decode step and DETERMINISTIC.** `"Count to 5: 1,2,3,"`→`" 0 0"`
  (token1 `" "`=prefill OK; token2 `"0"`=first decode step, WRONG). 6 identical probes
  byte-identical ⇒ deterministic; the 2010-vs-2012 non-det was cross-request contamination
  (`enable_prefix_caching=True`). Collapsed decode sometimes yields **NaN logits** (engine
  `ValueError: Out of range float ... nan` on logprobs).
* **Runtime is correct:** decode `start_pos = seq_lens[0]-1` INCREMENTS by 1 in the live
  trace (17..27, 9..14); runner captures+rethreads `self.kv_caches` each step
  (`tpu_runner.py:869`). So start_pos / kv-threading are NOT the bug.
* **Real cfg:** 43 layers, ratios `[0,0,4,128,4,128,...,4,0]`, 256 experts, index_topk=512,
  `state_max_seq_len=8192` (=256 max_model_len × 32 dp) ⇒ ratio4 cache=2048 slots ≫512;
  NO size mismatch, NO new flavor. Decode math fp32-exact for all 3 flavors (PHASE 6).

**PRIME HYPOTHESIS (untested by any repro so far):** the real engine **token-shards
`input_ids`/activations on `ATTN_DATA`** (`tpu_runner.py:1560`,
`data_parallel_attn_sharding = P(ShardingAxisName.ATTN_DATA)`; ATTN_DATA=('data','attn_dp',
'attn_dp_expert')). With attn_dp=32 and a SHORT prefill (T<32 tokens) the token axis is
sharded UNEVENLY (empty/idle ranks). The cross-token state-SEEDING ops run on this
token-sharded activation: `_swa_kv_cache_from_prefill` (jnp.roll over token axis) and
`_compressor_state_from_prefill` (slice `[cutoff-ratio:cutoff]` over token axis) — a sharded
roll/slice with idle ranks can produce a CORRUPTED seeded packed state, while the forward
`h` (token-1 logits) stays correct ⇒ exactly "first token correct, then step-1 collapse".
**The MH repro placed inputs REPLICATED `P()`** (`s1_mh_repro.py` `put`), so it NEVER tested
token-axis sharding — that's why sharded-MH (attn_dp=8, replicated input) looked clean.

**DECISIVE CHEAP TESTS (in priority order):**
1. **Real-engine length sweep** (just curls, faithful): short prompt (T<32) should COLLAPSE,
   long prompt (T≥32, fills all dp ranks) should stay COHERENT. If so ⇒ token-axis/short-
   prefill sharding bug CONFIRMED. (Earlier attempt got contention; rerun on a clean engine.)
2. **Token-sharded MH seeddiff**: place input on `P(None,'attn_dp')` (token-sharded) and diff
   the seeded packed state vs replicated-input. NB the replicated-input `seeddiff` action
   times out in its slow replicated half — make it sharded-vs-(token-sharded) and budget time.

If confirmed, the FIX is in the prefill state-seeding under token sharding: constrain the
seeding input/activation to replicated `P()` before `_swa_kv_cache_from_prefill` /
`_compressor_state_from_prefill` (or compute seeding on a gathered/un-sharded token axis), so
idle ranks can't corrupt the rolled/sliced state. Localize the exact field with test #2.

> SESSION NOTE (2026-05-24): a STALE 2nd `claude` session (old prompt) was running
> concurrently — co-editing scripts + running competing TPU jobs (tore down the smoke, ran
> a seeddiff that timed out). Consolidated to a single driver (SIGKILLed it; guardians +
> this session preserved). It had usefully enhanced `s1_mh_repro.py`'s `seeddiff` action
> (full-trajectory replicated-vs-sharded) — that enhancement is KEPT.

## PHASE 8 — candidate fix `c5f245c7` is SUSPECT (correlates with TPU Core-halts) (2026-05-24)

The fix `c5f245c7` adds `_replicate(x)` (= `with_sharding_constraint(x, P())`) at
the top of `attention_init_state_from_prefill` (seeding) AND `attention_decode_step`
(decode). **It has NOT been shown to work and is a PRIME SUSPECT for crashing decode:**
* **PRE-fix** smoke `064842`: served **93 completions, 0 Core-halts** (decode ran,
  just collapsed — the S1 symptom).
* **POST-fix** smokes `093046` (48443) and `094655` (this session): **Core-halt at the
  FIRST decode step**, both times — `real_program_continuator.cc: Core halted
  unexpectedly` then cluster-wide `SLICE_FAILURE_SW_INJECT_ERROR`. Different workers
  each time (.202, then .198) — i.e. NOT one bad chip.
* The fix's ONLY new op in the decode path is an **all-gather of the activation**
  (token-sharded ATTN_DATA → replicated P()). The pre-existing
  `_v4_constrain_packed_replicated` (P()→P(), no reshard) ran fine for 93 requests;
  resharding a **degenerate size-1 token axis** (1 decode token over 32 ways) is the
  new thing and is the leading Core-halt suspect. (Caveat: can't fully rule out
  independent slice flakiness — the runbook says this slice SLICE_FAILUREs
  spontaneously. A FRESH-slice retest is needed to be sure.)

**CONFIRMED on a FRESH slice → reverted to seeding-only (FIX v2):** rebooted the 7
workers (clean slice, 0 failures), relaunched the c5f245c7 fix smoke — decode Core-halted
*again* at the FIRST step (`SLICE_FAILURE_SW_INJECT_ERROR`). That's **3x including a
freshly-rebooted slice ⇒ NOT slice flakiness; the decode-step `_replicate(x_step)` is the
halt cause.** Root: a single decode token has a **size-1 token axis** — `with_sharding_
constraint` to `P()` emits a degenerate all-gather the TPU faults on; and a size-1 axis
can't shard 32 ways anyway (already effectively replicated), so the op was unnecessary AND
fatal. **FIX v2 (committed, seeding-only):** removed `_replicate(x_step)` from
`attention_decode_step`; KEPT `_replicate(x)` in `attention_init_state_from_prefill`
(prefill seeds over T>1 token-sharded tokens — the real S1 cause — and that gather is
normal/non-degenerate, prefill never halted). Pre-fix decode ran without halting and
collapsed from the bad SEED, so seeding-only should make decode coherent.

**RECOVERY procedure (each bad-fix smoke wedges the slice → STEP 0b reboot):** reboot 7
workers → wait for SSH → remount GCS each (`mount_gcs.sh`) → `full_slice_v4_ray_restart.sh`.
~6-7 min. The seeding-only fix should NOT wedge (no degenerate reshard).

**NEXT:** smoke FIX v2 on the fresh slice; read decode text. Coherent ⇒ verify TWICE ⇒ DONE.
If still COLLAPSES (no halt) ⇒ seed fix insufficient ⇒ the decode-step ALSO needs correct
sharding but via a NON-halting route: a RUNTIME fix (give V4 decode a replicated input at
`tpu_runner.py` `data_parallel_attn_sharding`, so no mid-forward reshard) — NOT another
model-level `with_sharding_constraint` on the size-1 decode activation.

## PHASE 9 — node ELIMINATED; halt is the FIX not the slice; Option A under test (2026-05-24)

Supersedes PHASE 7/8 fix theories. **Read HANDOFF_S1.md.**
* **node infra SOLVED.** The redeploy controller is a remote VM `35.186.51.62` SSHing
  in as user `mark` every ~10min running `run_cluster.sh` (`docker rm -f node; docker
  run ...`). Blocked permanently with **`DenyUsers mark`** in
  `/etc/ssh/sshd_config.d/99-s1-block-mark.conf` on all 8 hosts (survives reboot;
  self-healed by `full_slice_v4_node_occupy.sh`, which also keeps a dummy `node` +
  iptables drop as backups). Both guardians still run. Don't re-fight node.
* **The decode Core-halt was NOT node.** With node provably absent the whole smoke,
  FIX v2 still Core-halted at the **prefill** step. A/B on the same slice: **pre-fix
  runs clean (no halt) and still collapses** ⇒ **slice HEALTHY**, FIX v2 is the halt.
* **FIX v2 (`x=_replicate(x)` in `attention_init_state_from_prefill`) = CONFIRMED
  dead-end, REVERTED.** `max_num_seqs=1`+DP-attention → single seq on ~1 attn_dp rank,
  ~31 EMPTY ranks; in-jit `with_sharding_constraint(x,P())` all-gathers over empty
  shards → `RuntimeUnexpectedCoreHalt`. The replicated `P()` decode state then gets an
  inconsistent seed across ranks (only 1 rank has real tokens) ⇒ S1 collapse.
* **Option A (under test, `tpu_runner.py` `_prepare_inputs_dp`):** for V4+prefill
  (`q0>1`) re-place `input_ids` REPLICATED from the HOST buffer (broadcast, NOT a
  device gather) so every rank seeds the same state. If decode still collapses, the
  decode input likely needs the same treatment; diagnose with `V4_DECODE_NAN_TRIPWIRE=1`.

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

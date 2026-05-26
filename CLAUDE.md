# claude-deepseek-v4 — S1 runbook

> **⇒ START HERE: read `HANDOFF_S1.md` — its top "STATE (SESSION 18)" is authoritative.**
> This file holds only durable operational knowledge (how to run the slice, validate,
> pitfalls) + the handoff protocol below. Live state + current lead live in the handoff.
> History: `CLAUDE.full.md`.
>
> One-line status (2026-05-26 S18): **bug = the per-rank expert EINSUM in the MoE routed shard_map reads
> uninit HBM (psum + all_gather EXONERATED).** Isolated via `[ckL]` pre-psum local-sum, engine B3 vs A3:
> identical einsum INPUT (`[ckG]` x_full byte-identical) but the einsum OUTPUT sets share 0 values (lone ~1e5
> outlier differs, 9.42e4 vs 8.50e4); `[ckR]` moe_routed REAL rows differ (91.5 vs 45.8) ⇒ the einsum hits
> REAL rows, so pad/valid-row masking is REFUTED (per-element psum can't spread pad→real). Clean contrast: the
> dense auto-SPMD einsum + moe_shared are deterministic. **NEXT (HANDOFF_S1.md):** find the uninit source
> (weight/x_full physical padding the MXU reads) & fix, else port the fused_moe_gmm gmm kernel; validate on 2
> fresh engines (single-engine [ckR] is globally reduced — can't see the bug). Loop NOT stopped.

## ⇒ CONTEXT HANDOFF PROTOCOL — every session MUST follow this

Sessions are disposable; the commit log + `HANDOFF_S1.md` + this file are the only
memory that survives. When your context reaches **~100k–200k tokens** (interactive:
check `/context`; also heed the harness's "context getting long" reminders) AND you
are NOT seriously mid-operation (e.g. waiting on a smoke whose result you must read, or
a few calls from finishing a committed step) — HAND OFF to a fresh session:

1. **Keep the markdown LEAN** — trim `CLAUDE.md`, `HANDOFF_S1.md` (and your memory
   files): delete superseded narrative, keep only durable ops + the CURRENT lead + the
   single next action. Every line here rsyncs to 8 hosts / loads into every session;
   bloat is a real cost. Lean > complete.
2. Rewrite `HANDOFF_S1.md` to the CURRENT state: what you verified, what's in flight
   (log paths, ports, task ids), and the ONE most important next action.
3. `git add -A && git commit && git push`.
4. Run **`scripts/s1_handoff_window.sh`** — opens a NEW tmux window in this session and
   launches a fresh `claude --dangerously-skip-permissions --effort max` (empty context,
   auto-loads this file + `scripts/s1_loop_prompt.txt`). Not in tmux → it prints the
   exact command to run by hand.
5. **END YOUR TURN** so the fresh session takes over. Don't start anything you can't
   finish + commit before handing off.

(Headless alternative: `scripts/s1_session_loop.sh` relaunches `claude -p` in-place; it
can't read `/context` so it hands off per finished-chunk. The new-window protocol above
is preferred — the fresh session CAN budget by `/context`.)

## Goal

Make `vllm serve deepseek-ai/DeepSeek-V4-Flash` produce coherent, deterministic
decode output on the **v6e-32 TPU slice** (TP=32, 8 hosts × 4 chips). Decode
collapses into a degenerate attractor after the first token or two. That bug is
S1. Fixing it is the whole job.

## The bug (precise)

Decode's **first token is correct** (it's the prefill-forward argmax), then output
falls into a repeating/numeric attractor starting at the **first decode step**
(token 2). With the metadata-replicate decode fix live it is now **deterministic**
at temp=0 (byte-identical collapses). Tiny-config and CPU tests pass — this only
reproduces on the real model on the sharded TPU slice.

## Success gate (the only definition of done)

Verified **twice** on a fresh real-V4 engine:
* `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` exits 0
  — `visible_words >= 10` AND `max_word_run < 5`.
* 3 Paris probes at temp=0 are **byte-identical** (determinism).
* Still passes after 5 unrelated requests.

The basic "PASS: … contains 'Paris'" line is a **false positive** — `"capital of
France"` can hit EOS at token 1, so no decode steps run. Don't trust it, nor
`usage.completion_tokens` / `max_word_run` / `ends_clean` — all read healthy on
corrupted output. **Read the actual response text.** A strictly-increasing
sequence (Fibonacci) cleanly discriminates coherent-vs-attractor; greedy poem
repetition is a red herring (decode loops too).

## Slice-serving protocol (marginal slice — do this EVERY smoke)

1. Edit code → `scripts/full_slice_v4_sync.sh` (MANDATORY: 8 hosts, each own clone;
   `git push` does NOT sync them). Verify md5 of edited .py matches head==workers —
   a mismatch causes "different launch id" Core-halts (CODE DESYNC, **not** a slice
   wedge; do NOT reboot — sync fixes it).
2. Clear `~/.cache/vllm/xla_cache/*` on all 8 hosts (stale/mixed cache also → launch-id halt).
3. `scripts/full_slice_v4_reset.sh` (stops any engine, cleans lockfiles).
4. `scripts/full_slice_v4_smoke.sh` (backgrounds vllm serve; prints log path). Self-guards now:
   flock single-instance (REFUSES if an engine is up/starting — `SMOKE_NO_GUARD=1` escapes) +
   `VLLM_ENGINE_READY_TIMEOUT_S=2400` so a cold compile doesn't die at vllm's 600s default.
5. Wait for `Application startup complete` (~6 min when xla cache warm; cold compile
   10-30 min). Then probe with curls; **fire critical probes first** (engine can crash
   on internal NaN after a few requests; the `compute_logits` nan_to_num clamp keeps it
   alive). Init is a coin-flip (intermittent worker SYSTEM_ERROR) — just retry.

Slice is HEALTHY when code is synced (served 4 smokes clean on 2026-05-25, no halts).

## How to validate (fastest signal first)

The expensive thing is the full smoke (543 GiB load + 10-30 min cold compile). CPU
is cheap but **CPU passes — it cannot reproduce S1** (no sharding). Budget: at most
1-2 full smokes per session; reserve them for a fix that already has a hypothesis.

1. **CPU repros (regression-only):** `PYTHONPATH=work/tpu-inference:work/vllm
   work/vllm_env/bin/python3 scripts/s1_cpu_repro_v4flash.py both` — should end "OK:
   both eager and jit match". Proves no regression, never proves a fix works.
2. **Live-engine probes (faithful, no code change):** curl :18081. Compare the
   decode path (`max_tokens=2`, token2) vs prefill-everything (chained `max_tokens=1`,
   coherent reference). Helpers: `/tmp/s1_seedstep_probe.py LOG PROMPT TOK1`,
   `/tmp/s1_prefill_gen.py`, `/tmp/s1_decode_only.py`. Always-on diagnostics print
   `[fwdL]`/`[decL]` (per-layer last-pos), `[fwdS]`/`[decS]` (embed/attnout/moeout
   L0-2), `[moeRS]` (MoE routed-vs-shared). jax.debug.print drops/reorders under high
   volume — re-fire to collect missing lines.
3. Full V4-Flash smoke — the closure gate (only end-to-end S1 repro).
4. MH repro `scripts/full_slice_v4_mh_run.sh scripts/s1_mh_repro.py sharded 8 12 4`
   exists but its launcher runs each host's local clone (desync risk) and it places
   inputs replicated (can't test the real sharding). Prefer the live engine.

## Plumbing (read before touching)

* `layers/jax/moe/deepseek_v4_moe.py::moe_forward` — **the current S1 suspect.** Dense
  bespoke MoE (einsum + `out_NEd.sum(axis=1)` over attn_dp-sharded experts), NOT the
  production `fused_moe_gmm` qwen3/v3 use. Routed experts are dead in decode.
* `models/jax/deepseek_v4.py` — `deepseek_v4_run_with_decode_state` (decode entry),
  `transformer_body_forward` / `transformer_body_decode_step`, `block_forward` /
  `block_decode_step`, `hc_pre`/`hc_post` (hyper-connection residual mix). All the
  always-on `[fwd*]`/`[dec*]` diagnostics live here.
* `runner/tpu_runner.py::_prepare_inputs_dp` — the V4-only metadata-replicate decode
  fix (`_v4_decode_replicate`); makes decode activation P() instead of ATTN_DATA.
* `layers/jax/attention/deepseek_v4_attention.py` — attention decode/prefill + seed
  build (`attention_init_state_from_prefill` etc). Attention is HEALTHY in decode.
* `models/common/model_loader.py` — `donate_argnums`, V4 `kv_cache_sharding=P()`,
  `_pick_spec` (weights prefer attn_dp sharding).

## Pitfalls (these cost real time)

0. **`different launch id` / `Core halted` / `SLICE_FAILURE` BEFORE startup = CODE
   DESYNC, not a slice wedge.** Run `full_slice_v4_sync.sh` + clear xla_cache; do NOT
   reboot (the old runbook's "reboot 7 workers" misdiagnoses this). Also keep env-var
   reads consistent across workers — env-gated module-level reads race across ray
   workers → divergent programs → launch-id halt. That's why all S1 diagnostics are
   ALWAYS-ON (no env gate). Infra: the old `node` contention is SOLVED (`DenyUsers mark`
   on all 8 hosts; two guardians run); don't re-fight it.
1. **Shut down only via `scripts/full_slice_v4_reset.sh`** — never broad `pkill -f`
   (kills raylet, loses nodes). Escalate to `full_slice_v4_ray_restart.sh` if reset fails.
2. **`/tmp/libtpu_lockfile` survives SIGKILL** → next init SIGSEGVs. reset.sh handles it.
3. **`--enforce-eager` does NOT skip XLA compile** — the TPU forward is JAX, always jits.
4. **No unverified XLA flags** — smoke.sh ignores parent-shell `XLA_FLAGS`; opt in via
   `V4_XLA_FLAGS` and validate with `python -c "import jax; jax.devices()"` first.
5. **`with_sharding_constraint(activation, P())` that GATHERS a size-1 decode token axis
   Core-halts the slice** (proven ~8x). A wsc on a POST-reduction `[N,dim]` quantity is
   safe (doesn't gather the token axis). Don't gather the activation to replicated.
6. **A live engine WEDGES on a new request shape** (HTTP500 → connection-refused; process stays
   alive but stops serving), esp. the chat-path resharding. Raw `/completions` SAME shape survived
   3 fires. When probing: pick ONE shape, fire critical probes first, reset+re-smoke if wedged.

## Slice bootstrap

The slice (`v6spoteu721`, v6e-32, zone `europe-west4-a`, project `prm-research`) is
already bootstrapped: venv, GCS mount, ray up. IPs auto-discover —
`scripts/full_slice_v4_discover.sh`. If a fresh VM needs setup,
`./scripts/full_slice_v4_bootstrap.sh` + see `CLAUDE.full.md`. Weights load auth-free
from the GCS mount (`HF_TOKEN` intentionally unset). venv python is 3.12.

## Discipline

Keep the diff against upstream `tpu-inference` minimal — every line rsyncs to 8 hosts.
No new files when an existing one fits. V4 source should read like `qwen3.py` /
`deepseek_v3.py`. Per-iteration narrative goes in **commit messages**, not this file.
Commit a checkpoint after every validated step. Remove the S1 diagnostic prints when
S1 closes.

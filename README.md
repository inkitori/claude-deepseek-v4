# claude-deepseek-v4

Run DeepSeek-V4-Flash via `vllm serve` end-to-end on a Google Cloud
TPU **v6e-32** slice (8 hosts × 4 chips = 32 TPU chips, 992 GiB HBM
total). The model is a 543-GiB-bf16-expanded MoE (256 FP4 experts +
MLA attention + dense FP8) — this repo carries every patch the
upstream `tpu-inference` JAX backend needs to load and serve it.

The single optimization goal is **fast, mathematically correct
inference with the real V4-Flash weights**.

## TL;DR — fresh-slice bringup

You need: a TPU **v6e-32** slice already provisioned (8 hosts × 4
chips), SSH set up between the head and the 7 workers, a GCS bucket
already staged with the V4-Flash HuggingFace cache layout (config +
46 safetensors shards), and `uv` + `git` + `gcsfuse` installed on
every host (your cloud-init / infra responsibility — see "External
prereqs" below).

On the **head** host (worker 0, conventionally `10.164.0.41`):

```bash
git clone <this-repo> ~/claude-deepseek-v4
cd ~/claude-deepseek-v4
cp .env.example .env
$EDITOR .env                       # fill in HF_TOKEN, GCS_BUCKET, etc.

./run.sh bootstrap                 # one-shot: rsync + setup + GCS mount +
                                   # Ray start across all 8 hosts. ~10 min.
./run.sh serve                     # reset + sync + launch vllm serve.
                                   # ~5 min load + ~5-15 min cold compile.
./scripts/full_slice_v4_smoke_check.sh   # validates deterministic Paris
```

After the first successful `serve`, the per-host JAX compile cache is
populated; subsequent `./run.sh serve` cycles are sub-minute on the
compile path.

See [CLAUDE.md](CLAUDE.md) for the runbook (operational details,
optimization knobs, pitfalls, and the **minimum-delta rule** — agents
working on this repo should keep changes as small as possible while
preserving correctness + speed).

## `./run.sh` modes

| Mode | What it does |
|---|---|
| `./run.sh bootstrap` | Fresh-slice fan-out: rsync this repo to all 7 workers, run `setup.sh` on each, mount GCS, start Ray. One-shot, idempotent. |
| `./run.sh serve` | Cluster cleanup + source sync + launch `vllm serve` on the v6e-32 slice. Use after `bootstrap` (or after an iteration to relaunch). |
| `./run.sh agent` | Start the autonomous `claude -p prompt.md` loop on **this host only**. Default if no arg given (back-compat). |
| `./run.sh stop` | Stop the agent loop. |
| `./run.sh status` | Check whether the agent loop is running. |

## External prereqs

The bootstrap script does **not** install OS-level dependencies. On
every host (head + 7 workers) before running `./run.sh bootstrap`,
you need:

* **A Linux user with the same name on every host** (the script
  assumes `enyouki@<ip>` SSH targets and `~/claude-deepseek-v4`
  paths line up).
* `uv` (Python package manager) on PATH.
* `git` on PATH.
* `gcsfuse` on PATH (only required if `MOUNT_GCS=1`).
* SSH from the head reachable to every worker via
  `~/.ssh/google_compute_engine` (the GCE-provisioned identity).

On the head only, additionally:

* `claude` CLI on PATH (only needed if you'll use `./run.sh agent`).
* SSH agent unlocked for `id_ed25519` (only needed if you want
  `./run.sh agent` to push commits to GitHub after each iter).

## What each mode does (concretely)

**`./run.sh bootstrap`** — one-shot fan-out across the slice:
1. Reads `.env`.
2. Runs `scripts/setup.sh` on the head (creates `work/vllm_env`,
   installs vllm + tpu-inference editable from the worktree).
3. For each of the 7 workers: rsyncs the repo (excluding venv +
   logs + .env), copies `.env` over, runs `setup.sh` remotely.
4. If `MOUNT_GCS=1`: gcsfuse-mounts the bucket on every host.
5. Calls `scripts/full_slice_v4_ray_restart.sh` which `ray stop` +
   `ray start`s all 8 hosts. Verifies `0.0/32.0 TPU` available.

**`./run.sh serve`** — every change-then-test cycle:
1. `scripts/full_slice_v4_reset.sh` (cluster orphan cleanup).
2. `scripts/full_slice_v4_sync.sh` (rsync source to 7 workers).
3. `scripts/full_slice_v4_smoke.sh` (launch `vllm serve`).
4. Prints the log path; you validate with
   `scripts/full_slice_v4_smoke_check.sh`.

**`./run.sh agent`** — the autonomous-agent loop on this host
only (single-VM dev workflow). Reads `.env`, runs `setup.sh`,
optionally mounts GCS, runs preflight, then backgrounds
`scripts/loop.sh` which calls `claude -p prompt.md` repeatedly.

Pass criterion for serve: `full_slice_v4_smoke_check.sh` exits 0
with `PASS: deterministic completion contains 'Paris'`.

## Layout

| Path | What it is |
|---|---|
| `run.sh` | Top-level entry point — bootstrap + agent-loop launcher. |
| `scripts/full_slice_v4_*.sh` | Operational helpers for the v6e-32 deploy: reset, sync, smoke, ray restart, warm cache. |
| `scripts/setup.sh` | Idempotent venv bootstrap (uv + editable installs). |
| `scripts/mount_gcs.sh` | gcsfuse-mounts the GCS weights bucket onto the HF cache path. |
| `scripts/preflight.sh` | TPU/JAX sanity check. |
| `work/tpu-inference/` | JAX V4 implementation (this repo's main payload). Git subtree of the upstream `tpu-inference` repo. |
| `work/vllm/` | vLLM working tree. Git subtree. |
| `work/vllm_env/` | uv-managed venv. |
| `logs/` | Smoke logs, preflight log, agent-loop logs. `.gitignore`d. |
| `CLAUDE.md` | Runbook for the next agent / human picking this up. |
| `.env.example` | Documents every env var the scripts read. |

## Required external state

* **TPU v6e-32 slice** with all 8 hosts reachable from the head
  via `~/.ssh/google_compute_engine`.
* **GCS bucket** with `config.json` + 46 safetensors shards laid
  out in HuggingFace cache form under
  `gs://<bucket>/<dir>/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/<sha>/`.
* **HF token** with read access to `deepseek-ai/DeepSeek-V4-Flash`
  (gated model; the bucket is the auth-free fast path, but vLLM
  occasionally falls back to the HF API for tokenizer config).
* **One SSH key for GitHub** (`~/.ssh/id_ed25519` by convention) and
  one for cross-host SSH within the slice (`~/.ssh/google_compute_engine`,
  the GCE-provisioned identity).

## Status

Smoke gate **GREEN** as of 2026-04-30: cold
`./run.sh serve` returns deterministic `Paris` for "The capital of
France is" via `/v1/completions`; cold compile ~97s, warm-cache
curl sub-second. Loading is ~4 min for 35020 tensors; the MoE
forward is vectorized + inline-consolidated at load; the
persistent JAX compile cache populates per-host after first
success.

That's the demo path. For OpenRouter-grade production serving,
the prioritized work list — multi-seq decode, real concurrency,
chat/tool/reasoning surfaces, eval gates — lives in
[CLAUDE.md](CLAUDE.md) "Production-readiness backlog". Read that
before picking up any work.

# claude-deepseek-v4

Run DeepSeek-V4-Flash via `vllm serve` end-to-end on a Google Cloud
TPU **v6e-32** slice (8 hosts × 4 chips = 32 TPU chips, 992 GiB HBM
total). The model is a 543-GiB-bf16-expanded MoE (256 FP4 experts +
MLA attention + dense FP8) — this repo carries every patch the
upstream `tpu-inference` JAX backend needs to load and serve it.

The single optimization goal is **fast, mathematically correct
inference with the real V4-Flash weights**.

## TL;DR — fresh-VM bringup

You need: a TPU v6e-32 slice provisioned, a GCS bucket already staged
with the V4-Flash HuggingFace cache layout (config + 46 safetensors
shards), and SSH access between hosts.

```bash
git clone <this-repo> ~/claude-deepseek-v4
cd ~/claude-deepseek-v4
cp .env.example .env
$EDITOR .env                     # fill in HF_TOKEN, GCS_BUCKET, etc.
./run.sh                         # one-shot: setup -> mount -> preflight -> serve
```

`./run.sh` runs the full bringup: bootstrap the venv, mount the GCS
bucket (if `MOUNT_GCS=1`), TPU pre-flight, then launch the smoke
serve. Re-running it on the same host is safe (idempotent on the
already-done parts).

The first launch on a fresh VM compiles the V4 forward graph from
scratch (~5–15 min depending on compile cache state). Every launch
after that hits the persistent JAX compile cache and is sub-minute.

See [CLAUDE.md](CLAUDE.md) for the runbook (operational details,
optimization knobs, pitfalls).

## What `./run.sh` does

1. Reads `.env` (validates `CLAUDE_CODE_OAUTH_TOKEN`, `HF_TOKEN`).
2. Runs `scripts/setup.sh` — bootstraps `work/vllm_env` (uv-managed
   Python venv) with the right `vllm` + `tpu-inference` editable
   installs. Idempotent.
3. If `MOUNT_GCS=1`, runs `scripts/mount_gcs.sh` to gcsfuse-mount
   the V4-Flash bucket onto `~/.cache/huggingface/hub`.
4. Runs `scripts/preflight.sh` (JAX TPU sanity check, writes
   `logs/tpu-preflight.log`).
5. Launches a backgrounded agent loop that drives further
   iterations (this repo is wired for an autonomous-agent workflow;
   if you only want serve, run the smoke launcher directly — see
   below).

## The serve loop (manual; what an iteration looks like)

After the venv is bootstrapped, every change-then-test cycle is:

```bash
scripts/full_slice_v4_reset.sh   # cluster cleanup; safe to re-run
scripts/full_slice_v4_sync.sh    # mandatory after code edits — see CLAUDE.md
scripts/full_slice_v4_smoke.sh   # launch vllm serve; writes pid + log
scripts/full_slice_v4_smoke_check.sh  # validate /v1/completions when ready
```

Pass criterion: `full_slice_v4_smoke_check.sh` exits 0 with
`PASS: deterministic completion contains 'Paris'`.

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

Loading: 35020 V4-Flash tensors stream-load + per-host slice-aware
in ~4 minutes on the v6e-32 slice (~140 t/s/host). MoE forward is
vectorized (3 einsums per layer, not 256+ matmuls). Persistent JAX
compile cache populates after the first successful curl; subsequent
launches' first `/v1/completions` is sub-minute.

See [CLAUDE.md](CLAUDE.md) for current verified-working state and
known issues.

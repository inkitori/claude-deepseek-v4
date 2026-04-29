# Full-Slice DeepSeek V4-Flash Setup Plan

This is a handoff plan for Claude Code. The goal is to make the real
DeepSeek V4-Flash checkpoint serve on the full TPU slice, not on the
single local worker.

## Goal

Run:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash
```

on the full `v6e-32` TPU slice:

- 8 TPU VM workers.
- 4 local TPU chips per worker.
- 32 total TPU chips.
- `--tensor-parallel-size 32`.

Do not try to make full V4-Flash fit on one worker with `TP=4`. That
path OOMs by design because worker 0 only has 4 chips / 128 GiB HBM.

## Verified Current State

Node:

```text
v6dev-u4a-32-3
```

Accelerator:

```text
v6e-32
```

Worker internal IPs from TPU metadata:

```text
worker 0: 10.164.0.41
worker 1: 10.164.0.22
worker 2: 10.164.0.35
worker 3: 10.164.0.36
worker 4: 10.164.0.39
worker 5: 10.164.0.45
worker 6: 10.164.0.18
worker 7: 10.164.0.30
```

All 8 workers are reachable with the gcloud-managed SSH key:

```bash
ssh -T \
  -i /home/enyouki/.ssh/google_compute_engine \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/home/enyouki/.ssh/google_compute_known_hosts \
  enyouki@<worker-ip> hostname
```

Worker 0 currently has:

- `/home/enyouki/claude-deepseek-v4`
- `work/vllm_env`
- HF gcsfuse mount
- DeepSeek V4-Flash checkpoint path

Workers 1-7 currently do not have:

- the repo
- the venv
- the HF mount
- the checkpoint directory

The real checkpoint is mounted on worker 0 at:

```text
/home/enyouki/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash
```

Observed checkpoint footprint on worker 0:

```text
46 safetensors shards
159617149040 bytes
148.66 GiB
```

## Design

Worker 0 is the source of truth. Workers 1-7 are disposable mirrors.

Rules:

- Never edit code on workers 1-7.
- Never commit from workers 1-7.
- Sync repo/runtime from worker 0 to workers 1-7.
- Use `rsync --delete` to prevent worker drift.
- Do not copy model weights between workers.
- Mount the GCS Hugging Face cache on every worker instead.

This avoids repository divergence and avoids repeated 160 GB
worker-to-worker checkpoint transfers.

The first repo/runtime bootstrap can transfer several GiB to each worker.
After that, `rsync` should send deltas only.

## Scripts To Add

Prefer adding separate full-slice scripts instead of changing the existing
single-worker `run.sh` path. The current single-worker harness is useful for
tiny tests and should remain available.

### `scripts/full_slice_workers.sh`

Purpose: discover workers and verify SSH access.

Discover worker endpoints from metadata:

```bash
curl -fsS -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints
```

Extract the comma-separated IPs. The value shape is currently:

```text
unknown:unknown:10.164.0.41,unknown:unknown:10.164.0.22,...
```

Verify all 8 workers:

```bash
ssh -T \
  -i "$HOME/.ssh/google_compute_engine" \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile="$HOME/.ssh/google_compute_known_hosts" \
  "enyouki@$ip" \
  'printf "host=%s worker=" "$(hostname)"; curl -fsS -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number; printf "\n"'
```

Pass condition:

- exactly 8 workers
- worker IDs are `0` through `7`
- all report accelerator `v6e-32`

### `scripts/full_slice_sync.sh`

Purpose: mirror worker 0 repo/runtime to workers 1-7.

Default behavior:

- Abort if `git status --short` is non-empty, unless `ALLOW_DIRTY_SYNC=1`.
- Worker 0 is authoritative.
- Workers 1-7 are overwritten with `rsync --delete`.

Recommended sync excludes:

```text
.codex
.pytest_cache
logs
work/scratch
```

Do not exclude:

```text
.git
work/vllm
work/tpu-inference
work/vllm_env
scripts
```

The venv path is the same on all workers:

```text
/home/enyouki/claude-deepseek-v4/work/vllm_env
```

so copying the venv is acceptable. If import verification fails on any
worker, run the existing `scripts/setup.sh` on that worker rather than
debugging manually.

Post-sync verification on every worker:

```bash
cd /home/enyouki/claude-deepseek-v4
git rev-parse HEAD
work/vllm_env/bin/python -c "import vllm, tpu_inference; print('ok')"
```

Record a sync manifest under `logs/full-slice-sync-<timestamp>.txt` on
worker 0 containing:

- timestamp
- worker IPs
- `git rev-parse HEAD`
- whether dirty sync was allowed
- `git status --short`
- Python import result for each worker

### `scripts/full_slice_mount_gcs.sh`

Purpose: mount the model cache on every worker. Do not copy weights.

For each worker:

```bash
cd /home/enyouki/claude-deepseek-v4
GCS_BUCKET=personal-mark-eu \
GCS_ONLY_DIR=vllm/hub \
HF_HUB_MOUNT_DIR="$HOME/.cache/huggingface/hub" \
./scripts/mount_gcs.sh
```

Verify on every worker:

```bash
mountpoint -q "$HOME/.cache/huggingface/hub"
test -d "$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash"
```

Also verify `gcsfuse` exists before attempting the mount:

```bash
command -v gcsfuse
```

If `gcsfuse` is missing on workers 1-7, stop and report that as the next
blocker. Do not fall back to copying checkpoint shards between workers.

### `scripts/full_slice_ray.sh`

Purpose: start a Ray cluster across the 8 TPU workers.

Start Ray head on worker 0:

```bash
ray stop --force || true
ray start --head \
  --node-ip-address=10.164.0.41 \
  --port=6379 \
  --resources='{"TPU":4}'
```

Start Ray workers on workers 1-7:

```bash
ray stop --force || true
ray start \
  --address=10.164.0.41:6379 \
  --node-ip-address=<worker-ip> \
  --resources='{"TPU":4}'
```

Verify from worker 0:

```bash
cd /home/enyouki/claude-deepseek-v4
work/vllm_env/bin/python - <<'PY'
import ray
ray.init(address="auto")
print(ray.cluster_resources())
PY
```

Pass condition:

```text
TPU: 32
```

If Ray reports fewer than 32 TPU resources, do not run V4-Flash yet.

### `scripts/full_slice_v4_smoke.sh`

Purpose: run the real V4-Flash smoke through full-slice Ray.

Run this only on worker 0.

Environment:

```bash
export PATH="/home/enyouki/claude-deepseek-v4/work/vllm_env/bin:$PATH"
export VIRTUAL_ENV="/home/enyouki/claude-deepseek-v4/work/vllm_env"
export TPU_MULTIHOST_BACKEND=ray
export RAY_ADDRESS=auto
export JAX_PLATFORMS=''
export TPU_HOST_BOUNDS=2,4,1
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_PROCESS_BOUNDS=2,4,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEW_MODEL_DESIGN=1
```

First pass: eager, single sequence:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 32 \
  --max-model-len 256 \
  --max-num-seqs 1 \
  --port 18081 \
  --seed 0 \
  --trust-remote-code \
  --dtype bfloat16 \
  --enforce-eager \
  --additional_config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true}}}'
```

Wait for readiness:

```bash
curl -sf http://localhost:18081/v1/models
```

Completion smoke:

```bash
curl -s http://localhost:18081/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","prompt":"The capital of France is","max_tokens":8,"temperature":0,"seed":0}'
```

Send the exact same request twice.

Pass criteria:

- `/v1/models` returns HTTP 200.
- `/v1/completions` returns HTTP 200.
- both seeded completions are byte-identical.
- text starts with `Paris` or ` Paris`.

Second pass: production-shape smoke:

- remove `--enforce-eager`
- set `--max-num-seqs 4`
- send two identical seeded requests again
- send four concurrent requests if the two-request smoke passes

## Transfer And Cost Guidance

Avoid expensive transfers:

- Do not scp/rsync safetensors.
- Do not copy `~/.cache/huggingface/hub` between workers.
- Do not let each worker download from Hugging Face.
- Use the existing GCS bucket via gcsfuse on every worker.

Expected transfer costs:

- First repo/venv sync: about 4.6 GiB from worker 0 to each of 7 workers.
- Later repo syncs: rsync deltas only.
- Model reads: each worker reads needed shards from GCS through gcsfuse.
- TPU inter-worker traffic during inference is expected tensor-parallel
  communication and cannot be avoided for `TP=32`.

Avoid divergence:

- Worker 0 is the only editable worker.
- Workers 1-7 are mirrors.
- Sync with `rsync --delete`.
- Abort on dirty worker 0 tree unless explicitly allowed.
- After sync, verify `git rev-parse HEAD` on all workers.

## Acceptance Criteria

Preflight:

- all 8 workers reachable over SSH
- all 8 workers have the repo
- all 8 workers have working `work/vllm_env`
- all 8 workers mount the HF GCS cache
- all 8 workers can see `models--deepseek-ai--DeepSeek-V4-Flash`

Ray:

- Ray cluster has 8 nodes
- Ray reports `TPU: 32`
- every Ray worker can import `vllm` and `tpu_inference`

Model:

- `vllm serve` starts with `--tensor-parallel-size 32`
- eager smoke reaches `/v1/models`
- eager smoke returns a deterministic Paris completion
- production-shape smoke without `--enforce-eager` works with `--max-num-seqs 4`

Cleanup:

- failed runs kill the local `vllm serve` process
- Ray can be stopped with `ray stop --force` on all workers
- no stale full-slice process should be left running accidentally

## Known Pitfalls

- `gcloud compute tpus tpu-vm ssh --worker=all --output-directory=...` may
  have a CLI bug where it tries to write `None/0.log`. Direct SSH with
  `~/.ssh/google_compute_engine` works.
- Plain `ssh enyouki@<ip>` may fail with `Permission denied`; always use
  the explicit `google_compute_engine` key.
- The existing `scripts/t8_eager_smoke.sh` is single-worker `TP=4`; it is
  not the full-slice proof.
- If Ray reports fewer than 32 TPU resources, do not run the real model.
- If GCS mount is absent on workers 1-7, do not copy checkpoint shards as a
  fallback.

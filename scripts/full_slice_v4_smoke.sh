#!/usr/bin/env bash
# Launch vllm serve DeepSeek-V4-Flash on the full v6e-32 slice.
#
# Run on worker 0 (10.164.0.41) only. Assumes:
#   * Ray cluster is up (`ray status` shows TPU: 32, 8 nodes).
#   * GCS-mounted HF cache holds the V4-Flash checkpoint on every worker.
#   * The repo + venv are synced across all 8 workers.
#
# This calls into the streaming loader (deepseek_v4_loader.iter_v4_safetensors_dequant_torch)
# + per-tensor sharding (place_torch_as_jax_sharded) so the 543 GB bf16 model
# spreads evenly across 32 chips at ~17 GB / chip rather than OOMing.

set -euo pipefail

REPO_ROOT="/home/enyouki/claude-deepseek-v4"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/work/vllm_env"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPO_ROOT/logs/full-slice-v4-smoke-${TS}.log"
PORT="${PORT:-18081}"
MAX_LEN="${MAX_LEN:-256}"
MAX_SEQS="${MAX_SEQS:-1}"

mkdir -p "$REPO_ROOT/logs"

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONPATH="$REPO_ROOT/work/vllm:$REPO_ROOT/work/tpu-inference:$VENV/lib/python3.12/site-packages"

# Multi-host TPU env. These tell the JAX TPU plugin we're cooperating in a
# distributed slice (8 hosts × 4 chips). Without TPU_*_BOUNDS the plugin
# can't form the 32-chip mesh and falls back to a single-host view.
export TPU_MULTIHOST_BACKEND=ray
export RAY_ADDRESS=10.164.0.41:6379
export JAX_PLATFORMS=
export TPU_HOST_BOUNDS=2,4,1
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_PROCESS_BOUNDS=2,4,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1

# Force HF offline (we mount the checkpoint via gcsfuse; no internet).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# tpu-inference: enable the new model design path that supports the V4
# attention DP topology (`enable_dp_attention=true` requires this).
export NEW_MODEL_DESIGN=1

# Optional opt-in: parallel CPU dequant inside iter_v4_safetensors_dequant_torch.
# Default 0 = current sequential behavior. Set V4_LOADER_PREFETCH_WORKERS=4..8
# at the shell to overlap dequant with TPU placement and parallelize across
# cores per host. Forwarded to Ray workers via VLLM_RAY_EXTRA_ENV_VARS_TO_COPY.
export V4_LOADER_PREFETCH_WORKERS="${V4_LOADER_PREFETCH_WORKERS:-0}"
existing_extra="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-}"
if [ -n "$existing_extra" ]; then
    export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${existing_extra},V4_LOADER_PREFETCH_WORKERS"
else
    export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="V4_LOADER_PREFETCH_WORKERS"
fi

echo "[smoke] launching vllm serve | log=$LOG | prefetch_workers=$V4_LOADER_PREFETCH_WORKERS"
"$VENV/bin/vllm" serve deepseek-ai/DeepSeek-V4-Flash \
    --distributed-executor-backend ray \
    --tensor-parallel-size 32 \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --port "$PORT" \
    --seed 0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --enforce-eager \
    --additional_config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true}}}' \
    > "$LOG" 2>&1 &

SERVE_PID=$!
echo "[smoke] vllm serve pid=$SERVE_PID log=$LOG"
echo "$SERVE_PID" > "$REPO_ROOT/logs/full-slice-v4-smoke.pid"
echo "$LOG"

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

# Slice-aware loader: each host reads only its row range. Default 1.
# Set V4_LOADER_SLICE_AWARE=0 to fall back to full-tensor dequant per host.
export V4_LOADER_SLICE_AWARE="${V4_LOADER_SLICE_AWARE:-1}"

# Optional opt-in: parallel CPU dequant inside iter_v4_safetensors_dequant_torch.
# Default 0 = sequential. Set to 4-8 to overlap dequant with TPU placement.
# Note: empirically this didn't help on real V4 because placement (PCIe), not
# CPU dequant, is the bottleneck. Kept as a knob for future work.
export V4_LOADER_PREFETCH_WORKERS="${V4_LOADER_PREFETCH_WORKERS:-0}"

# Persistent JAX compile cache: every Ray worker reuses XLA-compiled modules
# across launches as long as the cache dir is reachable. Per-host (not GCS),
# so each worker has its own cache. Survives process restarts; lost if the
# worker host is rebuilt. Set V4_JAX_CACHE_DIR= to disable.
export V4_JAX_CACHE_DIR="${V4_JAX_CACHE_DIR:-/tmp/jax-compile-cache-v4}"
if [ -n "$V4_JAX_CACHE_DIR" ]; then
    mkdir -p "$V4_JAX_CACHE_DIR"
    export JAX_COMPILATION_CACHE_DIR="$V4_JAX_CACHE_DIR"
fi

# XLA: parallelize compile passes across CPU cores. Modest 10-30% compile
# speedup. Off by default in tpu-inference; turning it on for our smoke.
existing_xla="${XLA_FLAGS:-}"
xla_extra="--xla_tpu_impure_hlo_parallel_compile=true"
if [ -n "$existing_xla" ]; then
    export XLA_FLAGS="$existing_xla $xla_extra"
else
    export XLA_FLAGS="$xla_extra"
fi

# Bump Ray's compiled-graph channel timeout. Default is 300 seconds, which
# trips during the first inference if jit_run_model recompiles for the
# request shape (already burned us once at 5m1s). 1 hour gives the cold
# path room while still failing fast on actual hangs.
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-3600}"

# Cache even small / fast-to-compile modules. Default JAX policy skips
# them, but on a flagship MoE every cache hit shaves real seconds.
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES:-0}"
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS="${JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS:-0}"

# Forward these to Ray workers (vLLM only carries over a curated env-var
# set by default; non-VLLM_/HF_ vars need explicit opt-in).
existing_extra="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-}"
new_extra="V4_LOADER_PREFETCH_WORKERS,V4_LOADER_SLICE_AWARE,JAX_COMPILATION_CACHE_DIR,XLA_FLAGS,RAY_CGRAPH_get_timeout,JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES,JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS"
if [ -n "$existing_extra" ]; then
    export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${existing_extra},${new_extra}"
else
    export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${new_extra}"
fi

echo "[smoke] launching vllm serve | log=$LOG"
echo "[smoke]   slice_aware=$V4_LOADER_SLICE_AWARE prefetch_workers=$V4_LOADER_PREFETCH_WORKERS"
echo "[smoke]   jax_cache=$JAX_COMPILATION_CACHE_DIR"
echo "[smoke]   xla_flags=$XLA_FLAGS"
echo "[smoke]   ray_cgraph_timeout=${RAY_CGRAPH_get_timeout}s"
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

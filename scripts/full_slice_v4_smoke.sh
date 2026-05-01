#!/usr/bin/env bash
# Launch vllm serve DeepSeek-V4-Flash on the full v6e-32 slice. Run on
# worker 0 (10.164.0.41). Assumes Ray is up + repo/venv synced to all 8
# workers + GCS-mounted V4-Flash checkpoint visible on every host.

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

# Multi-host TPU bounds: form the 32-chip distributed mesh.
export TPU_MULTIHOST_BACKEND=ray
export RAY_ADDRESS=10.164.0.41:6379
export JAX_PLATFORMS=
export TPU_HOST_BOUNDS=2,4,1
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_PROCESS_BOUNDS=2,4,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1

# HF offline — checkpoint is mounted via gcsfuse, no internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# tpu-inference: required for V4's `enable_dp_attention=true` topology.
export NEW_MODEL_DESIGN=1

export V4_LOADER_SLICE_AWARE="${V4_LOADER_SLICE_AWARE:-1}"
export V4_LOADER_PLACE_WORKERS="${V4_LOADER_PLACE_WORKERS:-8}"
export V4_LOADER_PREFETCH_WORKERS="${V4_LOADER_PREFETCH_WORKERS:-0}"
# Per-decode-step NaN-localization tripwire (CLAUDE.md S1). Off by default;
# set V4_DECODE_NAN_TRIPWIRE=1 to emit per-sub-block NaN counts to the log.
export V4_DECODE_NAN_TRIPWIRE="${V4_DECODE_NAN_TRIPWIRE:-0}"
# One-shot finiteness audit of loaded weights (CLAUDE.md S1 hyp 1). Off by
# default; set V4_WEIGHT_NAN_AUDIT=1 to print `[weight_nan] {path}` for any
# leaf containing NaN/Inf right after `load_weights_from_dir` finishes.
export V4_WEIGHT_NAN_AUDIT="${V4_WEIGHT_NAN_AUDIT:-0}"

# Don't inherit parent-shell XLA_FLAGS (stale value SIGSEGVs workers; see
# CLAUDE.md pitfall #4). Opt-in via V4_XLA_FLAGS.
export XLA_FLAGS="${V4_XLA_FLAGS:-}"
case "$XLA_FLAGS" in
    *xla_tpu_impure_hlo_parallel_compile*)
        echo "[smoke] FATAL: XLA_FLAGS contains the known-bad flag --xla_tpu_impure_hlo_parallel_compile" >&2
        echo "[smoke]        That flag is not recognized by this libtpu build and SIGSEGVs every worker." >&2
        exit 2
        ;;
esac

# 300s default trips on first inference recompile; 1h covers cold path.
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-3600}"

# JAX 0.9 cache policy — cache even small / fast modules (older
# JAX_COMPILATION_CACHE_MIN_* names are silently ignored).
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:-0}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"

# Forward these to Ray workers (vLLM only carries over a curated env-var
# set by default; non-VLLM_/HF_ vars need explicit opt-in).
existing_extra="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-}"
new_extra="V4_LOADER_PREFETCH_WORKERS,V4_LOADER_SLICE_AWARE,V4_LOADER_PLACE_WORKERS,V4_DECODE_NAN_TRIPWIRE,V4_WEIGHT_NAN_AUDIT,XLA_FLAGS,RAY_CGRAPH_get_timeout,JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
if [ -n "$existing_extra" ]; then
    export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${existing_extra},${new_extra}"
else
    export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${new_extra}"
fi

# --chat-template is intentionally NOT passed — DeepseekV4Tokenizer
# resolves via upstream encode_messages and ignores it (see CLAUDE.md S4).

echo "[smoke] launching vllm serve | log=$LOG"
echo "[smoke]   slice_aware=$V4_LOADER_SLICE_AWARE place_workers=$V4_LOADER_PLACE_WORKERS prefetch_workers=$V4_LOADER_PREFETCH_WORKERS"
echo "[smoke]   xla_flags=$XLA_FLAGS  ray_cgraph_timeout=${RAY_CGRAPH_get_timeout}s"
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
    --reasoning-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v4 \
    --additional_config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true}}}' \
    > "$LOG" 2>&1 &

SERVE_PID=$!
echo "[smoke] vllm serve pid=$SERVE_PID log=$LOG"
echo "$SERVE_PID" > "$REPO_ROOT/logs/full-slice-v4-smoke.pid"
echo "$LOG"

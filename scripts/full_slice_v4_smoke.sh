#!/usr/bin/env bash
# Launch vllm serve DeepSeek-V4-Flash on the full v6e-16 slice. Run on
# worker 0 (the head). Assumes Ray is up + repo/venv synced to all 4
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

# Single-instance guard. Concurrent loop sessions otherwise race into two
# `vllm serve` engines that both grab the 32-chip placement group — one wins,
# the other deadlocks "Waiting for placement group", and both can wedge the
# slice (see CLAUDE.md). flock serializes the check+launch (closes the TOCTOU);
# the pgrep/port check refuses to launch while an engine is already running or
# still compiling (port not yet bound). When refused, DON'T relaunch — probe the
# live engine on :$PORT or `full_slice_v4_reset.sh` first. SMOKE_NO_GUARD=1 skips.
if [ "${SMOKE_NO_GUARD:-0}" != "1" ]; then
    exec 9>"/tmp/s1_smoke_launch.lock"
    if ! flock -n 9; then
        echo "[smoke] REFUSING: another smoke launch is in progress (lock held)." >&2
        exit 3
    fi
    if pgrep -f 'bin/vllm serve deepseek' >/dev/null 2>&1; then
        echo "[smoke] REFUSING: a 'vllm serve' is already running/starting. Probe :$PORT or reset first." >&2
        exit 3
    fi
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        echo "[smoke] REFUSING: port $PORT already bound — an engine is up. Probe it or reset first." >&2
        exit 3
    fi
fi

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONPATH="$REPO_ROOT/work/vllm:$REPO_ROOT/work/tpu-inference:$VENV/lib/python3.12/site-packages"

# Multi-host TPU bounds: form the 16-chip distributed mesh (v6e-16, topology
# 4x4 = 4 hosts x 4 chips). Matches GCP metadata HOST_BOUNDS/PROCESS_BOUNDS=2,2,1.
# (v6e-32 was 2,4,1 = 8 hosts; the chips-per-host/process 2,2,1 are unchanged.)
export TPU_MULTIHOST_BACKEND=ray
export RAY_ADDRESS="${RAY_ADDRESS:-$("$REPO_ROOT/scripts/full_slice_v4_discover.sh" head):6379}"
export JAX_PLATFORMS=
export TPU_HOST_BOUNDS=2,2,1
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_PROCESS_BOUNDS=2,2,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1

# HF offline — checkpoint is mounted via gcsfuse, no internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Pre-flight: the V4-Flash weights MUST be readable by enyouki on EVERY host, else
# the worker actors silently dummy-fallback to jnp.zeros and the engine wedges (the
# bringup mounts the GCS bucket enyouki-owned only on the head; workers get a
# root-only mount). Idempotent; hard-fail rather than burn a 10-30 min smoke on a
# guaranteed wedge. See full_slice_v4_mount_weights.sh for the full diagnosis.
if ! "$REPO_ROOT/scripts/full_slice_v4_mount_weights.sh"; then
    echo "[smoke] FATAL: weights not readable by enyouki on all hosts (would wedge in dummy-zeros). Fix mounts first." >&2
    exit 4
fi

# tpu-inference: required for V4's `enable_dp_attention=true` topology.
export NEW_MODEL_DESIGN=1

export V4_LOADER_SLICE_AWARE="${V4_LOADER_SLICE_AWARE:-1}"
export V4_LOADER_PLACE_WORKERS="${V4_LOADER_PLACE_WORKERS:-4}"
export V4_LOADER_PREFETCH_WORKERS="${V4_LOADER_PREFETCH_WORKERS:-0}"
# Per-decode-step NaN-localization tripwire. Off by default; set
# V4_DECODE_NAN_TRIPWIRE=1 to emit per-sub-block NaN counts to the log
# (the silent-callback path runs always — see `_v4_nan_tripwire`).
export V4_DECODE_NAN_TRIPWIRE="${V4_DECODE_NAN_TRIPWIRE:-0}"
# v6e-16 scheckne fix: run create_jit_model's state re-wrap EAGERLY (no @nnx.jit
# dispatch) so the fast rank-0 worker doesn't launch a 16-partition SPMD program
# ahead of the laggard workers (launch-id desync). On by default for this slice;
# model_loader defaults it OFF (shared infra). See model_loader.py:_get_nnx_model.
export V4_EAGER_CREATE_JIT_MODEL="${V4_EAGER_CREATE_JIT_MODEL:-1}"

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

# APIServer gives up waiting for EngineCore after this. vllm default is 600s,
# but a COLD XLA compile of V4-Flash on the slice is 10-30 min (CLAUDE.md) and
# blows 600s -> "Timed out waiting for engine core processes to start" and the
# whole serve dies. 2400s covers a cold compile; a warm start ignores it.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"

# JAX 0.9 cache policy — cache even small / fast modules (older
# JAX_COMPILATION_CACHE_MIN_* names are silently ignored).
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:-0}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"

# Forward these to Ray workers (vLLM only carries over a curated env-var
# set by default; non-VLLM_/HF_ vars need explicit opt-in).
existing_extra="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-}"
new_extra="V4_LOADER_PREFETCH_WORKERS,V4_LOADER_SLICE_AWARE,V4_LOADER_PLACE_WORKERS,V4_DECODE_NAN_TRIPWIRE,V4_EAGER_CREATE_JIT_MODEL,XLA_FLAGS,RAY_CGRAPH_get_timeout,JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
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
    --tensor-parallel-size 16 \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --port "$PORT" \
    --seed 0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --reasoning-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v4 \
    --additional_config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true}}}' \
    ${V4_PROFILER_ARGS:-} \
    > "$LOG" 2>&1 &

SERVE_PID=$!
echo "[smoke] vllm serve pid=$SERVE_PID log=$LOG"
echo "$SERVE_PID" > "$REPO_ROOT/logs/full-slice-v4-smoke.pid"
echo "$LOG"

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

# Multi-threaded per-host placement. Default 8 — most per-tensor work
# (read_dequant_slice, make_array_from_callback, host->device transfer)
# happens in JAX/safetensors C code that releases the GIL, so parallelism
# is real. Set to 1 to disable for single-thread parity testing.
export V4_LOADER_PLACE_WORKERS="${V4_LOADER_PLACE_WORKERS:-8}"

# Optional opt-in: parallel CPU dequant inside iter_v4_safetensors_dequant_torch.
# Default 0 = sequential. Set to 4-8 to overlap dequant with TPU placement.
# Note: empirically this didn't help on real V4 because placement (PCIe), not
# CPU dequant, is the bottleneck. Kept as a knob for future work.
export V4_LOADER_PREFETCH_WORKERS="${V4_LOADER_PREFETCH_WORKERS:-0}"

# Persistent JAX compile cache: tpu_inference's compilation_manager.py
# overrides jax_compilation_cache_dir to vllm_envs.VLLM_XLA_CACHE_PATH
# (default ~/.cache/vllm/xla_cache) at engine startup. Setting
# JAX_COMPILATION_CACHE_DIR here is a no-op — point users at the right
# path via VLLM_XLA_CACHE_PATH if they want to relocate.
#
# Per-host (not GCS), so each worker has its own cache. Survives process
# restarts; lost if the worker host is rebuilt. Cross-host rsync is
# *not* sound: SPMD compiles to host-specific binaries even when JAX's
# cache key is identical (verified 2026-04-30 via
# scripts/full_slice_v4_cache_fingerprint.sh — same filename, 8 distinct
# sha256s).

# NOTE on XLA flags: don't add unverified flags here. An earlier attempt
# at `--xla_tpu_impure_hlo_parallel_compile=true` *crashed every worker*
# on libtpu init with `Unknown flag in XLA_FLAGS`. The string appears in
# tpu_inference logs as a deepsea_compiler INTERNAL config option but
# isn't a recognized XLA flag in this build. Verify any addition with
# `python -c "import jax; jax.devices()"` under a candidate XLA_FLAGS
# value before wiring it into the smoke launcher.
#
# We intentionally do NOT inherit XLA_FLAGS from the parent shell — a
# stale bad value (e.g. left over from an autorunner env) silently
# SIGSEGVs every Ray worker. Opt in via V4_XLA_FLAGS if you really
# need a custom flag string for one launch.
export XLA_FLAGS="${V4_XLA_FLAGS:-}"
case "$XLA_FLAGS" in
    *xla_tpu_impure_hlo_parallel_compile*)
        echo "[smoke] FATAL: XLA_FLAGS contains the known-bad flag --xla_tpu_impure_hlo_parallel_compile" >&2
        echo "[smoke]        That flag is not recognized by this libtpu build and SIGSEGVs every worker." >&2
        exit 2
        ;;
esac

# Bump Ray's compiled-graph channel timeout. Default is 300 seconds, which
# trips during the first inference if jit_run_model recompiles for the
# request shape (already burned us once at 5m1s). 1 hour gives the cold
# path room while still failing fast on actual hangs.
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-3600}"

# Cache even small / fast-to-compile modules. Default JAX policy
# (min_compile_time_secs=1.0) silently skips them, which leaves
# /tmp/jax-compile-cache-v4 empty even after a successful smoke and
# defeats the whole point of the persistent cache.
#
# These env-var names follow JAX 0.9's `jax_persistent_cache_*` config
# option family. The older `JAX_COMPILATION_CACHE_MIN_*` names are
# silently ignored (no such config option exists), which masked this
# for previous launches — verify by `jax.config.jax_persistent_cache_min_compile_time_secs`.
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:-0}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"

# S1 gate: 1 flips __call__ from prefill-recompute baseline to the real
# decode orchestrator (deepseek_v4_run_with_decode_state, packed
# AttentionDecodeState threaded through kv_caches). Default 0 keeps the
# baseline until per-position decode-compile cost is amortized.
export V4_DECODE_STATE="${V4_DECODE_STATE:-0}"

# Side-effecting jax.debug.print of kv_caches[0] sum + nonfinite-count at
# orchestrator branch entry/exit. Only meaningful when V4_DECODE_STATE=1.
# Prints change the HLO (XLA can't DCE), so a green smoke under DIAG=1 is
# weaker than under DIAG=0.
export V4_DECODE_STATE_DIAG="${V4_DECODE_STATE_DIAG:-0}"

# Forward these to Ray workers (vLLM only carries over a curated env-var
# set by default; non-VLLM_/HF_ vars need explicit opt-in).
existing_extra="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-}"
new_extra="V4_LOADER_PREFETCH_WORKERS,V4_LOADER_SLICE_AWARE,V4_LOADER_PLACE_WORKERS,V4_DECODE_STATE,V4_DECODE_STATE_DIAG,XLA_FLAGS,RAY_CGRAPH_get_timeout,JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
if [ -n "$existing_extra" ]; then
    export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${existing_extra},${new_extra}"
else
    export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${new_extra}"
fi

# Chat template: NOT passed. vLLM auto-resolves tokenizer_mode='deepseek_v4'
# for DeepseekV4ForCausalLM (vllm/config/model.py:578), which loads
# DeepseekV4Tokenizer (vllm/tokenizers/deepseek_v4.py). That tokenizer's
# apply_chat_template ignores any --chat-template arg and calls
# encode_messages from vllm/tokenizers/deepseek_v4_encoding.py — the
# byte-equivalent of the encoder shipped at
# <hf-snapshot>/encoding/encoding_dsv4.py — across chat / thinking / tools
# / reasoning_effort. Pinned by TestVllmChatTemplateParity so a future
# upstream drift fires loudly. Passing --chat-template is a no-op here.

echo "[smoke] launching vllm serve | log=$LOG"
echo "[smoke]   slice_aware=$V4_LOADER_SLICE_AWARE place_workers=$V4_LOADER_PLACE_WORKERS prefetch_workers=$V4_LOADER_PREFETCH_WORKERS"
echo "[smoke]   v4_decode_state=$V4_DECODE_STATE (1=real decode orchestrator; 0=prefill-recompute baseline)"
echo "[smoke]   v4_decode_state_diag=$V4_DECODE_STATE_DIAG (1=jax.debug.print kv_caches[0] at orchestrator entry/exit)"
echo "[smoke]   jax_cache=\${VLLM_XLA_CACHE_PATH:-~/.cache/vllm/xla_cache} (set by tpu_inference)"
echo "[smoke]   xla_flags=$XLA_FLAGS"
echo "[smoke]   ray_cgraph_timeout=${RAY_CGRAPH_get_timeout}s"
echo "[smoke]   reasoning_parser=deepseek_v4 tool_call_parser=deepseek_v4 (auto-tool-choice on)"
# Reasoning + tool parsers: both registered upstream as "deepseek_v4"
# (vllm/reasoning/__init__.py and vllm/tool_parsers/__init__.py).
# Without these, chat responses return raw <think>...</think> blocks and
# DSML tool tokens in `content` instead of `reasoning_content` / `tool_calls`,
# which most clients treat as malformed. Safe to enable: when a request has
# no tools and no thinking, both parsers default to passthrough.
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

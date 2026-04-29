#!/usr/bin/env bash
# Idempotent host-side bootstrap for the DeepSeek-V4 / vLLM / tpu-inference
# stack. Safe to re-run; skips work that's already done. Designed to run
# directly on the TPU VM (no Docker).
#
# Layout (relative to repo root):
#   work/tpu-inference/   -- subtree (already in repo)
#   work/vllm/            -- subtree (already in repo)
#   work/vllm_env/        -- venv created here
#   work/scratch/         -- synthetic fixtures + any local working bytes
#   logs/                 -- iter-*.log, setup.log, tpu-preflight.log
#
# Exit codes: 0 on success / already-set-up; non-zero on a hard failure.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$REPO_DIR/work"
LOGS="$REPO_DIR/logs"
VENV="$WORK/vllm_env"
SETUP_LOG="$LOGS/setup.log"

mkdir -p "$WORK" "$LOGS" "$WORK/scratch"

ts() { date -u +%Y%m%d-%H%M%SZ; }
log() { echo "[setup $(ts)] $*" | tee -a "$SETUP_LOG"; }

log "repo=$REPO_DIR"

for bin in uv git claude; do
    if ! command -v "$bin" >/dev/null 2>&1; then
        log "FATAL: '$bin' not on PATH"
        exit 1
    fi
done

if [ ! -d "$WORK/tpu-inference" ] || [ ! -d "$WORK/vllm" ]; then
    log "FATAL: work/tpu-inference and work/vllm must already be present (they're subtrees of this repo)"
    exit 1
fi

if [ ! -d "$VENV" ]; then
    log "creating vllm_env (python 3.12) via uv"
    uv venv "$VENV" --python 3.12 2>&1 | tee -a "$SETUP_LOG"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! python -c "import vllm" 2>/dev/null; then
    log "installing vllm (TPU target) — this takes several minutes"
    # vllm uses setuptools-scm, which fails here because work/vllm/ is a
    # subtree of the parent repo (no own .git). The escape hatch is the
    # SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM env var; we tag it with the
    # pinned upstream commit for traceability.
    pinned_commit="$(tr -d '[:space:]' < "$WORK/vllm.commit" 2>/dev/null || true)"
    if [ -z "$pinned_commit" ]; then
        log "FATAL: $WORK/vllm.commit is missing or empty"
        exit 1
    fi
    # vllm's setup.py reads VLLM_VERSION_OVERRIDE before falling through to
    # setuptools-scm. setuptools-scm fails here because work/vllm/ is a
    # subtree of the parent repo (no own .git) — so we set the override.
    pretend_version="0.0.0+g${pinned_commit}"
    log "vllm VLLM_VERSION_OVERRIDE: $pretend_version"
    (
        cd "$WORK/vllm"
        export VLLM_VERSION_OVERRIDE="$pretend_version"
        uv pip install -r requirements/tpu.txt --torch-backend=cpu
        VLLM_TARGET_DEVICE=tpu uv pip install -e .
    ) 2>&1 | tee -a "$SETUP_LOG"
else
    log "vllm already importable; skipping install"
fi

tpu_inference_path="$(python -c "import tpu_inference; print(tpu_inference.__file__)" 2>/dev/null || true)"
expected_prefix="$WORK/tpu-inference/tpu_inference/"
if [ -z "$tpu_inference_path" ]; then
    log "installing tpu-inference (editable, from worktree)"
    (cd "$WORK/tpu-inference" && uv pip install -e . --no-deps) 2>&1 | tee -a "$SETUP_LOG"
elif [[ "$tpu_inference_path" != "$expected_prefix"* ]]; then
    # A non-editable / PyPI install is shadowing the worktree. Reinstall
    # from the worktree so test runs and `vllm serve` see our edits — and,
    # critically, so the worktree's vllm pin (which may not yet have new
    # symbols like SharedFusedMoE) lines up with the tpu_inference version.
    log "tpu_inference resolves to $tpu_inference_path (not worktree); reinstalling editable"
    (cd "$WORK/tpu-inference" && uv pip install -e . --no-deps) 2>&1 | tee -a "$SETUP_LOG"
else
    log "tpu_inference editable install already points at worktree; skipping"
fi

# gcsfuse is optional (only needed for the real-weight deploy gate); warn but
# don't fail if it's missing.
if ! command -v gcsfuse >/dev/null 2>&1; then
    log "WARN: gcsfuse not on PATH — Tier 8 real-weight gate will be unavailable until installed"
fi

log "complete"

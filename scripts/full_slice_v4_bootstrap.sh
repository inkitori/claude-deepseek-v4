#!/usr/bin/env bash
# Fan-out bootstrap: clone repo + setup venv + mount GCS + start Ray on
# every host of a freshly-provisioned v6e-32 slice.
#
# Idempotent: safe to re-run; skips workers that are already set up.
#
# Prereqs on the head:
#   * .env populated (CLAUDE_CODE_OAUTH_TOKEN, HF_TOKEN; GCS_BUCKET +
#     GCS_ONLY_DIR + MOUNT_GCS=1 if you want weights mounted).
#   * SSH access to each worker via ~/.ssh/google_compute_engine.
#   * uv, git on PATH.
#   * gcsfuse on PATH if MOUNT_GCS=1.
#
# Prereqs on each worker (your cloud-init / infra must install these
# — this script will NOT bootstrap OS-level deps):
#   * uv, git on PATH at the same paths as the head.
#   * gcsfuse on PATH if MOUNT_GCS=1.
#   * SSH from head reachable.
#   * Same Linux user as the head (so absolute paths line up).
#
# Usage:
#   scripts/full_slice_v4_bootstrap.sh
#
# Environment:
#   HEAD_IP   default: auto-discovered from GCP metadata (worker 0)
#   WORKERS   default: auto-discovered from GCP metadata (workers 1..N)

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO/.env"
HEAD_IP="${HEAD_IP:-$("$REPO/scripts/full_slice_v4_discover.sh" head)}"
WORKERS="${WORKERS:-$("$REPO/scripts/full_slice_v4_discover.sh" workers)}"
SSH_OPTS="-i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

log() { echo "[bootstrap $(date -u +%H:%M:%SZ)] $*"; }
fail() { echo "[bootstrap FATAL] $*" >&2; exit 1; }

[ -f "$ENV_FILE" ] || fail "$ENV_FILE missing — copy .env.example and fill in tokens"
# shellcheck disable=SC1091
set -a; source "$ENV_FILE"; set +a

# ---- 1. Head setup -----------------------------------------------------
log "head setup ($HEAD_IP)"
"$REPO/scripts/setup.sh" 2>&1 | tail -3 | sed 's/^/  /'

# ---- 2. Fan out to workers --------------------------------------------
# We rsync the repo (excluding the venv + logs + .env) to each worker, then
# run setup.sh remotely. Each worker builds its own venv from scratch — that
# parallelizes over the 7 workers and is faster + cleaner than rsyncing the
# venv binary tree.
RSYNC_EXCLUDES=(
    --exclude '.git/objects/pack'
    --exclude 'work/vllm_env'
    --exclude 'work/scratch'
    --exclude 'logs'
    --exclude '.env'
    --exclude '__pycache__'
    --exclude '*.pyc'
)

worker_setup_one() {
    local h="$1"
    log "[$h] ssh check"
    if ! ssh $SSH_OPTS "enyouki@$h" "true" 2>&1; then
        log "[$h] FAIL: ssh unreachable"
        return 1
    fi
    log "[$h] rsync repo -> ~/claude-deepseek-v4/"
    ssh $SSH_OPTS "enyouki@$h" "mkdir -p $HOME/claude-deepseek-v4" 2>&1 | sed 's/^/  /'
    rsync -e "ssh $SSH_OPTS" -ac --delete "${RSYNC_EXCLUDES[@]}" \
          "$REPO/" "enyouki@$h:$HOME/claude-deepseek-v4/" 2>&1 | tail -2 | sed 's/^/  /'
    log "[$h] rsync .env (workers need it for HF_TOKEN + gcs mount params)"
    rsync -e "ssh $SSH_OPTS" -ac "$ENV_FILE" "enyouki@$h:$HOME/claude-deepseek-v4/.env" 2>&1 | sed 's/^/  /'
    log "[$h] running scripts/setup.sh (this builds the venv; ~5 min)"
    ssh $SSH_OPTS "enyouki@$h" "cd $HOME/claude-deepseek-v4 && ./scripts/setup.sh" 2>&1 | tail -3 | sed 's/^/  /'
    if [ "${MOUNT_GCS:-0}" = "1" ]; then
        log "[$h] mounting GCS"
        ssh $SSH_OPTS "enyouki@$h" "cd $HOME/claude-deepseek-v4 && ./scripts/mount_gcs.sh" 2>&1 | tail -3 | sed 's/^/  /'
    fi
    log "[$h] OK"
}

failed_workers=()
for h in $WORKERS; do
    if ! worker_setup_one "$h"; then
        failed_workers+=("$h")
    fi
done

# ---- 3. Mount GCS on the head (if not already done by run.sh) ---------
if [ "${MOUNT_GCS:-0}" = "1" ]; then
    if ! mountpoint -q "${HF_HUB_MOUNT_DIR:-$HOME/.cache/huggingface/hub}" 2>/dev/null; then
        log "head: mounting GCS"
        "$REPO/scripts/mount_gcs.sh" 2>&1 | tail -3 | sed 's/^/  /'
    fi
fi

# ---- 4. Start Ray cluster ---------------------------------------------
if [ ${#failed_workers[@]} -gt 0 ]; then
    log "WARN: ${#failed_workers[@]} worker(s) failed setup: ${failed_workers[*]}"
    log "WARN: skipping ray-restart — fix workers first, then re-run this script"
    exit 1
fi
log "starting Ray cluster across all 8 hosts"
HEAD_IP="$HEAD_IP" WORKERS="$WORKERS" "$REPO/scripts/full_slice_v4_ray_restart.sh" 2>&1 | tail -10 | sed 's/^/  /'

# ---- 5. Optionally warm the JAX compile cache --------------------------
# Set WARM_CACHE_ON_BOOTSTRAP=1 in .env to fire one full smoke + curl
# right now so the first "real" `./run.sh serve` hits a populated cache.
# Adds ~10-15 min to bootstrap but every subsequent serve's first curl
# is sub-minute instead of cold-compile (5-15 min).
if [ "${WARM_CACHE_ON_BOOTSTRAP:-0}" = "1" ]; then
    log "WARM_CACHE_ON_BOOTSTRAP=1 — running one warm-up serve cycle to populate"
    log "  /tmp/jax-compile-cache-v4 on every host (~10-15 min one-time cost)"
    "$REPO/scripts/full_slice_v4_warm_cache.sh" 2>&1 | tail -15 | sed 's/^/  /'
else
    log "skipping warm-cache (set WARM_CACHE_ON_BOOTSTRAP=1 in .env to enable)"
fi

log "complete — cluster ready. Next: ./run.sh serve"

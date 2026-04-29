#!/usr/bin/env bash
# Sync the local tpu-inference + scripts source tree to every worker host.
#
# Each worker host has its own clone of ~/claude-deepseek-v4, so a `git push`
# from the head doesn't propagate code changes to workers — and Ray actors
# spawned on workers will run whatever code is on each worker's disk. That
# silently makes "code changes" only apply to 1/8 of the cluster.
#
# This rsync's only the directories that affect runtime (tpu-inference and
# the scripts dir), not the heavyweight venv (already synced once during
# bootstrap) or scratch fixtures.
#
# Usage: scripts/full_slice_v4_sync.sh
#
# Reads:
#   WORKERS  — space-separated worker IPs

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${WORKERS:-10.164.0.22 10.164.0.35 10.164.0.36 10.164.0.39 10.164.0.45 10.164.0.18 10.164.0.30}"
SSH_OPTS="-i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

log() { echo "[sync] $*"; }

paths=(
    "$REPO/work/tpu-inference/tpu_inference/"
    "$REPO/scripts/"
)

for h in $WORKERS; do
    log "syncing -> $h"
    for src in "${paths[@]}"; do
        # Mirror the same absolute path on the remote.
        dest="${src/\/home\/enyouki\/claude-deepseek-v4\//\/home\/enyouki\/claude-deepseek-v4\/}"
        rsync -e "ssh $SSH_OPTS" -ac --delete --exclude '__pycache__' --exclude '*.pyc' \
              "$src" "enyouki@$h:$dest" 2>&1 | tail -2 | sed 's/^/  /'
    done
done

log "verification: slice-aware code present on every host?"
for h in 10.164.0.41 $WORKERS; do
    if [ "$h" = "10.164.0.41" ]; then
        n=$(grep -c "iter_v4_safetensors_specs\|place_spec_as_jax_sharded\|slice_aware" \
            "$REPO/work/tpu-inference/tpu_inference/models/jax/deepseek_v4_loader.py" \
            "$REPO/work/tpu-inference/tpu_inference/models/jax/deepseek_v4.py" 2>/dev/null \
            | awk -F: '{s+=$2} END {print s}')
    else
        n=$(ssh $SSH_OPTS enyouki@"$h" \
            'grep -c "iter_v4_safetensors_specs\|place_spec_as_jax_sharded\|slice_aware" /home/enyouki/claude-deepseek-v4/work/tpu-inference/tpu_inference/models/jax/deepseek_v4_loader.py /home/enyouki/claude-deepseek-v4/work/tpu-inference/tpu_inference/models/jax/deepseek_v4.py 2>/dev/null | awk -F: "{s+=\$2} END {print s}"')
    fi
    if [ "${n:-0}" -ge 5 ]; then
        log "  [$h] OK ($n refs)"
    else
        log "  [$h] FAIL: only $n refs to slice-aware code"
    fi
done

#!/usr/bin/env bash
# Hard-reset the Ray cluster across the v6e-32 slice. Heavier than
# scripts/full_slice_v4_reset.sh (which only flushes lockfiles + orphan
# actors); this also `ray stop --force` + `ray start` on all 8 hosts so
# the underlying Ray worker pool is recreated fresh — which is required
# when libtpu state has been corrupted by SIGKILL'd EngineCore processes
# and subsequent init_device calls SIGSEGV on a clean lockfile.
#
# Usage:
#   scripts/full_slice_v4_ray_restart.sh
#
# Reads:
#   HEAD_IP   — head node ip (default: auto-discovered)
#   WORKERS   — space-separated worker ips
#   VENV      — path to vllm venv

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/work/vllm_env}"
HEAD_IP="${HEAD_IP:-$("$REPO/scripts/full_slice_v4_discover.sh" head)}"
WORKERS="${WORKERS:-$("$REPO/scripts/full_slice_v4_discover.sh" workers)}"
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

ray_bin="$VENV/bin/ray"

log() { echo "[ray-restart] $*"; }

log "stopping ray + cleaning libtpu state on head ($HEAD_IP)"
"$ray_bin" stop --force 2>&1 | tail -3 | sed 's/^/  /'
# Kill any leftover EngineCore not reaped by ray stop
pids=$(ps -eo pid,comm | awk '$2 == "VLLM::EngineCor" {print $1}')
[ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null
rm -f /tmp/libtpu_lockfile

for h in $WORKERS; do
    log "stopping ray + cleaning libtpu state on $h"
    $SSH enyouki@"$h" "
        $ray_bin stop --force 2>&1 | tail -3 | sed 's/^/  /'
        pids=\$(ps -eo pid,comm | awk '\$2 == \"VLLM::EngineCor\" {print \$1}')
        [ -n \"\$pids\" ] && echo \$pids | xargs kill -9 2>/dev/null
        rm -f /tmp/libtpu_lockfile
        echo '  cleaned'
    " 2>&1 | sed 's/^/  /' | tail -5
done

# Settle: give raylets a moment to fully exit before restart
sleep 4

log "starting ray head on $HEAD_IP"
"$ray_bin" start --head \
    --node-ip-address="$HEAD_IP" \
    --port=6379 \
    --resources='{"TPU":4,"TPU-v6e-16-head":1}' \
    --temp-dir=/tmp/ray-vllm \
    2>&1 | tail -8 | sed 's/^/  /'

# Wait for head to be reachable
sleep 3

for h in $WORKERS; do
    log "starting ray worker on $h"
    $SSH enyouki@"$h" "
        $ray_bin start \
            --address=$HEAD_IP:6379 \
            --node-ip-address=$h \
            --resources='{\"TPU\":4}' \
            --temp-dir=/tmp/ray-vllm
    " 2>&1 | tail -3 | sed 's/^/  /'
done

# Settle: ensure all 8 nodes register
sleep 5

log "verifying cluster"
RAY_ADDRESS="$HEAD_IP:6379" "$ray_bin" status 2>&1 | grep -E "node_|TPU|nodes" | sed 's/^/  /'

# Pass criterion: 16.0/16.0 TPU available (v6e-16 = 4 hosts x 4 chips)
got=$(RAY_ADDRESS="$HEAD_IP:6379" "$ray_bin" status 2>&1 | grep -oE '0\.0/[0-9]+\.0 TPU\b' | head -1)
log "TPU line: $got"
case "$got" in
    "0.0/16.0 TPU") log "OK: cluster up with 16 TPU"; exit 0 ;;
    *) log "FAIL: expected '0.0/16.0 TPU'"; exit 1 ;;
esac

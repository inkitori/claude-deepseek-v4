#!/usr/bin/env bash
# Guardian v3: keep mark's `node` ray-worker container permanently OUT by
# OCCUPYING the container NAME on every host with an inert dummy (see
# full_slice_v4_node_occupy.sh). The remote controller does
# `docker run --name node vllm/vllm-tpu...` (create+start, NEVER `rm` first —
# verified via `docker events`), so a pre-existing `node` makes its run fail
# with a name conflict and the real ray node can never join our ray / touch the
# TPU launch group -> no more SLICE_FAILURE_SW_INJECT_ERROR decode Core-halts.
#
# This loop runs the occupy script on every host every INTERVAL secs: it
# re-occupies if anything frees the name, and evicts + re-occupies if the real
# node ever wins the create race. The dummy is never started, so this loop never
# touches the TPU.
#
#   INTERVAL=3 setsid bash scripts/full_slice_v4_node_guardian.sh >logs/node_guardian.log 2>&1 </dev/null &
#
# Env: INTERVAL (default 3), HEAD_IP / WORKERS override discovery.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEAD_IP="${HEAD_IP:-$("$REPO/scripts/full_slice_v4_discover.sh" head)}"
WORKERS="${WORKERS:-$("$REPO/scripts/full_slice_v4_discover.sh" workers)}"
INTERVAL="${INTERVAL:-3}"
SSH_OPTS="-i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6"
# Each host runs its own clone at ~/claude-deepseek-v4 (Pitfall 1); the occupy
# script is rsync'd there by full_slice_v4_sync.sh and survives reboot.
OCCUPY='bash ~/claude-deepseek-v4/scripts/full_slice_v4_node_occupy.sh'

echo "[guardian v3] occupying 'node' name on $HEAD_IP $WORKERS every ${INTERVAL}s (Ctrl-C to stop)"
while true; do
    for h in $HEAD_IP $WORKERS; do
        { out=$(ssh $SSH_OPTS enyouki@"$h" "$OCCUPY" 2>/dev/null);
          [ -n "$out" ] && echo "[guardian] $out node on $h at $(date +%H:%M:%S)"; } &
    done
    wait
    sleep "$INTERVAL"
done

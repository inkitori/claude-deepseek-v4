#!/usr/bin/env bash
# Guardian: re-stop mark's reappearing `node` container (shared-slice tenant —
# a ray worker that joins ray at our head 10.164.0.192:6379 with
# restart=unless-stopped, and which a remote controller periodically REDEPLOYS)
# on every host, every INTERVAL seconds. `node` joining our ray cluster fights
# over the exclusive TPU slice; see CLAUDE.md Pitfall 0.
#
# Run in the background during any TPU work and kill it when done:
#   scripts/full_slice_v4_node_guardian.sh &   (or via Bash run_in_background)
# Loops forever; never exits on its own.
#
# Env: INTERVAL (secs, default 25), HEAD_IP / WORKERS override discovery.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEAD_IP="${HEAD_IP:-$("$REPO/scripts/full_slice_v4_discover.sh" head)}"
WORKERS="${WORKERS:-$("$REPO/scripts/full_slice_v4_discover.sh" workers)}"
INTERVAL="${INTERVAL:-25}"
SSH_OPTS="-i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8"

echo "[guardian] re-stopping 'node' on $HEAD_IP $WORKERS every ${INTERVAL}s (Ctrl-C to stop)"
while true; do
    stopped=0
    for h in $HEAD_IP $WORKERS; do
        ssh $SSH_OPTS enyouki@"$h" \
            'if sudo docker ps --filter name=^/node$ --format "{{.Names}}" 2>/dev/null | grep -q node; then
                 sudo docker update --restart=no node >/dev/null 2>&1
                 sudo docker stop -t3 node >/dev/null 2>&1
                 echo restopped
             fi' 2>/dev/null | grep -q restopped && { echo "[guardian] re-stopped node on $h at $(date +%H:%M:%S)"; stopped=1; }
    done
    sleep "$INTERVAL"
done

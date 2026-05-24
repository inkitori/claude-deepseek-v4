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
INTERVAL="${INTERVAL:-3}"
SSH_OPTS="-i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6"

# AGGRESSIVE: force-REMOVE `node` (not just stop) on every host every INTERVAL secs,
# in PARALLEL. The remote controller REDEPLOYS `node` (vllm/vllm-tpu:nightly, ray
# 2.54.1); it thrashes (joins our ray -> version-mismatch Exit 1 -> restart loop),
# touching the TPU and corrupting the launch group -> SLICE_FAILURE_SW_INJECT_ERROR
# (the decode Core-halts traced to this). `rm -f` neutralizes any state
# (created/running/restarting) and beats `docker run --name node` (name stays taken
# only until removed). A reboot won't auto-start a removed container. Short interval
# minimizes the TPU-touch window before a sensitive 32-way decode collective.
echo "[guardian] rm -f 'node' on $HEAD_IP $WORKERS every ${INTERVAL}s (Ctrl-C to stop)"
while true; do
    for h in $HEAD_IP $WORKERS; do
        ssh $SSH_OPTS enyouki@"$h" \
            'if sudo docker ps -a --filter name=^/node$ --format "{{.Names}}" 2>/dev/null | grep -q node; then
                 sudo docker rm -f node >/dev/null 2>&1; echo removed
             fi' 2>/dev/null | grep -q removed && echo "[guardian] rm -f node on $h at $(date +%H:%M:%S)" &
    done
    wait
    sleep "$INTERVAL"
done

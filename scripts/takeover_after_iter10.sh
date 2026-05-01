#!/usr/bin/env bash
# One-shot watcher: waits for iter 10's `claude -p` to exit, then kills
# the autonomous loop so the user can drive iter 11 manually.
#
# Writes status to logs/takeover.log so it can be tailed, and exits 0
# when the loop is stopped.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO_DIR/logs/takeover.log"
MARKER="$REPO_DIR/logs/takeover.done"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

log "watcher started; waiting for iter 10's claude -p to exit"

# Capture iter 10's claude PID. Loop until we find one (handle the case
# where this script starts during the 30s loop-sleep).
ITER_PID=""
for _ in $(seq 1 12); do
    ITER_PID=$(pgrep -f 'claude -p .*Agent prompt' | head -1 || true)
    [ -n "$ITER_PID" ] && break
    sleep 5
done

if [ -z "$ITER_PID" ]; then
    log "no claude -p process found; loop may already be between iters"
else
    log "watching claude -p PID=$ITER_PID"
    while kill -0 "$ITER_PID" 2>/dev/null; do
        sleep 15
    done
    log "claude -p PID=$ITER_PID has exited"
fi

# Now kill the loop manager during its sleep window.
LOOP_PID=$(pgrep -f 'bash scripts/loop.sh' | head -1 || true)
if [ -n "$LOOP_PID" ]; then
    log "killing loop.sh PID=$LOOP_PID"
    kill -TERM "$LOOP_PID" 2>/dev/null || true
    sleep 3
    if kill -0 "$LOOP_PID" 2>/dev/null; then
        log "loop.sh did not exit on TERM; sending KILL"
        kill -KILL "$LOOP_PID" 2>/dev/null || true
    fi
else
    log "loop.sh not running (already stopped?)"
fi

# Belt-and-suspenders: kill any lingering claude -p / timeout descendants.
pkill -f 'timeout.*claude -p' 2>/dev/null || true
pkill -f 'claude -p .*Agent prompt' 2>/dev/null || true

# Final state report.
log "git tip: $(git -C "$REPO_DIR" log --oneline -1)"
log "git status: $(git -C "$REPO_DIR" status --porcelain | wc -l) modified files"
log "remaining processes:"
pgrep -af 'loop.sh|claude -p|vllm serve|EngineCore' | tee -a "$LOG" || log "(none)"

touch "$MARKER"
log "takeover complete; marker written to $MARKER"
log "next step: cd $REPO_DIR && claude"

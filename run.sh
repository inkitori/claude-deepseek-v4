#!/usr/bin/env bash
# Host-direct launcher for the DeepSeek-V4 / vLLM / tpu-inference agent loop.
# No Docker. Designed for a single TPU VM (v6e-{4,8,16,32}-host) with the
# repo checked out at $HOME or wherever convenient.
#
# What it does:
#   1. Reads .env (CLAUDE_CODE_OAUTH_TOKEN, optional HF_TOKEN, optional GCS_*)
#   2. Runs scripts/setup.sh — bootstraps work/vllm_env and installs vllm +
#      tpu-inference (idempotent; fast no-op on re-run).
#   3. Optionally mounts the GCS weight bucket onto ~/.cache/huggingface/hub
#      via scripts/mount_gcs.sh, if MOUNT_GCS=1 in .env.
#   4. Runs scripts/preflight.sh — JAX TPU sanity, writes logs/tpu-preflight.log.
#   5. Backgrounds scripts/loop.sh (nohup), writes logs/loop.pid, returns.
#
# Re-running run.sh is safe: it'll bail with a clear error if a loop is
# already alive (use `./run.sh stop` to terminate, then re-launch).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$REPO_DIR/logs"
PID_FILE="$LOGS/loop.pid"

mkdir -p "$LOGS"

cmd="${1:-start}"

is_alive() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "$cmd" in
stop)
    if is_alive; then
        pid="$(cat "$PID_FILE")"
        echo "stopping loop pgid=$pid (group-kill, reaps children)"
        # The loop runs in its own session (setsid) — kill the whole group.
        kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
        fi
    else
        echo "no running loop"
    fi
    rm -f "$PID_FILE"
    exit 0
    ;;
status)
    if is_alive; then
        echo "loop alive pid=$(cat "$PID_FILE")"
    else
        echo "loop not running"
    fi
    exit 0
    ;;
start)
    : # fall through
    ;;
*)
    echo "usage: $0 [start|stop|status]" >&2
    exit 2
    ;;
esac

if is_alive; then
    echo "loop already running (pid=$(cat "$PID_FILE")). './run.sh stop' first." >&2
    exit 1
fi

if ! [ -f "$REPO_DIR/.env" ]; then
    echo "missing $REPO_DIR/.env — copy .env.example and fill in tokens" >&2
    exit 1
fi
# shellcheck disable=SC1091
set -a; source "$REPO_DIR/.env"; set +a

if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    cat >&2 <<EOF
CLAUDE_CODE_OAUTH_TOKEN is empty in .env.

To populate it:
  1. On this host (logged in as your user), run:  claude setup-token
  2. Copy the printed token.
  3. Edit $REPO_DIR/.env and set:
         CLAUDE_CODE_OAUTH_TOKEN=<paste>
  4. Re-run this script.
EOF
    exit 1
fi

# 1. Bootstrap venv + deps
"$REPO_DIR/scripts/setup.sh"

# 2. Optional GCS mount for real-weight deploy gate
if [ "${MOUNT_GCS:-0}" = "1" ]; then
    if [ -z "${GCS_BUCKET:-}" ] || [ -z "${GCS_ONLY_DIR:-}" ]; then
        echo "WARN: MOUNT_GCS=1 but GCS_BUCKET / GCS_ONLY_DIR not set; skipping mount" >&2
    else
        "$REPO_DIR/scripts/mount_gcs.sh"
    fi
fi

# 3. TPU pre-flight
"$REPO_DIR/scripts/preflight.sh" || true   # never block on preflight failure

# 4. Background the loop. Use `setsid` so loop.sh becomes its own session/
# process-group leader; that lets `./run.sh stop` kill the entire subtree
# (loop + timeout + claude + any agent-spawned pytest) by signaling the
# group, instead of orphaning grandchildren.
setsid nohup "$REPO_DIR/scripts/loop.sh" >>"$LOGS/loop.out" 2>&1 < /dev/null &
loop_pid=$!
echo "$loop_pid" > "$PID_FILE"
disown "$loop_pid" 2>/dev/null || true

cat <<EOF
loop started (pid=$loop_pid).

  live wrapper output:    tail -f $LOGS/loop.out
  live claude iteration:  tail -f $LOGS/iter-*.log
  setup progress:         tail -f $LOGS/setup.log
  loop events + push log: tail -f $LOGS/loop.log
  status snapshot:        ./monitor.sh status
  tail tool calls:        ./monitor.sh tools
  stop the loop:          ./run.sh stop
  loop status:            ./run.sh status

Resilience:
  - On rate limit / 5h-window reset, the loop parses the reset time and
    sleeps until reset+5min (fallback: 60min). Won't burn iterations
    against a wall.
  - Commits made by the agent are pushed to ${PUSH_REMOTE:-origin}/${PUSH_BRANCH:-main}
    at the end of every iter (set PUSH_ENABLED=0 in .env to disable).
    The agent is also instructed to push after every commit, so work is
    preserved even if the iter is killed mid-flight.
  - Loop does NOT auto-restart on host reboot — re-run \`./run.sh\`.

EOF

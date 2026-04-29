#!/usr/bin/env bash
# Reset cluster state between v6e-32 V4-Flash smoke runs.
#
# Cleans up the four orphan-state surfaces that bite a relaunch:
#   1. The api-server pid recorded in logs/full-slice-v4-smoke.pid
#      (kill by exact pid; never a broad regex per
#      memory/feedback_ray_cleanup.md).
#   2. Stale VLLM::EngineCore Ray actor children — these don't get
#      reaped when the api-server is SIGKILLed and they continue to
#      hold TPU resources via libtpu.
#   3. /tmp/libtpu_lockfile on each of the 8 hosts — left behind by
#      a SIGKILL'd EngineCore; libtpu refuses to initialize while it
#      exists.
#   4. Leaked Ray placement groups in the CREATED state.
#
# Does NOT touch raylet daemons or the Ray cluster itself. Assumes Ray
# is already running and reachable at $RAY_ADDRESS (default
# 10.164.0.41:6379) and that ssh ~/.ssh/google_compute_engine works
# host-to-host as the current user.
#
# Usage:
#   scripts/full_slice_v4_reset.sh            # full reset
#   QUIET=1 scripts/full_slice_v4_reset.sh    # only print errors
#
# Env vars (override defaults):
#   WORKERS  — space-separated worker IPs
#   RAY_ADDRESS — Ray head address (default 10.164.0.41:6379)
#   VENV     — path to vllm venv (default $REPO/work/vllm_env)

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/work/vllm_env}"
RAY_ADDRESS="${RAY_ADDRESS:-10.164.0.41:6379}"
WORKERS="${WORKERS:-10.164.0.22 10.164.0.35 10.164.0.36 10.164.0.39 10.164.0.45 10.164.0.18 10.164.0.30}"
PIDFILE="$REPO/logs/full-slice-v4-smoke.pid"
QUIET="${QUIET:-0}"

log() { [ "$QUIET" = "1" ] || echo "$@"; }
err() { echo "$@" >&2; }

log "[reset] === step 1: kill recorded api-server pid ==="
if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE" 2>/dev/null)
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null && log "[reset]   killed api-server pid=$pid"
        sleep 2
    else
        log "[reset]   pid $pid not running"
    fi
else
    log "[reset]   no pidfile; skipping"
fi

log "[reset] === step 2: kill orphan VLLM::EngineCore (head + 7 workers) ==="
killed_head=$(ps -eo pid,comm | awk '$2 == "VLLM::EngineCor" {print $1}')
if [ -n "$killed_head" ]; then
    n=$(echo "$killed_head" | wc -w)
    echo "$killed_head" | xargs kill -9 2>/dev/null
    log "[reset]   head: killed $n EngineCore"
fi
for h in $WORKERS; do
    n=$(ssh -i ~/.ssh/google_compute_engine -o BatchMode=yes \
            -o StrictHostKeyChecking=no -o ConnectTimeout=5 enyouki@"$h" '
        pids=$(ps -eo pid,comm | awk "\$2 == \"VLLM::EngineCor\" {print \$1}")
        [ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null
        echo "$pids" | wc -w
    ' 2>/dev/null | tail -1)
    log "[reset]   $h: killed $n EngineCore"
done

log "[reset] === step 3: remove /tmp/libtpu_lockfile ==="
rm -f /tmp/libtpu_lockfile && log "[reset]   head: removed (or absent)"
for h in $WORKERS; do
    ssh -i ~/.ssh/google_compute_engine -o BatchMode=yes \
        -o StrictHostKeyChecking=no -o ConnectTimeout=5 enyouki@"$h" \
        'rm -f /tmp/libtpu_lockfile' 2>/dev/null
    log "[reset]   $h: removed (or absent)"
done

log "[reset] === step 4: remove leaked CREATED placement groups ==="
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python3" - <<'PY' 2>&1 | sed 's/^/[reset]   /' || true
import sys
try:
    import ray
    from ray.util.placement_group import PlacementGroup
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
    pgs = ray.util.placement_group_table()
    n_removed = 0
    for pg_id_hex, info in pgs.items():
        if info.get("state") == "CREATED":
            try:
                pg = PlacementGroup(ray.PlacementGroupID(bytes.fromhex(pg_id_hex)))
                ray.util.remove_placement_group(pg)
                n_removed += 1
            except Exception as e:
                print(f"failed to remove {pg_id_hex[:16]}: {e}", file=sys.stderr)
    print(f"removed {n_removed} CREATED PG(s); table size: {len(pgs)}")
except Exception as e:
    print(f"PG cleanup error: {type(e).__name__}: {e}", file=sys.stderr)
PY

log "[reset] === ray status ==="
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status 2>/dev/null | grep -E "TPU|nodes|failures" | sed 's/^/[reset]   /' || true

log "[reset] done."

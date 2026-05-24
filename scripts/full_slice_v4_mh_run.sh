#!/usr/bin/env bash
# Launch a Python script as a coordinated multi-host jax.distributed job across
# ALL 8 TPU hosts of the slice (head = process 0). The script itself is expected
# to call no-arg `jax.distributed.initialize()` (Cloud-TPU metadata auto-detect),
# so this launcher only fans the script out near-simultaneously, cleans the
# stale libtpu lockfile (Pitfall 3), enforces a per-process timeout, and collects
# per-process logs.
#
# This is the CHEAP multi-host inner loop for S1: a small (random-weight, few
# layer) V4 config still needs all 8 hosts to bring up the v6e-32 TPU, but
# compiles in minutes with no 543 GiB weight load.
#
# Usage: scripts/full_slice_v4_mh_run.sh <repo-relative-script.py> [args...]
# Env:
#   MH_TIMEOUT  per-process remote timeout in seconds (default 600)
#   MH_ENV      extra "K=V K=V" env prefixed to each process (e.g. flags)
#   HEAD_IP / WORKERS  override slice discovery
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEAD_IP="${HEAD_IP:-$("$REPO/scripts/full_slice_v4_discover.sh" head)}"
WORKERS="${WORKERS:-$("$REPO/scripts/full_slice_v4_discover.sh" workers)}"
MH_TIMEOUT="${MH_TIMEOUT:-600}"
MH_ENV="${MH_ENV:-}"

SCRIPT="${1:?usage: full_slice_v4_mh_run.sh <script.py> [args...]}"; shift || true
ARGS="$*"
SSH_OPTS="-i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

HOSTS=("$HEAD_IP" $WORKERS)   # array index == jax process_id (head = 0)
NP=${#HOSTS[@]}
LOGDIR="$REPO/logs"; mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d-%H%M%S)

CMD_ENV="JAX_PLATFORMS=tpu PYTHONPATH=work/tpu-inference:work/vllm $MH_ENV"
RUN="rm -f /tmp/libtpu_lockfile 2>/dev/null; cd $REPO && env $CMD_ENV timeout $MH_TIMEOUT work/vllm_env/bin/python $SCRIPT $ARGS"

echo "[mh] launching '$SCRIPT $ARGS' across $NP hosts (timeout ${MH_TIMEOUT}s, env: $MH_ENV)"
pids=(); logs=()
for i in "${!HOSTS[@]}"; do
    h="${HOSTS[$i]}"
    log="$LOGDIR/mh-${TS}-p${i}-${h}.log"; logs+=("$log")
    if [ "$h" = "$HEAD_IP" ]; then
        bash -c "$RUN" > "$log" 2>&1 &
    else
        ssh $SSH_OPTS enyouki@"$h" "$RUN" > "$log" 2>&1 &
    fi
    pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then st="ok"; else st="FAIL(rc=$?)"; fail=1; fi
    echo "[mh] p$i ${HOSTS[$i]}: $st  -> ${logs[$i]}"
done

echo "[mh] === process 0 (head) tail ==="
tail -40 "${logs[0]}"
exit $fail

#!/usr/bin/env bash
# Launcher for the decode HOST-DISPATCH micro-benchmark (roadmap L1 -- the cheap
# perf inner loop). v6e-16 = 16 chips / 4 hosts / 4x4; the slice cannot boot TPU
# from a lone host, so this runs the MULTI-HOST distributed path across all 4
# hosts via full_slice_v4_mh_run.sh and builds the real 16-device mesh.
#
# A FAILED distributed run leaves JAX procs stuck in the coordination wait -- they
# ignore SIGTERM and POISON the next init. So we SIGKILL any perf_microbench_decode
# stragglers and clear /tmp/libtpu_lockfile on EVERY host FIRST. (The `[p]` glob
# keeps pkill from matching its own ssh command line.) TPU must be FREE first:
# `scripts/full_slice_v4_reset.sh`. SYNC to 4 hosts first (mh_run runs each clone).
#
#   scripts/perf_microbench_decode.sh --sweep         # leaf-count sweep (the table)
#   scripts/perf_microbench_decode.sh --n-leaves 1492
#   MH_TIMEOUT=300 scripts/perf_microbench_decode.sh --sweep
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/work/tpu-inference:$REPO/work/vllm"

# Pre-clean every host so a prior failed run can't poison this init.
HEAD="$("$REPO/scripts/full_slice_v4_discover.sh" head)"
WORKERS="$("$REPO/scripts/full_slice_v4_discover.sh" workers)"
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"
# NB: match the .py target ONLY -- this launcher is perf_microbench_decode.sh, so a
# bare 'perf_microbench_decode' pattern would SIGKILL the launcher itself (the [p]
# trick only spares the pkill command line, not the .sh that shares the prefix).
CLEAN="pkill -9 -f '[p]erf_microbench_decode\.py' 2>/dev/null; rm -f /tmp/libtpu_lockfile 2>/dev/null; true"
for h in $HEAD $WORKERS; do
  if [ "$h" = "$HEAD" ]; then bash -c "$CLEAN"; else $SSH enyouki@"$h" "$CLEAN" 2>/dev/null; fi
done

exec env MH_TIMEOUT="${MH_TIMEOUT:-300}" bash "$REPO/scripts/full_slice_v4_mh_run.sh" \
  scripts/perf_microbench_decode.py "$@" --distributed

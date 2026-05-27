#!/usr/bin/env bash
# Launcher for the sparse_attn micro-benchmark (Phase 0.0 — the cheap perf inner
# loop). The v6e-32 slice CANNOT boot TPU from a lone host, so this runs the
# MULTI-HOST distributed path across all 8 hosts via full_slice_v4_mh_run.sh
# (each process times its LOCAL per-device op; sparse_attn has no collectives).
#
# A FAILED distributed run leaves JAX procs stuck in the coordination wait — they
# ignore mh_run's SIGTERM and then POISON the next init (it hangs). So we SIGKILL
# any `perf_microbench` stragglers and clear /tmp/libtpu_lockfile on EVERY host
# FIRST. (The `[p]` glob keeps pkill from matching its own ssh command line.)
#
#   scripts/perf_microbench.sh --all                  # all 6 decode/prefill presets
#   scripts/perf_microbench.sh --preset prefill_csa --seq 4096
#   scripts/perf_microbench.sh --cpu-check            # CPU numerics gate (existing pytest)
#   MH_TIMEOUT=300 scripts/perf_microbench.sh --all   # per-process timeout override
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/work/vllm_env/bin/python3"
export PYTHONPATH="$REPO/work/tpu-inference:$REPO/work/vllm"

if [ "${1:-}" = "--cpu-check" ]; then
  JAX_PLATFORMS=cpu "$PY" -m pytest -q \
    "$REPO/work/tpu-inference/tests/models/jax/test_deepseek_v4.py::TestSparseAttnComponent"
  exit $?
fi

# Pre-clean every host so a prior failed run can't poison this init.
HEAD="$("$REPO/scripts/full_slice_v4_discover.sh" head)"
WORKERS="$("$REPO/scripts/full_slice_v4_discover.sh" workers)"
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"
CLEAN="pkill -9 -f '[p]erf_microbench_sparse_attn' 2>/dev/null; rm -f /tmp/libtpu_lockfile 2>/dev/null; true"
for h in $HEAD $WORKERS; do
  if [ "$h" = "$HEAD" ]; then bash -c "$CLEAN"; else $SSH enyouki@"$h" "$CLEAN" 2>/dev/null; fi
done

exec env MH_TIMEOUT="${MH_TIMEOUT:-300}" bash "$REPO/scripts/full_slice_v4_mh_run.sh" \
  scripts/perf_microbench_sparse_attn.py "$@" --distributed

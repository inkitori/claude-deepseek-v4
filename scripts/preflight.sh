#!/usr/bin/env bash
# TPU pre-flight: import jax, count chips, write a single-line JSON status to
# logs/tpu-preflight.log. The agent reads this on iter 1 to decide whether
# real-TPU tiers can run.
#
# Sets the v6e single-VM env vars before invoking JAX so this VM only sees
# its 4 local chips even though it's part of a larger pod (e.g. v6e-32). If
# you don't set these, jax.devices('tpu') hangs >60s waiting for non-existent
# peer hosts.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_DIR/work/vllm_env"
LOGS="$REPO_DIR/logs"
PREFLIGHT_LOG="$LOGS/tpu-preflight.log"

mkdir -p "$LOGS"

# v6e single-VM init (4 chips local). Override via .env if you need different
# topology semantics; do NOT remove without testing — JAX will hang waiting
# for absent peers otherwise.
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python - <<'PY' >"$PREFLIGHT_LOG" 2>&1
import json, time, os
t0 = time.time()
out = {}
try:
    import jax
    devs = jax.devices()
    tpu = [d for d in devs if d.platform == "tpu"]
    out = {
        "ok": len(tpu) == 4,
        "n_tpu": len(tpu),
        "kinds": [getattr(d, "device_kind", "?") for d in tpu],
        "tpu_host_bounds": os.environ.get("TPU_HOST_BOUNDS"),
        "tpu_chips_per_host_bounds": os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS"),
        "elapsed_s": round(time.time() - t0, 2),
    }
except Exception as e:
    out = {"ok": False, "error": repr(e), "elapsed_s": round(time.time() - t0, 2)}
print(json.dumps(out))
PY

head -n1 "$PREFLIGHT_LOG"

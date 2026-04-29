#!/usr/bin/env bash
set -uo pipefail

WORK=/workspace/work
LOGS=/workspace/logs
PROMPT_FILE=/workspace/prompt.md
ENV_FILE=/workspace/.env

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    echo "FATAL: CLAUDE_CODE_OAUTH_TOKEN is empty." >&2
    echo "Run 'claude setup-token' on the host, paste the token into .env, then restart." >&2
    exit 1
fi

if [ ! -w "${HOME:-/home/agent}" ]; then
    export HOME="$(mktemp -d)"
fi

git config --global user.name  "${GIT_AUTHOR_NAME:-claude-overnight}"  >/dev/null 2>&1 || true
git config --global user.email "${GIT_AUTHOR_EMAIL:-claude@overnight.local}" >/dev/null 2>&1 || true
git config --global init.defaultBranch main >/dev/null 2>&1 || true
git config --global safe.directory '*'      >/dev/null 2>&1 || true

mkdir -p "$LOGS" "$WORK"
cd "$WORK"

setup_log="$LOGS/setup.log"

if [ ! -d "$WORK/tpu-inference/.git" ]; then
    echo "[setup $(date -u +%H%M%SZ)] cloning tpu-inference" | tee -a "$setup_log"
    git clone https://github.com/vllm-project/tpu-inference.git 2>&1 | tee -a "$setup_log"

    VLLM_COMMIT_HASH="$(tr -d '[:space:]' < tpu-inference/.buildkite/vllm_lkg.version)"
    echo "[setup] vllm pinned to $VLLM_COMMIT_HASH" | tee -a "$setup_log"

    if [ ! -d "$WORK/vllm/.git" ]; then
        echo "[setup $(date -u +%H%M%SZ)] cloning vllm" | tee -a "$setup_log"
        git clone https://github.com/vllm-project/vllm.git 2>&1 | tee -a "$setup_log"
        git -C vllm checkout "$VLLM_COMMIT_HASH" 2>&1 | tee -a "$setup_log"
    fi
fi

if [ ! -d "$WORK/vllm_env" ]; then
    echo "[setup $(date -u +%H%M%SZ)] creating vllm_env (python 3.12)" | tee -a "$setup_log"
    uv venv "$WORK/vllm_env" --python 3.12 2>&1 | tee -a "$setup_log"
fi

# shellcheck disable=SC1091
source "$WORK/vllm_env/bin/activate"

if ! python -c "import vllm" 2>/dev/null; then
    echo "[setup $(date -u +%H%M%SZ)] installing vllm (TPU target) — this takes a while" | tee -a "$setup_log"
    (cd "$WORK/vllm" \
        && uv pip install -r requirements/tpu.txt --torch-backend=cpu \
        && VLLM_TARGET_DEVICE=tpu uv pip install -e .) 2>&1 | tee -a "$setup_log"
fi

if ! python -c "import tpu_inference" 2>/dev/null; then
    echo "[setup $(date -u +%H%M%SZ)] installing tpu-inference" | tee -a "$setup_log"
    (cd "$WORK/tpu-inference" && uv pip install -e .) 2>&1 | tee -a "$setup_log"
fi

echo "[setup $(date -u +%H%M%SZ)] complete" | tee -a "$setup_log"

# TPU pre-flight — agent reads $LOGS/tpu-preflight.log on iter 1 to decide
# whether tiers 5/6 (real-TPU) can run, or fall back to CPU-only tiers 1-3.
preflight_log="$LOGS/tpu-preflight.log"
python - <<'PY' >>"$preflight_log" 2>&1
import json, time
t0 = time.time()
try:
    import jax
    devs = jax.devices()
    tpu = [d for d in devs if d.platform == "tpu"]
    out = {"ok": len(tpu) == 4, "n_tpu": len(tpu),
           "kinds": [getattr(d, "device_kind", "?") for d in tpu],
           "elapsed_s": round(time.time() - t0, 2)}
except Exception as e:
    out = {"ok": False, "error": repr(e), "elapsed_s": round(time.time() - t0, 2)}
print(json.dumps(out))
PY
echo "[preflight] $(tail -n1 "$preflight_log")" | tee -a "$setup_log"

PROMPT="$(cat "$PROMPT_FILE")"
ITER_TIMEOUT_SEC="${ITER_TIMEOUT_SEC:-5400}"
iter=1

while true; do
    ts="$(date -u +%Y%m%d-%H%M%SZ)"
    log="$LOGS/iter-$(printf '%04d' "$iter")-${ts}.log"
    echo "=== iter $iter start @ $ts ===" | tee -a "$log"

    set +e
    timeout --signal=TERM --kill-after=120s "$ITER_TIMEOUT_SEC" \
        claude -p "$PROMPT" \
            --model opus \
            --effort xhigh \
            --dangerously-skip-permissions \
            --output-format stream-json --verbose \
            >>"$log" 2>&1
    rc=$?
    set -e

    end_ts="$(date -u +%H%M%SZ)"
    [ "$rc" = "124" ] && echo "[loop] iter $iter hit ${ITER_TIMEOUT_SEC}s timeout" | tee -a "$log"
    echo "=== iter $iter end exit=$rc @ $end_ts ===" | tee -a "$log"

    status_file="$WORK/tpu-inference/STATUS.md"
    if [ -f "$status_file" ]; then
        echo "[loop] STATUS@$(date -u +%H%MZ): $(grep -E '^Latest|^TPU|^Tier|^W[1-4]' "$status_file" | tr '\n' ' | ')" \
            | tee -a "$LOGS/loop.log"
    fi

    if grep -qE '"type":"(rate_limit|overloaded|billing)_error"|"status":"throttled"|insufficient_quota|quota_exceeded' "$log"; then
        echo "[loop] usage/rate limit detected, sleeping 30 min" | tee -a "$log"
        sleep 1800
    elif [ "$rc" -ne 0 ]; then
        echo "[loop] non-zero exit ($rc), sleeping 2 min" | tee -a "$log"
        sleep 120
    else
        echo "[loop] clean exit, sleeping 30s before next iter" | tee -a "$log"
        sleep 30
    fi
    iter=$((iter + 1))
done

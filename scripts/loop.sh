#!/usr/bin/env bash
# Agent loop: invokes `claude -p prompt.md` repeatedly with a per-iter
# timeout, persists logs under logs/iter-<N>-<ts>.log, sleeps on
# rate-limit / non-zero-exit, pushes commits after each iter, and
# continues forever.
#
# State persists between iterations under work/ — that's what makes the
# loop resumable. Each iteration is a fresh `claude` invocation that
# reads the prompt and figures out where to pick up.
#
# Resilience features:
#   - ssh-agent + ed25519 key unlocked at start (via SSH_ASKPASS helper)
#   - git push origin main after every iter (non-blocking on failure)
#   - smart rate-limit / session-usage handling: parse reset timestamp
#     when present, sleep until reset+5min; fall back to 60min otherwise.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$REPO_DIR/work"
LOGS="$REPO_DIR/logs"
PROMPT_FILE="$REPO_DIR/prompt.md"
ENV_FILE="$REPO_DIR/.env"
VENV="$WORK/vllm_env"
LOOP_LOG="$LOGS/loop.log"

if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    echo "FATAL: CLAUDE_CODE_OAUTH_TOKEN missing in .env" >&2
    exit 1
fi
if [ ! -d "$VENV" ]; then
    echo "FATAL: $VENV does not exist; run scripts/setup.sh first" >&2
    exit 1
fi
if [ ! -f "$PROMPT_FILE" ]; then
    echo "FATAL: $PROMPT_FILE missing" >&2
    exit 1
fi

# v6e single-VM TPU env (see scripts/preflight.sh for rationale)
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu}"

# Make the venv visible to subprocesses the agent spawns.
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"

git config --global user.name  "${GIT_AUTHOR_NAME:-claude-overnight}"  >/dev/null 2>&1 || true
git config --global user.email "${GIT_AUTHOR_EMAIL:-claude@overnight.local}" >/dev/null 2>&1 || true
git config --global init.defaultBranch main >/dev/null 2>&1 || true
git config --global safe.directory '*'      >/dev/null 2>&1 || true

mkdir -p "$LOGS"

# ----- ssh-agent: unlock the key once so post-iter `git push` works -----
ensure_ssh_agent() {
    if [ -n "${SSH_AUTH_SOCK:-}" ] && ssh-add -l 2>/dev/null | grep -q ed25519; then
        echo "[loop] ssh-agent already has ed25519 key" | tee -a "$LOOP_LOG"
        return 0
    fi
    eval "$(ssh-agent -s)" >/dev/null
    if [ -x "$HOME/.ssh/ssh_askpass_helper.sh" ]; then
        SSH_ASKPASS="$HOME/.ssh/ssh_askpass_helper.sh" \
            SSH_ASKPASS_REQUIRE=force \
            DISPLAY=:0 \
            ssh-add "$HOME/.ssh/id_ed25519" </dev/null >/dev/null 2>&1
    else
        ssh-add "$HOME/.ssh/id_ed25519" </dev/null >/dev/null 2>&1 || true
    fi
    if ssh-add -l 2>/dev/null | grep -q ed25519; then
        echo "[loop] ssh-agent unlocked ed25519 key (pid=${SSH_AGENT_PID:-?})" | tee -a "$LOOP_LOG"
        echo "${SSH_AGENT_PID:-}" > "$LOGS/ssh-agent.pid"
    else
        echo "[loop] WARN: could not unlock ssh key — git push will be skipped" | tee -a "$LOOP_LOG"
    fi
}

# Kill the ssh-agent we spawned (if any) on loop exit.
cleanup() {
    if [ -n "${SSH_AGENT_PID:-}" ]; then
        kill "$SSH_AGENT_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

ensure_ssh_agent

# ----- git push helper -----
PUSH_ENABLED="${PUSH_ENABLED:-1}"
PUSH_REMOTE="${PUSH_REMOTE:-origin}"
PUSH_BRANCH="${PUSH_BRANCH:-main}"

push_now() {
    [ "$PUSH_ENABLED" = "1" ] || return 0
    if ! git -C "$REPO_DIR" remote get-url "$PUSH_REMOTE" >/dev/null 2>&1; then
        return 0
    fi
    # Only push if there's something local that's not on the remote.
    local ahead
    ahead="$(git -C "$REPO_DIR" rev-list --count "$PUSH_REMOTE/$PUSH_BRANCH..HEAD" 2>/dev/null || echo 0)"
    if [ "$ahead" -eq 0 ] 2>/dev/null; then
        return 0
    fi
    echo "[loop] pushing $ahead commit(s) to $PUSH_REMOTE/$PUSH_BRANCH" | tee -a "$LOOP_LOG"
    if timeout 60s git -C "$REPO_DIR" push "$PUSH_REMOTE" "$PUSH_BRANCH" >>"$LOOP_LOG" 2>&1; then
        echo "[loop] push ok" | tee -a "$LOOP_LOG"
    else
        echo "[loop] push failed (rc=$?) — will retry next iter" | tee -a "$LOOP_LOG"
    fi
}

# ----- rate-limit / session-usage parser -----
# Returns sleep duration in seconds for the cooldown.
sleep_secs_for_limit() {
    local logfile="$1"
    local default_secs=3600   # 60 min fallback
    local now reset_epoch sleep_secs reset_str

    # Pull lines that mention reset/limit, take the last one.
    reset_str="$(grep -oE 'reset(s| at)? [^"\\]+' "$logfile" 2>/dev/null | tail -1)"
    if [ -z "$reset_str" ]; then
        reset_str="$(grep -oE 'available again at [^"\\]+' "$logfile" 2>/dev/null | tail -1)"
    fi

    if [ -n "$reset_str" ]; then
        # Strip the "reset at " / "resets " / "available again at " prefix
        local time_part
        time_part="$(echo "$reset_str" | sed -E 's/^(reset(s| at)?|available again at) +//')"
        # Try `date -d` (handles many natural-language forms)
        reset_epoch="$(date -d "$time_part" +%s 2>/dev/null || true)"
        if [ -n "$reset_epoch" ]; then
            now="$(date +%s)"
            sleep_secs=$(( reset_epoch - now + 300 ))   # +5min safety margin
            if [ "$sleep_secs" -gt 0 ] && [ "$sleep_secs" -lt 21600 ]; then
                echo "$sleep_secs"
                return
            fi
        fi
    fi

    echo "$default_secs"
}

# ----- main loop -----
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
        echo "[loop] STATUS@$(date -u +%H%MZ): $(grep -E '^Latest|^TPU|^Tier|^W[1-4]|^B[0-9]' "$status_file" | tr '\n' ' | ')" \
            | tee -a "$LOOP_LOG"
    fi

    # Push whatever the agent committed this iter (best-effort, non-blocking)
    push_now || true

    if grep -qE '"type":"(rate_limit|overloaded|billing)_error"|"status":"throttled"|insufficient_quota|quota_exceeded|usage limit reached|usage_limit|session limit|Claude AI usage limit' "$log"; then
        cooldown="$(sleep_secs_for_limit "$log")"
        echo "[loop] usage/rate limit detected, sleeping ${cooldown}s (until $(date -d "+${cooldown} seconds" -u +%H:%MZ))" | tee -a "$log" "$LOOP_LOG"
        sleep "$cooldown"
    elif [ "$rc" -ne 0 ]; then
        echo "[loop] non-zero exit ($rc), sleeping 2 min" | tee -a "$log"
        sleep 120
    else
        echo "[loop] clean exit, sleeping 30s before next iter" | tee -a "$log"
        sleep 30
    fi
    iter=$((iter + 1))
done

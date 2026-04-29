#!/usr/bin/env bash
# Validate the running v6e-32 v4-flash smoke once the server is ready.
#
# Polls /v1/models until 200, fires the deterministic completion request
# twice, asserts the responses are byte-identical, and prints the produced
# text. Exits non-zero on any failure so it can be wired into a watcher.
#
# Usage:
#   scripts/full_slice_v4_smoke_check.sh           # default: localhost:18081
#   PORT=18082 scripts/full_slice_v4_smoke_check.sh

set -euo pipefail

PORT="${PORT:-18081}"
HOST="${HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
PROMPT='The capital of France is'
MAX_TOK=8
SEED=0
TIMEOUT_S="${TIMEOUT_S:-1800}"

readiness_wait() {
    local deadline=$(( $(date +%s) + TIMEOUT_S ))
    while true; do
        if curl -sf -o /dev/null --max-time 5 "${URL}/v1/models"; then
            echo "[smoke-check] server ready at ${URL}"
            return 0
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "[smoke-check] timed out waiting for ${URL}/v1/models" >&2
            return 1
        fi
        sleep 5
    done
}

fire_completion() {
    curl -sf --max-time 60 "${URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$(printf '{"model":"%s","prompt":"%s","max_tokens":%d,"temperature":0,"seed":%d}' \
              "$MODEL" "$PROMPT" "$MAX_TOK" "$SEED")"
}

extract_text() {
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['text'])"
}

main() {
    readiness_wait
    echo "[smoke-check] firing request 1"
    R1="$(fire_completion)"
    echo "[smoke-check] firing request 2"
    R2="$(fire_completion)"

    T1="$(printf '%s' "$R1" | extract_text)"
    T2="$(printf '%s' "$R2" | extract_text)"

    echo "[smoke-check] text 1: $T1"
    echo "[smoke-check] text 2: $T2"

    if [ "$T1" != "$T2" ]; then
        echo "[smoke-check] FAIL: completions differ between runs (non-deterministic)" >&2
        exit 2
    fi

    case "$T1" in
        *Paris*|*paris*) ok=1 ;;
        *) ok=0 ;;
    esac
    if [ "$ok" -ne 1 ]; then
        echo "[smoke-check] FAIL: expected text containing 'Paris', got: $T1" >&2
        exit 3
    fi

    echo "[smoke-check] PASS: deterministic completion contains 'Paris'"
}

main "$@"

#!/usr/bin/env bash
# Validate the running v6e-32 v4-flash smoke once the server is ready.
#
# Polls /v1/models until 200, then runs two checks:
#   1. /v1/completions: fires the deterministic completion twice, asserts
#      responses are byte-identical and contain "Paris".
#   2. /v1/chat/completions: fires a chat probe, asserts the response
#      contains "Paris". This validates scripts/v4_chat_template.jinja —
#      without that template vllm falls back to a generic format and the
#      model returns garbled text.
# Exits non-zero on any failure so it can be wired into a watcher.
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
# Readiness poll cap (server-up wait).
TIMEOUT_S="${TIMEOUT_S:-1800}"
# Per-request curl cap. The FIRST inference call against a cold engine
# triggers the jit_run_model compile (~5–10 min on V4-Flash); 900s gives
# real headroom while still failing fast on a true hang. Subsequent calls
# are sub-second.
CURL_MAX_TIME="${CURL_MAX_TIME:-900}"

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
    curl -sf --max-time "$CURL_MAX_TIME" "${URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$(printf '{"model":"%s","prompt":"%s","max_tokens":%d,"temperature":0,"seed":%d}' \
              "$MODEL" "$PROMPT" "$MAX_TOK" "$SEED")"
}

fire_chat() {
    curl -sf --max-time "$CURL_MAX_TIME" "${URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(printf '{"model":"%s","messages":[{"role":"user","content":"What is the capital of France? Answer with just the city name."}],"max_tokens":16,"temperature":0,"seed":%d}' \
              "$MODEL" "$SEED")"
}

extract_text() {
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['text'])"
}

extract_chat_message() {
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"
}

contains_paris() {
    case "$1" in
        *Paris*|*paris*) return 0 ;;
        *) return 1 ;;
    esac
}

main() {
    readiness_wait
    echo "[smoke-check] firing /v1/completions request 1"
    R1="$(fire_completion)"
    echo "[smoke-check] firing /v1/completions request 2"
    R2="$(fire_completion)"

    T1="$(printf '%s' "$R1" | extract_text)"
    T2="$(printf '%s' "$R2" | extract_text)"

    echo "[smoke-check] completions text 1: $T1"
    echo "[smoke-check] completions text 2: $T2"

    if [ "$T1" != "$T2" ]; then
        echo "[smoke-check] FAIL: completions differ between runs (non-deterministic)" >&2
        exit 2
    fi

    if ! contains_paris "$T1"; then
        echo "[smoke-check] FAIL: expected completions text containing 'Paris', got: $T1" >&2
        exit 3
    fi

    echo "[smoke-check] firing /v1/chat/completions request"
    RC="$(fire_chat)"
    TC="$(printf '%s' "$RC" | extract_chat_message)"
    echo "[smoke-check] chat message: $TC"

    if ! contains_paris "$TC"; then
        echo "[smoke-check] FAIL: expected chat message containing 'Paris', got: $TC" >&2
        echo "[smoke-check]        likely the chat-template (scripts/v4_chat_template.jinja) is missing or wrong" >&2
        exit 4
    fi

    echo "[smoke-check] PASS: /v1/completions deterministic+'Paris', /v1/chat/completions contains 'Paris'"
}

main "$@"

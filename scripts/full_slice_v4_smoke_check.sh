#!/usr/bin/env bash
# Validate the running v6e-32 v4-flash smoke once the server is ready.
#
# Polls /v1/models until 200, fires the deterministic /v1/completions
# request twice, asserts responses are byte-identical and contain
# "Paris", and prints the produced text. Exits non-zero on any failure
# so it can be wired into a watcher.
#
# After the completions gate passes, also fires a /v1/chat/completions
# probe and prints the response. This is INFORMATIONAL — set
# CHAT_REQUIRED=1 to make a missing/empty chat response fail the gate
# (exit 4). Default is best-effort: the chat path lands in a bigger
# prefill bucket (1024 total tokens for an 18-token chat prompt vs 256
# for a 5-token completion prompt) and on a tight HBM budget the
# engine may need TpuLoadedExecutable's OOM-defragment-retry to land
# the program — it usually succeeds but adds ~30s to first-chat
# latency. Asserting on chat content also requires a separate
# heuristic we don't have a robust one for yet (the model's first
# 16 tokens at temp=0/seed=0 don't reliably contain a fixed string,
# even when the template is applied correctly).
#
# Two further probes are gated off by default and turn on individually:
#   REASONING_REQUIRED=1 — fires a thinking-mode chat (chat_template_kwargs
#       {"thinking": true}) with a reasoning-eliciting prompt and asserts
#       message.reasoning is non-empty. Pins the S3 runtime: --reasoning-parser
#       deepseek_v4 actually emitting <think>...</think> → reasoning field.
#       Adds ~30s on cold cache (lands in chat path, may OOM-retry once).
#       Exit 5 on empty reasoning.
#   STREAMING_REQUIRED=1 — fires the /v1/completions probe a second time with
#       stream=true, reassembles the SSE chunks, and asserts the reassembled
#       text byte-equals the non-streaming response from the first probe.
#       Pins S7: streaming path produces the same output as non-streaming.
#       Exit 6 on mismatch / no chunks.
#
# Usage:
#   scripts/full_slice_v4_smoke_check.sh                # default: localhost:18081
#   PORT=18082 scripts/full_slice_v4_smoke_check.sh
#   CHAT_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh         # fail on empty chat
#   REASONING_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh    # fail on empty reasoning
#   STREAMING_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh    # fail on stream/non-stream mismatch

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
CHAT_REQUIRED="${CHAT_REQUIRED:-0}"
REASONING_REQUIRED="${REASONING_REQUIRED:-0}"
STREAMING_REQUIRED="${STREAMING_REQUIRED:-0}"

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

# Reasoning probe: thinking-mode chat that should produce a non-empty
# message.reasoning when --reasoning-parser deepseek_v4 is active. Multiplication
# is the cheapest reasoning-eliciting prompt — short, deterministic, and the
# <think> block almost always contains digits + intermediate products.
fire_chat_thinking() {
    curl -sf --max-time "$CURL_MAX_TIME" "${URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(printf '{"model":"%s","messages":[{"role":"user","content":"What is 17 multiplied by 23? Show your reasoning step by step."}],"max_tokens":256,"temperature":0,"seed":%d,"chat_template_kwargs":{"thinking":true}}' \
              "$MODEL" "$SEED")"
}

# Streaming probe: same prompt/params as fire_completion, but stream=true.
# vLLM emits SSE on /v1/completions: lines of `data: {...}\n\n` plus a
# terminating `data: [DONE]\n\n`. The -N flag turns off curl's output buffer.
fire_completion_stream() {
    curl -sf --max-time "$CURL_MAX_TIME" -N "${URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$(printf '{"model":"%s","prompt":"%s","max_tokens":%d,"temperature":0,"seed":%d,"stream":true}' \
              "$MODEL" "$PROMPT" "$MAX_TOK" "$SEED")"
}

extract_text() {
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['text'])"
}

extract_chat_message() {
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"
}

extract_chat_reasoning() {
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message'].get('reasoning') or '')"
}

# Reassemble streaming completion deltas into a single text. Tolerates
# `data: ` lines, the terminating `data: [DONE]`, and blank lines.
reassemble_stream() {
    python3 -c "
import json, sys
parts = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: '):
        continue
    payload = line[len('data: '):]
    if payload == '[DONE]':
        break
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        continue
    if not chunk.get('choices'):
        continue
    choice = chunk['choices'][0]
    delta = choice.get('text', '') or (choice.get('delta') or {}).get('content', '') or ''
    parts.append(delta)
print(''.join(parts))
"
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

    case "$T1" in
        *Paris*|*paris*) ok=1 ;;
        *) ok=0 ;;
    esac
    if [ "$ok" -ne 1 ]; then
        echo "[smoke-check] FAIL: expected completions text containing 'Paris', got: $T1" >&2
        exit 3
    fi

    echo "[smoke-check] firing /v1/chat/completions probe (informational; CHAT_REQUIRED=$CHAT_REQUIRED)"
    if RC="$(fire_chat 2>/dev/null)"; then
        TC="$(printf '%s' "$RC" | extract_chat_message 2>/dev/null || true)"
        echo "[smoke-check] chat message: ${TC:-<empty>}"
        if [ -z "$TC" ] && [ "$CHAT_REQUIRED" = "1" ]; then
            echo "[smoke-check] FAIL: chat message empty (CHAT_REQUIRED=1)" >&2
            exit 4
        fi
    else
        echo "[smoke-check] chat probe failed (curl non-zero)"
        if [ "$CHAT_REQUIRED" = "1" ]; then
            echo "[smoke-check] FAIL: chat probe failed (CHAT_REQUIRED=1)" >&2
            exit 4
        fi
    fi

    if [ "$REASONING_REQUIRED" = "1" ]; then
        echo "[smoke-check] firing /v1/chat/completions thinking-mode probe (REASONING_REQUIRED=1)"
        if RR="$(fire_chat_thinking 2>/dev/null)"; then
            TR="$(printf '%s' "$RR" | extract_chat_reasoning 2>/dev/null || true)"
            # Trim to first 80 chars for the log so a long <think> block doesn't drown it.
            if [ "${#TR}" -gt 80 ]; then
                echo "[smoke-check] reasoning prefix: ${TR:0:80}..."
            else
                echo "[smoke-check] reasoning: ${TR:-<empty>}"
            fi
            if [ -z "$TR" ]; then
                echo "[smoke-check] FAIL: thinking-mode chat returned empty reasoning (REASONING_REQUIRED=1)" >&2
                echo "[smoke-check]       --reasoning-parser deepseek_v4 is registered but produced no <think> output." >&2
                exit 5
            fi
        else
            echo "[smoke-check] FAIL: thinking-mode chat probe failed (curl non-zero, REASONING_REQUIRED=1)" >&2
            exit 5
        fi
    fi

    if [ "$STREAMING_REQUIRED" = "1" ]; then
        echo "[smoke-check] firing /v1/completions streaming probe (STREAMING_REQUIRED=1)"
        if RS="$(fire_completion_stream 2>/dev/null)"; then
            TS="$(printf '%s' "$RS" | reassemble_stream 2>/dev/null || true)"
            echo "[smoke-check] stream reassembled: ${TS:-<empty>}"
            if [ -z "$TS" ]; then
                echo "[smoke-check] FAIL: streaming probe produced no chunks (STREAMING_REQUIRED=1)" >&2
                exit 6
            fi
            if [ "$TS" != "$T1" ]; then
                echo "[smoke-check] FAIL: streaming output != non-streaming (STREAMING_REQUIRED=1)" >&2
                echo "[smoke-check]       non-streaming: $T1" >&2
                echo "[smoke-check]       streaming    : $TS" >&2
                exit 6
            fi
        else
            echo "[smoke-check] FAIL: streaming probe failed (curl non-zero, STREAMING_REQUIRED=1)" >&2
            exit 6
        fi
    fi

    echo "[smoke-check] PASS: deterministic completion contains 'Paris'"
}

main "$@"

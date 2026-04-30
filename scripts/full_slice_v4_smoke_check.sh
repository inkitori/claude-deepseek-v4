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
#   SAMPLING_REQUIRED=1 — fires a /v1/completions probe with temperature>0,
#       top_p, and frequency_penalty set, and asserts the response has
#       non-empty text and a valid finish_reason. Pins S6: the sampling code
#       path (temperature scaling + top-p filter + token penalties) doesn't
#       crash or produce empty/garbage output. Cheap (same prefill bucket as
#       the existing completions probe). Exit 7 on empty text / invalid
#       finish_reason / curl failure.
#   STOP_REQUIRED=1 — fires a /v1/completions probe with `stop=["Paris"]`
#       and asserts (a) the response text does NOT contain "Paris" and (b)
#       finish_reason is "stop". Pins S6 broader-matrix: the stop-sequence
#       handler in vLLM's TPU runner actually truncates the generation at
#       the matched token and reports it. Same prefill bucket as the
#       existing completions probe. Exit 8 on bad text / wrong finish_reason
#       / curl failure.
#   LOGPROBS_REQUIRED=1 — fires a /v1/completions probe with `logprobs=5`
#       and asserts every emitted token's `logprobs.top_logprobs` entry has
#       at least 5 alternatives. Pins S6 broader-matrix: the logprobs
#       postprocessing path emits the per-token alternative distribution.
#       Same prefill bucket. Exit 9 on missing logprobs / wrong cardinality
#       / curl failure.
#   TOPK_REQUIRED=1 — fires a /v1/completions probe with `temperature=0.7`
#       and `top_k=10` and asserts the response has non-empty text plus a
#       valid finish_reason. Pins S6 broader-matrix: the top-k filter on
#       the TPU sampling path doesn't crash or zero-out the candidate set.
#       Same prefill bucket. Exit 10 on empty / invalid / curl failure.
#   PRESENCE_REQUIRED=1 — fires a /v1/completions probe with
#       `temperature=0.7` and `presence_penalty=0.5` and asserts the
#       response has non-empty text plus a valid finish_reason. Pins
#       S6 broader-matrix: presence-penalty (per-token, distinct from
#       frequency_penalty's per-occurrence semantics) is correctly
#       applied on the TPU sampling path. Same prefill bucket. Exit 11
#       on empty / invalid / curl failure.
#   N_REQUIRED=1 — fires a /v1/completions probe with `n=2` and asserts
#       the response has at least 2 choices, each with non-empty text.
#       Pins S6 broader-matrix: multi-completion (n>1) in a single request
#       actually emits n choices on the TPU runner. Under --max-num-seqs=1
#       this likely sequentializes, but the choices array must still have
#       length n. Same prefill bucket. Exit 12 on missing/short choices /
#       empty text / curl failure.
#
# Usage:
#   scripts/full_slice_v4_smoke_check.sh                # default: localhost:18081
#   PORT=18082 scripts/full_slice_v4_smoke_check.sh
#   CHAT_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh         # fail on empty chat
#   REASONING_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh    # fail on empty reasoning
#   STREAMING_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh    # fail on stream/non-stream mismatch
#   SAMPLING_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh     # fail on broken sampling path
#   STOP_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh         # fail on broken stop-sequence path
#   LOGPROBS_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh     # fail on broken logprobs path
#   TOPK_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh         # fail on broken top-k path
#   PRESENCE_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh     # fail on broken presence-penalty path
#   N_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh            # fail on broken n>1 (multi-choice) path

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
SAMPLING_REQUIRED="${SAMPLING_REQUIRED:-0}"
STOP_REQUIRED="${STOP_REQUIRED:-0}"
LOGPROBS_REQUIRED="${LOGPROBS_REQUIRED:-0}"
TOPK_REQUIRED="${TOPK_REQUIRED:-0}"
PRESENCE_REQUIRED="${PRESENCE_REQUIRED:-0}"
N_REQUIRED="${N_REQUIRED:-0}"

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
#
# max_tokens is set to ~third of the smoke-launcher MAX_LEN (default 256) so
# `prompt + max_tokens <= max-model-len` and vLLM doesn't 400 the request.
# 96 leaves ~160 tokens for the prompt + thinking template (the actual encoded
# prompt is ~30-40 tokens; headroom covers the <think> preamble). Even if the
# generation is truncated mid-<think>, the parser still emits whatever it
# captured, so non-empty assertion still holds.
fire_chat_thinking() {
    local max_thinking_tokens="${REASONING_MAX_TOKENS:-96}"
    curl -sf --max-time "$CURL_MAX_TIME" "${URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(printf '{"model":"%s","messages":[{"role":"user","content":"What is 17 multiplied by 23? Show your reasoning step by step."}],"max_tokens":%d,"temperature":0,"seed":%d,"chat_template_kwargs":{"thinking":true}}' \
              "$MODEL" "$max_thinking_tokens" "$SEED")"
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

# Sampling probe: exercises the temperature>0 + top-p + frequency-penalty path
# in one request. We don't assert on a specific token (that depends on the
# model's distribution under sampling); we only assert the response is well-
# formed (non-empty text, valid finish_reason). Top-k / presence_penalty / n>1 /
# logprobs are deliberately omitted — each is a separate code path with
# different known quirks under vLLM's TPU runner, and a minimum-delta probe
# should cover the most-used parameters first. Same prompt/length budget as
# the existing completions probe so it lands in the same prefill bucket.
#
# Note: no `seed` field. vLLM's TPU runner rejects per-request seed for
# non-greedy paths with HTTP 400 ("JAX does not support per-request seed.")
# — sending `seed` together with `temperature>0` makes the probe fail
# spuriously even when the sampling code path itself is healthy.
fire_completion_sampling() {
    curl -sf --max-time "$CURL_MAX_TIME" "${URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$(printf '{"model":"%s","prompt":"%s","max_tokens":%d,"temperature":0.7,"top_p":0.9,"frequency_penalty":0.1}' \
              "$MODEL" "$PROMPT" "$MAX_TOK")"
}

# Stop-sequence probe: same deterministic prompt as fire_completion, but with
# stop=["Paris"]. The model's first generated token at temp=0/seed=0 is
# " Paris" (the deterministic baseline asserts this), so a working stop-sequence
# handler must intercept that token, truncate the response before it, and
# report finish_reason="stop". A broken handler either lets the token through
# (response still contains "Paris") or emits the wrong finish_reason.
fire_completion_stop() {
    curl -sf --max-time "$CURL_MAX_TIME" "${URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$(printf '{"model":"%s","prompt":"%s","max_tokens":%d,"temperature":0,"seed":%d,"stop":["Paris"]}' \
              "$MODEL" "$PROMPT" "$MAX_TOK" "$SEED")"
}

# Logprobs probe: same deterministic prompt as fire_completion, but with
# logprobs=5. vLLM emits a per-position object containing `tokens`,
# `token_logprobs`, and `top_logprobs` (a list of {alt_token: logprob} dicts);
# we assert every position has at least 5 alternatives. Greedy decode at
# temp=0 keeps the chosen-token logprob deterministic, but the alternatives
# matter to clients that use logprobs for confidence/scoring.
fire_completion_logprobs() {
    curl -sf --max-time "$CURL_MAX_TIME" "${URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$(printf '{"model":"%s","prompt":"%s","max_tokens":%d,"temperature":0,"seed":%d,"logprobs":5}' \
              "$MODEL" "$PROMPT" "$MAX_TOK" "$SEED")"
}

# Top-k probe: temperature>0 + top_k=10. Distinct from the existing sampling
# probe (which exercises top_p + frequency_penalty); top-k gates the candidate
# set by rank rather than cumulative-prob mass. We don't assert on a specific
# token (depends on the model's distribution) — only that text is non-empty
# and finish_reason is recognised. No `seed` (vLLM/TPU runner rejects per-
# request seed on non-greedy paths with HTTP 400).
fire_completion_topk() {
    curl -sf --max-time "$CURL_MAX_TIME" "${URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$(printf '{"model":"%s","prompt":"%s","max_tokens":%d,"temperature":0.7,"top_k":10}' \
              "$MODEL" "$PROMPT" "$MAX_TOK")"
}

# Presence-penalty probe: temperature>0 + presence_penalty=0.5. Distinct from
# frequency_penalty (which scales linearly with token count); presence_penalty
# is a fixed deduction the first time a token appears. Same well-formedness
# assertion as the sampling/topk probes: non-empty text + valid finish_reason.
fire_completion_presence() {
    curl -sf --max-time "$CURL_MAX_TIME" "${URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$(printf '{"model":"%s","prompt":"%s","max_tokens":%d,"temperature":0.7,"presence_penalty":0.5}' \
              "$MODEL" "$PROMPT" "$MAX_TOK")"
}

# Multi-completion probe: n=2 in one request. vLLM expands this into n
# parallel sequences sharing the prompt; under --max-num-seqs=1 they run
# sequentially but the response's choices array must still have length n
# (each with its own non-empty text). Probe asserts len(choices) >= 2 and
# every emitted choice has non-empty text. Uses temperature>0 because at
# temp=0 some samplers de-dupe identical greedy outputs; we want to
# verify the request shape is honoured, not the determinism of the result.
fire_completion_n() {
    curl -sf --max-time "$CURL_MAX_TIME" "${URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$(printf '{"model":"%s","prompt":"%s","max_tokens":%d,"temperature":0.7,"n":2}' \
              "$MODEL" "$PROMPT" "$MAX_TOK")"
}

extract_text() {
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['text'])"
}

extract_chat_message() {
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"
}

extract_chat_reasoning() {
    # Print the reasoning field with `<NL>` substituted for raw newlines.
    # Bash command substitution `$(...)` strips trailing \n, so a reasoning
    # field that's all whitespace would otherwise look empty to the caller.
    # The replacement keeps the visible length non-zero AND makes log lines
    # readable; the caller does its own emptiness/whitespace-only check
    # against the original via `extract_chat_reasoning_len`.
    python3 -c "
import json, sys
d = json.load(sys.stdin)
r = d['choices'][0]['message'].get('reasoning') or ''
print(r.replace('\n', '<NL>'))
"
}

extract_chat_reasoning_len() {
    # Print the byte length of the reasoning field's *non-whitespace* run.
    # 0 means the field was missing, null, empty, or whitespace-only — the
    # smoke gate treats all four as a failed reasoning emission.
    python3 -c "
import json, sys
d = json.load(sys.stdin)
r = d['choices'][0]['message'].get('reasoning') or ''
print(len(r.strip()))
"
}

# Print two lines: non-whitespace text length, and finish_reason.
# A length of 0 means empty or whitespace-only output — the sampling
# code path silently produced nothing useful. finish_reason on the
# second line lets the caller assert it's a recognised stop signal.
extract_completion_text_finish() {
    python3 -c "
import json, sys
d = json.load(sys.stdin)
c = (d.get('choices') or [{}])[0]
print(len((c.get('text') or '').strip()))
print(c.get('finish_reason') or '')
"
}

# Print three lines: contains-needle ("1" if response text contains the
# argument, "0" otherwise), non-whitespace text length, and finish_reason.
# Used by the STOP_REQUIRED probe to detect both leak ("Paris" still in
# the text) and wrong finish_reason in one call.
extract_completion_stop_check() {
    local needle="$1"
    NEEDLE="$needle" python3 -c "
import json, os, sys
needle = os.environ.get('NEEDLE', '')
d = json.load(sys.stdin)
c = (d.get('choices') or [{}])[0]
text = (c.get('text') or '')
print('1' if needle and needle in text else '0')
print(len(text.strip()))
print(c.get('finish_reason') or '')
"
}

# Print one line: minimum number of `top_logprobs` alternatives across
# all emitted positions, or -1 if the logprobs object is missing/empty.
# A value below the requested logprobs count = vLLM silently dropped
# alternatives on the TPU sampling path.
extract_completion_logprobs_min_alts() {
    python3 -c "
import json, sys
d = json.load(sys.stdin)
c = (d.get('choices') or [{}])[0]
lp = c.get('logprobs')
if not isinstance(lp, dict):
    print(-1); sys.exit(0)
top = lp.get('top_logprobs')
if not top:
    print(-1); sys.exit(0)
alts = []
for entry in top:
    if isinstance(entry, dict):
        alts.append(len(entry))
print(min(alts) if alts else -1)
"
}

# Print two lines: number of choices in the response, and the count of
# choices whose text is non-empty (after .strip()). The N_REQUIRED probe
# asserts n_choices >= 2 AND non_empty == n_choices.
extract_completion_n_choices() {
    python3 -c "
import json, sys
d = json.load(sys.stdin)
ch = d.get('choices') or []
n = len(ch)
ne = sum(1 for c in ch if (c.get('text') or '').strip())
print(n)
print(ne)
"
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
            TR_DISPLAY="$(printf '%s' "$RR" | extract_chat_reasoning 2>/dev/null || true)"
            TR_LEN="$(printf '%s' "$RR" | extract_chat_reasoning_len 2>/dev/null || echo 0)"
            # Trim display to first 80 chars so a long <think> block doesn't drown the log.
            if [ "${#TR_DISPLAY}" -gt 80 ]; then
                echo "[smoke-check] reasoning prefix (len=$TR_LEN): ${TR_DISPLAY:0:80}..."
            else
                echo "[smoke-check] reasoning (len=$TR_LEN): ${TR_DISPLAY:-<empty>}"
            fi
            if [ "${TR_LEN:-0}" = "0" ]; then
                echo "[smoke-check] FAIL: thinking-mode chat returned empty/whitespace-only reasoning (REASONING_REQUIRED=1)" >&2
                echo "[smoke-check]       --reasoning-parser deepseek_v4 is registered but the model emitted no <think> content." >&2
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

    if [ "$SAMPLING_REQUIRED" = "1" ]; then
        echo "[smoke-check] firing /v1/completions sampling probe (SAMPLING_REQUIRED=1)"
        if RP="$(fire_completion_sampling 2>/dev/null)"; then
            # extract_completion_text_finish prints two lines: text len, finish_reason.
            # Read both with a single python invocation (cheaper than two) and split.
            mapfile -t SAMP < <(printf '%s' "$RP" | extract_completion_text_finish 2>/dev/null || printf '0\n\n')
            SAMP_LEN="${SAMP[0]:-0}"
            SAMP_FIN="${SAMP[1]:-}"
            echo "[smoke-check] sampling text len=$SAMP_LEN finish_reason=${SAMP_FIN:-<empty>}"
            if [ "${SAMP_LEN:-0}" = "0" ]; then
                echo "[smoke-check] FAIL: sampling probe returned empty/whitespace-only text (SAMPLING_REQUIRED=1)" >&2
                echo "[smoke-check]       temperature>0 + top_p + frequency_penalty path produced no usable output." >&2
                exit 7
            fi
            case "$SAMP_FIN" in
                stop|length) ;;
                *)
                    echo "[smoke-check] FAIL: sampling probe finish_reason=${SAMP_FIN:-<empty>}, expected stop/length (SAMPLING_REQUIRED=1)" >&2
                    exit 7
                    ;;
            esac
        else
            echo "[smoke-check] FAIL: sampling probe failed (curl non-zero, SAMPLING_REQUIRED=1)" >&2
            exit 7
        fi
    fi

    if [ "$STOP_REQUIRED" = "1" ]; then
        echo "[smoke-check] firing /v1/completions stop-sequence probe (STOP_REQUIRED=1)"
        if RT="$(fire_completion_stop 2>/dev/null)"; then
            mapfile -t STP < <(printf '%s' "$RT" | extract_completion_stop_check "Paris" 2>/dev/null || printf '0\n0\n\n')
            STP_HIT="${STP[0]:-0}"
            STP_LEN="${STP[1]:-0}"
            STP_FIN="${STP[2]:-}"
            echo "[smoke-check] stop probe contains_needle=$STP_HIT text_len=$STP_LEN finish_reason=${STP_FIN:-<empty>}"
            if [ "$STP_HIT" = "1" ]; then
                echo "[smoke-check] FAIL: stop probe response still contains 'Paris' (STOP_REQUIRED=1)" >&2
                echo "[smoke-check]       stop=[\"Paris\"] should have truncated before that token." >&2
                exit 8
            fi
            if [ "$STP_FIN" != "stop" ]; then
                echo "[smoke-check] FAIL: stop probe finish_reason=${STP_FIN:-<empty>}, expected 'stop' (STOP_REQUIRED=1)" >&2
                exit 8
            fi
        else
            echo "[smoke-check] FAIL: stop probe failed (curl non-zero, STOP_REQUIRED=1)" >&2
            exit 8
        fi
    fi

    if [ "$LOGPROBS_REQUIRED" = "1" ]; then
        echo "[smoke-check] firing /v1/completions logprobs probe (LOGPROBS_REQUIRED=1)"
        if RL="$(fire_completion_logprobs 2>/dev/null)"; then
            LP_MIN="$(printf '%s' "$RL" | extract_completion_logprobs_min_alts 2>/dev/null || echo -1)"
            echo "[smoke-check] logprobs probe min_alternatives=$LP_MIN (requested 5)"
            if [ "${LP_MIN:--1}" -lt 5 ]; then
                echo "[smoke-check] FAIL: logprobs probe min_alternatives=$LP_MIN < 5 (LOGPROBS_REQUIRED=1)" >&2
                echo "[smoke-check]       requested logprobs=5; vLLM TPU sampling path dropped alternatives." >&2
                exit 9
            fi
        else
            echo "[smoke-check] FAIL: logprobs probe failed (curl non-zero, LOGPROBS_REQUIRED=1)" >&2
            exit 9
        fi
    fi

    if [ "$TOPK_REQUIRED" = "1" ]; then
        echo "[smoke-check] firing /v1/completions top-k probe (TOPK_REQUIRED=1)"
        if RTK="$(fire_completion_topk 2>/dev/null)"; then
            mapfile -t TK < <(printf '%s' "$RTK" | extract_completion_text_finish 2>/dev/null || printf '0\n\n')
            TK_LEN="${TK[0]:-0}"
            TK_FIN="${TK[1]:-}"
            echo "[smoke-check] top-k text len=$TK_LEN finish_reason=${TK_FIN:-<empty>}"
            if [ "${TK_LEN:-0}" = "0" ]; then
                echo "[smoke-check] FAIL: top-k probe returned empty/whitespace-only text (TOPK_REQUIRED=1)" >&2
                echo "[smoke-check]       temperature>0 + top_k=10 path produced no usable output." >&2
                exit 10
            fi
            case "$TK_FIN" in
                stop|length) ;;
                *)
                    echo "[smoke-check] FAIL: top-k probe finish_reason=${TK_FIN:-<empty>}, expected stop/length (TOPK_REQUIRED=1)" >&2
                    exit 10
                    ;;
            esac
        else
            echo "[smoke-check] FAIL: top-k probe failed (curl non-zero, TOPK_REQUIRED=1)" >&2
            exit 10
        fi
    fi

    if [ "$PRESENCE_REQUIRED" = "1" ]; then
        echo "[smoke-check] firing /v1/completions presence-penalty probe (PRESENCE_REQUIRED=1)"
        if RPP="$(fire_completion_presence 2>/dev/null)"; then
            mapfile -t PP < <(printf '%s' "$RPP" | extract_completion_text_finish 2>/dev/null || printf '0\n\n')
            PP_LEN="${PP[0]:-0}"
            PP_FIN="${PP[1]:-}"
            echo "[smoke-check] presence text len=$PP_LEN finish_reason=${PP_FIN:-<empty>}"
            if [ "${PP_LEN:-0}" = "0" ]; then
                echo "[smoke-check] FAIL: presence probe returned empty/whitespace-only text (PRESENCE_REQUIRED=1)" >&2
                echo "[smoke-check]       temperature>0 + presence_penalty=0.5 path produced no usable output." >&2
                exit 11
            fi
            case "$PP_FIN" in
                stop|length) ;;
                *)
                    echo "[smoke-check] FAIL: presence probe finish_reason=${PP_FIN:-<empty>}, expected stop/length (PRESENCE_REQUIRED=1)" >&2
                    exit 11
                    ;;
            esac
        else
            echo "[smoke-check] FAIL: presence probe failed (curl non-zero, PRESENCE_REQUIRED=1)" >&2
            exit 11
        fi
    fi

    if [ "$N_REQUIRED" = "1" ]; then
        echo "[smoke-check] firing /v1/completions n=2 probe (N_REQUIRED=1)"
        if RN="$(fire_completion_n 2>/dev/null)"; then
            mapfile -t NC < <(printf '%s' "$RN" | extract_completion_n_choices 2>/dev/null || printf '0\n0\n')
            NC_TOTAL="${NC[0]:-0}"
            NC_NONEMPTY="${NC[1]:-0}"
            echo "[smoke-check] n=2 probe choices=$NC_TOTAL non_empty=$NC_NONEMPTY (requested n=2)"
            if [ "${NC_TOTAL:-0}" -lt 2 ]; then
                echo "[smoke-check] FAIL: n=2 probe returned only $NC_TOTAL choices, expected >=2 (N_REQUIRED=1)" >&2
                echo "[smoke-check]       vLLM TPU runner did not expand n>1 into multiple choices." >&2
                exit 12
            fi
            if [ "${NC_NONEMPTY:-0}" -lt "${NC_TOTAL:-0}" ]; then
                echo "[smoke-check] FAIL: n=2 probe has $NC_NONEMPTY/$NC_TOTAL non-empty choices (N_REQUIRED=1)" >&2
                exit 12
            fi
        else
            echo "[smoke-check] FAIL: n=2 probe failed (curl non-zero, N_REQUIRED=1)" >&2
            exit 12
        fi
    fi

    echo "[smoke-check] PASS: deterministic completion contains 'Paris'"
}

main "$@"

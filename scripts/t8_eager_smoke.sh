#!/usr/bin/env bash
# Tier 8 eager-mode smoke test: real DeepSeek-V4-Flash on TPU through
# vllm serve, single sequence, --enforce-eager, max-num-seqs 1.
#
# Usage: scripts/t8_eager_smoke.sh [PORT]
#
# Outputs:
#   logs/T8-eager-serve-<TS>.log    — full vllm-serve stderr/stdout
#   logs/T8-eager-curl-<TS>.txt     — curl roundtrip results
#   logs/T8-eager-result-<TS>.json  — parsed result line + metadata

set -uo pipefail
PORT="${1:-18081}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPO="/home/enyouki/claude-deepseek-v4"
LOGS="$REPO/logs"
SERVE_LOG="$LOGS/T8-eager-serve-$TS.log"
CURL_LOG="$LOGS/T8-eager-curl-$TS.txt"
RESULT="$LOGS/T8-eager-result-$TS.json"
mkdir -p "$LOGS"

cd "$REPO/work/tpu-inference"

echo "[t8] starting vllm serve on port $PORT — log: $SERVE_LOG"
JAX_PLATFORMS=tpu HF_HUB_OFFLINE=1 NEW_MODEL_DESIGN=1 \
    vllm serve deepseek-ai/DeepSeek-V4-Flash \
      --tensor-parallel-size 4 \
      --max-model-len 256 --max-num-seqs 1 \
      --port "$PORT" --seed 0 \
      --trust-remote-code --dtype bfloat16 \
      --enforce-eager \
      --additional_config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' \
      > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
echo "[t8] vllm serve pid=$SERVE_PID"

# Wait up to 30 minutes for /v1/models 200 (gcsfuse weight scan + compile)
ready=0
for i in $(seq 1 360); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[t8] vllm serve died early; check $SERVE_LOG"
        ready=-1
        break
    fi
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[t8] /v1/models ready after ${i}*5s"
        ready=1
        break
    fi
    sleep 5
done

if [ "$ready" != "1" ]; then
    echo "[t8] FAILED — server did not become ready"
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
    echo "{\"ok\":false,\"reason\":\"server_not_ready\",\"pid\":$SERVE_PID,\"log\":\"$SERVE_LOG\"}" > "$RESULT"
    exit 2
fi

# Two identical seeded /v1/completions
PROMPT='The capital of France is'
PAYLOAD="{\"model\":\"deepseek-ai/DeepSeek-V4-Flash\",\"prompt\":\"$PROMPT\",\"max_tokens\":8,\"temperature\":0,\"seed\":0}"
echo "[t8] sending request 1" | tee -a "$CURL_LOG"
RESP1=$(curl -s "http://localhost:$PORT/v1/completions" \
  -H "Content-Type: application/json" -d "$PAYLOAD")
echo "$RESP1" | tee -a "$CURL_LOG"

echo "[t8] sending request 2 (determinism check)" | tee -a "$CURL_LOG"
RESP2=$(curl -s "http://localhost:$PORT/v1/completions" \
  -H "Content-Type: application/json" -d "$PAYLOAD")
echo "$RESP2" | tee -a "$CURL_LOG"

# Parse text from response 1
TEXT1=$(echo "$RESP1" | python3 -c "import sys, json; j=json.loads(sys.stdin.read()); print(j['choices'][0]['text'])" 2>/dev/null || echo "<parse-error>")
TEXT2=$(echo "$RESP2" | python3 -c "import sys, json; j=json.loads(sys.stdin.read()); print(j['choices'][0]['text'])" 2>/dev/null || echo "<parse-error>")
echo "[t8] text1=$TEXT1" | tee -a "$CURL_LOG"
echo "[t8] text2=$TEXT2" | tee -a "$CURL_LOG"

# Pass criteria: starts with " Paris" or "Paris", and text1==text2
STARTS_PARIS=0
case "$TEXT1" in
    " Paris"*|"Paris"*) STARTS_PARIS=1 ;;
esac
DETERMINISTIC=0
[ "$TEXT1" = "$TEXT2" ] && DETERMINISTIC=1

python3 -c "
import json, sys
print(json.dumps({
  'ok': $STARTS_PARIS and $DETERMINISTIC,
  'starts_with_paris': bool($STARTS_PARIS),
  'deterministic': bool($DETERMINISTIC),
  'text1': $(printf '%s' "$TEXT1" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'),
  'text2': $(printf '%s' "$TEXT2" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'),
  'serve_log': '$SERVE_LOG',
  'curl_log': '$CURL_LOG',
}, indent=2))
" > "$RESULT" 2>&1 || echo "{\"ok\":false,\"reason\":\"result-parse-error\"}" > "$RESULT"

cat "$RESULT"

echo "[t8] killing vllm serve pid=$SERVE_PID"
kill "$SERVE_PID" 2>/dev/null || true
wait "$SERVE_PID" 2>/dev/null || true

if [ "$STARTS_PARIS" = "1" ] && [ "$DETERMINISTIC" = "1" ]; then
    exit 0
fi
exit 1

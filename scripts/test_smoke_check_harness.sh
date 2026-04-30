#!/usr/bin/env bash
# End-to-end test of `full_slice_v4_smoke_check.sh` against a stdlib mock
# OpenAI server. Validates the smoke harness's success/failure paths
# independently of vLLM/TPU. Useful as a pre-deploy sanity check.
#
# Scenarios:
#   1. happy_path           — completions ok                          → PASS (0)
#   2. flaky_readiness      — server 503 for first 3 /v1/models       → PASS (0)
#   3. wrong_content        — server returns "Berlin"                 → FAIL (3)
#   4. timeout_when_down    — server never starts                     → FAIL (1)
#   5. chat_required_empty  — completions ok, empty chat,
#                             CHAT_REQUIRED=1                         → FAIL (4)
#   6. reasoning_required_present — thinking mode returns reasoning,
#                                   REASONING_REQUIRED=1              → PASS (0)
#   7. reasoning_required_empty   — thinking mode returns no reasoning,
#                                   REASONING_REQUIRED=1              → FAIL (5)
#   7b. reasoning_required_whitespace — thinking mode returns only newlines,
#                                       REASONING_REQUIRED=1          → FAIL (5)
#                                       (caught the real V4 thinking-mode
#                                        regression where greedy decode at
#                                        temp=0 emits N newlines after <think>;
#                                        bash $(...) strips trailing \n so the
#                                        check uses str.strip() len, not -z)
#   8. streaming_required_match   — stream reassembles to non-stream,
#                                   STREAMING_REQUIRED=1              → PASS (0)
#   9. streaming_required_mismatch — stream text differs from non-stream
#                                   STREAMING_REQUIRED=1              → FAIL (6)
#
# Exit 0 if all scenarios produced their expected exit codes; non-zero
# otherwise (with details printed inline).

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/work/vllm_env/bin/python3"
SCRIPT="${ROOT}/scripts/full_slice_v4_smoke_check.sh"
MOCK="${ROOT}/scripts/_mock_openai_server.py"

fails=0

run_scenario() {
    local name="$1"; local port="$2"; local expected="$3"; shift 3
    local with_server=1
    if [ "$name" = "timeout_when_down" ]; then with_server=0; fi

    local chat_required=0
    if [ "$name" = "chat_required_empty" ]; then chat_required=1; fi

    local reasoning_required=0
    case "$name" in
        reasoning_required_*) reasoning_required=1 ;;
    esac

    local streaming_required=0
    case "$name" in
        streaming_required_*) streaming_required=1 ;;
    esac

    local mpid=
    if [ "$with_server" -eq 1 ]; then
        "$PY" "$MOCK" --port "$port" "$@" >"/tmp/mock-${port}.log" 2>&1 &
        mpid=$!
        sleep 0.5
    fi

    # Flaky readiness scenario waits up to 3 retries × the smoke-check's
    # 5s poll interval. timeout_when_down only needs ~3s.
    local timeout_s=20
    if [ "$name" = "timeout_when_down" ]; then timeout_s=3; fi
    PORT="$port" TIMEOUT_S="$timeout_s" \
        CHAT_REQUIRED="$chat_required" \
        REASONING_REQUIRED="$reasoning_required" \
        STREAMING_REQUIRED="$streaming_required" \
        "$SCRIPT" >"/tmp/check-${port}.log" 2>&1
    local rc=$?

    if [ -n "$mpid" ]; then
        kill "$mpid" 2>/dev/null
        wait "$mpid" 2>/dev/null
    fi

    if [ "$rc" = "$expected" ]; then
        echo "  [pass] $name  exit=$rc"
    else
        echo "  [FAIL] $name  exit=$rc expected=$expected"
        echo "    --- check.log ---"
        sed 's/^/      /' "/tmp/check-${port}.log"
        fails=$((fails+1))
    fi
}

echo "Testing $SCRIPT against $MOCK"
run_scenario happy_path                    18091 0
run_scenario flaky_readiness               18092 0 --flaky-readiness 3
run_scenario wrong_content                 18093 3 --text " Berlin."
run_scenario timeout_when_down             18095 1
run_scenario chat_required_empty           18096 4 --chat-text ""
run_scenario reasoning_required_present    18097 0 --reasoning-text "17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391"
run_scenario reasoning_required_empty      18098 5 --reasoning-text ""
run_scenario reasoning_required_whitespace 18099 5 --reasoning-text "$(printf '\n\n\n\n\n')"
run_scenario streaming_required_match      18100 0
run_scenario streaming_required_mismatch   18101 6 --stream-text " Berlin."

echo
if [ "$fails" -eq 0 ]; then
    echo "OK: 10/10 harness scenarios pass"
    exit 0
fi
echo "FAIL: ${fails} scenario(s) failed"
exit 1

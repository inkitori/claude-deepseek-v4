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
#  10. sampling_required_match    — temp>0 returns non-empty text + valid
#                                   finish_reason, SAMPLING_REQUIRED=1 → PASS (0)
#  11. sampling_required_empty    — temp>0 returns empty text,
#                                   SAMPLING_REQUIRED=1               → FAIL (7)
#  12. sampling_required_bad_finish — temp>0 returns non-empty text but an
#                                   invalid finish_reason (e.g. abort),
#                                   SAMPLING_REQUIRED=1               → FAIL (7)
#  13. stop_required_match        — stop=["Paris"] truncates before token,
#                                   finish_reason="stop",
#                                   STOP_REQUIRED=1                   → PASS (0)
#  14. stop_required_leak         — server ignores stop, leaves "Paris" in
#                                   text, STOP_REQUIRED=1             → FAIL (8)
#  15. logprobs_required_match    — logprobs=5 returns >=5 alternatives per
#                                   token, LOGPROBS_REQUIRED=1        → PASS (0)
#  16. logprobs_required_missing  — logprobs=5 but server omits the object,
#                                   LOGPROBS_REQUIRED=1               → FAIL (9)
#  17. logprobs_required_dropped  — logprobs=5 but server emits only 3 alts,
#                                   LOGPROBS_REQUIRED=1               → FAIL (9)
#  18. topk_required_match        — temp>0+top_k=10 returns non-empty text +
#                                   valid finish_reason, TOPK_REQUIRED=1 → PASS (0)
#  19. topk_required_empty        — temp>0+top_k=10 returns empty text,
#                                   TOPK_REQUIRED=1                   → FAIL (10)
#  20. presence_required_match    — temp>0+presence_penalty=0.5 returns
#                                   well-formed, PRESENCE_REQUIRED=1  → PASS (0)
#  21. presence_required_empty    — temp>0+presence_penalty=0.5 returns
#                                   empty text, PRESENCE_REQUIRED=1   → FAIL (11)
#  22. n_required_match           — n=2 returns 2 non-empty choices,
#                                   N_REQUIRED=1                      → PASS (0)
#  23. n_required_short           — server caps choices at 1 despite
#                                   n=2, N_REQUIRED=1                 → FAIL (12)
#  24. long_gen_required_match    — max_tokens=64 returns 30+ tokens of
#                                   well-formed text, LONG_GEN_REQUIRED=1
#                                                                     → PASS (0)
#  25. long_gen_required_short    — usage.completion_tokens=10 (under 30),
#                                   LONG_GEN_REQUIRED=1               → FAIL (13)
#  26. long_gen_required_loop     — text shows the same word 5+ times in a
#                                   row, LONG_GEN_REQUIRED=1          → FAIL (13)
#  27. long_gen_required_dirty_end — text ends in "...." (non-alphanumeric
#                                   tail), LONG_GEN_REQUIRED=1        → FAIL (13)
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

    local sampling_required=0
    case "$name" in
        sampling_required_*) sampling_required=1 ;;
    esac

    local stop_required=0
    case "$name" in
        stop_required_*) stop_required=1 ;;
    esac

    local logprobs_required=0
    case "$name" in
        logprobs_required_*) logprobs_required=1 ;;
    esac

    local topk_required=0
    case "$name" in
        topk_required_*) topk_required=1 ;;
    esac

    local presence_required=0
    case "$name" in
        presence_required_*) presence_required=1 ;;
    esac

    local n_required=0
    case "$name" in
        n_required_*) n_required=1 ;;
    esac

    local long_gen_required=0
    case "$name" in
        long_gen_required_*) long_gen_required=1 ;;
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
        SAMPLING_REQUIRED="$sampling_required" \
        STOP_REQUIRED="$stop_required" \
        LOGPROBS_REQUIRED="$logprobs_required" \
        TOPK_REQUIRED="$topk_required" \
        PRESENCE_REQUIRED="$presence_required" \
        N_REQUIRED="$n_required" \
        LONG_GEN_REQUIRED="$long_gen_required" \
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
run_scenario sampling_required_match       18102 0
run_scenario sampling_required_empty       18103 7 --sampling-text ""
run_scenario sampling_required_bad_finish  18104 7 --sampling-text " ok." --sampling-finish "abort"
run_scenario stop_required_match           18105 0
run_scenario stop_required_leak            18106 8 --stop-honor 0
run_scenario logprobs_required_match       18107 0
run_scenario logprobs_required_missing     18108 9 --logprobs-alts 0
run_scenario logprobs_required_dropped     18109 9 --logprobs-alts 3
run_scenario topk_required_match           18110 0
run_scenario topk_required_empty           18111 10 --sampling-text ""
run_scenario presence_required_match       18112 0
run_scenario presence_required_empty       18113 11 --sampling-text ""
run_scenario n_required_match              18114 0
run_scenario n_required_short              18115 12 --n-cap 1
run_scenario long_gen_required_match       18116 0
run_scenario long_gen_required_short       18117 13 --long-gen-tokens 10
run_scenario long_gen_required_loop        18118 13 \
    --long-gen-text "robot robot robot robot robot robot wandered the plain steadily"
run_scenario long_gen_required_dirty_end   18119 13 \
    --long-gen-text "The robot rolled across the rust-red plain ...."
# Real-V4 decode-state corruption emits pad/control tokens that decode to
# empty strings: completion_tokens=64 (from usage) looks healthy but visible
# text is just one word. Pin the gate against this exact pattern.
run_scenario long_gen_required_invisible   18120 13 \
    --long-gen-tokens 64 --long-gen-text " This"

echo
if [ "$fails" -eq 0 ]; then
    echo "OK: 29/29 harness scenarios pass"
    exit 0
fi
echo "FAIL: ${fails} scenario(s) failed"
exit 1

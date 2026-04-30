#!/usr/bin/env bash
# Live monitor for the host-direct claude-overnight loop.
#
# Usage:
#   ./monitor.sh           # default: every tool call (Bash, Read, Edit, Write, ...)
#   ./monitor.sh bash      # only bash commands, as plain shell lines
#   ./monitor.sh narrate   # the model's text-block prose (its public-facing reasoning)
#   ./monitor.sh results   # tool results — what each tool call returned to the model
#   ./monitor.sh all       # chronological feed: tool calls + narration + tool results + thinking-marker
#   ./monitor.sh stream    # raw stream-json (verbose)
#   ./monitor.sh status    # one-shot snapshot: loop, resources, commits, latest smoke
#   ./monitor.sh tail      # tail -F the latest iter log
#
# NOTE on "what the model is thinking":
#   With --effort xhigh the model uses extended thinking, but Anthropic's API
#   redacts the plaintext (you'll see only an encrypted .signature blob, never
#   the .thinking text). This is fixed across all auth modes and CLI flags.
#   `narrate` and `all` show the next-best signal: the model's text blocks
#   (open prose between tool calls) plus its tool calls and their results.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$REPO_DIR/logs"
PID_FILE="$LOGS/loop.pid"

latest_log() {
    ls -1t "$LOGS"/iter-*.log 2>/dev/null | head -1
}

wait_for_log() {
    while [ -z "$(latest_log)" ]; do
        echo "(no iter log yet — first-run setup may still be in progress)"
        echo "  setup progress:  tail -f $LOGS/setup.log"
        echo "  loop wrapper:    tail -f $LOGS/loop.out"
        sleep 5
    done
}

mode="${1:-tools}"

case "$mode" in
tools)
    wait_for_log
    log="$(latest_log)"
    echo "tailing tool calls from $log"
    echo "---"
    tail -n 200 -F "$log" | grep --line-buffered '^{' | \
        jq -rR --unbuffered '
            fromjson? |
            select(.type=="assistant") | .message.content[]? |
            select(.type=="tool_use") |
            "[\(.name)] " + (
                if .name == "Bash" then .input.command
                elif .name == "Read" then .input.file_path
                elif .name == "Edit" or .name == "Write" then .input.file_path
                elif .name == "Grep" then ((.input.pattern // "?") + "  in  " + (.input.path // "."))
                elif .name == "TodoWrite" or .name == "TaskCreate" then (.input | tostring)
                else (.input | tostring)
                end
              ) | .[:300]
        '
    ;;

bash)
    wait_for_log
    log="$(latest_log)"
    echo "tailing bash commands from $log"
    echo "---"
    tail -n 200 -F "$log" | grep --line-buffered '^{' | \
        jq -rR --unbuffered '
            fromjson? |
            select(.type=="assistant") | .message.content[]? |
            select(.type=="tool_use" and .name=="Bash") | .input.command
        '
    ;;

narrate)
    wait_for_log
    log="$(latest_log)"
    echo "tailing text/narration blocks from $log"
    echo "---"
    tail -n 500 -F "$log" | grep --line-buffered '^{' | \
        jq -rR --unbuffered '
            fromjson? |
            select(.type=="assistant") | .message.content[]? |
            select(.type=="text") | .text + "\n"
        '
    ;;

results)
    wait_for_log
    log="$(latest_log)"
    echo "tailing tool results from $log"
    echo "---"
    tail -n 500 -F "$log" | grep --line-buffered '^{' | \
        jq -rR --unbuffered '
            fromjson? |
            select(.type=="user") | .message.content[]? |
            select(.type=="tool_result") |
            "← " + (
                if (.content | type) == "string" then .content
                else (.content | tostring)
                end
              ) | .[:600]
        '
    ;;

all)
    wait_for_log
    log="$(latest_log)"
    echo "chronological feed from $log (text + tool_use + tool_result + thinking-marker)"
    echo "---"
    tail -n 500 -F "$log" | grep --line-buffered '^{' | \
        jq -rR --unbuffered '
            fromjson? |
            if .type=="assistant" then
                .message.content[]? |
                if .type=="text" then "📝 " + .text
                elif .type=="thinking" then "💭 (thinking, \(.signature | length) chars redacted)"
                elif .type=="tool_use" then
                    "→ [\(.name)] " + (
                        if .name == "Bash" then .input.command
                        elif .name == "Read" or .name == "Edit" or .name == "Write" then .input.file_path
                        elif .name == "Grep" then ((.input.pattern // "?") + "  in  " + (.input.path // "."))
                        else (.input | tostring)
                        end
                    )[:300]
                else empty end
            elif .type=="user" then
                .message.content[]? | select(.type=="tool_result") |
                "← " + (
                    if (.content | type) == "string" then .content
                    else (.content | tostring)
                    end
                )[:300]
            elif .type=="system" and .subtype=="init" then "⚡ session init: model=\(.model)"
            elif .type=="result" then "✓ result: \(.subtype // "?")  cost=$\(.total_cost_usd // "?")"
            else empty
            end
        '
    ;;

stream)
    wait_for_log
    log="$(latest_log)"
    echo "tailing raw stream-json from $log"
    tail -n 50 -F "$log"
    ;;

tail)
    wait_for_log
    log="$(latest_log)"
    echo "tailing $log"
    tail -n 50 -F "$log"
    ;;

status)
    echo "=== loop ==="
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        pid="$(cat "$PID_FILE")"
        echo "alive pid=$pid  uptime=$(ps -o etime= -p "$pid" | tr -d ' ')"
    else
        echo "(no loop running — start with ./run.sh)"
    fi
    echo
    echo "=== resources ==="
    df -h "$REPO_DIR" / /dev/shm 2>/dev/null | head -5
    echo
    free -h | head -2
    echo
    echo "=== preflight ==="
    [ -f "$LOGS/tpu-preflight.log" ] && head -1 "$LOGS/tpu-preflight.log" || echo "(none)"
    echo
    echo "=== iter logs (most recent 5) ==="
    ls -lh "$LOGS"/iter-*.log 2>/dev/null | tail -5 || echo "(none yet)"
    echo
    echo "=== commits in repo ==="
    git -C "$REPO_DIR" log --oneline -10 2>/dev/null
    echo
    echo "=== latest smoke result ==="
    if [ -f "$REPO_DIR/logs/full-slice-v4-smoke.pid" ]; then
        latest_smoke=$(ls -1t "$REPO_DIR"/logs/full-slice-v4-smoke-*.log 2>/dev/null | head -1)
        if [ -n "$latest_smoke" ]; then
            echo "log=$latest_smoke"
            grep -aE "POST /v1/(completions|chat)|Application startup complete|HbmOom|Worker exit" \
                "$latest_smoke" | tail -10
        fi
    else
        echo "(no smoke recorded)"
    fi
    ;;

*)
    grep -E '^#( |$)' "$0" | sed 's/^# \?//'
    exit 1
    ;;
esac

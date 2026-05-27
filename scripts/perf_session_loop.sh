#!/usr/bin/env bash
# perf_session_loop.sh — self-perpetuating Claude Code session loop for the
# DeepSeek-V4-Flash PERFORMANCE campaign. (Headless sibling of
# perf_handoff_window.sh; same design as the S1 loop, retargeted at perf.)
#
# ============================================================================
# WHAT THIS IS
# ----------------------------------------------------------------------------
# A `while true` wrapper that repeatedly launches a FRESH-CONTEXT Claude Code
# session to work the HANDOFF_PERF.md roadmap. Each session:
#   * starts with EMPTY context (we do NOT use --continue / -c),
#   * auto-loads CLAUDE.md (the runbook),
#   * is given scripts/perf_loop_prompt.txt as its initial prompt,
#   * works until it judges its context is near ~200k tokens (or hits a natural
#     stopping point), at which point the prompt instructs it to trim the
#     markdown, refresh HANDOFF_PERF.md, commit+push, and END ITS TURN.
# When the session process exits, this wrapper sleeps briefly and launches the
# NEXT fresh session — so context stays lean indefinitely while perf progress
# accumulates in git + HANDOFF_PERF.md + CLAUDE.md.
#
# CAVEAT (by design): `/context` is a terminal-UI feature, NOT available in `-p`
# mode, so a headless session cannot read its own token count. The prompt handles
# this by telling the session to hand off after a bounded unit of work. To let a
# session literally eyeball /context, run it interactively (see below) or use the
# new-window protocol (perf_handoff_window.sh) instead — that is PREFERRED.
#
# ============================================================================
# HOW TO RUN
# ----------------------------------------------------------------------------
#   tmux new-window -n perfloop         # detachable window so it survives logout
#   scripts/perf_session_loop.sh        # start the loop
#   # detach: Ctrl-b d   reattach: tmux attach
# or with nohup:
#   nohup scripts/perf_session_loop.sh >logs/perf_session_loop.log 2>&1 &
#
# RUN INTERACTIVELY (optional, single session, lets you watch /context):
#   claude --effort max --dangerously-skip-permissions "$(cat scripts/perf_loop_prompt.txt)"
#
# ============================================================================
# HOW TO STOP  (read this — there is a `pkill -f` FOOTGUN below)
# ----------------------------------------------------------------------------
#   touch /tmp/perf_loop_stop
# The loop checks for that file at the top of every iteration and BEFORE each
# relaunch, so it stops cleanly without killing an in-flight session. To also
# interrupt a session mid-run, Ctrl-C the wrapper (or kill the tmux window) AFTER
# touching the stop file.
#
# !!! pkill -f FOOTGUN !!!  Do NOT `pkill -f perf_session_loop` from a shell whose
# own command line contains that pattern — `pkill -f` matches the FULL command
# line and can kill the very shell running it. Use the stop file. If you must
# pkill, use a bracket-class pattern from an UNRELATED shell:
#     pkill -f '[p]erf_session_loop.sh'
# ============================================================================

set -uo pipefail   # deliberately NOT `set -e` — a single failed session must
                   # never tear down the loop; we want it to relaunch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT_FILE="$SCRIPT_DIR/perf_loop_prompt.txt"
STOP_FILE="${PERF_LOOP_STOP_FILE:-/tmp/perf_loop_stop}"
LOG_DIR="$REPO_ROOT/logs"

RELAUNCH_SLEEP="${RELAUNCH_SLEEP:-5}"

CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || true)}"
if [[ -z "$CLAUDE_BIN" && -x "$HOME/.local/bin/claude" ]]; then
  CLAUDE_BIN="$HOME/.local/bin/claude"
fi

if [[ -z "$CLAUDE_BIN" ]]; then
  echo "FATAL: 'claude' CLI not found on PATH or at \$HOME/.local/bin/claude." >&2
  exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "FATAL: prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

# Flags (verified against claude --help, claude 2.1.x):
#   --print/-p                        headless, agentic, tool-capable; exits when
#                                     the model ends its turn (our relaunch cue).
#   --effort max                      "max thinking".
#   --dangerously-skip-permissions    bypass permission prompts (sandbox box).
#   --verbose                         stream the agent's work to our log.
CLAUDE_FLAGS=(--print --effort max --dangerously-skip-permissions --verbose)

echo "=============================================================="
echo " perf_session_loop starting"
echo "   claude:      $CLAUDE_BIN"
echo "   flags:       ${CLAUDE_FLAGS[*]}"
echo "   prompt:      $PROMPT_FILE"
echo "   repo:        $REPO_ROOT"
echo "   stop file:   $STOP_FILE   (touch it to stop the loop)"
echo "   relaunch in: ${RELAUNCH_SLEEP}s between sessions"
echo "=============================================================="

if [[ -e "$STOP_FILE" ]]; then
  echo "Stop file $STOP_FILE already exists. Remove it to start the loop:" >&2
  echo "    rm -f $STOP_FILE" >&2
  exit 1
fi

iter=0
while true; do
  if [[ -e "$STOP_FILE" ]]; then
    echo "[loop] stop file $STOP_FILE present — exiting loop cleanly."
    break
  fi

  iter=$((iter + 1))
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  session_log="$LOG_DIR/perf_session_${ts}.log"
  echo
  echo "[loop] ===== launching session #${iter} at ${ts} ====="
  echo "[loop] session log: $session_log"

  # SINGLE foreground invocation; prompt on stdin (avoids argv quoting pitfalls).
  # `|| true` keeps the loop alive even if a session exits non-zero.
  (
    cd "$REPO_ROOT" || exit 1
    "$CLAUDE_BIN" "${CLAUDE_FLAGS[@]}" < "$PROMPT_FILE"
  ) 2>&1 | tee "$session_log" || true

  rc=${PIPESTATUS[0]}
  echo "[loop] session #${iter} exited (rc=${rc}) at $(date -u +%Y%m%dT%H%M%SZ)."

  if [[ -e "$STOP_FILE" ]]; then
    echo "[loop] stop file $STOP_FILE present after session — exiting loop cleanly."
    break
  fi

  echo "[loop] relaunching a FRESH session in ${RELAUNCH_SLEEP}s ..."
  sleep "$RELAUNCH_SLEEP" || true
done

echo "[loop] stopped after ${iter} session(s). To restart: rm -f $STOP_FILE && scripts/perf_session_loop.sh"

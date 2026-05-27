#!/usr/bin/env bash
# perf_handoff_window.sh — hand off to a FRESH-context Claude Code session in a
# NEW tmux window of the CURRENT tmux session, for the PERFORMANCE campaign.
#
# Mechanism behind the CONTEXT HANDOFF PROTOCOL in CLAUDE.md: when a session's
# context nears 100k-200k tokens (and it is NOT seriously mid-operation), it trims
# the markdown lean, refreshes HANDOFF_PERF.md, commits+pushes, then runs THIS
# script and ENDS ITS TURN. A brand-new max-thinking session
# (`claude --dangerously-skip-permissions --effort max`) starts in a new window
# with EMPTY context, auto-loads CLAUDE.md, and is handed scripts/perf_loop_prompt.txt.
#
# Why a new window (not `claude -p` headless like perf_session_loop.sh): an
# interactive session CAN read its own token count via /context, so it can make
# the 100k-200k handoff decision the loop wrapper can't. The new window also leaves
# the finished session's scrollback intact for a human to inspect.
#
# Usage (from inside tmux):  scripts/perf_handoff_window.sh
# Stop the chain: just don't call it (or `touch /tmp/perf_loop_stop`, which the
# headless wrapper honours).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT_FILE="$SCRIPT_DIR/perf_loop_prompt.txt"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || true)}"
[[ -z "$CLAUDE_BIN" && -x "$HOME/.local/bin/claude" ]] && CLAUDE_BIN="$HOME/.local/bin/claude"
[[ -z "$CLAUDE_BIN" ]] && { echo "FATAL: 'claude' CLI not found on PATH or ~/.local/bin." >&2; exit 1; }

EFFORT_FLAG="--effort max"
"$CLAUDE_BIN" --help 2>&1 | grep -q -- "--effort" || EFFORT_FLAG=""

# ---- LAUNCH MODE: re-invocation inside the freshly-created window ----------
# Runs the actual fresh session. Kept in this same file (re-invoked with
# _PERF_LAUNCH=1) so there is no second script and no inline-quoting of the
# multi-line prompt through tmux.
if [[ "${_PERF_LAUNCH:-}" == "1" ]]; then
  # CRITICAL: clear the flag BEFORE exec'ing claude. `exec` inherits the env, so a
  # leaked _PERF_LAUNCH=1 persists in the spawned session, is inherited by every Bash
  # tool subprocess, and makes that session's NEXT handoff mis-take THIS launch
  # branch (exec claude in-line, no new window). Unsetting here makes every spawned
  # session start clean so its own handoff correctly reaches orchestrate mode below.
  unset _PERF_LAUNCH
  cd "$REPO_ROOT" || exit 1
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "FATAL: prompt file missing: $PROMPT_FILE" >&2; exec bash
  fi
  # shellcheck disable=SC2086  # EFFORT_FLAG is intentionally word-split (may be empty)
  exec "$CLAUDE_BIN" --dangerously-skip-permissions $EFFORT_FLAG "$(cat "$PROMPT_FILE")"
fi

# ---- ORCHESTRATE MODE: open the new window --------------------------------
if [[ -z "${TMUX:-}" ]]; then
  echo "FATAL: not inside tmux (\$TMUX is empty). Start one (tmux new -s claude) or" >&2
  echo "       run a fresh session manually:" >&2
  echo "       $CLAUDE_BIN --dangerously-skip-permissions $EFFORT_FLAG \"\$(cat $PROMPT_FILE)\"" >&2
  exit 1
fi

SESSION="$(tmux display-message -p '#S')"

# Launch-mode `exec`s claude, so when a handoff session's claude EXITS its window
# closes on its own. We deliberately do NOT auto-kill any window: a live interactive
# claude reports pane_current_command=claude, so a name/command heuristic would risk
# killing a session that is still working. Close old finished-but-idle windows by hand.
WIN="perffresh-$(date -u +%H%M%SZ)"
# `-d` (background, don't steal focus) + redirect this invocation's std{in,out,err} to
# /dev/null. Without the redirection a NON-INTERACTIVE caller HANGS: the detached
# interactive `claude` inherits the caller's stdout pipe, so the caller blocks until
# that claude exits. This form returns cleanly and the window persists with the fresh
# session running.
tmux new-window -d -t "$SESSION:" -n "$WIN" -c "$REPO_ROOT" "_PERF_LAUNCH=1 '$SELF'" </dev/null >/dev/null 2>&1
echo "[handoff] fresh max-thinking PERF session launched in tmux session '$SESSION', window '$WIN' (background)."
echo "[handoff] Switch to it:  tmux select-window -t '$SESSION:$WIN'   (or Ctrl-b then its number)."
echo "[handoff] This session should now END ITS TURN so the fresh one takes over."

#!/usr/bin/env bash
# s1_handoff_window.sh — hand off to a FRESH-context Claude Code session in a
# NEW tmux window of the CURRENT tmux session.
#
# This is the mechanism behind the CONTEXT HANDOFF PROTOCOL in CLAUDE.md: when a
# session's context nears 100k-200k tokens (and it is NOT seriously mid-operation),
# it trims the markdown lean, refreshes HANDOFF_S1.md, commits+pushes, then runs
# THIS script and ENDS ITS TURN. A brand-new max-thinking session
# (`claude --dangerously-skip-permissions --effort max`) starts in a new window
# with EMPTY context, auto-loads CLAUDE.md, and is handed scripts/s1_loop_prompt.txt.
#
# Why a new window (not `claude -p` headless like s1_session_loop.sh): an
# interactive session CAN read its own token count via /context, so it can make
# the 100k-200k handoff decision the loop wrapper can't. The new window also
# leaves the finished session's scrollback intact for a human to inspect.
#
# Usage (from inside tmux):  scripts/s1_handoff_window.sh
# Stop the chain: just don't call it (or touch /tmp/s1_loop_stop, which the
# prompt also honours when S1 is DONE).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT_FILE="$SCRIPT_DIR/s1_loop_prompt.txt"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || true)}"
[[ -z "$CLAUDE_BIN" && -x "$HOME/.local/bin/claude" ]] && CLAUDE_BIN="$HOME/.local/bin/claude"
[[ -z "$CLAUDE_BIN" ]] && { echo "FATAL: 'claude' CLI not found on PATH or ~/.local/bin." >&2; exit 1; }

EFFORT_FLAG="--effort max"
"$CLAUDE_BIN" --help 2>&1 | grep -q -- "--effort" || EFFORT_FLAG=""

# ---- LAUNCH MODE: re-invocation inside the freshly-created window ----------
# Runs the actual fresh session. Kept in this same file (re-invoked with
# _S1_LAUNCH=1) so there is no second script and no inline-quoting of the
# multi-line prompt through tmux.
if [[ "${_S1_LAUNCH:-}" == "1" ]]; then
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
# closes on its own — nothing to garbage-collect. We deliberately do NOT auto-kill
# any window: a live interactive claude reports pane_current_command=claude, so a
# name/command heuristic would risk killing a session that is still working. Old
# finished-but-idle windows (claude ended its turn and is waiting) are harmless and
# useful for reference; close them by hand if they pile up.
WIN="s1fresh-$(date -u +%H%M%SZ)"
# `-d` (create in background, don't steal focus) + redirect THIS invocation's std{in,out,err}
# to /dev/null. Without the redirection a NON-INTERACTIVE caller HANGS: the detached interactive
# `claude` in the new window inherits the caller's stdout pipe, so the caller (s1_session_loop,
# or a Claude Code Bash tool performing the handoff) blocks until that claude exits. A bare
# `tmux new-window` only works from a human's interactive terminal. Proven: the bare form hangs
# a non-interactive caller and never returns + never creates the window; this form returns
# cleanly and the window persists with the fresh session running.
tmux new-window -d -t "$SESSION:" -n "$WIN" -c "$REPO_ROOT" "_S1_LAUNCH=1 '$SELF'" </dev/null >/dev/null 2>&1
echo "[handoff] fresh max-thinking session launched in tmux session '$SESSION', window '$WIN' (background)."
echo "[handoff] Switch to it:  tmux select-window -t '$SESSION:$WIN'   (or Ctrl-b then its number)."
echo "[handoff] This session should now END ITS TURN so the fresh one takes over."

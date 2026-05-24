#!/usr/bin/env bash
# s1_session_loop.sh — self-perpetuating Claude Code session loop for the S1 bug.
#
# ============================================================================
# WHAT THIS IS
# ----------------------------------------------------------------------------
# A `while true` wrapper that repeatedly launches a FRESH-CONTEXT Claude Code
# session to work on the S1 TPU-decode bug. Each session:
#   * starts with EMPTY context (we do NOT use --continue / -c, which would
#     resume the SAME growing context and defeat the whole point),
#   * auto-loads CLAUDE.md (the runbook),
#   * is given scripts/s1_loop_prompt.txt as its initial prompt,
#   * works until it judges its context is near ~200k tokens (or hits a natural
#     stopping point), at which point the prompt instructs it to trim CLAUDE.md,
#     refresh HANDOFF_S1.md, commit+push, and END ITS TURN.
# When the session process exits, this wrapper sleeps briefly and launches the
# NEXT fresh session — so context stays lean indefinitely while S1 progress
# accumulates in git + HANDOFF_S1.md + CLAUDE.md.
#
# ============================================================================
# WHY `claude -p` (headless print mode) AND NOT tmux send-keys
# ----------------------------------------------------------------------------
# Verified on this box (claude 2.1.150): `claude -p "<prompt>"` is AGENTIC and
# TOOL-CAPABLE in print mode — it runs Bash/Edit/Read/subagents and only exits
# when the model ends its turn. That exit is exactly our relaunch trigger, so we
# do NOT need fragile `tmux send-keys` puppeteering to drive the agent. (tmux IS
# still useful, but only to HOST this wrapper in a detachable window so the loop
# survives SSH disconnects — see HANDOFF_S1.md "Session continuity". This script
# does not require tmux to function.)
#
# CAVEAT (documented, by design): the `/context` slash command is a terminal-UI
# feature and is NOT available in `-p` mode, so a headless session cannot read
# its own token count. The prompt handles this by telling the session to hand
# off after completing a bounded unit of work, which keeps each session short
# and context lean WITHOUT needing /context. If you instead want a session to
# literally eyeball /context, run it interactively (see RUN INTERACTIVELY below).
#
# ============================================================================
# HOW TO RUN
# ----------------------------------------------------------------------------
#   tmux new-window -n s1loop          # detachable window so it survives logout
#   scripts/s1_session_loop.sh         # start the loop
#   # detach: Ctrl-b d   reattach: tmux attach
# or with nohup:
#   nohup scripts/s1_session_loop.sh >logs/s1_session_loop.log 2>&1 &
#
# RUN INTERACTIVELY (optional, single session, lets you watch /context):
#   claude --effort max --dangerously-skip-permissions "$(cat scripts/s1_loop_prompt.txt)"
#
# ============================================================================
# HOW TO STOP  (read this — there is a `pkill -f` FOOTGUN below)
# ----------------------------------------------------------------------------
#   touch /tmp/s1_loop_stop
# The loop checks for that file at the top of every iteration and BEFORE each
# relaunch, so it stops cleanly without killing an in-flight session. To also
# interrupt a session that is mid-run, Ctrl-C the wrapper (or kill the tmux
# window) AFTER touching the stop file.
#
# !!! pkill -f FOOTGUN !!!
# Do NOT try to stop this loop with something like
#     pkill -f s1_session_loop
# from another shell that ALSO has "s1_session_loop" on its own command line:
# `pkill -f` matches against the FULL command line, so such a command can match
# (and kill) the very shell that is running it, or kill the wrapper at a moment
# that orphans a child. Use the stop file. If you must pkill, use a bracket-class
# pattern from an UNRELATED shell, e.g.  pkill -f '[s]1_session_loop.sh'  — the
# bracket prevents the pattern string itself from matching. The clean mechanism
# is the stop file; prefer it.
# ============================================================================

set -uo pipefail   # NOTE: deliberately NOT `set -e` — a single failed session
                   # must never tear down the loop; we want it to relaunch.

# --- resolve paths (work from anywhere) ------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT_FILE="$SCRIPT_DIR/s1_loop_prompt.txt"
STOP_FILE="${S1_LOOP_STOP_FILE:-/tmp/s1_loop_stop}"
LOG_DIR="$REPO_ROOT/logs"

# Seconds to wait between a session exiting and the next launching.
RELAUNCH_SLEEP="${RELAUNCH_SLEEP:-5}"

# Locate the claude binary (PATH, then the known install location).
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || true)}"
if [[ -z "$CLAUDE_BIN" && -x "$HOME/.local/bin/claude" ]]; then
  CLAUDE_BIN="$HOME/.local/bin/claude"
fi

# --- preflight -------------------------------------------------------------
if [[ -z "$CLAUDE_BIN" ]]; then
  echo "FATAL: 'claude' CLI not found on PATH or at \$HOME/.local/bin/claude." >&2
  exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "FATAL: prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

# The flags, verified against `claude --help` (claude 2.1.150):
#   -p / --print                     headless, agentic, tool-capable; exits when
#                                    the model ends its turn (our relaunch cue).
#   --effort max                     "max thinking" (valid choice: low|medium|high|xhigh|max).
#   --dangerously-skip-permissions   bypass all permission prompts (sandbox box).
#   --verbose                        stream the agent's work to our log so a human
#                                    can watch each session live.
# We pass the prompt on stdin (--print reads the prompt arg OR stdin); using
# stdin avoids any argv-length / quoting pitfalls with the multi-line prompt.
CLAUDE_FLAGS=(--print --effort max --dangerously-skip-permissions --verbose)

echo "=============================================================="
echo " s1_session_loop starting"
echo "   claude:      $CLAUDE_BIN"
echo "   flags:       ${CLAUDE_FLAGS[*]}"
echo "   prompt:      $PROMPT_FILE"
echo "   repo:        $REPO_ROOT"
echo "   stop file:   $STOP_FILE   (touch it to stop the loop)"
echo "   relaunch in: ${RELAUNCH_SLEEP}s between sessions"
echo "=============================================================="

# If a stale stop file exists from a previous run, refuse to start silently in
# the stopped state — make the operator clear it intentionally.
if [[ -e "$STOP_FILE" ]]; then
  echo "Stop file $STOP_FILE already exists. Remove it to start the loop:" >&2
  echo "    rm -f $STOP_FILE" >&2
  exit 1
fi

iter=0
while true; do
  # --- stop check (top of loop) --------------------------------------------
  if [[ -e "$STOP_FILE" ]]; then
    echo "[loop] stop file $STOP_FILE present — exiting loop cleanly."
    break
  fi

  iter=$((iter + 1))
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  session_log="$LOG_DIR/s1_session_${ts}.log"
  echo
  echo "[loop] ===== launching session #${iter} at ${ts} ====="
  echo "[loop] session log: $session_log"

  # --- launch ONE fresh session --------------------------------------------
  # Always run in the repo root so CLAUDE.md auto-discovery + relative scripts
  # resolve. We feed the prompt on stdin. `|| true` keeps the loop alive even
  # if this session exits non-zero (crash, transient API error, etc.).
  #
  # IMPORTANT: this is a SINGLE foreground invocation. We never background the
  # claude process and then pkill it — that is the footgun described above. The
  # loop's control flow is simply: run session -> it exits -> relaunch.
  (
    cd "$REPO_ROOT" || exit 1
    "$CLAUDE_BIN" "${CLAUDE_FLAGS[@]}" < "$PROMPT_FILE"
  ) 2>&1 | tee "$session_log" || true

  rc=${PIPESTATUS[0]}
  echo "[loop] session #${iter} exited (rc=${rc}) at $(date -u +%Y%m%dT%H%M%SZ)."

  # --- stop check (before relaunch) ----------------------------------------
  # A session may itself `touch /tmp/s1_loop_stop` when S1 is DONE.
  if [[ -e "$STOP_FILE" ]]; then
    echo "[loop] stop file $STOP_FILE present after session — exiting loop cleanly."
    break
  fi

  echo "[loop] relaunching a FRESH session in ${RELAUNCH_SLEEP}s ..."
  sleep "$RELAUNCH_SLEEP" || true
done

echo "[loop] stopped after ${iter} session(s). To restart: rm -f $STOP_FILE && scripts/s1_session_loop.sh"

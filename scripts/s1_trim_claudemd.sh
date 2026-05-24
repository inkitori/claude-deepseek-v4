#!/usr/bin/env bash
# s1_trim_claudemd.sh — keep CLAUDE.md LEAN for the self-perpetuating S1 handoff loop.
#
# WHAT IT DOES
#   CLAUDE.md accumulates a long historical PHASE narrative (PHASE 2, 3, 4, ...).
#   Each fresh loop session auto-loads CLAUDE.md, so the more PHASE history piles
#   up, the more tokens every session burns before doing any work. This script
#   moves the OBSOLETE PHASE sections out of CLAUDE.md and APPENDS them verbatim to
#   CLAUDE.full.md (the project's long-term archive), while KEEPING in CLAUDE.md:
#     * top matter: the title, "## Goal", "## The bug", "## Success gate"
#     * the "## First action" and "## If it fails" sections (operational, not history)
#     * the MOST RECENT 2 "## PHASE N" sections (current diagnosis state)
#     * "## How to validate", "## Plumbing", "## Pitfalls"
#     * "## Slice bootstrap" and "## Discipline"
#   i.e. it removes only PHASE sections OLDER than the most-recent two. Nothing is
#   ever deleted — everything trimmed is appended to CLAUDE.full.md first.
#
# SAFETY / IDEMPOTENCE
#   * Backs up CLAUDE.md -> CLAUDE.md.bak before writing anything.
#   * Idempotent: if there are <=2 PHASE sections, it makes NO changes (exit 0).
#   * Re-running after a trim is a no-op (the old PHASEs are already gone from
#     CLAUDE.md; only <=2 remain).
#   * Pure awk text surgery on whole "## " sections — never edits inside a section.
#
# USAGE
#   bash scripts/s1_trim_claudemd.sh           # trim in place
#   DRY_RUN=1 bash scripts/s1_trim_claudemd.sh # print plan + counts, change nothing
#
# NOTE: a "## PHASE" section runs from its "## PHASE ..." heading line up to (but
#   not including) the next top-level "## " heading (or EOF). KEEP=2 most recent.

set -euo pipefail

# --- locate the repo root so this works from anywhere ----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLAUDE_MD="$REPO_ROOT/CLAUDE.md"
FULL_MD="$REPO_ROOT/CLAUDE.full.md"
BAK="$REPO_ROOT/CLAUDE.md.bak"

# How many of the most-recent PHASE sections to KEEP in CLAUDE.md.
KEEP="${KEEP:-2}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$CLAUDE_MD" ]]; then
  echo "ERROR: $CLAUDE_MD not found" >&2
  exit 1
fi

# --- count PHASE sections (lines beginning with '## PHASE') ----------------
total_phases="$(grep -cE '^## PHASE' "$CLAUDE_MD" || true)"
before_lines="$(wc -l < "$CLAUDE_MD")"

echo "CLAUDE.md before: ${before_lines} lines, ${total_phases} PHASE sections (KEEP=${KEEP})"

if [[ "$total_phases" -le "$KEEP" ]]; then
  echo "Nothing to trim (<= ${KEEP} PHASE sections). No changes made."
  exit 0
fi

# Number of PHASE sections to MOVE = all but the most-recent KEEP.
to_move=$(( total_phases - KEEP ))
echo "Will move the ${to_move} oldest PHASE section(s) to CLAUDE.full.md, keep the ${KEEP} newest."

# --- awk splitter ----------------------------------------------------------
# Emits trimmed CLAUDE.md to fd 3 (KEEP_OUT) and the moved sections to fd 4
# (MOVE_OUT). We process whole "## " sections. A PHASE section is "moved" only
# if its 1-based PHASE index <= to_move; everything else (non-PHASE sections and
# the most-recent KEEP PHASE sections) is kept.
#
# Implementation: buffer the current "## " section; on the next "## " heading
# (or END) decide keep-vs-move for the buffered section and flush it.
awk_prog='
BEGIN { phase_seen = 0; have = 0; is_phase = 0; }

function flush() {
  if (!have) return;
  if (is_phase && phase_seen <= TO_MOVE) {
    printf "%s", buf > MOVE_OUT;     # archive this old PHASE section
  } else {
    printf "%s", buf > KEEP_OUT;     # keep everything else
  }
  have = 0; buf = ""; is_phase = 0;
}

# A new top-level "## " heading starts a new section.
/^## / {
  flush();
  have = 1;
  buf = $0 ORS;
  if ($0 ~ /^## PHASE/) { is_phase = 1; phase_seen++; }
  else                  { is_phase = 0; }
  next;
}

# Any other line: part of the current section, or (before the first "## ")
# part of the top matter, which we always keep.
{
  if (have) { buf = buf $0 ORS; }
  else      { printf "%s%s", $0, ORS > KEEP_OUT; }   # top matter, pre-first-heading
}

END { flush(); }
'

TMP_KEEP="$(mktemp)"
TMP_MOVE="$(mktemp)"
cleanup() { rm -f "$TMP_KEEP" "$TMP_MOVE"; }
trap cleanup EXIT

awk -v TO_MOVE="$to_move" \
    -v KEEP_OUT="$TMP_KEEP" \
    -v MOVE_OUT="$TMP_MOVE" \
    "$awk_prog" "$CLAUDE_MD"

# Sanity: no content may be lost. (kept + moved) line count must equal the
# original. awk's section buffers preserve every byte, but verify anyway.
kept_lines="$(wc -l < "$TMP_KEEP")"
moved_lines="$(wc -l < "$TMP_MOVE")"
sum=$(( kept_lines + moved_lines ))
if [[ "$sum" -ne "$before_lines" ]]; then
  echo "ERROR: line accounting mismatch (kept ${kept_lines} + moved ${moved_lines} = ${sum} != ${before_lines}). Aborting, no changes made." >&2
  exit 1
fi

if [[ "$moved_lines" -eq 0 ]]; then
  echo "Computed 0 lines to move; no changes made."
  exit 0
fi

echo "Plan: keep ${kept_lines} lines in CLAUDE.md, append ${moved_lines} lines to CLAUDE.full.md."

if [[ "$DRY_RUN" != "0" ]]; then
  echo "DRY_RUN=1 — no files written. (CLAUDE.full.md would gain ${moved_lines} lines.)"
  exit 0
fi

# --- commit the change -----------------------------------------------------
cp -f "$CLAUDE_MD" "$BAK"
echo "Backed up CLAUDE.md -> ${BAK}"

# Append moved sections to CLAUDE.full.md with a dated banner so the archive
# stays navigable. CLAUDE.full.md is created if it does not yet exist.
{
  echo ""
  echo "<!-- ===== trimmed from CLAUDE.md by s1_trim_claudemd.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ) ===== -->"
  echo ""
  cat "$TMP_MOVE"
} >> "$FULL_MD"

# Replace CLAUDE.md with the trimmed version.
cp -f "$TMP_KEEP" "$CLAUDE_MD"

after_lines="$(wc -l < "$CLAUDE_MD")"
after_phases="$(grep -cE '^## PHASE' "$CLAUDE_MD" || true)"
full_lines="$(wc -l < "$FULL_MD")"

echo "DONE."
echo "  CLAUDE.md after:      ${after_lines} lines, ${after_phases} PHASE sections (was ${before_lines} / ${total_phases})"
echo "  CLAUDE.full.md after: ${full_lines} lines (appended ${moved_lines})"
echo "  Backup:               ${BAK}"

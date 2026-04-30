# PROGRESS.md (archived)

This file held a chronological build log of the v1–v8 overnight
sessions that built the V4 implementation from scratch (Phase 0
"setup" through Phase 7 "hardening"). It was append-only and
session-spec-mandated.

The narrative is preserved in `git log` for anyone who needs the
history. Day-to-day, working memory lives in:

* [`../../CLAUDE.md`](../../CLAUDE.md) — runbook, current verified
  state, optimization knobs, pitfalls.
* `git log --oneline work/tpu-inference/` — actual change history.

If you find yourself wanting to append to a file like this during
a future agent iteration, **don't** — write the change in the
commit message and update CLAUDE.md if the change is durable
(operational knowledge that survives this session).

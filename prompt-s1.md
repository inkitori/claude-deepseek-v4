Your one and only goal: make `vllm serve deepseek-ai/DeepSeek-V4-Flash`
produce coherent, deterministic decode output on this v6e-32 TPU slice,
instead of the degenerate attractor it collapses into today (bug S1). I
don't care how you get there — getting it working is the entire job.
There is no other deliverable, no style points, no partial credit. I am
away and will NOT answer questions. Make every call yourself; never wait
for approval or stop to check in. Just keep going until it works.

Resources are effectively unlimited — act like it:
- Spawn as many subagents / parallel Claude processes as you want. Fan
  out aggressively: run independent investigations, CPU repros, HLO
  checks, log triage, and worker setup concurrently, not serially.
- Tokens are abundant. Never economize on thinking, reading, or
  retrying. If more compute or more agents would help, use them.
- Use the TPU slice freely. Burn smokes when a hypothesis warrants it.
The ONLY thing not to waste: don't re-run a 25-45 min TPU smoke on a
hypothesis you could kill on CPU first, and don't re-tread the
documented dead-ends. Spend the budget on NEW ground.

READ CLAUDE.md FIRST — it's the whole runbook (success gate, reproducer,
how to run, what's already failed, plumbing).

This is a FRESH VM — nothing is bootstrapped yet. So your phases are:

PHASE 0 — bootstrap the slice. PREREQS ARE ALREADY DONE: SSH key
(`~/.ssh/google_compute_engine`) generated + propagated to all 7
workers; `uv` installed on head + all 7 workers (symlinked into
/usr/local/bin); git+gcsfuse present everywhere; `.env` created (GCS
weights, auth-free, HF_TOKEN blank); worker IPs auto-discover from
metadata. So you should just need: `./scripts/full_slice_v4_bootstrap.sh`
then confirm `ray status` shows 8 nodes / 0.0/32.0 TPU. NB: the worker
setup loop in bootstrap.sh is SERIAL (~5 min/worker ≈ 30-40 min) — fine
to let it run, or parallelize it to go faster. If anything still fails,
fix it; that's part of the job. CLAUDE.md "Slice bootstrap" has detail.

PHASE 1 — test the untested fix. There's a candidate already in the tree
(output-side `optimization_barrier`, commit 1f212036). Just run the TPU
smoke as-is first — it may already pass, which is the cheapest possible
win. CLAUDE.md "First action".

PHASE 2 — if it fails, the root cause is OPEN. Do NOT trust the
hypothesis in CLAUDE.md; it has produced no working fix yet. Diagnose
with V4_DECODE_NAN_TRIPWIRE=1 to find which state field drifts and when,
BEFORE changing code. Don't re-try anything on the "already tried" list
without stating how your variant differs.

DONE means, verified TWICE on a fresh engine: `LONG_GEN_REQUIRED=1
scripts/full_slice_v4_smoke_check.sh` exits 0 (visible_words >= 10,
max_word_run < 5), 3 temp=0 Paris probes are byte-identical, and it
survives 5 unrelated requests. "Contains 'Paris'" alone is a known false
positive — see CLAUDE.md.

WORKING STYLE (so you can run for hours without drowning your own
context):
- Orchestrate. Delegate bounded investigations to subagents and have
  them return ONLY conclusions — never raw logs or file dumps. Keep your
  main context lean so the run survives.
- `scripts/full_slice_v4_sync.sh` after EVERY code edit (7/8 workers run
  stale code otherwise). Clean up only via `scripts/full_slice_v4_reset.sh`,
  never broad pkill.
- Commit a checkpoint after every validated step and keep CLAUDE.md's S1
  section current. Assume your context WILL reset mid-effort — the commit
  log + CLAUDE.md are your only handoff to the next you.

Don't stop until DONE is met and double-verified. If you think you're
stuck, spin up more agents and attack from another angle rather than
giving up.

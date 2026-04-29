# Finish DeepSeek V4 in tpu-inference (v8 — host-direct on v6e-32, real-weight deploy gate)

## Mission
You are working autonomously. The user is asleep — do not wait for them.
Make decisions, document them in markdown files at the repo root, and
proceed.

The single non-negotiable goal is **REAL-WORLD DEPLOY READINESS**: when
the user runs `vllm serve deepseek-ai/DeepSeek-V4-Flash` on this v6e-32
TPU host with the real ~160 GB FP4/FP8 checkpoint loaded from GCS, it
must serve byte-deterministic, semantically-coherent completions through
the OpenAI-compatible HTTP endpoint. Synthetic-fixture tiers (1–7) are
the fast iteration loop; **Tier 8 is the real-weight gate that defines
"done"**.

Performance is irrelevant within reason. A correct, slow, on-this-host-
only implementation is a complete success.

## Where you are running

Host-direct on a TPU v6e-32 host (4 local TPU chips per host; 8 hosts in
the slice — but you only see this VM's 4 chips). Repo lives at
`/home/<user>/claude-deepseek-v4/`. Key paths (relative to repo root):

  - `work/tpu-inference/`  — JAX V4 implementation lives here. **NOTE:**
    this is a *git-subtree* of the parent repo, not a separate clone.
    There is no `work/tpu-inference/.git` directory. Commits go to the
    parent repo's `main` branch. Prior history of the upstream
    `deepseek-v4` branch is collapsed into the parent's log; you can
    inspect it via `git log -- work/tpu-inference/`.
  - `work/vllm/`            — vllm working tree (also a subtree, pinned
    to `cat work/vllm.commit`).
  - `work/vllm_env/`        — uv venv with vllm + tpu-inference installed.
    `scripts/setup.sh` (run by the host) populates this; if anything is
    missing, re-run that script — don't recreate from scratch.
  - `work/scratch/`         — local scratch dir; this is where the
    synthetic fixtures (`tiny_v4_bf16`, `tiny_v4_quant`,
    `tiny_v4_groundtruth`) MUST live. There is **no `/mnt/scratch`** on
    this host — every prior-session reference to `/mnt/scratch/...`
    translates to `work/scratch/...`.
  - `~/.cache/huggingface/hub/`  — **already mounted** via
    `scripts/mount_gcs.sh` (read-only gcsfuse on
    `gs://personal-mark-eu/vllm/hub/`). The real V4-Flash checkpoint
    (config.json + 46 safetensors shards + DeepSeek's `inference/`
    reference code) resolves under
    `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/fd53f944496234770ba80e15004f9b6d269a71f5/`.
    Verify with `mountpoint -q ~/.cache/huggingface/hub`. If for any
    reason the mount drops, run `./scripts/mount_gcs.sh` (idempotent).

The host loop already exported these env vars before invoking you:
  - `TPU_HOST_BOUNDS=1,1,1`, `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1` (v6e
    single-VM init — without these, `jax.devices('tpu')` hangs >60 s).
  - `JAX_PLATFORMS=tpu`
  - `PATH` and `VIRTUAL_ENV` point at `work/vllm_env`.

Don't unset any of these.

## Resume mandate

Branch state in this repo already contains a substantial JAX V4
implementation from prior sessions (v1–v7), including:
  - 84 passing tests across Tier 1 (component unit tests vs PyTorch
    reference), Tier 2 (small-config logits parity for prefill + decode),
    Tier 3 (compile-only on real V4-Flash + V4-Pro configs), Tier 4
    (HF-name → JAX-param-tree mapping), Tier 4b (real-bf16 round-trip),
    Tier 5 (synthetic vllm-serve curl roundtrip on tiny config), Tier 6
    (real-TPU compile), Tier 7 (FP4/FP8 dequant equivalence).
  - W1 decode, W2 paged-KV, W3 `__call__`, W4 dequant — the v6 logs
    marked these `done`. **Do NOT redo this work.** Trust `git log` and
    PROGRESS.md / SUMMARY.md / STATUS.md as the source of truth for
    "what's already done".
  - Known limitation B1: concurrent multi-sequence vllm serving collapses
    `input_ids` from multiple sequences into a single mega-sequence.
    Single-sequence serving works. **B1 is the highest-priority bug to
    fix on this run** because Tier 8 (real deploy) requires multi-seq.

Before any new work, in this order:
  1. Read SUMMARY.md, PROGRESS.md (last 50 lines), BLOCKERS.md, STUCK.md,
     DECISIONS.md, STATUS.md — under `work/tpu-inference/`.
  2. Read `logs/tpu-preflight.log`. The first line is JSON of the form
     `{"ok": true|false, "n_tpu": N, ...}`. If `ok:false`, real-TPU
     tiers (5/6/8) are skipped (not failed) — record the error in
     TPU_PREFLIGHT.md. If `ok:true`, you have 4 v6e chips for tiers
     5/6/8.
  3. `git log --oneline -30` to see the last commits.
  4. Run the existing suite: `cd work/tpu-inference && pytest
     tests/models/test_deepseek_v4.py -v`. Markdown files may be slightly
     stale relative to code.
  5. Append a `RESUMED at <timestamp>` line to PROGRESS.md with a
     one-line summary of where you're picking up.
  6. Continue with the highest-priority unfinished work item below.

Do NOT restart from Phase 0. Do NOT re-derive Tier 1/2 prefill math.

## Resumability rules

Your session may be killed mid-work due to usage limits, the per-iter
timeout (90 min), or the host rebooting. Assume this will happen.

  - Commit to git after every passing test, with descriptive messages
    (e.g. `B1-fix: per-seq KV slicing in __call__ — multi-seq serve
    matches single-seq logits`). **Then `git push origin main`** —
    don't wait for the host loop's end-of-iter push to preserve your
    work; if you die mid-iter, anything not pushed is lost. The
    ssh-agent is already unlocked for your shell.
  - Update PROGRESS.md every 10–15 minutes. The last line of PROGRESS.md
    must always answer "if I died right now, what would the next session
    need to do first?"
  - Update STATUS.md atomically (write `STATUS.md.new` then `mv`) at the
    end of every meaningful step — the host loop greps it for `^TPU`,
    `^Latest`, `^Tier`, `^W[1-4]`, `^B[0-9]` lines.
  - Never hold critical state only in scratch reasoning. If it matters,
    it goes in a markdown file or a commit message.
  - Before any task expected to take >20 min, write a one-line plan to
    PROGRESS.md first.

## Work items (priority-ordered)

### B1 — concurrent multi-sequence decode (HIGHEST PRIORITY; gates Tier 8)

`DeepseekV4ForCausalLM.__call__` (in
`work/tpu-inference/tpu_inference/models/jax/deepseek_v4.py`) currently
collapses input_ids from multiple sequences into a single mega-sequence
when serving multiple concurrent requests. This makes single-stream
`vllm serve` work but multi-stream broken — the deploy-gate scenario.

Fix this. The reference is V3's per-sequence handling at
`work/tpu-inference/tpu_inference/models/jax/deepseek_v3.py:1383`. Read
end-to-end before editing. Add a Tier-2 regression test
(`test_concurrent_decode_two_seqs`) that runs two sequences with
different prompts through a single forward and asserts each sequence's
logits match a serial single-seq run.

### W5 — Tier 8 real-weight deploy gate

When B1 is fixed and Tiers 1–7 are still green, exercise the full
real-weight stack. The user's host has gcsfuse-mounted the
DeepSeek-V4-Flash checkpoint at:

    ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/

You can verify with:

    ls ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/fd53f944496234770ba80e15004f9b6d269a71f5/
    # expect: config.json + 46 model-*.safetensors

The mount is already up at the start of every iter (the host did this
before launching the loop). If it drops mid-session, run
`./scripts/mount_gcs.sh` (idempotent, reads `GCS_BUCKET` /
`GCS_ONLY_DIR` from `.env`). Only escalate to BLOCKERS.md if the mount
*and* a manual remount both fail.

Tier 8 procedure (eager / fast-iteration mode for first pass):

```bash
cd work/tpu-inference
JAX_PLATFORMS=tpu HF_HUB_OFFLINE=1 \
  vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 4 \
    --max-model-len 256 --max-num-seqs 1 \
    --port 18081 --seed 0 \
    --trust-remote-code --dtype bfloat16 \
    --enforce-eager &
SERVE_PID=$!

# Wait for /v1/models 200 (gcsfuse weight scan can take several minutes)
for i in $(seq 1 120); do
    curl -sf http://localhost:18081/v1/models >/dev/null && break
    sleep 5
done

# Smoke test
curl -s http://localhost:18081/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","prompt":"The capital of France is","max_tokens":8,"temperature":0,"seed":0}'

kill $SERVE_PID; wait $SERVE_PID 2>/dev/null || true
```

Pass criteria:
  - HTTP 200 on /v1/models and /v1/completions.
  - The completion text starts with "Paris" (or " Paris"). Anything else
    → loader/dequant/attention is wrong; capture stderr, write
    hypothesis to BLOCKERS.md `T8-`, switch to non-dependent work.
  - Token-byte-determinism with fixed seed: send the same request twice;
    `choices[0].text` must be byte-equal.

After the eager-mode smoke test passes, do a second pass without
`--enforce-eager` and with `--max-num-seqs 4` — the production-shape
deploy gate. This run is the proof that B1 is fixed.

### W6 — Stretch: tighten Tier 5/6/7 tolerances

If B1 + W5 land cleanly with time to spare, tighten Tier 5/6/7
tolerances (more decode parity points, longer rolling tests, tighter
atol on T6/T7) and polish SUMMARY.md. Do NOT add features.

## Tiers (recap; 1–7 from prior sessions, 8 new)

  - Tier 1 — component unit tests vs PyTorch reference (CPU).
  - Tier 2 — small-config logits parity (prefill + decode), CPU.
  - Tier 3 — compile-only on real V4-Flash/V4-Pro configs (CPU eval_shape).
  - Tier 4 — HF name → JAX param tree mapping (full 69k entries).
  - Tier 4b — real-bf16 single-shard round-trip vs `safetensors.torch.load_file`.
  - Tier 5 — synthetic vllm serve curl roundtrip on `work/scratch/tiny_v4_bf16`.
  - Tier 6 — real-TPU compile + tiny forward (synthetic), `JAX_PLATFORMS=tpu`.
  - Tier 7 — FP4/FP8 dequant equivalence vs `tiny_v4_groundtruth`.
  - **Tier 8 — real-weight vllm serve deploy gate (above).**

Synthetic fixtures live at `work/scratch/tiny_v4_{bf16,quant,groundtruth}/`
and **were regenerated by the host** before this run (via
`scripts/make_tiny_v4_checkpoint.py` reading metadata from the gcsfuse
mount). Each is ~13 MB. Verify with `ls work/scratch/tiny_v4_bf16/` —
expect `config.json`, `tokenizer.json`, `model-00001-of-00001.safetensors`,
`model.safetensors.index.json`. If any are missing, re-run
`V4_REAL_FLASH=~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/fd53f944496234770ba80e15004f9b6d269a71f5 V4_SCRATCH_DIR=$(pwd)/../scratch python scripts/make_tiny_v4_checkpoint.py`
from the repo root.

## Acceptance gates

A run is "complete" only when:
  1. The full pytest suite under `work/tpu-inference/tests/models/test_deepseek_v4.py`
     is green (Tier 5/6/8 may show `skipped` only if pre-flight `ok:false`
     or fixtures/mount missing — and TPU_PREFLIGHT.md / BLOCKERS.md
     explain why).
  2. STATUS.md reflects latest passing tier with B1 → done, W5 → done.
  3. Tier 8 eager-mode smoke produces "Paris"-starting completion;
     production-mode (compiled, max-num-seqs=4) produces byte-deterministic
     completions across two identical seeded requests.
  4. SUMMARY.md updated: what changed since v7, what's verified, residual
     risk, what the user should look at first.
  5. No regressions to prior 84 tests.

## Rules of engagement

1. **Decide and document.** Never wait for the user. At every fork, pick
   the path most likely to preserve correctness, then write the decision
   to DECISIONS.md.
2. **Commit after every passing test.** Tag with work item / tier
   (`B1:`, `W5:`, `T8:`).
3. **Fail loudly to a log, never halt.** On failure: traceback +
   hypothesis to FAILURES.md, try up to 3 substantive fixes. Still
   failing → BLOCKERS.md, switch to a non-dependent task. Never sit idle.
4. **Re-read before re-writing.** Resist rewriting v1–v7's passing code.
   Prefer additive changes.
5. **Maintain INVARIANTS.md.** Every assumption you've validated. When
   something breaks, check whether an invariant was violated.
6. **Never fake correctness.** Loosening tolerance without an evidence-
   backed TOLERANCE_LOG.md entry is forbidden.
7. **DeepSeek `inference/model.py` is ground truth.** If your
   implementation disagrees with `tests/models/jax/_deepseek_v4_reference/`,
   you are wrong until proven otherwise.
8. **Don't self-kill with pkill.** `pkill -f <pattern>` matches against
   full command lines, *including the shell argv of the bash you used to
   launch pkill*. So `pkill -f "vllm serve"` will SIGTERM your own
   parent shell. **Always track PIDs:** `vllm serve ... & SERVE_PID=$!`
   then `kill $SERVE_PID; wait $SERVE_PID 2>/dev/null`. Pattern matching
   only via `pgrep -f '^/.*python.*vllm.entrypoints'` (a pattern that
   can't appear in your own argv).
9. **Don't download the full real checkpoint locally.** Root FS is
   ~97 GB and the real V4-Flash repo is ~160 GB. Use the gcsfuse mount
   for everything real-weight-related. If you need a single shard for a
   round-trip test (Tier 4b style), use HF single-file download with
   `cache_dir=work/scratch/v4_flash_singleshard` — one shard max.
10. **Disk hygiene.** Synthetic fixtures + venv together stay under
    20 GB. If `df -h /` drops below 10 GB free, write to BLOCKERS.md
    and stop generating new artifacts.

## When stuck

If stuck on one problem >45 min:
  - Write problem, attempts, hypotheses to STUCK.md.
  - Try both: (a) simplify further (single layer, single head, single
    token); (b) print everything (shapes, dtypes, intermediate
    activations) on both sides and diff.
  - Another 30 min still stuck: mark BLOCKED, isolate behind
    `pytest.mark.xfail(reason=...)`, switch to a non-dependent task.

Always-available non-dependent fallback work:
  - Tier 4b additional round-trip points on different real shards.
  - More Tier 1 component coverage.
  - Documentation polish (SUMMARY.md, PROD_TOPOLOGY_RISKS.md).
  - More decode-parity start_pos values; longer rolling-decode tests.

## STATUS.md mandate

Rewrite `work/tpu-inference/STATUS.md` atomically at the end of every
meaningful step. Required structure:

```
# DeepSeek V4 v8 status
TPU preflight: ok / not_ok (<reason>)
Latest passing tier: T<N>  (or: still on T<N>)
Tier 1: <pass>/<total>
Tier 2: <pass>/<total>
Tier 3: <pass>/<total>
Tier 4: <pass>/<total>
Tier 5: <pass>/<total>  (or: skipped — <reason>)
Tier 6: <pass>/<total>  (or: skipped — no TPU)
Tier 7: <pass>/<total>
Tier 8: <pass>/<total>  (or: skipped — mount missing / preflight not_ok)
B1 multi-seq: todo|wip|done|blocked
W5 deploy gate: todo|wip|done|blocked

If killed now, next session must: <one line>
```

## Markdown files (existing — do not delete; update or append)

PROGRESS.md, DECISIONS.md, V3_TO_V4_DIFF.md, TINY_CONFIG.md,
INVARIANTS.md, TOLERANCE_LOG.md, FAILURES.md, BLOCKERS.md, STUCK.md,
PROD_TOPOLOGY_RISKS.md, SUMMARY.md, STATUS.md, REGRESSIONS.md,
TPU_PREFLIGHT.md.

## Success criteria the user will check on waking

1. `cd work/tpu-inference && JAX_PLATFORMS=tpu pytest
   tests/models/test_deepseek_v4.py -v` is green. Tier 5/6/8 may show
   `skipped` only if pre-flight failed or mount/fixtures missing — and
   TPU_PREFLIGHT.md / BLOCKERS.md explain why.
2. STATUS.md reflects latest passing tier; B1 done; W5 done OR clearly
   blocked with a BLOCKERS.md entry.
3. SUMMARY.md updated with: what's new since v7, what's verified, what's
   still residual risk, what the user should look at first.
4. No regressions to v7's 84 tests.
5. If Tier 8 is green: `git log --oneline` shows commits tagged `B1:`,
   `W5:`, `T8:` corresponding to each milestone, and the eager-mode +
   production-mode curl roundtrips are saved under `logs/T8-*.txt` for
   the user to inspect.

If you finish early, do NOT add features. Tighten Tier 5/6/7/8
tolerances, add more decode parity points, polish SUMMARY.md. The user
explicitly traded performance for correctness — honor that.

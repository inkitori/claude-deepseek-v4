# Production-readiness blockers

This file is a short pointer at the prioritized work list. The
authoritative backlog with file:line references and full
rationale lives in **[../../CLAUDE.md](../../CLAUDE.md)**
"Production-readiness backlog". Read that first.

## Status

The Tier-8 deploy gate is GREEN as of 2026-04-30 — `./run.sh
serve` plus `scripts/full_slice_v4_smoke_check.sh` returns
`PASS: deterministic completion contains 'Paris'` reliably,
cold compile ~97s, warm-cache curl sub-second. Older blockers
(B1 multi-seq dispatch, B4 vLLM MLA-classification gate, the
MoE-vectorize HBM OOM, the 126 `Involuntary full
rematerialization` warnings on `compressor.ape`) are all
resolved. History lives in `git log`.

## Current production-readiness blockers (in priority order)

### Tier S — silent correctness bombs (fix first)

* **S1.** Decode is not real decode — every step recomputes
  prefill on the full prompt+generated context. Headline bug;
  unblocks A1, B1, S5.
* **S2.** Multi-sequence dispatch is a Python loop in eager
  mode; doesn't jit; one user blocks all others.
* **S3.** `--reasoning-parser deepseek_v4` and
  `--tool-call-parser deepseek_v4` not enabled in the smoke
  launcher. The parsers exist upstream — one-line launcher fix.
* **S4.** Chat encoding — RESOLVED upstream. vLLM's
  `DeepseekV4Tokenizer` (auto-loaded via `tokenizer_mode='deepseek_v4'`)
  calls upstream `encode_messages` directly, byte-equal to the
  V4-Flash reference encoder across chat / thinking / tools /
  reasoning_effort. Pinned by `TestVllmChatTemplateParity`.
  `latest_reminder` isn't reachable via the chat-completions API
  surface; punt unless a downstream consumer surfaces a need.
* **S5.** MTP speculative-decoding hook not wired into
  `tpu_inference/runner/speculative_decoding_manager.py`. 1.5–2×
  decode throughput on the table.
* **S6.** Sampling parameters (`temperature>0`, `top_p`,
  `top_k`, penalties, `n>1`, `logprobs`) untested under load.
* **S7.** Streaming (SSE) responses unverified.

### Tier A — production deployment infra

* **A1.** `MAX_LEN=256, MAX_SEQS=1` hard-coded in the smoke
  launcher (depends on S1).
* **A2.** Persistent compile cache is host-local and ephemeral.
* **A3.** No engine crash recovery / drain-on-SIGTERM.
* **A4.** No metrics / observability (vLLM's `--enable-metrics`
  not set).
* **A5.** No TLS / authentication / per-key rate limiting.
* **A6.** Single slice — no horizontal scale.

### Tier B — known performance work

* **B1.** Sparse-attention Pallas kernel
  (`layers/jax/attention/deepseek_v4_attention.py:131` is
  fully-materialized; correctness over perf per DECISIONS D2).
* **B2.** True sparse MoE dispatch via existing
  `kernels/megablox/gmm.py` (today's `moe_forward` is
  vectorized-dense; 32× over true sparse for top_k=8 / E=256).
* **B3.** SPMD `Involuntary full rematerialization` warning
  audit. `compressor.ape` family already eliminated; sweep the
  rest.
* **B4.** AOT compile + binary persist (per-host because of
  the cache-fingerprint finding).

### Tier C — quality gates

C1 lm-eval-harness scores, C2 long-context functional, C3 math
regression under load, C4 tokenizer edges, C5 refusal
preservation. **C1 is the gate for claiming "we serve
V4-Flash" honestly.**

### Tier D — code-hygiene

D1 test bloat consolidation
(`tests/models/jax/test_deepseek_v4.py` is 2997 LOC, 30
classes; ~4× peer models), D2 stale comment cleanup
(`__call__` docstring at `models/jax/deepseek_v4.py:1395-1409`
references archived B1).

For full rationale, file:line references, and verification
patterns for each item, read **CLAUDE.md "Production-readiness
backlog"**.

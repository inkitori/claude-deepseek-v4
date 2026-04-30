# Production-readiness blockers

The authoritative backlog (with file:line references and full
rationale) lives in **[../../CLAUDE.md](../../CLAUDE.md)**
"Production-readiness backlog". Read that first.

## Status

Smoke gate is GREEN — `./run.sh serve` plus
`scripts/full_slice_v4_smoke_check.sh` returns
`PASS: deterministic completion contains 'Paris'` reliably,
cold compile ~97s, warm-cache curl sub-second.

## Current blockers (priority order)

* **Tier S — silent correctness bombs** (S2 multi-seq
  dispatch; S3 tool-call runtime probe; S5 MTP hook). S1 done
  (env-gated behind `V4_DECODE_STATE=1`, default-flip needs
  REASONING_REQUIRED reverification); S4/S6/S7 resolved.
* **Tier A — production infra** (A1 max-len lift depends on
  S1; A2–A6 cache durability, crash recovery, metrics, TLS,
  multi-slice).
* **Tier B — perf** (B1 sparse-attn Pallas, B2 megablox MoE,
  B3 remat audit, B4 AOT persist).
* **Tier C — quality gates** (C1 lm-eval is the gate for
  claiming "we serve V4-Flash" honestly).
* **Tier D — janitorial** (D1 test bloat).

For full rationale, file:line references, and verification
patterns per item, read **CLAUDE.md "Production-readiness
backlog"**.

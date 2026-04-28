# DeepSeek V4 Implementation Progress

Started 2026-04-28 ~08:36 UTC. Single autonomous overnight session.

## Phase 0 — Setup
- [x] Branch `deepseek-v4` created.
- [x] transformers 5.6.2 installed (does NOT have deepseek_v4 model_type yet — see DECISIONS.md).
- [x] Downloaded V4-Pro and V4-Flash artifacts to /mnt/scratch: config.json, tokenizer files, safetensors index, encoding/, inference/{model.py, kernel.py, convert.py, generate.py}.
- [x] Read V3 model file (tpu_inference/models/jax/deepseek_v3.py, 1450 lines).
- [x] Read V4 reference inference/model.py end-to-end.
- [x] Wrote V3_TO_V4_DIFF.md.
- [x] Confirmed JAX runs on CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=N`. TPU unavailable on this host (mmap EAGAIN); see DECISIONS.md.

## Phase 1 — Reference oracle
- [x] Wrote `tests/models/jax/_deepseek_v4_reference/{__init__.py, model.py, kernel_stubs.py}`.
- [x] Smoke-tested: tiny-config forward returns `[B, S, vocab_size]` fp32 logits; prefill, decode (with KV state), multi-batch, and MTP all produce the right shape. Reproducible across reset+rerun.

## Phase 2 — Components + Tier 1
TBD.

## Phase 3 — Assembly + Tier 2
TBD.

## Phase 4 — Tier 3 compile-only (v4-8 + simulated v6e-32)
TBD.

## Phase 5 — Weight loader smoke test
TBD.

## Phase 6 — Registration
TBD.

## Phase 7 — Hardening + SUMMARY
TBD.

---

## Resume hint
**If I died right now**, the next session should: read V3_TO_V4_DIFF.md, then run `git log --oneline` to see committed work, then continue Phase 1 (write the PyTorch reference oracle at `tests/models/_v4_reference.py`).

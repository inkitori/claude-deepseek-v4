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
- [x] Wrote `tpu_inference/layers/jax/attention/deepseek_v4_attention.py` — RMSNorm, RoPE, sparse_attn, sinkhorn, Compressor, Indexer, Attention prefill (SWA / CSA / HCA).
- [x] Wrote `tpu_inference/layers/jax/moe/deepseek_v4_moe.py` — Gate (sqrtsoftplus + hash), Expert SwiGLU, MoE dense dispatch.
- [x] Wrote `tpu_inference/models/jax/deepseek_v4.py` — Block (mHC), MTP block, Transformer prefill.
- [x] **All 25 Tier 1 tests pass.** (RMSNorm, RoPE, sparse_attn, sinkhorn doubly-stochastic, Compressor, Indexer, Attention SWA+CSA+HCA, Block at SWA+CSA+HCA+trailing-SWA layers, Gate hash+non-hash, Expert, MoE both modes.)

## Phase 3 — Assembly + Tier 2
- [x] **All 6 Tier 2 E2E tests pass.** prefill 16/32/64, multibatch (4×16), argmax agreement ≥95%, MTP forward parity.

## Phase 4 — Tier 3 compile-only (v4-8 + simulated v6e-32)
- [x] `eval_shape` succeeds on full V4-Flash (43 layers + 1 MTP) and V4-Pro (61 layers + 1 MTP) configs.
- [x] Per-device byte budgets reported.
  - V4-Flash: 543 GB total bf16, 17 GB/device on simulated v6e-32 (matches expectation: 284B params at bf16 ≈ 540 GB).
  - V4-Pro: 2982 GB total bf16, 93 GB/device on simulated v6e-32 (will OOM on real 32GB HBM — needs FP4/FP8 in production; documented in PROD_TOPOLOGY_RISKS.md).
  - KV cache @ 1M context: 0.17–0.97 GB/device (well within HBM).
- [x] `jit().lower().compile()` smoke test on first-2-layers truncated configs of both V4-Flash and V4-Pro — succeeds.

## Phase 5 — Weight loader smoke test
- [x] HF→JAX name mapping covers all 69,187 parameter names in V4-Flash safetensors index. No unmapped names.
- [x] Downloaded shard 1 (~1 GB) and verified each tensor's shape matches the abstract param tree.

## Phase 5 — Weight loader smoke test
TBD.

## Phase 6 — Registration
TBD.

## Phase 7 — Hardening + SUMMARY
TBD.

---

## Resume hint
**If I died right now**, the next session should: read V3_TO_V4_DIFF.md, then run `git log --oneline` to see committed work, then continue Phase 1 (write the PyTorch reference oracle at `tests/models/_v4_reference.py`).

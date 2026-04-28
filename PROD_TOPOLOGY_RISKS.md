# Production-topology risks

Items that I could not validate from this dev box and that the user should test first when they have v6e-32 access.

## TPU access
This dev host does not have working TPU access. JAX `init_backend('tpu')` fails with `FAILED_PRECONDITION: Couldn't mmap [/dev/accel*]: Resource temporarily unavailable`. All testing was done on CPU, including the "v4-8 mesh" version of Tier 3 — see DECISIONS.md D4.

## Risks
1. **Real XLA-on-TPU compilation differences.** CPU XLA may accept HLO that TPU XLA rejects (or vice versa) — e.g. layout constraints, dtype lowering, custom-call availability. Tier 3 verifies `eval_shape` and `lower(...).compile()` succeed on CPU, which catches *most* sharding bugs but cannot exhaust TPU-specific lowering rules.
2. **Pallas / ragged_paged_attention not exercised.** This implementation does NOT use `tpu_inference.kernels.ragged_paged_attention.v3`. It uses dense attention with explicit masking. On v6e-32 with 1M context, the dense path will be too slow / OOM. The user will need to swap in a sparse-attention Pallas kernel (or build one from `inference/kernel.py`'s `sparse_attn_kernel`). Math correctness is unaffected.
3. **HBM budget on v6e-32.** Tier 3 prints estimated bytes-per-device assuming bf16 weights. On v6e-32 with V4-Pro-Base (1.6T params, bf16), per-device weight memory ≈ 100 GB. v6e has 32 GB HBM/chip. This implies that for production the user MUST use the FP4/FP8 mixed precision (real V4 ships FP4 experts + FP8 dense). Our weight loader (Phase 5) does NOT yet dequantize FP4 weights — see Phase-5 limitation below. Without that, V4-Pro will OOM on v6e-32 even before forward.
4. **KV cache size.** For 1M context, even with sliding-window-128 + compressed-pool of `(1M / 4 + 1M / 128) ≈ 258k entries × head_dim 512 × bf16 = 264MB per layer × 61 layers = 16 GB`. Distributed across 32 chips this is ~500MB/chip, well within HBM. But this is per-request — concurrent requests will multiply.
5. **Sharding-axis names.** I used `('data', 'model')` axis names matching `tpu_inference/layers/common/sharding.py`. If the production v6e-32 mesh uses different axis names (e.g. `('attn_data','attn_model','expert')`), the PartitionSpec annotations will silently produce replicated tensors. Documented in test as a CHECK_ME.
6. **Real-weight dtype handling.** Real V4 weights are stored in FP8/FP4 with separate `weight.scale` arrays. The current weight loader (Phase 5) is a name-mapping smoke test only — it does not handle the per-block scale tensor or the FP8/FP4 dequantization. The user should treat the loader as a *map* of name correspondences, and plug in dequantization before any real-weights run.
7. **MTP block.** The MTP block is implemented and tested for forward equivalence. But its integration into vLLM's speculative-decoding hook is NOT done. vLLM-side wiring is required before MTP outputs flow back to the scheduler.

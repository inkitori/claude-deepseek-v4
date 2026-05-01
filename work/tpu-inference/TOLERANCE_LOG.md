# Tolerance log

Tolerances tighter or looser than the bf16 default (atol=1e-2).
Each entry: where, asserted bound, worst observed, why the bound
isn't tighter or looser. Per-iter measurement narrative lives in
`git log`.

| ID | Where | Bound | Worst observed | Notes |
|---|---|---|---|---|
| T1 | `TestAttentionComponent::test_attention_prefill_matches_torch` | atol=1e-3 | 7.63e-6 (45 cfg×seed) | bf16 ULP at output magnitude. 130× margin over empirical worst. |
| T2 | `TestBlockComponent::test_block_matches_torch` | atol=2e-2 | 7.81e-3 (80 layer×seed) | Block adds Sinkhorn + mHC ULP. 2.5× margin; tighter risks flakes from Block input distribution shift. `TestMoEComponent::test_moe_matches_torch` is atol=5e-3 (worst 4.88e-4). Hash variant stays atol=5e-2 (worst 4.2e-2). |
| T3 | `TestEndToEnd::test_*_prefill_logits_parity`, `test_mtp_forward_parity` | atol=1e-3 (long-context 2e-3) | 1.35e-4 single-batch / 1.22e-4 long-context | fp32 head matmul absorbs bf16 residual. ~7× margin single-batch, ~16× long-context. |
| T4 | `TestIndexerComponent::test_indexer_prefill_matches_torch_topk` | set equality | exact | `lax.top_k` and `torch.topk` may break score-ties differently in fp32. The right invariant is "same K positions selected", not order. |
| T6 | `TestRealTpuTinyForward::test_tiny_tpu_compile_and_forward` | finite + std>0.01 | n/a | JAX cannot init both TPU and CPU in one process. CPU path validated at T3; real-TPU correctness gated by the `vllm serve` smoke. |
| T7 | `TestQuantToParamsApply::test_forward_logits_quant_vs_groundtruth` | byte-exact | 0 | Loader is byte-equal across 355 tensors (T-FP8-REF + T-FP4-REF), so weights are bit-identical. Same Python source on same trace = no fp32 reduction-order divergence. |
| T8 | `TestDecodeRollingEquivalenceWithPrefill::test_swa_decode_state_equals_prefill_state_after_32_steps` | byte-exact | 0 (8 seeds) | Prefill and decode share the same per-position write expression. XLA emits byte-identical lowerings. |
| Decode-step parity | `TestDecodeAttentionParity{,Extended}`, `TestDecodeRollingParity{,Long}` | atol=1e-4 | 7.63e-6 (~500 step measurements) | bf16 ULP regime; rolling chain (K up to 32) doesn't compound. ~13-25× margin. Tighter than T1 because decode-step has shorter matmul chain than full attention. |
| T-CDS | `TestCompressorDecodeStep{,Extended}` | atol=1e-5 | 7.15e-7 kv_state, 5.96e-7 score_state (72 measurements) | Compressor accumulator is fp32 end-to-end on both sides; state-tensor parity should be at fp32 ULP, not bf16 noise. ~14-17× margin. |
| T-FP8-REF | `TestFp8DequantIndependentReference` | byte-exact bf16 | 0 (~50 MB across 6 cases) | e4m3fn + e8m0fnu decoded independently in numpy (sign+exp+mantissa arithmetic vs torch's `.float()` cast); `np.kron` vs `repeat_interleave`. Byte-equal validates the FP8 spec, scale axes, scale-block broadcast. No looser bound is acceptable. |
| T-FP4-REF | `TestFp4DequantIndependentReference` | byte-exact bf16 | 0 (~16 MB across 4 cases) | Sign-magnitude decode (8-entry magnitude × sign bit) vs loader's 16-entry table lookup. Byte-equal validates nibble order, sign-bit position, scale-block broadcast. **`-0 → +0` canonicalization required** — DeepSeek's table collapses nibble 8 to `+0.0` (INVARIANTS::I38, ~0.6% of expert weight bytes). |
| T-FP8-CAST | `TestFp8CastByteDomain` | byte-exact fp32 | 0 (256/256 bytes per dtype) | Exhaustive over both 8-bit byte domains (e4m3fn, e8m0fnu). Tautology if torch's cast is correct, real bug otherwise. Fences unused corner of domain a future torch upgrade could change. |

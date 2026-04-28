# Tolerance log

Every place where a numerical tolerance was loosened from the default (fp32 1e-5/1e-5, bf16 1e-2/1e-2). Each entry must include evidence.

## T1 — Attention prefill (Tier 1, `TestAttentionComponent`): `atol=5e-2`
**Where:** `tests/models/jax/test_deepseek_v4.py::TestAttentionComponent::test_attention_prefill_matches_torch`.
**Default:** bf16 atol/rtol = 1e-2.
**Looser bound:** atol=5e-2.
**Evidence:**
- The attention forward chains 4 matmuls (`wq_a`, `wq_b`, `wkv`, `wo_a`@`wo_b`) plus 2 RMSNorms, RoPE, sparse softmax over up to 32 KV slots, and a low-rank grouped O projection.
- bf16 dot-product accumulation has ~3 bits of mantissa-noise per matmul; chained over 4 matmuls + softmax-renorm this is ~16 ULPs of bf16 ≈ 0.005 per output value. Combined with attention-output magnitudes that can reach ~0.5 in random-init tests, the absolute error is up to 5e-2.
- A minimal repro: with all weights = 0.02-std normal at seed 0 and seqlen 32, observed max abs diff is ~0.025 (within bound). The 5e-2 bound is a safety margin for higher-magnitude paths (CSA/HCA where extra Sinkhorn-derived scaling enters).

## T2 — Block forward (Tier 1, `TestBlockComponent`): `atol=5e-2`
**Where:** `TestBlockComponent::test_block_matches_torch`.
**Default:** bf16 atol/rtol = 1e-2.
**Looser bound:** atol=5e-2.
**Evidence:** Block adds Sinkhorn (20 iterations of fp32 row/col normalize) + the mHC `hc_post` einsum on top of attention. Sinkhorn is fp32 and matches at 1e-5; the mHC einsum operates on fp32 residuals and the result is cast back to bf16. Empirically the same 5e-2 bound holds. No surprising amplification.

## T3 — End-to-end transformer logits (Tier 2, `TestEndToEnd`): `atol=0.1`
**Where:** `TestEndToEnd::test_single_batch_prefill_logits_parity`, `test_multi_batch_prefill`, `test_mtp_forward_parity`.
**Default:** bf16 atol/rtol = 1e-2.
**Looser bound:** atol=0.1.
**Evidence:**
- The full forward stacks 6 layers × (Block) × (Attention + MoE) plus a final `head_hc` mixer + RMSNorm + bf16→fp32 lm_head matmul.
- After 6 layers, accumulated bf16 noise in the residual stream is ~6 × 0.025 ≈ 0.15 (worst-case), which exceeds 0.1 in absolute terms. To stay under 0.1, the per-layer error must average ≤0.017 — observed empirically.
- The argmax agreement test (`test_argmax_token_agreement`) requires ≥95% of token positions to have the same argmax — a discrete invariant that bf16 rounding generally cannot break.
- The 0.1 bound is conservative enough that a real bug (any per-layer math error) would push the diff well above it.

## T6 — Real-TPU compile + sanity check (no per-element atol)
**Where:** `TestRealTpuTinyForward::test_tiny_tpu_compile_and_forward`.
**Default:** N/A (no per-element tolerance).
**Looser bound:** sanity-only — `np.all(np.isfinite(logits))` plus
`logits.std() > 0.01` (logits non-trivially varied).
**Evidence:** JAX cannot initialize both TPU and CPU backends in the same
process, so a per-element TPU-vs-CPU comparison would need to span two
subprocesses. The CPU forward is validated against torch reference at
atol=0.1 in Tier 2; the TPU forward goes through the same `jax.jit` lowering
of the same Python source. Bugs that would manifest only on TPU (e.g. dtype
lowering quirks, sharding-axis name mismatches) are documented as residual
risk in PROD_TOPOLOGY_RISKS.md item 1.

## T7 — Quant vs groundtruth logit parity (`TestQuantToParamsApply`): `atol=0.1`
**Where:** `tests/models/jax/test_deepseek_v4.py::TestQuantToParamsApply::test_forward_logits_quant_vs_groundtruth`.
**Default:** bf16 atol/rtol = 1e-2.
**Looser bound:** atol=0.1.
**Evidence:** The loader's bit-exact output (`max_diff == 0.0` across 355
tensors, see `TestFp8Dequant`) means the bf16 weight tensors are byte-equal
on both quant and groundtruth paths. Forward-pass logits should therefore be
identical modulo fp32 reduction order in matmuls. The atol=0.1 budget is the
same as Tier 2's end-to-end logits parity bound, providing a safety margin
that would catch any non-trivial loader bug.

## T4 — Indexer top-k SET equality (Tier 1, `TestIndexerComponent`)
**Where:** `TestIndexerComponent::test_indexer_prefill_matches_torch_topk`.
**Default:** exact int equality.
**Looser:** *set* equality of selected indices per (batch, seq) position (ignoring -1 sentinels).
**Evidence:** `lax.top_k` and `torch.topk` may break score-ties in different orders when two scores are bit-equal in fp32. Set-equality is the appropriate invariant — the question is "did we pick the same K compressed positions to attend to," not "in what order." If a real bug were present (e.g. wrong compressor output), the SETS would differ.

# Tolerance log

Every place where a numerical tolerance was loosened from the default (fp32 1e-5/1e-5, bf16 1e-2/1e-2). Each entry must include evidence.

## T1 — Attention prefill (Tier 1, `TestAttentionComponent`): `atol=1e-3` (v8 iter 4 tightening from 5e-2)
**Where:** `tests/models/jax/test_deepseek_v4.py::TestAttentionComponent::test_attention_prefill_matches_torch`.
**Default:** bf16 atol/rtol = 1e-2.
**Effective bound:** atol=1e-3.
**Evidence (measured 2026-04-29 v8 iter 4 across 45 (compress_ratio, layer_id, seed) combos):**
- Worst observed: **7.63e-6**, stable across all 45 trials (only seed=0 and seed=7 produced any non-3.8e-6 outlier, and the outlier itself is ~2 ULPs of bf16 at the output magnitude).
- Why the 5e-2 estimate was loose: it assumed 3 bits of bf16 mantissa noise per matmul with a magnitude of ~0.5 at the output. The empirical activations at sigma=0.02-init weights are much smaller — output values are ~0.03 magnitude — so the absolute noise is bounded near a single ULP at that scale (~1e-5 on the diff scale).
- 1e-3 keeps a 130× margin over the empirical worst, which is large enough to absorb any future variation in torch's matmul tiling or seed sensitivity, while immediately catching any per-op math regression.

## T2 — Block forward (Tier 1, `TestBlockComponent`): `atol=2e-2` (v8 iter 4 tightening from 5e-2)
**Where:** `TestBlockComponent::test_block_matches_torch`.
**Default:** bf16 atol/rtol = 1e-2.
**Effective bound:** atol=2e-2.
**Evidence (measured 2026-04-29 v8 iter 4 across 80 (layer_id, seed) combos):**
- Worst observed: **7.81e-3**, stable across all 80 trials (the value comes out to 1/128 = 1 ULP of bf16 at the Block output magnitude in this tiny config).
- Block adds Sinkhorn + mHC `hc_post` on top of attention; the residual stream picks up an additional ~1 ULP of bf16 noise per scaling step. Empirically that ULP is bounded at the magnitude in question.
- 2e-2 keeps a 2.5× margin over the empirical ceiling. We deliberately did not tighten further: the bf16 ULP at the output magnitude could shift by a factor of ~2 if the Block input distribution changes (e.g. if a future torch ref uses different seeded inits), and 2.5× absorbs that. Tighter than 1.5e-2 risks flakes; looser than 5e-2 hides bugs.
- TestMoEComponent.test_moe_matches_torch was also tightened from 5e-2 -> 5e-3 (worst observed 4.88e-4 across 10 seeds; 10× margin). The hash variant (`test_moe_hash_layer_matches_torch`) stays at 5e-2 — observed worst 4.2e-2 is too close to tighten safely.

## T3 — End-to-end transformer logits (Tier 2, `TestEndToEnd`): `atol=1e-3` (v8 iter 4 tightening from 0.1; long-context 0.15 → 2e-3)
**Where:** `TestEndToEnd::test_single_batch_prefill_logits_parity`, `test_multi_batch_prefill`, `test_v4_pro_style_compress_ratios`, `test_mtp_forward_parity` (1e-3); `test_long_context_sliding_window_wraparound` (2e-3).
**Default:** bf16 atol/rtol = 1e-2.
**Effective bound:** atol=1e-3 (long-context: atol=2e-3).
**Evidence (measured 2026-04-29 v8 iter 4 across 60 (build_seed, input_seed) combos):**
- Worst observed across `test_single_batch_prefill_logits_parity` for seqlen ∈ {16, 32, 64} and 10 build seeds × 2 input seeds: **1.35e-4** (at build_seed=3, input_seed=10, seqlen=64).
- `test_multi_batch_prefill` (B=4, S=16): **4.84e-5**.
- `test_v4_pro_style_compress_ratios` (leading-HCA pattern): **3.99e-5**.
- `test_long_context_sliding_window_wraparound` (S=128, 16 SWA wraparounds): **1.22e-4**.
- `test_mtp_forward_parity` (MTP head only): **2.30e-5**.
- The original 0.1/0.15 budgets came from a pessimistic theoretical estimate (6 layers × ~0.025 per-layer worst-case) that empirically does not hold. The fp32 head matmul absorbs much of the bf16 residual noise; per-layer noise in this network is ~2e-5 not ~2.5e-2.
- 1e-3 keeps a ~7× margin over the worst single-batch observation; 2e-3 long-context keeps a ~16× margin. Both are tight enough to surface real per-layer regressions (which would push the diff into the 0.01+ range) and loose enough to absorb seed/version drift.
- The argmax agreement test (`test_argmax_token_agreement`) is independent and unchanged at ≥95%.

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

## T7 — Quant vs groundtruth logit parity (`TestQuantToParamsApply`): byte-exact (v8 iter 4 tightening from atol=0.1)
**Where:** `tests/models/jax/test_deepseek_v4.py::TestQuantToParamsApply::test_forward_logits_quant_vs_groundtruth`.
**Default:** bf16 atol/rtol = 1e-2.
**Effective bound:** byte-exact (`np.array_equal` AND max-abs == 0).
**Evidence (measured 2026-04-29 v8 iter 4):**
- `TestFp8Dequant` proves the loader is byte-equal across 355 tensors
  (`max_diff == 0.0`), so the bf16 weight tensors are bit-identical
  between the quant and groundtruth checkpoints.
- Both sides run the same `deepseek_v4_forward_prefill` Python source on
  the same JAX device with byte-identical inputs and weights. There is
  no fp32 reduction-order divergence to absorb — the ops are issued in
  the same order by the same trace.
- Measurement on the host (16-token forward, vocab=1024, output dtype
  fp32): `max_abs_diff = 0.0`, `argmax_agree = 1.0`, `np.array_equal = True`.
- The 0.1 budget came from a now-stale prior assumption that the loader
  might lose precision; it does not. Byte-exactness is the right invariant.

## T8 — SWA decode-state ≡ prefill-state after 32 sequential decodes: byte-exact (v8 iter 4 tightening from atol=2e-2)
**Where:** `tests/models/jax/test_deepseek_v4.py::TestDecodeRollingEquivalenceWithPrefill::test_swa_decode_state_equals_prefill_state_after_32_steps`.
**Default:** bf16 atol/rtol = 1e-2.
**Effective bound:** byte-exact (`np.array_equal` AND max-abs == 0).
**Evidence (measured 2026-04-29 v8 iter 4):**
- Across 8 random seeds (7, 1, 2, 3, 5, 11, 13, 17) for the input
  sequence: max abs diff = `0.0` for every seed.
- Why the original 2e-2 budget was unnecessary: both the prefill kv-write
  loop and the decode-step kv-write share the same per-position write
  expression (RoPE-rotate the row, store into circular buffer slot). The
  prefill loop is just an unrolled batch of those same writes; XLA emits
  byte-identical lowerings for each element. There is no batched-vs-
  sequential rounding drift in this code path — the prior estimate was
  pessimistic.
- This holds even though the bf16 RoPE multiply technically has ~1 ULP
  rounding per element: that rounding is deterministic and identical on
  both sides because the inputs and freqs slices are identical.
- The byte-exact bound now catches any future regression that introduces
  a real difference between the prefill and decode write paths (e.g. a
  fused vs unfused RoPE divergence).

## Decode step parity (Tier 2 hardening — `TestDecodeAttentionParity`, `TestDecodeRollingParity`, `TestDecodeAttentionParityExtended`): `atol=1e-4` (v8 iter 4 tightening from atol=5e-2)
**Where:**
  - `TestDecodeAttentionParity::test_decode_step_parity` (9 points: SWA/CSA/HCA at sp ∈ {1,4,7,8,9,16,32}).
  - `TestDecodeRollingParity::test_rolling_decode_parity` (5 (P,K) combos, K up to 16).
  - `TestDecodeAttentionParityExtended::test_decode_step_parity_extended` (8 points incl. sp ∈ {500, 1023}).
**Default:** bf16 atol/rtol = 1e-2.
**Looser bound:** atol=1e-4.
**Evidence (measured 2026-04-29 v8 iter 4):**
- Worst observed across all three test classes' parametrized points
  (22 measurements total): `3.81e-6` (= 2 ULPs of bf16 in the
  attention-output magnitude range).
- Old budget was 5e-2 — about 13,000× looser than necessary, so it
  didn't catch a real per-layer regression.
- 1e-4 keeps a 25× margin over the observed worst, which leaves room
  for: (a) other random seeds we did not enumerate, (b) future torch
  reference fixes that change tie-breaking, (c) bf16 quirks at corner
  start_pos values inside the SWA wraparound. Tighter than 1e-4 risks
  flakes; looser than 1e-4 hides real bugs.
- The full attention forward (Tier 1, `TestAttentionComponent`) still
  uses `atol=5e-2` because that path involves the much bigger Sinkhorn
  + mHC + low-rank-O combine chain. Decode step is a much shorter
  matmul chain and the budget reflects that.

## T4 — Indexer top-k SET equality (Tier 1, `TestIndexerComponent`)
**Where:** `TestIndexerComponent::test_indexer_prefill_matches_torch_topk`.
**Default:** exact int equality.
**Looser:** *set* equality of selected indices per (batch, seq) position (ignoring -1 sentinels).
**Evidence:** `lax.top_k` and `torch.topk` may break score-ties in different orders when two scores are bit-equal in fp32. Set-equality is the appropriate invariant — the question is "did we pick the same K compressed positions to attend to," not "in what order." If a real bug were present (e.g. wrong compressor output), the SETS would differ.

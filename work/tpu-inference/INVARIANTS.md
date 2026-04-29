# Invariants — DeepSeek V4 implementation

Things that must remain true for the math to work. When something breaks, check these first.

## Architecture
- I1. `head_dim = nope_head_dim + rope_head_dim`. RoPE is applied only to the last `rope_head_dim` slots of each q/k vector.
- I2. `num_key_value_heads = 1`. K and V are shared across all `n_heads` query heads (true MQA).
- I3. `n_heads * head_dim` must be divisible by `o_groups`. The grouped O projection treats heads as `(o_groups, n_heads*head_dim/o_groups)`.
- I4. Sliding window size `win` divides `kv_cache_size = win + max_seq_len/ratio` cleanly when wraparound writes happen at decode time.
- I5. `compress_ratio` is one of `{0, 4, 128}`. `0` disables compressor and indexer entirely. `4` enables both (CSA). `128` enables compressor only (HCA).
- I6. The compressor's `coff = 2 if ratio==4 else 1` toggles overlap mode. With overlap, the compressor produces TWO compressed positions per ratio-step (overlap window + non-overlap window).

## mHC (manifold-constrained Hyper-Connections)
- I7. Hidden state is `[B, S, hc_mult, D]` everywhere inside the block stack. Embedding produces `[B, S, D]` then `unsqueeze(2).repeat(1,1,hc_mult,1)`.
- I8. Each `hc_pre` reduces hc dim → 1 (returns `[B, S, D]`); each `hc_post` rebuilds hc dim ≥ 1 (returns `[B, S, hc_mult, D]`).
- I9. Sinkhorn input `mixes: [B*S, mix_hc]` where `mix_hc = (2 + hc_mult) * hc_mult`. The first `hc_mult` slots (after `hc_scale[0]` + `hc_base[0:hc]`) are sigmoid-gated → `pre`. The next `hc_mult` slots → `post` (with factor 2). The remaining `hc_mult * hc_mult` slots → `comb` (after Sinkhorn iterations).
- I10. Sinkhorn iterations are: softmax along last axis + eps, divide by col-sum + eps, then `iters-1` more rounds of (row-norm, col-norm). `comb` is approximately doubly-stochastic at end.

## RoPE
- I11. `apply_rotary_emb` uses YaRN-augmented frequencies when `original_seq_len > 0`, else plain. It is applied IN PLACE to the last `rope_head_dim` slots of x's last axis, where slots are interpreted as complex pairs (`unflatten(-1, (-1, 2))`).
- I12. The inverse RoPE in `Attention.forward` (after sparse_attn) is `apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)`, i.e. multiplied by complex conjugate. This UNDOES the rotation that was applied to q in the rope dims.
- I13. Pure-SWA layers use `original_seq_len=0, rope_theta=args.rope_theta`. CSA/HCA layers use `original_seq_len=args.original_seq_len, rope_theta=args.compress_rope_theta`.

## Attention
- I14. Sparse attention uses learned `attn_sink: [n_heads]` as an extra logit. Numerically: `softmax = exp(s - max) / (sum(exp(s' - max)) + exp(attn_sink - max))`. (See sparse_attn_kernel: the `sum_exp[i] += exp(attn_sink[i] - scores_max[i])` line.)
- I15. The KV that flows into sparse_attn is BF16. Quantization (act_quant, fp4_act_quant) inplace=True simulates QAT noise but for our tests it is a no-op (see DECISIONS D2).
- I16. The KV passed to sparse_attn is the concatenation `[kv (current step) ; kv_cache (compressed/window)]` for prefill, and for decode it is just `kv_cache[:bsz]`. The `topk_idxs` index into this concatenation.
- I17. Indexer and Compressor share `kv_cache` storage (set lazily inside Attention.__init__).

## MoE Gate
- I18. Hash-routing layers use `tid2eid: [vocab_size, n_activated_experts]` int32 lookup keyed by INPUT TOKEN ID, not by hidden state. Routing weights are still derived from the score path.
- I19. Non-hash layers: `original_scores = sqrt(softplus(scores_raw))`. Bias is added BEFORE topk but routing weights are gathered from `original_scores` (pre-bias) and then renormalized to sum=1, then scaled by `route_scale`.
- I20. There is NO group-routing (no `n_group, topk_group` like V3). Topk is over all `n_routed_experts`.

## Tiny config tradeoffs (testing)
- I21. Tiny config shrinks `head_dim`, `index_head_dim`, `q_lora_rank`, `o_lora_rank`, `vocab_size`, `n_routed_experts`, `dim`, `intermediate_size`. None of these affect *which code path* runs; they only affect arithmetic intensity. The `compress_ratios` pattern, `hc_mult`, and presence/absence of indexer are preserved exactly.
- I22. Random weights for tiny tests are seeded identically on the PyTorch reference and the JAX implementation.

## Decode (v2)
- I23. **Compressor state shape:** `kv_state, score_state` are `[B, coff*ratio, coff*head_dim]` with `coff = 2 if ratio==4 else 1`. `score_state` is initialized to `-inf` so unfilled positions softmax-out to 0.
- I24. **Compressor compression event:** the compressor emits a new compressed kv when `(start_pos+1) % ratio == 0`. Otherwise it just updates state. This is statically known from start_pos so JIT can branch on it.
- I25. **Overlap-mode buffer shift:** when `ratio==4` and a compression event happens, `kv_state[:, :ratio]` is set to `kv_state[:, ratio:]`, "scrolling" the buffer up by one ratio group.
- I26. **Attention kv_cache layout:** `[B, win + (max_seq_len/ratio if ratio else 0), head_dim]`. Slots `[0, win)` are the SWA circular buffer; slots `[win, win+max/ratio)` are the compressor cache. Decoding writes to `start_pos % win` (SWA) and `win + start_pos // ratio` (compressor, when compressed).
- I27. **Indexer kv_cache** is a SEPARATE buffer from attention's, shape `[B, max/ratio, index_head_dim]`. The indexer's compressor state is also separate from attention's compressor state (different params, different head_dim).
- I28. **Topk K is statically bounded** at decode time: `K = min(index_topk, (start_pos+1)/ratio)` for the indexer; window topk is exactly `window_size`.

## Quantization (v2)
- I29. **FP8 dequant:** weight is `float8_e4m3fn` with shape `[O, I]`; scale is `float8_e8m0fnu` with shape `[O/block, I/block]`. Block size from `quantization_config.weight_block_size` (32 in tiny, 128 in real). Dequant: `weight.float() * 2 ** (scale.uint8() - 127)` per-block.
- I30. **FP4 dequant:** weight is `int8` with shape `[O, I/2]` (logical FP4 shape `[O, I]`). Each byte's low nibble at logical index `2k`, high nibble at `2k+1`. FP4_TABLE[low/high] gives the fp32 value. Block size from `fp4_block_size` (8 in tiny, 32 in real). Per-block scale is `float8_e8m0fnu` shape `[O, I/fp4_block]`.
- I31. **Bit-exact loader:** `tiny_v4_quant` dequantized via I29 + I30 produces bf16 tensors that are *byte-identical* to `tiny_v4_groundtruth` (which was pre-dequantized using DeepSeek's `convert.py` recipe). Verified across all 355 tensors.

## vLLM-runtime integration (v6)
- I32. **vLLM mis-classifies V4 as MLA** in `model_arch_config_convertor.is_deepseek_mla` because V4's HF config has `compress_ratios` + `head_dim`. V4 is NOT MLA — it has CSA/HCA/SWA hybrid sparse attention with no `kv_lora_rank`. The `KVCacheManager` mitigates by detecting `model_type == "deepseek_v4"` at __init__ and forcing `self.use_mla = False`, so the kv_cache spec branch reads `head_dim` / `num_key_value_heads` (which V4 has). V3 is unaffected — it falls through the original branch.
- I33. **vLLM `VllmConfig` pydantic gate (B4)** still requires `NEW_MODEL_DESIGN=1` env + `--additional_config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}'` because V4 inherits the MLA-name pattern. enable_dp_attention is the right setting for V4 anyway (sliding-window attention is not improved by intra-attention TP), so the workaround is also the production-correct setting.
- I34. **vLLM kv_caches passthrough.** V4's per-layer compressor + indexer state lives in the model's params dataclass tree (specifically `Compressor.kv_state / score_state`, `Indexer.kv_state / score_state / kv_cache`), NOT in vllm's `kv_caches: List[jax.Array]`. The vllm-runtime `__call__` returns `kv_caches` UNCHANGED — vllm's per-layer cache is a placeholder. Decode-time state must be threaded through self.params_v.value when implemented for the multi-step decode path; at present `__call__` is prefill-only (single sequence, all positions [0, T)).
- I35. **load_weights → load_weights_from_dir.** vllm calls `model.load_weights(rng)` after `nnx.eval_shape`. Our `load_weights` reads `self.vllm_config.model_config.model`; if it's a local checkpoint dir with `config.json`, it dispatches to `load_weights_from_dir` (uses the W4 deepseek_v4_loader). Falls back to zero-fill if the path is non-local-loadable.

## Quantization spec correctness (v8 iter 7)
- I36. **FP8 dequant is byte-equal to a bit-level reference** on real V4-Flash tensors. Independently decoding e4m3fn (sign+exp+mant arithmetic) and e8m0fnu (`2^(byte-127)`) in numpy and broadcasting the scale block via `np.kron` produces the same bf16 bytes as the loader's `weight.float() * scale.float().repeat_interleave(...)` path. Verified on `layers.0.attn.{wq_a, wkv}`. A regression here means the loader has diverged from the spec, not from a tolerance.
- I37. **FP4 dequant is byte-equal to a sign-magnitude reference** on real V4-Flash expert tensors. The loader's 16-entry `_FP4_TABLE_T` lookup and an independently-derived sign-magnitude decomposition (8-entry magnitude table + sign bit + conditional negate) produce the same bf16 bytes for `layers.{2,0}.ffn.experts.0.{w1, w2}`. The two code paths share only the canonical magnitude set `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`, so byte-equality validates nibble ordering, sign-bit position, and scale-block axis.
- I38. **DeepSeek's FP4 codebook collapses negative-zero to positive-zero.** Nibble 8 (sign=1, mag_idx=0) decodes to `+0.0`, not `-0.0`. The reference dequant must canonicalize -0 → +0 to match the loader's table-lookup output byte-equally; this affects every nibble-8 byte (~0.6% of real expert weights) and nothing else.

## Quantization byte-domain coverage (v8 iter 8)
- I39. **e4m3fn 256-byte cast: numpy reference and torch's `.float()` cast agree bit-for-bit** on every one of the 256 possible input bytes. Non-NaN values match by uint32 reinterpretation; NaN positions match exactly the FN spec's `{0x7F, 0xFF}`. This locks our independent decoder against torch's native cast at the byte boundary — any future torch behavior change (e.g., redefining FN-mode handling of 0xFE / 0x7E, or a subnormal rounding regression) trips this test before it propagates into Tier 4b real-data tests.
- I40. **e8m0fnu 256-byte cast: numpy reference and torch's `.float()` cast agree bit-for-bit** on all 256 possible input bytes. Non-NaN values match exactly; NaN position matches the spec's lone `{0xFF}`. The byte=0 → 2^-127 subnormal-edge case and byte=254 → 2^127 max-finite case are exercised explicitly.
- I41. **FP8 dequant byte-equal coverage spans 6 distinct real V4-Flash tensors** (iter 7: 2; iter 8: +4) — `layers.0.attn.{wq_a, wkv}`, `layers.20.attn.wq_b`, `layers.10.attn.wo_a`, `layers.5.attn.wo_b`, `layers.40.ffn.shared_experts.w1`. Coverage spans 5 distinct projections, 4 distinct shapes (1024×4096, 512×4096, 32768×1024 (out>>in), 8192×4096, 4096×8192 (in>out), 2048×4096), 4 distinct shards, and layers in {0, 5, 10, 20, 40}. The shared-experts case is the only FP8 path outside attn — it covers the dense FFN.
- I42. **FP4 dequant byte-equal coverage spans 4 distinct real V4-Flash expert tensors** (iter 7: 2; iter 8: +2) — `layers.{2,0}.ffn.experts.0.{w1, w2}`, `layers.30.ffn.experts.128.w1`, `layers.10.ffn.experts.50.w3`. Coverage spans all three SwiGLU projections (w1/w2/w3), expert ids in {0, 50, 128}, and layers {0, 2, 10, 30}.

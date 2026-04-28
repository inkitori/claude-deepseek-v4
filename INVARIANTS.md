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

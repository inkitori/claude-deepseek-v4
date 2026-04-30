# V3 → V4 architectural diff

Built from DeepSeek's `inference/model.py` reference (vendored at
`tests/models/jax/_deepseek_v4_reference/model.py`) and the V4-Pro /
V4-Flash `config.json` files, compared against
`tpu_inference/models/jax/deepseek_v3.py`. See README.md in the V4
HuggingFace repo for the marketing version.

## Headline changes
1. **Multi-head Latent Attention (MLA) → CSA / HCA / SWA hybrid.** Each layer's attention is one of three flavors selected via `compress_ratios[layer_id]`:
   - `0` → pure sliding-window attention (SWA), window 128.
   - `4` → Compressed Sparse Attention (CSA): SWA + DSA-indexed top-k over learned-pooled compressed KV (ratio 4, with overlap).
   - `128` → Heavily Compressed Attention (HCA): SWA + dense over a much-more-compressed KV (ratio 128, no DSA index, no overlap).
2. **Residual connections → manifold-constrained Hyper-Connections (mHC).** Hidden state is `[b, s, hc_mult, d]` with `hc_mult=4` parallel residual streams. Each block consumes them via `hc_pre` (Sinkhorn-mixed weighted sum into a single stream) and emits via `hc_post` (rebuilds the `hc_mult` streams). The same applies to the LM head and to the MTP block.
3. **MoE gate is "sqrtsoftplus" with optional hash routing.** No more sigmoid + auxiliary-loss-free `noaux_tc` topk over expert groups. The first `n_hash_layers` (default 3) MoE layers route via a fixed `tid2eid[input_id]` lookup; the remaining layers route via `topk(softplus(scores).sqrt() + bias)`. Group structure (`n_group`, `topk_group` in V3) is gone.
4. **Multi-Token Prediction (MTP) heads.** `n_nextn_predict_layers=1` extra block sits after the main stack, sharing the embedding and LM head, used for speculative-style multi-token output.
5. **One KV head, learnable attention sink.** `num_key_value_heads=1` (true MQA-like). Each layer has a learnable `attn_sink: [n_heads]` parameter that contributes a non-data-dependent term to the softmax denominator.
6. **Grouped low-rank output projection.** `wo = wo_a (per-group, head_dim/groups → o_lora_rank) ; wo_b (groups*o_lora_rank → dim)` with `o_groups=8 or 16`. V3's `wo` was a single dense linear.
7. **YaRN RoPE only inside compressed attention.** SWA-only layers use base RoPE (`rope_theta=10000`, no YaRN). CSA/HCA layers use `compress_rope_theta=160000` with YaRN scaling.
8. **FP4 expert weights.** `expert_dtype=fp4` in V4 configs (V3 was BF16). Dequantize at load time.
9. **swiglu_limit=10.0** clamps SwiGLU activations (V3 was unbounded).

## Per-component diff

### Embedding
- V3: `JaxEmbed`, replicated.
- V4: same shape `[vocab, dim]`. Output is then `unsqueeze(2).repeat(1, 1, hc_mult, 1)` to enter the HC space.

### Attention
- V3 `DeepseekV3MLA`:
  - `wq_a (dim → q_lora_rank)`, `q_norm`, `wq_b (q_lora_rank → n_heads * (qk_nope_head_dim + qk_rope_head_dim))`
  - `wkv_a (dim → kv_lora_rank + qk_rope_head_dim)`, `kv_norm`, `wkv_b (kv_lora_rank → n_heads * (qk_nope_head_dim + v_head_dim))`
  - `o (n_heads*v_head_dim → dim)`
  - `mla_attention` kernel
- V4 `Attention`:
  - `wq_a (dim → q_lora_rank)`, `q_norm`, `wq_b (q_lora_rank → n_heads * head_dim)` where `head_dim = 512` and `head_dim = nope + rope`, `rope_head_dim = 64`.
  - `wkv (dim → head_dim)` — single shared K/V vector across all heads (`n_kv_heads = 1`). RMSNorm on the full kv, then RoPE applied to last 64 dims.
  - `wo_a (n_heads*head_dim/o_groups → o_groups*o_lora_rank, BF16)` and `wo_b (o_groups*o_lora_rank → dim)` for grouped low-rank O projection.
  - `attn_sink: [n_heads]` learnable.
  - Sliding window of 128 entries kept in `kv_cache[:, :win]`. If `compress_ratio>0`, an additional compressed buffer in `kv_cache[:, win:]`.
  - `topk_idxs = concat(get_window_topk_idxs(win), DSA top-k or get_compress_topk_idxs)`. Then `sparse_attn(q, kv, attn_sink, topk_idxs)`.
  - Inverse RoPE applied to last 64 dims of `o` (`apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)`).
- New sub-modules:
  - `Compressor(args, compress_ratio, head_dim, rotate=False)` — learned-gated pooling over `compress_ratio` consecutive tokens, with overlap when ratio==4. Per-position-in-window APE bias. Handles both prefill and decode paths.
  - `Indexer(args, compress_ratio=4)` — its own Compressor (`rotate=True`) plus a query/key projection (`wq_b`, `weights_proj`) that produces top-k compressed positions.
- See section "Attention parameter inventory" below for the full param tree.

### MoE Gate
- V3 used grouped sigmoid + topk_group + topk + bias (auxiliary-loss-free). All gone.
- V4 `Gate`:
  - For first `n_hash_layers` MoE layers: a fixed `tid2eid: [vocab_size, n_activated_experts]` int32 lookup keyed by token id. No score-based topk.
  - For remaining layers: `scores = softplus(linear(x, weight)).sqrt()`. Add `bias` for selection only (not for routing weights). `topk(scores + bias, n_activated_experts)`. Routing weights are gathered from `original_scores` (pre-bias) then renormalized to sum=1, then scaled by `route_scale`.
- The `score_func` config is `"sqrtsoftplus"` for V4.
- All MoE layers are MoE; there is no `first_k_dense_replace` dense prefix in V4 (V3 had `first_k_dense_replace=3`).

### MoE Expert
- V3: `silu(w1(x)) * w3(x); w2(...)`.
- V4: same, plus optional `swiglu_limit` clamp on `gate` and `up`. With `swiglu_limit=10.0` in real V4 and `0` in tiny-config, this is just a clamp.

### Block (mHC)
- V3: `x = x + attn(rmsnorm(x)); x = x + ffn(rmsnorm(x))`.
- V4: hidden state is `[b, s, hc_mult, d]`. For each sub-step:
  ```
  residual = x  # [b, s, hc, d]
  pre, post, comb = sinkhorn_mix(rms(flatten(x)), hc_fn, hc_scale, hc_base)
  x = sum_hc(pre[..., None] * residual)         # [b, s, d]
  x = sub_norm(x)
  x = sub(x)                                    # attn or ffn output
  x = post[..., None] * x[..., None, :] + sum_hc(comb[..., None] * residual)
  # x is now [b, s, hc, d] again
  ```
- The Sinkhorn iteration runs ~20 times to make `comb` doubly-stochastic-ish. See `hc_split_sinkhorn` reference; we re-implement in JAX.

### LM Head
- V3: replicated linear, fp32 weight.
- V4: `ParallelHead` with `hc_head` mixer (sigmoid-gated weighted sum over hc dim, no Sinkhorn) before applying RMSNorm and final linear. The output of `forward` returns logits for `x[:, -1]` only (last position), but for our equivalence tests we expose all-position logits via a small wrapper.

### MTP block
- A `MTPBlock(layer_id, args)` subclass of `Block` with extra:
  - `e_proj`, `h_proj` linears, `enorm`, `hnorm` RMS norms.
  - `norm` final RMSNorm (separate from the parent stack's).
  - `hc_head_fn`, `hc_head_base`, `hc_head_scale` for an additional HC mix at head-time.
- Forward: `e = embed(input_ids); h = h_proj(hnorm(x)) + e_proj(enorm(e)).unsqueeze(2); out = super().forward(h, ...); logits = head(out, hc_head_fn, ...)`.

## Attention parameter inventory (V4)

For a layer with `compress_ratio>0`:
- `attn_sink` — `[n_heads]` fp32
- `wq_a.weight` — `[q_lora_rank, dim]`
- `q_norm.weight` — `[q_lora_rank]` fp32
- `wq_b.weight` — `[n_heads*head_dim, q_lora_rank]`
- `wkv.weight` — `[head_dim, dim]`
- `kv_norm.weight` — `[head_dim]` fp32
- `wo_a.weight` — `[o_groups*o_lora_rank, n_heads*head_dim/o_groups]` bf16
- `wo_b.weight` — `[dim, o_groups*o_lora_rank]`
- `compressor.ape` — `[ratio, coff*head_dim]` fp32, `coff=2 if ratio==4 else 1`
- `compressor.wkv.weight` — `[coff*head_dim, dim]` fp32
- `compressor.wgate.weight` — `[coff*head_dim, dim]` fp32
- `compressor.norm.weight` — `[head_dim]` fp32
- If `ratio==4`:
  - `indexer.wq_b.weight` — `[index_n_heads * index_head_dim, q_lora_rank]`
  - `indexer.weights_proj.weight` — `[index_n_heads, dim]` bf16
  - `indexer.compressor.*` — same shape skeleton, with `head_dim=index_head_dim=128`, `rotate=True`.

For `compress_ratio==0`: only the `attn_sink, wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b` are present (no compressor, no indexer).

## Block parameter inventory (V4)
- All Attention params above (varies by compress_ratio).
- `attn_norm.weight`, `ffn_norm.weight`
- `ffn` (MoE):
  - `gate.weight: [n_routed_experts, dim]` fp32
  - `gate.bias: [n_routed_experts]` fp32 (or `tid2eid: [vocab_size, top_k]` int32 for hash layers)
  - For each routed expert `i`: `experts[i].w1, w2, w3` — each `[hidden_or_out, in]`
  - `shared_experts.w1, w2, w3` — same shape skeleton
- mHC params:
  - `hc_attn_fn: [(2+hc_mult)*hc_mult, hc_mult*dim]` fp32
  - `hc_ffn_fn: [(2+hc_mult)*hc_mult, hc_mult*dim]` fp32
  - `hc_attn_base, hc_ffn_base: [(2+hc_mult)*hc_mult]` fp32
  - `hc_attn_scale, hc_ffn_scale: [3]` fp32

## Top-level (Transformer) params
- `embed.weight`, `head.weight` (separate; no tie_word_embeddings)
- `norm.weight` (final RMSNorm before head)
- `hc_head_fn, hc_head_base, hc_head_scale` for the head's HC mixer (one mixer for whole model)
- `mtp[0]` block params

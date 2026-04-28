# Blockers

Items where the current scope was deemed infeasible to complete in this session
and were deferred. The user should look at these first.

## B1 — W2 (ragged_paged_attention integration) — DEFERRED

**Status:** Not attempted.

**Why deferred:** V4 attention is *not* a drop-in replacement for V3-style MLA
in the `ragged_paged_attention` interface. Real V4 attention requires:

  * **Sliding-window write+read** of raw bf16 KV (not paged-KV). The window is
    a circular buffer of fixed length 128.
  * **Compressed-position cache** at indices `[win, win + max_seq_len/ratio)`
    that is written by the per-layer `Compressor` only when
    `(start_pos+1) % ratio == 0`.
  * **Per-layer compressor kv_state and score_state** of shape
    `[B, coff*ratio, coff*head_dim]` (`coff = 2 if ratio==4 else 1`). This is
    *separate* from the kv_cache and must persist across decode steps.
  * **Per-layer indexer state** (only on `ratio==4` layers): its own
    `kv_state`, `score_state`, and `kv_cache`.
  * **Sparse top-K attention** over the union of window + compressed slots,
    with a learnable per-head sink term.

`tpu_inference.kernels.ragged_paged_attention.v3` assumes a flat
`[total_pages, page_size, kv_heads_x2, head_dim]` KV layout with a single
`[max_num_seqs * pages_per_seq]` index buffer. It supports causal masking and
optional fixed-size sliding windows, but it does **not** model:

  - per-token sparse top-K selection,
  - learnable attention sinks in the softmax denominator,
  - the dual-buffer (window + compressed) layout V4 needs.

Adapting `ragged_paged_attention.v3` to V4 would require either:
  (a) writing a new Pallas kernel from scratch that fuses
      sparse_attn(q, [SWA-window || compressed], topk_idxs) into a paged KV
      layout, or
  (b) keeping the current dense/materialized `sparse_attn` and threading the
      additional compressor/indexer state through vLLM's per-layer KV-cache
      slot.

Both are substantial engineering efforts beyond what's feasible for a single
overnight session that also has to deliver decode + dequant correctness. The
existing functional `attention_prefill` and `attention_decode_step`
implementations are mathematically correct and serve as the contract that
any future paged-KV adapter must match.

**Recommended next step for the user:** decide whether to (i) write a V4-
specific Pallas kernel, or (ii) extend vLLM's kv_cache schema so V4 can
register a custom multi-tensor cache (kv_cache + compressor_state +
indexer_state). Option (ii) is structurally simpler.

## B2 — W3 (DeepseekV4ForCausalLM.__call__) — DEFERRED

**Status:** The class exists at `tpu_inference/models/jax/deepseek_v4.py:777`
and is registered in the model registry. Its `__call__` still raises
`NotImplementedError`.

**Why deferred:** The vLLM-runtime `__call__` signature is

```python
def __call__(self, kv_caches: List[jax.Array], input_ids,
             attention_metadata, ...) -> (kv_caches, hidden, [])
```

Implementing this body requires:
  1. **Owning weights as `nnx.Variable`**s (not a plain dataclass tree). V3's
     `DeepseekV3` is a real `nnx.Module` and uses `JaxRmsNorm`, `JaxEinsum`,
     etc. so that `JaxAutoWeightsLoader` can populate it. Wrapping V4's
     functional core in nnx is a non-trivial port — every parameter (~256
     leaves in the tiny config, ~70k in real V4-Flash) must be registered as
     an `nnx.Param` with sharding annotations, and every functional helper
     (`block_forward`, `attention_prefill`, `moe_forward`, etc.) must read
     params via the module hierarchy.
  2. **Decoding the `attention_metadata` shape** — `cu_q_lens`, `kv_lens`,
     `block_tables`, etc. — and dispatching to either `attention_prefill` or
     `attention_decode_step` per layer. The decode-step path additionally
     needs the `compressor_kv_state`, `compressor_score_state`,
     `indexer_kv_state`, `indexer_score_state`, `indexer_kv_cache` fields,
     which vLLM's per-layer `kv_caches[i]` does not yet provide (B1).
  3. **Logit head** via `compute_logits` interface (V3 uses `JaxEinsum` with
     a `JaxAutoWeightsLoader`-compatible name).

This is structurally a port of the V3 `DeepseekV3ForCausalLM` class to V4,
while keeping the math exactly as our functional core implements it.

**Recommended next step for the user:** treat this as a separate PR. Start by
copying `DeepseekV3ForCausalLM` and replacing each MLA / dense-FFN call with
the corresponding functional V4 call, then add the V4-specific compressor /
indexer state plumbing.

## B3 — T5 (vLLM serve smoke test) — BLOCKED ON B1 + B2

**Status:** Not attempted.

**Why blocked:** Tier 5 sends two `/v1/completions` requests through
`vllm serve /mnt/scratch/tiny_v4_bf16` and asserts both return non-empty,
deterministic-with-seed outputs. This requires the full vLLM scheduling
pipeline: tokenize → schedule into batches → call
`DeepseekV4ForCausalLM.__call__` (B2) → write to paged kv_cache (B1) →
generate next token → repeat. With B1 and B2 unimplemented, the serve will
either (a) fail at registry time, (b) raise `NotImplementedError` from
`__call__`, or (c) hang.

**Confirmed pre-flight:** the fixtures `/mnt/scratch/tiny_v4_bf16` (and
`tiny_v4_quant`, `tiny_v4_groundtruth`) exist and are loadable by our W4
loader. So once B1+B2 land, the only remaining T5 work is the curl
round-trip itself.

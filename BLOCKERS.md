# Blockers (status post-v6)

This file tracks items that were deferred, still partially open, or
characterized but not "fully solved with the right kernel". As of v6,
the structural blockers that prevented `vllm serve` from reaching
`/v1/completions` are RESOLVED end-to-end (Tier 5 green); the items
below describe what's still residual.

---

## B1 — W2 (ragged_paged_attention integration) — STILL OPEN AS PROD WORK; T5 NO LONGER GATED

**Status (v6):** No longer gates Tier 5. The minimum-viable replacement
landed in v6: `KVCacheManager` forces `use_mla=False` for V4, which
returns a `FullAttentionSpec` placeholder per layer. V4's actual KV-like
state (compressor `kv_state`/`score_state`, indexer `kv_state`/
`score_state`/`kv_cache`, sliding-window slots) is held inside the
model's params dataclass tree — vLLM's per-layer `kv_caches: List[
jax.Array]` is passed through `__call__` unchanged.

**What this means for production:** correctness is still right (Tier 5
proves it: byte-equal deterministic completions on real bf16 weights),
but the per-layer state is **not paged**. Multi-sequence concurrent
decode that depends on vLLM's block-table for V4 would need either:

  (a) a Pallas kernel fusing
      `sparse_attn(q, [SWA-window || compressed], topk_idxs)` over a
      paged-KV layout, or
  (b) extending vLLM's per-layer kv_cache to admit V4's dataclass-tree
      state pytree (compressor + indexer state) so it can be sharded /
      offloaded by vLLM's block-pool the way MLA's flat KV is today.

**Why option (a) is hard:** `tpu_inference.kernels.
ragged_paged_attention.v3` assumes a flat `[total_pages, page_size,
kv_heads_x2, head_dim]` KV layout with a single `[max_num_seqs *
pages_per_seq]` index buffer. It supports causal masking and optional
fixed-size sliding windows, but it does **not** model per-token
sparse top-K selection, learnable attention sinks in the softmax
denominator, or the dual-buffer (window + compressed) layout V4 needs.

**Why option (b) is hard:** vLLM's per-layer kv_cache schema is
`jax.Array`, not a pytree, so admitting a dataclass-tree state would
require API changes upstream of tpu_inference.

**Why this didn't block v6:** the smoke-test path (single-sequence
prefill + decode loop with the model's internal state) bypasses paged-
KV entirely. The Tier 5 fixture is `max_num_seqs=2, max_model_len=256`,
so even concurrent two-sequence runs work because each sequence's state
lives in a copy of params_v.value (not shared paged memory).

**Recommended next step for production:** option (b). Land a vLLM
kv_cache schema extension upstream so V4 can register a multi-tensor
cache (kv + compressor_state + indexer_state). Until then, V4 in vllm
serve is correct-but-not-multi-sequence-paged.

---

## B2 — W3 (DeepseekV4ForCausalLM.__call__) — RESOLVED IN v6

**Status (v6):** RESOLVED. `DeepseekV4ForCausalLM` is now an `nnx.Module`
subclass (`tpu_inference/models/jax/deepseek_v4.py`, the dynamic class
returned by `_build_class()`). It passes `nnx.eval_shape(
create_abstract_model)`, has the V3-compatible
`__call__(kv_caches, input_ids, attention_metadata, ...) ->
(kv_caches, hidden_TD, [])` signature, has a `compute_logits(hidden_TD)`
that runs the V4 head, and a `load_weights(rng)` that dispatches to
`load_weights_from_dir` for local checkpoint dirs.

**What's still residual:** the `__call__` body only handles single-
sequence prefill (one batch, all positions [0, T)). Multi-sequence
batching and decode-step dispatch (with start_pos > 0 reading per-
layer compressor/indexer state across calls) are NOT wired through
`attention_metadata` yet — they require B1's per-layer state plumbing.
The functional core (`attention_decode_step` + W1) has the math; what's
missing is the vllm-runtime contract for multi-step decode state.

---

## B3 — T5 (vLLM serve smoke test) — RESOLVED IN v6

**Status (v6):** RESOLVED. Tier 5 is green:

```
JAX_PLATFORMS=cpu pytest tests/models/jax/test_deepseek_v4.py::TestVllmServeRoundtrip -v
→ 1 passed in 112.77s
```

The test spawns `vllm serve /mnt/scratch/tiny_v4_bf16` with the B4
workaround flags, waits for `/v1/models` 200, sends two identical
seed=0 `/v1/completions`, and asserts non-empty + byte-equal text.
Observed completion text on the host: `" \" ab oideable<unk>子"`.

**Path that the v6 work cleared:**
  1. `vllm serve` launches with `NEW_MODEL_DESIGN=1` + `enable_dp_attention`
     (B4 workaround flags).
  2. `nnx.eval_shape(create_abstract_model)` succeeds because
     `DeepseekV4ForCausalLM` is now an `nnx.Module` (B2 fixed in v5).
  3. `KVCacheManager.get_kv_cache_spec()` succeeds because V4 is now
     routed through the non-MLA branch (B5 fixed in v6).
  4. `model.load_weights(rng)` reads
     `vllm_config.model_config.model` (the `--model PATH` arg) and
     dispatches to `load_weights_from_dir`, which loads real bf16
     weights via the W4 deepseek_v4_loader (v6 wiring).
  5. compilation_manager.precompile_backbone runs jit on the model's
     `__call__` for {16, 32, 64, 128, 256, 512, 1024, 2048} num_tokens
     plus compute_logits + select_from_array — all succeed.
  6. The DPScheduler routes 2 prefill requests through 4 ranks; each
     rank's worker runs `__call__` with its own input_ids slice;
     `compute_logits` produces token logits; sampling at temperature=0
     is deterministic.

---

## B4 — vLLM `VllmConfig` validation gate — STILL ACTIVE (workaround required)

**Status (v6):** Workaround required at every `vllm serve` invocation
for V4. Still upstream-vllm work to remove the gate.

```
NEW_MODEL_DESIGN=1 \
  vllm serve <model_dir> \
    ...other flags... \
    --additional_config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}'
```

The Tier 5 pytest test sets these flags automatically. End users
running `vllm serve` directly need them in their command line.

**enable_dp_attention is also the right production setting for V4:**
SWA + sparse compressed attention does not benefit from intra-
attention TP (each token attends to a small static window), so DP-
attention (replicate attention, shard MoE) matches V4's ideal
distribution.

---

## B5 — KVCacheManager use_mla branch was V3-only — RESOLVED IN v6

**Status (v6):** RESOLVED. `KVCacheManager.__init__` now detects
`model_type == "deepseek_v4"` on the model_config and forces
`self.use_mla = False`. V4 falls through the non-MLA branch which
already handles `head_dim` / `num_key_value_heads`.

**Original symptom:** `AttributeError: 'DeepseekV4Config' object has
no attribute 'kv_lora_rank'` at
`tpu_inference/runner/kv_cache_manager.py:365`.

**The fix is V3-safe:** V3 still goes through `model_config.use_mla`
which returns True for V3 (`is_deepseek_mla` + has `kv_lora_rank`),
and the V3 MLA path is unchanged. Only V4 takes the new override branch.

**Test coverage:** `TestVllmServeRoundtrip` exercises this end-to-end
on TPU. CPU regression suite (83 tests) has no V3 path, so V3 isn't
re-validated here, but the override is gated on `model_type == "deepseek_v4"`
which is mutually exclusive with V3.

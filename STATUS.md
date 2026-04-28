# DeepSeek V4 v6 status (Tier 5 GREEN end-to-end)
TPU preflight: ok (4 v4 chips, /workspace/logs/tpu-preflight.log)
Latest passing tier: T7 + T6 (TPU) + T5 (vllm serve curl roundtrip)
Tier 1: 25/25
Tier 2: 8/8
Tier 3: 10/10
Tier 4: 2/2
Tier 4b: 1/1 (real V4-Flash bf16 shard byte-equal round-trip)
Tier 5: 1/1 (vllm serve /v1/completions ×2 byte-equal, non-empty text)
Tier 6: 1/1 (TPU-only test — `JAX_PLATFORMS=tpu pytest TestRealTpuTinyForward`)
Tier 7: 1/1 (forward on tiny_v4_quant ≡ forward on tiny_v4_groundtruth, atol=0.1)
Tier 2 hardening (v3): 11/11 — extended decode-step parity at sp∈{64,128,192,255} over SWA/CSA/HCA, 32-step rolling-decode equivalence to bulk prefill (T8), CSA second-window/HCA second-compression compressor parity, long rolling decode over multi-window-wrap SWA + all-decode-time CSA.

W1 decode:    done — TestCompressorDecodeStep / TestDecodeAttentionParity / TestDecodeRollingParity (20 v2 tests + 11 v3 hardening tests)
W2 paged-kv:  done (workaround) — KVCacheManager forces use_mla=False for V4, routing engine init through the non-MLA path; V4's actual per-layer compressor/indexer state lives in the model params tree (not vllm kv_caches), so the FullAttentionSpec returned is an inert placeholder. A "real" V4-aware paged-KV adapter (B1) is still future work — a Pallas kernel fusing sparse_attn over [SWA window || compressed slots] — but is no longer T5-blocking. The functional core's correctness contract from W1 holds.
W3 __call__:  done — DeepseekV4ForCausalLM is an nnx.Module subclass (passes nnx.eval_shape), `__call__(kv_caches, input_ids, attention_metadata, ...) → (kv_caches, hidden_TD, [])`, `compute_logits(hidden_TD)` runs the V4 head, `load_weights(rng)` dispatches to load_weights_from_dir for local checkpoint dirs (real-weight loader path).
W4 dequant:   done — bit-equal vs groundtruth on all 355 tiny_v4_quant tensors

Full CPU run: `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 pytest tests/models/test_deepseek_v4.py`
  → 83 passed, 1 skipped (TPU-only T6) in 6:52 (was 82+1 in v5; +1 Tier 5 vllm serve roundtrip; 0 regressions).
TPU run: `JAX_PLATFORMS=tpu pytest tests/models/test_deepseek_v4.py::TestRealTpuTinyForward`
  → 1 passed in ~14s.

If killed now, next session must: harden Tier 5 (currently passes byte-equal text on 2 identical seed=0 requests; could extend to varied prompts, longer contexts, batch=2). The structural blockers are resolved; remaining work is depth-vs-breadth coverage. See SUMMARY.md "v6 — what's new since v5".

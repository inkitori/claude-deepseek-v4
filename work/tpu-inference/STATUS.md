# DeepSeek V4 v8 status (host-direct on v6e-32; fixtures+mount in place)
TPU preflight: ok (4 v6e chips, logs/tpu-preflight.log)
Host: TPU v6e-32 single-VM (4 local chips). No docker. Real V4-Flash weights mounted via gcsfuse at ~/.cache/huggingface/hub/. Synthetic fixtures regenerated under work/scratch/{tiny_v4_bf16,tiny_v4_quant,tiny_v4_groundtruth}.
Latest passing tier: T7 + T6 (TPU) + T5 (vllm serve curl roundtrip) — v6 baseline; v8 first iter has not yet rerun on this host
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

If killed now, next session must: (1) reconfirm baseline `pytest tests/models/test_deepseek_v4.py -v` on this host now that fixtures + GCS mount are present (expect formerly-skipped Tier 4 shard / 4b / 5 / 6 / 7 / W3-helper tests to switch from skip to pass) — record the new pass/skip count here; (2) attack B1 (multi-seq concurrent decode in `DeepseekV4ForCausalLM.__call__`) — gates Tier 8; (3) execute Tier 8 real-weight `vllm serve deepseek-ai/DeepSeek-V4-Flash` per prompt §W5. v6/v7 numbers above are historical; rewrite this whole STATUS.md per the prompt §"STATUS.md mandate" once you have v8 numbers.

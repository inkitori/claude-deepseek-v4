# DeepSeek V4 v2 status
TPU preflight: ok (4 v4 chips, 10.4s self-check at /workspace/logs/tpu-preflight.log)
Latest passing tier: T7 (and T6 separately on TPU)
Tier 1: 25/25
Tier 2: 8/8
Tier 3: 10/10
Tier 4: 2/2
Tier 4b: 1/1 (real V4-Flash bf16 shard byte-equal round-trip)
Tier 5: blocked — see BLOCKERS.md B1 + B2 + B3
Tier 6: 1/1 (TPU-only test — `JAX_PLATFORMS=tpu pytest TestRealTpuTinyForward`)
Tier 7: 1/1 (forward on tiny_v4_quant ≡ forward on tiny_v4_groundtruth, atol=0.1)

W1 decode:    done — TestCompressorDecodeStep / TestDecodeAttentionParity / TestDecodeRollingParity (20 tests)
W2 paged-kv:  blocked — BLOCKERS B1 (sparse_attn shape doesn't fit ragged_paged_attention.v3)
W3 __call__:  partial — forward_prefill + load_weights_from_dir helpers work; full vllm-runtime __call__ blocked on B2 (nnx.Module port + per-layer kv_cache schema for compressor/indexer state)
W4 dequant:   done — bit-equal vs groundtruth on all 355 tiny_v4_quant tensors

Full CPU run: `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 pytest tests/models/test_deepseek_v4.py`
  → 68 passed, 1 skipped (TPU-only test) in 5:25.
TPU run: `JAX_PLATFORMS=tpu pytest tests/models/test_deepseek_v4.py::TestRealTpuTinyForward`
  → 1 passed in ~22s (compile + 1×16-token prefill on real V4 chip 0).

If killed now, next session must: address W2/W3 to unblock T5. The two
required pieces (per BLOCKERS.md): (a) extend vLLM's per-layer kv_cache
schema to admit V4's compressor/indexer state pytrees, or write a custom
Pallas kernel that fuses sparse_attn over the [SWA window || compressed
slots] layout (B1); (b) port DeepseekV3ForCausalLM's nnx.Module structure
to V4 (B2). With those done, T5's curl round-trip is a ~30-line script.

# DeepSeek V4 v2 status (v3 final pass)
TPU preflight: ok (4 v4 chips, 10.4s self-check at /workspace/logs/tpu-preflight.log)
Latest passing tier: T7 + T6 (TPU); +11 v3 hardening tests pass
Tier 1: 25/25
Tier 2: 8/8
Tier 3: 10/10
Tier 4: 2/2
Tier 4b: 1/1 (real V4-Flash bf16 shard byte-equal round-trip)
Tier 5: blocked — see BLOCKERS.md B1 + B2 + B3 + B4 (failure now *characterized*, not speculative)
Tier 6: 1/1 (TPU-only test — `JAX_PLATFORMS=tpu pytest TestRealTpuTinyForward`)
Tier 7: 1/1 (forward on tiny_v4_quant ≡ forward on tiny_v4_groundtruth, atol=0.1)
Tier 2 hardening (v3): 11/11 — extended decode-step parity at sp∈{64,128,192,255} over SWA/CSA/HCA, 32-step rolling-decode equivalence to bulk prefill (T8), CSA second-window/HCA second-compression compressor parity, long rolling decode over multi-window-wrap SWA + all-decode-time CSA.

W1 decode:    done — TestCompressorDecodeStep / TestDecodeAttentionParity / TestDecodeRollingParity (20 v2 tests + 11 v3 hardening tests)
W2 paged-kv:  blocked — BLOCKERS B1 (sparse_attn shape doesn't fit ragged_paged_attention.v3)
W3 __call__:  blocked — BLOCKERS B2 confirmed via vllm probe: nnx.eval_shape rejects DeepseekV4ForCausalLM as not-a-JAX-type. forward_prefill + load_weights_from_dir helpers work for non-vllm callers; full vllm-runtime __call__ requires nnx.Module port + per-layer kv_cache schema for compressor/indexer state
W4 dequant:   done — bit-equal vs groundtruth on all 355 tiny_v4_quant tensors

Full CPU run: `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 pytest tests/models/test_deepseek_v4.py`
  → 81 passed, 1 skipped (TPU-only test) in 5:39 (was 70 + 1 in v2; +11 hardening, 0 regressions).
TPU run: `JAX_PLATFORMS=tpu pytest tests/models/test_deepseek_v4.py::TestRealTpuTinyForward`
  → 1 passed in ~22s (compile + 1×16-token prefill on real V4 chip 0).

If killed now, next session must: address W2/W3 to unblock T5. The two
required pieces (per BLOCKERS.md B1/B2/B3/B4): (a) extend vLLM's per-layer
kv_cache schema to admit V4's compressor/indexer state pytrees, or write
a custom Pallas kernel that fuses sparse_attn over the
[SWA window || compressed slots] layout (B1); (b) port
DeepseekV3ForCausalLM's nnx.Module structure to V4 (B2 — failure now
characterized via vllm probe at BLOCKERS B3); (c) note that vllm requires
NEW_MODEL_DESIGN=1 + --additional_config enable_dp_attention to reach
the model class at all (B4). With (a)+(b) done, T5's curl round-trip is
a ~30-line script.

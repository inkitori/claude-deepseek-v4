# DeepSeek V4 v2 status
TPU preflight: ok (4 v4 chips, 10.4s self-check)
Latest passing tier: T7 (FP4/FP8 dequant logits parity green)
Tier 1: 25/25
Tier 2: 8/8
Tier 3: 10/10
Tier 4: 2/2
Tier 4b: 1/1 (real V4-Flash bf16 shard byte-equal round-trip)
Tier 5: not started (depends on W2/W3)
Tier 6: not started
Tier 7: 1/1 (forward on tiny_v4_quant ≡ forward on tiny_v4_groundtruth, atol=0.1)

W1 decode:    done — 20 new tests under TestDecodeAttentionParity / TestDecodeRollingParity / TestCompressorDecodeStep
W2 paged-kv:  todo (lower priority — ragged_paged_attention integration requires nnx.Module wrapping)
W3 __call__:  todo (lower priority — depends on W2)
W4 dequant:   done — 1 unit test (TestFp8Dequant: 355 tensors bit-equal)

If killed now, next session must: attempt Tier 6 real-TPU compile of deepseek_v4_forward_prefill on tiny_v4_bf16 (needs JAX_PLATFORMS=tpu, eval_shape + jit().lower().compile() + small forward). Then tackle W3 (nnx.Module __call__) so vllm-serve can dispatch — that unblocks T5.

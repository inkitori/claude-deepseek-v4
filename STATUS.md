# DeepSeek V4 v2 status
TPU preflight: ok (4 v4 chips, 10.4s self-check)
Latest passing tier: T4 (carried from v1: 41 tests passing on CPU)
Tier 1: 25/25
Tier 2: 8/8
Tier 3: 10/10 (V4-Pro full-config compile is in deselected slot)
Tier 4: 2/2
Tier 5: not started
Tier 6: not started
Tier 7: not started
W1 decode:    todo
W2 paged-kv:  todo
W3 __call__:  todo
W4 dequant:   todo

If killed now, next session must: read SUMMARY.md §4-§5, then implement decode-step JAX functions (compressor_decode_step, indexer_decode_step, attention_decode_step) under tpu_inference/layers/jax/attention/deepseek_v4_attention.py, mirroring start_pos>0 branches from tests/models/jax/_deepseek_v4_reference/model.py.

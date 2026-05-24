"""Confirm the peaked-CPU decode-vs-prefill divergence is STRUCTURAL (large
hidden-state error = the real S1 bug), not a benign near-tie float flip."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import sys
sys.path.insert(0, "scripts")
import numpy as np, jax, jax.numpy as jnp
from s1_cpu_repro_v4flash import make_v4_flash_truncated_cfg
from s1_cpu_repro_peaked import make_scaled_params
from tpu_inference.models.jax.deepseek_v4 import (
    make_freqs_cis, transformer_body_forward, head_forward,
    deepseek_v4_run_with_decode_state, v4_layer_packed_sizes_from_cfg)

scale = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
T, N = 8, 12
cfg = make_v4_flash_truncated_cfg(n_layers=4, n_experts=8)
params = make_scaled_params(cfg, scale, seed=0)
max_seq = 4096
swa, comp = make_freqs_cis(cfg, max_seq)
rng = np.random.default_rng(seed=1234)
ids_full = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(1, T+N)), dtype=jnp.int32)

# prefill reference: full-sequence hidden states (the CORRECT path)
h_full = transformer_body_forward(ids_full, params, swa, comp, cfg)  # [1, T+N, dim]

sizes = v4_layer_packed_sizes_from_cfg(cfg, max_seq, batch_size=1)
kv = [jnp.zeros((s,), dtype=jnp.float32) for s in sizes]
kv, _ = deepseek_v4_run_with_decode_state(kv, ids_full[:, :T], params, swa, comp, cfg,
    state_max_seq_len=max_seq, is_decode_step=False, start_pos=jnp.int32(0))

print(f"scale={scale}  per-step: relErr=||h_dec - h_pre||/||h_pre||, argmax match")
for step in range(N):
    pos = T + step
    kv, h_step = deepseek_v4_run_with_decode_state(kv, ids_full[:, pos:pos+1], params,
        swa, comp, cfg, state_max_seq_len=max_seq, is_decode_step=True, start_pos=jnp.int32(pos))
    h_pre = h_full[:, pos:pos+1]               # [1,1,dim]
    rel = float(jnp.linalg.norm(h_step - h_pre) / (jnp.linalg.norm(h_pre) + 1e-9))
    cos = float((h_step.ravel() @ h_pre.ravel()) / (jnp.linalg.norm(h_step)*jnp.linalg.norm(h_pre)+1e-9))
    a_dec = int(jnp.argmax(head_forward(h_step, params.head_w, params.final_norm_w,
        params.hc_head_fn, params.hc_head_scale, params.hc_head_base, cfg.rms_norm_eps, cfg.hc_eps)[0,0]))
    a_pre = int(jnp.argmax(head_forward(h_pre, params.head_w, params.final_norm_w,
        params.hc_head_fn, params.hc_head_scale, params.hc_head_base, cfg.rms_norm_eps, cfg.hc_eps)[0,0]))
    print(f"  step={step:>2} pos={pos:>2}  relErr={rel:7.4f}  cos={cos:7.4f}  "
          f"argmax {'OK ' if a_dec==a_pre else 'XX '} dec={a_dec} pre={a_pre}")

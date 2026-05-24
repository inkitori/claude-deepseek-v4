"""Detect _pack/_unpack shape mismatch: compare each layer's ACTUAL
AttentionDecodeState field shapes (from the real init/decode path) against
the layout used by _unpack_layer_state. Any mismatch => pack/unpack misalign."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import sys
sys.path.insert(0, "scripts")
import numpy as np, jax, jax.numpy as jnp
import tpu_inference.models.jax.deepseek_v4 as m
from s1_cpu_repro_v4flash import make_v4_flash_truncated_cfg
from s1_cpu_repro_peaked import make_scaled_params

cfg = make_v4_flash_truncated_cfg(n_layers=4, n_experts=8)
params = make_scaled_params(cfg, 0.5, seed=0)
max_seq = 4096
swa, comp = m.make_freqs_cis(cfg, max_seq)

orig_pack = m._pack_layer_state
def traced_pack(state, layout):
    for name, shape, dtype in layout:
        arr = getattr(state, name)
        asz = int(np.prod(arr.shape)); lsz = int(np.prod(shape))
        flag = "" if asz == lsz else "  <<< MISMATCH"
        if asz != lsz or tuple(arr.shape) != tuple(shape):
            print(f"    field={name:22s} actual={tuple(arr.shape)} (sz={asz})  layout={tuple(shape)} (sz={lsz}){flag}", flush=True)
    return orig_pack(state, layout)
m._pack_layer_state = traced_pack

sizes = m.v4_layer_packed_sizes_from_cfg(cfg, max_seq, batch_size=1)
kv = [jnp.zeros((s,), dtype=jnp.float32) for s in sizes]
rng = np.random.default_rng(seed=1234)
ids = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(1, 9)), dtype=jnp.int32)
print("=== PREFILL init pack (per layer, only printing fields w/ shape!=layout) ===")
kv, _ = m.deepseek_v4_run_with_decode_state(kv, ids[:, :8], params, swa, comp, cfg,
    state_max_seq_len=max_seq, is_decode_step=False, start_pos=jnp.int32(0))
print("=== DECODE step pack ===")
kv, _ = m.deepseek_v4_run_with_decode_state(kv, ids[:, 8:9], params, swa, comp, cfg,
    state_max_seq_len=max_seq, is_decode_step=True, start_pos=jnp.int32(8))
print("done (no MISMATCH lines above = shapes all agree)")

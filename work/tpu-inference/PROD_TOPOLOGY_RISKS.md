# PROD_TOPOLOGY_RISKS.md (archived)

This file listed risks an early CPU-only dev session couldn't validate
("This dev host does not have working TPU access", "Pallas /
ragged_paged_attention not exercised", FP4/FP8 dequant not yet wired,
etc.). Most of those have been resolved on the current main branch:

* TPU access: working — v6e-32 slice deploys via `./run.sh bootstrap`
  + `./run.sh serve`.
* FP4/FP8 dequant: implemented in `tpu_inference/models/jax/deepseek_v4_loader.py`
  and verified byte-equal vs independent numpy reference on real
  V4-Flash tensors.
* Real-weight load: 35020 tensors, ~4 min on the 32-chip slice.

Still real:

* Dense attention vs sparse Pallas — V4 forward uses `sparse_attn`
  (top-k + sink) implemented in JAX, not the upstream
  `ragged_paged_attention` Pallas kernel. Performance-only; math is
  correct.
* SPMD `Involuntary full rematerialization` warnings during
  `jit_run_model` compile. See [BLOCKERS.md](BLOCKERS.md).

For current state, read [`../../CLAUDE.md`](../../CLAUDE.md).

# Decisions log — DeepSeek V4 implementation

## D1 — Reference oracle is the official DeepSeek inference/model.py, not HuggingFace transformers
**When:** 2026-04-28 09:00 UTC.
**Decision:** Use `/mnt/scratch/v4_pro/inference/model.py` as the ground truth for V4 architectural math. The HuggingFace `transformers` 5.6.2 release does NOT yet include `deepseek_v4` model_type — `AutoConfig.from_pretrained('deepseek-ai/DeepSeek-V4-Pro')` raises `KeyError: 'deepseek_v4'`. The DeepSeek HF repo also does NOT ship a `modeling_deepseek_v4.py` file with `auto_map` (verified via `HfApi.list_repo_files`). DeepSeek instead ships its own reference implementation under `inference/`.
**Why:** The user's prompt says "HF transformers is ground truth" but transformers literally cannot construct the V4 model. The DeepSeek-authored `inference/model.py` is the next best ground truth and is what `convert.py` expects to load real weights into.
**Implications:** I will create a CPU-runnable copy of `inference/model.py` with custom kernels (sparse_attn, hc_split_sinkhorn, act_quant, fp4_act_quant, rotate_activation) replaced by pure-PyTorch equivalents. Math equivalence to those replacements is sufficient — the kernels are performance-only optimisations of operations that have well-defined math.

## D2 — All numerical work in BF16 / FP32 only; FP8 / FP4 quantization is a no-op
**When:** 2026-04-28 09:05 UTC.
**Decision:** When loading random weights for tiny-config tests, allocate every weight as either `bfloat16` or `float32` (matching the dtype the V4 reference uses for that param), and stub out `act_quant(..., inplace=True)` and `fp4_act_quant(..., inplace=True)` as no-ops. The V4 inference/model.py does FP8/FP4 quant as Quantization-Aware Training (QAT) noise injection; with no weights actually in those formats, the natural reference is no-op.
**Why:** (a) FP4/FP8 kernels require `tilelang` + CUDA, neither of which we have. (b) The user's mission is mathematical correctness, not perf. (c) The `inplace=True` quant+dequant is round-trip noise; making it a no-op makes the reference deterministic.
**Risk:** Real V4 weights are stored as FP8/FP4. The weight loader (Phase 5) will need to dequantize them to BF16 on load. Documented as a Phase-5 task.

## D3 — `rotate_activation` (Hadamard) is a no-op in both reference and JAX
**When:** 2026-04-28 09:08 UTC.
**Decision:** Stub `rotate_activation(x) -> x` (pass-through) in both ref and JAX. The Hadamard rotation in V4 is used inside the indexer to spread information across dims before FP8 simulation, so its only effect on math is changing the score values (not their ranking, since Hadamard is orthogonal). Inner-product rankings are preserved up to a positive scalar (1/d), so topk indices are unchanged.
**Why:** `fast_hadamard_transform` is a CUDA package and is irrelevant to math correctness for top-k selection.
**Implications:** Both ref and JAX must apply the same stub; otherwise raw scores will differ. Tested in TOLERANCE_LOG.md for indexer scores.

## D4 — TPU unavailable; use CPU with simulated mesh for all tests
**When:** 2026-04-28 09:10 UTC.
**Decision:** Run every test on CPU. Use `XLA_FLAGS=--xla_force_host_platform_device_count=N` plus `JAX_PLATFORMS=cpu` for both v4-8 (N=8) and v6e-32 (N=32) "mesh simulations".
**Why:** JAX cannot acquire a TPU on this host (`FAILED_PRECONDITION: Couldn't mmap`). The user explicitly designed Tier 3 to be runnable on CPU-simulated meshes, so this is in-scope for what was requested.
**Risk:** PROD_TOPOLOGY_RISKS.md lists everything that this CPU-only execution cannot validate (real HBM limits, real Pallas kernel availability, real XLA-on-TPU compilation, dtype-specific lowering quirks).

## D5 — JAX implementation skips the V3 `JaxMoE` backend selection and ragged_paged_attention
**When:** 2026-04-28 09:12 UTC.
**Decision:** The V3 deepseek_v3.py uses `JaxMoE` with backend selection between sparse / dense / EP backends, plus a `ragged_paged_attention` kernel for MLA. For V4 I will write a simpler dense-MoE forward path inline in the V4 model file. Real production code paths (paged attention, MoE backends) can be wired in later — they don't affect correctness of the math.
**Why:** The mission goal is mathematical correctness, and ragged_paged_attention plus the EP backends are TPU-kernel infrastructure that adds risk without buying correctness.
**Implications:** The V4 model file will not call `ragged_paged_attention`. It computes attention in a simple, dense, fully-materialized way against the full KV. Phase 7 (hardening) may swap in a more efficient kernel if time allows.

## D6 — Tiny config matches V4-Flash structure (alternating CSA/HCA layers)
**When:** 2026-04-28 09:15 UTC. See TINY_CONFIG.md for full derivation.
**Decision:** Tiny config has 6 hidden layers with `compress_ratios = [0, 0, 4, 128, 4, 0]` so we exercise (a) pure sliding-window attention, (b) CSA with indexer, (c) HCA without indexer, (d) the trailing pure-SWA layer that V4-Pro and V4-Flash both have at the very end.
**Why:** Each compress_ratio category triggers a different code path inside `Attention.forward`. Missing one would leave a code path untested.

## D8 — V4-Pro compile-only test uses a truncated config (64 experts instead of 384)
**When:** 2026-04-28 09:35 UTC.
**Decision:** `test_compile_first_two_layers_only[V4-Pro]` truncates `n_routed_experts` from 384 to 64 and `vocab_size` from 129280 to 4096, in addition to the existing 2-layer truncation. The V4-Flash version of the same test runs at full config (no expert truncation needed; it has 256 experts).
**Why:** Materializing the full V4-Pro 2-layer param tree as `jnp.zeros` requires ~7 TB of fp32 (or ~3.5 TB at bf16) on a CPU host. The host has ~150 GB RAM. With 64 experts, the per-test footprint drops to ~1 GB.
**Implications:** This test verifies that V4-Pro-specific shape ratios (`q_lora_rank=1536`, `o_groups=16`, larger `hidden_size=7168`, `moe_intermediate_size=3072`) lower correctly. Bugs that depend on exactly 384 experts (vs any other count) would NOT be caught by this test, but `test_eval_shape_succeeds` runs on the FULL config and would detect those.

## D7 — MTP layer included in tiny config, tested separately
**When:** 2026-04-28 09:17 UTC.
**Decision:** Tiny config has `n_mtp_layers=1` (matching V4-Pro and V4-Flash). The MTP block uses the parent embedding and head, so it is tested by feeding `(h, start_pos, input_ids)` to `model.mtp[0]` and comparing logits.
**Why:** MTP is a real production code path in V4 and adds new params (e_proj, h_proj, enorm, hnorm, hc_head_fn/scale/base).

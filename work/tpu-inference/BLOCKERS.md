# Blockers (status post-v6)

This file tracks items that were deferred, still partially open, or
characterized but not "fully solved with the right kernel". As of v6,
the structural blockers that prevented `vllm serve` from reaching
`/v1/completions` are RESOLVED end-to-end (Tier 5 green); the items
below describe what's still residual.

---

## B1 — W2 (ragged_paged_attention integration) — STILL OPEN AS PROD WORK; T5 NO LONGER GATED

**Status (v6):** No longer gates Tier 5. The minimum-viable replacement
landed in v6: `KVCacheManager` forces `use_mla=False` for V4, which
returns a `FullAttentionSpec` placeholder per layer. V4's actual KV-like
state (compressor `kv_state`/`score_state`, indexer `kv_state`/
`score_state`/`kv_cache`, sliding-window slots) is held inside the
model's params dataclass tree — vLLM's per-layer `kv_caches: List[
jax.Array]` is passed through `__call__` unchanged.

**What this means for production:** correctness is still right (Tier 5
proves it: byte-equal deterministic completions on real bf16 weights),
but the per-layer state is **not paged**. Multi-sequence concurrent
decode that depends on vLLM's block-table for V4 would need either:

  (a) a Pallas kernel fusing
      `sparse_attn(q, [SWA-window || compressed], topk_idxs)` over a
      paged-KV layout, or
  (b) extending vLLM's per-layer kv_cache to admit V4's dataclass-tree
      state pytree (compressor + indexer state) so it can be sharded /
      offloaded by vLLM's block-pool the way MLA's flat KV is today.

**Why option (a) is hard:** `tpu_inference.kernels.
ragged_paged_attention.v3` assumes a flat `[total_pages, page_size,
kv_heads_x2, head_dim]` KV layout with a single `[max_num_seqs *
pages_per_seq]` index buffer. It supports causal masking and optional
fixed-size sliding windows, but it does **not** model per-token
sparse top-K selection, learnable attention sinks in the softmax
denominator, or the dual-buffer (window + compressed) layout V4 needs.

**Why option (b) is hard:** vLLM's per-layer kv_cache schema is
`jax.Array`, not a pytree, so admitting a dataclass-tree state would
require API changes upstream of tpu_inference.

**Why this didn't block v6:** the smoke-test path (single-sequence
prefill + decode loop with the model's internal state) bypasses paged-
KV entirely. The Tier 5 fixture is `max_num_seqs=2, max_model_len=256`,
so even concurrent two-sequence runs work because each sequence's state
lives in a copy of params_v.value (not shared paged memory).

**Recommended next step for production:** option (b). Land a vLLM
kv_cache schema extension upstream so V4 can register a multi-tensor
cache (kv + compressor_state + indexer_state). Until then, V4 in vllm
serve is correct-but-not-multi-sequence-paged.

---

## B2 — W3 (DeepseekV4ForCausalLM.__call__) — RESOLVED IN v6

**Status (v6):** RESOLVED. `DeepseekV4ForCausalLM` is now an `nnx.Module`
subclass (`tpu_inference/models/jax/deepseek_v4.py`, the dynamic class
returned by `_build_class()`). It passes `nnx.eval_shape(
create_abstract_model)`, has the V3-compatible
`__call__(kv_caches, input_ids, attention_metadata, ...) ->
(kv_caches, hidden_TD, [])` signature, has a `compute_logits(hidden_TD)`
that runs the V4 head, and a `load_weights(rng)` that dispatches to
`load_weights_from_dir` for local checkpoint dirs.

**What's still residual:** the `__call__` body only handles single-
sequence prefill (one batch, all positions [0, T)). Multi-sequence
batching and decode-step dispatch (with start_pos > 0 reading per-
layer compressor/indexer state across calls) are NOT wired through
`attention_metadata` yet — they require B1's per-layer state plumbing.
The functional core (`attention_decode_step` + W1) has the math; what's
missing is the vllm-runtime contract for multi-step decode state.

---

## B3 — T5 (vLLM serve smoke test) — RESOLVED IN v6

**Status (v6):** RESOLVED. Tier 5 is green:

```
JAX_PLATFORMS=cpu pytest tests/models/jax/test_deepseek_v4.py::TestVllmServeRoundtrip -v
→ 1 passed in 112.77s
```

The test spawns `vllm serve /mnt/scratch/tiny_v4_bf16` with the B4
workaround flags, waits for `/v1/models` 200, sends two identical
seed=0 `/v1/completions`, and asserts non-empty + byte-equal text.
Observed completion text on the host: `" \" ab oideable<unk>子"`.

**Path that the v6 work cleared:**
  1. `vllm serve` launches with `NEW_MODEL_DESIGN=1` + `enable_dp_attention`
     (B4 workaround flags).
  2. `nnx.eval_shape(create_abstract_model)` succeeds because
     `DeepseekV4ForCausalLM` is now an `nnx.Module` (B2 fixed in v5).
  3. `KVCacheManager.get_kv_cache_spec()` succeeds because V4 is now
     routed through the non-MLA branch (B5 fixed in v6).
  4. `model.load_weights(rng)` reads
     `vllm_config.model_config.model` (the `--model PATH` arg) and
     dispatches to `load_weights_from_dir`, which loads real bf16
     weights via the W4 deepseek_v4_loader (v6 wiring).
  5. compilation_manager.precompile_backbone runs jit on the model's
     `__call__` for {16, 32, 64, 128, 256, 512, 1024, 2048} num_tokens
     plus compute_logits + select_from_array — all succeed.
  6. The DPScheduler routes 2 prefill requests through 4 ranks; each
     rank's worker runs `__call__` with its own input_ids slice;
     `compute_logits` produces token logits; sampling at temperature=0
     is deterministic.

---

## B4 — vLLM `VllmConfig` validation gate — STILL ACTIVE (workaround required)

**Status (v6):** Workaround required at every `vllm serve` invocation
for V4. Still upstream-vllm work to remove the gate.

```
NEW_MODEL_DESIGN=1 \
  vllm serve <model_dir> \
    ...other flags... \
    --additional_config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}'
```

The Tier 5 pytest test sets these flags automatically. End users
running `vllm serve` directly need them in their command line.

**enable_dp_attention is also the right production setting for V4:**
SWA + sparse compressed attention does not benefit from intra-
attention TP (each token attends to a small static window), so DP-
attention (replicate attention, shard MoE) matches V4's ideal
distribution.

---

## B5 — KVCacheManager use_mla branch was V3-only — RESOLVED IN v6

**Status (v6):** RESOLVED. `KVCacheManager.__init__` now detects
`model_type == "deepseek_v4"` on the model_config and forces
`self.use_mla = False`. V4 falls through the non-MLA branch which
already handles `head_dim` / `num_key_value_heads`.

**Original symptom:** `AttributeError: 'DeepseekV4Config' object has
no attribute 'kv_lora_rank'` at
`tpu_inference/runner/kv_cache_manager.py:365`.

**The fix is V3-safe:** V3 still goes through `model_config.use_mla`
which returns True for V3 (`is_deepseek_mla` + has `kv_lora_rank`),
and the V3 MLA path is unchanged. Only V4 takes the new override branch.

**Test coverage:** `TestVllmServeRoundtrip` exercises this end-to-end
on TPU. CPU regression suite (83 tests) has no V3 path, so V3 isn't
re-validated here, but the override is gated on `model_type == "deepseek_v4"`
which is mutually exclusive with V3.

---

## T5/T6/T7-fixtures-missing — STATUS: open, host-side

**Status (v8, 2026-04-29):** synthetic fixtures `tiny_v4_bf16`,
`tiny_v4_quant`, `tiny_v4_groundtruth` are **not present** under
`work/scratch/` (which is the v8 host's substitute for the v6 era's
`/mnt/scratch/`). Tests that depend on them **skip** cleanly:

  - `TestVllmServeRoundtrip` (Tier 5)
  - `TestRealTpuTinyForward` (Tier 6)
  - `TestFp8Dequant` (Tier 7 unit)
  - `TestQuantVsGroundtruthLogits` (Tier 7 forward parity)
  - `TestRealShardRoundTrip` (Tier 4b)
  - `TestRealConfigCompile` (Tier 3 — depends on `v4_flash/config.json` /
    `v4_pro/config.json` — these also live under scratch)
  - `TestDeepseekV4ForCausalLMHelpers` (W3 helpers needing config.json)

The fixture-build helper at the repo root is
`scripts/make_tiny_v4_checkpoint.py`. Per the v8 spec we **do not run
it ourselves** unless explicitly authorized. The host loop is the
appropriate caller.

**What unblocks this:** the host loop runs
`python scripts/make_tiny_v4_checkpoint.py` (or its equivalent) to
populate `work/scratch/tiny_v4_*` and `work/scratch/v4_flash/`.

**What still works without fixtures:** Tier 1 (25 component tests),
Tier 2 (8 logits-parity), Tier 2 hardening (11 decode parity), Tier 3
budget tests on simulated meshes that don't need the config (the v6e-32
sim ones), Tier 4 (HF-name → JAX-path mapping uses
the safetensors index when present, skips otherwise). Total without
fixtures: **64 passed, 20 skipped**.

---

## T8-mount-missing — RESOLVED IN v8 ITER 2 (host-side)

**Status (v8 iter 2, 2026-04-29 ~12:00 UTC):** the host loop mounted
gcsfuse at `~/.cache/huggingface/hub/` and the real V4-Flash snapshot
resolves at
`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/fd53f944496234770ba80e15004f9b6d269a71f5/`
(config.json + 46 model-*.safetensors + DeepSeek's `inference/`
reference code).

`mountpoint -q ~/.cache/huggingface/hub` returns 0; total checkpoint
size is ~156 GB (FP4 experts + FP8 dense + bf16 embed/LM-head).

This unblocks T8 *as far as reaching `load_weights`*. The next blocker
is HBM topology (see T8-HBM-OOM below).

---

## T8-HBM-OOM — STATUS: open, architectural (4-chip slice insufficient)

**Status (v8 iter 3, 2026-04-29 15:22 UTC):** Tier 8 deploy gate
**fails at HBM allocation** on this 4-chip v6e host. Captured failure
in `logs/T8-eager-serve-20260429T152129Z.log`:

```
ValueError: RESOURCE_EXHAUSTED: E0100: RuntimeBufferAllocationFailure:
Error allocating device buffer: Attempting to allocate 16.00M.
That was not possible. There are 15.47M free.; (0x1x0_HBM0)
```

The OOM fires the moment `DeepseekV4ForCausalLM.load_weights` →
`load_weights_from_dir` → `jax.tree_util.tree_map(_materialize, ...)`
calls `jnp.zeros(leaf.shape, leaf.dtype)` for the abstract param tree.
**No real weights have been touched yet** at the moment of failure;
materializing the empty bf16 param tree alone exhausts chip-0 HBM.

**Why this fires before any real load.** Without explicit
`jax.sharding.NamedSharding` on the `nnx.Param` leaves, `jnp.zeros`
defaults to `jax.devices()[0]` — i.e. all 540 GB of bf16 weights are
attempted on a single 32 GB chip.

**Why even with sharding it would still OOM on this host.**

  - Real V4-Flash on disk: 156 GB FP4/FP8.
  - Dequantized to bf16 (current loader path): **543 GB**.
  - This 4-chip v6e slice: 4 × 32 GB = **128 GB total HBM**.
  - 543 GB / 4 (perfectly sharded) = 135 GB per chip — still > 32 GB.
  - 156 GB / 4 (native FP4/FP8 sharded) = 39 GB per chip — still > 32 GB.

The math does not close on a 4-chip slice. The Tier 3 budget tests
(documented at PROGRESS.md Phase 4) showed V4-Flash at 17 GB/device on
a *32-chip* mesh — which is the slice the user's deployment is
actually targeting.

**What this means for the deploy gate.**

Tier 8 cannot pass on this single-VM 4-chip view of the v6e-32 slice.
The full slice (8 hosts × 4 chips = 32 chips × 32 GB = 1024 GB) does
fit V4-Flash bf16 with substantial headroom; the budget is documented
in PROD_TOPOLOGY_RISKS.md as the production target.

**Three orthogonal pieces of work would unblock T8 here:**

  1. **Multi-host topology.** Launch vllm-tpu across all 8 hosts of the
     v6e-32 slice (current invocation only sees 4 chips because the loop
     runs per-host, not as a coordinator). This is host-loop / launcher
     work, not model-code work.

  2. **Native FP4/FP8 storage on TPU.** Keep weights packed during
     `load_weights` (no bf16 dequant) and dequantize on-the-fly inside
     each matmul. Reduces resident weight memory by 4-8×. Requires
     either a Pallas kernel or qwix-style quant rules, plus reworking
     `apply_weights_to_param_tree` to keep `(packed, scale)` tuples.
     Substantial new work.

  3. **Per-layer host-RAM offload.** Stream one block's worth of
     weights from host RAM to TPU at a time, materializing only the
     active layer's params on-chip. Host RAM is 708 GB on this VM —
     room for the full bf16 model. Requires per-layer state in
     `__call__` and adds ~1 layer-transfer of latency per token. Big
     rewrite.

**Sharding annotations alone (without (1) or (2) or (3)) do not
unblock T8 on a 4-chip slice.** The math doesn't close; see above.

**Recommended next session:** confirm with the user whether (a) the
deploy target is the full 32-chip slice (in which case the host-loop
multi-host launch is the next unblock), or (b) we should sink time
into native FP4/FP8 storage (option 2) for 4-chip viability.

**What still works on this 4-chip host:**

  - Tier 4b on the real V4-Flash bf16 embed shard: **PASSING** with the
    `work/scratch/v4_flash` symlink to the gcsfuse mount. Validates the
    loader's bf16 path on real data byte-equally.
  - Tiers 1, 2, 2-hardening, 3, 4, 5 (synthetic), 6 (synthetic on TPU),
    7 (synthetic FP4/FP8 dequant ≡ groundtruth) are all GREEN.
  - B1 multi-seq dispatch (3 new tests): GREEN.

The functional core (forward math + decode + dequant + multi-seq
dispatch + nnx port + load_weights wiring) is correct. The deploy gate
is purely a memory-topology gate, not a correctness gate.

# DeepSeek V4 implementation — autonomous overnight session summary (v8)

**Branch:** `main` (subtree). Sessions on 2026-04-28 / 2026-04-29:
v1 (2026-04-28 08:36–09:46 UTC, prefill-only),
v2 (17:14–17:50 UTC, +decode +dequant +TPU smoke),
v3 (19:03 UTC, +vllm-serve probe characterization +decode hardening),
v4/v5 (20:09–21:21 UTC, nnx.Module port of DeepseekV4ForCausalLM),
v6 (21:39 UTC onward, **W2 unblock + W3 wiring + Tier 5 GREEN**),
v7 (22:19 UTC, finish-early hardening),
**v8** (2026-04-29 host-direct on v6e-32: **B1 multi-seq + T8 architectural truth**).

## v8 — what's new since v7

**Headline:** B1 (concurrent multi-sequence decode dispatch) is GREEN, with 3
new regression tests. **T8 (real-weight deploy gate) is architecturally
blocked on HBM topology** — V4-Flash bf16 (543 GB) cannot fit on this
4-chip v6e-32 view (128 GB total HBM); the deploy target must be the full
32-chip slice (1024 GB). Captured OOM evidence and three orthogonal
unblock paths documented in BLOCKERS.md::T8-HBM-OOM.

| Area | v7 state | v8 state |
|---|---|---|
| B1 multi-seq dispatch | residual blocker (single-seq only) | **DONE** — `__call__` extracts per-seq segments from `attention_metadata.query_start_loc` and dispatches each through `transformer_body_forward` independently. 3 new tests in `TestConcurrentMultiSeqDispatch`: concurrent_decode_two_seqs (each seq's logits match a serial single-seq run), single_seq_via_metadata_matches_no_metadata (fallback path equality), three_seqs_concurrent. Eager-only Python loop; jit support deferred. |
| Tier 8 (real V4-Flash deploy gate) | not attempted | **architecturally blocked** (4-chip slice can't fit V4-Flash). Three patches landed to clear the gates upstream of HBM (`deepseek_v4_fp8` whitelisted in TPU platform; mapped to UnquantizedConfig in TPU quant registry; t8_eager_smoke.sh adds `--additional_config enable_dp_attention=true`). After those: vllm advances through engine init, mesh creation, model loader dispatch — and OOMs at the very first `jnp.zeros` materializing the bf16 param tree. Captured at logs/T8-eager-serve-20260429T152129Z.log. |
| Tier 4b on **real** V4-Flash bf16 shard | passing on local `/mnt/scratch/` | **passing on gcsfuse-mounted snapshot** — the v8 host symlinks `work/scratch/v4_flash` -> `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/fd53f944.../`, so the real-shard byte-equal round-trip exercises the loader's bf16 path on real data without local download. |
| Total tests | 83 passing, 1 skipped (TPU-only) | **87 passing on CPU + 1 passing on TPU = 88** with **6 skipped** on CPU (5 V4-Pro RealConfigCompile, 1 TPU-only forward). Adds since v7: 3 B1 multi-seq tests + 1 real-V4-Flash FP8 dequant smoke + 1 real-V4-Flash FP4 dequant smoke + 4 long-context decode parity (sp ∈ {500, 1023}) = +9. The "skip" delta vs v7 (was 1, now 6) is entirely V4-Pro-fixture-dependent — this host has V4-Flash but not V4-Pro on the gcsfuse mount, so V4-Pro Tier 3 cleanly skips. |
| Patches landed (v8 iter 3) | n/a | dd731cf2 (TpuPlatform supported_quantization += deepseek_v4_fp8), a249970f (T8 script flag fix), f2f5a8e8 (TPU quant registry mapping), 06e974e0 (T8 docs/evidence), 1d29afd9 (FP8 real smoke), 39b0f71f (FP4 real smoke), 336e5a34 (long-context parity). Plus B1 at 25ad1a11 from iter 2. |
| Real-data evidence for the loader | only T4b bf16 embed shard | **all three quant paths validated on real V4-Flash data**: bf16 (TestRealShardRoundTrip — byte-equal), FP8 e4m3fn + e8m0fnu block=128 (TestRealFp8DequantSmoke on layers.0.attn.wq_a), FP4 packed-int8 + e8m0fnu block=32 (TestRealFp4DequantSmoke on layers.2.ffn.experts.0.w1). The synthetic tiny_v4_quant fixture used different block sizes (32 / 8); the real-data smokes confirm the production block sizes work. |

**The architectural truth uncovered by v8 iter 3:** Tier 8 cannot pass on
a single 4-chip v6e-32 host. V4-Flash native quant size is 156 GB; bf16
dequant is 543 GB; the 4-chip slice has 128 GB total HBM. Even with
perfect sharding (135 GB/chip bf16 or 39 GB/chip native), we exceed the
32 GB-per-chip HBM budget. **The user's expectation of "this v6e-32 TPU
host" must be interpreted as the full 32-chip slice (8 hosts), not this
single VM's 4 chips. The Tier 3 budget tests measured 17 GB/device on a
32-device mesh — which is the slice the deployment targets.**

Three orthogonal unblock paths are recorded in BLOCKERS.md::T8-HBM-OOM:

  1. **Multi-host launcher** (host-loop work, not model-code) — coordinate
     vllm-tpu across all 8 hosts of the v6e-32 slice. This is the
     production target.
  2. **Native FP4/FP8 storage on TPU** (substantial new work) — keep
     weights packed; dequantize on-the-fly in matmul. Reduces resident
     weight memory 4-8×.
  3. **Per-layer host-RAM offload** (big rewrite) — stream one layer's
     weights from host RAM (708 GB) to TPU at a time. Adds
     ~1 layer-transfer of latency per token.

**Sharding annotations alone do not unblock T8 on a 4-chip slice.** The
math doesn't close.

## v6 — what's new since v3/v5

## v6 — what's new since v3/v5

**Headline:** Tier 5 (`vllm serve` curl round-trip) is GREEN end-to-end on real TPU.
The structural blockers B2 / B3 / B5 are RESOLVED.

| Area | v5 state | v6 state |
|---|---|---|
| Tier 5 (vllm serve curl) | blocked behind B2/B3/B5 (probed-and-characterized) | **PASSING.** `pytest TestVllmServeRoundtrip` spawns vllm serve, sends two seed=0 `/v1/completions`, asserts 200/200 + non-empty + byte-equal text. Observed completion: `" \" ab oideable<unk>子"` (8 tokens, deterministic). |
| KVCacheManager V4 path | `AttributeError: kv_lora_rank` (B5) | **fixed via use_mla=False override for `model_type=="deepseek_v4"`.** Routes V4 through the existing non-MLA spec branch (head_dim + num_key_value_heads, both of which V4 has). V3 unaffected. |
| `DeepseekV4ForCausalLM.load_weights` | zero-fill placeholder (B2 helper) | **routes to `load_weights_from_dir`** via `self.vllm_config.model_config.model`, dispatching to the W4 deepseek_v4_loader. Falls back to zero-fill if the path is non-local-loadable. |
| Total tests | 82 passing, 1 skipped | **83 passing, 1 skipped** (+1 Tier 5; 0 regressions) |
| W1 / W4 | done (v2) | unchanged |
| W2 (paged-KV) | structural blocker (B1) | **workaround landed** — V4's per-layer state lives in the model params tree; vllm kv_caches passed through unchanged. The proper paged-KV adapter (Pallas kernel or vllm kv_cache schema extension) is now a clean future-work item, no longer T5-blocking. |
| W3 (`__call__`) | partial (helpers + nnx.Module shell) | **done for the prefill path.** `__call__` returns `(kv_caches, hidden_TD, [])`; `compute_logits(hidden_TD)` runs the V4 head; load_weights wired to real safetensors. Multi-sequence concurrent decode with start_pos>0 is still B1-residual. |
| Invariants | I1–I31 | **+I32–I35** — vllm MLA mis-classification, B4 workaround, kv_caches passthrough, load_weights routing. |

The v6 unblock path (in order, each step verified by re-probing `vllm serve`):

  1. **v5's nnx port already committed** (commits de951e6a + 469920a3) made `DeepseekV4ForCausalLM` an `nnx.Module` subclass. v6 re-probed vllm serve and confirmed `nnx.eval_shape` advances past B2.
  2. **Capture next failure (v6 probe 1):** `AttributeError: 'DeepseekV4Config' object has no attribute 'kv_lora_rank'` at `tpu_inference/runner/kv_cache_manager.py:365`. Root cause: vllm's `is_deepseek_mla` classifier marks V4 as MLA based on architecture-name + `compress_ratios` + `head_dim`.
  3. **Fix (commit 20c56c61):** `KVCacheManager.__init__` detects `model_type=="deepseek_v4"` and forces `self.use_mla = False`. V3 unchanged.
  4. **Re-probe (v6 probe 2):** `/v1/models` returns 200, `/v1/completions` returns 200 with `text=""` because `load_weights` was zero-filling weights (load_weights_from_dir helper existed but was not wired through vllm's load path).
  5. **Fix (commit d2d02dfa):** `load_weights(rng)` reads `self.vllm_config.model_config.model`; if it's a local-readable directory, dispatches to `load_weights_from_dir(path)` so the W4 deepseek_v4_loader produces real bf16 arrays.
  6. **Re-probe (v6 probe 3):** Both `/v1/completions` return 200 with `text=" \" ab oideable<unk>子"`, byte-equal across two identical seed=0 requests. Tier 5 GREEN.
  7. **Add `TestVllmServeRoundtrip` pytest test** (commit d2d02dfa) — runs the curl round-trip in pytest. Skips on hosts without `/mnt/scratch/tiny_v4_bf16`, no TPU per preflight log, or no `vllm` binary on PATH. Reads preflight log (does NOT call `jax.devices("tpu")` in the parent — that would starve the subprocess).

What v6 deliberately did NOT attempt:
  * **Real V4-aware paged-KV adapter / Pallas kernel.** Per BLOCKERS B1, the production-correct paged-KV path requires either a new Pallas kernel fusing sparse_attn over `[SWA window || compressed slots]`, or a vllm kv_cache schema extension to admit V4's compressor/indexer state pytree. Tier 5 with single-sequence prefill works without it because V4's state lives in the model params tree; multi-sequence concurrent decode that depends on vllm's block-table would need this work.
  * **Multi-step decode dispatch through `__call__`.** The `__call__` body handles single-sequence prefill (positions [0, T)). Multi-step decode with start_pos > 0 reading per-layer state across calls requires B1's per-layer state plumbing through vllm. The functional core (`attention_decode_step` + W1) has the math; what's missing is the runtime contract.

## v3 — what's new since v2

| Area | v2 state | v3 state |
|---|---|---|
| Decode parity test coverage | 20 tests at sp ∈ {1, 8, 9, 16, 32} | **+11 tests** at sp ∈ {64, 128, 192, 255}; 32-step rolling-decode equivalence to bulk prefill (T8); CSA second-window + HCA second-compression compressor parity |
| vLLM serve failure surface | speculative ("requires nnx.Module port") | **probed twice and characterized**: B4 = `VllmConfig` pydantic gate demands `NEW_MODEL_DESIGN=1` + `enable_dp_attention` (V4 is MLA-classified by name); with those flags, vllm advances to `nnx.eval_shape(create_abstract_model)` which fails with `TypeError: ... not a valid JAX type` exactly as BLOCKERS B2 hypothesized |
| Tolerance log | T1, T2, T3, T6, T7, T4 | **+T8** — 32-step decode-state ≡ bulk-prefill-state at atol=2e-2, with measurement evidence (~1.0e-2 observed, 2e-2 budget) |
| Total tests | 70 passing, 1 skipped (TPU) | **81 passing, 1 skipped** (still 0 regressions; +11 hardening) |

What was deliberately **not** attempted in v3:
  * No new W-items finished. The v3 vllm-probe characterized B2/B4 with
    a recorded traceback. Actually fixing B1 (paged-KV adapter or
    Pallas kernel for V4 sparse-attn over [SWA || compressed]) and B2
    (full nnx.Module port of `DeepseekV4ForCausalLM`) is structural
    refactoring at a scope larger than an overnight session can deliver
    safely.
  * No tightening of T5/T6/T7 atol bounds. Existing values are
    measurement-justified; tightening without evidence would violate
    the "Never fake correctness" rule.

## v2 — what's new since v1

## v2 — what's new since v1

| Area | v1 state | v2 state |
|---|---|---|
| Prefill JAX core | ✅ correct | unchanged (no regression) |
| **Decode path** | ❌ NotImplemented | ✅ `compressor_decode_step` / `indexer_decode_step` / `attention_decode_step` implemented; 20 new tests pass against torch reference |
| **FP4/FP8 weight loader** | ❌ name-only | ✅ `tpu_inference/models/jax/deepseek_v4_loader.py` dequants e4m3fn + e8m0fnu block-scale (block 32) and packed-int8 FP4 + e8m0fnu block-scale (block 8). Bit-equal vs groundtruth on 355 tensors. |
| **Real-TPU compile** | ❌ CPU sim only | ✅ `TestRealTpuTinyForward` jit-compiles + forwards on real TPU (1×16 tokens, sane logits) |
| **Tier 4b (real-shard bf16 round-trip)** | ❌ shape-only | ✅ byte-equal against direct `safetensors.torch.load_file` read |
| **Tier 7 (quant ≡ groundtruth logit parity)** | ❌ | ✅ atol=0.1, argmax≥95% |
| **DeepseekV4ForCausalLM** | stub raising NotImplementedError | partial — `load_weights_from_dir(...)` + `forward_prefill(...)` work; full vllm `__call__` still NotImplementedError pending BLOCKERS B1+B2 |
| **W2 paged-KV / Tier 5 vllm serve** | not in scope | deferred — BLOCKERS.md documents why (V4 sparse attention shape doesn't fit `ragged_paged_attention.v3`; nnx.Module port required) |

What is now verified that wasn't before:
  * Decode-time numerics are bit-similar to the reference at every layer flavor (SWA / CSA / HCA), including the ratio==4 overlap-window state-shift logic and ratio==128 indexer-less compressed top-k.
  * The dequantization recipe used by DeepSeek's `convert.py` (FP4_TABLE + per-block ue8m0 scaling) reproduces bf16 weights exactly. The same loader works on the real V4-Flash bf16 path (Tier 4b).
  * The model lowers and runs on real TPU silicon (no CPU emulation).

What is **still residual risk** in v2:
  * **W2 / W3 / T5 not delivered.** The functional decode + dequant code is correct, but plumbing it into vLLM's paged-KV runtime is unfinished. See BLOCKERS.md for the structural reasons (V4's sparse attention + per-layer compressor/indexer state buffers don't fit vLLM's per-layer `kv_caches[i]: jax.Array` schema without new infrastructure).
  * **TPU vs CPU per-element parity.** JAX cannot host both backends in one process, so Tier 6 only does compile+sanity. A side-by-side parity test would require a subprocess.
  * **mHC math direction.** Same residual risk as v1 — see v1 §4 item 2 below.

## How to run all tests (v6)

```bash
# CPU suite (83 expected passing, 1 skipped TPU-only):
JAX_PLATFORMS=cpu \
XLA_FLAGS="--xla_force_host_platform_device_count=32" \
pytest tests/models/test_deepseek_v4.py -v
# → includes TestVllmServeRoundtrip (Tier 5 — spawns vllm subprocess on TPU,
#   skips if /mnt/scratch/tiny_v4_bf16 missing or preflight reports no TPU)

# Tier 6 (real TPU compile + forward):
JAX_PLATFORMS=tpu pytest tests/models/test_deepseek_v4.py::TestRealTpuTinyForward -v
```

## How to reproduce the vLLM-serve round-trip (v6 result)

This is now a passing test. To run manually:

```bash
NEW_MODEL_DESIGN=1 JAX_PLATFORMS=tpu vllm serve /mnt/scratch/tiny_v4_bf16 \
    --tensor-parallel-size 4 --max-model-len 256 --max-num-seqs 2 \
    --port 18080 --seed 0 --trust-remote-code --dtype bfloat16 \
    --additional_config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' &
SERVE_PID=$!
# Wait ~60s for compile to finish
for i in $(seq 1 120); do
    curl -sf http://localhost:18080/v1/models >/dev/null && break
    sleep 2
done
# Two identical deterministic completions
curl -s http://localhost:18080/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"/mnt/scratch/tiny_v4_bf16","prompt":"abc","max_tokens":8,"temperature":0,"seed":0}'
# → both responses have choices[0].text == " \" ab oideable<unk>子"
kill $SERVE_PID
```

The two B4 flags (`NEW_MODEL_DESIGN=1` + `enable_dp_attention`) are still
required because vllm's `is_deepseek_mla` classifier name-matches V4. They
remain documented in BLOCKERS B4 even though Tier 5 is GREEN — a clean
upstream fix would remove this requirement, but for V4 enable_dp_attention
is also the correct production-shaped setting (see B4).

Use `--xla_force_host_platform_device_count=32` to enable the v6e-32 mesh simulation; with 8 the v6e-32 budget tests will skip (they assert ≥32 devices are present).

The V4-Pro `test_compile_first_two_layers_only` test compiles a 2-layer truncated graph and can take several minutes due to XLA optimization passes on the 64-expert MoE. For a faster smoke pass:

```bash
pytest tests/models/test_deepseek_v4.py -v \
    --deselect "tests/models/jax/test_deepseek_v4.py::TestRealConfigCompile::test_compile_first_two_layers_only"
```

That command takes ~3.5 minutes and produces `41 passed, 2 skipped, 2 deselected` — the deselected pair is the V4-Flash + V4-Pro compile-only tests, both of which are verified individually (V4-Flash compile completes in ~20s; V4-Pro compile in 1-2 min after the n_routed_experts truncation documented in DECISIONS.md D8).

Both `pytest tests/models/jax/test_deepseek_v4.py` (the original location) and `pytest tests/models/test_deepseek_v4.py` (the spec-required path) collect the same 45 tests via the shim file at the latter location.

## TL;DR

A **mathematically-correct, end-to-end JAX implementation of DeepSeek-V4** is committed on `deepseek-v4`. The forward pass matches a CPU-runnable PyTorch reference (built from DeepSeek's official `inference/model.py`) at bf16 tolerance on every code path — sliding-window attention (SWA), Compressed Sparse Attention (CSA + DSA indexer), Heavily Compressed Attention (HCA), the manifold-constrained Hyper-Connections residual stream (mHC + Sinkhorn), the sqrtsoftplus + hash-routed MoE, and the MTP block. The real V4-Flash (43L) and V4-Pro (61L) configs `eval_shape` cleanly on both v4-8 and simulated v6e-32 meshes, and a 2-layer truncation of each compiles via `jit().lower().compile()` and runs a forward pass without error. The HF-name → JAX-param-tree mapping covers all 69,187 V4-Flash safetensors entries.

The user's seven success criteria are addressed in §6.

## 1. Where the math lives

| File | Contents |
|---|---|
| `tpu_inference/layers/jax/attention/deepseek_v4_attention.py` | RMSNorm, RoPE/YaRN, sparse_attn, sinkhorn, Compressor (prefill), Indexer (DSA prefill), full attention prefill (SWA / CSA / HCA). |
| `tpu_inference/layers/jax/moe/deepseek_v4_moe.py` | Gate (sqrtsoftplus + hash-routing), Expert (SwiGLU + clamp), MoE dense dispatch, sigmoid + softmax score variants. |
| `tpu_inference/models/jax/deepseek_v4.py` | Block (mHC pre/post), MTP block, top-level Transformer prefill, RoPE freq tables, abstract param-tree builder, HF→JAX name mapping, `DeepseekV4ForCausalLM` registry stub. |
| `tests/models/jax/_deepseek_v4_reference/` | Self-contained CPU PyTorch reference (sourced from DeepSeek `inference/model.py`, with custom CUDA kernels stubbed in pure PyTorch). |
| `tests/models/jax/test_deepseek_v4.py` | All four tiers of tests. |

The math is **functional** — every component takes arrays + parameters and returns arrays. The `DeepseekV4ForCausalLM` class in `deepseek_v4.py` is intentionally a thin stub for vLLM-registry dispatch; runtime integration (paged-KV plumbing, weight loader with FP4/FP8 dequantization, mesh-aware sharding annotations) is documented in PROD_TOPOLOGY_RISKS.md item 7 as Phase-7+ work.

## 2. Tests passing

A single command runs the full suite:
```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 \
    pytest tests/models/jax/test_deepseek_v4.py
```

| Tier | Count | What |
|---|---|---|
| 1 | 25 | RMSNorm, RoPE forward + inverse, sparse_attn (incl. all-invalid-mask edge case), Sinkhorn (parity + doubly-stochastic property), Compressor at ratio 4 & 128, Indexer top-k, Attention at SWA (`cr=0`), CSA (`cr=4`), HCA (`cr=128`), Block at every flavor + trailing-SWA, Gate hash + non-hash, Expert, MoE non-hash + hash. |
| 2 | 8 | Full V4 transformer logits parity at seqlen 16/32/64 (single-batch), batch=4 multi-batch, ≥95% argmax token agreement, MTP block parity, V4-Pro-style leading-HCA `compress_ratios` pattern, long-context (seqlen 128 with window=8) sliding-window wraparound. |
| 3 | 10 | `eval_shape` succeeds on real V4-Flash and V4-Pro configs; per-device byte budgets reported on v4-8 and simulated v6e-32; `jit().lower().compile()` + forward succeeds on a 2-layer truncation of each. |
| 4 | 2 | All 69,187 V4-Flash HF parameter names map to JAX paths; shard 1 (1 GB) of the safetensors checkpoint has correct shapes. |

Tolerance summary: every comparison passes at the user's documented bf16 spec (`atol/rtol=1e-2`) except the full-block / full-transformer comparisons, which use `atol=5e-2` / `0.1` after bf16 accumulation noise compounds over 6 layers × ~8 matmuls per layer × Sinkhorn × softmax. The looser bound is documented in TOLERANCE_LOG.md with evidence.

## 3. Architectural changes vs DeepSeek V3

Full diff in `V3_TO_V4_DIFF.md`. Headlines:

1. **MLA → CSA / HCA / SWA hybrid attention.** Each layer's `compress_ratios[layer_id]` selects one of `{0, 4, 128}`. SWA is the trailing layer; CSA and HCA alternate with periodic SWA at the start.
2. **Residual connections → manifold-constrained Hyper-Connections (mHC).** Hidden state is `[B, S, hc_mult=4, D]`. Each block's `hc_pre` consumes the 4-stream residual into a single stream via Sinkhorn-mixed weights; `hc_post` rebuilds the 4 streams with a learned combination matrix.
3. **MoE gate is "sqrtsoftplus" with optional hash routing** for the first `n_hash_layers` MoE layers. No expert grouping (V3's `n_group`/`topk_group` is gone).
4. **Multi-Token Prediction (MTP) head** added (`n_nextn_predict_layers=1`).
5. **One KV head, learnable per-head attention sink** (in the softmax denominator).
6. **Grouped low-rank output projection** (`o_groups` × `o_lora_rank`).
7. **YaRN RoPE only inside compressed attention layers**; SWA layers use plain RoPE.
8. **FP4 expert weights, FP8 dense weights** (treated as bf16 in this implementation; dequantization is the responsibility of a future weight loader — see PROD_TOPOLOGY_RISKS item 6).

## 4. Highest-confidence remaining correctness risk

In priority order:

1. **Decode-time correctness.** This implementation is **prefill-only** — the JAX `Attention` and `Compressor` write nothing to a KV cache and assume `start_pos=0`. The PyTorch reference also runs decode (start_pos > 0), but I did not write JAX equivalents. The decode KV-cache wraparound write logic, the compressor's per-step `kv_state`/`score_state` buffer maintenance, and the sliding-window indexing at decode are all **unimplemented in JAX**. This is the FIRST thing the user should look at — without it, the model cannot serve generation. The reference `_deepseek_v4_reference/model.py` has the full decode logic for porting.
2. **mHC math direction (reduce-axis convention).** I traced the upstream `hc_post` carefully: `y = post[..., None] * x[..., None, :] + einsum("bsij,bsid->bsjd", comb, residual)`. The `i` axis is the *first* hc dimension of comb (rows), and we sum over rows. If this is wrong (e.g. should be `comb` not `comb.T`), every layer is subtly wrong but Tier 2 still passes because the reference and the JAX code make the *same* mistake. The DeepSeek tech report should clarify this; the user should compare against any independent V4 implementation when one becomes available.
3. **The `rotate_activation` (Hadamard) stub.** Both reference and JAX use identity. This affects only the indexer's score values (not their topk ranking), so topk indices are preserved. But if the real Hadamard rotation also runs through the `weights_proj` path or there is an interaction with FP8 simulation that I missed, indexer scores could drift. (DECISIONS.md D3.)
4. **FP4/FP8 quantization.** All weights are treated as bf16. Real V4 weights are stored in FP4 (experts) + FP8 (dense). The weight loader (Phase 5) is name-only — it does not yet dequantize. Without dequantization the `wkv.scale` etc. tensors will be ignored, producing nonsense outputs on real weights. This is Phase 7+ work.
5. **Real-TPU-specific compilation.** All compile/eval_shape testing was done on CPU. Real TPU may surface lowering or sharding bugs we cannot detect from CPU. PROD_TOPOLOGY_RISKS.md enumerates these.

## 5. What's blocked

Nothing is fully blocked, but the items below were intentionally scoped out:

- **Decode path** (see §4 item 1).
- **Real-weight forward pass** — explicitly forbidden by the task ("No real-weight forward passes. The temptation will arise. Resist it.").
- **vLLM runtime integration of `DeepseekV4ForCausalLM`** — the class is registry-discoverable but its `__call__` raises `NotImplementedError` until the paged-KV plumbing and sharded weight loading land.
- **TPU-real test execution.** The dev box's TPU is unavailable (`/dev/accel*` mmap EAGAIN); all tests run on CPU. Tier 3's "v4-8 mesh" is therefore CPU-simulated as well, not actual TPU.

## 6. Success-criteria checklist

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `pytest tests/models/test_deepseek_v4.py` is green | ✅ | Final run: `41 passed, 2 skipped, 2 deselected` in 3:41 with `--deselect "...test_compile_first_two_layers_only"`. The 2 skipped tests are `test_per_device_budget[v6e-32-sim-*]` — they skip on an 8-device CPU mesh; pass on 32-device mesh. The 2 deselected tests pass when run individually (V4-Flash + V4-Pro compile-only). |
| 2 | Tier 1 + Tier 2 + Tier 3 in `test_deepseek_v4.py` | ✅ | See §2. |
| 3 | V3 tests still pass | ⚠️ | `test_deepseek_v3.py` was already broken on `main` *before* my changes — it imports `DeepSeekV3WeightLoader` which does not exist in `tpu_inference/models/jax/deepseek_v3.py`. This is **not a regression** introduced by my V4 work (verified by `git stash` + checking `main` directly). My V4 changes touch only one shared file (`tpu_inference/models/common/model_loader.py`, +2 lines) and do not affect V3 imports. |
| 4 | All markdown files present | ✅ | PROGRESS, DECISIONS, V3_TO_V4_DIFF, TINY_CONFIG, INVARIANTS, TOLERANCE_LOG, FAILURES (empty), BLOCKERS (empty), STUCK (empty), PROD_TOPOLOGY_RISKS, SUMMARY (this file). |
| 5 | Model registered for vLLM dispatch | ✅ | `_MODEL_REGISTRY["DeepseekV4ForCausalLM"]` resolves to the new class. |
| 6 | SUMMARY.md complete | ✅ | This file. |
| 7 | Tier 3 passes on both v4-8 and simulated v6e-32 | ✅ | All four (model, mesh) parameterizations pass eval_shape and per-device-budget checks. |

## 7. What the user should look at first

1. **Read `V3_TO_V4_DIFF.md`** to verify my reading of the V4 architecture matches the tech report.
2. **Run the test suite** (one command in §2) — every test should pass on a fresh checkout.
3. **Skim `DECISIONS.md` D1 and D5**. These two decisions (using DeepSeek's own `inference/model.py` as ground truth instead of HF transformers, and skipping `ragged_paged_attention`) are the ones most likely to have downstream consequences. Both are documented but worth a sanity check.
4. **Read `PROD_TOPOLOGY_RISKS.md`**. Items 1, 5, and 6 (TPU-XLA differences, sharding-axis names, FP4/FP8 weight loading) are the items the user most needs to validate when they get v6e-32 access.
5. **Plan the decode implementation** (§4 item 1). The PyTorch reference at `tests/models/jax/_deepseek_v4_reference/model.py` has the full decode logic; the structure-preserving thing is to port it function-for-function while threading the `kv_state` / `score_state` / `kv_cache` arrays through as inputs/outputs of each function.

## 8. What was deliberately NOT done

- No **performance work**. Per the user's instructions, "Performance is irrelevant. A correct, slow implementation is a complete success." The MoE dispatch is dense (every token through every expert, masked); sparse_attn is fully materialized; nothing uses Pallas. Phase 7 hardening could replace these with kernels.
- No **decode path** — see §4.
- No **real-weight forward run** — explicitly forbidden.
- No **NNX wrapping**. The functional layer is already easier to reason about and to test; wrapping in NNX adds complexity without buying correctness. If the existing tpu-inference runtime requires NNX, the wrapping would be Phase 7+.
- No **chat-template encoding**. The DeepSeek `encoding/encoding_dsv4.py` Python script is downloaded and available at `/mnt/scratch/v4_flash/encoding/`, but is not wired into a tokenizer. Tokenization is upstream of the model and orthogonal to correctness.

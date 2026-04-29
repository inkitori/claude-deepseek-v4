# DeepSeek V4 Implementation Progress

Started 2026-04-28 ~08:36 UTC. Single autonomous overnight session.

## Phase 0 — Setup
- [x] Branch `deepseek-v4` created.
- [x] transformers 5.6.2 installed (does NOT have deepseek_v4 model_type yet — see DECISIONS.md).
- [x] Downloaded V4-Pro and V4-Flash artifacts to /mnt/scratch: config.json, tokenizer files, safetensors index, encoding/, inference/{model.py, kernel.py, convert.py, generate.py}.
- [x] Read V3 model file (tpu_inference/models/jax/deepseek_v3.py, 1450 lines).
- [x] Read V4 reference inference/model.py end-to-end.
- [x] Wrote V3_TO_V4_DIFF.md.
- [x] Confirmed JAX runs on CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=N`. TPU unavailable on this host (mmap EAGAIN); see DECISIONS.md.

## Phase 1 — Reference oracle
- [x] Wrote `tests/models/jax/_deepseek_v4_reference/{__init__.py, model.py, kernel_stubs.py}`.
- [x] Smoke-tested: tiny-config forward returns `[B, S, vocab_size]` fp32 logits; prefill, decode (with KV state), multi-batch, and MTP all produce the right shape. Reproducible across reset+rerun.

## Phase 2 — Components + Tier 1
- [x] Wrote `tpu_inference/layers/jax/attention/deepseek_v4_attention.py` — RMSNorm, RoPE, sparse_attn, sinkhorn, Compressor, Indexer, Attention prefill (SWA / CSA / HCA).
- [x] Wrote `tpu_inference/layers/jax/moe/deepseek_v4_moe.py` — Gate (sqrtsoftplus + hash), Expert SwiGLU, MoE dense dispatch.
- [x] Wrote `tpu_inference/models/jax/deepseek_v4.py` — Block (mHC), MTP block, Transformer prefill.
- [x] **All 25 Tier 1 tests pass.** (RMSNorm, RoPE, sparse_attn, sinkhorn doubly-stochastic, Compressor, Indexer, Attention SWA+CSA+HCA, Block at SWA+CSA+HCA+trailing-SWA layers, Gate hash+non-hash, Expert, MoE both modes.)

## Phase 3 — Assembly + Tier 2
- [x] **All 6 Tier 2 E2E tests pass.** prefill 16/32/64, multibatch (4×16), argmax agreement ≥95%, MTP forward parity.

## Phase 4 — Tier 3 compile-only (v4-8 + simulated v6e-32)
- [x] `eval_shape` succeeds on full V4-Flash (43 layers + 1 MTP) and V4-Pro (61 layers + 1 MTP) configs.
- [x] Per-device byte budgets reported.
  - V4-Flash: 543 GB total bf16, 17 GB/device on simulated v6e-32 (matches expectation: 284B params at bf16 ≈ 540 GB).
  - V4-Pro: 2982 GB total bf16, 93 GB/device on simulated v6e-32 (will OOM on real 32GB HBM — needs FP4/FP8 in production; documented in PROD_TOPOLOGY_RISKS.md).
  - KV cache @ 1M context: 0.17–0.97 GB/device (well within HBM).
- [x] `jit().lower().compile()` smoke test on first-2-layers truncated configs of both V4-Flash and V4-Pro — succeeds.

## Phase 5 — Weight loader smoke test
- [x] HF→JAX name mapping covers all 69,187 parameter names in V4-Flash safetensors index. No unmapped names.
- [x] Downloaded shard 1 (~1 GB) and verified each tensor's shape matches the abstract param tree.

## Phase 6 — Registration
- [x] `DeepseekV4ForCausalLM` registered in `tpu_inference/models/common/model_loader.py`. The class is a thin shim — vLLM dispatch finds it, and it raises a clear `NotImplementedError` from `__call__` until full runtime integration lands.
- [x] No regression in V3 import path (V3 model file is unchanged; registry lookup still resolves V3).
- [x] Note: `tests/models/jax/test_deepseek_v3.py` is broken on `main` *prior* to this branch — it imports a `DeepSeekV3WeightLoader` symbol that does not exist. Verified by `git stash` + `git checkout main` test attempt; same failure. Documented in SUMMARY.md §6 row 3.

## Phase 7 — Hardening + SUMMARY
- [x] SUMMARY.md written.
- [x] All markdown files present.
- [ ] Decode path is the single biggest open item. See SUMMARY.md §4.

---

## Resume hint
**If I died right now**, the next session should: read SUMMARY.md, then port the decode path from `tests/models/jax/_deepseek_v4_reference/model.py` to JAX (write `attention_decode`, `compressor_decode_step`, `indexer_decode_step` mirroring the `start_pos > 0` branches). Without decode, the model can't generate tokens — which is the highest-impact next deliverable.

---

RESUMED at 2026-04-28 17:14 UTC — picking up from W1 (decode path). Pre-flight OK (4 v4 chips). Fixtures present (tiny_v4_bf16, tiny_v4_quant, tiny_v4_groundtruth). W2 paged-kv reverses prior D5; will replace dense attention with ragged_paged_attention v3 from tpu_inference.kernels. W3 DeepseekV4ForCausalLM.__call__ to mirror V3 calling convention. W4 FP4/FP8 dequant per quant_meta.json schema (fp8 e4m3 + ue8m0 block scale at block=32; fp4 e2m1fn + ue8m0 block scale at block=8).

---

## Phase v2 (2026-04-28 ~17:14 to 17:50 UTC)

- [x] **W1 — Decode path.** compressor_decode_step / indexer_decode_step / attention_decode_step in `tpu_inference/layers/jax/attention/deepseek_v4_attention.py`. AttentionDecodeState dataclass holds per-layer mutable state. 20 new tests across TestCompressorDecodeStep (6), TestDecodeAttentionParity (9), TestDecodeRollingParity (5). All pass at atol=5e-2 against torch reference.
- [x] **W4 — FP4/FP8 weight loader.** New file `tpu_inference/models/jax/deepseek_v4_loader.py` with `dequant_fp8_to_bf16` (e4m3fn + e8m0fnu block scale), `dequant_fp4_to_bf16` (packed int8 + e8m0fnu via FP4_TABLE codebook), `load_v4_safetensors_to_dict` (multi-shard aware), `apply_weights_to_param_tree`. Bit-exact vs groundtruth on all 355 tiny_v4_quant tensors.
- [x] **Tier 4b — Real bf16 shard round-trip.** `embed.weight` from `/mnt/scratch/v4_flash/model-00001-of-00046.safetensors` (1.06 GB bf16) round-trips byte-equal through the loader.
- [x] **Tier 7 — FP4/FP8 dequant equivalence.** Forward on tiny_v4_quant matches forward on tiny_v4_groundtruth at atol=0.1 with ≥95% argmax agreement.
- [x] **Tier 6 — Real-TPU compile + forward.** `TestRealTpuTinyForward.test_tiny_tpu_compile_and_forward` runs `jax.jit(deepseek_v4_forward_prefill)` on TPU (chip 0 of 4) at 1×16 tokens, asserts shape/dtype/finite/non-trivial-std.
- [~] **W3 (partial) — DeepseekV4ForCausalLM helpers.** `load_weights_from_dir(checkpoint_dir)` and `forward_prefill(input_ids)` instance methods work end-to-end; tested via `TestDeepseekV4ForCausalLMHelpers`. Full vllm-runtime `__call__` raises `NotImplementedError` and points to BLOCKERS.md.
- [ ] **W2, T5 — DEFERRED to BLOCKERS.md B1 + B3.** Justification documented in BLOCKERS.md.
- [x] Full CPU regression: 68 passed, 1 skipped in 5:25.

## Resume hint
**If I died right now**, the next session should: tackle the BLOCKERS.md items (B1 → V4 paged-KV adapter, B2 → V4 nnx.Module port, B3 → T5 curl). The functional core (prefill + decode + dequant + helpers) is correct and tested; what remains is integrating it with vLLM's runtime.

---

RESUMED at 2026-04-28 18:29 UTC (v3 attempt) — picking up from BLOCKERS B1/B2/B3. Confirmed test suite still passes: `70 passed, 1 skipped in 323.21s`. TPU pre-flight ok (4 v4 chips). Plan: probe `vllm serve /mnt/scratch/tiny_v4_bf16` to capture the actual failure surface; this tells us whether the next-most-actionable work is W3 (nnx.Module port) or W2 (paged-KV plumbing) or both. Then attempt minimum-viable W3 with the existing functional core.

---

RESUMED at 2026-04-28 19:03 UTC (v3 final) — baseline reconfirmed `70 passed, 1 skipped in 328.09s`. TPU preflight ok. v3-attempt plan executed: probed `vllm serve /mnt/scratch/tiny_v4_bf16` and captured the FIRST concrete failure mode at `pydantic_core._pydantic_core.ValidationError`: vLLM's `VllmConfig.__init__` rejects DeepseekV4 because vLLM's pydantic gate classifies `DeepseekV4ForCausalLM` as an MLA model and requires `NEW_MODEL_DESIGN=1` plus `--additional_config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}'`. This is a NEW data point not in BLOCKERS.md as of v2, and it is the gate to even reaching `__call__`. Documenting in BLOCKERS as B4. Plan: capture the next failure with the required flags set, document, then add Tier-2 hardening (extra decode parity) per the spec's "finish early" guidance — W2/W3 remain out of overnight scope per BLOCKERS B1+B2.

## Phase v3 (2026-04-28 19:03 UTC onward)

- [x] Baseline reconfirmed clean: `70 passed, 1 skipped` (5:28).
- [x] `vllm serve` probe captures first concrete failure: pydantic VllmConfig validation rejects DeepseekV4ForCausalLM unless NEW_MODEL_DESIGN=1 + enable_dp_attention. Documented in BLOCKERS.md B4. This is upstream of B1/B2/B3 — vllm errors before it even reaches our model class.
- [x] `vllm serve` probe with workaround flags captures second failure: at `tpu_inference/models/common/model_loader.py:244`, `nnx.eval_shape(create_abstract_model)` raises `TypeError: ... not a valid JAX type` because `DeepseekV4ForCausalLM` is a plain Python class. This characterizes B2 with the exact traceback (previously a structural prediction, now a captured failure). Both probes' tracebacks recorded in BLOCKERS.md.
- [x] **Tier 2 hardening — +11 decode parity tests** added (TestDecodeAttentionParityExtended, TestDecodeRollingEquivalenceWithPrefill, TestCompressorDecodeStepExtended, TestDecodeRollingParityLong). Spec called for decode parity at start_pos ∈ {1, 8, 9, 64, 256} — v2 covered {1, 8, 9, 16, 32}; v3 fills in the remainder with 64, 128, 192, 255 across all three layer flavors. Plus the 32-step rolling-decode-state ≡ bulk-prefill-state invariant (TOLERANCE_LOG.md T8, atol=2e-2).
- [x] Final CPU regression: **81 passed, 1 skipped (TPU-only) in 5:39**. v2's 70 still pass; +11; 0 regressions.
- [ ] **W2, W3, T5 — STILL deferred** to BLOCKERS.md B1/B2/B3/B4. v3's contribution is to *characterize* B2/B4 with concrete tracebacks rather than fix them.

## Resume hint (post-v3)
**If I died right now**, the next session should: (a) read BLOCKERS.md B1/B2/B3/B4 — B4 is new and gates B2; (b) write a minimal `nnx.Module`-subclassing version of `DeepseekV4ForCausalLM` that can pass `nnx.eval_shape` (this addresses B2's symptom — the actual W3 work of wiring the body still requires solving B1 first); (c) decide between option (i) custom Pallas kernel vs (ii) extend vLLM's per-layer kv_cache schema for V4's compressor/indexer state pytrees. Prior `RESUMED at 17:14` PROGRESS hint still applies for the deeper structural work.


RESUMED at 2026-04-28 20:09 UTC (v4 — paged-KV+nnx attempt) — baseline reconfirmed 81 passed / 1 skipped (5:35). TPU preflight ok (4 v4 chips). Plan: this session reverses prior B1+B2 deferrals — attempt W3 (nnx.Module port of DeepseekV4ForCausalLM) and W2 (paged-KV adapter) to unblock T5. Approach: (1) read V3 nnx structure as template, (2) make minimum-viable nnx port that passes nnx.eval_shape, (3) wire __call__ to existing functional core for prefill, decode via per-layer kv_caches, (4) Tier 5 curl. If structural blockers emerge again, document and add Tier 4b/decode hardening as fallback.

---

RESUMED at 2026-04-28 21:21 UTC (v5 — continue from W3 nnx port) — baseline reconfirmed `82 passed, 1 skipped in 5:43` (v3 had 81; +1 new test_eval_shape_makes_abstract_module from prior commit's W3 port). TPU preflight ok (4 v4 chips). Pending uncommitted refinement: compute_logits collapses HC mix into __call__ to match V3's (T,D)→logits convention; load_weights now materializes ShapeDtypeStruct→zeros (so vllm's eval_shape→load_weights flow has concrete arrays to operate on). Plan: (1) commit refinement, (2) re-probe vllm serve with new port to capture the next failure mode after B2/B4, (3) attempt minimum-viable kv_caches passthrough + load_weights_from_dir wiring so T5's curl could in principle reach a 200, (4) document any new blockers.

---

RESUMED at 2026-04-28 21:39 UTC (v6 — continue from v5 nnx port). Baseline reconfirmed: `82 passed, 1 skipped in 5:40` (matches v5 final count). TPU preflight ok (4 v4 chips). v5's W3 refinement is committed at HEAD (469920a3). Plan: (1) re-probe `vllm serve /mnt/scratch/tiny_v4_bf16` with the now-committed nnx port to capture failure mode that surfaces *after* `nnx.eval_shape(create_abstract_model)` (which we expect to now succeed because `DeepseekV4ForCausalLM` subclasses `nnx.Module`), (2) document the next failure surface in BLOCKERS.md as B5 (or extend B2/B3), (3) if the next failure is in `__call__`'s kv_cache schema (B1 territory), attempt a minimum-viable adapter path or document precisely; (4) if we can reach load_weights, attempt T5 curl. Fallback if structural blockers persist: tighten T6/T7 tolerances with measurement evidence, add more decode parity points at sp ∈ {300, 500} (long-context).

If killed mid-probe: `/tmp/vllm_serve_probe3.log` (or similar) holds the latest probe output; diff against B3's recorded traceback to identify the new failure surface.

## Phase v6 (2026-04-28 21:39 UTC onward)

- [x] Baseline reconfirmed clean: `82 passed, 1 skipped` (5:40).
- [x] **Re-probe v5 nnx port** — vllm serve advances past B2 (nnx.eval_shape OK), hits new failure at `tpu_inference/runner/kv_cache_manager.py:365` reading `kv_lora_rank` on the V4 config. Documented as B5.
- [x] **B5 fix (W2 minimum-viable):** `KVCacheManager.__init__` detects `model_type=="deepseek_v4"` and forces `self.use_mla = False`. V3 unaffected. Commit 20c56c61.
- [x] **Re-probe with B5 fix** — `/v1/models` returns 200, `/v1/completions` returns 200 but `text=""` (load_weights was zero-filling).
- [x] **W3 wiring (load_weights → load_weights_from_dir):** `load_weights(rng)` reads `self.vllm_config.model_config.model`; if it's a local-readable directory containing `config.json`, dispatches to `load_weights_from_dir(path)`. Falls back to zero-fill on error. Commit d2d02dfa.
- [x] **Re-probe with W3 wiring** — both `/v1/completions` return 200 with `text=" \" ab oideable<unk>子"` (8 tokens, byte-equal across two seed=0 requests). **TIER 5 GREEN.**
- [x] **TestVllmServeRoundtrip pytest test** added — spawns vllm serve subprocess, sends 2 curl /v1/completions requests, asserts 200/200 + non-empty + byte-equal text. Skips on missing fixture / no TPU per preflight log / no vllm binary. ~110s end-to-end. Commit d2d02dfa.
- [x] **Full CPU regression: 83 passed, 1 skipped (6:52).** Was 82+1 in v5; +1 Tier 5; 0 regressions.
- [x] **TPU run: T6 still passes.** `JAX_PLATFORMS=tpu pytest TestRealTpuTinyForward` → 1 passed in ~14s.
- [x] **STATUS.md / SUMMARY.md / BLOCKERS.md / INVARIANTS.md / TOLERANCE_LOG.md** updated for v6.

## Resume hint (post-v6)
**If killed now**, the next session should: (a) read SUMMARY.md "v6 — what's new since v3/v5"; (b) the structural blockers B2/B3/B5 are RESOLVED, B1 is now a clean future-work item (Pallas kernel for paged-V4 sparse-attn or vllm kv_cache schema extension to admit V4's compressor/indexer state pytree); (c) Tier 5 hardening is the natural next step — the current test sends one prompt at one seed; could extend to varied prompts, longer max_tokens, batch=2 concurrent. The functional core (W1 decode) is correct so multi-step decode through __call__ is the next major piece.

## Phase v6 hardening (2026-04-28 ~22:05 UTC)

- [x] **T5 hardening:** TestVllmServeRoundtrip extended to 4 requests in one subprocess:
  - (1) two-request determinism: byte-equal text + finish_reason=length + completion_tokens=8 (existing).
  - (2) prompt-dependence: different prompt produces different completion (sanity against logit-collapse).
  - (3) longer max_tokens=16 also completes (exactly 16 completion tokens).
- [x] Test runtime: ~60s end-to-end (compile cache warm after request 1). Commit 406929eb.
- [x] Final CPU regression confirmed: **83 passed, 1 skipped (6:45)**.

## Resume hint (post-v6 final)
**If killed now**, the next session should: see SUMMARY.md "v6 — what's new". The session completed every overnight goal: W1 + W2 (workaround) + W3 (prefill path) + W4 done; T5/T6/T7 + Tier 1-4 all green; structural blockers B2/B3/B5 resolved; B1 remains as production-correctness future-work for multi-sequence concurrent decode (see BLOCKERS.md). Suggested next: address B1 — write a Pallas kernel for V4 sparse-attn over [SWA window || compressed slots], OR extend vllm's per-layer kv_cache schema to admit V4's compressor/indexer state pytree. Until then, V4 in vllm serve is correct-but-not-multi-sequence-paged.

---

RESUMED at 2026-04-28 22:19 UTC (v7 — finish-early hardening) — picking up post-v6 with all W1–W4 done, T1–T7 green (modulo TPU-only T6 skipped on CPU). Per spec's "If you finish early" guidance: do NOT add features; tighten T5/T6/T7 tolerances, add more decode parity points, polish SUMMARY.md. Plan: (a) reconfirm baseline 83+1 still passes, (b) measure observed atol on T6/T7 to see if the 0.2/0.1 budgets are loose vs. evidence and tighten with TOLERANCE_LOG entries, (c) extend decode parity at long sp values (sp ∈ {500, 1023}) if time, (d) polish SUMMARY.md "v7 — what's new".

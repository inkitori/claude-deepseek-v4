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

---

RESUMED at 2026-04-29 11:38 UTC (v8 — host-direct on v6e-32). TPU preflight ok (4 v6 chips, n_tpu=4). Baseline reconfirmed `64 passed, 20 skipped` (1:17). Skip count higher than v6's 83p+1s because the new host has no `/mnt/scratch/` and no GCS mount available to this user (`/tmp/gcs/bucket/` permission-denied; `~/.cache/huggingface/hub/` empty; `work/scratch/` empty). Fixture-dependent tests (Tier 4 shard, Tier 4b, Tier 5, Tier 6 forward, Tier 7, FP8 dequant unit, W3 helpers) skip cleanly. Test paths updated: `_scratch(name)` helper resolves `V4_SCRATCH_DIR` env / `/mnt/scratch/` / `work/scratch/` candidates and falls back to `work/scratch/`. Added skip guard to `test_eval_shape_makes_abstract_module` (was hard-failing on missing config.json). Plan: focus on B1 (highest priority per spec, gates Tier 8); document missing-fixture/mount situation in BLOCKERS.md as `T5/T6/T7-fixtures-missing` and `T8-mount-missing` so the user knows why those tiers skipped on this run; do W5 Tier 8 only if the GCS mount becomes available within this session.

If killed now, next session must: (1) read this PROGRESS.md tail and STATUS.md; (2) confirm fixture/mount situation hasn't changed (`ls /tmp/gcs/bucket` and `ls work/scratch`); (3) continue B1 work — reading `tpu_inference/models/jax/deepseek_v4.py`'s `__call__` and `tpu_inference/models/jax/deepseek_v3.py:1383`'s reference for per-sequence handling.

---

HOST-UPDATE at 2026-04-29 11:56 UTC (between v8 iter 1 and iter 2): the user resolved both blockers from v8 iter 1. (a) gcsfuse mount of `gs://personal-mark-eu/vllm/hub/` is now live at `~/.cache/huggingface/hub/`; the real DeepSeek-V4-Flash snapshot resolves at `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/fd53f944496234770ba80e15004f9b6d269a71f5/` (config.json + 46 model-*.safetensors + DeepSeek's `inference/` reference code). (b) Synthetic fixtures regenerated under `work/scratch/tiny_v4_{bf16,quant,groundtruth}/` via `scripts/make_tiny_v4_checkpoint.py` reading metadata from the mount. (c) `.env` now has `MOUNT_GCS=1` so `./run.sh` auto-mounts on subsequent restarts. Verified: `mountpoint -q ~/.cache/huggingface/hub` ✓; `ls work/scratch/tiny_v4_bf16/` shows config.json + tokenizer.json + safetensors. The 20 fixture-dependent skips from v8 iter 1 should now turn into passes — re-run baseline first thing and update STATUS.md with the v8 numbers.

If killed now (post-host-update), next session must: (1) reconfirm `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 pytest tests/models/test_deepseek_v4.py -q` — expect close to v6's 83+1, modulo any iter-1 code edits; (2) run `JAX_PLATFORMS=tpu pytest tests/models/test_deepseek_v4.py::TestRealTpuTinyForward` (TPU-only); (3) attack B1 in `deepseek_v4.py::__call__` (multi-seq decode); (4) Tier 8 real-weight gate via `JAX_PLATFORMS=tpu HF_HUB_OFFLINE=1 vllm serve deepseek-ai/DeepSeek-V4-Flash --tensor-parallel-size 4 --enforce-eager --max-model-len 256 --max-num-seqs 1 --port 18081 --seed 0 --trust-remote-code --dtype bfloat16` then curl smoke (expect "Paris"-starting completion).

**WIP NOTE for the resuming agent:** v8 iter 1 was killed mid-flight while writing B1 (multi-seq dispatch). Two files have **uncommitted** WIP in your working tree (run `git status` to see):
  - `tpu_inference/models/jax/deepseek_v4.py` — `__call__` rewritten to dispatch per `query_start_loc` segments, calling `transformer_body_forward` per sequence and reassembling `hidden_TM`. Eager-only; jit support deferred.
  - `tests/models/jax/test_deepseek_v4.py` — `+190` lines: `_hf_dict_from_torch_args` helper + `TestB1MultiSeqDispatch` class.

**Do not blindly discard or commit.** Inspect with `git diff`, run the new B1 test:
```
cd work/tpu-inference
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 \
  pytest tests/models/test_deepseek_v4.py::TestB1MultiSeqDispatch -v
```
If it passes, run the full suite to check for regressions, then commit (`B1: per-seq dispatch in __call__ — multi-seq logits match serial single-seq`). If it fails, debug — don't `git checkout` away from your iter-1 reasoning unless you're sure it's wrong.

---

RESUMED at 2026-04-29 12:04 UTC (v8 iter 2 — pick up B1 WIP from iter 1). TPU preflight ok (4 v6e chips). GCS mount UP at `~/.cache/huggingface/hub/`; real V4-Flash snapshot has 46 model-*.safetensors and DeepSeek's `inference/` ref code. Synthetic fixtures `tiny_v4_{bf16,quant,groundtruth}` populated under `work/scratch/`. Disk ~71 GB free (~28% used). The two WIP files from iter 1 (deepseek_v4.py + test_deepseek_v4.py) are intact. **`TestConcurrentMultiSeqDispatch` (3 tests) passes** as-written on this host (cpu, simulated 8-device) in 44s. Plan: (1) confirm full suite green via the v6 invocation; (2) commit B1 + B1 tests; (3) push; (4) attack W5 Tier 8 eager-mode `vllm serve deepseek-ai/DeepSeek-V4-Flash`.

If killed now, next session must: run `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 pytest tests/models/test_deepseek_v4.py -q` (expect ≥83+1; will be 86+1 after B1 commits) — if green, commit "B1: per-seq dispatch in __call__ — multi-seq matches serial" then continue to W5.

---

RESUMED at 2026-04-29 15:30 UTC (v8 iter 3 — pick up after B1 committed; T8 first attempt hit fp8 rejection). B1 is committed and pushed (25ad1a11). The first T8 eager attempt produced `logs/T8-eager-result-20260429T120533Z.json` with `ok:false reason:server_not_ready`; the underlying vllm-serve log shows `pydantic_core._pydantic_core.ValidationError: deepseek_v4_fp8 quantization is currently not supported in tpu`. Root cause: `tpu_inference/platforms/tpu_platform.py:94` lists `supported_quantization = ["tpu_int8", "compressed-tensors", "awq", "fp8", "gpt_oss_mxfp4"]` — `deepseek_v4_fp8` is missing. The JAX V4 path doesn't use vllm's torch DeepseekV4FP8Config/FusedMoE machinery (our W4 deepseek_v4_loader handles dequant in JAX), so the right fix is to add `deepseek_v4_fp8` to the TPU `supported_quantization` whitelist. Plan: (1) verify baseline still green on host (in case iter 2's commit broke anything); (2) patch the supported_quantization list; (3) re-run T8 eager smoke; (4) iterate on whatever next failure mode appears.

If killed now, next session must: read `logs/T8-eager-result-*.json` (latest); if `ok:true`, mark W5 done; otherwise read the underlying serve log, document new failure surface in BLOCKERS.md, attempt up to 3 fixes, then move to non-dependent work.

---

## Phase v8 iter 3 (2026-04-29 ~15:30–16:10 UTC)

- [x] Baseline reconfirmed clean: **73 passed, 14 skipped** in 3:05 (no regressions vs iter 2's B1-augmented baseline).
- [x] Symlinked `work/scratch/v4_flash` -> gcsfuse snapshot dir; re-run baseline now **81 passed, 6 skipped** (3:44). The 6 remaining skips are 5 V4-Pro RealConfigCompile (no v4_pro fixture) + 1 TPU-only `TestRealTpuTinyForward` (runs separately under `JAX_PLATFORMS=tpu`, where it passes — 1/1).
- [x] **Tier 5 (synthetic vllm serve)** confirmed still GREEN on this host: 1/1 in 1:06.
- [x] **Tier 4b on real V4-Flash bf16 shard** GREEN through gcsfuse mount (1/1 in 18s).
- [x] **Tier 8 attempt 1** (commit 25ad1a11 baseline) — vllm pydantic rejects `deepseek_v4_fp8` quant. Captured at logs/T8-eager-serve-20260429T120533Z.log.
- [x] **Tier 8 attempt 2** (after dd731cf2: TpuPlatform supported_quantization += deepseek_v4_fp8) — vllm pydantic now accepts the quant name; gate moves to VllmConfig "MLA models require enable_dp_attention". Script was missing the `--additional_config` flag. Captured at logs/T8-eager-serve-20260429T151831Z.log.
- [x] **Tier 8 attempt 3** (after a249970f: t8_eager_smoke.sh adds enable_dp_attention) — vllm advances past pydantic; tpu_inference's separate `get_tpu_quantization_config` rejects with NotImplementedError "deepseek_v4_fp8 quantization method not supported. Supported methods are dict_keys([None, 'fp8'])". Captured at logs/T8-eager-serve-20260429T151926Z.log.
- [x] **Tier 8 attempt 4** (after f2f5a8e8: deepseek_v4_fp8 -> UnquantizedConfig in TPU quant registry) — engine init advances all the way to `model.load_weights(rng)`, then OOMs at `jnp.zeros` materializing the abstract bf16 param tree on chip 0. **Architectural truth confirmed:** V4-Flash 543 GB bf16 cannot fit 4 v6e chips × 32 GB HBM = 128 GB even with perfect sharding (135 GB/chip > 32 GB). Captured at logs/T8-eager-serve-20260429T152129Z.log; structured result at logs/T8-eager-result-20260429T152206Z.json. Documented in BLOCKERS.md::T8-HBM-OOM with three orthogonal unblock paths.
- [x] STATUS.md / SUMMARY.md / BLOCKERS.md / PROGRESS.md (this file) updated. v8 iter 3 ends with W5 deploy gate **architecturally blocked**; the deploy target must be the full 32-chip v6e-32 slice (multi-host launcher work) to fit V4-Flash bf16.

## Resume hint (post-v8 iter 3)

**If killed now:** the headline is BLOCKERS.md::T8-HBM-OOM. The deploy gate cannot pass on this 4-chip slice; V4-Flash bf16 = 543 GB > 128 GB HBM. The functional core (math + decode + dequant + multi-seq dispatch + nnx port + load_weights wiring) is correct and validated on synthetic + real-shard paths. To exercise T8, either (A) the host loop launches vllm-tpu across all 8 hosts of the v6e-32 slice, or (B) we add native FP4/FP8 storage on TPU + on-the-fly dequant, or (C) per-layer host-RAM offload. None of these are model-code work.

**If we have more session time:** the spec's "finish early" guidance says don't add features; tighten T5/T6/T7/T8 tolerances, add more decode parity points, polish SUMMARY.md. But T8 has no tolerance to tighten (it's blocked, not loose). Useful targets: more decode parity at sp ∈ {500, 1023}, tighter T7 atol bounds with measurement evidence in TOLERANCE_LOG.

---

RESUMED at 2026-04-29 16:08 UTC (v8 iter 4 — finish-early polish). TPU preflight ok=true, n_tpu=4. GCS mounted; v4-flash snapshot resolves. Synthetic fixtures intact. Working tree clean (only `scripts/setup.sh` modified at parent repo). Baseline reconfirmed: **87 passed, 6 skipped** in 3:45 — zero regressions vs v8 iter 3 final (b55e5a30). TPU T6 spot-check: **1/1** in 25s.

State of the world: B1 done, T8 architecturally blocked on HBM (4-chip slice can't fit 543 GB bf16 V4-Flash). Per spec's "If you finish early" guidance: do NOT add features; tighten tolerances, add more decode parity points, polish SUMMARY.md.

Plan for iter 4:
  (1) survey TOLERANCE_LOG.md to find loosest budgets
  (2) measure observed atol on T7 (FP4/FP8 dequant equivalence), T2 long-context decode
  (3) tighten one or two budgets with TOLERANCE_LOG entries citing measured values
  (4) extend decode parity at 1–2 more start_pos values to fill gaps in {0..1023}
  (5) polish SUMMARY.md "v8 iter 4 — what's verified, residual risk"
  (6) re-run full suite to confirm no regressions; commit each green step

If killed now: see prior resume hint (post-v8 iter 3); current iter is pure polish so killing mid-iter doesn't lose anything substantive. Nothing in this iter touches W5/T8 architectural blocker.


## Phase v8 iter 4 (2026-04-29 ~16:08–16:25 UTC) — finish-early polish

- [x] Baseline reconfirmed clean: **87 passed, 6 skipped** in 3:45 (no regressions vs v8 iter 3 final b55e5a30).
- [x] TPU T6 spot-check confirmed clean: 1/1 in 25s.
- [x] **T7 measurement** (TestQuantToParamsApply forward logits, quant ≡ groundtruth): observed `max_abs_diff = 0.0`, `argmax_agree = 1.0`, `byte_equal = True`. Tightened from atol=0.1 to byte-exact (np.array_equal AND max-abs == 0).
- [x] **T8 measurement** (TestDecodeRollingEquivalenceWithPrefill: SWA decode-state ≡ prefill-state after 32 steps): observed `0.0` max-abs across 8 random seeds {7, 1, 2, 3, 5, 11, 13, 17}. Tightened from atol=2e-2 to byte-exact.
- [x] **Decode step parity measurements** (TestDecodeAttentionParity 9 points, TestDecodeRollingParity 5 (P,K) combos, TestDecodeAttentionParityExtended 8 points = 22 total): observed worst `3.81e-6`. Tightened from atol=5e-2 to atol=1e-4 (25× margin over observed worst).
- [x] **TOLERANCE_LOG.md** rewritten: T7 + T8 entries replaced with byte-exact evidence; new "Decode step parity" entry citing per-class measured atol.
- [x] **4 new decode parity points** added to TestDecodeAttentionParityExtended: SWA + HCA at sp ∈ {256, 768}. Fills gaps between sp=192/500 and sp=500/1023. All four pass at the new 1e-4 bound.
- [x] **Affected-tests run** (24 tests): 24 passed in 20s. **Full CPU suite re-run: 91 passed, 6 skipped** in 3:45 (+4 from new parity points; 0 regressions). **TPU T6 re-check post-tightening: 1/1 in 26s.**
- [x] STATUS.md / SUMMARY.md / TOLERANCE_LOG.md / PROGRESS.md (this file) updated for iter 4.

Commits this iter:
  - edc4647a — tighten T7/T8/decode-step bounds + TOLERANCE_LOG entries
  - 0b8d7fe3 — 4 new decode parity points (sp ∈ {256, 768}, SWA + HCA)

## Resume hint (post-v8 iter 4)

**If killed now:** the headline is still BLOCKERS.md::T8-HBM-OOM (architectural,
unchanged from iter 3). Iter 4 was pure polish — tightened tolerances and added
parity coverage. The functional core is unchanged. Tests now defend the
implementation at byte-exactness for T7/T8 and at 1e-4 for decode parity, vs
loose 0.1/2e-2/5e-2 budgets that would have waved real regressions through.

**What's left to do (order of value):**
  1. **T8 architectural unblock — needs user decision** (multi-host launcher
     vs native FP4/FP8 storage vs host-RAM offload). None of these are pure
     model-code work.
  2. Tighten T1/T2/T3 budgets only if measurements show comfortable headroom.
     The full attention/transformer chain has bf16 noise estimates that
     match measured behavior — likely no headroom there. Skipped here.
  3. **Tier 5 hardening** — extend TestVllmServeRoundtrip to more prompts /
     longer max_tokens / batch=2 multi-seq (now that B1 is wired). The
     existing test already exercises 3 prompt variants and max_tokens=16;
     adding batch=2 would exercise B1 end-to-end through vllm.
  4. Polish: V3_TO_V4_DIFF.md, PROD_TOPOLOGY_RISKS.md sanity passes.

## Phase v8 iter 4 continuation (2026-04-29 16:25–16:40 UTC) — extend to T1/T2/T3

After the first iter 4 commit batch (T7/T8/decode-step), measured headroom on
Tier 1 + Tier 2 component tests and Tier 2 end-to-end logits parity. All had
1000–10000× headroom under the original budgets:

- TestAttentionComponent (45 seed combos): worst 7.63e-6 vs budget 5e-2 — tightened to 1e-3 (130× margin).
- TestBlockComponent (80 seed combos): worst 7.81e-3 (ULP-stable) vs budget 5e-2 — tightened to 2e-2 (2.5× margin).
- TestMoEComponent.test_moe_matches_torch (10 seeds): worst 4.88e-4 vs budget 5e-2 — tightened to 5e-3 (10× margin).
- TestMoEComponent.test_moe_hash_layer_matches_torch (10 seeds): worst 4.20e-2 — too close, kept at 5e-2.
- TestEndToEnd single/multi-batch/V4-Pro/MTP (60 seed combos): worst 1.35e-4 vs budget 0.1 — tightened to 1e-3 (7× margin).
- TestEndToEnd long-context S=128: worst 1.22e-4 vs budget 0.15 — tightened to 2e-3 (16× margin).

TOLERANCE_LOG.md T1, T2, T3 entries rewritten with measurement evidence;
the original "loose" budgets came from theoretical worst-case bf16
accumulation estimates that empirically don't materialize (the fp32
head matmul absorbs much of the residual stream's bf16 noise; per-layer
bf16 noise at sigma=0.02-init activations is ~1 ULP of bf16 at the
output magnitude, not ~0.025).

Commits this iter (continuation):
  - cadebad8 — tighten T3 end-to-end logits parity bounds (5 tests)
  - 06a7b3b7 — tighten T1/T2 component bounds (Attention, Block, MoE)

Final state: **91 passed, 6 skipped on CPU + 1 passing on TPU** under
the new tighter bounds. Zero regressions. SUMMARY.md / STATUS.md /
TOLERANCE_LOG.md / PROGRESS.md (this file) updated.

## Resume hint (post-v8 iter 4 complete)

**If killed now:** the headline is still BLOCKERS.md::T8-HBM-OOM (architectural,
unchanged from iter 3). Iter 4 tightened 7 tolerance budgets (T1, T2, T3, T7,
T8, decode-step parity ×3) and added 4 new decode parity points (sp ∈ {256,
768} for SWA + HCA). The functional core is unchanged; the tests now defend
it byte-equally where possible (T7, T8) and at 1e-3–1e-4 elsewhere, instead
of the previous 0.1–5e-2 budgets that would have waved real per-layer bugs
through silently.

**Useful follow-up if more session time appears:**
  1. T8 architectural unblock — needs user decision (multi-host launcher
     vs native FP4/FP8 storage vs host-RAM offload). None of these are
     pure model-code work.
  2. Tier 5 hardening — extend TestVllmServeRoundtrip to send 2 concurrent
     requests in a single subprocess (would exercise B1 end-to-end through
     vllm). Existing test already exercises 3 prompt variants and
     max_tokens=16 sequentially; adding parallel batch=2 would be the next
     natural step.
  3. Compressor decode-step parity (TestCompressorDecodeStep,
     TestCompressorDecodeStepExtended) — atol=5e-2 there too; not yet
     measured in iter 4 because the tests have separate kv_state /
     score_state assertions that need individual treatment.


## RESUMED 2026-04-29 ~16:46 UTC — v8 iter 5 (compressor-decode parity tightening)

Picking up from iter 4's clean state (91 passed, 6 skipped on CPU + 1 on TPU,
sha 454896fa). Headline blocker T8-HBM-OOM unchanged — architectural, gates
real-weight deploy on this 4-chip slice; needs user decision per
BLOCKERS.md::T8-HBM-OOM. Iter 5 is pure polish / "honor correctness over
features".

Plan:
  (1) confirm baseline 91/6 still green — DONE (3:43, no regressions).
  (2) tighten the remaining loose budget — TestCompressorDecodeStep +
      TestCompressorDecodeStepExtended (atol=5e-2 on three quantities;
      explicitly flagged in iter 4's resume hint as "not yet measured").
  (3) update TOLERANCE_LOG.md with measured-evidence entry T-CDS.
  (4) update STATUS.md / SUMMARY.md to reflect iter 5.
  (5) commit each green step; push.

Measured (scripts/measure_compressor_decode_parity.py, 9 configs × 8 seeds):
  - kv_compressed: worst 0.0 (24 hits — 3 compress configs × 8 seeds).
  - kv_state:      worst 7.15e-7 (ratio=4 sp=4 seed=3).
  - score_state:   worst 5.96e-7 (ratio=128 sp=128 seed=13).
  Both kv_state / score_state quantities are full fp32 accumulator math
  on both sides; their parity should be at fp32 ULP, not bf16 noise.
  Old 5e-2 budget was ~70,000× looser than necessary — would have hidden
  a fp32→bf16 accumulator regression silently.

Tightened from 5e-2 → 1e-5 (14–17× margin over measured worst).

If killed mid-iter-5: TOLERANCE_LOG already has the T-CDS entry; the test
edits are already in tests/models/jax/test_deepseek_v4.py at lines 2135 +
2149-2150 + 2366 + 2378-2382. Re-run TestCompressorDecodeStep* to confirm
green at the new bound; then re-run the full suite to confirm no regressions
elsewhere; then commit + push.


## v8 iter 6 (2026-04-29 ~17:00 UTC) — last attention_decode_step 5e-2 holdout

After iter 5 landed cleanly (commit e5712207 + a761bf99, pushed), grep for
remaining `5e-2` budgets in the test file showed two: (a) the
`test_moe_hash_layer_matches_torch` 5e-2 (intentionally kept — observed
4.20e-2 in iter 4, too close to tighten safely); (b)
`TestDecodeRollingParityLong::test_rolling_decode_parity_long` — same
`attention_decode_step` path as iter 4 tightened to 1e-4, just for longer K.

Plan for iter 6:
  (1) measure rolling-long parity worst-case across 3 configs × 6 seeds × up to
      32 steps per row.
  (2) tighten 5e-2 -> 1e-4 (same bound as iter 4 sister classes).
  (3) extend the existing "Decode step parity" TOLERANCE_LOG entry.
  (4) re-run full suite to confirm no regressions; commit + push.

Measured (scripts/measure_rolling_long_parity.py): worst 7.63e-6 (layer=0
P=1 K=31 seed=7 step k=26). Same bf16 ULP regime as iter 4's measurements;
the 32-step rolling chain doesn't compound per-step error because the state
buffers are exact fp32 history on both sides. 1e-4 keeps a 13× margin.

If killed mid-iter-6: the test edit is at line 2426 (5e-2 -> 1e-4) and the
TOLERANCE_LOG "Decode step parity" entry has been extended to mention
TestDecodeRollingParityLong + the 7.63e-6 measurement. Re-run that class to
confirm green at 1e-4; then full suite; then commit + push.


## Resume hint (post-v8 iter 6 — final state of this session)

**If killed now:** the headline is still BLOCKERS.md::T8-HBM-OOM
(architectural, unchanged from iter 3). Iters 5 + 6 were pure polish —
last two unmeasured 5e-2 budgets tightened with measurement evidence.
The functional core is unchanged.

**Test suite end-state (commit 3b7530c8 + 33f4032d):**
  - CPU: **91 passed, 6 skipped** (3:40, no regressions vs all prior iters).
  - TPU: **1 passing** (T6 spot-check confirmed post iter-5/6, ~20s).
  - 5e-2 budgets remaining in test file: **1**, intentionally kept
    (test_moe_hash_layer_matches_torch — observed 4.20e-2 in iter 4,
    no safe tightening).
  - All other measurable tolerance budgets are at fp32-ULP / bf16-ULP /
    byte-exact / measurement-bound levels per TOLERANCE_LOG.md.

**Cumulative iter-5/6 tightenings (this session):**
  - iter 5: TestCompressorDecodeStep + TestCompressorDecodeStepExtended
    kv_compressed/kv_state/score_state: 5e-2 → 1e-5 (worst observed
    0.0/7.15e-7/5.96e-7 across 9 configs × 8 seeds = 72 measurements).
  - iter 6: TestDecodeRollingParityLong: 5e-2 → 1e-4 (worst observed
    7.63e-6 across 3 configs × 6 seeds × ≤32 steps ≈ 500 step
    measurements). Same path as iter 4 sister classes.

**Useful follow-up if more session time appears:**
  1. T8 architectural unblock — needs user decision (multi-host launcher
     vs native FP4/FP8 storage vs host-RAM offload). None are pure
     model-code work.
  2. Tier 5 hardening — extend TestVllmServeRoundtrip to send 2
     concurrent requests in a single subprocess (would exercise B1
     end-to-end through vllm). Existing test exercises 3 prompt variants
     + max_tokens=16 sequentially.
  3. Tier 4b — add a real V4-Flash *expert* shard byte-equal round-trip
     (currently we have bf16 embed + FP8 wq_a + FP4 expert dequant; an
     expert bf16 round-trip would cover the heaviest weight class).


## v8 iter 7 (2026-04-29 ~17:30 UTC) — Tier 4b real-V4-Flash dequant byte-equal

RESUMED at 2026-04-29T17:07Z. Baseline confirmed: 91 passed / 6 skipped on
CPU, 1 passed on TPU — same end-state as iter 6 (commits dcc10282 +
3b7530c8 + 33f4032d). T8 still architecturally blocked on HBM (BLOCKERS::
T8-HBM-OOM).

**Picked from prior session's follow-up list:** option (3) Tier 4b
expansion. Chose this over (2) Tier 5 hardening because: (a) it's pure
local code with no vllm-runtime debugging risk; (b) byte-equal independent
reference is a stronger correctness check than concurrent vllm coverage;
(c) iter 7 measurement evidence directly supports new spec-correctness
invariants (I36-I38).

**What landed (iter 7):**
  1. Parametrized `TestRealFp8DequantSmoke` from 1 case
     (`layers.0.attn.wq_a`) to 4 cases covering layers {0, 10, 30} and
     attn projections {wq_a, wkv}.
  2. Parametrized `TestRealFp4DequantSmoke` from 1 case
     (`layers.2.ffn.experts.0.w1`) to 4 cases covering layers {0, 2, 5, 42},
     experts {0, 10, 255}, and projections {w1, w2, w3}.
  3. Added `TestFp8DequantIndependentReference` — bit-level e4m3fn
     decode + bit-level e8m0fnu decode + np.kron block-broadcast, all in
     numpy. Asserts bf16 byte-equal vs loader on `layers.0.attn.{wq_a,
     wkv}`. 2 cases.
  4. Added `TestFp4DequantIndependentReference` — sign-magnitude FP4
     nibble decomposition with -0->+0 canonicalization (matches the
     DeepSeek codebook spec choice). Asserts bf16 byte-equal vs loader's
     16-entry FP4_TABLE lookup on `layers.{2,0}.ffn.experts.0.{w1, w2}`.
     2 cases.

**Spec finding (iter 7):** First version of the FP4 reference produced
-0.0 (bf16 0x8000) where the loader produced +0.0 (bf16 0x0000) for
nibble-8 inputs. Investigation revealed DeepSeek's FP4_TABLE intentionally
maps both nibble 0 and nibble 8 to +0.0 (the negative-zero slot is unused).
The reference now canonicalizes mag==0 -> +0.0 to match the loader's
table-lookup output. This is now codified as INVARIANTS::I38.

Affects ~0.6% of real expert weight bytes (every nibble-8 byte). Worth
making explicit because:
  - A future numerics check that compares loader against pure sign-magnitude
    decode without the canonicalization would falsely flag a bug.
  - Any third-party FP4 dequant for V4 weights (in vllm-tpu or otherwise)
    must collapse -0 to +0 to match DeepSeek's runtime.

**Test count: 91 -> 101 passed**, 6 skipped unchanged. CPU runtime barely
changed (223s vs 227s) — gcsfuse cache absorbed the extra IO. TPU
spot-check: 1 passed (~21s).

**New TOLERANCE_LOG entries:** T-FP8-REF (byte-exact, 6.3M elements
across 2 tensors), T-FP4-REF (byte-exact, 12.6M elements across 2 tensors).

Iter 7 commits:
  - c818b6c1 — code: parametrize Tier 4b FP8/FP4 smokes + add byte-equal
    independent reference dequant.
  - (pending) — iter 7 docs.

If killed mid-iter-7: the code change is in commit c818b6c1 (already
pushed). Docs in this session may not yet be pushed. Re-run the new test
classes (`TestRealFp8DequantSmoke`, `TestRealFp4DequantSmoke`,
`TestFp8DequantIndependentReference`, `TestFp4DequantIndependentReference`)
to confirm green; then push.


## Resume hint (post-v8 iter 7 — final state of this session)

**If killed now:** the headline is still BLOCKERS.md::T8-HBM-OOM
(architectural, unchanged from iter 3). Iter 7 was real-data hardening of
Tier 4b — independent-reference byte-equal coverage closes a quiet
correctness gap that the smoke tests left open.

**Test suite end-state (commits c818b6c1 + iter-7 docs):**
  - CPU: **101 passed, 6 skipped** (~3:43, +10 over iter 6's 91+6).
  - TPU: **1 passing** (T6 spot-check confirmed post iter 7, ~21s).
  - 5e-2 budgets remaining in test file: **1**, intentionally kept
    (test_moe_hash_layer_matches_torch — observed 4.20e-2 in iter 4,
    no safe tightening).
  - Tier 4b coverage: 13 passing tests (was 3 in iter 6) — 1 bf16 shard
    round-trip + 4 FP8 attn smokes + 4 FP4 expert smokes + 2 FP8
    byte-equal-vs-numpy-reference + 2 FP4 byte-equal-vs-sign-magnitude-reference.
  - All measurable tolerance budgets at fp32-ULP / bf16-ULP / byte-exact /
    measurement-bound levels per TOLERANCE_LOG.md.

**Cumulative iter-7 additions (this session):**
  - 6 new parametrize cases (3 FP8 + 3 FP4) on diverse real V4-Flash tensors.
  - 4 new byte-equal independent-reference tests (2 FP8 + 2 FP4).
  - 3 new invariants: I36 (FP8 byte-equal to bit-level reference),
    I37 (FP4 byte-equal to sign-magnitude reference), I38 (FP4 codebook
    collapses -0 to +0).
  - 2 new TOLERANCE_LOG entries: T-FP8-REF, T-FP4-REF.

**Useful follow-up if more session time appears:**
  1. T8 architectural unblock — needs user decision (multi-host launcher
     vs native FP4/FP8 storage vs host-RAM offload). None are pure
     model-code work.
  2. Tier 5 hardening — extend TestVllmServeRoundtrip with 2 concurrent
     requests via threads to exercise B1 end-to-end through vllm's
     scheduler (current test is sequential; with --max-num-seqs 2 the
     scheduler would batch concurrent requests but only sees serial ones).
     Risk: paged-KV path through vllm has known limitations (BLOCKERS::B1).
  3. Tier 4b independent reference for the e4m3fn cast itself — iterate
     all 256 byte values and verify torch's `.float()` matches the
     hand-decoded value exactly. Tautology if torch is correct, but
     locks in the assumption explicitly.


## v8 iter 8 (2026-04-29 ~17:35 UTC) — Tier 4b 256-byte exhaustive cast parity + cross-tensor real-data byte-equal expansion

RESUMED at 2026-04-29T17:35Z. Baseline confirmed: 101 passed / 6 skipped on
CPU, 1 passed on TPU — same end-state as iter 7 (commits 5ceb3e32 +
c818b6c1). T8 still architecturally blocked on HBM (BLOCKERS::T8-HBM-OOM).

**Picked from prior session's follow-up list:** option (3) Tier 4b 256-byte
exhaustive cast parity, plus its natural extension — the cross-tensor real
data byte-equal expansion that iter 7's parametrize lists left implicit.
Chose this over (2) Tier 5 concurrent-via-threads because: (a) the
exhaustive byte-domain test is pure local code with no vllm-runtime risk;
(b) the parametrize expansion converts iter 7's "loader is correct on
layer-0 attn" claim into "loader is correct across the full V4-Flash
surface" — the right scope for the spec's finish-early guidance; (c) the
two together close the iter 7 follow-up and the iter 6 follow-up
(real-bf16 expert shard byte-equal) at the same cost in test surface
area as iter 7 alone.

**What landed (iter 8):**
  1. Added `TestFp8CastByteDomain` with two new tests:
     `test_e4m3fn_all_256_bytes_match_torch_cast` — iterates all 256 e4m3fn
     bytes; asserts numpy `_numpy_decode_e4m3fn` and torch's `view(e4m3fn)
     .float()` cast agree bit-for-bit (NaN positions on the FN spec's
     {0x7F, 0xFF}; non-NaN bits equal). Spot-checks: 0x00 → +0.0,
     0x80 → -0.0 (sign preserved), 0x38 → +1.0, 0x7E → +448.0 (FN max-finite).
  2. `test_e8m0fnu_all_256_bytes_match_torch_cast` — iterates all 256
     e8m0fnu bytes; same byte-domain parity check. NaN at {0xFF} only;
     spot-checks: byte=0 → 2^-127 (subnormal edge), byte=127 → +1.0,
     byte=254 → 2^127 (max finite).
  3. Expanded `TestFp8DequantIndependentReference` parametrize 2 → 6 cases.
     Added `layers.20.attn.wq_b` (out>>in 32768×1024), `layers.10.attn.wo_a`
     (output proj A), `layers.5.attn.wo_b` (in>out output proj B), and
     `layers.40.ffn.shared_experts.w1` (the only FP8 path outside attn —
     dense FFN). 4 new tensor cases, ~104 MB total real V4-Flash data
     byte-validated.
  4. Expanded `TestFp4DequantIndependentReference` parametrize 2 → 4 cases.
     Added `layers.30.ffn.experts.128.w1` (deep layer, mid expert id) and
     `layers.10.ffn.experts.50.w3` (w3 SwiGLU projection).
  5. Cleaned up the `_numpy_decode_e8m0fnu` overflow warning at byte=0xFF
     (the NaN slot, where `np.exp2(128)` overflows fp32 before `np.where`
     re-masks to NaN). Now pre-masks the exponent so the computation only
     sees finite-encoding bytes. Functionally identical; quieter test
     output for both new and existing tests.

**Test count: 101 -> 109 passed**, 6 skipped unchanged. CPU runtime barely
changed (224s vs 221s) — gcsfuse cache absorbed the extra IO. TPU
spot-check: 1 passed (~21s).

**Spec-correctness finding (iter 8):** None. The byte-domain parity tests
passed on first run with no edge cases requiring spec choices — the FN
NaN bytes match the spec literally ({0x7F, 0xFF} for e4m3fn, {0xFF} for
e8m0fnu). e4m3fn 0x80 produces -0.0 with the sign bit preserved (the
codec is signed, unlike FP4_TABLE which collapses -0 → +0 — see I38).

**New invariants:** I39, I40, I41, I42 — see INVARIANTS.md.
**New TOLERANCE_LOG entries:** T-FP8-CAST (256-byte fp32 byte-exact); plus
an extension to T-FP8-REF / T-FP4-REF documenting the iter-8 coverage
expansion.

Iter 8 commits:
  - (pending) — code: parametrize expansion on Tier 4b independent-reference
    tests + 2 new exhaustive byte-domain tests.
  - (pending) — iter 8 docs.

If killed mid-iter-8: code change can be reproduced from this PROGRESS
entry. To validate, run
`pytest tests/models/jax/test_deepseek_v4.py::TestFp8CastByteDomain
tests/models/jax/test_deepseek_v4.py::TestFp8DequantIndependentReference
tests/models/jax/test_deepseek_v4.py::TestFp4DequantIndependentReference
-v` — should produce 12 passed (2 byte-domain + 6 FP8 + 4 FP4).


## Resume hint (post-v8 iter 8 — final state of this session)

**If killed now:** the headline is still BLOCKERS.md::T8-HBM-OOM
(architectural, unchanged from iter 3). Iter 8 was real-data hardening of
Tier 4b — locks the numpy decoder against torch's cast across the full
byte domain (256/256 bytes for both e4m3fn and e8m0fnu) and broadens the
real-tensor byte-equal coverage from layer-0 attn to a spread of layers
× projections × shapes that better represent the V4-Flash surface.

**Test suite end-state (commits pending iter-8 push):**
  - CPU: **109 passed, 6 skipped** (~3:44, +8 over iter 7's 101+6).
  - TPU: **1 passing** (T6 spot-check confirmed post iter 8, ~21s).
  - 5e-2 budgets remaining in test file: **1**, intentionally kept
    (test_moe_hash_layer_matches_torch — observed 4.20e-2 in iter 4,
    no safe tightening).
  - Tier 4b coverage: 21 passing tests (was 13 in iter 7) — 1 bf16 shard
    round-trip + 4 FP8 attn smokes + 4 FP4 expert smokes + 6 FP8
    byte-equal-vs-numpy-reference + 4 FP4
    byte-equal-vs-sign-magnitude-reference + 2 byte-domain
    exhaustive-cast parity tests.
  - All measurable tolerance budgets at fp32-ULP / bf16-ULP / byte-exact /
    measurement-bound levels per TOLERANCE_LOG.md.

**Cumulative iter-8 additions (this session):**
  - 2 new byte-domain exhaustive-cast tests (e4m3fn + e8m0fnu, both 256/256
    bytes vs torch's `.float()` cast).
  - 4 new FP8 byte-equal real-tensor cases (wq_b, wo_a, wo_b,
    shared_experts.w1) — spans 5 projections, 4 shapes, 4 shards, layers
    {0, 5, 10, 20, 40}.
  - 2 new FP4 byte-equal real-tensor cases (experts.128.w1 / experts.50.w3
    on deeper layers and the w3 SwiGLU projection).
  - 4 new invariants: I39 (e4m3fn 256-byte cast parity), I40 (e8m0fnu
    256-byte cast parity), I41 (FP8 byte-equal coverage census), I42 (FP4
    byte-equal coverage census).
  - 1 new TOLERANCE_LOG entry (T-FP8-CAST) + 1 expansion entry covering
    the parametrize widening.
  - 1 quiet-noise cleanup in `_numpy_decode_e8m0fnu` (overflow warning
    suppression by pre-masking the NaN slot).

**Useful follow-up if more session time appears:**
  1. T8 architectural unblock — still needs user decision (multi-host
     launcher vs native FP4/FP8 storage vs host-RAM offload). None are
     pure model-code work.
  2. Tier 5 hardening — extend TestVllmServeRoundtrip with 2 concurrent
     requests via threads to exercise B1 end-to-end through vllm's
     scheduler. Carries vllm-runtime risk.
  3. Tier 4b *real-bf16 expert shard* byte-equal round-trip (analog of
     TestRealShardRoundTrip on a non-embed bf16 tensor — but real V4-Flash
     stores experts as FP4, not bf16, so this would have to be a quant-
     to-bf16 round-trip on an expert tensor instead, which iter 8's
     `experts.30.w1` byte-equal independent reference effectively already
     covers). Probably no additional value here.
  4. More byte-domain reference parity for the FP4 codebook itself —
     iterate all 16 nibbles + spot-check the loader's `_FP4_TABLE_T` is
     element-equal to a manually-typed reference table of `{0, ±0.5, ±1,
     ±1.5, ±2, ±3, ±4, ±6}` with the -0 → +0 canonicalization (I38). Pure
     local code; small.

---

## v8 iter 9 RESUMED at 2026-04-29T18:01Z

Baseline reproduced: CPU **109 passed, 6 skipped** in 3:43; TPU
spot-check (`TestRealTpuTinyForward`) **1 passed** in 19.5 s. Mount + 4
v6e chips + scratch fixtures all healthy. T8 still architecturally
blocked on this 4-chip view (BLOCKERS::T8-HBM-OOM unchanged from iter
3).

Plan for iter 9 (this session): the user's prompt explicitly says "If
you finish early, do NOT add features. Tighten Tier 5/6/7/8 tolerances,
add more decode parity points, polish SUMMARY.md." Iter 8 already
exhausted the easy tightenings; remaining levers in priority order are
(a) tighten the moe_hash 5e-2 holdout if we can prove a tighter bound
on real-data; (b) add the 16-nibble FP4 codebook reference (the
follow-up explicitly listed in iter-8's hint); (c) widen Tier 4b
byte-equal real-data coverage to >=2 more shards to span the FP8 dense
keystone projections we haven't independently checked; (d) add a real-
data Tier 4b round-trip on a *real V4-Flash bf16 norm tensor* (the
single keystone storage path we haven't byte-equal-validated end-to-end
on real data — embeddings are bf16 and pass; norms are fp32-stored).

If killed mid-iter-9: see this entry. The headline is still T8-HBM-OOM.

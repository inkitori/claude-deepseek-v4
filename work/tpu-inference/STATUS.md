# DeepSeek V4 v8 status (host-direct on v6e-32; iter 7 polish — Tier 4b expanded with byte-equal independent reference dequant)
TPU preflight: ok (4 v6e chips, logs/tpu-preflight.log)
Host: TPU v6e-32 single-VM (4 local chips of 32 GB HBM each = 128 GB total HBM,
  708 GB host RAM). No docker. Real V4-Flash weights mounted via gcsfuse at
  `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/fd53f944.../`
  (config.json + 46 model-*.safetensors). Synthetic fixtures regenerated under
  `work/scratch/{tiny_v4_bf16,tiny_v4_quant,tiny_v4_groundtruth}`. v8 iter 3
  added a symlink `work/scratch/v4_flash` -> the gcsfuse snapshot dir so the
  Tier 3/4b real-config + real-shard tests resolve.

Latest passing tier: T1-T7 + B1 multi-seq dispatch + T4b on real V4-Flash
  bf16/FP8/FP4 + decode parity at sp ∈ {1, 4, 7, 8, 9, 16, 32, 64, 128, 192,
  256, 500, 768, 1023}, all under tightened tolerances; iter 5 added tightened
  compressor-decode-step parity (atol=1e-5); iter 6 tightened
  TestDecodeRollingParityLong (5e-2 -> 1e-4); iter 7 expanded Tier 4b
  real-V4-Flash dequant coverage from 2 single-tensor smokes to 8 parametrized
  smokes + 4 byte-equal independent-reference checks. T8 (deploy gate) BLOCKED
  on HBM OOM (architectural; see BLOCKERS.md::T8-HBM-OOM).

Tier 1: 25/25  (T1 attention: 5e-2 -> 1e-3; T2 block: 5e-2 -> 2e-2;
  T2 MoE: 5e-2 -> 5e-3; MoE-hash kept at 5e-2 — all tightenings backed
  by measured worst-case in TOLERANCE_LOG.md)
Tier 2: 8/8    (T3 logits parity: 0.1 -> 1e-3; long-context: 0.15 -> 2e-3)
Tier 2 hardening (v3 + v8 long-context + iter 4 256/768): 19/19 — 11 from v3,
  4 from v8 iter 3 (sp ∈ {500, 1023}), 4 new in v8 iter 4 (sp ∈ {256, 768}
  SWA + HCA), all at the tightened atol=1e-4 bound (was 5e-2).
Tier 3: 10/10  (V4-Flash full + 2-layer compile; V4-Pro skipped — no v4_pro fixture)
Tier 4: 2/2    (HF->JAX name mapping)
Tier 4b: 13/13 (was 3/3 in iter 6) — real V4-Flash byte-equal bf16 shard +
  4 FP8 attn smokes (was 1) + 4 FP4 expert smokes (was 1) + 2 FP8
  byte-equal-vs-numpy-reference + 2 FP4 byte-equal-vs-sign-magnitude-reference,
  all on the gcsfuse-mounted snapshot. The reference tests catch FP8/FP4 spec
  divergence at the bit level — see TOLERANCE_LOG::T-FP8-REF / T-FP4-REF
  and INVARIANTS::I36-I38.
Tier 5: 1/1    (vllm serve /v1/completions byte-equal — synthetic tiny_v4_bf16)
Tier 6: 1/1    (TPU-only — `JAX_PLATFORMS=tpu pytest TestRealTpuTinyForward`)
Tier 7: 1/1    (forward on tiny_v4_quant ≡ forward on tiny_v4_groundtruth —
  v8 iter 4 tightened 0.1 -> byte-exact, measured 0.0)
Tier 2 v8 (B1 multi-seq dispatch): 3/3 — concurrent_decode_two_seqs,
  single_seq_via_metadata_matches_no_metadata, three_seqs_concurrent.
Compressor decode-step parity (iter 5): 9/9 at the new atol=1e-5
  (TestCompressorDecodeStep 6 + TestCompressorDecodeStepExtended 3). Was 5e-2.
Tier 8 deploy gate: BLOCKED — `vllm serve deepseek-ai/DeepSeek-V4-Flash` OOMs
  during load_weights at HBM allocation time. V4-Flash bf16 = 543 GB; this
  4-chip slice has 128 GB total HBM. Math doesn't close even with perfect
  sharding. Captured failure at logs/T8-eager-serve-20260429T152129Z.log;
  result summary at logs/T8-eager-result-20260429T152206Z.json. Three
  orthogonal unblocks documented in BLOCKERS.md::T8-HBM-OOM.

B1 multi-seq: done — `DeepseekV4ForCausalLM.__call__` extracts per-seq segments
  from `attention_metadata.query_start_loc` and dispatches each through
  `transformer_body_forward` independently. Eager-only Python loop;
  jit-compiled multi-seq remains future work but is not gated by Tier 8 with
  `--enforce-eager`. Three regression tests in TestConcurrentMultiSeqDispatch.
W5 deploy gate: BLOCKED on HBM topology — see Tier 8 above and BLOCKERS::T8-HBM-OOM.
  The deploy gate cannot pass on a 4-chip slice; the user's deployment target
  must be the full 32-chip v6e-32 slice (8 hosts) to fit V4-Flash bf16.

Tolerance tightenings (cumulative through iter 7 — every entry has measured-worst evidence in TOLERANCE_LOG.md):
  - T1 attention prefill (3 tests): 5e-2 -> 1e-3 (worst 7.63e-6 / 45 seeds; 130x margin).
  - T2 block forward (4 tests): 5e-2 -> 2e-2 (worst 7.81e-3 / 80 seeds; 2.5x margin).
  - T2 MoE forward: 5e-2 -> 5e-3 (worst 4.88e-4 / 10 seeds; 10x margin).
  - T3 logits parity (5 tests): 0.1 -> 1e-3 (worst 1.35e-4 / 60 seeds; 7x margin).
  - T3 long-context: 0.15 -> 2e-3 (worst 1.22e-4; 16x margin).
  - T7 quant≡groundtruth logits: 0.1 -> byte-exact (np.array_equal). Measured 0.0.
  - T8 SWA decode-state ≡ prefill-state: 2e-2 -> byte-exact. Measured 0.0 / 8 seeds.
  - Decode step parity (3 test classes, 26 points): 5e-2 -> 1e-4 (worst 3.81e-6; 25x margin).
  - iter 5: Compressor decode-step parity (kv_compressed/kv_state/score_state, 9 configs × 8 seeds):
    5e-2 -> 1e-5 (worst 0.0/7.15e-7/5.96e-7 — fp32 ULP floor; 14–17x margin).
  - iter 6: TestDecodeRollingParityLong (3 configs × 6 seeds × up to 32 steps ≈ 500 measurements):
    5e-2 -> 1e-4 (worst 7.63e-6; 13x margin). Last 5e-2 holdout on the
    attention_decode_step codepath (the only 5e-2 left is the intentionally-kept
    moe_hash_layer test from iter 4, observed 4.20e-2 — too close to tighten).
  - **iter 7: Tier 4b real-V4-Flash dequant byte-equal coverage** — independent
    numpy references (bit-level e4m3fn/e8m0fnu decode + np.kron block scale;
    sign-magnitude FP4 nibble decomposition with -0->+0 canonicalization) match
    the loader byte-equally on `layers.0.attn.{wq_a,wkv}` (FP8) and
    `layers.{2,0}.ffn.experts.0.{w1,w2}` (FP4). 4 new byte-equal cases. No
    tolerance tightening (these were never tolerance-bounded — the smoke tests
    only verified plausibility); the reference tests upgrade them to
    spec-correctness. See TOLERANCE_LOG::T-FP8-REF / T-FP4-REF.

New invariants from iter 7 (INVARIANTS.md):
  - I36 — FP8 dequant byte-equal to bit-level reference on real V4-Flash.
  - I37 — FP4 dequant byte-equal to sign-magnitude reference on real V4-Flash.
  - I38 — DeepSeek's FP4 codebook collapses nibble 8 to +0.0 (not -0.0).

Patches committed this iter (v8 iter 7):
  - (pending commit): iter 7 docs (STATUS/SUMMARY/PROGRESS/INVARIANTS/TOLERANCE_LOG)
  - c818b6c1 v8 iter 7 polish: parametrize Tier 4b FP8/FP4 smokes + add byte-equal independent reference dequant
  - dcc10282 v8 iter 6 final: resume hint documents end-state (91+1/6 skipped)
  - 3b7530c8 v8 iter 6 docs: STATUS/SUMMARY/PROGRESS reflect rolling-long tightening
  - 33f4032d v8 iter 6 polish: tighten TestDecodeRollingParityLong (5e-2 -> 1e-4)
  - a761bf99 v8 iter 5 docs: STATUS/SUMMARY/PROGRESS reflect compressor-decode tightening
  - e5712207 v8 iter 5 polish: tighten compressor decode-step parity (5e-2 -> 1e-5)

Full CPU run target: `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 pytest tests/models/jax/test_deepseek_v4.py`
  -> **101 passed, 6 skipped** at iter 7 bounds (was 91+6 in iters 4-6; +10 new
  tests in iter 7: 6 from parametrize expansion of FP8/FP4 smokes + 4
  byte-equal independent-reference cases).
  Skipped: 5 V4-Pro RealConfigCompile tests (no v4_pro fixture on this host)
  + 1 TPU-only forward (TestRealTpuTinyForward, needs JAX_PLATFORMS=tpu).
TPU spot-check target: `JAX_PLATFORMS=tpu pytest tests/models/jax/test_deepseek_v4.py::TestRealTpuTinyForward`
  -> 1 passed (~21 s). Confirmed post iter 7 — iter 7 only touches Tier 4b
  CPU-side dequant tests, so no impact on T6.

If killed now, next session must:
  (1) read BLOCKERS.md::T8-HBM-OOM end-to-end (it's still the headline);
  (2) confirm with the user whether the deploy target is the full 32-chip
      v6e-32 slice or this 4-chip view — if 4-chip, T8 is architecturally
      infeasible without options (B) or (C) in BLOCKERS;
  (3) if the answer is "full slice", the next unblock is launcher work in
      the host loop (multi-host vllm-tpu coordination), NOT model-code work;
  (4) if "stay on 4 chips and make it fit", begin design of native FP4/FP8
      TPU storage (B) — start in tpu_inference/models/jax/deepseek_v4_loader.py.

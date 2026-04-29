# DeepSeek V4 v8 status (host-direct on v6e-32; iter 4 polish — tightened tolerances + 4 new parity points)
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
  256, 500, 768, 1023}. T8 (deploy gate) BLOCKED on HBM OOM (architectural;
  see BLOCKERS.md::T8-HBM-OOM).

Tier 1: 25/25
Tier 2: 8/8
Tier 2 hardening (v3 + v8 long-context + iter 4 256/768): 19/19 — 11 from v3,
  4 from v8 iter 3 (sp ∈ {500, 1023}), 4 new in v8 iter 4 (sp ∈ {256, 768}
  SWA + HCA), all at the tightened atol=1e-4 bound.
Tier 3: 10/10  (V4-Flash full + 2-layer compile; V4-Pro skipped — no v4_pro fixture)
Tier 4: 2/2    (HF->JAX name mapping)
Tier 4b: 3/3   (real V4-Flash byte-equal bf16 shard + FP8 + FP4 dequant smokes
  on the gcsfuse-mounted snapshot)
Tier 5: 1/1    (vllm serve /v1/completions byte-equal — synthetic tiny_v4_bf16)
Tier 6: 1/1    (TPU-only — `JAX_PLATFORMS=tpu pytest TestRealTpuTinyForward`)
Tier 7: 1/1    (forward on tiny_v4_quant ≡ forward on tiny_v4_groundtruth —
  v8 iter 4 tightened 0.1 -> byte-exact, measured 0.0)
Tier 2 v8 (B1 multi-seq dispatch): 3/3 — concurrent_decode_two_seqs,
  single_seq_via_metadata_matches_no_metadata, three_seqs_concurrent.
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

Tolerance tightenings (v8 iter 4, evidence in TOLERANCE_LOG.md):
  - T7 quant≡groundtruth logits: 0.1 -> byte-exact (np.array_equal). Measured 0.0.
  - T8 SWA decode-state ≡ prefill-state: 2e-2 -> byte-exact. Measured 0.0 across 8 seeds.
  - Decode step parity (3 test classes, 22 points): 5e-2 -> 1e-4. Worst measured 3.81e-6.

Patches committed this iter (v8 iter 4):
  - 0b8d7fe3 v8 iter 4: 4 new decode parity points (sp ∈ {256, 768}, SWA + HCA)
  - edc4647a v8 iter 4 polish: tighten T7 + T8 + decode-step parity bounds

Full CPU run target: `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 pytest tests/models/test_deepseek_v4.py`
  -> **91 passed, 6 skipped (3:45)**. Up from iter 3's 87+6; +4 net from iter 4
  (4 new decode parity points; tightenings reused existing test bodies).
  Skipped: 5 V4-Pro RealConfigCompile tests (no v4_pro fixture on this host)
  + 1 TPU-only forward (TestRealTpuTinyForward, needs JAX_PLATFORMS=tpu).
TPU spot-check target: `JAX_PLATFORMS=tpu pytest tests/models/test_deepseek_v4.py::TestRealTpuTinyForward`
  -> 1 passed (~26 s).

If killed now, next session must:
  (1) read BLOCKERS.md::T8-HBM-OOM end-to-end (it's still the headline);
  (2) confirm with the user whether the deploy target is the full 32-chip
      v6e-32 slice or this 4-chip view — if 4-chip, T8 is architecturally
      infeasible without options (B) or (C) in BLOCKERS;
  (3) if the answer is "full slice", the next unblock is launcher work in
      the host loop (multi-host vllm-tpu coordination), NOT model-code work;
  (4) if "stay on 4 chips and make it fit", begin design of native FP4/FP8
      TPU storage (B) — start in tpu_inference/models/jax/deepseek_v4_loader.py.

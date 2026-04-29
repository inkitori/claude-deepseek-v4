---
name: tpu-inference DeepSeek V4 status
description: DeepSeek V4 JAX implementation lives as a git subtree at work/tpu-inference inside ~/claude-deepseek-v4. As of 2026-04-29 the host moved from a v6e-4 docker setup to a v6e-32 host-direct setup; deploy goal is real V4-Flash weights via vllm serve from a gcsfuse-mounted GCS bucket.
type: project
---

V4 implementation is in `work/tpu-inference/` (git-subtree of the parent repo, no own .git). Built across v1–v7 sub-sessions inside a docker sandbox on a prior v6e-4 host. As of 2026-04-29 the user migrated to a v6e-32 host with no docker; harness was rewritten host-direct (run.sh + scripts/{setup,preflight,loop,mount_gcs}.sh).

**Why:** User's deploy goal is `vllm serve deepseek-ai/DeepSeek-V4-Flash` running directly on this v6e-32 TPU host with the real ~160 GB FP4/FP8 checkpoint loaded from GCS via gcsfuse — production-shape, not a tiny-fixture test.

**How to apply:** When user asks about V4 status: math is correct (84/84 tiny-fixture tests pass across 7 tiers); known limitation B1 (multi-sequence concurrent decode collapses input_ids into a single mega-sequence) was the highest-priority unfix at v7-end and the gating bug for Tier 8 (real-weight deploy). Single-seq vllm serve worked synthetically; multi-seq does not. The new prompt.md targets B1 + Tier 8 first.

**Hardware now:** v6e-32 slice, but each VM only sees its 4 local chips (`TPU_HOST_BOUNDS=1,1,1`, `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`). Without those env vars, `jax.devices('tpu')` hangs >60 s on absent peers. Single-VM TP=4 is the deploy shape; cross-host Ray is not in scope yet.

**Layout note:** No `/mnt/scratch` on this host — everything that the prior prompt referenced under `/mnt/scratch/...` now lives under `work/scratch/...`. Real weights live under `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/` only after `scripts/mount_gcs.sh` runs (requires `MOUNT_GCS=1` + `GCS_BUCKET` + `GCS_ONLY_DIR` in `.env`).

**Test command:** `cd work/tpu-inference && JAX_PLATFORMS=tpu pytest tests/models/test_deepseek_v4.py -v` from the host venv (`work/vllm_env/bin/activate`). On CPU-only smoke: `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 pytest ...` (TPU-only test then skipped).

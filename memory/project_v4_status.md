---
name: tpu-inference DeepSeek V4 status
description: Overnight session built a JAX DeepSeek V4 implementation on the `deepseek-v4` branch in work/tpu-inference. As of 2026-04-28 the JAX core math is fully correct vs PyTorch reference, but concurrent multi-sequence vllm serving is broken (B1 limitation).
type: project
---

V4 implementation is on branch `deepseek-v4` in `work/tpu-inference/`. Built across 7 sub-sessions (v1-v7) by previous Claude instances inside a docker sandbox; testable from host via `sudo docker run claude-overnight` image (UID 2010).

**Why:** User wants to run real V4-Flash weights on TPU.
**How to apply:** When user asks about V4 status, the math is correct (84/84 tests pass on tiny fixture covering all attention flavors), but B1 (multi-sequence concurrent decode through `__call__` collapses input_ids into a single mega-sequence) is unfixed. Single-sequence vllm serve works; concurrent multi-user serving does not.

Test command: `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 pytest tests/models/test_deepseek_v4.py` inside the docker (83 pass + 1 TPU-only skipped on CPU; the 1 TPU-only passes when run with `JAX_PLATFORMS=tpu`).

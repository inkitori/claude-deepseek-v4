---
name: V4-Flash full-slice deploy session (2026-04-29)
description: In-flight state of the v6e-32 TP=32 deploy — streaming/sharded loader written, two bugs fixed, currently iterating on load-time observability + Ray cleanup hygiene.
type: project
---

Branch: `main` of `~/claude-deepseek-v4`. Worker 0 = `10.164.0.41`. Workers
1–7 = `10.164.0.{22,35,36,39,45,18,30}`. Single TPU slice `v6e-32` =
8 hosts × 4 chips × 32 GB HBM = 992 GiB total. Real V4-Flash bf16 = 543 GB
(~17 GB / chip if sharded 32-way).

**Why:** User wants `vllm serve deepseek-ai/DeepSeek-V4-Flash` running
end-to-end on the full 32-chip slice. Codex handed off saying the OOM was
in `load_weights_from_dir`'s zero-tree materialization; deeper root cause
is V4 had no per-tensor sharding so each chip tried to hold the full
543 GB.

**How to apply:** Loader rewrite already committed locally (not pushed yet
as of 22:35 UTC):

  * `work/tpu-inference/tpu_inference/models/jax/deepseek_v4_loader.py`
    — added `iter_v4_safetensors_dequant_torch` (streaming generator),
    `place_torch_as_jax_sharded` (sharded jax.Array placement via
    `make_array_from_callback`), `pick_partition_spec` (auto-pick
    sharding axis — prefers `attn_dp`).
  * `work/tpu-inference/tpu_inference/models/jax/deepseek_v4.py` ::
    `load_weights_from_dir` — no longer materializes a zero param tree;
    streams weights one-at-a-time + assigns into the dataclass tree;
    progress logging every 200 placements.

Bugs fixed mid-session:
  1. `fp4_block` defaulted to None when HF config.json lacked
     `fp4_block_size`. Now defaults to 32 if `expert_dtype == "fp4"`,
     matching DeepSeek's `inference/model.py:18`.
  2. `_torch_to_numpy_view` + `_np_view_as_jax_dtype` did a raw
     `.view(dtype)` on numpy that collapsed shape when source/target
     itemsizes differed (bf16 torch → fp32 leaf halved the array
     length). Replaced with `_torch_to_numpy_preserve` returning a
     dtype-faithful numpy (via `ml_dtypes.bfloat16`) + `.astype(target)`
     for semantic conversion.

**Open issue at end of session (22:35 UTC):** Ray worker daemons taken
out by overly-broad pkill (see feedback_ray_cleanup.md). Rebuild from
CODEX_PLAN.md::"Ray setup" section. Then relaunch
`scripts/full_slice_v4_smoke.sh` and watch
`/home/enyouki/claude-deepseek-v4/logs/full-slice-v4-smoke-<TS>.log`.

**Update at 23:00 UTC:** Ray cluster rebuilt, smoke re-launched
(pid file at `logs/full-slice-v4-smoke.pid`, log
`logs/full-slice-v4-smoke-20260429T223800Z.log`). Loader is healthy:
mesh `attn_dp=32` correct, 23400/33000 tensors placed at 19/s on layer
29/43 by minute 20, no errors. ETA ~28 min total load.

Pushed to `origin/main` (`inkitori/claude-deepseek-v4`):
  * `97111371` — main streaming loader rewrite + 2 bug fixes
  * `fed58cfa` — opt-in prefetch-pool (env: V4_LOADER_PREFETCH_WORKERS,
    default 0; verified byte-identical vs sequential on tiny_v4_quant
    355 tensors). Forwarded to Ray workers via
    VLLM_RAY_EXTRA_ENV_VARS_TO_COPY (vLLM mechanism).
  * `48b3277a` — `scripts/full_slice_v4_smoke_check.sh`: post-startup
    polls /v1/models, fires the deterministic Paris completion twice,
    asserts byte-identical + contains "Paris".

For retries that need a faster load, set
`V4_LOADER_PREFETCH_WORKERS=4..8` before invoking the smoke script.

SSH to remote git: key at `~/.ssh/id_ed25519`, passphrase known to user.

**Performance concern:** First successful run (before pkill killed Ray)
showed only ONE Ray worker at 100% CPU + 10 GB RES, others idle, log
silent for 5+ minutes mid-load. Each host independently dequantizes the
FULL 543 GB — single-threaded torch ops in `dequant_fp4_to_bf16`. For
real-load viability, the next iteration likely needs (a) parallelize
dequant within a host, or (b) shard safetensors keys across hosts so
each host only dequants its 1/8 share. `make_array_from_callback` is
per-host so option (b) is correct.

**Known good signals to look for in log:**
  * `Init mesh | mesh=Mesh('data': 1, 'attn_dp': 32, ...)` — TP=32 mesh.
  * `[deepseek_v4] load_weights_from_dir: streaming '<path>'` — entered.
  * `[deepseek_v4] placed N tensors (R/s, last=<hf_name>)` — heartbeat.
  * `[deepseek_v4] load_weights_from_dir done: placed=N skipped=M elapsed=Ts`.
  * `XLA::TPU program HBM usage: <real GB> / 31.25G` — real fit.
  * `Application startup complete` — server ready, port 18081.

Smoke pass: `curl -s http://localhost:18081/v1/completions -H 'Content-Type:
application/json' -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","prompt":"The
capital of France is","max_tokens":8,"temperature":0,"seed":0}'` — must
return text starting with " Paris" / "Paris", byte-identical on repeat.

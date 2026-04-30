---
name: Ray cleanup hygiene
description: Use narrow pkill patterns for vllm/Ray worker cleanup — `RayWorkerWrapper` and `EngineCore` substring matches kill the raylet daemon too.
type: feedback
---

When stopping a failed `vllm serve` run on a Ray cluster, do NOT use
broad regex like:

```bash
pkill -9 -f "EngineCore|RayWorkerWrapper|vllm"
```

**Why:** the raylet's process command line on each worker contains
strings that match those patterns (substring `RayWorker*` shows up in
internal raylet args). Running the broad regex on remote workers via
ssh kills the raylet itself, the Ray cluster loses 7/8 nodes, the
placement group leaks, and the full-slice deploy needs a
`ray stop --force` + restart sequence (see
`scripts/full_slice_v4_ray_restart.sh`). Cost the session ~10 min the
first time it happened.

**How to apply:** When killing leftover vllm processes, prefer either:

  * Kill the engine pid recorded by `scripts/full_slice_v4_smoke.sh`
    (`logs/full-slice-v4-smoke.pid`) — the script writes one.
  * Kill ONLY the api-server frontend, e.g. `pkill -f "bin/vllm serve"`
    (matches the entrypoint script, never the raylet).
  * Use Ray's own API: `ray.util.placement_group_table()` plus
    `ray.util.remove_placement_group(...)` to drop leaked PGs without
    touching daemons.

For a clean reset of just the model engine without rebuilding Ray:
`pkill -f "bin/vllm serve"` on the head host is sufficient — Ray
automatically tears down RayWorkerWrappers when their owning driver
exits.

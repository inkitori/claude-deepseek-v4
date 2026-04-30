# DeepSeek V4 status

> **Authoritative status now lives in [`../../CLAUDE.md`](../../CLAUDE.md)
> ("What's been optimized + verified" + "Known bloat / consolidation
> candidates" sections). This file is retained as a stub so historical
> references resolve.**

Snapshot at the time this stub was written:

* **Load (real V4-Flash, 35020 tensors):** ~4 min on the v6e-32 slice.
  Verified end-to-end (`load_weights_from_dir done: placed=35020 skipped=0`).
* **MoE forward:** vectorized; math byte-equivalent vs the per-expert
  reference loop on a synthetic fixture (maxabs=0 across 5 seeds);
  HLO instruction count for `jit_run_model` drops 477,014 → 103,540
  (4.6×). **But:** the vectorize introduces a compile-time HBM OOM
  on real V4-Flash via an unsharded 4 GB all-gather of the stacked
  expert weights. **First curl currently fails.** See
  [`../../CLAUDE.md`](../../CLAUDE.md) "Current state" for the full
  breakdown + 4 ranked attack lanes.
* **First curl:** broken (HBM OOM at compile time). Fixing this is the
  next agent's #1 task; lane 1 (`with_sharding_constraint` on the
  stacked MoE weights) should be a one-line fix.

For the current state, read CLAUDE.md.

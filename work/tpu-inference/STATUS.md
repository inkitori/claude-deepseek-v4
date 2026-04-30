# DeepSeek V4 status

> **Authoritative status now lives in [`../../CLAUDE.md`](../../CLAUDE.md)
> ("What's been optimized + verified" + "Known bloat / consolidation
> candidates" sections). This file is retained as a stub so historical
> references resolve.**

Snapshot at the time this stub was written:

* **Load (real V4-Flash, 35020 tensors):** ~4 min on the v6e-32 slice.
  Verified end-to-end (`load_weights_from_dir done: placed=35020 skipped=0`).
* **MoE forward:** vectorized to 3 einsums per layer; HLO instruction
  count for `jit_run_model` dropped from 477,014 → 103,540 (4.6×). Math
  byte-equivalent vs the per-expert reference loop on a synthetic fixture
  (maxabs=0 across 5 seeds).
* **Cold first curl:** still ~10–15 min on a clean compile cache. Not yet
  reduced to "fast"; this is the primary objective for the next agent
  iteration (see `prompt.md` "Your job — primary objective").

For the current state, run the iterate loop and read CLAUDE.md.

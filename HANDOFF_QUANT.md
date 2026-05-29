# Handoff — DeepSeek-V4-Flash QUANTIZED-WEIGHT-LOADING campaign (v6e-16)

> **Phase = QUANT / FIT-ON-v6e-16.** Make `vllm serve deepseek-ai/DeepSeek-V4-Flash`
> LOAD AND SERVE CORRECTLY on **v6e-16** by **NOT dequantizing the FP4 experts to bf16 at load**.
> Durable slice ops + pitfalls: `CLAUDE.md`. Prior campaigns (history): `HANDOFF_PERF.md`,
> `HANDOFF_S1.md`. This doc = the loop's memory.
>
> **One-line status (2026-05-29):** **THE FULL MODEL NOW LOADS + FITS.** Host-gathering ALL routed
> expert leaves (commit `e1b434f8`) eliminated the layer-0 consolidation `device_put` collectives
> that core-halted every prior smoke — load completes: **`placed=68812 skipped=0 elapsed=233.1s,
> host_gather_groups=264`, ZERO scheckne during load.** **NEW BLOCKER: a SECOND, distinct `scheckne`
> launch-id halt fires ~4 min later during POST-LOAD FORWARD/warmup COMPILE** (log
> `logs/full-slice-v4-smoke-20260529T025539Z.log`). Same assertion pc, now in the forward path, not
> consolidation. **NEXT = confirm it's deterministic (1 retry; runbook says init can be flaky), then
> HLO-dump the post-load compile + diff across the 4 hosts** (the technique that nailed the
> consolidation), prime suspect = a host-divergent forward program (env-gated trace-time branch?).

---

## THE PROBLEM (one table)
V4-Flash ships natively quantized: dense=FP8, 256 routed experts=**FP4 (=MXFP4, codebook ≡
`jnp.float4_e2m1fn`, e8m0 block scale, block 32)**. On-disk: routed `layers.{L}.ffn.experts.{0-255}.{w1,w2,w3}.weight`
= **I8** packed + `.scale` = **F8_E8M0**; shared `ffn.shared_experts` = FP8 (E4M3); 43 layers +
1 MTP, hidden=4096, moe_inter=2048. Old loader dequantized everything → ~542 GiB > 512 GiB → OOM.

| scheme | HBM | fits v6e-16 (512 GiB)? |
|---|---:|---|
| bf16 (old load path) | ~542 GiB | **NO** (OOM) |
| **fp4-experts-kept + dense bf16 (Strategy C)** | **~155 GiB** | **yes** — and now LOADS (68812 tensors) |

---

## WHAT LANDED (committed, CPU-gated) — Strategy C loader + the FIT FIX
- `26e4023d` Q.1 + MoE consumer: routed experts declared FP4, dequant-in-trace (`_dequant_fp4_experts`).
- `26318abf` Q.2 loader: emit FP4 experts COMPRESSED (uint8 weight + scale leaves), no bf16 dequant.
- `580b1f83` scales stored as **uint8** (not on-device `float8_e8m0fnu`) — REFUTED the e8m0-halt theory
  (crash was byte-identical with uint8). Kept: it's correct (matches gpt_oss/qwix/the consumer).
- `6eb5241f` un-swallow loader exception (`deepseek_v4.py:2026` re-raises unless
  `V4_ALLOW_DUMMY_FALLBACK=1`) — REFUTED the swallowed-exception theory (re-raise never fired ⇒ no
  exception). Correctness win: never silently serve zero-weighted garbage. + host-gather w1/w3/scales
  (left w2 on device_put) — STILL crashed, proving even ONE consolidation device_put diverges here.
- **`e1b434f8` THE FIT FIX:** host-gather **EVERY** routed/mtp expert leaf (w1/w2/w3 + all scales) ⇒
  ZERO consolidation `device_put` reshard collectives. Edits in `deepseek_v4.py`: `_is_stash_leaf`
  (`return m is not None`) + `use_host_gather` (`all(k in _expert_host_np ...)`) match all routed
  leaves; `deepseek_v4_loader.py:981` `place_spec_as_jax_sharded` full-reads + returns host_np when
  `return_host_np=True` even for axis-0 leaves (so host-gather has the data).
  - **WHY it was needed:** uint8 packing makes w1/w3 SQUARE `[2048,2048]` → `pick_partition_spec`
    strict-`>` tie-break (`deepseek_v4_loader.py:~513`) picks **axis-0** → slice-aware path → host_np
    was None → host-gather DORMANT → `device_put` reshard. Those reshards are cross-host collectives;
    on the v6e-16 4×4 topology they desync the SPMD launch group (HLO diff proved: HEAD launched a
    real-data consolidation stack while the 3 WORKERS launched whole-tree scalar-fills). bf16's lone
    w2 device_put worked on v6e-32 but NOT here — only ZERO consolidation collectives passes. (This is
    the same axis-0 consolidation S1's S27 tried + reverted — see `HANDOFF_S1.md`.)
  - host-gather = `make_array_from_callback` from full per-expert host numpy; collective-free + byte-
    clean (no uninit-HBM reshard), so S1-safe. Load is now fully collective-free → completes.

---

## ⚠️ THE NEW BLOCKER — SECOND scheckne in the POST-LOAD FORWARD COMPILE
Smoke `025539Z`: load done at 02:56:14 (`placed=68812 … host_gather_groups=264`). Then ~4 min of
forward/warmup jit compiles (`jit_broadcast_in_dim` fingerprints, `jit_exp/multiply/outer/iota`,
`make_freqs_cis`-looking ops). At 03:00:16 → **`Core halted … scheckne` at `TensorCoreSequencer:1:0xba`**
(SAME pc as the consolidation halt) → worker on `.17`/`.8` dies (`SLICE_FAILURE_SW_INJECT_ERROR` →
SYSTEM_ERROR "connection error code 2"). Engine init fails. This is a DISTINCT divergence from the
(now-fixed) consolidation — it's in the forward path, post-load.

**ROADMAP / NEXT ACTIONS (do in order):**
1. **Re-smoke once, plain.** The runbook notes init is sometimes a flaky worker SYSTEM_ERROR "just
   retry". This crash is a `scheckne` (looks deterministic, like the consolidation), but confirm: if a
   clean retry reaches `Application startup complete`, it was flaky → go straight to the GATE.
2. **If deterministic → HLO-dump the POST-LOAD compile + diff across the 4 hosts** (the decisive
   technique that nailed consolidation). `V4_XLA_FLAGS=--xla_dump_to=/tmp/hlo_dump` (validates clean;
   propagates to workers via ray env), clear xla_cache+/tmp/hlo_dump on all 4 first. After crash:
   rsync each host's `/tmp/hlo_dump/*optimizations.txt` to head, compare the module **opname multiset
   + ENTRY signatures** head-vs-worker (helper recipe below). The op that differs across hosts is the
   divergence.
3. **Prime suspect = a host-divergent FORWARD program.** The forward is SPMD-jitted so it SHOULD be
   identical; a launch-id split usually means a **trace-time host-dependent branch**. Check (CLAUDE.md
   pitfall #0): any env-var read or `process_index`/device-ownership branch reached during forward
   trace. Specifically: `layers/jax/moe/deepseek_v4_moe.py::moe_forward` — the `use_shard_map` gate
   (~:211) choosing dense-all-256 (decode) vs sharded `gmm_v2` (prefill), and the in-trace
   `_dequant_fp4_experts` (uint8+e8m0→bf16) over the E-sharded experts. Also `make_freqs_cis`.
4. After it serves: **GATE** below (establish the v6e-16 baseline md5).

HLO-diff helper (proven this session):
```
for ip in 8 17 16; do rsync -az --include='*.before_optimizations.txt' --exclude='*' \
  -e "ssh -i ~/.ssh/google_compute_engine" enyouki@10.164.0.$ip:/tmp/hlo_dump/ /tmp/hlo_cmp/h$ip/; done
# head (15) is local: cp /tmp/hlo_dump/*.before_optimizations.txt /tmp/hlo_cmp/h15/
# then per host: ls .../*.before_optimizations.txt | sed -E 's#.*/module_[0-9]+\.##;s#\.cl_[0-9]+.*##' | sort|uniq -c
# divergent count/signature between h15 and h8/h17/h16 = the culprit op.
```

---

## <a name="GATE"></a>GATE (non-negotiable) — for v6e-16
Old v6e-32 md5 `5bf42256` is DEAD. Bar:
- `LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` → rc=0 (port 18081; visible_words ≥ 10,
  max_word_run < 5). (smoke_check uses prompt 'The capital of France is', temp=0.)
- FIB decode: **correct Fibonacci (21,34,55,89,144)** in a longer decode + **N=2 md5 byte-identical
  across 2 fresh engines** via `python3 /tmp/s1_probe2.py N` (RECREATED this session — prompt "Here is
  the start of the Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13,", temp=0 seed=0, auto-discovers model id;
  ⚠️ /tmp is ephemeral — the file may be gone next session, the prompt above reproduces it). Establish
  the NEW baseline hash once; confirm identical ×2 engines. READ the actual text ("contains Paris" is a
  false positive). Do NOT gate on a long-tail md5 (nondeterministic at temp=0).

---

## VALIDATION TIERS (cheapest first)
1. **CPU loader fp4 check** (~40s, no slice): `JAX_PLATFORMS=cpu PYTHONPATH=work/tpu-inference:work/vllm
   work/vllm_env/bin/python3 scripts/quant_loader_fp4_check.py` → "OK …". (Needs `work/scratch/tiny_v4_quant`
   + `tiny_v4_groundtruth` fixtures.) **All CPU gates currently PASS.**
1b. **CPU oracle** `scripts/s1_cpu_repro_v4flash.py both` (eager==jit, bad=0/12) + **fp4 dequant**
   `scripts/quant_fp4_dequant_check.py` (max|Δ|=0). NOTE: CPU has mesh=None ⇒ consolidation/host-gather
   is SKIPPED, so CPU CANNOT validate the host-gather or forward-compile paths — those need a smoke.
2. **TPU microbench** (`scripts/perf_microbench*`).
3. **Full smoke + GATE** — now reaches `placed=68812` (~233s load) then the forward-compile crash
   ~4 min later. Cheap to iterate the forward-compile blocker (crashes ~5-6 min in).

---

## INFRA STATUS / v6e-16
- Slice `v6spoteu719`, zone `europe-west4-a`, project `prm-research`. **v6e-16, topology 4×4**, 16
  chips, 4 hosts: head `10.164.0.15` + workers `10.164.0.8 / .17 / .16`. Ray healthy (16 TPU). TP=16.
- ⚠️ **numpy MUST be `<2.4` (pinned `2.3.5`)** — 2.4.x breaks `import numba`, crashes the APIServer
  before any TPU work. venv has no pip (uv): `~/.local/bin/uv pip install --python work/vllm_env/bin/python3
  'numpy==2.3.5'` per host. Pre-smoke: `python3 -c "import numba"` per host.
- ⚠️ Ray "version mismatch" = mark's rogue ray container; **FIX = keep BOTH guardians alive**
  (`ps -eo pid,cmd | grep -E 'node_guard[i]an|meta_guard[i]an'`; restart per the loop prompt,
  meta_guardian needs `10.164.0.15:6379`). Ray (re)start: `scripts/full_slice_v4_ray_restart.sh`.
- After ANY code edit: `full_slice_v4_sync.sh` (rsync to 4 hosts; `git push` does NOT sync them) +
  clear `~/.cache/vllm/xla_cache/*` on all 4 + verify md5 head==workers (mismatch → launch-id halt).
  Shut down ONLY via `full_slice_v4_reset.sh`. HLO dump propagates to workers via the ray env (XLA_FLAGS).

# S1 handoff — LOCALIZED: per-process garbage enters in LAYER-0 FORWARD (downstream of attn kv-seed); fp32 REFUTED

Goal: coherent, **deterministic** decode for `vllm serve deepseek-ai/DeepSeek-V4-Flash` on the
v6e-32 slice. Ops in `CLAUDE.md`; this is live state.

## STATE (2026-05-26, SESSION 12) — two decisive results

**1. fp32 matmul band-aid REFUTED.** Global `jax_default_matmul_precision=highest` via CODE, 2
engines same warm cache: ENG1 `fef3a2e4` ("21,34,55,…"+prose) ≠ ENG2 `50aae3c0`
("…377,521,618,139…"). Per-process nondeterminism SURVIVES full fp32. Root: V4 is ALREADY mostly
fp32 (logits deepseek_v4.py:2008, `_linear` :469, MoE gate/up W1/W3 pinned) — only MoE W2 + misc
flipped — so uninit-HBM garbage propagates regardless of matmul precision. fp32 reverted.

**2. LOCALIZED to LAYER 0 FORWARD via seed checksums.** Added always-on `_v4_checksum`
(deepseek_v4_attention.py) printing sum+absmax at 4 seed-build points; ran 2 fresh engines (ENG3
`764fe7cd`, ENG4 `ad51abcd` — EVERY process gives a DIFFERENT FIB continuation), diffed the
`[ckS]` global-all-reduce sums (`/tmp/s1_ckdiff.py LOG_A LOG_B`):
* **Layer-0 seed BYTE-IDENTICAL across processes** (seed_x_in 1.8616e3 both; kv/kv_cache identical).
* **Layer-1 input (`seed_x_in`) DIFFERS** (1683.6 vs 1719.4, Δ≈36, real not float-noise) and grows
  with depth (NaN by L36-42; the `_linear` clamp turns NaN→finite so kv_postlinear stays
  finite-but-divergent). 111/172 checksums differ.
⇒ `seed_x_in(L)` = layer L's residual input; L0 input (embedding) is identical ⇒ **per-process
garbage FIRST enters during LAYER 0's forward, DOWNSTREAM of the attention kv-seed** (L0 seed is
identical). REFOCUSES off the attention seed (S5-S10 dead-end) onto layer 0 `block_forward`.

## NEXT ACTION — pinpoint WITHIN layer 0, then fix.
`block_forward` (deepseek_v4.py:265-291): residual→hc_pre→rms→**attention_prefill(:280)**→
hc_post(:281)→residual→hc_pre→**moe_forward(:290)**→hc_post(:291). Add `_v4_checksum` (gate to
layer_idx<2; import from deepseek_v4_attention) on: y after :280 (attn out), x after :281, y after
:290 (MoE out), out after :291. sync→clear xla_cache→cold smoke→`/tmp/s1_warmup.py`→grep `[ckS]`→
reset→warm 2nd engine→warmup→`/tmp/s1_ckdiff.py`. FIRST of {attn_out, moe_out} that differs across
the 2 processes = the culprit op. **MoE is prime suspect**: bespoke `out_NEd.sum(axis=1)` over
attn_dp-sharded experts (deepseek_v4_moe.py `moe_forward`) — in prefill, idle ranks (no tokens) can
sum uninit-HBM garbage; it's CLAUDE.md's standing suspect.
THEN fix at that op: zero idle-rank/idle-token contributions (shard_map + `jnp.where(is_live, real,
0.0)` + `lax.psum`, NO token-axis gather — precedents in CLAUDE.md). Re-verify: 2-engine FIB
byte-identical + coherent.

## DONE gate (unchanged): FIB coherent through 1597 (READ TEXT) + byte-identical across TWO fresh
engines + survives 5 reqs. Engine CORE-HALTS on the PARIS shape — FIB-only when probing. NB: every
process now gives a DIFFERENT coherent-ish FIB ("21,34,55…"+prose or +numeric-derail) ⇒ COHERENCE
is no longer the discriminator; CROSS-PROCESS determinism is.

## Tools / ops
* `/tmp/s1_warmup.py` (1 FIB, 2400s timeout, absorbs the DEFERRED cold compile → result file).
  `/tmp/s1_fib2.py <label>` (FIB×3 within-engine). `/tmp/s1_ckdiff.py LOG_A LOG_B` (diff [ckS] sums).
* `_v4_checksum(name,x,layer_idx)` always-on in deepseek_v4_attention.py + 4 calls in
  attention_init_state_from_prefill — **REMOVE all `[ckS]`/helper when S1 closes.**
* ENG3 log `logs/...034437Z.log`, ENG4 log `logs/...035449Z.log` (both have `[ckS]`).
* Reset CLEAN (0/32). Guardians up (node 497956 + meta 4039835). Slice HEALTHY (4 clean smokes
  this session). Cold compile is DEFERRED to the 1st request (startup completes ~6min even cold).

## DEAD fixes (do NOT retry)
* **fp32 matmul highest (S12)** — survives; model already fp32.
* **attention KV-SEED fixes (S12)** — L0 seed is byte-identical across processes; garbage is in the
  layer-0 FORWARD, not the seed. (Subsumes: :775 seed-replicate=benign all-gather S11; zero
  compressor/indexer seed pad-slots=regressed S6.)
* `wsc(activation,P())` gathering empty/idle shards → Core-halts (~8×).
* mask matmul INPUT x / position<n_real → no-op (garbage on idle RANKS not pad positions).

## Durable lessons
* `wsc(ACTIVATION,P())` gathering empty/idle shards Core-halts. decode-vs-prefill ON SLICE is the test.
* `different launch id`/`Core halted` BEFORE startup = CODE DESYNC → sync + clear xla_cache. Don't reboot.
* (`/tmp/s1_loop_stop` NOT set.)

# Decisions log — DeepSeek V4 implementation

Durable architectural decisions that shape the V4 implementation.
Per-session decisions (which Tier to attack next, which test to add)
are recorded in commit messages and live in `git log`.

## D1 — Reference oracle is DeepSeek's `inference/model.py`, not HuggingFace transformers

**Decision:** Use DeepSeek's official `inference/model.py` (under the
HF repo's `inference/` dir) as the architectural ground truth for V4
math.

**Why:** HuggingFace `transformers` (5.6.x at the time of writing)
does not include a `deepseek_v4` model_type — `AutoConfig` raises
`KeyError`. The DeepSeek HF repo also does not ship a
`modeling_deepseek_v4.py` with `auto_map`. DeepSeek instead ships
its own reference implementation. That reference is what
`convert.py` expects to load real weights into.

**Implications:** A CPU-runnable copy of `inference/model.py` lives
under `tests/models/jax/_deepseek_v4_reference/`, with custom CUDA
kernels (`sparse_attn`, `hc_split_sinkhorn`, `act_quant`,
`fp4_act_quant`, `rotate_activation`) replaced by pure-PyTorch
equivalents. Math equivalence to those replacements is sufficient —
they're performance-only optimisations of well-defined operations.

## D2 — JAX implementation does not use V3's MoE backend selection or `ragged_paged_attention`

**Decision:** V4 has its own MoE forward and its own `sparse_attn`
in `layers/jax/{moe,attention}/deepseek_v4_*.py`. It does NOT route
through V3's `JaxMoE` backend selector or
`tpu_inference.kernels.ragged_paged_attention.v3`.

**Why:** V3's `JaxMoE` selects between sparse / dense / EP backends
designed for V3's grouped-routing pattern, which V4 does not have.
V3's paged attention assumes a flat KV layout, which V4's
sliding-window + compressed-pool dual-buffer attention doesn't fit.
Building separate V4 implementations was lower-risk and lower-LOC
than generalizing V3's machinery.

**Implications:** When `ragged_paged_attention` adds support for
top-k + attn_sink + dual-buffer KV, the V4 attention can swap to it
for a perf win without changing the math. The current
fully-materialized `sparse_attn` is correctness-only; a real Pallas
kernel is backlog item B1 (CLAUDE.md "Production-readiness
backlog"). Similarly, `kernels/megablox/gmm.py` is the path forward
for sparse MoE dispatch (B2) once V4's hash-routing variant is
integrated.

## D3 — Tiny config matches V4-Flash structure (alternating CSA/HCA layers)

**Decision:** The tiny test config uses 6 hidden layers with
`compress_ratios = [0, 0, 4, 128, 4, 0]` so we exercise pure
sliding-window attention, CSA-with-indexer, HCA-without-indexer,
and the trailing pure-SWA layer that V4-Flash and V4-Pro both
have at the very end.

**Why:** Each `compress_ratio` value triggers a different code path
inside `Attention.forward`. Missing one in the tiny config would
leave a code path untested on the fast loop.

See [TINY_CONFIG.md](TINY_CONFIG.md) for the full derivation.

## D4 — MTP layer is in the tiny config; speculative-decoding integration is NOT wired

**Decision:** Tiny config has `n_mtp_layers=1` (matching V4-Pro and
V4-Flash). The MTP block is tested by feeding `(h, start_pos,
input_ids)` to `model.mtp[0]` and comparing logits.

**Why:** MTP is a real production code path in V4 and adds new
params (`e_proj`, `h_proj`, `enorm`, `hnorm`, `hc_head_fn/scale/base`).
Forward equivalence is verifiable on the tiny fixture; vLLM's
speculative-decoding hook integration was originally downstream
work, but is now active backlog item S5 (CLAUDE.md
"Production-readiness backlog") since the math foundation is
solid and 1.5–2× decode throughput is on the table once S1
(real decode plumbing) lands.

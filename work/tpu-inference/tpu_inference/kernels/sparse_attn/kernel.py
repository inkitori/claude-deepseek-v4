# SPDX-License-Identifier: Apache-2.0
"""Fused sparse-attention TPU kernel for DeepSeek-V4.

Replaces the materialized KV gather in
`tpu_inference/layers/jax/attention/deepseek_v4_attention.py::sparse_attn`
(the profiled bottleneck — `jnp.take_along_axis` building a `[B,M,K,D]` tensor:
99% of prefill, 66% of a decode step, at ~0.02-0.05% of HBM bandwidth).

Math (preserved BIT-FOR-BIT vs the oracle `sparse_attn_torch` and the JAX
`sparse_attn` it replaces — both single-pass fp32 softmax):

  logits[h,k] = (q[h] . kv[idx[k]]) * softmax_scale          # idx<0 clamped to 0
  mask: logits[k] = -inf where topk_idxs[k] == -1
  m = max( max_k logits , attn_sink[h] );  m = 0 if non-finite (all-masked row)
  p = exp(logits - m); p = 0 where invalid
  denom = sum_k p + exp(attn_sink[h] - m)     # sink adds to DENOMINATOR only
  out[h] = (p / denom) . kv_gathered          # no sink value vector

Single shared KV head broadcast across all H query heads: gather the K rows once
per (b,m) and reuse across heads. Read bf16 kv, accumulate in fp32, cast to bf16
on output. K is bounded (<=640 decode-CSA) so `[K,D]` fits VMEM → SINGLE-PASS
softmax over the gathered tile (matches the oracle exactly; no online-softmax
accumulation-order drift). Output stays RoPE-rotated; the caller applies inverse
RoPE — the kernel does not touch RoPE.

GATHER MECHANISM = ONE-HOT MATMUL. Each program handles one (b,m) query; the K
selected rows are gathered as `onehot[k,n] = (safe[k]==n)` (exact 0/1 in bf16)
times the resident `kv[N,D]`, so `onehot @ kv` reproduces `kv[safe[k]]`
BIT-IDENTICALLY (one nonzero term per row — correct even for duplicate indices,
matching `take_along_axis`). Why not a dynamic per-row load: Mosaic cannot
dynamically index the sublane-tiled (8,128) VMEM row axis at an arbitrary
(non-8-aligned) offset (verified — `E2003 ...UnprovenMemoryAccessAlignment`), so
`kv_ref[0, pl.ds(idx,1), :]` does NOT lower on v6e. The one-hot matmul is pure
MXU work => lowers cleanly for BOTH call sites (microbench-confirmed).

  *** The onehot READS ALL N. Fine for decode (latency-bound; microbench
  decode_csa 5.44ms baseline -> 0.131ms kernel, ~41x) and prefill (bandwidth win
  is from reusing kv across queries, not from skipping rows). For the LARGE-N
  decode-CSA regime (msl huge => reading all N dominates), the production gather
  should switch to the proven gen-6 DMA idiom (ragged_paged_attention/v3 +
  mla/v2): SMEM scalar-prefetch the index array
  (`PrefetchScalarGridSpec(num_scalar_prefetch=...)`), kv as an HBM operand
  (`pl.BlockSpec(memory_space=pltpu.HBM)`), per-row
  `pltpu.make_async_copy(kv_hbm.at[pl.ds(idx_ref[j]*D, sz)], scratch.at[...], sem)`
  in a statically-unrolled loop, size-0 copy for the -1 sentinel. The SOFTMAX
  MATH is identical for either gather — only the kvg-construction line changes. ***

Status: MATH validated bit-parity vs `sparse_attn_torch` + the JAX `sparse_attn`
on CPU (`interpret=True`, `tests/.../test_deepseek_v4.py::TestSparseAttnKernel`);
TPU-COMPILES + runs for decode (M=1) and prefill (M=S) via the microbench
(`scripts/perf_microbench.sh --kernel`). NOT yet wired into the model — wiring
into both call sites + the full S1 smoke gate is the next step.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _sparse_attn_kernel(
    q_ref,      # [1, 1, H, D]  bf16
    kv_ref,     # [1, N, D]     bf16  (single shared KV head; resident across the M grid)
    sink_ref,   # [H]           fp32
    topk_ref,   # [1, 1, 1, K]  int32  (-1 == ignore)
    out_ref,    # [1, 1, H, D]  out dtype (== q.dtype, bf16)
    *,
    softmax_scale: float,
):
    idx = topk_ref[0, 0, 0, :]                    # [K] int32
    safe = jnp.maximum(idx, 0)                    # [K]  clamp -1 -> 0 (oracle: clamp(min=0))
    N = kv_ref.shape[1]

    # Gather the K selected kv rows via a ONE-HOT MATMUL, not dynamic per-row
    # loads: Mosaic cannot dynamically index the sublane-tiled (8,128) VMEM row
    # axis at an arbitrary (non-8-aligned) offset (E2003). `onehot[k,n] =
    # (safe[k] == n)` is exact 0/1 in bf16, so `onehot @ kv` reproduces
    # kv[safe[k]] BIT-IDENTICALLY (exactly one nonzero term per row — correct
    # even for duplicate indices, matching `take_along_axis`). Pure matmul =>
    # lowers cleanly. Reads all N; for the large-N decode-CSA regime the DMA
    # idiom in the module docstring avoids the full-N read (future optimization).
    kv = kv_ref[0]                                # [N, D] bf16 (resident)
    onehot = (safe[:, None]
              == jnp.arange(N, dtype=jnp.int32)[None, :]).astype(kv.dtype)  # [K, N]
    kvg = lax.dot_general(
        onehot, kv, (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32)        # [K, D] fp32

    qf = q_ref[0, 0, :, :].astype(jnp.float32)    # [H, D]
    # logits[h,k] = (qf[h] . kvg[k]) * scale ; contract over D.
    logits = lax.dot_general(
        qf, kvg, (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32)        # [H, K]
    logits = logits * softmax_scale

    valid = (idx != -1)                            # [K]
    logits = jnp.where(valid[None, :], logits,
                       jnp.full_like(logits, -jnp.inf))
    sink = sink_ref[:].astype(jnp.float32)         # [H]
    m_max = jnp.maximum(jnp.max(logits, axis=-1), sink)        # [H]
    m_max = jnp.where(jnp.isfinite(m_max), m_max, jnp.zeros_like(m_max))
    p = jnp.exp(logits - m_max[:, None])           # [H, K]
    p = jnp.where(valid[None, :], p, jnp.zeros_like(p))
    denom = jnp.sum(p, axis=-1) + jnp.exp(sink - m_max)        # [H]
    p = p / denom[:, None]
    # out[h,d] = sum_k p[h,k] * kvg[k,d] ; contract over K.
    out = lax.dot_general(
        p, kvg, (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32)        # [H, D]
    out_ref[0, 0, :, :] = out.astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("softmax_scale", "interpret"))
def sparse_attn_kernel(
    q: jnp.ndarray,          # [B, M, H, D]  bf16
    kv: jnp.ndarray,         # [B, N, D]     bf16
    attn_sink: jnp.ndarray,  # [H]           fp32
    topk_idxs: jnp.ndarray,  # [B, M, K]     int32 (-1 == ignore)
    softmax_scale: float,
    *,
    interpret: bool = False,
) -> jnp.ndarray:            # [B, M, H, D]  bf16 (== q.dtype)
    """Fused sparse multi-head attention with a learnable per-head sink.

    Drop-in for `deepseek_v4_attention.sparse_attn` (identical signature + math).
    One program per (b, m) query; the KV gather + fp32 softmax + sink + output
    are fused into a single kernel launch per layer (no `[B,M,K,D]` materialized
    to HBM, no separate softmax pass).
    """
    B, M, H, D = q.shape
    N = kv.shape[1]
    K = topk_idxs.shape[-1]

    # topk_idxs reshaped [B,M,K] -> [B,M,1,K] so the grid-tiled M axis (block 1)
    # leaves the sublane/lane-tiled last-two dims (Pallas requires those either
    # equal the array dims or be divisible by (8,128)); q/kv/out are unaffected
    # since M is not in their last-two dims.
    return pl.pallas_call(
        functools.partial(_sparse_attn_kernel, softmax_scale=softmax_scale),
        grid=(B, M),
        in_specs=[
            pl.BlockSpec((1, 1, H, D), lambda b, m: (b, m, 0, 0)),  # q
            pl.BlockSpec((1, N, D), lambda b, m: (b, 0, 0)),        # kv (resident)
            pl.BlockSpec((H,), lambda b, m: (0,)),                  # attn_sink
            pl.BlockSpec((1, 1, 1, K), lambda b, m: (b, m, 0, 0)),  # topk_idxs [B,M,1,K]
        ],
        out_specs=pl.BlockSpec((1, 1, H, D), lambda b, m: (b, m, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((B, M, H, D), q.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=interpret,
    )(q, kv, attn_sink, topk_idxs.reshape(B, M, 1, K))

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

GATHER MECHANISM (current = the bounded-N regime): each program handles one
(b,m) query; it gathers its K kv rows with per-row dynamic `pl.ds` slices on a
VMEM-resident kv block, then `concatenate`s to `[K,D]`. This is correct + lowers
for N small enough that `kv[N,D]` fits VMEM (decode SWA/HCA, short prefill).

  *** For the LARGE-N regime (long-context decode-CSA, where reading all N would
  dominate and kv may not fit VMEM) the production gather should use the proven
  gen-6 idiom from ragged_paged_attention/v3 + mla/v2: scalar-prefetch the index
  array into SMEM (`PrefetchScalarGridSpec(num_scalar_prefetch=...)`), keep kv as
  an HBM operand (`pl.BlockSpec(memory_space=pltpu.HBM)`), and DMA each selected
  row into a VMEM scratch via
  `pltpu.make_async_copy(kv_hbm.at[pl.ds(idx_ref[j]*D, sz)], scratch.at[...], sem)`
  in a statically-unrolled loop, double-buffered semaphores, size-0 copy for the
  -1 sentinel. The SOFTMAX MATH below is identical for either gather — only the
  `_gather_kv` step changes. ***

Status: MATH validated bit-parity vs `sparse_attn_torch` on CPU (`interpret=True`)
— see `tests/.../test_deepseek_v4.py::TestSparseAttnKernel`. NOT yet wired into
the model; TPU compile + microbench + the S1 gate are the next step.
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
    topk_ref,   # [1, 1, K]     int32  (-1 == ignore)
    out_ref,    # [1, 1, H, D]  out dtype (== q.dtype, bf16)
    *,
    softmax_scale: float,
    K: int,
):
    idx = topk_ref[0, 0, :]                       # [K] int32
    safe = jnp.maximum(idx, 0)                    # [K]  clamp -1 -> 0 (oracle: clamp(min=0))

    # Gather the K selected kv rows (bf16), then upcast to fp32 — bit-identical
    # to upcasting kv first (gather is pure indexing). Per-row dynamic slice on
    # the VMEM-resident kv block; `k` is static so `safe[k]` is a plain scalar
    # value used as a dynamic slice START (the legal `pl.ds` dynamic-start form).
    rows = [kv_ref[0, pl.ds(safe[k], 1), :]
            for k in range(K)]                    # each [1, D] bf16
    kvg = jnp.concatenate(rows, axis=0).astype(jnp.float32)   # [K, D] fp32

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

    return pl.pallas_call(
        functools.partial(_sparse_attn_kernel, softmax_scale=softmax_scale, K=K),
        grid=(B, M),
        in_specs=[
            pl.BlockSpec((1, 1, H, D), lambda b, m: (b, m, 0, 0)),  # q
            pl.BlockSpec((1, N, D), lambda b, m: (b, 0, 0)),        # kv (resident)
            pl.BlockSpec((H,), lambda b, m: (0,)),                  # attn_sink
            pl.BlockSpec((1, 1, K), lambda b, m: (b, m, 0)),        # topk_idxs
        ],
        out_specs=pl.BlockSpec((1, 1, H, D), lambda b, m: (b, m, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((B, M, H, D), q.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=interpret,
    )(q, kv, attn_sink, topk_idxs)

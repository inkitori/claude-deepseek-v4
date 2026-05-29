#!/usr/bin/env python3
"""TPU micro-benchmark: the V4 DECODE MoE expert-FFN device cost (roadmap #1).

P.4 split the 220 ms/decode-step into device_wait ~208 ms (96% of the wall) +
host-dispatch 8 ms. First-principles (HANDOFF_PERF.md NEXT ACTION) attributes the
~183 ms device COMPUTE to the DENSE decode MoE path (`deepseek_v4_moe.py:296-318`),
which bf16-dequants ALL 16 local FP4 experts IN-TRACE per token and runs dense
einsums over the local-E axis -- a ~10x HBM amplifier over the FP4-once floor.

This bench CONFIRMS that attribution AND measures the proposed fuse, on the real
16-chip mesh with SYNTHETIC FP4 experts at the real V4-Flash dims, N=1 token,
E-sharded on 'attn_dp' (16 local experts/chip) -- NO full-model load, NO vllm.

Two functions are timed (each freshly jitted), both ending in the E-sum psum:
  BASELINE = the production dense path: `_dequant_fp4_experts` (FP4->bf16) x3 +
             three dense einsums over the local-E axis + mask + sum-over-E (psum).
  FUSED    = the proposed fix: a shard_map over 'attn_dp' that feeds the FP4 leaves
             into gmm_v2 as fp8 codes + per-block rhs_scale (the prefill QUANT path,
             `_fp4_rhs_and_scale`), token replicated to lhs=[E_local, dim], one
             group/expert. gmm streams fp8 (1 byte) + dequants per-block in-register
             -> NO bf16 weight materialized. Output masked + summed + psum.

Both import the EXACT production helpers (`_dequant_fp4_experts`, `_fp4_rhs_and_scale`,
`_shard_e_*`) so the comparison -- and any fix mirrored from it -- is faithful.

Decision the numbers drive (HANDOFF roadmap #1):
  * baseline_per_layer * 43 ~ 183 ms  => MoE-dequant attribution CONFIRMED.
  * fused / baseline                  => the fuse win (expect HBM-bound ~2-4x: at
                                         N=1 every path is matrix-vector, the win is
                                         fp8 1-byte streaming + a tuned kernel, NOT MXU).
  * max-rel-err(baseline, fused)      => sanity (fp8 lhs+rhs vs bf16; real GATE = smoke).

Run (TPU must be FREE; SYNC to 4 hosts first -- mh_run runs each host's own clone):
    scripts/full_slice_v4_sync.sh
    MH_TIMEOUT=900 scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_moe_decode.py
"""
import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2
from tpu_inference.layers.jax.moe.deepseek_v4_moe import (
    _dequant_fp4_experts, _fp4_rhs_and_scale, _shard_e_first, _shard_e_last,
    _shard_e_mid)
from tpu_inference.layers.common.quantization import (
    MXFP4_BLOCK_SIZE, e8m0_to_fp32, u8_unpack_e2m1)

# The drafted dense FP8-code in-register-dequant kernel lives beside this script.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_dense_fp8_moe_kernel import dense_fp8_moe_matvec  # noqa: E402

# Real DeepSeek-V4-Flash MoE dims (config.json).
DIM = 4096          # hidden_size
INTER = 2048        # moe_intermediate_size
E = 256             # n_routed_experts
TOP_K = 6           # num_experts_per_tok
N_MOE_LAYERS = 43   # num_hidden_layers (all MoE; first 3 hash-routed -> same FFN cost)
DEVICE_COMPUTE_MS = 183.0  # P.4 measured decode device compute (the budget to attribute)
BLK = MXFP4_BLOCK_SIZE     # 32


def _local_shape(idx, global_shape):
    return tuple(len(range(*s.indices(g))) for s, g in zip(idx, global_shape))


def make_fp4_stacked(out_dim, in_dim, mesh, seed):
    """Synthetic STACKED packed-FP4 expert weight [E, out, in/2] uint8 + e8m0 scale
    [E, out, in/BLK] uint8, E-sharded on 'attn_dp'. Random uint8 weights are valid
    e2m1 (every byte = 2 codes); scale bytes are constrained near exp 127 so the
    dequanted magnitude stays O(1) (avoids silu/exp overflow on garbage scales)."""
    sh = NamedSharding(mesh, P('attn_dp', None, None))
    w_global = (E, out_dim, in_dim // 2)
    s_global = (E, out_dim, in_dim // BLK)

    def w_cb(idx):
        rng = np.random.default_rng(seed + 7919 * (idx[0].start or 0))
        return rng.integers(0, 256, size=_local_shape(idx, w_global), dtype=np.uint8)

    def s_cb(idx):
        rng = np.random.default_rng(seed + 104729 * (idx[0].start or 0) + 1)
        # exponent bytes in [125,130] -> scale 2^(b-127) in [0.25, 8].
        return rng.integers(125, 131, size=_local_shape(idx, s_global), dtype=np.uint8)

    w = jax.make_array_from_callback(w_global, sh, w_cb)
    s = jax.make_array_from_callback(s_global, sh, s_cb)
    return w, s


def make_replicated(shape, mesh, seed, dtype=jnp.bfloat16):
    sh = NamedSharding(mesh, P())
    rng = np.random.default_rng(seed)
    base = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    return jax.make_array_from_callback(shape, sh,
                                        lambda idx, _b=base: _b[idx]).astype(dtype)


def make_pew(mesh, seed):
    """Per-(token, expert) routing weight [1, E] fp32, E-sharded on 'attn_dp'.
    ~TOP_K of E nonzero (the rest masked to 0); values O(1). Only the SAME tensor
    fed to both paths matters for the numerics compare; timing is dense regardless."""
    sh = NamedSharding(mesh, P(None, 'attn_dp'))
    rng = np.random.default_rng(seed)
    full = np.zeros((1, E), np.float32)
    sel = rng.choice(E, size=TOP_K, replace=False)
    full[0, sel] = rng.uniform(0.3, 1.5, size=TOP_K).astype(np.float32)
    return jax.make_array_from_callback((1, E), sh, lambda idx, _f=full: _f[idx])


def baseline_expert_ffn(x, W1u, W3u, W2u, S1, S3, S2, pew):
    """Production dense decode path (deepseek_v4_moe.py:296-318), expert FFN only."""
    fp32, bf16 = jnp.float32, jnp.bfloat16
    W1 = _shard_e_first(_dequant_fp4_experts(W1u, S1))   # [E,inter,dim] bf16
    W3 = _shard_e_first(_dequant_fp4_experts(W3u, S3))
    W2 = _shard_e_first(_dequant_fp4_experts(W2u, S2))   # [E,dim,inter] bf16
    gate = _shard_e_mid(jnp.einsum('nd,eid->nei', x, W1, preferred_element_type=fp32))
    up = _shard_e_mid(jnp.einsum('nd,eid->nei', x, W3, preferred_element_type=fp32))
    h = jax.nn.silu(gate) * up                            # [1,E,inter] fp32
    h = _shard_e_mid(h * _shard_e_last(pew)[..., None])
    out = _shard_e_mid(jnp.einsum('nei,edi->ned', h.astype(bf16), W2.astype(bf16)))
    return out.astype(fp32).sum(axis=1)                   # [1,dim] (sum over E = psum)


def _dequant_fp4_lean(w_u8, scale):
    """LEAN bf16 dequant: BIT-IDENTICAL to _dequant_fp4_experts (fp4 codes and the
    pow2 e8m0 scale are both exact in bf16, so the bf16-domain product == the fp32
    one), but (a) no fp32 intermediates and (b) a broadcast-multiply over the 32-wide
    block instead of `jnp.repeat` (which materializes a full-size fp32 scale). Cheaper
    elementwise => XLA can fuse more of it into the matmul operand load."""
    if scale.dtype != jnp.uint8:
        scale = jax.lax.bitcast_convert_type(scale, jnp.uint8)
    codes = u8_unpack_e2m1(w_u8).astype(jnp.bfloat16)         # [..., K] bf16
    s = e8m0_to_fp32(scale).astype(jnp.bfloat16)              # [..., K/BLK] bf16 (exact pow2)
    K = codes.shape[-1]
    blk = codes.reshape(*codes.shape[:-1], K // BLK, BLK)     # [..., K/BLK, BLK]
    return (blk * s[..., None]).reshape(*codes.shape[:-1], K)  # [..., K] bf16


def baseline_lean_ffn(x, W1u, W3u, W2u, S1, S3, S2, pew, wsc=True):
    """Dense path with the LEAN dequant. wsc=False drops the intermediate
    _shard_e_first on the dequanted weights (a possible dequant->matmul fusion
    barrier); E-sharding is still carried by the sharded inputs + _shard_e_mid."""
    fp32, bf16 = jnp.float32, jnp.bfloat16
    _sf = _shard_e_first if wsc else (lambda z: z)
    W1 = _sf(_dequant_fp4_lean(W1u, S1))
    W3 = _sf(_dequant_fp4_lean(W3u, S3))
    W2 = _sf(_dequant_fp4_lean(W2u, S2))
    gate = _shard_e_mid(jnp.einsum('nd,eid->nei', x, W1, preferred_element_type=fp32))
    up = _shard_e_mid(jnp.einsum('nd,eid->nei', x, W3, preferred_element_type=fp32))
    h = jax.nn.silu(gate) * up
    h = _shard_e_mid(h * _shard_e_last(pew)[..., None])
    out = _shard_e_mid(jnp.einsum('nei,edi->ned', h.astype(bf16), W2.astype(bf16)))
    return out.astype(fp32).sum(axis=1)


def _dequant_fp8_lean(codes_f8, scale):
    """LEAN dequant from PRE-UNPACKED fp8 codes (the fp8-CODES-RESIDENT scheme): identical
    to _dequant_fp4_lean but the e2m1->fp8 unpack is ALREADY done (at load), so this skips
    u8_unpack_e2m1 (the bit-unpack) and reads 1-byte fp8 [E,out,in] (2x the packed-fp4 HBM).
    Still BIT-IDENTICAL to the fp4 lean dequant (fp8 e4m3 holds the e2m1 codebook exactly)."""
    if scale.dtype != jnp.uint8:
        scale = jax.lax.bitcast_convert_type(scale, jnp.uint8)
    codes = codes_f8.astype(jnp.bfloat16)                     # [..., K] bf16 (NO unpack)
    s = e8m0_to_fp32(scale).astype(jnp.bfloat16)              # [..., K/BLK] bf16
    K = codes.shape[-1]
    blk = codes.reshape(*codes.shape[:-1], K // BLK, BLK)
    return (blk * s[..., None]).reshape(*codes.shape[:-1], K)


def lean_dequant_fp8_ffn(x, W1c, W3c, W2c, S1, S3, S2, pew):
    """Decode dense path with fp8-CODES-RESIDENT experts: lean dequant from fp8 codes (no
    bit-unpack, 2x HBM read vs packed fp4) + the production einsums (no wsc). Measures the
    DECODE MoE cost IF experts were stored fp8-codes-resident -- the decode SIDE of the P.9
    prefill rhs-prep lever (which removes 225 ms/forward from prefill). LOSSLESS, so this
    must be bit-identical to baseline; the question its TIMING answers is whether the 2x
    expert HBM read regresses decode vs the current packed-fp4 lean-noWSC (2.46 ms/layer)."""
    fp32, bf16 = jnp.float32, jnp.bfloat16
    W1 = _dequant_fp8_lean(W1c, S1)
    W3 = _dequant_fp8_lean(W3c, S3)
    W2 = _dequant_fp8_lean(W2c, S2)
    gate = _shard_e_mid(jnp.einsum('nd,eid->nei', x, W1, preferred_element_type=fp32))
    up = _shard_e_mid(jnp.einsum('nd,eid->nei', x, W3, preferred_element_type=fp32))
    h = jax.nn.silu(gate) * up
    h = _shard_e_mid(h * _shard_e_last(pew)[..., None])
    out = _shard_e_mid(jnp.einsum('nei,edi->ned', h.astype(bf16), W2.astype(bf16)))
    return out.astype(fp32).sum(axis=1)


def dequant_only(W1u, W3u, W2u, S1, S3, S2):
    """Isolate the FP4->bf16 dequant + bf16 materialization (no matmul)."""
    W1 = _shard_e_first(_dequant_fp4_experts(W1u, S1))
    W3 = _shard_e_first(_dequant_fp4_experts(W3u, S3))
    W2 = _shard_e_first(_dequant_fp4_experts(W2u, S2))
    return W1, W3, W2


def einsum_only(x, W1, W3, W2, pew):
    """Isolate the dense matmul+mask+sum on ALREADY-bf16 experts (no dequant)."""
    fp32, bf16 = jnp.float32, jnp.bfloat16
    gate = _shard_e_mid(jnp.einsum('nd,eid->nei', x, W1, preferred_element_type=fp32))
    up = _shard_e_mid(jnp.einsum('nd,eid->nei', x, W3, preferred_element_type=fp32))
    h = jax.nn.silu(gate) * up
    h = _shard_e_mid(h * _shard_e_last(pew)[..., None])
    out = _shard_e_mid(jnp.einsum('nei,edi->ned', h.astype(bf16), W2.astype(bf16)))
    return out.astype(fp32).sum(axis=1)


def einsum_fp8w(x, W1, W3, W2, pew):
    """KERNEL-CEILING proxy: the matmul reading FP8 weights (1 byte/elem) instead of
    materialized bf16 (2 byte) -- the HBM read an in-register-dequant kernel achieves
    (it streams fp8 codes + applies the e8m0 scale in VMEM; the scale array is 1/BLK
    the size, negligible). Activation stays bf16 (the production decode operand). W*
    are pre-cast fp8 OUTSIDE timing so the matmul genuinely reads fp8 from HBM (no
    dequant materialization in the timed region). NOT bit-identical; timing floor."""
    fp32, bf16 = jnp.float32, jnp.bfloat16
    gate = _shard_e_mid(jnp.einsum('nd,eid->nei', x, W1, preferred_element_type=fp32))
    up = _shard_e_mid(jnp.einsum('nd,eid->nei', x, W3, preferred_element_type=fp32))
    h = jax.nn.silu(gate) * up
    h = _shard_e_mid(h * _shard_e_last(pew)[..., None])
    out = _shard_e_mid(jnp.einsum('nei,edi->ned', h.astype(bf16), W2,
                                  preferred_element_type=fp32))
    return out.astype(fp32).sum(axis=1)


def einsum_fp8(x, W1, W3, W2, pew):
    """As einsum_fp8w but the ACTIVATION is fp8 too (native fp8xfp8 MXU on v6e) -- the
    upper bound on the fp8 win. x is pre-cast fp8 outside timing. NOT bit-identical."""
    fp32, f8 = jnp.float32, jnp.float8_e4m3fn
    gate = _shard_e_mid(jnp.einsum('nd,eid->nei', x, W1, preferred_element_type=fp32))
    up = _shard_e_mid(jnp.einsum('nd,eid->nei', x, W3, preferred_element_type=fp32))
    h = jax.nn.silu(gate) * up
    h = _shard_e_mid(h * _shard_e_last(pew)[..., None])
    out = _shard_e_mid(jnp.einsum('nei,edi->ned', h.astype(f8), W2,
                                  preferred_element_type=fp32))
    return out.astype(fp32).sum(axis=1)


def fused_expert_ffn(x, W1u, W3u, W2u, S1, S3, S2, pew, mesh):
    """Proposed fuse: shard_map over 'attn_dp'; gmm_v2 dequants fp8 codes in-kernel.
    Token replicated to lhs=[E_local, dim], group_sizes=[1]*E_local (1 token/expert)."""
    fp32, bf16 = jnp.float32, jnp.bfloat16
    axis = mesh.shape['attn_dp']
    EP = E // axis                                        # local experts/chip (16)

    def _local(x_l, W1u_l, W3u_l, W2u_l, S1_l, S3_l, S2_l, pew_l):
        # Build the gmm rhs (fp8 codes) + per-block scale from the FP4 leaves,
        # exactly as the prefill QUANT path (rank-local: touches axis 1/2 only).
        r1, q1 = _fp4_rhs_and_scale(W1u_l, S1_l)          # [EP,dim,inter] fp8
        r3, q3 = _fp4_rhs_and_scale(W3u_l, S3_l)
        W13 = jnp.concatenate([r1, r3], axis=2)           # [EP,dim,2*inter]
        Q13 = jnp.concatenate([q1, q3], axis=3)
        W2t, Q2t = _fp4_rhs_and_scale(W2u_l, S2_l)        # [EP,inter,dim]
        lhs = jnp.broadcast_to(x_l, (EP, DIM)).astype(bf16)   # token -> all EP experts
        gsz = jnp.ones(EP, jnp.int32)                     # 1 row per group(expert)
        goff = jnp.asarray([0], jnp.int32)
        g1 = gmm_v2(lhs, W13, gsz, rhs_scale=Q13, group_offset=goff,
                    zero_initialize=False, preferred_element_type=fp32)  # [EP,2*inter]
        gate, up = jnp.split(g1, 2, axis=-1)
        h = (jax.nn.silu(gate) * up).astype(bf16)         # [EP,inter]
        g2 = gmm_v2(h, W2t, gsz, rhs_scale=Q2t, group_offset=goff,
                    zero_initialize=False, preferred_element_type=fp32)  # [EP,dim]
        local = (g2 * pew_l.reshape(EP, 1)).sum(axis=0, keepdims=True)   # [1,dim]
        return jax.lax.psum(local, 'attn_dp')             # [1,dim] full E sum
    return jax.shard_map(
        _local, mesh=mesh,
        in_specs=(P(), P('attn_dp', None, None), P('attn_dp', None, None),
                  P('attn_dp', None, None), P('attn_dp', None, None),
                  P('attn_dp', None, None), P('attn_dp', None, None),
                  P(None, 'attn_dp')),
        out_specs=P(), check_vma=False,
    )(x, W1u, W3u, W2u, S1, S3, S2, pew)


def kernel_fp8_ffn(x, W1f8, W3f8, W2f8, S1, S3, S2, pew, mesh, tile_out=256):
    """The drafted DENSE FP8-code in-register-dequant kernel (perf_dense_fp8_moe_kernel)
    for all three projections + silu + mask + E-sum, in a shard_map so the pallas_call
    runs on each chip's EP=16 local experts (like FUSED). fp8 codes are pre-unpacked
    OUTSIDE timing (the production load-time fp4->fp8 unpack; v6e can't unpack fp4
    in-kernel). The kernel reads fp8 (1 byte) + dequants the e8m0 scale IN-REGISTER ->
    NO bf16 weight materialized (the 1.70 ms/layer round-trip the production lean path
    still pays). TIMING-ONLY for the down proj: the kernel takes a SHARED activation but
    down is per-expert, so down feeds h[:,0] as a shared-x timing proxy (identical weight
    read / FLOPs; numerically wrong -> this variant is NOT bit-checked, correctness is
    the CPU oracle in perf_dense_fp8_moe_kernel)."""
    fp32, bf16 = jnp.float32, jnp.bfloat16
    EP = E // mesh.shape['attn_dp']

    def _local(x_l, W1_l, W3_l, W2_l, S1_l, S3_l, S2_l, pew_l):
        gate = dense_fp8_moe_matvec(x_l, W1_l, S1_l, tile_out=tile_out)  # [1,EP,inter]
        up = dense_fp8_moe_matvec(x_l, W3_l, S3_l, tile_out=tile_out)
        h = jax.nn.silu(gate) * up                       # [1,EP,inter] fp32
        h = h * pew_l.reshape(1, EP, 1)
        hx = h[:, 0, :].astype(bf16)                     # [1,inter] (down timing proxy)
        out = dense_fp8_moe_matvec(hx, W2_l, S2_l, tile_out=tile_out)    # [1,EP,dim]
        local = out.astype(fp32).sum(axis=1)             # [1,dim]
        return jax.lax.psum(local, 'attn_dp')            # [1,dim] full E sum
    return jax.shard_map(
        _local, mesh=mesh,
        in_specs=(P(), P('attn_dp', None, None), P('attn_dp', None, None),
                  P('attn_dp', None, None), P('attn_dp', None, None),
                  P('attn_dp', None, None), P('attn_dp', None, None),
                  P(None, 'attn_dp')),
        out_specs=P(), check_vma=False,
    )(x, W1f8, W3f8, W2f8, S1, S3, S2, pew)


def measure(f, iters, warmup=8):
    for _ in range(warmup):
        jax.block_until_ready(f())
    total = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        total.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(total), min(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--distributed", action="store_true")
    a = ap.parse_args()
    if a.distributed:
        jax.distributed.initialize()
    nd = jax.device_count()
    p0 = (not a.distributed) or jax.process_index() == 0
    devices = np.array(jax.devices()).reshape(nd)
    mesh = jax.sharding.Mesh(devices, ('attn_dp',))

    with jax.set_mesh(mesh):
        x = make_replicated((1, DIM), mesh, 1)
        W1u, S1 = make_fp4_stacked(INTER, DIM, mesh, 10)   # [E,inter,dim/2]
        W3u, S3 = make_fp4_stacked(INTER, DIM, mesh, 20)
        W2u, S2 = make_fp4_stacked(DIM, INTER, mesh, 30)   # [E,dim,inter/2]
        pew = make_pew(mesh, 40)

        base_f = jax.jit(baseline_expert_ffn)
        lean_f = jax.jit(baseline_lean_ffn)
        leannw_f = jax.jit(lambda *aa: baseline_lean_ffn(*aa, wsc=False))
        fused_f = jax.jit(lambda *args: fused_expert_ffn(*args, mesh))
        dq_f = jax.jit(dequant_only)
        es_f = jax.jit(einsum_only)
        e8w_f = jax.jit(einsum_fp8w)
        e8_f = jax.jit(einsum_fp8)
        kern_f = jax.jit(lambda *aa: kernel_fp8_ffn(*aa, mesh))
        ldf8_f = jax.jit(lean_dequant_fp8_ffn)   # fp8-codes-resident decode (P.9 lever)
        args = (x, W1u, W3u, W2u, S1, S3, S2, pew)

        # Pre-dequant bf16 experts ONCE (outside timing) for the einsum-only bench.
        W1b, W3b, W2b = jax.block_until_ready(dq_f(W1u, W3u, W2u, S1, S3, S2))
        # Pre-cast fp8 experts + fp8 activation ONCE (outside timing) -> the fp8-read
        # floor a streaming in-register-dequant kernel would hit (fp8 = 1 byte/elem).
        f8 = jnp.float8_e4m3fn
        cast8 = jax.jit(lambda a, b, c: (a.astype(f8), b.astype(f8), c.astype(f8)))
        W1f, W3f, W2f = jax.block_until_ready(cast8(W1b, W3b, W2b))
        x8 = jax.block_until_ready(jax.jit(lambda z: z.astype(f8))(x))
        # Pre-UNPACK the fp4 codes to fp8 ONCE (outside timing) for the KERNEL variant
        # = the production load-time fp4->fp8 unpack (lossless: e2m1 subset of e4m3).
        unpack8 = jax.jit(lambda w: u8_unpack_e2m1(w).astype(f8))
        W1c = jax.block_until_ready(unpack8(W1u))          # [E,inter,dim] fp8 codes
        W3c = jax.block_until_ready(unpack8(W3u))
        W2c = jax.block_until_ready(unpack8(W2u))          # [E,dim,inter] fp8 codes
        kargs = (x, W1c, W3c, W2c, S1, S3, S2, pew)

        # BIT-IDENTITY: lean dequant must match the production dequant exactly.
        yb = np.asarray(jax.device_get(jax.block_until_ready(base_f(*args))), np.float32)
        yl = np.asarray(jax.device_get(jax.block_until_ready(lean_f(*args))), np.float32)
        max_abs = float(np.abs(yb - yl).max())
        # fp8-weight drift (informational: gauges the md5-shift an fp8 kernel implies).
        ye8 = np.asarray(jax.device_get(jax.block_until_ready(
            e8w_f(x, W1f, W3f, W2f, pew))), np.float32)
        max_abs_fp8 = float(np.abs(yb - ye8).max())
        rel_fp8 = max_abs_fp8 / (float(np.abs(yb).max()) + 1e-9)
        # KERNEL sanity on v6e: confirm it lowers (no Mosaic error) + emits finite output.
        # (Not a correctness check -- down uses a shared-x timing proxy; the CPU oracle in
        #  perf_dense_fp8_moe_kernel validated correctness at 4.1e-7 rel err.)
        yk = np.asarray(jax.device_get(jax.block_until_ready(kern_f(*kargs))), np.float32)
        kern_finite = bool(np.isfinite(yk).all())
        # fp8-CODES-RESIDENT decode (P.9 lever): must be BIT-IDENTICAL to baseline (fp8
        # holds e2m1 exactly) -- proves the lever is lossless on the decode side too.
        yldf8 = np.asarray(jax.device_get(jax.block_until_ready(
            ldf8_f(x, W1c, W3c, W2c, S1, S3, S2, pew))), np.float32)
        max_abs_ldf8 = float(np.abs(yb - yldf8).max())

        bmed, bmin = measure(lambda: base_f(*args), a.iters)
        lmed, lmin = measure(lambda: lean_f(*args), a.iters)
        nmed, nmin = measure(lambda: leannw_f(*args), a.iters)
        ldf8med, ldf8min = measure(
            lambda: ldf8_f(x, W1c, W3c, W2c, S1, S3, S2, pew), a.iters)
        dmed, dmin = measure(lambda: dq_f(W1u, W3u, W2u, S1, S3, S2), a.iters)
        emed, emin = measure(lambda: es_f(x, W1b, W3b, W2b, pew), a.iters)
        e8wmed, e8wmin = measure(lambda: e8w_f(x, W1f, W3f, W2f, pew), a.iters)
        e8med, e8min = measure(lambda: e8_f(x8, W1f, W3f, W2f, pew), a.iters)
        kmed, kmin = measure(lambda: kern_f(*kargs), a.iters)
        fmed, fmin = measure(lambda: fused_f(*args), a.iters)

    if p0:
        # HBM floor/layer: per-chip resident FP4 ~8.57 GiB / 43 layers; v6e 1638 GiB/s.
        fp4_layer_gib = 8.57 / N_MOE_LAYERS
        floor_fp4 = fp4_layer_gib / 1638 * 1e3            # read fp4 once
        floor_bf16 = fp4_layer_gib * 4 / 1638 * 1e3       # read materialized bf16 once
        print(f"\n=== V4 DECODE MoE expert-FFN microbench (16-chip, N=1, E={E}/16=16 local) ===")
        print(f"dims dim={DIM} inter={INTER} top_k={TOP_K} | backend={jax.default_backend()} devices={nd}")
        print(f"\nBASELINE   (current _dequant_fp4 + einsum)    : med {bmed:7.2f} ms  min {bmin:7.2f} ms  /layer")
        print(f"LEAN       (bf16 broadcast dequant + einsum)  : med {lmed:7.2f} ms  min {lmin:7.2f} ms  /layer")
        print(f"LEAN-noWSC (lean, no intermediate _shard_e_first): med {nmed:7.2f} ms  min {nmin:7.2f} ms  /layer  <- PRODUCTION")
        print(f"LEAN-fp8res(lean dequant from fp8 codes, no unpack): med {ldf8med:7.2f} ms  min {ldf8min:7.2f} ms  /layer")
        print(f"FUSED      (gmm_v2 fp8 codes, shard_map)      : med {fmed:7.2f} ms  min {fmin:7.2f} ms  /layer")
        print(f"KERNEL     (dense fp8 in-reg dequant, shard_map): med {kmed:7.2f} ms  min {kmin:7.2f} ms  /layer")
        print(f"  dequant-only (current, materialize x3)      : med {dmed:7.2f} ms  min {dmin:7.2f} ms  /layer")
        print(f"  einsum-only  (matmul+mask+sum on bf16)      : med {emed:7.2f} ms  min {emin:7.2f} ms  /layer")
        print(f"  einsum-fp8w  (fp8 weights, bf16 act)        : med {e8wmed:7.2f} ms  min {e8wmin:7.2f} ms  /layer")
        print(f"  einsum-fp8   (fp8 weights + fp8 act)        : med {e8med:7.2f} ms  min {e8min:7.2f} ms  /layer")
        print(f"\nLEAN bit-identity vs baseline: max|Δ| = {max_abs:.3e}  ({'IDENTICAL' if max_abs==0 else 'DIFFERS'})")
        # P.9 fp8-codes-resident DECODE side: does the 2x expert HBM read regress decode?
        _f8v = ('NO REGRESSION' if ldf8min <= nmin * 1.05 else
                f'REGRESSES {ldf8min/nmin:.2f}x')
        print(f"fp8-RESIDENT decode (P.9 lever, LOSSLESS): lean-fp8 {ldf8min:.2f} vs production "
              f"lean-noWSC {nmin:.2f} ms/layer => {_f8v}  (bit-ident max|Δ|={max_abs_ldf8:.3e}; "
              f"MoE step {ldf8min*N_MOE_LAYERS:.0f} vs {nmin*N_MOE_LAYERS:.0f} ms)")
        best = min(lmin, nmin)
        print(f"DECOMPOSITION: dequant {dmed:.2f} + einsum {emed:.2f} (dequant {100*dmed/(dmed+emed):.0f}% of split)")
        print(f"WIN: lean/baseline {bmin/lmin:4.2f}x | lean-noWSC/baseline {bmin/nmin:4.2f}x | gmm {bmin/fmin:4.2f}x")
        print(f"ATTRIBUTION: baseline x {N_MOE_LAYERS} = {bmin*N_MOE_LAYERS:.1f} ms vs {DEVICE_COMPUTE_MS:.0f} ms budget "
              f"({100*bmin*N_MOE_LAYERS/DEVICE_COMPUTE_MS:.0f}%)")
        print(f"PROJECTED step device-compute: baseline {bmin*N_MOE_LAYERS:.0f} -> best {best*N_MOE_LAYERS:.0f} ms "
              f"(einsum-floor {emin*N_MOE_LAYERS:.0f}); other (attn/logits) unchanged.")
        print(f"HBM floor/layer: fp4-read-once {floor_fp4:.3f} ms | bf16-read-once {floor_bf16:.3f} ms")
        # KERNEL CEILING (roadmap #1): an in-register-dequant kernel removes the bf16
        # materialization (lean-noWSC {n}) and reads fp8 codes (1 byte) -> floor ~= the
        # fp8-read matmul. Two wins stacked: kill materialization, halve the read.
        print(f"\nFP8 matmul floor: einsum bf16 {emin:.2f} -> fp8w {e8wmin:.2f} ({emin/e8wmin:4.2f}x) "
              f"| fp8+fp8act {e8min:.2f} ({emin/e8min:4.2f}x)")
        print(f"KERNEL CEILING vs production lean-noWSC: {nmin:.2f} -> ~{e8wmin:.2f} ms/layer "
              f"({nmin/e8wmin:4.2f}x) => MoE step ~{nmin*N_MOE_LAYERS:.0f} -> ~{e8wmin*N_MOE_LAYERS:.0f} ms")
        print(f"fp8-weight drift vs baseline: max|Δ|={max_abs_fp8:.3e} rel={rel_fp8:.2e} "
              f"(informational; an fp8 kernel shifts the GATE md5)")
        # MEASURED kernel on v6e (the drafted dense fp8 in-register-dequant kernel):
        print(f"\nKERNEL on v6e: lowers+finite={kern_finite} | {kmin:.2f} ms/layer vs "
              f"lean-noWSC {nmin:.2f} ({nmin/kmin:4.2f}x) vs matmul-floor {emin:.2f} ({kmin/emin:4.2f}x)")
        print(f"  => MoE step ~{nmin*N_MOE_LAYERS:.0f} -> ~{kmin*N_MOE_LAYERS:.0f} ms "
              f"(timing-only; down=shared-x proxy; correctness=CPU oracle 4.1e-7)")


if __name__ == "__main__":
    main()

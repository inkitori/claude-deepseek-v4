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
        args = (x, W1u, W3u, W2u, S1, S3, S2, pew)

        # Pre-dequant bf16 experts ONCE (outside timing) for the einsum-only bench.
        W1b, W3b, W2b = jax.block_until_ready(dq_f(W1u, W3u, W2u, S1, S3, S2))

        # BIT-IDENTITY: lean dequant must match the production dequant exactly.
        yb = np.asarray(jax.device_get(jax.block_until_ready(base_f(*args))), np.float32)
        yl = np.asarray(jax.device_get(jax.block_until_ready(lean_f(*args))), np.float32)
        max_abs = float(np.abs(yb - yl).max())

        bmed, bmin = measure(lambda: base_f(*args), a.iters)
        lmed, lmin = measure(lambda: lean_f(*args), a.iters)
        nmed, nmin = measure(lambda: leannw_f(*args), a.iters)
        dmed, dmin = measure(lambda: dq_f(W1u, W3u, W2u, S1, S3, S2), a.iters)
        emed, emin = measure(lambda: es_f(x, W1b, W3b, W2b, pew), a.iters)
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
        print(f"LEAN-noWSC (lean, no intermediate _shard_e_first): med {nmed:7.2f} ms  min {nmin:7.2f} ms  /layer")
        print(f"FUSED      (gmm_v2 fp8 codes, shard_map)      : med {fmed:7.2f} ms  min {fmin:7.2f} ms  /layer")
        print(f"  dequant-only (current, materialize x3)      : med {dmed:7.2f} ms  min {dmin:7.2f} ms  /layer")
        print(f"  einsum-only  (matmul+mask+sum on bf16)      : med {emed:7.2f} ms  min {emin:7.2f} ms  /layer")
        print(f"\nLEAN bit-identity vs baseline: max|Δ| = {max_abs:.3e}  ({'IDENTICAL' if max_abs==0 else 'DIFFERS'})")
        best = min(lmin, nmin)
        print(f"DECOMPOSITION: dequant {dmed:.2f} + einsum {emed:.2f} (dequant {100*dmed/(dmed+emed):.0f}% of split)")
        print(f"WIN: lean/baseline {bmin/lmin:4.2f}x | lean-noWSC/baseline {bmin/nmin:4.2f}x | gmm {bmin/fmin:4.2f}x")
        print(f"ATTRIBUTION: baseline x {N_MOE_LAYERS} = {bmin*N_MOE_LAYERS:.1f} ms vs {DEVICE_COMPUTE_MS:.0f} ms budget "
              f"({100*bmin*N_MOE_LAYERS/DEVICE_COMPUTE_MS:.0f}%)")
        print(f"PROJECTED step device-compute: baseline {bmin*N_MOE_LAYERS:.0f} -> best {best*N_MOE_LAYERS:.0f} ms "
              f"(einsum-floor {emin*N_MOE_LAYERS:.0f}); other (attn/logits) unchanged.")
        print(f"HBM floor/layer: fp4-read-once {floor_fp4:.3f} ms | bf16-read-once {floor_bf16:.3f} ms")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TPU micro-benchmark: the V4 PREFILL *NON-MoE* device cost (roadmap #1, fork B).

Companion to `perf_microbench_moe_prefill.py`, which characterized the prefill MoE
(275/288/331/393 ms/fwd at GLOBAL N=512/1024/2048/4096, DOMINATED by the 225 ms/fwd
FP4->fp8 rhs-prep). To RANK prefill levers we need the FULL prefill wall = MoE +
NON-MoE; only the MoE half is measured so far. This script measures the per-chip
NON-MoE prefill ops so we can compute non-MoE's share and find the next lever.

SHARDING MODEL (confirmed): the prefill activation is token/sequence-sharded on the
attn_dp axis (tpu_runner.py:1431, `PartitionSpec(ATTN_DATA)`) -> each of the 16 chips
processes n = N/16 tokens with REPLICATED dense weights (SEQUENCE-parallel: the dense
projection / gate / HC weights are small enough to replicate per chip, unlike the
EP-sharded experts and the column-sharded vocab head). So per-chip cost = the op on
[n, dim] with full weights and NO per-projection collective. (The only true prefill
collectives are the MoE all_gather/psum, already measured in moe_prefill.) ATTENTION is
the EXCEPTION: the Mosaic kernel can't be auto-partitioned, so prefill all-gathers the
activation to REPLICATED and runs the kernel on the FULL sequence (M=N) on EVERY chip
(deepseek_v4_attention.py:239 docstring) -> attention is measured replicated at M=N (not
n). This 16x-redundant replication is a documented large-prefill inefficiency AND a lever.
The dense ops mirror moe_prefill's "replicated P() + jit -> per-chip cost" convention
(each of the 16 identical ranks runs in parallel, so timing one chip = per-chip cost).

Ops (REAL fns imported where possible; mirrored matmuls otherwise), at real V4-Flash
dims, per chip at n=N/16, rolled up by the REAL per-layer counts (43 layers =
2 SWA + 21 CSA + 20 HCA; HC pre+post run 2x/layer = x86):
  PROJ      q_a/q_b/kv/o_a(grouped)/o_b bf16 matmuls (x43)
  GATE      real `gate_forward`: [n,dim]@[dim,256] f32 + top_k(6) (x43)
  HC        real `hc_pre`+`hc_post`: [n,16384]@[16384,24] mix + 19-iter sinkhorn on
            [n,4,4] + the bsij,bsid combine, fp32 (x2x43 = x86)
  NORM      real `rms_norm` [n,dim] fp32, costed x4x43 (attn+ffn+q+kv, UPPER bound: all
            at the largest [n,dim] size; q/kv norms are smaller)
  ATTN      real `sparse_attn_kernel` (Mosaic; the PRODUCTION prefill path,
            deepseek_v4_attention.py:950 -> :239) under a fully-REPLICATED shard_map at
            the prefill_swa/csa/hca presets, M=N (every chip runs the full-seq kernel;
            the 16x-redundant replication is a documented inefficiency + lever). (x2/x21/x20)
  COMPRESS  CSA compressor coff=2 ([n,dim]@[dim,1024] x2, x21) + HCA coff=1 (x512, x20)
  INDEXER   CSA only: wq_b [n,1024]@[1024,8192] + weights_proj [n,dim]@[dim,64] fp32 (x21)
  LOGITS    head: [1,dim]@[dim,vocab/16] f32 (column-sharded head_w, last-token) (x1)

Run (TPU FREE; SYNC first -- mh_run runs each host's own clone):
    scripts/full_slice_v4_sync.sh
    MH_TIMEOUT=900 scripts/full_slice_v4_mh_run.sh scripts/perf_microbench_prefill_nonmoe.py --distributed
"""
import argparse
import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from tpu_inference.models.jax.deepseek_v4 import hc_pre, hc_post
from tpu_inference.layers.jax.attention.deepseek_v4_attention import rms_norm
from tpu_inference.kernels.sparse_attn.kernel import sparse_attn_kernel
from tpu_inference.layers.jax.moe.deepseek_v4_moe import gate_forward, GateParams

# Reuse the decode bench's timing harness + the shared dims.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_microbench_moe_decode import DIM, E, TOP_K, N_MOE_LAYERS, measure  # noqa: E402

# --- real V4-Flash config (config.json) ---
HEAD_DIM = 512          # head_dim (also MLA kv_lora_rank)
N_HEADS = 64            # num_attention_heads
Q_LORA = 1024           # q_lora_rank
O_LORA = 1024           # o_lora_rank
O_GROUPS = 8            # o_groups
VOCAB = 129280          # vocab_size
HC = 4                  # hc_mult
SINK_ITERS = 20         # hc_sinkhorn_iters (code does iters-1 = 19 sinkhorn steps)
EPS = 1e-6              # rms_norm_eps
HC_EPS = 1e-6           # hc_eps
QKV_OUT = N_HEADS * HEAD_DIM      # 32768  (q_b output; attn output before o)
O_A_OUT = O_GROUPS * O_LORA       # 8192   (o_a output / o_b input)

# attention preset dims (mirror perf_microbench_sparse_attn.py)
H, D, WIN = N_HEADS, HEAD_DIM, 128
SCALE = 1.0 / (D ** 0.5)
# prefill presets -> (B, M, K, N) at query count s (per-chip n)
PREFILL_PRESETS = {
    "swa": lambda s: (1, s, min(s, WIN), s),
    "csa": lambda s: (1, s, min(s, WIN) + min(512, s // 4), s + s // 4),
    "hca": lambda s: (1, s, min(s, WIN) + s // 128, s + s // 128),
}

# REAL per-layer counts (config compress_ratios: 2 SWA(0) + 21 CSA(4) + 20 HCA(128) = 43)
L = N_MOE_LAYERS  # 43
N_SWA, N_CSA, N_HCA = 2, 21, 20
N_COMPRESS = N_CSA + N_HCA  # 41 layers run a compressor
N_INDEXER = N_CSA           # 21 layers run an indexer (CSA only)

# MoE per-forward ms (moe_prefill P.9; for the % share readout)
MOE_FWD = {512: 275.0, 1024: 288.0, 2048: 331.0, 4096: 393.0}


def make_rep(shape, mesh, seed, dtype=jnp.bfloat16):
    """Replicated (P()) synthetic array, value O(1). Per-chip cost == one rank."""
    sh = NamedSharding(mesh, P())
    rng = np.random.default_rng(seed)
    base = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    return jax.make_array_from_callback(
        shape, sh, lambda idx, _b=base: _b[idx]).astype(dtype)


def make_rep_raw(arr, mesh):
    """Replicated array preserving `arr`'s numpy dtype (for int idx tensors)."""
    sh = NamedSharding(mesh, P())
    return jax.make_array_from_callback(arr.shape, sh, lambda idx, _a=arr: _a[idx])


def mm(x, w):
    """x[.., in] @ w[out, in].T -> [.., out], bf16 in / fp32 accum / bf16 out."""
    return jnp.einsum('ni,oi->no', x, w, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16)


def mm_grouped(xg, wg):
    """grouped o_a: xg[n, G, C] @ wg[G, R, C] -> [n, G, R] (block-diagonal o-proj)."""
    return jnp.einsum('ngc,grc->ngr', xg, wg,
                      preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def mm_f32(x, w):
    """fp32 matmul (indexer / logits): x[.., in] @ w[out, in].T -> [.., out] fp32."""
    return jnp.einsum('ni,oi->no', x, w, preferred_element_type=jnp.float32)


def hc_block(x, hc_fn, hc_scale, hc_base):
    """One attn-or-ffn sub-block's HC overhead: hc_pre (mix + 19-iter sinkhorn) +
    hc_post (the bsij,bsid combine). Excludes the attn/moe in between."""
    y, post, comb = hc_pre(x, hc_fn, hc_scale, hc_base, HC, SINK_ITERS, EPS, HC_EPS)
    return hc_post(y, x, post, comb)


def attn_kernel_rep(q, kv, sink, idx, mesh):
    """Production prefill attention (deepseek_v4_attention.py:239): the Mosaic
    `sparse_attn_kernel` under a fully-REPLICATED shard_map -- every chip runs the
    whole kernel on the full (all-gathered) prefill sequence. Mirrors the prod call."""
    return jax.shard_map(
        functools.partial(sparse_attn_kernel, softmax_scale=SCALE),
        mesh=mesh, in_specs=(P(),) * 4, out_specs=P(), check_vma=False,
    )(q, kv, sink, idx)


def attn_kernel_sharded(q, kv, sink, idx, mesh):
    """THE LEVER: shard the prefill attention map over the QUERY/token axis (M) so each
    chip runs the kernel on only M/axis queries (kv stays REPLICATED -- all keys needed).
    This is the kernel docstring's 'future optimization is to shard the map over the
    prefill token axis'. Per-chip cost ~ replicated/axis. Decode is untouched (M=1)."""
    return jax.shard_map(
        functools.partial(sparse_attn_kernel, softmax_scale=SCALE),
        mesh=mesh,
        in_specs=(P(None, 'attn_dp', None, None), P(), P(), P(None, 'attn_dp', None)),
        out_specs=P(None, 'attn_dp', None, None), check_vma=False,
    )(q, kv, sink, idx)


def make_attn_rep(B, M, K, N, mesh, seed):
    """Replicated sparse_attn inputs (q/kv bf16, sink fp32, topk int32 w/ 3% -1 pad)."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, M, H, D)).astype(np.float32)
    kv = rng.standard_normal((B, N, D)).astype(np.float32)
    sink = rng.standard_normal(H).astype(np.float32)
    idx = rng.integers(0, N, size=(B, M, K)).astype(np.int32)
    idx[rng.random((B, M, K)) < 0.03] = -1
    return (make_rep_raw(q, mesh).astype(jnp.bfloat16),
            make_rep_raw(kv, mesh).astype(jnp.bfloat16),
            make_rep_raw(sink, mesh), make_rep_raw(idx, mesh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--seqs", type=str, default="512,1024,2048,4096",
                    help="GLOBAL prefill seq_len sweep (per-chip n = N/16)")
    a = ap.parse_args()
    if a.distributed:
        jax.distributed.initialize()
    nd = jax.device_count()
    p0 = (not a.distributed) or jax.process_index() == 0
    devices = np.array(jax.devices()).reshape(nd)
    mesh = jax.sharding.Mesh(devices, ('attn_dp',))
    seqs = [int(s) for s in a.seqs.split(",")]
    assert all(s % nd == 0 for s in seqs), f"each seq must be divisible by {nd}"

    rows = []
    with jax.set_mesh(mesh):
        # ---- n-INDEPENDENT weights (built once) ----
        wq_a = make_rep((Q_LORA, DIM), mesh, 1)            # [1024, 4096]
        wq_b = make_rep((QKV_OUT, Q_LORA), mesh, 2)        # [32768, 1024]
        wkv = make_rep((HEAD_DIM, DIM), mesh, 3)           # [512, 4096]
        wo_a = make_rep((O_GROUPS, O_LORA, QKV_OUT // O_GROUPS), mesh, 4)  # [8,1024,4096]
        wo_b = make_rep((DIM, O_A_OUT), mesh, 5)           # [4096, 8192]
        nrm_w = make_rep((DIM,), mesh, 6, jnp.float32)
        hc_fn = make_rep(((2 + HC) * HC, HC * DIM), mesh, 7, jnp.float32)   # [24, 16384]
        hc_scale = make_rep((3,), mesh, 8, jnp.float32)
        hc_base = make_rep(((2 + HC) * HC,), mesh, 9, jnp.float32)          # [24]
        gp = GateParams(weight=make_rep((E, DIM), mesh, 10, jnp.float32),
                        bias=make_rep((E,), mesh, 11, jnp.float32),
                        tid2eid=None, score_func="sigmoid", route_scale=1.0, top_k=TOP_K)
        wcmp_csa = make_rep((2 * HEAD_DIM, DIM), mesh, 12)   # coff=2 -> [1024, 4096]
        wcmp_hca = make_rep((HEAD_DIM, DIM), mesh, 13)       # coff=1 -> [512, 4096]
        widx_q = make_rep((N_HEADS * 128, Q_LORA), mesh, 14, jnp.float32)  # [8192,1024]
        widx_w = make_rep((N_HEADS, DIM), mesh, 15, jnp.float32)           # [64, 4096]
        head_w = make_rep((VOCAB // nd, DIM), mesh, 16, jnp.float32)       # col-sharded

        # ---- jit once (shape-keyed recompile across n) ----
        f_qa = jax.jit(mm); f_qb = jax.jit(mm); f_kv = jax.jit(mm)
        f_oa = jax.jit(mm_grouped); f_ob = jax.jit(mm)
        f_gate = jax.jit(lambda x, ids: gate_forward(x, ids, gp))
        f_hc = jax.jit(hc_block)
        f_nrm = jax.jit(lambda x: rms_norm(x, nrm_w, EPS))
        f_cmp_csa = jax.jit(lambda x: (mm(x, wcmp_csa), mm(x, wcmp_csa)))
        f_cmp_hca = jax.jit(lambda x: (mm(x, wcmp_hca), mm(x, wcmp_hca)))
        f_idx = jax.jit(lambda qr, x: (mm_f32(qr, widx_q), mm_f32(x, widx_w)))
        f_attn = jax.jit(lambda q, kv, s, idx: attn_kernel_rep(q, kv, s, idx, mesh))
        f_attn_sh = jax.jit(lambda q, kv, s, idx: attn_kernel_sharded(q, kv, s, idx, mesh))
        f_log = jax.jit(mm_f32)

        # logits: last-token only, once per forward (n-independent).
        xl = make_rep((1, DIM), mesh, 900, jnp.float32)
        jax.block_until_ready(f_log(xl, head_w))
        _, log_min = measure(lambda: f_log(xl, head_w), a.iters)

        def t(f, *args):
            jax.block_until_ready(f(*args))
            return measure(lambda: f(*args), a.iters)[1]

        for N in seqs:
            n = N // nd
            x = make_rep((n, DIM), mesh, 100 + N)              # [n, 4096] bf16
            qr = make_rep((n, Q_LORA), mesh, 110 + N)          # [n, 1024]
            xg = make_rep((n, O_GROUPS, QKV_OUT // O_GROUPS), mesh, 120 + N)  # [n,8,4096]
            xob = make_rep((n, O_A_OUT), mesh, 130 + N)        # [n, 8192]
            xf = make_rep((n, DIM), mesh, 140 + N, jnp.float32)
            ids = make_rep_raw(
                np.random.default_rng(N).integers(0, VOCAB, n).astype(np.int32), mesh)
            xhc = make_rep((1, n, HC, DIM), mesh, 150 + N)     # [1, n, 4, 4096]
            xn = make_rep((1, n, DIM), mesh, 160 + N)          # [1, n, 4096]

            r = dict(N=N, n=n)
            r['qa'] = t(f_qa, x, wq_a)
            r['qb'] = t(f_qb, qr, wq_b)
            r['kv'] = t(f_kv, x, wkv)
            r['oa'] = t(f_oa, xg, wo_a)
            r['ob'] = t(f_ob, xob, wo_b)
            r['proj'] = r['qa'] + r['qb'] + r['kv'] + r['oa'] + r['ob']
            r['gate'] = t(f_gate, xf, ids)
            r['hc'] = t(f_hc, xhc, hc_fn, hc_scale, hc_base)
            r['norm'] = t(f_nrm, xn)
            r['cmp_csa'] = t(f_cmp_csa, x)
            r['cmp_hca'] = t(f_cmp_hca, x)
            r['idx'] = t(f_idx, qr, xf)
            # production attention = REPLICATED at full seq (M=N); the LEVER = sharded
            # over the token axis (M=N/axis). Measure both; guard the (new) sharded path.
            attn, attn_sh = {}, {}
            for name, fn in PREFILL_PRESETS.items():
                B, M, K, Nk = fn(N)
                ai = make_attn_rep(B, M, K, Nk, mesh, 800 + N)
                attn[name] = t(f_attn, *ai)
                try:
                    attn_sh[name] = t(f_attn_sh, *ai)
                except Exception as e:  # noqa: BLE001 -- feasibility probe
                    attn_sh[name] = float('nan')
                    if p0 and name == 'csa':
                        print(f"  [sharded-attn N={N} {type(e).__name__}: {str(e)[:140]}]")
            r['attn_swa'], r['attn_csa'], r['attn_hca'] = (
                attn['swa'], attn['csa'], attn['hca'])
            r['asw_sh'], r['acs_sh'], r['ahc_sh'] = (
                attn_sh['swa'], attn_sh['csa'], attn_sh['hca'])
            rows.append(r)

    if p0:
        print(f"\n=== V4 PREFILL NON-MoE microbench (16-chip seq-parallel, n=N/{nd}) ===")
        print(f"backend={jax.default_backend()} devices={nd} | dim={DIM} heads={N_HEADS} "
              f"head_dim={HEAD_DIM} vocab={VOCAB}")
        print(f"layers: {L} = {N_SWA} SWA + {N_CSA} CSA + {N_HCA} HCA; HC x2/layer; "
              f"logits last-token x1 = {log_min:.3f} ms")
        print(f"NOTE: ATTN (aSWA/aCSA/aHCA) is the Mosaic kernel REPLICATED at M=N (full "
              f"seq, every chip); all other ops are seq-parallel at n=N/{nd}.")
        print(f"\n{'N':>5} {'n':>4} | per-chip min ms/CALL: "
              f"{'proj':>6} {'gate':>6} {'hc':>6} {'norm':>6} "
              f"{'aSWA':>6} {'aCSA':>6} {'aHCA':>6} {'cCSA':>6} {'cHCA':>6} {'idx':>6}")
        for r in rows:
            print(f"{r['N']:>5} {r['n']:>4} | "
                  f"{r['proj']:>6.3f} {r['gate']:>6.3f} {r['hc']:>6.3f} {r['norm']:>6.3f} "
                  f"{r['attn_swa']:>6.3f} {r['attn_csa']:>6.3f} {r['attn_hca']:>6.3f} "
                  f"{r['cmp_csa']:>6.3f} {r['cmp_hca']:>6.3f} {r['idx']:>6.3f}")

        print(f"\n=== PER-FORWARD ROLLUP (per chip, ms) — non-MoE vs MoE ===")
        print(f"{'N':>5} | {'PROJ':>7} {'GATE':>6} {'HC':>7} {'NORM':>6} {'ATTN':>7} "
              f"{'CMP':>6} {'IDX':>6} {'LOG':>5} | {'nonMoE':>7} {'MoE':>6} {'share':>6}")
        for r in rows:
            proj = r['proj'] * L
            gate = r['gate'] * L
            hc = r['hc'] * 2 * L                       # pre+post, attn + ffn sub-blocks
            norm = r['norm'] * 4 * L                   # attn+ffn+q+kv (UPPER bound)
            attn = (r['attn_swa'] * N_SWA + r['attn_csa'] * N_CSA
                    + r['attn_hca'] * N_HCA)
            cmp = r['cmp_csa'] * N_CSA + r['cmp_hca'] * N_HCA
            idx = r['idx'] * N_INDEXER
            log = log_min
            nonmoe = proj + gate + hc + norm + attn + cmp + idx + log
            moe = MOE_FWD.get(r['N'], float('nan'))
            share = 100.0 * nonmoe / (nonmoe + moe) if moe == moe else float('nan')
            print(f"{r['N']:>5} | {proj:>7.1f} {gate:>6.1f} {hc:>7.1f} {norm:>6.1f} "
                  f"{attn:>7.1f} {cmp:>6.1f} {idx:>6.1f} {log:>5.1f} | "
                  f"{nonmoe:>7.1f} {moe:>6.0f} {share:>5.1f}%")
        print(f"\n=== ATTENTION LEVER: shard prefill attn over token axis (M=N -> M=N/{nd}) ===")
        print(f"{'N':>5} | {'ATTN_repl':>9} {'ATTN_shrd':>9} {'saving':>7} | "
              f"{'wall_now':>8} {'wall_lever':>10} {'wall_-%':>7}")
        for r in rows:
            a_repl = r['attn_swa'] * N_SWA + r['attn_csa'] * N_CSA + r['attn_hca'] * N_HCA
            a_shrd = r['asw_sh'] * N_SWA + r['acs_sh'] * N_CSA + r['ahc_sh'] * N_HCA
            floor = (r['proj'] + r['gate']) * L + r['hc'] * 2 * L + r['norm'] * 4 * L \
                + r['cmp_csa'] * N_CSA + r['cmp_hca'] * N_HCA + r['idx'] * N_INDEXER + log_min
            moe = MOE_FWD.get(r['N'], float('nan'))
            wall_now = floor + a_repl + moe
            wall_lever = floor + a_shrd + moe
            pct = 100.0 * (wall_now - wall_lever) / wall_now
            print(f"{r['N']:>5} | {a_repl:>9.1f} {a_shrd:>9.1f} {a_repl - a_shrd:>7.1f} | "
                  f"{wall_now:>8.1f} {wall_lever:>10.1f} {pct:>6.1f}%")

        print(f"\nREAD: non-MoE 'share' = nonMoE/(nonMoE+MoE) of the prefill device wall. "
              f"If small, prefill stays MoE-dominated (rhs-prep lever, roadmap #2). "
              f"CAVEATS: ATTN runs REPLICATED at M=N (production kernel path) -> the "
              f"16x-redundant replication is itself a lever; NORM x4 is an UPPER bound.")


if __name__ == "__main__":
    main()

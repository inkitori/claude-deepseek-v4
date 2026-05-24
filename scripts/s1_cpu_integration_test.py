"""S1 decisive full-LAYER integration test (CPU, seconds).

Isolates ONE attention layer (default layer 2, ratio=4 / HCA) of the truncated
V4-Flash cfg with PEAKED weights (scale 0.5) and checks whether the decode path
(`attention_init_state_from_prefill` + `attention_decode_step` stepped to P)
matches the fresh-prefill reference (`attention_prefill(x[:P+1])[:, P]`) — the
same prefill-vs-decode equivalence S1 violates, but on a SINGLE attention layer
(no MoE, no residual, no multi-layer accumulation).

Freqs dispatch mirrors transformer_body_forward (source of truth): ratio>0 ->
compress freqs, ratio==0 -> swa freqs.

Stage 1: relErr(y_decode, y_prefill) per decode position. >0.05 => bug in THIS
layer's attention_decode_step integration.

Stage 2 (auto, at first diverging P): monkeypatch sparse_attn to capture
(q, kv_buffer, topk_idxs, o) for both the prefill[:P+1] call and the decode-at-P
call, then:
  - relErr(q): should be ~0 (same x_step, same freqs).
  - relErr(o): the sparse_attn output. If q matches but o differs => bug is in
    the kv buffer or the topk SELECTION (not the output projection).
  - map both paths' topk indices to LOGICAL positions (SWA position set +
    compressed-position set) and compare. Equal sets + different o => wrong KV
    VALUES (compressor/seed/threading). Different sets => wrong SELECTION (topk).

Usage: python scripts/s1_cpu_integration_test.py [scale=0.5] [n_layers=4] [T=8] [N=12] [seed=0] [layer=2]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp

from s1_cpu_repro_v4flash import make_v4_flash_truncated_cfg
from s1_cpu_repro_peaked import make_scaled_params
from tpu_inference.models.jax.deepseek_v4 import make_freqs_cis
from tpu_inference.layers.jax.attention import deepseek_v4_attention as A


def relerr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def decode_window_logical_positions(win, P):
    """For decode at start_pos=P, ring slot r holds logical position
    p = largest p<=P with p%win==r."""
    out = {}
    for r in range(win):
        p = P - ((P - r) % win)
        if 0 <= p <= P:
            out[r] = p
    return out


def main():
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    LAYER = int(sys.argv[6]) if len(sys.argv) > 6 else 2

    cfg = make_v4_flash_truncated_cfg(n_layers=n_layers, n_experts=8)
    params = make_scaled_params(cfg, scale, seed=seed)
    max_seq = 4096
    swa, comp = make_freqs_cis(cfg, max_seq)

    attn = params.layers[LAYER].attn
    ratio = attn.compress_ratio
    fc = comp if ratio > 0 else swa
    win = attn.window_size
    idx_hd = cfg.index_head_dim if (ratio == 4 and attn.indexer is not None) else 0
    print(f"Setup: layer={LAYER} ratio={ratio} win={win} "
          f"indexer={'yes' if attn.indexer is not None else 'no'} "
          f"freqs={'comp' if ratio > 0 else 'swa'} scale={scale} T={T} N={N}")

    B = 1
    dim = cfg.hidden_size
    rng = np.random.default_rng(seed=1234)
    # Attention input (post hc_pre + rms_norm); bf16 to match production dtype.
    x = jnp.asarray(rng.standard_normal((B, T + N, dim)).astype(np.float32),
                    dtype=jnp.bfloat16)

    # ---- Stage 1: per-position y relErr ----
    print("\n=== Stage 1: relErr(y_decode, y_prefill) per decode position ===")
    state = A.attention_init_state_from_prefill(
        x[:, :T], attn, fc, cfg_max_seq_len=max_seq,
        cfg_index_head_dim=idx_hd, dtype=jnp.bfloat16)
    first_bad_P = None
    for step in range(N):
        P = T + step
        y_ref = A.attention_prefill(x[:, :P + 1], attn, fc)[:, P]
        state, y_dec = A.attention_decode_step(
            x[:, P:P + 1], jnp.int32(P), attn, fc, state)
        e = relerr(y_dec[:, 0], y_ref)
        bad = e > 0.05
        if bad and first_bad_P is None:
            first_bad_P = P
        print(f"  {'**' if bad else '  '} step={step:>2} pos={P:>2}  relErr(y)={e:.4f}")

    if first_bad_P is None:
        print("\nNO divergence in this layer's attention_decode_step "
              "(y matches prefill). Bug is downstream (MoE/residual/multi-layer).")
        return 0

    # ---- Stage 2: localize at first diverging position ----
    P = first_bad_P
    print(f"\n=== Stage 2: localize at first diverging pos P={P} ===")
    captured = {}
    orig = A.sparse_attn

    def spy(tag):
        def f(q, kv, attn_sink, topk_idxs, softmax_scale):
            out = orig(q, kv, attn_sink, topk_idxs, softmax_scale)
            captured.setdefault(tag, []).append(
                dict(q=np.asarray(q, np.float32), kv=np.asarray(kv, np.float32),
                     topk=np.asarray(topk_idxs), out=np.asarray(out, np.float32)))
            return out
        return f

    # Prefill capture (single call, query rows 0..P; we want row P).
    A.sparse_attn = spy("pre")
    _ = A.attention_prefill(x[:, :P + 1], attn, fc)
    # Decode capture: fresh state, step T..P; keep the last call (at P).
    A.sparse_attn = spy("dec")
    st = A.attention_init_state_from_prefill(
        x[:, :T], attn, fc, cfg_max_seq_len=max_seq,
        cfg_index_head_dim=idx_hd, dtype=jnp.bfloat16)
    for p in range(T, P + 1):
        st, _ = A.attention_decode_step(x[:, p:p + 1], jnp.int32(p), attn, fc, st)
    A.sparse_attn = orig

    pre = captured["pre"][0]
    dec = captured["dec"][-1]  # call at start_pos=P
    S = P + 1

    # q at position P
    q_pre = pre["q"][:, P]          # [B,H,D]
    q_dec = dec["q"][:, 0]
    print(f"  relErr(q@P)           = {relerr(q_dec, q_pre):.4f}")
    # sparse_attn output o at position P
    o_pre = pre["out"][:, P]
    o_dec = dec["out"][:, 0]
    print(f"  relErr(o=sparse_attn) = {relerr(o_dec, o_pre):.4f}")

    # ---- topk -> logical position sets ----
    tp_pre = pre["topk"][0, P]     # indices into kv_full [0,S)U[S,S+S//ratio)
    tp_dec = dec["topk"][0, 0]     # indices into [0,win)U[win,win+extra)
    # prefill: <S => SWA position; >=S => compressed pos (idx-S)
    pre_swa = sorted({int(i) for i in tp_pre if 0 <= i < S})
    pre_cmp = sorted({int(i) - S for i in tp_pre if i >= S})
    # decode: <win => ring slot -> logical pos; >=win => compressed pos (idx-win)
    ringmap = decode_window_logical_positions(win, P)
    dec_swa = sorted({ringmap[int(i)] for i in tp_dec if 0 <= i < win and int(i) in ringmap})
    dec_cmp = sorted({int(i) - win for i in tp_dec if i >= win})

    print(f"  SWA positions  prefill={pre_swa}")
    print(f"  SWA positions  decode ={dec_swa}")
    print(f"  SWA set equal? {pre_swa == dec_swa}")
    print(f"  CMP positions  prefill={pre_cmp}")
    print(f"  CMP positions  decode ={dec_cmp}")
    print(f"  CMP set equal? {pre_cmp == dec_cmp}")

    # ---- compare gathered KV values for the SHARED logical positions ----
    # Build position->kv map for both buffers.
    def gather_map(buf, idxs, idx_to_pos):
        m = {}
        for i in idxs:
            i = int(i)
            if i in idx_to_pos:
                m[idx_to_pos[i]] = buf[0, i]
        return m
    # prefill kv_full: row j<S is SWA pos j; row S+k is comp pos k
    pre_swa_map = {j: pre["kv"][0, j] for j in pre_swa}
    pre_cmp_map = {k: pre["kv"][0, S + k] for k in pre_cmp}
    dec_swa_map = {ringmap[int(i)]: dec["kv"][0, int(i)]
                   for i in tp_dec if 0 <= i < win and int(i) in ringmap}
    dec_cmp_map = {int(i) - win: dec["kv"][0, int(i)] for i in tp_dec if i >= win}

    shared_swa = sorted(set(pre_swa_map) & set(dec_swa_map))
    if shared_swa:
        es = max(relerr(dec_swa_map[p], pre_swa_map[p]) for p in shared_swa)
        print(f"  max relErr(SWA kv) over shared positions = {es:.4f}")
    shared_cmp = sorted(set(pre_cmp_map) & set(dec_cmp_map))
    if shared_cmp:
        ec = max(relerr(dec_cmp_map[p], pre_cmp_map[p]) for p in shared_cmp)
        print(f"  max relErr(CMP kv) over shared positions = {ec:.4f}  "
              f"(positions {shared_cmp})")
        for p in shared_cmp:
            print(f"      cmp pos {p}: relErr={relerr(dec_cmp_map[p], pre_cmp_map[p]):.4f}")

    print("\n=== VERDICT ===")
    if relerr(q_dec, q_pre) > 0.05:
        print("q diverges -> bug upstream of sparse_attn (q computation/freqs).")
    elif pre_swa != dec_swa or pre_cmp != dec_cmp:
        print("topk SELECTION differs -> bug in topk index assembly "
              "(window/indexer/get_compress_topk).")
    elif shared_cmp and max(relerr(dec_cmp_map[p], pre_cmp_map[p]) for p in shared_cmp) > 0.05:
        print("CMP kv VALUES differ -> bug in compressed-kv threading/seed "
              "(compressor_decode_step / init_state compressed seed).")
    elif shared_swa and max(relerr(dec_swa_map[p], pre_swa_map[p]) for p in shared_swa) > 0.05:
        print("SWA kv VALUES differ -> bug in SWA ring-buffer write/seed.")
    else:
        print("q, topk sets, and gathered kv all match but o differs -> "
              "re-examine sparse_attn inputs (sink / dtype / ordering).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

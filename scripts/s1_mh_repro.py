"""Multi-host SHARDED reproducer + state-diff for S1 on the real v6e-32 TPU.

Runs the truncated V4-Flash config (random weights, few layers) across ALL 8
hosts / 32 chips under a JAX mesh, in the same sharding+donation context S1
lives in. CPU and a lone TPU host can't reproduce S1 (CPU passes; a single host
can't even boot the slice), so this is the cheap inner loop: minutes to compile,
no 543 GiB load.

Production V4 parallelism (layers/common/sharding.py): num_kv_heads=1 + bf16 =>
attn_dp == tensor_parallelism == 32, model=expert=1. KV cache is replicated
P() (model_loader get_flax_model). So the bug context is: experts + attention
sharded on attn_dp, KV state buffers replicated P(), buffers donated -> reshard
between sharded compute and the replicated donated buffers.

Launch:
  scripts/full_slice_v4_mh_run.sh scripts/s1_mh_repro.py [mode T N n_layers action n_experts]
  mode    : 'replicated' (all P(), proxy for single-device) | 'sharded' (attn_dp).
  action  : 'repro' (default; donated decode vs teacher-forced oracle, prints VERDICT)
          | 'diff'  (donated vs NON-donated decode; per-(step,layer,field) buffer diff
                     to localize which packed state field the donation corrupts).
  n_experts: experts in the truncated config (default 8). Use 32 for attn_dp=32.

Every process prints the VERDICT line so the launcher's head-log tail shows it.
"""
import os
import sys
import time
import functools

import numpy as np
import jax
import jax.numpy as jnp

jax.distributed.initialize()  # Cloud-TPU metadata auto-detect; needs all 8 hosts

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_cpu_repro_v4flash import make_v4_flash_truncated_cfg, make_random_params  # noqa: E402

from tpu_inference.models.jax.deepseek_v4 import (  # noqa: E402
    deepseek_v4_run_with_decode_state,
    transformer_body_forward,
    head_forward,
    make_freqs_cis,
    transformer_body_layout,
    v4_layer_packed_sizes_from_cfg,
)

P = jax.sharding.PartitionSpec
PID = jax.process_index()
FIELD_NAMES = ("kv_cache", "compressor_kv_state", "compressor_score_state",
               "indexer_kv_state", "indexer_score_state", "indexer_kv_cache")


def log(*a):
    if PID == 0:
        print(*a, flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "replicated"
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    n_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    action = sys.argv[5] if len(sys.argv) > 5 else "repro"
    n_experts = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    if mode not in ("replicated", "sharded"):
        raise SystemExit(f"mode '{mode}' not in (replicated, sharded)")
    if action not in ("repro", "diff"):
        raise SystemExit(f"action '{action}' not in (repro, diff)")

    log(f"[s1-mh] mode={mode} action={action} T={T} N={N} n_layers={n_layers} "
        f"n_experts={n_experts} devices={jax.device_count()} procs={jax.process_count()}")

    cfg = make_v4_flash_truncated_cfg(n_layers=n_layers, n_experts=n_experts)
    params = make_random_params(cfg, seed=0)
    max_seq = 4096
    swa, comp = make_freqs_cis(cfg, max_seq)
    sizes = v4_layer_packed_sizes_from_cfg(cfg, max_seq, batch_size=1)

    rng = np.random.default_rng(seed=1234)
    ids_full = rng.integers(0, cfg.vocab_size, size=(1, T + N)).astype(np.int32)

    # Mesh carries production's named axes (the functional fwd references
    # 'attn_dp' for MoE expert sharding). 'replicated': everything size-1 except
    # a dummy 'model' holding all devices (fully replicated; proxy for
    # single-device TPU+donation). 'sharded': experts/attention parallel across
    # attn_dp like production, KV state still replicated P().
    AXES = ("data", "attn_dp", "attn_dp_expert", "expert", "model")
    nd = jax.device_count()
    if mode == "replicated":
        shape = (1, 1, 1, 1, nd)
    else:
        adp = nd
        while adp > 1 and (n_experts % adp or nd % adp):
            adp //= 2
        shape = (1, adp, 1, 1, nd // adp)
    # Plain Mesh ctor -> Auto axis types (classic GSPMD; with_sharding_constraint
    # reshards). jax.make_mesh defaults to Explicit (assert-only) -> wrong here.
    devices = np.array(jax.devices()).reshape(shape)
    mesh = jax.sharding.Mesh(devices, AXES)
    rep = jax.sharding.NamedSharding(mesh, P())
    log(f"[s1-mh] mesh shape={dict(zip(AXES, shape))}")

    def put(x):
        # multi-controller-safe replicated placement (device_put-resharding
        # process-local arrays to global P() triggers a tiled allgather OOM).
        x = np.asarray(x)
        return jax.make_array_from_callback(x.shape, rep, lambda idx, _x=x: _x[idx])

    put_tree = lambda t: jax.tree_util.tree_map(put, t)

    with jax.set_mesh(mesh):
        params_r = put_tree(params)
        swa_r, comp_r = put(swa), put(comp)

        def _head(h):
            return head_forward(
                h, params_r.head_w, params_r.final_norm_w,
                params_r.hc_head_fn, params_r.hc_head_scale, params_r.hc_head_base,
                cfg.rms_norm_eps, cfg.hc_eps)

        @jax.jit
        def jit_oracle(ids):
            h = transformer_body_forward(ids, params_r, swa_r, comp_r, cfg)
            return jnp.argmax(_head(h), axis=-1)  # [1, T+N]

        def _prefill(kv_caches, ids):
            new_kv, _h = deepseek_v4_run_with_decode_state(
                kv_caches, ids, params_r, swa_r, comp_r, cfg,
                state_max_seq_len=max_seq, is_decode_step=False, start_pos=jnp.int32(0))
            return new_kv

        def _decode(kv_caches, ids_step, start_pos):
            new_kv, h = deepseek_v4_run_with_decode_state(
                kv_caches, ids_step, params_r, swa_r, comp_r, cfg,
                state_max_seq_len=max_seq, is_decode_step=True, start_pos=start_pos)
            return new_kv, jnp.argmax(_head(h)[0, 0])

        def make_jits(donate):
            da = (0,) if donate else ()
            return (jax.jit(_prefill, donate_argnums=da),
                    jax.jit(_decode, donate_argnums=da))

        def run_inc(jit_prefill, jit_decode, snapshot):
            kv = [put(np.zeros((s,), dtype=np.float32)) for s in sizes]
            kv = jit_prefill(kv, put(ids_full[:, :T]))
            ams, snaps = [], []
            for step in range(N):
                pos = T + step
                kv, am = jit_decode(kv, put(ids_full[:, pos:pos + 1]), put(np.int32(pos)))
                ams.append(int(am))
                if snapshot:
                    snaps.append([np.asarray(b) for b in kv])
            return ams, snaps

        t0 = time.time()
        argmax_full = jit_oracle(put(ids_full))
        argmax_full.block_until_ready()
        oracle = [int(argmax_full[0, T + s]) for s in range(N)]
        log(f"[s1-mh] oracle compiled+ran in {time.time()-t0:.1f}s")

        if action == "repro":
            t0 = time.time()
            jp, jd = make_jits(donate=True)
            ams, _ = run_inc(jp, jd, snapshot=False)
            n_bad = sum(a != o for a, o in zip(ams, oracle))
            log(f"[s1-mh] donated decode ran in {time.time()-t0:.1f}s, bad={n_bad}/{N}")
            for s, (a, o) in enumerate(zip(ams, oracle)):
                log(f"  {'  ' if a == o else '**'} step={s:>2} pos={T+s:>2} "
                    f"decode={a:>6} oracle={o:>6} {'OK' if a == o else 'MISMATCH'}")
            verdict = "NO_S1" if n_bad == 0 else "S1_REPRODUCED"
            print(f"[p{PID}] VERDICT: {verdict} (bad={n_bad}/{N})", flush=True)
            return

        # action == diff: donated vs non-donated, localize corrupted field.
        jp_d, jd_d = make_jits(donate=True)
        jp_n, jd_n = make_jits(donate=False)
        ams_d, snaps_d = run_inc(jp_d, jd_d, snapshot=True)
        ams_n, snaps_n = run_inc(jp_n, jd_n, snapshot=True)
        bad_d = sum(a != o for a, o in zip(ams_d, oracle))
        bad_n = sum(a != o for a, o in zip(ams_n, oracle))
        log(f"[s1-mh] DONATE bad={bad_d}/{N}  NON-DONATE bad={bad_n}/{N}")

        layouts = transformer_body_layout(params, cfg, max_seq, 1)
        first = None
        for step in range(N):
            for layer in range(n_layers):
                bd, bn = snaps_d[step][layer], snaps_n[step][layer]
                if bd.shape != bn.shape or not np.array_equal(bd, bn):
                    off = 0
                    for name, fshape, _dt in layouts[layer]:
                        n = int(np.prod(fshape)) if len(fshape) else 1
                        if n == 0:
                            continue
                        d = bd[off:off + n]
                        nn = bn[off:off + n]
                        mad = float(np.max(np.abs(d - nn))) if d.shape == nn.shape else float("inf")
                        if mad > 0:
                            log(f"  DIFF step={step:>2} layer={layer} field={name:<22} "
                                f"max|donate-nondonate|={mad:.4g} "
                                f"(donate_argmax={ams_d[step]} nondonate_argmax={ams_n[step]} oracle={oracle[step]})")
                            if first is None:
                                first = (step, layer, name, mad)
                        off += n
        if first is None:
            log("[s1-mh] NO buffer diff between donate and non-donate paths "
                f"(bad_d={bad_d}, bad_n={bad_n}).")
        else:
            log(f"[s1-mh] FIRST divergence: step={first[0]} layer={first[1]} "
                f"field={first[2]} max_abs={first[3]:.4g}")
        print(f"[p{PID}] DIFF_DONE first={first} bad_d={bad_d} bad_n={bad_n}", flush=True)


if __name__ == "__main__":
    main()

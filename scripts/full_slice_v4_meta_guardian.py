#!/usr/bin/env python3
"""Metadata guardian: keep the ray GCS CLUSTER_METADATA pinned to OUR ray
version (2.55.1), self-healing within ~0.5s of any poisoning.

WHY: mark's `node` docker container (ray 2.54.1) periodically joins our GCS,
calls put_cluster_metadata(overwrite=True) which stamps CLUSTER_METADATA to
2.54.1, then crashes on the version mismatch. That poisoned value makes OUR
own 2.55.1 clients/workers fail their `ray.init()` version check. The node
container can't actually hold the TPU (it Exits 1), so the only residual harm
is this metadata flip — which this loop continuously corrects.

Run in background during TPU work; kill via
`pkill -f "[f]ull_slice_v4_meta_guardian"`.
"""
import json, time, sys
import ray
from ray._raylet import GcsClient

ADDR = sys.argv[1] if len(sys.argv) > 1 else "10.164.0.192:6379"
KEY = b"CLUSTER_METADATA"
NS = b"cluster"
TARGET_V = ray.__version__       # 2.55.1
TARGET_C = ray.__commit__        # 237c2455...

gc = GcsClient(address=ADDR)
print(f"[meta-guardian] pinning {ADDR} CLUSTER_METADATA -> ray_version={TARGET_V}", flush=True)
n_fixed = 0
while True:
    try:
        v = gc.internal_kv_get(KEY, namespace=NS)
        md = json.loads(v.decode()) if v else {}
        if md.get("ray_version") != TARGET_V:
            md["ray_version"] = TARGET_V
            md["git_commit"] = TARGET_C
            md["python_version"] = "3.12.13"
            gc.internal_kv_put(KEY, json.dumps(md).encode(), True, namespace=NS)
            n_fixed += 1
            print(f"[meta-guardian] re-stamped (#{n_fixed}) at {time.strftime('%H:%M:%S')}", flush=True)
    except Exception as e:
        print(f"[meta-guardian] err: {e}", flush=True)
        time.sleep(1)
    time.sleep(0.5)

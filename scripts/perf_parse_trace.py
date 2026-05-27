#!/usr/bin/env python3
"""Durable TPU profiler-trace parser for the V4 perf campaign.

Parses a torch/XLA Chrome-format profiler trace (gzipped JSON, ~hundreds of MB)
and ranks the on-device XLA ops by wall-time, plus a structural bucket summary
that tells you AT A GLANCE whether the gather, the fused custom-call, collectives,
matmul, etc. dominate a given trace.

Streams with ijson (NEVER json.load — the file is too big to fit in RAM).

Trace layout (verified): top-level {displayTimeUnit, metadata, traceEvents:[...]}.
  * metadata events: ph=="M", name=="process_name"  -> args.name e.g. "/device:TPU:0"
                     ph=="M", name=="thread_name"   -> args.name e.g. "XLA Ops"
  * op events:       ph=="X" with name (HLO/fusion) + dur (microseconds).
The device ops we want live on a TPU-device pid in the lane named "XLA Ops".
(Beware: the host CPU pid carries far MORE total X-dur via Python tracing spans,
and the "XLA Modules" lane duplicates "XLA Ops" totals as outer spans — so we
must target the TPU pid's "XLA Ops" lane specifically, not the biggest pid.)
"""
import argparse, gzip, sys
from collections import defaultdict
import ijson

# (bucket label, substrings matched case-insensitively against the op name).
# Order matters: first match wins, so the specific buckets precede the general.
BUCKETS = [
    ("gather/take_along", ("gather", "take_along", "dynamic-slice", "dynamicslice")),
    ("custom-call/Mosaic", ("custom-call", "custom_call", "mosaic", "tpu_custom", "sparse_attn")),
    ("all-reduce",         ("all-reduce", "allreduce", "reduce-scatter")),
    ("all-gather",         ("all-gather", "allgather")),
    ("topk/sort",          ("top_k", "top-k", "topk", "sort", "approx_max")),
    ("copy/transpose",     ("copy", "transpose", "bitcast")),
    ("matmul/dot/einsum",  ("dot", "fusion", "einsum", "convolution")),
]

def classify(name):
    low = name.lower()
    for label, subs in BUCKETS:
        if any(s in low for s in subs):
            return label
    return "other"

def main():
    ap = argparse.ArgumentParser(description="Rank on-device XLA ops in a TPU profiler trace.")
    ap.add_argument("trace", help="path to a *.trace.json.gz (Chrome trace format)")
    ap.add_argument("--top", type=int, default=30, help="number of top ops to print (default 30)")
    ap.add_argument("--bucket-ops", type=int, nargs="?", const=8, default=0, metavar="N",
                    help="ATTRIBUTE buckets: for each structural bucket print its top N "
                         "(default 8) contributing op names (name, total dur_ms, count). "
                         "Tells you WHAT is inside e.g. the gather/take_along or 'other' bucket.")
    args = ap.parse_args()

    # ---- Pass 1: metadata. Build pid->name and (pid,tid)->lane name. ----
    proc_name, lane_name = {}, {}
    with gzip.open(args.trace, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            if ev.get("ph") != "M":
                continue
            nm = ev.get("name")
            if nm == "process_name":
                proc_name[ev.get("pid")] = ev.get("args", {}).get("name", "")
            elif nm == "thread_name":
                lane_name[(ev.get("pid"), ev.get("tid"))] = ev.get("args", {}).get("name", "")

    # TPU device pids by metadata name; their "XLA Ops" lanes are the target.
    tpu_pids = {p for p, n in proc_name.items() if "tpu" in n.lower() or "device" in n.lower()}
    xla_ops_lanes = {(p, t) for (p, t), n in lane_name.items()
                     if p in tpu_pids and n.strip().lower() == "xla ops"}

    # ---- Pass 2: aggregate X-event dur+count by op name on the target lane(s). ----
    # If we identified XLA-Ops lanes, restrict to them. Otherwise fall back to a
    # per-lane scan and pick the busiest lane on a TPU pid (or, last resort, the
    # busiest lane overall) — robust to differently-named metadata.
    by_name_lane = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # lane -> name -> [dur,cnt]
    have_lanes = bool(xla_ops_lanes)
    with gzip.open(args.trace, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            if ev.get("ph") != "X":
                continue
            lane = (ev.get("pid"), ev.get("tid"))
            if have_lanes and lane not in xla_ops_lanes:
                continue
            d = float(ev.get("dur") or 0.0)
            slot = by_name_lane[lane][ev.get("name", "<noname>")]
            slot[0] += d
            slot[1] += 1

    if not have_lanes:
        # Fallback: rank lanes by total dur, prefer a TPU pid, take the single busiest.
        lane_tot = {lane: sum(v[0] for v in d.values()) for lane, d in by_name_lane.items()}
        tpu_lanes = {l: t for l, t in lane_tot.items() if l[0] in tpu_pids}
        pool = tpu_lanes or lane_tot
        chosen = max(pool, key=pool.get) if pool else None
        xla_ops_lanes = {chosen} if chosen else set()
        sys.stderr.write(f"[fallback] no 'XLA Ops' metadata; chose busiest lane {chosen} "
                         f"(pid name={proc_name.get(chosen[0]) if chosen else '?'})\n")

    # Merge the (possibly multiple, symmetric per-chip) chosen lanes by op name.
    agg = defaultdict(lambda: [0.0, 0])
    for lane in xla_ops_lanes:
        for name, (d, c) in by_name_lane.get(lane, {}).items():
            agg[name][0] += d
            agg[name][1] += c

    total = sum(v[0] for v in agg.values()) or 1.0  # us; guard div-by-zero
    lanes_desc = ", ".join(f"pid{p}/tid{t}({proc_name.get(p,'?')})" for p, t in sorted(xla_ops_lanes))
    print(f"trace: {args.trace}")
    print(f"device lanes parsed: {lanes_desc or '<none>'}")
    print(f"total on-device op time: {total/1000:.1f} ms  ({len(agg)} distinct ops)\n")

    # ---- Top-N op table. ----
    ranked = sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)
    print(f"{'#':>3}  {'dur_ms':>11}  {'pct':>6}  {'count':>8}  name")
    for i, (name, (d, c)) in enumerate(ranked[:args.top], 1):
        print(f"{i:>3}  {d/1000:>11.2f}  {100*d/total:>5.1f}%  {c:>8}  {name[:90]}")

    # ---- Structural bucket summary. ----
    buckets = defaultdict(float)
    bucket_ops = defaultdict(list)  # label -> [(name, dur_us, count), ...]
    for name, (d, c) in agg.items():
        label = classify(name)
        buckets[label] += d
        bucket_ops[label].append((name, d, c))
    print(f"\n{'STRUCTURAL BUCKET':>20}  {'total_ms':>12}  {'pct':>6}")
    for label, d in sorted(buckets.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{label:>20}  {d/1000:>12.2f}  {100*d/total:>5.1f}%")

    # ---- Per-bucket op attribution (--bucket-ops). ----
    if args.bucket_ops:
        n = args.bucket_ops
        print(f"\n=== BUCKET ATTRIBUTION (top {n} ops by dur per bucket) ===")
        for label, _d in sorted(buckets.items(), key=lambda kv: kv[1], reverse=True):
            ops = sorted(bucket_ops[label], key=lambda t: t[1], reverse=True)
            shown = sum(o[1] for o in ops[:n])
            print(f"\n[{label}]  {buckets[label]/1000:.2f} ms total, "
                  f"{len(ops)} distinct op-names  (top {min(n,len(ops))} = "
                  f"{100*shown/(buckets[label] or 1.0):.0f}% of bucket)")
            print(f"    {'dur_ms':>10}  {'pct_tot':>7}  {'count':>7}  name")
            for name, dd, cc in ops[:n]:
                print(f"    {dd/1000:>10.2f}  {100*dd/total:>6.1f}%  {cc:>7}  {name[:80]}")

if __name__ == "__main__":
    main()

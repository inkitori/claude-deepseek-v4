#!/usr/bin/env python3
"""Concurrent multi-request DECODE probe for the QUANT v6e-16 serving gate.

Usage: python3 scripts/quant_concurrent_probe.py [K] [N]
       K = concurrency (default 4, match --max-num-seqs), N = max_tokens (default 32)

Fires K IDENTICAL deterministic (temp=0, seed=0) Fibonacci-continuation requests
CONCURRENTLY at the live engine (port 18081), plus one baseline before and one
state-check after. Long-ish N forces the K requests to co-decode, so the engine
actually runs num_reqs>1 decode steps (the path _v4_decode_replicate does NOT
cover: num_reqs>1 -> ATTN_DATA-sharded activation into the dense MoE einsum's
implicit collective = the S1-buggy combination, since gmm needs N>=16 and
replication needs num_reqs==1).

PRIMARY verdict = FIB CORRECTNESS of every concurrent response. The predicted
failure mode (uninitialised-HBM S1-class corruption) yields INCOHERENT/garbage
output, NOT merely reordered-but-correct numbers, so correctness — not byte
identity — is the decisive signal. (CLAUDE.md: the free-form FIB *tail* is
benignly nondeterministic at temp=0 even single-request, so a long-tail md5
mismatch alone is NOT proof of the concurrent bug; an INCORRECT Fibonacci is.)
SECONDARY = first-decoded-token agreement + byte-identity (informational).
"""
import sys, json, hashlib, urllib.request, concurrent.futures, time

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N = int(sys.argv[2]) if len(sys.argv) > 2 else 32
BASE = "http://127.0.0.1:18081"
PROMPT = "Here is the start of the Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13,"
# Canonical correct continuation of "...8, 13,". The 4-term prefix needs few
# tokens (robust to small N); the 5-term incl. 144 is the full gate tail.
FIB4 = " 21, 34, 55, 89"
FIB5 = " 21, 34, 55, 89, 144"

_models = json.load(urllib.request.urlopen(f"{BASE}/v1/models", timeout=60))
MODEL = _models["data"][0]["id"]


def fire(_=None):
    body = json.dumps({
        "model": MODEL, "prompt": PROMPT,
        "max_tokens": N, "temperature": 0, "seed": 0,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
    except Exception as e:                       # a wedge must not crash the probe
        return {"err": f"{type(e).__name__}: {e}", "text": "", "dt": time.time() - t0}
    c = r["choices"][0]
    return {"text": c["text"], "finish": c.get("finish_reason"),
            "toks": r["usage"]["completion_tokens"], "dt": time.time() - t0}


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()[:8]


def fib_ok(s):
    return FIB4 in s          # the decisive correctness check (4-term, robust)


def show(tag, res):
    if "err" in res:
        print(f"  [{tag}] ERROR {res['err']}")
        return
    t = res["text"]
    print(f"  [{tag}] md5={md5(t)} fib4={'Y' if FIB4 in t else 'N'} "
          f"fib5={'Y' if FIB5 in t else 'N'} finish={res['finish']!r} "
          f"toks={res['toks']} dt={res['dt']:.1f}s")
    print(f"        TEXT={t!r}")


print(f"=== concurrent decode probe: K={K} concurrent, N={N} max_tokens ===")
print(f"MODEL={MODEL}")

print("\n--- (1) baseline: single request ---")
base = fire()
show("base", base)

print(f"\n--- (2) {K} CONCURRENT identical requests ---")
with concurrent.futures.ThreadPoolExecutor(max_workers=K) as ex:
    conc = list(ex.map(fire, range(K)))
for i, res in enumerate(conc):
    show(f"c{i}", res)

print("\n--- (3) state-check: single request AFTER the concurrent batch ---")
post = fire()
show("post", post)

# ---- verdicts ----
print("\n=== VERDICT ===")
ok_conc = [r for r in conc if "err" not in r]
n_err = len(conc) - len(ok_conc)
texts = [r["text"] for r in ok_conc]
md5s = {md5(t) for t in texts}
n_fib = sum(1 for t in texts if fib_ok(t))
base_ok = "err" not in base and fib_ok(base["text"])
post_ok = "err" not in post and fib_ok(post["text"])

def line(name, passed, detail=""):
    print(f"  {'PASS' if passed else 'FAIL'}  {name:24} {detail}")

line("BASELINE_FIB_CORRECT", base_ok)
line("NO_CONCURRENT_ERRORS", n_err == 0, f"({n_err}/{K} errored — wedge?)" if n_err else "")
line("ALL_CONCURRENT_CORRECT", n_err == 0 and n_fib == len(ok_conc),
     f"({n_fib}/{len(ok_conc)} FIB-correct)")
line("STATE_CLEAN_AFTER", post_ok)
line("CONC_BYTE_DETERMINISTIC", len(md5s) == 1,
     f"(unique md5s among concurrent: {sorted(md5s)}) [secondary: long tail benignly nondet]")
print("\n  DECISIVE: ALL_CONCURRENT_CORRECT is the real gate. If it FAILs with coherent")
print("  but reordered numbers, that's benign tail nondeterminism; if responses are")
print("  GARBAGE/incoherent, that's the predicted S1-class concurrent-decode corruption.")

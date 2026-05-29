#!/usr/bin/env python3
"""FIB decode probe for the QUANT v6e-16 GATE.

Usage: python3 scripts/s1_probe2.py [N]   (N = max_tokens, default 2)

Fires a DETERMINISTIC (temperature=0, seed=0) Fibonacci-continuation prompt at
the live vllm engine (port 18081), prints the decoded text + an 8-hex md5 of it.
GATE: run on 2 FRESH engines and confirm the N=2 md5 is byte-identical, and that
a longer decode (e.g. N=24) contains the correct Fibonacci tail 21, 34, 55, 89,
144. The exact md5 is the NEW v6e-16 baseline (the old v6e-32 5bf42256 is dead).
"""
import sys, json, hashlib, urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
BASE = "http://127.0.0.1:18081"
PROMPT = "Here is the start of the Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13,"

models = json.load(urllib.request.urlopen(f"{BASE}/v1/models", timeout=60))
model = models["data"][0]["id"]
body = json.dumps({
    "model": model, "prompt": PROMPT,
    "max_tokens": N, "temperature": 0, "seed": 0,
}).encode()
req = urllib.request.Request(
    f"{BASE}/v1/completions", data=body,
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=900))
text = r["choices"][0]["text"]
print(f"N={N} md5={hashlib.md5(text.encode()).hexdigest()[:8]} "
      f"finish={r['choices'][0].get('finish_reason')!r}")
print(f"PROMPT={PROMPT!r}")
print(f"TEXT={text!r}")

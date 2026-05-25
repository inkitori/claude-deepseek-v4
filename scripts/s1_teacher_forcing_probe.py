"""S1 teacher-forcing localizer — distinguishes a broken DECODE-STATE path
from a broken forward, using only curls to a running smoke (no instrumentation).

Idea: `max_tokens=1` returns the pure prefill-FORWARD next token (no decode step
runs). Chaining `max_tokens=1` calls (append the greedy token, re-prefill) builds
the CORRECT greedy trajectory entirely from the forward path — which is known good
(token1 of every generation is correct). The `max_tokens=N` request runs the same
prefill then N-1 DECODE-STATE steps. Compare:

  decode-path[i]  vs  teacher-forced[i]

Both are single-sequence (n_active==1 -> the packed decode-state path; NO multi-seq
code-path switch). If they agree at i=0 (forward) but diverge at i>=1 while the
teacher-forced trajectory stays coherent, the decode-state path (seed + step) is
the bug, and the forward is fine. Run several times: determinism check.

Usage: python scripts/s1_teacher_forcing_probe.py [PROMPT] [N] [URL]
"""
import json
import sys
import urllib.request

URL = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:18081/v1/completions"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def complete(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "seed": 0, "logprobs": 1,
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    toks = lp.get("tokens") or []
    return ch.get("text", ""), toks


def main():
    print(f"URL={URL}\nPROMPT={PROMPT!r}  N={N}\n")

    # --- determinism: 3 identical decode-path requests ---
    print("=== decode-path (max_tokens=%d), 3x determinism ===" % N)
    runs = []
    for k in range(3):
        text, toks = complete(PROMPT, N)
        runs.append((text, toks))
        print(f"  run{k}: text={text!r}")
        print(f"        toks={toks}")
    det = all(runs[i][0] == runs[0][0] for i in range(3))
    print(f"  byte-identical across 3 runs: {det}\n")

    decode_text, decode_toks = runs[0]

    # --- teacher-forced reference: chain max_tokens=1 (pure forward) ---
    print("=== teacher-forced reference (chained max_tokens=1, pure forward) ===")
    cur = PROMPT
    ref_toks = []
    for i in range(N):
        _t, toks = complete(cur, 1)
        if not toks:
            print(f"  step{i}: NO TOKEN returned; stop")
            break
        tok = toks[0]
        ref_toks.append(tok)
        cur = cur + tok
        print(f"  step{i}: +{tok!r}")
    print(f"  reference continuation text: {cur[len(PROMPT):]!r}\n")

    # --- align & report first divergence ---
    print("=== alignment (decode-path vs teacher-forced) ===")
    first_div = None
    for i in range(min(len(decode_toks), len(ref_toks))):
        same = decode_toks[i] == ref_toks[i]
        mark = "  " if same else "**"
        print(f" {mark} i={i:>2} decode={decode_toks[i]!r:>14} teacher={ref_toks[i]!r:>14}"
              f" {'OK' if same else 'DIVERGE'}")
        if not same and first_div is None:
            first_div = i
    if first_div is None:
        print("  decode-path matches teacher-forced for all compared positions.")
    else:
        print(f"\n  FIRST DIVERGENCE at i={first_div} "
              f"(i=0 is the pure-forward token; i>=1 are decode-state steps).")
        print(f"  => decode-state path diverges starting at decode step {first_div}.")


if __name__ == "__main__":
    main()

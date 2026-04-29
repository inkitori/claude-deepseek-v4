"""Stress test for DeepSeek V4 vllm serve roundtrip.

Exercises the full architecture (SWA + CSA + HCA + mHC + MoE + MTP) on a
real-shape tiny config (compress_ratios=[0,0,4,128,4,0,0]) to flush out
any wiring bug before pointing at V4-Flash. We deliberately do NOT
assume what the model outputs — the v6 commit baked in its observed
'abc' completion to one test, but here we just verify structural and
behavioral invariants that any working causal LM must satisfy.
"""

from __future__ import annotations
import concurrent.futures
import json
import sys
import time
import urllib.request

PORT = 18080
MODEL = "/mnt/scratch/tiny_v4_bf16"
URL = f"http://localhost:{PORT}/v1/completions"


def post(prompt: str, max_tokens: int = 8, seed: int = 0,
         temperature: float = 0.0, n: int = 1) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "n": n,
    }).encode("utf-8")
    req = urllib.request.Request(
        URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: {a!r} != {b!r}")


def check(label: str, fn):
    t0 = time.monotonic()
    try:
        fn()
        dt = time.monotonic() - t0
        print(f"  PASS  {label}  ({dt:.1f}s)")
    except Exception as e:
        dt = time.monotonic() - t0
        print(f"  FAIL  {label}  ({dt:.1f}s)  {type(e).__name__}: {e}")
        raise


def t1_two_seed0_byte_equal():
    j1 = post("abc", max_tokens=8)
    j2 = post("abc", max_tokens=8)
    assert_eq(j1["choices"][0]["text"], j2["choices"][0]["text"],
              "seed=0 determinism")
    assert j1["choices"][0]["text"] != "", "non-empty"
    assert_eq(j1["usage"]["completion_tokens"], 8, "tok count")


def t2_prompt_dependence():
    j1 = post("abc", max_tokens=8)
    j2 = post("hello world this is a longer prompt", max_tokens=8)
    j3 = post("the quick brown fox jumps over the lazy dog", max_tokens=8)
    texts = {j1["choices"][0]["text"], j2["choices"][0]["text"],
             j3["choices"][0]["text"]}
    assert len(texts) >= 2, f"all three prompts collapsed to same output: {texts}"


def t3_long_max_tokens():
    j = post("abc", max_tokens=32)
    assert_eq(j["usage"]["completion_tokens"], 32, "max_tokens=32")
    assert j["choices"][0]["text"] != "", "non-empty"


def t4_very_long_max_tokens():
    j = post("abc", max_tokens=64)
    assert_eq(j["usage"]["completion_tokens"], 64, "max_tokens=64")


def t5_prefix_consistency():
    """At temp=0, the first N tokens of an N+M-token completion should
    match the entire N-token completion (greedy is prefix-stable)."""
    j8 = post("abc", max_tokens=8)
    j16 = post("abc", max_tokens=16)
    text8 = j8["choices"][0]["text"]
    text16 = j16["choices"][0]["text"]
    assert text16.startswith(text8), (
        f"greedy 16-token completion does not extend the 8-token completion:\n"
        f"  text8={text8!r}\n  text16={text16!r}"
    )


def t6_concurrent_batch():
    """Send 4 different prompts concurrently — exercises continuous-
    batching path with max_num_seqs=4."""
    prompts = ["abc", "xyz", "hello world", "the quick brown fox"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda p: post(p, max_tokens=8), prompts))
    for j, p in zip(results, prompts):
        text = j["choices"][0]["text"]
        assert text != "", f"prompt {p!r}: empty text"
        assert_eq(j["usage"]["completion_tokens"], 8,
                  f"prompt {p!r}: tok count")


def t7_concurrent_same_prompt_determinism():
    """4 concurrent identical seed=0 requests → all should be byte-equal."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda _: post("abc", max_tokens=8), range(4)))
    texts = {j["choices"][0]["text"] for j in results}
    assert len(texts) == 1, (
        f"4 concurrent identical seed=0 requests gave {len(texts)} "
        f"different completions: {texts}"
    )


def t8_varied_prompt_lengths():
    """Prompt lengths from 1 char to ~100 chars, max_tokens=4. Verifies
    the prefill path handles different seqlens (each goes through SWA
    bucket selection for input length padding)."""
    prompts = [
        "a",
        "abc",
        "the quick brown",
        "hello world this is a moderately long prompt to test prefill",
        "a" * 100,
    ]
    for p in prompts:
        j = post(p, max_tokens=4)
        text = j["choices"][0]["text"]
        assert text != "", f"prompt len={len(p)}: empty text"
        assert_eq(j["usage"]["completion_tokens"], 4,
                  f"prompt len={len(p)}: tok count")


def t9_finish_reason_length():
    j = post("abc", max_tokens=8)
    assert_eq(j["choices"][0]["finish_reason"], "length",
              "no natural stop token in tiny config")


def t10_logprobs_shape():
    """logprobs=1 should return at least one logprob per token."""
    payload = json.dumps({
        "model": MODEL, "prompt": "abc", "max_tokens": 4,
        "temperature": 0, "seed": 0, "logprobs": 1,
    }).encode("utf-8")
    req = urllib.request.Request(
        URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        j = json.loads(r.read())
    lp = j["choices"][0].get("logprobs")
    assert lp is not None, "no logprobs in response"
    # Must have one logprob entry per generated token
    assert_eq(len(lp["tokens"]), 4, "logprobs/tokens count")
    assert_eq(len(lp["token_logprobs"]), 4, "logprobs/token_logprobs count")
    # All logprobs must be finite negative (≤0 since they're log-probabilities)
    for x in lp["token_logprobs"]:
        assert x is None or x <= 0.0, f"logprob > 0: {x}"


def t11_finite_outputs_no_nan():
    """Stress: send a few different prompts, ensure no NaN/Inf surfaces
    via logprobs."""
    for p in ["abc", "hello", "the cat sat on the mat", "x" * 50]:
        payload = json.dumps({
            "model": MODEL, "prompt": p, "max_tokens": 4,
            "temperature": 0, "seed": 0, "logprobs": 1,
        }).encode("utf-8")
        req = urllib.request.Request(
            URL, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            j = json.loads(r.read())
        for x in j["choices"][0]["logprobs"]["token_logprobs"]:
            if x is not None:
                assert x == x, f"NaN logprob for prompt {p!r}"
                assert x != float("-inf"), f"-inf logprob for prompt {p!r}"


def t12_long_context_decode_through_window_wrap():
    """SWA window in tiny config is 8 tokens. Generate well past that
    to exercise the sliding-window wraparound in decode."""
    j = post("the quick brown fox jumps over the lazy dog and then some",
             max_tokens=64)
    assert_eq(j["usage"]["completion_tokens"], 64, "long decode")
    text = j["choices"][0]["text"]
    assert text != "", "long decode produced empty"


def main():
    print(f"Running stress tests against {URL}")
    tests = [
        ("seed=0 byte-equal determinism (×2)", t1_two_seed0_byte_equal),
        ("prompt dependence (3 distinct prompts)", t2_prompt_dependence),
        ("max_tokens=32", t3_long_max_tokens),
        ("max_tokens=64", t4_very_long_max_tokens),
        ("greedy prefix consistency (8 ⊂ 16)", t5_prefix_consistency),
        ("4 concurrent different prompts", t6_concurrent_batch),
        ("4 concurrent identical seed=0", t7_concurrent_same_prompt_determinism),
        ("varied prompt lengths", t8_varied_prompt_lengths),
        ("finish_reason=length", t9_finish_reason_length),
        ("logprobs response shape", t10_logprobs_shape),
        ("logprobs finite (no NaN/-inf)", t11_finite_outputs_no_nan),
        ("decode through SWA wrap (max_tokens=64)",
         t12_long_context_decode_through_window_wrap),
    ]
    failures = 0
    for label, fn in tests:
        try:
            check(label, fn)
        except Exception:
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()

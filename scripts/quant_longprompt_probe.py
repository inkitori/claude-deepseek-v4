#!/usr/bin/env python3
"""Long-prompt probe for the QUANT v6e-16 larger-config hardening.

Fires a deterministic (temp=0, seed=0) LONG prompt (~500+ tokens) at the live
vllm engine (:18081) to exercise the fp8-gmm compressed-expert PREFILL path at
large N (the Q.13 change) — the short FIB/Paris probes only hit small N.

Reports prompt_tokens (actual prefill size), output text, md5, finish_reason,
and whether two identical fires are byte-identical (intra-engine determinism).
A coherent one-sentence summary + no NaN/garbage + deterministic = PASS.

Usage: python3 /tmp/quant_longprompt_probe.py [max_tokens]   (default 32)
"""
import sys, json, hashlib, urllib.request

MAXTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 32
BASE = "http://127.0.0.1:18081"

PASSAGE = (
    "The printing press, developed by Johannes Gutenberg in the middle of the "
    "fifteenth century, was one of the most consequential inventions in human "
    "history. Before its arrival, books in Europe were copied by hand, usually "
    "by monks working in monastery scriptoria, a process so slow and costly "
    "that a single volume could take months to produce and only the wealthiest "
    "institutions could afford a substantial library. Gutenberg's key "
    "innovation was movable type cast from a durable metal alloy, combined with "
    "an oil-based ink and a press adapted from the screw presses already used "
    "for pressing grapes and olives. Together these allowed identical pages to "
    "be reproduced quickly and cheaply in large numbers. The effect on European "
    "society was profound and rapid. Within fifty years of the first printed "
    "Bible, presses had spread to most major cities on the continent, and "
    "millions of books were in circulation where before there had been only "
    "thousands of manuscripts. The falling price of books widened access to "
    "knowledge far beyond the clergy and aristocracy, helping to fuel rising "
    "literacy among merchants, artisans, and eventually ordinary townspeople. "
    "Scholars could now compare standardized editions of the same text rather "
    "than divergent hand-copied versions riddled with the errors that "
    "inevitably crept in during transcription. This standardization accelerated "
    "the work of science, because experimental results, mathematical tables, "
    "and anatomical drawings could be shared accurately across great distances. "
    "The press also played a central role in the Protestant Reformation: Martin "
    "Luther's tracts and his translation of the Bible into German were printed "
    "and distributed on a scale that no earlier reform movement had ever "
    "achieved, allowing his ideas to outrun the authorities trying to suppress "
    "them. In the longer run, the cheap and rapid circulation of printed "
    "material helped knit together communities of readers who shared the same "
    "books, newspapers, and pamphlets, laying groundwork for the public sphere "
    "and the national languages of modern Europe. Historians often describe the "
    "printing revolution as a turning point comparable to the later arrival of "
    "the internet, because both technologies dramatically lowered the cost of "
    "copying and spreading information, and in doing so reshaped religion, "
    "politics, commerce, and the everyday life of the people who lived through "
    "them."
)
PROMPT = (
    "Read the following passage carefully and then summarize its single most "
    "important point in one clear sentence.\n\nPassage:\n" + PASSAGE +
    "\n\nOne-sentence summary:"
)


def fire():
    models = json.load(urllib.request.urlopen(f"{BASE}/v1/models", timeout=60))
    model = models["data"][0]["id"]
    body = json.dumps({
        "model": model, "prompt": PROMPT,
        "max_tokens": MAXTOK, "temperature": 0, "seed": 0,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    ch = r["choices"][0]
    usage = r.get("usage") or {}
    return ch["text"], ch.get("finish_reason"), usage.get("prompt_tokens")


t1, f1, pt1 = fire()
t2, f2, pt2 = fire()
print(f"prompt_tokens={pt1} (prefill size N entering MoE; chunked if > chunk)")
print(f"fire1 md5={hashlib.md5(t1.encode()).hexdigest()[:8]} finish={f1!r}")
print(f"fire2 md5={hashlib.md5(t2.encode()).hexdigest()[:8]} finish={f2!r}")
print(f"DETERMINISTIC_INTRA_ENGINE={t1 == t2}")
print(f"TEXT1={t1!r}")

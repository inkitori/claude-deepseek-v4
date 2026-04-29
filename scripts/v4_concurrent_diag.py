"""Diagnose concurrent-decode behavior under V4 (B1 territory)."""
import json, urllib.request, concurrent.futures, time

URL = "http://localhost:18080/v1/completions"
MODEL = "/mnt/scratch/tiny_v4_bf16"


def post(prompt, **kw):
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": 8,
               "temperature": 0, "seed": 0}
    payload.update(kw)
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


print("=== A) 4 sequential identical requests ===")
texts = []
for i in range(4):
    j = post("abc")
    print(f"  i={i}  finish={j['choices'][0]['finish_reason']:6}  toks={j['usage']['completion_tokens']}  text={j['choices'][0]['text']!r}")
    texts.append(j['choices'][0]['text'])
print(f"  unique texts: {len(set(texts))}")

print()
print("=== B) 2 concurrent identical requests ===")
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    results = list(ex.map(lambda _: post("abc"), range(2)))
texts = []
for j in results:
    print(f"  finish={j['choices'][0]['finish_reason']:6}  toks={j['usage']['completion_tokens']}  text={j['choices'][0]['text']!r}")
    texts.append(j['choices'][0]['text'])
print(f"  unique texts: {len(set(texts))}")

print()
print("=== C) 4 concurrent identical requests (max_num_seqs=4) ===")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(lambda _: post("abc"), range(4)))
texts = []
for j in results:
    print(f"  finish={j['choices'][0]['finish_reason']:6}  toks={j['usage']['completion_tokens']}  text={j['choices'][0]['text']!r}")
    texts.append(j['choices'][0]['text'])
print(f"  unique texts: {len(set(texts))}")

print()
print("=== D) 4 concurrent different prompts ===")
prompts = ["abc", "xyz", "hello world", "the quick brown fox"]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(lambda p: (p, post(p)), prompts))
for p, j in results:
    text = j['choices'][0]['text']
    finish = j['choices'][0]['finish_reason']
    toks = j['usage']['completion_tokens']
    print(f"  prompt={p!r:42s} finish={finish:6} toks={toks} text={text!r}")

print()
print("=== E) 1 sequential request after concurrent batch (state still clean?) ===")
j = post("abc")
print(f"  finish={j['choices'][0]['finish_reason']:6}  toks={j['usage']['completion_tokens']}  text={j['choices'][0]['text']!r}")

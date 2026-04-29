"""TP=1 test: with single rank, concurrent same-prompt requests should
all return the same completion if the model truly is per-sequence
deterministic. If they still diverge, the bug is in shared in-process
state, not cross-rank divergence."""
import os
import json, urllib.request, concurrent.futures

URL = "http://localhost:18082/v1/completions"
MODEL = os.environ.get("V4_SCRATCH_MODEL", os.path.expanduser("~/claude-deepseek-v4/work/scratch/tiny_v4_bf16"))


def post(prompt, max_tokens=8, seed=0, temperature=0.0):
    payload = {
        "model": MODEL, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


print("=== TP=1, max-num-seqs=4 ===")

print()
print("=== A) 4 sequential identical seed=0 ===")
texts = [post("abc")["choices"][0]["text"] for _ in range(4)]
print(f"  unique: {len(set(texts))}")
print(f"  text: {texts[0]!r}")

print()
print("=== B) 4 client-concurrent identical seed=0 ===")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(lambda _: post("abc"), range(4)))
texts = [j["choices"][0]["text"] for j in results]
for j in results:
    print(f"  {j['choices'][0]['text']!r}")
print(f"  unique: {len(set(texts))}")

print()
print("=== C) 2 client-concurrent identical seed=0 ===")
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    results = list(ex.map(lambda _: post("abc"), range(2)))
texts = [j["choices"][0]["text"] for j in results]
for j in results:
    print(f"  {j['choices'][0]['text']!r}")
print(f"  unique: {len(set(texts))}")

print()
print("=== D) 4 different prompts, concurrent ===")
prompts = ["abc", "xyz", "hello world", "the quick brown fox"]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(lambda p: (p, post(p)), prompts))
for p, j in results:
    text = j['choices'][0]['text']
    finish = j['choices'][0]['finish_reason']
    toks = j['usage']['completion_tokens']
    print(f"  prompt={p!r:42s} finish={finish:6} toks={toks} text={text!r}")

print()
print("=== E) Same 4 prompts, sequential ===")
for p in prompts:
    j = post(p)
    text = j['choices'][0]['text']
    finish = j['choices'][0]['finish_reason']
    toks = j['usage']['completion_tokens']
    print(f"  prompt={p!r:42s} finish={finish:6} toks={toks} text={text!r}")

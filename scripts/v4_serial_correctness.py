"""With --max-num-seqs=1, every request runs serially regardless of
client concurrency. Verify the math is fully correct in this mode."""
import json, urllib.request, concurrent.futures

URL = "http://localhost:18081/v1/completions"
MODEL = "/mnt/scratch/tiny_v4_bf16"


def post(prompt, max_tokens=8, seed=0, temperature=0.0):
    payload = {
        "model": MODEL, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


print("=== With --max-num-seqs=1, all requests are serialized ===")

print()
print("=== A) 4 sequential identical seed=0 ===")
texts = [post("abc")["choices"][0]["text"] for _ in range(4)]
for i, t in enumerate(texts):
    print(f"  i={i}  {t!r}")
print(f"  unique: {len(set(texts))}")

print()
print("=== B) 4 client-concurrent identical seed=0 (server serializes) ===")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(lambda _: post("abc"), range(4)))
texts = [j["choices"][0]["text"] for j in results]
for j in results:
    print(f"  {j['choices'][0]['text']!r}")
print(f"  unique: {len(set(texts))}")

print()
print("=== C) 4 client-concurrent different prompts (server serializes) ===")
prompts = ["abc", "xyz", "hello world", "the quick brown fox"]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(lambda p: (p, post(p)), prompts))
for p, j in results:
    text = j['choices'][0]['text']
    finish = j['choices'][0]['finish_reason']
    toks = j['usage']['completion_tokens']
    print(f"  prompt={p!r:42s} finish={finish:6} toks={toks} text={text!r}")

print()
print("=== D) Then a sequential request — same as before? ===")
j = post("abc")
print(f"  {j['choices'][0]['text']!r}")

print()
print("=== E) max_tokens=64 — long decode crosses SWA window 8× ===")
j = post("abc", max_tokens=64)
print(f"  toks={j['usage']['completion_tokens']}  text={j['choices'][0]['text']!r}")

print()
print("=== F) 16 sequential identical → all byte-equal? ===")
texts = [post("abc")["choices"][0]["text"] for _ in range(16)]
print(f"  unique: {len(set(texts))}")
print(f"  first text: {texts[0]!r}")

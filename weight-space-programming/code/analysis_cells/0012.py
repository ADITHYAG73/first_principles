# Build pass-2 request: teacher-forced read of the model's OWN pass-1 tokens (exact ids), original weights.
# 16 loop texts (prompt + answer up to ~1100 tokens past loop onset) + 12 clean correct finishes (whole answer), all <= 4096 tokens.
import random, json
rng = random.Random(2)
T = 4096
def loop_cands(src):
    out = []
    for r in cut:
        L = LOOPS[r["rid"]]
        if r["source"] != src or L["onset"] is None or L["period"] > 200: continue
        npr = len(r["prompt_token_ids"]); end = L["onset"] + 1100
        if L["onset"] >= 1200 and npr + end <= T: out.append(r)
    return out
def pick(cands, k):
    rng.shuffle(cands); seen, out = set(), []
    for r in cands:
        if r["id"] not in seen: seen.add(r["id"]); out.append(r)
        if len(out) == k: break
    return out
lc = {s: loop_cands(s) for s in ("gsm8k", "math")}
print("loop candidates", {s: len(v) for s, v in lc.items()})
loopsel = pick(lc["gsm8k"], 8) + pick(lc["math"], 8)
natc = {s: [r for r in nat if r["correct"] and r["source"] == s and len(r["prompt_token_ids"]) + r["n_tokens"] <= T] for s in ("gsm8k", "math")}
print("natural candidates", {s: len(v) for s, v in natc.items()})
natsel = pick(natc["gsm8k"], 6) + pick(natc["math"], 6)
texts, meta = [], []
for r in loopsel:
    L = LOOPS[r["rid"]]; ids = r["prompt_token_ids"] + r["token_ids"][:L["onset"] + 1100]
    texts.append(ids); meta.append(dict(kind="loop", rid=r["rid"], n_prompt=len(r["prompt_token_ids"]), onset=L["onset"], period=L["period"], len=len(ids)))
for r in natsel:
    ids = r["prompt_token_ids"] + r["token_ids"]
    texts.append(ids); meta.append(dict(kind="natural", rid=r["rid"], n_prompt=len(r["prompt_token_ids"]), len=len(ids)))
req2 = {"type": "teacher_forced", "model": "original", "token_ids": texts,
        "return": {"token_logprobs": True, "topk": 20, "hidden_states": list(range(25))}}
lens = [len(t) for t in texts]
est = len(texts) * max(lens) * (1024 * 25 * 2 + 4 + 8 * 20)
print("texts", len(texts), "kinds", collections.Counter(m["kind"] for m in meta), "max len", max(lens), "total", sum(lens), f"est {est/1e9:.2f} GB (limit {6*1024**3/1e9:.2f})")
json.dump(req2, open("work/req_pass2.json", "w")); json.dump(meta, open("work/req_pass2_meta.json", "w"), indent=1)
P2META = meta

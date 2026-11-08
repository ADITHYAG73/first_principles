# Pass-6 request (NOT launched yet): TF on the E1 model over E1's own remaining loops (pass 4) + E1 clean correct finishes.
rng6 = random.Random(6); T6 = 12400
lc6 = [r for r in cutE if LE[r["rid"]] and LE[r["rid"]][1] <= 200 and LE[r["rid"]][0] >= 1200
       and len(G4["edited"][r["rid"]]["token_ids"]) and 106 + LE[r["rid"]][0] + 1100 <= T6]
rng6.shuffle(lc6); seen, loop6 = set(), []
for r in lc6:
    if r["id"] not in seen and len(loop6) < 20: seen.add(r["id"]); loop6.append(r)
TOK4E = {json.loads(l)["rid"]: json.loads(l) for l in open("work/gen_4_edited.tokens.jsonl")}
nat6c = [r for r in G4["edited"].values() if r["correct"] and len(TOK4E[r["rid"]]["prompt_token_ids"]) + r["n_tokens"] <= T6 and r["id"] not in seen]
rng6.shuffle(nat6c); nat6 = []
for r in nat6c:
    if r["id"] not in seen and len(nat6) < 8: seen.add(r["id"]); nat6.append(r)
texts6, meta6 = [], []
for r in loop6:
    o, P, _ = LE[r["rid"]]; pr = TOK4E[r["rid"]]["prompt_token_ids"]; ids = pr + r["token_ids"][:o + 1100]
    texts6.append(ids); meta6.append(dict(kind="loop", rid=r["rid"], n_prompt=len(pr), onset=o, period=P, len=len(ids)))
for r in nat6:
    pr = TOK4E[r["rid"]]["prompt_token_ids"]; ids = pr + r["token_ids"]
    texts6.append(ids); meta6.append(dict(kind="natural", rid=r["rid"], n_prompt=len(pr), len=len(ids)))
lens = [len(t) for t in texts6]
req6 = {"type": "teacher_forced", "model": "edited", "token_ids": texts6,
        "return": {"token_logprobs": True, "topk": 20, "hidden_states": [11, 15, 19, 23, 24]}}
print("loop candidates", len(lc6), "| texts", len(texts6), collections.Counter(m["kind"] for m in meta6),
      "| max", max(lens), "total", sum(lens), f"| est {len(texts6)*max(lens)*(1024*5*2+4+160)/1e9:.2f} GB")
json.dump(req6, open("work/req_pass6.json", "w")); json.dump(meta6, open("work/req_pass6_meta.json", "w"), indent=1)
P6META = meta6

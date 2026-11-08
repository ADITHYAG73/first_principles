# Pass 4 (original vs E1, same problems + seeds): sanity of the gain.
def loadgen(m):
    R = [json.loads(l) for l in open(f"work/gen_4_{m}.jsonl")]
    T = {json.loads(l)["rid"]: json.loads(l)["token_ids"] for l in open(f"work/gen_4_{m}.tokens.jsonl")}
    for r in R: r["token_ids"] = T[r["rid"]]; r["level"] = PRACTICE[r["id"]].get("level")
    return {r["rid"]: r for r in R}
G4 = {m: loadgen(m) for m in ("original", "edited")}
for m, R in G4.items():
    rs = list(R.values()); fin = [r for r in rs if r["category"] != "cut_off"]
    print(m, dict(collections.Counter(r["category"] for r in rs)),
          "| finished:", len(fin), "acc among finished", round(np.mean([bool(r["correct"]) for r in fin]), 3),
          "| finished length q", np.percentile([r["n_tokens"] for r in fin], [10, 25, 50, 75, 90]).astype(int).tolist(),
          "| think closed among finished", sum(r["think_closed"] for r in fin))
    for lv in ("1", "2", "3", "4"):
        x = [r for r in rs if r["source"] == "math" and str(r["level"]) == lv]
        print("   MATH L", lv, sum(bool(r["correct"]) for r in x), "/", len(x))
cutE = [r for r in G4["edited"].values() if r["category"] == "cut_off"]
per = [token_loop(r["token_ids"]) for r in cutE]
print("edited cut-offs with exact periodic tail:", sum(p is not None for p in per), "/", len(cutE),
      "| onset q", np.percentile([p[0] for p in per if p], [25, 50, 75]).astype(int).tolist(),
      "| period q", np.percentile([p[1] for p in per if p], [25, 50, 75]).astype(int).tolist())
# right->wrong cases
rw = [k for k in G4["original"] if G4["original"][k]["correct"] and not G4["edited"][k]["correct"]]
print("right->wrong:", [(k, G4["edited"][k]["category"], G4["edited"][k]["n_tokens"]) for k in rw])
# thinking length before </think> in edited correct vs original correct
END = 248069
def think_len(r): return r["token_ids"].index(END) if END in r["token_ids"] else None
for m in G4:
    tl = [think_len(r) for r in G4[m].values() if r["correct"]]
    print(m, "thinking tokens before </think> among correct, quartiles", np.percentile([t for t in tl if t], [10, 25, 50, 75, 90]).astype(int).tolist())
# 'Wait' count per 1k tokens in the thinking of correct answers: is the model still checking?
wid = tok.encode("Wait").ids
for m in G4:
    v = [sum(1 for t in r["token_ids"][:think_len(r)] if t in wid) / think_len(r) * 1000 for r in G4[m].values() if r["correct"] and think_len(r)]
    print(m, "'Wait' tokens per 1k thinking tokens (correct answers): median", round(float(np.median(v)), 2))

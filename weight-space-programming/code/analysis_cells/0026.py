# Pass 5 (confirmation of E1, fresh problems): same checks as pass 4 (cell 22), plus paired SE recomputed locally.
def loadgen5(m):
    R = [json.loads(l) for l in open(f"work/gen_5_{m}.jsonl")]
    T = {json.loads(l)["rid"]: json.loads(l)["token_ids"] for l in open(f"work/gen_5_{m}.tokens.jsonl")}
    for r in R: r["token_ids"] = T[r["rid"]]; r["level"] = PRACTICE[r["id"]].get("level")
    return {r["rid"]: r for r in R}
G5 = {m: loadgen5(m) for m in ("original", "edited")}
assert set(G5["original"]) == set(G5["edited"]) and len(G5["original"]) == 256
used4 = {r["id"] for r in G4["original"].values()}; used5 = {r["id"] for r in G5["original"].values()}
print("problem overlap pass4 vs pass5:", len(used4 & used5))
for m, R in G5.items():
    rs = list(R.values()); fin = [r for r in rs if r["category"] != "cut_off"]
    print(m, dict(collections.Counter(r["category"] for r in rs)),
          "| finished:", len(fin), "acc among finished", round(np.mean([bool(r["correct"]) for r in fin]), 3),
          "| finished length q", np.percentile([r["n_tokens"] for r in fin], [10, 25, 50, 75, 90]).astype(int).tolist(),
          "| think closed among finished", sum(r["think_closed"] for r in fin))
    for lv in ("1", "2", "3", "4"):
        x = [r for r in rs if r["source"] == "math" and str(r["level"]) == lv]
        print("   MATH L", lv, sum(bool(r["correct"]) for r in x), "/", len(x))
# paired by problem (2 samples each): per-problem mean diff, SE over 128 problems
byp = collections.defaultdict(list)
for k in G5["original"]:
    byp[G5["original"][k]["id"]].append(int(bool(G5["edited"][k]["correct"])) - int(bool(G5["original"][k]["correct"])))
d = np.array([np.mean(v) for v in byp.values()])
print("paired per-problem: n", len(d), "mean", round(d.mean(), 4), "SE", round(d.std(ddof=1) / np.sqrt(len(d)), 4))
cut5 = [r for r in G5["edited"].values() if r["category"] == "cut_off"]
per5 = [token_loop(r["token_ids"]) for r in cut5]
print("edited cut-offs exact loop:", sum(p is not None for p in per5), "/", len(cut5),
      "| onset q", np.percentile([p[0] for p in per5 if p], [25, 50, 75]).astype(int).tolist())
rw = [k for k in G5["original"] if G5["original"][k]["correct"] and not G5["edited"][k]["correct"]]
print("right->wrong:", [(k, G5["edited"][k]["category"], G5["edited"][k]["n_tokens"]) for k in rw])
wa = [r for r in G5["edited"].values() if r["category"] == "wrong_answer"]
print("edited wrong answers:", len(wa), "by source", dict(collections.Counter(r["source"] for r in wa)),
      "| lengths", sorted(r["n_tokens"] for r in wa))
for m in G5:
    tl = [think_len(r) for r in G5[m].values() if r["correct"]]
    print(m, "thinking tokens before </think> among correct q", np.percentile([t for t in tl if t], [10, 25, 50, 75, 90]).astype(int).tolist())

# Passes 9 (E3) vs 10 (E1): same 256 fresh problems, same seeds. Pre-registered rule: E3 if paired E3-E1 > 0.
def loadg(path):
    R = [json.loads(l) for l in open(path)]
    T = {json.loads(l)["rid"]: json.loads(l)["token_ids"] for l in open(path.replace(".jsonl", ".tokens.jsonl"))}
    for r in R: r["token_ids"] = T[r["rid"]]; r["level"] = PRACTICE[r["id"]].get("level")
    return {r["rid"]: r for r in R}
F3, F1 = loadg("work/gen_9.jsonl"), loadg("work/gen_10.jsonl")
assert set(F3) == set(F1) and len(F3) == 512 and all(F3[k]["seed"] == F1[k]["seed"] for k in F3)
assert not ({r["id"] for r in F3.values()} & ({r["id"] for r in E1.values()}))
for name, M in (("E1", F1), ("E3", F3)):
    rs = list(M.values()); fin = [r for r in rs if r["category"] != "cut_off"]
    print(f"{name} acc {np.mean([bool(r['correct']) for r in rs]):.4f}", dict(collections.Counter(r["category"] for r in rs)),
          "| acc among finished", round(np.mean([bool(r["correct"]) for r in fin]), 3),
          "| GSM8K", round(np.mean([bool(r["correct"]) for r in rs if r["source"] == "gsm8k"]), 3),
          "MATH", round(np.mean([bool(r["correct"]) for r in rs if r["source"] == "math"]), 3))
    print("   MATH by level of 64:", {lv: sum(bool(r["correct"]) for r in rs if r["source"] == "math" and str(r["level"]) == lv) for lv in "1234"})
d = paired(F1, F3); print("FRESH paired E3 - E1 (mean, SE, wrong->right, right->wrong):", d)
for src in ("gsm8k", "math"):
    print(src, paired({k: v for k, v in F1.items() if v["source"] == src}, {k: F3[k] for k, v in F1.items() if v["source"] == src}))
print("DECISION (pre-registered: E3 if > 0):", "E3" if d[0] > 0 else "E1")

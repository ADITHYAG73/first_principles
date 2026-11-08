# Pass 8: E3 (E1 + L23H5,H4 x0.5) on the same 512 seeds as E1/E2/original.
R8 = [json.loads(l) for l in open("work/gen_8.jsonl")]
T8 = {json.loads(l)["rid"]: json.loads(l)["token_ids"] for l in open("work/gen_8.tokens.jsonl")}
for r in R8: r["token_ids"] = T8[r["rid"]]; r["level"] = PRACTICE[r["id"]].get("level")
E3 = {r["rid"]: r for r in R8}
assert set(E3) == set(E1) and all(E3[k]["seed"] == E1[k]["seed"] for k in E3)
for name, M in (("original", OR), ("E1", E1), ("E2", E2), ("E3", E3)):
    rs = list(M.values()); fin = [r for r in rs if r["category"] != "cut_off"]
    print(f"{name:8s} acc {np.mean([bool(r['correct']) for r in rs]):.4f}", dict(collections.Counter(r["category"] for r in rs)),
          "| acc among finished", round(np.mean([bool(r["correct"]) for r in fin]), 3),
          "| GSM8K", round(np.mean([bool(r["correct"]) for r in rs if r["source"] == "gsm8k"]), 3),
          "MATH", round(np.mean([bool(r["correct"]) for r in rs if r["source"] == "math"]), 3))
print("E3 - E1:", paired(E1, E3), "| E3 - E2:", paired(E2, E3), "| E3 - orig:", paired(OR, E3))
for part, keys in (("pass4 problems", G4["edited"]), ("pass5 problems", G5["edited"])):
    print(part, "E3-E1", paired({k: E1[k] for k in keys}, {k: E3[k] for k in keys}))
for lv in ("1", "2", "3", "4"):
    print("MATH L", lv, {n: sum(bool(r["correct"]) for r in M.values() if r["source"] == "math" and str(r["level"]) == lv) for n, M in (("orig", OR), ("E1", E1), ("E3", E3))}, "of 64")
per8 = [token_loop(r["token_ids"]) for r in E3.values() if r["category"] == "cut_off"]
per1 = [token_loop(r["token_ids"]) for r in E1.values() if r["category"] == "cut_off"]
print("exact-loop cut-offs: E1", sum(p is not None for p in per1), "E3", sum(p is not None for p in per8))
na8 = [r for r in E3.values() if r["category"] == "no_answer"]
print("E3 no_answer:", [(r["rid"], r["think_closed"], r["n_tokens"]) for r in na8])
for m, M in (("E1", E1), ("E3", E3)):
    tl = [think_len(r) for r in M.values() if r["correct"]]
    print(m, "thinking tokens before </think> among correct q", np.percentile([t for t in tl if t], [10, 25, 50, 75, 90]).astype(int).tolist())

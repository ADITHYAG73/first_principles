# Pass 7: E2 (E1 + L23H5,L23H4 off) on pass-4+5 problems, same seeds. Paired vs E1 (gen_4/5 edited) and original.
R7 = [json.loads(l) for l in open("work/gen_7.jsonl")]
T7 = {json.loads(l)["rid"]: json.loads(l)["token_ids"] for l in open("work/gen_7.tokens.jsonl")}
for r in R7: r["token_ids"] = T7[r["rid"]]; r["level"] = PRACTICE[r["id"]].get("level")
E2 = {r["rid"]: r for r in R7}
E1 = {**G4["edited"], **G5["edited"]}; OR = {**G4["original"], **G5["original"]}
assert set(E2) == set(E1) == set(OR) and len(E2) == 512
print("seed match E2 vs E1:", sum(E2[k]["seed"] == E1[k]["seed"] for k in E2), "/ 512")
for name, M in (("original", OR), ("E1", E1), ("E2", E2)):
    rs = list(M.values()); fin = [r for r in rs if r["category"] != "cut_off"]
    print(f"{name:8s} acc {np.mean([bool(r['correct']) for r in rs]):.4f}", dict(collections.Counter(r["category"] for r in rs)),
          "| acc among finished", round(np.mean([bool(r["correct"]) for r in fin]), 3),
          "| GSM8K", round(np.mean([bool(r["correct"]) for r in rs if r["source"] == "gsm8k"]), 3),
          "MATH", round(np.mean([bool(r["correct"]) for r in rs if r["source"] == "math"]), 3))
def paired(A, B):
    byp = collections.defaultdict(list)
    for k in A: byp[A[k]["id"]].append(int(bool(B[k]["correct"])) - int(bool(A[k]["correct"])))
    d = np.array([np.mean(v) for v in byp.values()])
    wr = sum(1 for k in A if not A[k]["correct"] and B[k]["correct"]); rw = sum(1 for k in A if A[k]["correct"] and not B[k]["correct"])
    return round(d.mean(), 4), round(d.std(ddof=1) / np.sqrt(len(d)), 4), wr, rw
print("E2 - E1  (mean, SE, wrong->right, right->wrong):", paired(E1, E2))
print("E2 - orig:", paired(OR, E2), "| E1 - orig:", paired(OR, E1))
for part, keys in (("pass4 problems", G4["edited"]), ("pass5 problems", G5["edited"])):
    print(part, "E2-E1", paired({k: E1[k] for k in keys}, {k: E2[k] for k in keys}))
# what happened to E1's cut-offs and E1's correct answers under E2
tr = collections.Counter((E1[k]["category"], E2[k]["category"]) for k in E2)
print("transition E1 category -> E2 category:", sorted(tr.items(), key=lambda x: -x[1]))
na = [r for r in E2.values() if r["category"] == "no_answer"]
print("E2 no_answer: think_closed", sum(r["think_closed"] for r in na), "/", len(na), "| finish", collections.Counter(r["finish_reason"] for r in na), "| lengths", sorted(r["n_tokens"] for r in na))
per7 = [token_loop(r["token_ids"]) for r in E2.values() if r["category"] == "cut_off"]
print("E2 cut-offs with exact loop:", sum(p is not None for p in per7), "/", len(per7))
wa7 = [r for r in E2.values() if r["category"] == "wrong_answer"]
print("E2 wrong answers by source", collections.Counter(r["source"] for r in wa7), "| length q", np.percentile([r["n_tokens"] for r in wa7], [10, 25, 50, 75, 90]).astype(int).tolist())

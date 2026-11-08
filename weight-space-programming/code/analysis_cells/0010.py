# Full pass 1 (512 samples): taxonomy by source/level, token-level loop stats for every cut-off.
import json, collections, numpy as np
P1 = [json.loads(l) for l in open("work/gen_1.jsonl")]
TOK = {json.loads(l)["rid"]: json.loads(l) for l in open("work/gen_1.tokens.jsonl")}
for r in P1:
    r["token_ids"] = TOK[r["rid"]]["token_ids"]; r["prompt_token_ids"] = TOK[r["rid"]]["prompt_token_ids"]
    r["level"] = PRACTICE[r["id"]].get("level")
print("n", len(P1), dict(collections.Counter(r["category"] for r in P1)))
for src in ("gsm8k", "math"):
    rs = [r for r in P1 if r["source"] == src]
    print(src, len(rs), dict(collections.Counter(r["category"] for r in rs)), "acc", round(np.mean([bool(r["correct"]) for r in rs]), 4))
print("MATH by level: (n, correct, cut)")
for lv in sorted(set(r["level"] for r in P1 if r["source"] == "math"), key=str):
    rs = [r for r in P1 if r["source"] == "math" and r["level"] == lv]
    print(" ", lv, len(rs), sum(bool(r["correct"]) for r in rs), sum(r["category"] == "cut_off" for r in rs))
nat = [r for r in P1 if r["category"] != "cut_off"]
print("natural finishes", len(nat), "correct", sum(bool(r["correct"]) for r in nat), "token quartiles",
      np.percentile([r["n_tokens"] for r in nat], [0, 25, 50, 75, 100]).astype(int).tolist())
cut = [r for r in P1 if r["category"] == "cut_off"]
loops = {}
for r in cut:
    b = token_loop(r["token_ids"])
    loops[r["rid"]] = dict(onset=b[0], period=b[1], periodic_tail=b[2]) if b else dict(onset=None, period=None, periodic_tail=0)
LOOPS = loops
found = [v for v in loops.values() if v["onset"] is not None]
for src in ("gsm8k", "math"):
    c = [r for r in cut if r["source"] == src]
    f = [loops[r["rid"]] for r in c if loops[r["rid"]]["onset"] is not None]
    print(src, "cut", len(c), "exact periodic tail", len(f),
          "onset q", np.percentile([x["onset"] for x in f], [0, 25, 50, 75, 100]).astype(int).tolist() if f else None,
          "period q", np.percentile([x["period"] for x in f], [0, 25, 50, 75, 100]).astype(int).tolist() if f else None)
nonper = [r for r in cut if loops[r["rid"]]["onset"] is None]
print("cut-offs without exact periodic tail:", len(nonper), [(r["rid"], r["dup_line_share"]) for r in nonper])
# per-problem: both samples cut?
byp = collections.defaultdict(list)
for r in P1: byp[r["id"]].append(r)
print("problems with 0/1/2 samples cut:", collections.Counter(sum(x["category"] == "cut_off" for x in v) for v in byp.values()))

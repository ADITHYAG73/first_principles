# Pass 5 = confirmation of E1 on FRESH problems (not in pass 1 or pass 4). Same composition as pass 4.
req4 = json.load(open("work/req_pass4.json"))
used_all = set(json.load(open("work/used_problems.json"))) | set(req4["problems"])
json.dump(sorted(used_all), open("work/used_problems.json", "w"))
rng5 = random.Random(5)
g = sorted(p for p in PRACTICE if p.startswith("gsm8k-train") and p not in used_all); rng5.shuffle(g)
sel5 = g[:64]
for lv in (1, 2, 3, 4):
    c = sorted(p for p in PRACTICE if p.startswith("math-train") and p not in used_all and str(PRACTICE[p].get("level")) == str(lv)); rng5.shuffle(c)
    sel5 += c[:16]
assert not (set(sel5) & used_all) and len(set(sel5)) == 128
req5 = {"type": "generate", "model": "both", "problems": sel5, "n": 2, "max_tokens": 32768}
json.dump(req5, open("work/req_pass5.json", "w"))
print("used so far", len(used_all), "| pass-5 problems", len(sel5), "| edit", edited())

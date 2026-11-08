# Pass-4 problems: fresh (not used in pass 1), seeded. 64 GSM8K train + 64 MATH train (levels 1-4, 16 each; level 5 was 0/50
# in pass 1 so it adds noise but no measurable signal). n=2 -> 256 per model ("both").
used = set(json.load(open("work/used_problems.json")))
rng4 = random.Random(4)
g = sorted(p for p in PRACTICE if p.startswith("gsm8k-train") and p not in used); rng4.shuffle(g)
sel = g[:64]
for lv in (1, 2, 3, 4):
    c = sorted(p for p in PRACTICE if p.startswith("math-train") and p not in used and str(PRACTICE[p].get("level")) == str(lv)); rng4.shuffle(c)
    sel += c[:16]
print(len(sel), len(set(sel)), collections.Counter(PRACTICE[p].get("level") for p in sel if p.startswith("math")))
req4 = {"type": "generate", "model": "both", "problems": sel, "n": 2, "max_tokens": 32768}
json.dump(req4, open("work/req_pass4.json", "w"))
print(edited())

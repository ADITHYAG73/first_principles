# Pass-9 problems: fresh = not in pass 1, 4, 5 (or 7/8, which reused 4+5). 128 GSM8K train + 128 MATH train L1-4 (32 each). Seeded.
used = set(json.load(open("work/used_problems.json"))) | {r["id"] for r in G4["original"].values()} | {r["id"] for r in G5["original"].values()}
print("excluded", len(used))
rng9 = random.Random(9)
g = sorted(p for p in PRACTICE if p.startswith("gsm8k-train") and p not in used); rng9.shuffle(g); sel9 = g[:128]
for lv in (1, 2, 3, 4):
    c = sorted(p for p in PRACTICE if p.startswith("math-train") and p not in used and str(PRACTICE[p].get("level")) == str(lv)); rng9.shuffle(c)
    sel9 += c[:32]
assert len(set(sel9)) == 256 and not (set(sel9) & used)
print(len(sel9), collections.Counter(PRACTICE[p].get("level") for p in sel9 if p.startswith("math")))
json.dump(sorted(used | set(sel9)), open("work/used_problems.json", "w"))
req9 = {"type": "generate", "model": "edited", "problems": sel9, "n": 2, "max_tokens": 32768}
json.dump(req9, open("work/req_pass9.json", "w"))
print(edited())   # must still be E3: L15, L19 (E1) + L23 x0.5
w23 = load("model.language_model.layers.23.self_attn.o_proj.weight"); o23 = load_original("model.language_model.layers.23.self_attn.o_proj.weight")
print("L23 cols 1024-1535 ratio edited/orig:", float((w23[:, 1024:1536].float() / o23[:, 1024:1536].float()).nanmedian()), "| rest identical:", bool((w23[:, :1024] == o23[:, :1024]).all() and (w23[:, 1536:] == o23[:, 1536:]).all()))

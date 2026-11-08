# Edit E2 = E1 + zero o_proj columns of L23H5 (1280-1535) and L23H4 (1024-1279). Chosen from pass 6 (cells 29-30):
# loop-specific first-order harm 5.0x (H5) and 4.5x (H4) natural; strongest remaining copiers (copy-all 0.49, 0.29).
name = "model.language_model.layers.23.self_attn.o_proj.weight"
W = load_original(name).clone()
for h in (5, 4):
    W[:, h * 256:(h + 1) * 256] = 0.0
save_edit(name, W)
print(edited())
probs7 = sorted({r["id"] for r in G4["original"].values()} | {r["id"] for r in G5["original"].values()})
print("pass-7 problems", len(probs7), collections.Counter(p.split("-")[0] for p in probs7))
req7 = {"type": "generate", "model": "edited", "problems": probs7, "n": 2, "max_tokens": 32768}
json.dump(req7, open("work/req_pass7.json", "w"))

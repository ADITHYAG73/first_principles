# Pass 10 needs exactly E1: drop the L23 edit (E3 bundle archived in work/edits_archive/E3), verify L15/L19 are E1.
drop_edit("model.language_model.layers.23.self_attn.o_proj.weight")
print(edited())
for L, hs in ((15, [6, 7]), (19, [0])):
    n_ = f"model.language_model.layers.{L}.self_attn.o_proj.weight"; w = load(n_); o = load_original(n_)
    mask = torch.zeros(w.shape[1], dtype=torch.bool)
    for h in hs: mask[h*256:(h+1)*256] = True
    print(L, "zeroed cols all zero:", bool((w[:, mask] == 0).all()), "| other cols identical:", bool((w[:, ~mask] == o[:, ~mask]).all()))
n23 = "model.language_model.layers.23.self_attn.o_proj.weight"; print("L23 equals original:", bool((load(n23) == load_original(n23)).all()))
req10 = dict(req9); json.dump(req10, open("work/req_pass10.json", "w")); print("req10 problems", len(req10["problems"]), req10["model"])

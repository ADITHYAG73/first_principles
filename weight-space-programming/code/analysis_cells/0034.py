# E3 = E1 + L23H5, L23H4 o_proj columns scaled by 0.5 (half strength). Reason: E2 (full zero) broke loops (exact-loop cut-offs
# 218 -> 72 of 512) but lost \boxed formatting (23 no_answer, 19 of them numerically right) and tripled wrong answers (24 -> 67);
# paired E2 - E1 = -1.4 pts (SE 2.4). Only 2 of 2048 columns-blocks of one tensor are scaled: not a whole-tensor scale.
name = "model.language_model.layers.23.self_attn.o_proj.weight"
W = load_original(name).clone()
for h in (5, 4):
    W[:, h * 256:(h + 1) * 256] *= 0.5
save_edit(name, W)
chk = load_original(name)
print(edited(), "| max |W-orig| outside heads 4,5:", float((W[:, :1024] - chk[:, :1024]).abs().max()), float((W[:, 1536:] - chk[:, 1536:]).abs().max()))
req8 = dict(req7); json.dump(req8, open("work/req_pass8.json", "w")); print("req8 problems", len(req8["problems"]), "n", req8["n"])

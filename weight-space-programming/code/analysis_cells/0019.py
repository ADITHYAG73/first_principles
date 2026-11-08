# Edit E1: ablate copy heads L15H7, L15H6, L19H0 (zero their o_proj input columns; head h = columns h*256:(h+1)*256).
ABL = {15: [7, 6], 19: [0]}
for L, hs in ABL.items():
    name = f"model.language_model.layers.{L}.self_attn.o_proj.weight"
    W = load_original(name).clone()
    for h in hs:
        W[:, h * 256:(h + 1) * 256] = 0.0
    save_edit(name, W)
print(edited())
req3 = dict(req2); req3["model"] = "edited"
req3["return"] = {"token_logprobs": True, "topk": 20, "hidden_states": [15, 19, 23, 24]}
json.dump(req3, open("work/req_pass3.json", "w"))
print("req3 texts", len(req3["token_ids"]), "est GB", 28 * 4094 * (1024 * 4 * 2 + 4 + 160) / 1e9)

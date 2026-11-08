# HARNESS SMOKE TEST ONLY (dummy edit, dropped right after): perturb 4 rows of a layer-5 down_proj
w = load_original("model.language_model.layers.5.mlp.down_proj.weight").clone()
w[:4] *= 1.5
print(save_edit("model.language_model.layers.5.mlp.down_proj.weight", w))

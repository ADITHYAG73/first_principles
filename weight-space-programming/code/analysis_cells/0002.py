# harness smoke test over: remove the dummy edit
drop_edit("model.language_model.layers.5.mlp.down_proj.weight")
print(edited())

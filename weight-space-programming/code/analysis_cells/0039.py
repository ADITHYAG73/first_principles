# Restore E3 (decision per pre-registered rule) from work/edits_archive/E3, verify byte-identical to the bundle used in passes 8-9.
from safetensors.torch import load_file
import hashlib, glob, os
for f in sorted(glob.glob("work/edits_archive/E3/*.safetensors")):
    d = load_file(f); (k, W), = d.items(); save_edit(k, W)
print(edited())
for f in sorted(glob.glob("work/edits_archive/E3/*.safetensors")):
    g = "work/edited/" + os.path.basename(f)
    print(os.path.basename(f)[:60], hashlib.sha256(open(f, "rb").read()).hexdigest() == hashlib.sha256(open(g, "rb").read()).hexdigest())
n23 = "model.language_model.layers.23.self_attn.o_proj.weight"; w = load(n23); o = load_original(n23)
print("L23 heads4-5 ratio", float((w[:, 1024:1536].float() / o[:, 1024:1536].float()).nanmedian()), "| rest identical", bool((w[:, :1024] == o[:, :1024]).all() and (w[:, 1536:] == o[:, 1536:]).all()))

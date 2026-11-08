import importlib, sys, torch
sys.path.insert(0, "wsp"); import recon; importlib.reload(recon)
ATT = {L: recon.AttnLayer(load_original, L) for L in (3, 7, 11, 15, 19, 23)}
WU = load_original("model.language_model.embed_tokens.weight")      # tied unembedding (read-only analysis)
GF = load_original("model.language_model.norm.weight")
print({L: tuple(a.q.shape) for L, a in ATT.items()}, tuple(WU.shape), float(GF.abs().mean()))

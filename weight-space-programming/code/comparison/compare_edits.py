"""Post-hoc comparison of my edit (E3) with the authors' released edits (p10c, p25c, p50c) on Qwen3.5-0.8B.
Weight analysis + recorded teacher-forced hidden states only; no model is run here.

usage: python -I compare_edits.py <base.safetensors> <hf_ckpt.safetensors> <logs_dir> <work_dir> <out.json>
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

BASE, CKPT, LOGS, WORK, OUT = map(Path, sys.argv[1:6])
torch.set_grad_enabled(False)
R = {}
P = "model.language_model.layers.{}."


def sha_bf16(t):
    return hashlib.sha256(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- 1. checkpoint vs base, every tensor
with safe_open(str(BASE), "pt") as fb, safe_open(str(CKPT), "pt") as fc:
    kb, kc = set(fb.keys()), set(fc.keys())
    R["keys_only_in_base"], R["keys_only_in_ckpt"] = sorted(kb - kc)[:20], sorted(kc - kb)[:20]
    n_all = n_lm = 0
    changed = {}
    for k in sorted(kb & kc):
        a, b = fb.get_tensor(k), fc.get_tensor(k)
        n_all += a.numel()
        if k.startswith("model.language_model.") or k.startswith("lm_head"):
            n_lm += a.numel()
        if a.dtype != b.dtype or a.shape != b.shape:
            changed[k] = {"dtype_or_shape_differs": [str(a.dtype), str(b.dtype), list(a.shape), list(b.shape)]}
            continue
        same = (a.view(torch.int16) == b.view(torch.int16)) if a.dtype == torch.bfloat16 else (a == b)
        if not bool(same.all()):
            m = ~same
            af, bf = a.float(), b.float()
            changed[k] = {"shape": list(a.shape), "changed_entries": int(m.sum()),
                          "changed_cols": int(m.any(0).sum()) if a.dim() == 2 else None,
                          "rel_change": float((bf - af).norm() / af.norm()),
                          "base_sha256": sha_bf16(a), "ckpt_sha256": sha_bf16(b)}
    R["n_params_all_tensors"], R["n_params_language_model"] = n_all, n_lm
    R["ckpt_changed_tensors"] = changed

man50 = json.load(open(LOGS / "exp8_math/runs/p50c/edit/manifest.json"))
R["p50c_manifest_check"] = [{"name": t["name"], "base_sha_match": changed.get(t["name"], {}).get("base_sha256") == t["base_sha256"],
                              "edited_sha_match": changed.get(t["name"], {}).get("ckpt_sha256") == t["edited_sha256"],
                              "manifest_changed_entries": t["changed_entries"]} for t in man50["tensors"]]

# ---------------------------------------------------------------- 2. p50c directions from the checkpoint diff
U50, COLS50 = {}, {}
with safe_open(str(BASE), "pt") as fb, safe_open(str(CKPT), "pt") as fc:
    for L in (15, 19):
        n = P.format(L) + "mlp.down_proj.weight"
        W0, W1 = fb.get_tensor(n).double(), fc.get_tensor(n).double()
        D = W1 - W0
        cols = torch.nonzero((D != 0).any(0)).flatten()
        u_, s, vt = torch.linalg.svd(D[:, cols], full_matrices=False)
        u = u_[:, 0]
        proj0 = u @ W0[:, cols]
        alpha = -(u @ D[:, cols]) / proj0
        U50[L], COLS50[L] = u.float(), cols
        R[f"p50c_L{L}"] = {"changed_cols": int(len(cols)), "sv_top5": [float(x) for x in s[:5]],
                           "rank1_energy": float(s[0] ** 2 / (s ** 2).sum()),
                           "alpha_median": float(alpha.median()), "alpha_q10_q90": [float(alpha.quantile(.1)), float(alpha.quantile(.9))],
                           "rel_change": float(D.norm() / W0.norm())}
R["p50c_cos_u15_u19"] = float(U50[15] @ U50[19])

# ---------------------------------------------------------------- 3. p10c (rank-1 factors) and p25c (sparse) from edit.npz
man10 = json.load(open(LOGS / "exp8_math/runs/p10c/edit/manifest.json"))
z10 = np.load(LOGS / "exp8_math/runs/p10c/edit/edit.npz")
U10 = {}
for i, t in enumerate(man10["tensors"]):
    L = int(t["name"].split(".layers.")[1].split(".")[0])
    a = z10[f"t{i}_a"][:, 0]
    U10.setdefault(L, []).append((t["name"], a / np.linalg.norm(a), t.get("relative_change")))
R["p10c_tensors"] = len(man10["tensors"])
R["p10c_same_u_within_layer"] = {L: float(abs(v[0][1] @ v[1][1])) if len(v) > 1 else None for L, v in sorted(U10.items())}
man25 = json.load(open(LOGS / "exp8_math/runs/p25c/edit/manifest.json"))
z25 = np.load(LOGS / "exp8_math/runs/p25c/edit/edit.npz")
t25 = man25["tensors"][0]
idx = z25["t0_idx"]
R["p25c"] = {"name": t25["name"], "changed_entries": t25["changed_entries"],
             "cols": sorted(set((idx % t25["shape"][1]).tolist()))[:3] + ["..."] + sorted(set((idx % t25["shape"][1]).tolist()))[-2:],
             "n_cols": len(set((idx % t25["shape"][1]).tolist())), "all_new_values_zero": bool((z25["t0_val"] == 0).all())}
for L in (15, 19):
    if L in U10:
        R[f"cos_p10c_u_vs_p50c_u_L{L}"] = float(abs(torch.from_numpy(U10[L][0][1]).float() @ U50[L]))
# p10c direction for layer l is applied to layer l's writers; the residual "after layer l" = hidden_L{l+1}

# ---------------------------------------------------------------- 4. which attention heads write along the p50c directions
FULL = (3, 7, 11, 15, 19, 23)
with safe_open(str(BASE), "pt") as fb:
    O = {L: fb.get_tensor(P.format(L) + "self_attn.o_proj.weight").float() for L in FULL}
    DOWN = {L: fb.get_tensor(P.format(L) + "mlp.down_proj.weight").float() for L in (15, 19)}
g = torch.Generator().manual_seed(0)
RND = torch.randn(1024, 2000, generator=g)
RND /= RND.norm(dim=0, keepdim=True)


def head_scores(u):
    """ratio ||u^T W_O^h|| / RMS over random unit directions of ||r^T W_O^h|| (1.0 = no preference for u)."""
    out = {}
    for L in FULL:
        for h in range(8):
            B = O[L][:, h * 256:(h + 1) * 256]
            out[f"L{L}H{h}"] = float((u @ B).norm() / (RND.T @ B).norm(dim=1).pow(2).mean().sqrt())
    return out


R["head_write_along_u"] = {}
for L in (15, 19):
    hs = head_scores(U50[L])
    rk = sorted(hs.items(), key=lambda kv: -kv[1])
    R["head_write_along_u"][f"p50c_u{L}"] = {"top10": rk[:10],
                                              "mine": {k: (hs[k], 1 + [x for x, _ in rk].index(k)) for k in ("L15H6", "L15H7", "L19H0", "L23H4", "L23H5")}}
    # same metric for the 64 selected MLP neurons vs all neurons of that layer (sanity: they should be high)
    colnorm = DOWN[L].norm(dim=0)
    along = (U50[L] @ DOWN[L]).abs() / colnorm
    sel = torch.zeros(3584, dtype=torch.bool); sel[COLS50[L]] = True
    R["head_write_along_u"][f"p50c_u{L}"]["mlp_selected_vs_rest_cos"] = [float(along[sel].mean()), float(along[~sel].mean())]

# ---------------------------------------------------------------- 5. recorded hidden states: is u a loop direction in MY data, and does E1 shrink it?
meta = json.load(open(WORK / "req_pass2_meta.json"))
H2 = {l: np.load(WORK / f"fwd_2/hidden_L{l}.npy", mmap_mode="r") for l in (16, 19, 20, 23)}
H3 = {l: np.load(WORK / f"fwd_3/hidden_L{l}.npy", mmap_mode="r") for l in (19, 23)}


def regions(m):
    n, a0 = m["len"], m["n_prompt"]
    if m["kind"] == "loop":
        o, Pd = a0 + m["onset"], m["period"]
        return {"loop": np.arange(o + Pd, n - 1), "pre": np.arange(a0 + 100, o - 1)}
    return {"natural": np.arange(a0 + 100, n - 1)}


def proj_stats(Hd, layer, dirs):
    """per region: list over texts of per-token projections onto each direction (dirs: (1024, k))."""
    acc = {"loop": [], "pre": [], "natural": []}
    for i, m in enumerate(meta):
        X = torch.from_numpy(Hd[layer][i, :m["len"]].astype(np.float32))
        Pj = X @ dirs
        for rg, pos in regions(m).items():
            acc[rg].append(Pj[torch.from_numpy(pos)])
    return {rg: torch.cat(v) for rg, v in acc.items()}


def dprime(st):
    a, b = st["loop"], st["natural"]
    return (a.mean(0) - b.mean(0)) / torch.sqrt(0.5 * (a.var(0) + b.var(0)))


R["hidden_projection"] = {}
for name, L_u, lay in (("p50c_u15@hidden_L16(orig)", 15, 16), ("p50c_u19@hidden_L20(orig)", 19, 20)):
    dirs = torch.cat([U50[L_u][:, None], RND[:, :1000]], 1)
    st = proj_stats(H2, lay, dirs)
    d = dprime(st)
    R["hidden_projection"][name] = {"dprime_u": float(d[0]), "abs_dprime_random_q50_q95_q99": [float(d[1:].abs().quantile(q)) for q in (.5, .95, .99)],
                                    "rank_of_u_among_1000_random": int((d[1:].abs() > d[0].abs()).sum()),
                                    "mean_loop_pre_natural": [float(st[r][:, 0].mean()) for r in ("loop", "pre", "natural")]}
for name, L_u, lay in (("p50c_u15@hidden_L19", 15, 19), ("p50c_u19@hidden_L23", 19, 23)):
    dirs = U50[L_u][:, None]
    s2, s3 = proj_stats(H2, lay, dirs), proj_stats(H3, lay, dirs)
    R["hidden_projection"][name + "_orig_vs_E1"] = {
        "orig_mean_loop_pre_natural": [float(s2[r].mean()) for r in ("loop", "pre", "natural")],
        "E1_mean_loop_pre_natural": [float(s3[r].mean()) for r in ("loop", "pre", "natural")],
        "orig_gap_loop_minus_natural": float(s2["loop"].mean() - s2["natural"].mean()),
        "E1_gap_loop_minus_natural": float(s3["loop"].mean() - s3["natural"].mean()),
        "orig_dprime": float(dprime(s2)[0]), "E1_dprime": float(dprime(s3)[0])}

# ---------------------------------------------------------------- 6. my edit (E3) size, cast to bf16 as the evaluation does
from safetensors.torch import load_file
E3 = {}
with safe_open(str(BASE), "pt") as fb:
    for f in sorted((WORK / "edits_archive/E3").glob("*.safetensors")):
        name = f.name[:-len(".safetensors")]
        w = load_file(str(f))["w"].to(torch.bfloat16)
        b = fb.get_tensor(name)
        m = w.view(torch.int16) != b.view(torch.int16)
        E3[name] = {"changed_entries": int(m.sum()), "rel_change": float((w.float() - b.float()).norm() / b.float().norm())}
R["E3"] = E3
R["E3_total_changed"] = sum(v["changed_entries"] for v in E3.values())
R["p50c_total_changed"] = sum(v["changed_entries"] for v in changed.values() if "changed_entries" in v)

json.dump(R, open(OUT, "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))

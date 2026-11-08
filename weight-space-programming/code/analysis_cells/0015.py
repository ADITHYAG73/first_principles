# Per-head copy attention + direct logit attribution (DLA) on the realised next token, 6 full-attention layers x 8 heads.
# DLA_h(t) = c_h(t) . (u[next] - mean_v u[v]) / rms(r24(t)),  u = W_U * (1 + g_final)  (final RMSNorm linearised at t).
import numpy as np, torch, collections
U = WU * (1.0 + GF)[None]                  # (V, 1024)
Ubar = U.mean(0)
def dla_vec(i, pos):
    r24 = torch.from_numpy(H[24][i, pos].astype(np.float32))
    s = torch.rsqrt(r24.pow(2).mean(-1, keepdim=True) + 1e-6)
    nxt = torch.from_numpy(TID[i, pos + 1].astype(np.int64))
    return (U[nxt] - Ubar[None]) * s           # (n, 1024)
res = collections.defaultdict(list)        # key (L, region) -> list of (8,) arrays per text
copy = collections.defaultdict(list)       # key L -> per text (8,) attention to t-P+1 in loop region
copy_base = collections.defaultdict(list)  # same offset measured in the pre-onset region (control)
lp_stats = collections.defaultdict(list)
for i, m in enumerate(P2META):
    n = m["len"]; a0 = m["n_prompt"]
    if m["kind"] == "loop":
        o, Pd = a0 + m["onset"], m["period"]
        regions = {"loop": np.arange(o + Pd, n - 1), "pre": np.arange(a0 + 100, o - 1)}
    else:
        regions = {"natural": np.arange(a0 + 100, n - 1)}
    for rg, pos in regions.items():
        lp_stats[rg].append(np.nanmean(LP[i, pos + 1]))
    loop_pos = torch.from_numpy(regions["loop"]) if "loop" in regions else None
    for L, layer in ATT.items():
        r_in = torch.from_numpy(H[L][i, :n].astype(np.float32))
        _, c, rows = layer.heads(r_in, keep_attn_rows=loop_pos)
        for rg, pos in regions.items():
            d = dla_vec(i, pos)                                      # (n_pos, 1024)
            res[(L, rg)].append(torch.einsum("thd,td->th", c[torch.from_numpy(pos)], d).mean(0).numpy())
        if loop_pos is not None:
            tgt = loop_pos - Pd + 1
            copy[L].append(rows[:, torch.arange(len(loop_pos)), tgt].mean(1).numpy())
print("mean token logprob by region:", {k: round(float(np.mean(v)), 3) for k, v in lp_stats.items()})
np.set_printoptions(precision=3, suppress=True, linewidth=160)
for L in ATT:
    print(f"L{L:2d} copy-attn(loop) {np.mean(copy[L], 0)}")
    for rg in ("loop", "pre", "natural"):
        print(f"     DLA {rg:7s} {np.mean(res[(L, rg)], 0)}")
HEADRES, HEADCOPY = res, copy

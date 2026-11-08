# Pass 6 (TF on E1, its own remaining loops): which heads still copy? Reconstruction uses EDITED weights (load).
import time
t0 = time.time()
F6 = "work/fwd_6"
H6 = {l: np.load(f"{F6}/hidden_L{l}.npy", mmap_mode="r") for l in (11, 15, 19, 23, 24)}
TID6 = np.load(f"{F6}/token_ids.npy"); LP6 = np.load(f"{F6}/token_logprobs.npy")
TKI6 = np.load(f"{F6}/topk_ids.npy"); TKL6 = np.load(f"{F6}/topk_logprobs.npy")
P6META = json.load(open("work/req_pass6_meta.json"))
req6 = json.load(open("work/req_pass6.json"))
assert all((TID6[i, :len(t)] == np.array(t)).all() for i, t in enumerate(req6["token_ids"]))
ATT6 = {L: recon.AttnLayer(load, L) for L in (11, 15, 19, 23)}
for L, hs in ((15, [6, 7]), (19, [0])):
    for h in hs: assert ATT6[L].o[:, h*256:(h+1)*256].abs().max() == 0   # E1 really is in these weights
i = 27; n = P6META[i]["len"]
print("fidelity L23 (edited) on text", i, recon.check_layer(ATT6[23], torch.from_numpy(H6[23][i, :n].astype(np.float32)), torch.from_numpy(H6[24][i, :n].astype(np.float32))))
def dla6(i, pos):
    r24 = torch.from_numpy(H6[24][i, pos].astype(np.float32))
    s = torch.rsqrt(r24.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (U[torch.from_numpy(TID6[i, pos + 1].astype(np.int64))] - Ubar[None]) * s
res6 = collections.defaultdict(list); acopy6 = collections.defaultdict(list); ctrl6 = collections.defaultdict(list)
lp6 = collections.defaultdict(list); byk6 = collections.defaultdict(list); C6 = {}
for i, m in enumerate(P6META):
    n = m["len"]; a0 = m["n_prompt"]
    if m["kind"] == "loop":
        o, Pd = a0 + m["onset"], m["period"]
        regions = {"loop": np.arange(o + Pd, n - 1), "pre": np.arange(a0 + 100, o - 1)}
        for k in range(12):
            s, e = o + k * Pd, min(o + (k + 1) * Pd, n - 1)
            if e - s < Pd // 2: break
            byk6[k].append(float(np.nanmean(LP6[i, s + 1:e + 1])))
    else:
        regions = {"natural": np.arange(a0 + 100, n - 1)}
    for rg, pos in regions.items(): lp6[rg].append(float(np.nanmean(LP6[i, pos + 1])))
    lpos = regions.get("loop")
    for L, layer in ATT6.items():
        _, c, rows = layer.heads(torch.from_numpy(H6[L][i, :n].astype(np.float32)),
                                 keep_attn_rows=None if lpos is None else torch.from_numpy(lpos))
        C6[(i, L)] = c.half()   # keep for the ablation screen
        for rg, pos in regions.items():
            res6[(L, rg)].append(torch.einsum("thd,td->th", c[torch.from_numpy(pos)], dla6(i, pos)).mean(0).numpy())
        if lpos is not None:
            A = np.zeros((8, len(lpos))); Cc = np.zeros((8, len(lpos)))
            for j, t in enumerate(lpos):
                ks = np.arange(t - Pd + 1, a0, -Pd); ks = ks[ks >= o]
                kc = ks + Pd // 2; kc = kc[kc < t]
                A[:, j] = rows[:, j, ks].sum(-1).numpy(); Cc[:, j] = rows[:, j, kc].sum(-1).numpy() if len(kc) else 0
            acopy6[L].append(A.mean(1)); ctrl6[L].append(Cc.mean(1))
np.set_printoptions(precision=3, suppress=True, linewidth=160)
print("E1 mean token logprob by region:", {k: round(float(np.mean(v)), 4) for k, v in lp6.items()}, {k: len(v) for k, v in lp6.items()})
print("E1 confidence by repetition k:", [(k, len(v), round(float(np.mean(v)), 4)) for k, v in sorted(byk6.items())])
for L in ATT6:
    print(f"L{L:2d} copy-all E1   {np.mean(acopy6[L], 0)}\n    copy-all orig {np.mean(ALLCOPY[L], 0)} (pass 2, different texts)\n    control E1    {np.mean(ctrl6[L], 0)}")
    for rg in ("loop", "pre", "natural"):
        print(f"     DLA E1 {rg:7s} {np.mean(res6[(L, rg)], 0)}")
print("elapsed", round(time.time() - t0), "s")

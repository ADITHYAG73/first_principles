# (1) Layer-level DLA (whole layer = mixer + MLP, r_{l} - r_{l-1}) for all 24 layers, loop vs pre vs natural.
# (2) Copy attention summed over ALL previous repetitions (t - kP + 1, k>=1) vs a control offset (t - kP + 1 + P//2).
lay = collections.defaultdict(list)
for i, m in enumerate(P2META):
    n = m["len"]; a0 = m["n_prompt"]
    if m["kind"] == "loop":
        o, Pd = a0 + m["onset"], m["period"]; regions = {"loop": np.arange(o + Pd, n - 1), "pre": np.arange(a0 + 100, o - 1)}
    else:
        regions = {"natural": np.arange(a0 + 100, n - 1)}
    for rg, pos in regions.items():
        d = dla_vec(i, pos)
        v = []
        for l in range(1, 25):
            delta = torch.from_numpy((H[l][i, pos].astype(np.float32) - H[l - 1][i, pos].astype(np.float32)))
            v.append(float((delta * d).sum(-1).mean()))
        lay[rg].append(v)
print("layer-level DLA (layer 1..24; * = full attention):")
for l in range(24):
    tag = "*" if l in ATT else " "
    print(f" layer {l:2d}{tag} loop {np.mean([x[l] for x in lay['loop']]):7.3f}  pre {np.mean([x[l] for x in lay['pre']]):7.3f}  natural {np.mean([x[l] for x in lay['natural']]):7.3f}")
allcopy, ctrl = collections.defaultdict(list), collections.defaultdict(list)
for i, m in enumerate(P2META):
    if m["kind"] != "loop": continue
    n = m["len"]; a0 = m["n_prompt"]; o, Pd = a0 + m["onset"], m["period"]
    pos = np.arange(o + Pd, n - 1); tp = torch.from_numpy(pos)
    for L, layer in ATT.items():
        _, _, rows = layer.heads(torch.from_numpy(H[L][i, :n].astype(np.float32)), keep_attn_rows=tp)
        A = np.zeros((8, len(pos))); C = np.zeros((8, len(pos)))
        for j, t in enumerate(pos):
            ks = np.arange(t - Pd + 1, a0, -Pd); ks = ks[ks >= o]                 # earlier copies inside the loop + loop start
            kc = ks + Pd // 2; kc = kc[kc < t]
            A[:, j] = rows[:, j, ks].sum(-1).numpy(); C[:, j] = rows[:, j, kc].sum(-1).numpy() if len(kc) else 0
        allcopy[L].append(A.mean(1)); ctrl[L].append(C.mean(1))
for L in ATT:
    print(f"L{L:2d} copy-all {np.mean(allcopy[L], 0)}\n    control  {np.mean(ctrl[L], 0)}")
LAYDLA, ALLCOPY, CTRLCOPY = lay, allcopy, ctrl

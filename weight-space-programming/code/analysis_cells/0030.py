# First-order ablation screen on E1 (same method as cell 17): remove c_h from r24, recompute final norm+softmax,
# change in logprob of the actual next token. Lower bound (ignores downstream). Up to 300 seeded positions per region per text.
gen = np.random.default_rng(0)
CAND6 = [(19, 5), (15, 5), (23, 5), (23, 4), (23, 0), (23, 6), (23, 7), (23, 2), (11, 6), (11, 1), (19, 2), (15, 3)]
GROUPS = {"g_clean(19.5+15.5)": [(19, 5), (15, 5)], "g_clean+11.1": [(19, 5), (15, 5), (11, 1)]}
dlp6 = collections.defaultdict(list)
for i, m in enumerate(P6META):
    n = m["len"]; a0 = m["n_prompt"]
    if m["kind"] == "loop":
        o, Pd = a0 + m["onset"], m["period"]; regions = {"loop": np.arange(o + Pd, n - 1), "pre": np.arange(a0 + 100, o - 1)}
    else:
        regions = {"natural": np.arange(a0 + 100, n - 1)}
    for rg, pos in regions.items():
        pos = np.sort(gen.choice(pos, size=min(300, len(pos)), replace=False)); tp = torch.from_numpy(pos)
        r24 = torch.from_numpy(H6[24][i, pos].astype(np.float32)); nxt = torch.from_numpy(TID6[i, pos + 1].astype(np.int64))
        base = logprob_actual(r24, nxt)
        for (L, h) in CAND6:
            dlp6[(L, h, rg)].append(float((logprob_actual(r24 - C6[(i, L)][tp, h].float(), nxt) - base).mean()))
        for g, hs in GROUPS.items():
            dlp6[(g, 0, rg)].append(float((logprob_actual(r24 - sum(C6[(i, L)][tp, h].float() for L, h in hs), nxt) - base).mean()))
print("E1: mean change in logprob of actual next token when head removed (first order); ratio = loop / natural harm")
for key in CAND6 + [(g, 0) for g in GROUPS]:
    v = {rg: float(np.mean(dlp6[(*key, rg)])) for rg in ("loop", "pre", "natural")}
    print(key, {k: round(x, 4) for k, x in v.items()}, "ratio", round(v["loop"] / min(v["natural"], -1e-4), 2))

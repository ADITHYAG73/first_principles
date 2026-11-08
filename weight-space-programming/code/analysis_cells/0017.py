# First-order ablation screen: remove head h's contribution c_h(t) from the final residual r24(t), recompute final norm + full softmax,
# measure change in logprob of the actual next token. Ignores downstream propagation (later heads reading this head) -> lower bound.
# Positions: up to 300 random per region per text (seeded).
gen = np.random.default_rng(0)
CAND = [(15, 7), (15, 6), (19, 0), (19, 5), (23, 0), (23, 5), (23, 4), (11, 1)]
def logprob_actual(r24, nxt):
    x = recon.rms(r24, GF); lg = x @ WU.T
    return lg.log_softmax(-1).gather(1, nxt[:, None])[:, 0]
dlp = collections.defaultdict(list)
for i, m in enumerate(P2META):
    n = m["len"]; a0 = m["n_prompt"]
    if m["kind"] == "loop":
        o, Pd = a0 + m["onset"], m["period"]; regions = {"loop": np.arange(o + Pd, n - 1), "pre": np.arange(a0 + 100, o - 1)}
    else:
        regions = {"natural": np.arange(a0 + 100, n - 1)}
    heads_c = {}
    for L in set(l for l, _ in CAND):
        _, c, _ = ATT[L].heads(torch.from_numpy(H[L][i, :n].astype(np.float32))); heads_c[L] = c
    for rg, pos in regions.items():
        pos = np.sort(gen.choice(pos, size=min(300, len(pos)), replace=False))
        r24 = torch.from_numpy(H[24][i, pos].astype(np.float32)); nxt = torch.from_numpy(TID[i, pos + 1].astype(np.int64))
        base = logprob_actual(r24, nxt)
        for (L, h) in CAND:
            dlp[(L, h, rg)].append(float((logprob_actual(r24 - heads_c[L][torch.from_numpy(pos), h], nxt) - base).mean()))
        allc = sum(heads_c[L][torch.from_numpy(pos), h] for L, h in CAND[:3])
        dlp[("trio", 0, rg)].append(float((logprob_actual(r24 - allc, nxt) - base).mean()))
print("mean change in logprob of actual next token when the head is removed (first order):")
for key in CAND + [("trio", 0)]:
    print(key, {rg: round(float(np.mean(dlp[(*key, rg)])), 4) for rg in ("loop", "pre", "natural")})

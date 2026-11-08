# Remaining loops under E1 (pass 4 tokens, local only).
cutO = [r for r in G4["original"].values() if r["category"] == "cut_off"]
LO = {r["rid"]: token_loop(r["token_ids"]) for r in cutO}
LE = {r["rid"]: token_loop(r["token_ids"]) for r in cutE}
fo = [v for v in LO.values() if v]; fe = [v for v in LE.values() if v]
print("original cut", len(cutO), "exact loops", len(fo), "onset q", np.percentile([v[0] for v in fo], [25, 50, 75]).astype(int).tolist(), "period q", np.percentile([v[1] for v in fo], [25, 50, 75]).astype(int).tolist())
print("E1       cut", len(cutE), "exact loops", len(fe), "onset q", np.percentile([v[0] for v in fe], [25, 50, 75]).astype(int).tolist(), "period q", np.percentile([v[1] for v in fe], [25, 50, 75]).astype(int).tolist())
# E1 cut-offs WITHOUT exact loop: what are they? (unique-line share in last 4k tokens)
nl = [r for r in cutE if LE[r["rid"]] is None]
def uniq_tail(r):
    lines = tok.decode(r["token_ids"][-4000:]).split("\n"); lines = [l.strip() for l in lines if l.strip()]
    return round(len(set(lines)) / max(1, len(lines)), 3)
print("E1 cut-offs without exact loop:", len(nl), "unique-line share of last 4k tokens:", sorted(uniq_tail(r) for r in nl))
# Did E1 reach the gold answer before looping? (gold string appears in text before loop onset)
def gold_before(r, onset):
    txt = tok.decode(r["token_ids"][:onset]); g = str(r["gold"]).strip()
    return g in txt
ge = [gold_before(r, LE[r["rid"]][0]) for r in cutE if LE[r["rid"]]]
print("E1 loops where gold string appears before onset:", sum(ge), "/", len(ge))
# Confidence growth inside E1 loops is not measurable without a TF pass (no logprobs from generation). 
# Period start tokens and a few loop samples (<=250 chars).
rng = random.Random(11)
for r in rng.sample([r for r in cutE if LE[r["rid"]]], 3):
    o, P, _ = LE[r["rid"]]; print("===", r["rid"], "onset", o, "period", P); print(repr(tok.decode(r["token_ids"][o:o + P])[:250]))

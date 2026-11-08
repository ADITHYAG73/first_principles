# E1 vs original on identical tokens: change in logprob of the actual token by region and by repetition index.
LP3 = np.load("work/fwd_3/token_logprobs.npy"); TID3 = np.load("work/fwd_3/token_ids.npy")
assert (np.nan_to_num(TID3) == np.nan_to_num(TID)).all()
reg = collections.defaultdict(list); byk3 = collections.defaultdict(list)
for i, m in enumerate(P2META):
    n = m["len"]; a0 = m["n_prompt"]
    if m["kind"] == "loop":
        o, Pd = a0 + m["onset"], m["period"]
        regions = {"loop": np.arange(o + Pd, n - 1), "pre": np.arange(a0 + 100, o - 1)}
        for k in range(12):
            s, e = o + k * Pd, min(o + (k + 1) * Pd, n - 1)
            if e - s < Pd // 2: break
            byk3[k].append((float(np.nanmean(LP[i, s + 1:e + 1])), float(np.nanmean(LP3[i, s + 1:e + 1]))))
    else:
        regions = {"natural": np.arange(a0 + 100, n - 1)}
    for rg, pos in regions.items():
        reg[rg].append((float(np.nanmean(LP[i, pos + 1])), float(np.nanmean(LP3[i, pos + 1]))))
for rg, v in reg.items():
    v = np.array(v); print(f"{rg:8s} orig {v[:,0].mean():.4f}  E1 {v[:,1].mean():.4f}  delta {np.mean(v[:,1]-v[:,0]):+.4f}  (per-text deltas min {np.min(v[:,1]-v[:,0]):+.3f} max {np.max(v[:,1]-v[:,0]):+.3f})")
print("by repetition k: orig -> E1")
for k in sorted(byk3):
    v = np.array(byk3[k]); print(k, len(v), f"{v[:,0].mean():.4f} -> {v[:,1].mean():.4f}")
# per-token: in loop region, share of tokens whose prob fell below 0.5 (where sampling could leave the loop)
for name, L_ in (("orig", LP), ("E1", LP3)):
    sh = [np.mean(L_[i, a0 + m["onset"] + m["period"] + 1: m["len"]] < np.log(0.5)) for i, m in enumerate(P2META) if m["kind"] == "loop" for a0 in [m["n_prompt"]]]
    print(name, "loop tokens with p<0.5:", round(float(np.mean(sh)), 4))

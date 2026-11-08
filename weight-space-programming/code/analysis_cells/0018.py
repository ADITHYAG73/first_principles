# Confidence by repetition index k (k=0: first time the period text is written, k=1: first copy, ...), and </think> availability.
END = tok.token_to_id("</think>"); print("</think> id", END)
byk = collections.defaultdict(list); end_lp = collections.defaultdict(list)
for i, m in enumerate(P2META):
    if m["kind"] != "loop": continue
    n = m["len"]; a0 = m["n_prompt"]; o, Pd = a0 + m["onset"], m["period"]
    for k in range(0, 12):
        s, e = o + k * Pd, min(o + (k + 1) * Pd, n - 1)
        if e - s < Pd // 2: break
        byk[k].append(float(np.nanmean(LP[i, s + 1:e + 1])))
        # logprob of </think> at period start (if in top-20; else < min top-20)
        tk = TKI[i, s - 1]; tl = TKL[i, s - 1]
        end_lp[k].append(float(tl[list(tk).index(END)]) if END in tk else float("nan"))
print("k, n_texts, mean logprob per token in period k, share of texts with </think> in top-20 at period start")
for k in sorted(byk):
    v = np.array(end_lp[k]); print(k, len(byk[k]), round(float(np.mean(byk[k])), 3), f"{np.mean(~np.isnan(v)):.2f}")
# natural texts: where is </think>, and what was its logprob there + 200 tokens before
for i, m in enumerate(P2META):
    if m["kind"] != "natural": continue
    ids = TID[i, :m["len"]]; w = np.where(ids == END)[0]
    if len(w) == 0: print(m["rid"], "no </think>"); continue
    t = w[0]; print(m["rid"], "</think> at", t - m["n_prompt"], "logprob", round(float(LP[i, t]), 3))

print(sorted(wa[0].keys()))
for r in sorted(wa, key=lambda r: r["rid"]):
    print(r["rid"], {k: r[k] for k in r if k not in ("text", "token_ids", "prompt")})

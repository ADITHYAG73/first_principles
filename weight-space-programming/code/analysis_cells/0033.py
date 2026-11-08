for r in sorted(na, key=lambda r: r["rid"]):
    print("=" * 100); print(r["rid"], "| gold", repr(r["gold"]), "| E1:", E1[r["rid"]]["category"], "| n_tokens", r["n_tokens"])
    print(r["text"].split("</think>")[-1])

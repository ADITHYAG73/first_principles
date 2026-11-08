# E1 wrong answers in pass 5: real errors or grader artefacts? Print gold, extracted pred, and the full final answer text after </think>.
for r in sorted(wa, key=lambda r: r["rid"]):
    o = G5["original"][r["rid"]]
    after = r["text"].split("</think>")[-1]
    print("=" * 100); print(r["rid"], "| gold:", repr(r.get("gold")), "| pred:", repr(r.get("pred")), "| orig:", o["category"], o["n_tokens"])
    print(after)

# What does the text look like at loop onset? 6 random cut-offs (seeded): ~250 chars before onset and one loop period (<=300 chars).
import random
rng = random.Random(7)
samp = rng.sample([r for r in cut if LOOPS[r["rid"]]["onset"] is not None], 6)
for r in samp:
    L = LOOPS[r["rid"]]; o, P = L["onset"], L["period"]; ids = r["token_ids"]
    before = tok.decode(ids[max(0, o - 80):o]); per = tok.decode(ids[o:o + P])
    print(f"=== {r['rid']} onset={o} period={P} gold={r['gold']}")
    print("BEFORE:", repr(before[-250:])); print("PERIOD:", repr(per[:300]))
# Statistics: does the loop text contain the gold answer? does it contain 'Wait'?
import re
has_wait = sum("Wait" in tok.decode(r["token_ids"][LOOPS[r["rid"]]["onset"]:LOOPS[r["rid"]]["onset"] + LOOPS[r["rid"]]["period"]]) for r in cut if LOOPS[r["rid"]]["onset"] is not None)
print("loop periods containing 'Wait':", has_wait, "/", len(found))
# first-token ids of the period: which token starts each loop period (most common)
c = collections.Counter(r["token_ids"][LOOPS[r["rid"]]["onset"]] for r in cut if LOOPS[r["rid"]]["onset"] is not None)
print("most common period-start token ids:", c.most_common(8))

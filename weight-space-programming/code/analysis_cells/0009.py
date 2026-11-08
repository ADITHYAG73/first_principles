# Hypothesis: loops are exact token-level periodic repeats; find onset and period in TOKEN space for each cut-off,
# to build teacher-forced windows that contain the loop onset + several repetitions (needed to find copy heads).
import numpy as np
def token_loop(ids, min_period=8, max_period=400, reps=3):
    """Earliest position t such that ids[t:t+reps*P] is P-periodic for some P; returns (onset, period)."""
    a = np.array(ids)
    n = len(a)
    best = None
    for P in range(min_period, max_period + 1):
        eq = a[P:] == a[:-P]                      # eq[i]: ids[i+P] == ids[i]
        # need a run of reps*P-P consecutive matches
        need = (reps - 1) * P
        run = 0
        for i in range(len(eq) - 1, -1, -1):      # scan backwards: loop persists to the end
            if eq[i]:
                run += 1
            else:
                break
        if run >= need:
            onset = len(eq) - run
            if best is None or onset < best[0]:
                best = (onset, P, run)
    return best
cut = [r for r in P1 if r["category"] == "cut_off"]
loops = []
for r in cut:
    b = token_loop(r["token_ids"])
    loops.append(dict(rid=r["rid"], onset=b[0] if b else None, period=b[1] if b else None,
                      periodic_tail=b[2] if b else 0))
found = [l for l in loops if l["onset"] is not None]
print(f"exact periodic tail found in {len(found)}/{len(cut)} cut-offs")
on = np.array([l["onset"] for l in found]); per = np.array([l["period"] for l in found])
print("onset (token) quartiles", np.percentile(on, [0, 25, 50, 75, 100]).astype(int).tolist())
print("period (tokens) quartiles", np.percentile(per, [0, 25, 50, 75, 100]).astype(int).tolist())
print("periodic tail length (tokens) median", int(np.median([l["periodic_tail"] for l in found])))
LOOPS = {l["rid"]: l for l in loops}

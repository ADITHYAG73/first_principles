"""Authors' released per-problem results (unedited, p10c, p25c, p50c) restricted to (a) my 64 held-out problems,
re-derived from remote/prepare_heldout.py's seeded selection, and (b) the user's replication subset (first 64 MATH-500).
Prints indices / ids and accuracies only (no problem text).

usage: python -I same_problems.py <logs_dir> <math500_test.jsonl> <out.json>
"""
import glob
import json
import random
import sys
from pathlib import Path

LOGS, M5, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
math500 = [json.loads(l) for l in open(M5)]
assert len(math500) == 500
N_GSM_TEST = 1319
rng = random.Random(20261104)                       # exactly as remote/prepare_heldout.py
m_idx = sorted(rng.sample(range(len(math500)), 32))
g_idx = sorted(rng.sample(range(N_GSM_TEST), 32))
my_m_ids = [math500[i]["unique_id"] for i in m_idx]
my_g_ids = [f"gsm8k_test_{i}" for i in g_idx]
first64 = [math500[i]["unique_id"] for i in range(64)]

R = {"my_math500_indices": m_idx, "my_gsm8k_indices": g_idx}
for model in ("unedited", "p10c", "p25c", "p50c"):
    per = {}
    for bench in ("math500", "gsm8k"):
        for f in glob.glob(str(LOGS / f"exp8_math/eval/{model}/parts/{bench}_s*/{bench}_per_problem.json")):
            for r in json.load(open(f)):
                per[r["id"]] = r["correct"]

    def acc(ids):
        miss = [i for i in ids if i not in per]
        got = [c for i in ids if i in per for c in per[i]]
        return {"acc": round(100 * sum(got) / len(got), 2) if got else None, "samples": len(got), "missing": len(miss)}

    full_m = [c for k, v in per.items() if not k.startswith("gsm8k") for c in v]
    full_g = [c for k, v in per.items() if k.startswith("gsm8k") for c in v]
    R[model] = {"my_heldout_math500": acc(my_m_ids), "my_heldout_gsm8k": acc(my_g_ids),
                "my_heldout_both": acc(my_m_ids + my_g_ids),
                "user_first64_math500": acc(first64),
                "full_math500": round(100 * sum(full_m) / len(full_m), 2), "full_gsm8k": round(100 * sum(full_g) / len(full_g), 2),
                "n_full_math500_problems": sum(1 for k in per if not k.startswith("gsm8k")),
                "n_full_gsm8k_problems": sum(1 for k in per if k.startswith("gsm8k"))}
json.dump(R, open(OUT, "w"), indent=1)
print(json.dumps(R, indent=1))

# Hypothesis: the dominant failure is runaway thinking that hits the 32k cap (likely a repetition loop).
# Grade the 152 finished pass-1 answers (worker died before grading) and build the failure taxonomy.
import sys, collections, statistics
sys.path.insert(0, "remote"); sys.path.insert(0, "wsp")  # worker imports common.py from wsp/
import importlib, worker as W  # grading/category helpers only (pure-python, no model code)
P1 = [json.loads(l) for l in open("work/pass1_partial/gen_original_gpu0.jsonl")]
items = {pid: dict(gold=PRACTICE[pid]["answer"]) for pid in set(r["id"] for r in P1)}
P1 = W.finalize_gen(P1, items)
for r in P1:
    r["level"] = PRACTICE[r["id"]].get("level")
print("finished:", len(P1), "by source", collections.Counter(r["source"] for r in P1))
for src in ("gsm8k", "math"):
    rs = [r for r in P1 if r["source"] == src]
    if rs:
        print(src, "n", len(rs), "categories", dict(collections.Counter(r["category"] for r in rs)),
              "acc", round(sum(bool(r["correct"]) for r in rs) / len(rs), 3))
cut = [r for r in P1 if r["category"] == "cut_off"]
print("cut-offs:", len(cut), "| think closed among cut:", sum(r["think_closed"] for r in cut),
      "| dup_line_share median:", statistics.median(r["dup_line_share"] for r in cut) if cut else None,
      "| max_window_repeats>=3:", sum(r["max_window_repeats"] >= 3 for r in cut))
nat = [r for r in P1 if r["category"] != "cut_off"]
print("natural finishes:", len(nat), "lengths median", statistics.median(r["n_tokens"] for r in nat) if nat else None,
      "acc among natural", round(sum(bool(r["correct"]) for r in nat) / max(1, len(nat)), 3))

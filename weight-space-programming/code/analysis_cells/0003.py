# Hypothesis: the base model's math failures fall into a few categories (wrong answer / no answer / cut off
# at 32k while looping); pass 1 measures that taxonomy on a stratified practice sample, with activations.
import random
rng = random.Random(1)
gsm = sorted(k for k, v in PRACTICE.items() if v["source"] == "gsm8k")
math_by_level = {L: sorted(k for k, v in PRACTICE.items() if v["source"] == "math" and v.get("level") == L) for L in range(1, 6)}
p1_gsm = rng.sample(gsm, 128)
p1_math = [pid for L in range(1, 6) for pid in rng.sample(math_by_level[L], [26, 26, 26, 25, 25][L - 1])]
p1 = p1_gsm + p1_math
req1 = {"type": "generate", "model": "original", "problems": p1, "n": 2, "max_tokens": 32768,
        "activations": {"layers": list(range(25)), "mlp": True}}
json.dump(req1, open("work/req_pass1.json", "w"))
USED = set(p1)   # problems used in diagnosis: never reuse them for confirmation
json.dump(sorted(USED), open("work/used_problems.json", "w"))
import collections
print(len(p1), collections.Counter(PRACTICE[p]["source"] for p in p1), collections.Counter(PRACTICE[p].get("level") for p in p1_math),
      collections.Counter(PRACTICE[p].get("subject") for p in p1_math))

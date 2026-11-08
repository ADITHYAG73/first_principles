"""Runs ON THE WORKER ONLY. Downloads the held-out benchmarks (MATH-500, GSM8K test) and fixes a
seeded selection of 32 + 32 problems for the final edited-vs-original comparison.
Nothing from here is ever copied back to the analysis machine except aggregate scores.
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

OUT = Path("/opt/wsp/data/heldout")
N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 32
SEED = 20261104


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m5 = HfApi().dataset_info("HuggingFaceH4/MATH-500")
    f = [s.rfilename for s in m5.siblings if s.rfilename.endswith(".jsonl")]
    assert len(f) == 1, f
    math500 = [json.loads(l) for l in open(hf_hub_download("HuggingFaceH4/MATH-500", f[0], repo_type="dataset",
                                                           revision=m5.sha))]
    g = HfApi().dataset_info("openai/gsm8k")
    gt = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset",
                                          revision=g.sha))
    rng = random.Random(SEED)
    rows = []
    for i in sorted(rng.sample(range(len(math500)), N_EACH)):
        r = math500[i]
        rows.append(dict(id=f"heldout-math500-{i:03d}", source="math500", problem=r["problem"], answer=r["answer"]))
    for i in sorted(rng.sample(range(len(gt)), N_EACH)):
        r = gt.iloc[i]
        rows.append(dict(id=f"heldout-gsm8k-{i:04d}", source="gsm8k_test", problem=r["question"],
                         answer=r["answer"].split("####")[-1].strip().replace(",", "")))
    with open(OUT / "selection.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    json.dump(dict(math500_rev=m5.sha, gsm8k_rev=g.sha, seed=SEED, n_each=N_EACH, n_math500=len(math500),
                   n_gsm8k_test=len(gt)), open(OUT / "selection_meta.json", "w"))
    print(f"held-out selection written: {len(rows)} problems (contents not printed)")


if __name__ == "__main__":
    main()

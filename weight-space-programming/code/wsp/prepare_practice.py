"""Build the practice sets (TRAIN splits only) as data/practice/*.jsonl.

Held-out evaluation sets (MATH-500, GSM8K test) are deliberately NOT downloaded here:
they are prepared on the remote worker only and never read locally.
Schema per line: id, source, problem, answer, [solution, subject, level, year].
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parent.parent / "data" / "practice"
REVS = {  # pinned dataset revisions (sha of the HF repo at download time)
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "di-zhang-fdu/AIME_1983_2024": "3e2cc86390666c5c756622afc0eeb9e6194496bc",
}
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def last_boxed(s: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}) with brace matching."""
    idx = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j]
    return None


def fetch(repo: str, filename: str) -> str:
    return hf_hub_download(repo, filename, repo_type="dataset", revision=REVS[repo])


def gsm8k() -> list[dict]:
    df = pd.read_parquet(fetch("openai/gsm8k", "main/train-00000-of-00001.parquet"))
    rows = []
    for i, r in df.iterrows():
        sol, ans = r["answer"].split("####")
        rows.append(dict(id=f"gsm8k-train-{i:04d}", source="gsm8k", problem=r["question"],
                         answer=ans.strip().replace(",", ""), solution=sol.strip()))
    return rows


def math_train() -> list[dict]:
    rows, skipped = [], 0
    for subj in MATH_SUBJECTS:
        df = pd.read_parquet(fetch("EleutherAI/hendrycks_math", f"{subj}/train-00000-of-00001.parquet"))
        for i, r in df.iterrows():
            ans = last_boxed(r["solution"])
            if ans is None:
                skipped += 1
                continue
            lvl = str(r["level"]).replace("Level ", "")
            rows.append(dict(id=f"math-train-{subj}-{i:04d}", source="math", problem=r["problem"],
                             answer=ans, solution=r["solution"], subject=subj,
                             level=int(lvl) if lvl.isdigit() else None))
    print(f"math: kept {len(rows)}, skipped {skipped} without a boxed answer")
    return rows


def aime() -> list[dict]:
    df = pd.read_csv(fetch("di-zhang-fdu/AIME_1983_2024", "AIME_Dataset_1983_2024.csv"))
    print("aime columns:", list(df.columns))
    df = df[df["Year"] <= 2023]
    rows = []
    for _, r in df.iterrows():
        rows.append(dict(id=f"aime-{r['ID']}", source="aime", problem=r["Question"],
                         answer=str(r["Answer"]).strip(), year=int(r["Year"])))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [("gsm8k_train", gsm8k()), ("math_train", math_train()), ("aime", aime())]:
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} problems -> {OUT / (name + '.jsonl')}")


if __name__ == "__main__":
    main()

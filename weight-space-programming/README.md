# Weight-space programming, blind: Claude Opus 5.5 on the Astra task

Article: [index.html](index.html) (live at https://adithyag73.github.io/first_principles/weight-space-programming/).

Claude Opus 5.5 was given only the task prompt from
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang,
Khattab, MIT CSAIL, 2026-11-02), whose experiments used OpenAI's [GPT-6 Astra](https://openai.com/index/gpt-6-astra/).
It built its own harness on one AWS g5.xlarge, ran 10 forward passes over 2026-11-04/05, and edited 3 attention output
matrices of Qwen3.5-0.8B. Everything behind the article's numbers is here.

**To replicate with a coding agent** (Claude Code, Codex): follow [SKILL.md](SKILL.md).

| Folder | What it holds |
|---|---|
| `task/` | The exact task prompt (identical to the authors' prompt, word for word). |
| `code/wsp/`, `code/remote/` | The harness Opus wrote: analysis kernel, forward-pass client, AWS launcher, GPU worker, grader. `code/wsp/recon.py` is the local re-implementation of the attention layers that breaks the task's rule (b); see the article. |
| `code/analysis_cells/` | Every analysis step that ran (`.py`) and its printed output (`.out`), in order. |
| `code/comparison/` | Post-hoc comparison with the authors' released edits and per-problem results. |
| `records/` | `journal.md` (preflights, results, the pre-registered decision rule), `notes.md` (findings F1–F5), `writeup.md`, `comparison_with_article.md`, held-out aggregates, compressed run logs and pass requests. |
| `generations/` | Every graded practice answer, gzipped JSONL, one record per answer (`rid`, `seed`, `text`, `correct`, `category`, ...). Passes: 1 (original), 4 and 5 (original vs E1), 7 (E2), 8 and 9 (E3), 10 (E1). |
| `edits/E3/` | The final edit. The full checkpoint is on Hugging Face: [TokenGod/Qwen3.5-0.8B-opus-blind-weight-edit](https://huggingface.co/TokenGod/Qwen3.5-0.8B-opus-blind-weight-edit). The folder's README has the exact changes, hashes and a three-line rebuild. |
| `audit_by_astra/` | GPT-6 Astra's independent audit of this run (evidence and mechanism audits, regrade of all 512 held-out answers). |

Not included: activations and teacher-forced hidden states (≈ 10 GB), token-id files, held-out problem texts and
answers. The AWS account ID is replaced with `ACCOUNTID` in the copied files.

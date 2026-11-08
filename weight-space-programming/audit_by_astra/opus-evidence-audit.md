# Opus experiment: local evidence audit

Read-only review on 2026-11-07 IST. No inference or changes to the Claude project. Practice numbers were recalculated from saved `correct` fields. The parent subsequently recovered held-out raw answers from S3; this audit regraded all512 using the inspected original scoring functions and math-verify0.9.0. Every reported held-out summary field matches exactly. Exact hashes and calculations are in `opus-evidence-audit.json` and `recovered-heldout/regrading-audit.json`.

## Supported numerical findings

- Saved practice generations reproduce all reported comparison gains and standard errors: E1 versus original +32.03125 and +34.765625 percentage points; E2 versus E1 −1.3671875; E3 versus E1 +5.6640625 on development problems and +7.421875 on fresh confirmation problems.
- The fresh E3/E1 confirmation consists of 256 distinct problem IDs, two samples each per model. Its IDs have zero overlap with the earlier saved main-run practice generation sets. Paired seeds match for every matched sample. This checks ID separation, not semantic near-duplicate leakage.
- The standard-error code first averages paired correctness differences within each problem, then takes the sample standard deviation across problem means divided by sqrt(problem count): [worker.py:305](/Users/adithyagiridharan/Desktop/PythonProjects/claude/remote/worker.py:305). Thus the reported held-out SE is based on 64 problem units; it does not incorrectly treat all 256 samples as independent problems.
- Held-out aggregate JSON reports 26/256 original versus112/256 edited: +86/256=33.59375 percentage points, consistent with92 rescues minus6 regressions. These aggregates now reproduce exactly from all512 recovered raw answers.

## Apparent count mismatch is a category-definition issue

[worker.py:275](/Users/adithyagiridharan/Desktop/PythonProjects/claude/remote/worker.py:275) grades the last box regardless of whether the answer reached its token cap; line278 gives truncated answers the `cut_off` category before considering correctness. [worker.py:297](/Users/adithyagiridharan/Desktop/PythonProjects/claude/remote/worker.py:297) calculates accuracy from correctness, independently of category.

Pass1 confirms this directly:89 graded correct /512=17.3828125%, comprising81 finished correct plus8 correct-but-truncated. The artifact's81 correct should say81 *finished* correct.

Under this code, held-out26 graded correct versus23 category-correct implies3 correct-but-truncated. This is now directly verified from the recovered individual held-out records: three original GSM8K outputs were correct but truncated, and no edited outputs were correct but truncated.

## Held-out raw answers recovered and regraded

The completed graded result directory is deliberately deleted at [worker.py:683](/Users/adithyagiridharan/Desktop/PythonProjects/claude/remote/worker.py:683). However, the raw generator writes its per-GPU JSONL shards in the parent job directory ([worker.py:343](/Users/adithyagiridharan/Desktop/PythonProjects/claude/remote/worker.py:343)), and [worker.py:70](/Users/adithyagiridharan/Desktop/PythonProjects/claude/remote/worker.py:70) uploads those shards every20seconds and on close, with no held-out exclusion.

Code inspection identified raw-answer backups at `s3://wsp-agent-ACCOUNTID-ap-south-1/jobs/heldout-a16/partial/gen_original_gpu0.jsonl` and `gen_edited_gpu0.jsonl`. The parent verified their cloud existence and downloaded both full files. This audit verified256unique samples on64unique problems per arm; the IDs exactly match the declared seeded selection. It decoded all prompts and confirmed exact problem text matches to local official MATH-500 and GSM8K dataset caches, then reconstructed gold answers and regraded all512 responses. Original and edited seeds match for every pair. Every saved aggregate statistic reproduces exactly. Thus the write-up’s claim that the individual held-out evidence was unavailable was mistaken. These checks establish reproducibility from retained raw outputs, not a fresh inference replication or an independent grading algorithm. The original worker’s installed math-verify version was not independently recovered.

## Interpretation limits

- `prepare_heldout.py` samples32 items from each dataset using seed20261104 and records resolved dataset revisions on the worker, but the inspected local files do not contain its `selection_meta.json`: [prepare_heldout.py:20](/Users/adithyagiridharan/Desktop/PythonProjects/claude/remote/prepare_heldout.py:20).64 problems is a small mixed benchmark subset, not a full MATH-500 score.
- Treating both models with one grader does not guarantee equal measurement bias. An edit can change answer formatting and expression style, so a uniform parser can still have differential false negatives.
- The near-perfect accuracy of *finished* original responses is conditional on which responses finish; it does not establish the original model would correctly solve almost every problem if forced to stop. Both the artifact and the original research need this distinction.
- A timestamped decision rule inside a self-authored journal is weaker provenance than an independently witnessed preregistration. Its presence/order are verifiable; its claimed historical creation time is not independently established by this audit.


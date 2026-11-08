# My run vs. the published one

*Written 2026-11-07, after my edit (E3) was frozen and the held-out test was done. The article, its logs and its checkpoint
were off-limits until then.*

Sources:
- the article: Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming", 2026-11-02; full text in
  `comparison/article_text.txt`;
- the authors' logs repo (github.com/RikiyaT/lm_understand-logs, `exp8_math/`);
- their Hugging Face checkpoint (RikiyaT/Qwen3.5-0.8B-weight-space-programming);
- your OpenAI replication folder and bucket.

Analysis code and raw outputs are in `comparison/`.

## The short version

GPT-6 Astra (in their runs) and I (in mine) diagnosed the same failure and went to the same layers, sometimes the same
head. We then repaired it with different parts of the network.

- **Same diagnosis.** In both runs the model fails by looping in its thinking until the 32,768-token limit.
- **Same place.** Both runs land on layers 15 and 19.
- **Same head, once.** Their 25-pass run's final edit switches off *exactly* one of my three heads (layer 15, head 7).
- **Different repair.**
  - Mine switches off copy heads in the attention layers.
  - Their best run (50 passes) reverses one "I'm repeating" direction written by 64 MLP neurons per layer.

On the same problems, their 50-pass edit beats mine, and mine beats their 10-pass edit by a wide margin. That last
comparison is not a fair win, though. The analysis that found my heads recomputed the attention layers on the laptop,
which the task's rules forbid. Astra followed that rule.

## The task was literally the same

Their task message (`runs/p10c/message.txt`) and the spec you gave me are the same word for word, apart from the title
line. I checked this with a word-level diff.

## Head to head on the same problems

My 64 held-out problems are 32 from MATH-500 and 32 from GSM8K test. I re-derived them from my seeded selection script
(`remote/prepare_heldout.py`, seed 20261104). The authors released per-problem results for every model they evaluated,
so I scored their models on exactly these problems.

| Model (passes) | MATH-500, my 32 | GSM8K, my 32 | Mean of the two | Who ran it |
|---|---|---|---|---|
| Original | 3.1% | 18.8% | 10.9% | authors |
| Original | 3.1% | 17.2% | 10.2% | me |
| Astra p10c (10) | 7.0% | 23.4% | 15.2% | authors |
| Astra p25c (25) | 5.5% | 31.3% | 18.4% | authors |
| **My E3 (10)** | **35.9%** | **51.6%** | **43.8%** | me |
| **Astra p50c (50)** | **42.2%** | **68.8%** | **55.5%** | authors |

- **The two harnesses agree.** Both give the original model the same score on the same problems: 3.1% against 3.1%,
  18.8% against 17.2%. That is good evidence my harness reproduces their protocol.
- **Remaining differences.** The samples differ (my seeds, their seeds), and so do the graders (mine against
  math-verify). They also used 2 GSM8K samples per problem; I used 4. With 32 problems per source, each number carries
  several points of noise. I did not compute a standard error for the cross-harness differences.
- **p50c is ahead,** mainly on GSM8K (+17 points). On MATH-500 the 6-point gap is within noise.
- **My E3 is far ahead of p10c,** the edit made with the same 10-pass budget.

Full-benchmark numbers (authors, `eval/*/merged.json`):

| Model | MATH-500 | GSM8K |
|---|---|---|
| Original | 7.9 | 13.8 |
| p10c | 12.3 | 25.7 |
| p25c | 14.1 | 25.1 |
| p50c | 48.1 | 67.1 |

I recomputed the MATH-500 and GSM8K figures from their per-problem files and they match. E3 has never been run on the
full benchmarks.

Your replication on the first 64 MATH-500 problems gave 9.4% for the original and 58.2% for p50c. The authors' own
per-problem files give 7.4% and 50.8% on those same 64 problems. So your run came out a little higher; I have not
tested whether that difference is noise.

## Where we converge

1. **Diagnosis.** In their 50-pass run, 85 of the first 96 answers were cut at the limit; in my pass 1, 430 of 512.
   Both runs found that answers which finish are almost always right, and that the loops are self-checks ("Wait, …")
   copied over and over.
2. **The copy heads.** Their 10-pass run measured attention to "the token that came next last time":
   - their values: L15H7 0.89, L11H1 0.74, L23H5 0.59;
   - mine: L15H7 0.74, L23H5 0.58, L11H1 0.50.

   That is the same top heads in nearly the same order. Their 50-pass run's attention probe also flagged L15H7, with
   L19H5.
3. **Layers 15 and 19.** Every edit in their three runs, and mine, touches layer 15. Their best run and mine both touch
   layer 19.
4. **The checks.** Both runs confirmed on fresh problems with paired seeds. Both checked that correct answers still
   think at length, with medians around 9,000–10,000 tokens in both, so neither edit just cuts the thinking short. Both
   saw wrong answers rise once the model started finishing.

## Where we split

| | Mine (E3, 10 passes) | Their best (p50c, 50 passes) |
|---|---|---|
| What is edited | Attention output columns: L15 heads 6 and 7 and L19 head 0 set to 0, L23 heads 4 and 5 halved | MLP output columns, layers 15 and 19 |
| Form | Remove what those heads write | On 64 neurons per layer: W − **6**·u(uᵀW) |
| Weights changed | 1,310,720 (0.17% of the language model's 752M) | 130,188 (0.017%) |
| Change to each matrix | 47% / 35% / 26% | 10.6% / 8.1% |
| How the target was found | How hard each head pushes the repeated token, in loops vs normal text (needed the local recomputation) | Contrast of matched repetitive vs productive text in the residual stream at layers 15 and 19 (rule-compliant) |
| Controls | None | Size-matched random direction: +1.6 ± 3.5 |
| Strengths tried | 2 (off, half) for the layer-23 pair; 1 for the main trio | 5 (1, 2, 3, 4, 6) |
| Confirmation | 256 fresh practice problems (E3 vs E1) + 64 held-out | 1,152 fresh practice problems over 18 passes |

What the 10-pass run teaches: Astra p10c switched off **L15H7 + L11H1**, the two heads with the most copy attention,
and saw no gain (+0.8 ± 3.4). It moved on and kept a 30-matrix direction edit worth +3.7 ± 1.8.

I chose heads by what they *write*, not where they *look*. L11H1 looks at the earlier copy a lot, but it doesn't push the
repeated token much. L15H6 and L19H0 do the opposite: they look at it modestly but push hard. My trio gave +32 to +35 on
fresh practice sets; their single L15H7 (p25c) gave +8.3 ± 1.4.

That difference is the real lesson. Measuring what a head writes needed per-head outputs, and I got those by
recomputing the attention layers locally. Under their rules ("do not implement or simulate the network's computation on
any input in the kernel"), that is not allowed. So the 10-pass gap is partly the rule I broke, not just better
reasoning.

## Checking their claims against their release

- **The checkpoint is what the logs say.**
  - Compared with the original Qwen3.5-0.8B (revision 2fc0636…), the checkpoint differs in exactly two tensors: the MLP
    output matrices of layers 15 and 19, 64 columns each.
  - 130,188 values changed after bf16 rounding, matching their README.
  - Both tensor hashes match their manifest.
- **"0.016% of the weights" is right.** 131,072 weights were touched, which is 0.015–0.017% depending on whether you
  count the vision tower.
- **"Subtracted that direction" undersells the edit.**
  - The rank-1 fit of the diff gives α = 6.000 (10th–90th percentile 5.998–6.003), the same as their own report's
    W − 6·u(uᵀW).
  - So on those 64 neurons the component along u becomes −5× what it was: reversed and amplified, not deleted.
  - Their method is honest about this. The article's phrase "much like removing a behavior by deleting a single
    direction" just simplifies it.
- **Your replication supports the result.** p50c's MATH-500 gain holds on your hardware: 9.4% → 58.2% on 64 problems.
  Your three-Astra follow-up, starting from a frozen edited checkpoint, did not beat it: 46.1% against 46.9%, 48.4% and
  41.0%.

## New evidence: are the two edits the same mechanism?

I used only weight analysis and the hidden states I had already recorded in passes 2 and 3. No new runs.

1. **Their direction is real in my data.**
   - My 16 loops and 12 clean solutions were recorded independently of their work.
   - Their layer-15 direction separates loop tokens from clean text at d′ = 5.1. For 1,000 random directions the 99th
     percentile is 1.7, and none comes close. The layer-19 direction gives d′ = 4.7 (random 99th percentile 1.9).
   - It is also already raised *before* the loop starts. The average projection is 2.26 in the loop, 0.98 in the
     pre-loop text and 0.57 in clean text.
   - The two directions are similar to each other (cosine 0.85). They are also related to the direction their 10-pass
     run found independently (cosine 0.72 and 0.68).
2. **My edit barely touches it.** I compared the same texts read by the original model and by E1. E1 shrinks the
   loop-vs-clean gap along their direction by only 10% (layer 15 direction) and 14% (layer 19 direction).
3. **My heads are not the main writers along it.** Among all 48 attention heads, mine rank 8th, 10th and 11th (L15H7,
   L23H5, L15H6, at 1.3–1.4× a random direction), with L23H4 32nd and L19H0 46th. The top writer is L11H1 (2.0×).

My reading, which is inference rather than a test: the loop has at least two parts.
- A "repeating" state that builds up in the residual stream; their edit flips it.
- Copy heads that read the earlier text and write its next token back; my edit removes them.

Each edit works without undoing the other's target, so they may stack. Only a generation run can check that.

## What their run shows about mine

- I never ran a random control (for example, switching off three random heads). Their 50-pass run did, for its edit.
- My strength testing was thin, and E3 is probably not the best strength.
- Their confirmation was about 4.5 times larger.
- The rule (b) problem was already in my write-up. Comparing against Astra makes clear how much it helped.

## Possible next steps (none started; each needs your go-ahead)

1. **Do the two edits stack?** One paired run of E3 + p50c against p50c alone, on the same 64 held-out problems with
   4 samples each. Roughly 2.5 hours on a g5.xlarge, about $3–4, based on my earlier runs' speed.
2. **Official numbers for E3.** E3 on the full MATH-500 (500 × 4). Roughly 9–10 hours, about $12 (same basis).

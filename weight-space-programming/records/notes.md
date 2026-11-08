# notes.md: weight-space programming on Qwen3.5-0.8B

Status (2026-11-05 05:30 IST): DONE. All 10 passes used. Final edit = E3. Held-out (64 problems, n=4): original 10.2% -> E3 43.8%, paired +33.6 pts (SE 4.4). GPU instance terminated, zero left.

## Plan for the 10 passes (draft)
1. Diagnostic generation (original), GSM8K + MATH across levels, n=2, full 32k budget, with activations.
   Goal: failure taxonomy (wrong / no answer / cut-off / looping) and per-sample residual summaries.
2-3. Teacher-forced passes on the model's own failing vs good continuations: localize what drives
   the dominant failure (attention heads / MLP neurons / directions), with token-level signal.
4-8. Edit candidates, measured with paired "both" generations on fresh problems at several strengths.
9-10. Confirmation on problems never used to choose the edit.
Final: held-out 32 MATH-500 + 32 GSM8K-test, n=4, both models (outside the budget, aggregate only).

## Harness facts (smoke tests, 2026-11-04, not counted)
- Generation: vLLM 0.32.0 on 1x A10G; engine start about 3.5-4 min per model load (compile + warmup).
  FlashInfer sampler disabled (VLLM_USE_FLASHINFER_SAMPLER=0, no ninja/nvcc); PyTorch sampler with per-request seeds.
- Paired seeds work: original and edited share the seed per (problem, sample) and diverge only after the edit bites.
- vLLM casts the 18 linear_attn.norm.weight (F32 on disk) to bf16 when serving, so tiny edits there round away.
- Teacher-forced OK: chat template renders as expected; P('5' | ...boxed{) = 0.993 on '2+3'; attention rows sum to 1.
  Mean residual-token norm on a short chat: L0 0.70, L12 1.88, L24 10.7.
- transformers returns attention maps only for the 6 full-attention layers.

## Findings
### F1. Repetition loops at the 32k cap are the dominant failure (pass-1 partial, 152 answers, mostly GSM8K)
- 152 finished answers (148 GSM8K, 4 MATH) from attempt 1-a5, graded locally (cell 5).
- GSM8K: 33 correct, **115 cut off at 32,768 tokens (78%)**, 0 wrong, 0 no-answer. All 37 natural finishes are correct
  (median length 3,802 tokens).
- All 115 cut-offs are verbatim line-repetition loops (cell 7): the loop starts at a median of 14% of the answer
  (quartiles 0.106-0.191); the last 200 lines contain a median of 4 distinct lines; `</think>` is never closed.
  Typical loop line: "Wait, if he earns $14, he needs $0." repeated hundreds of times.
- In 81/115 loops the gold answer already appears in the first half of the thinking: the model had solved the
  problem, then got stuck re-checking.
- Caveats: GSM8K-heavy sample (request order put GSM8K first); MATH is still unmeasured. Per sample, not per problem.
### F2. Full pass 1 (job 1-a6; 512 answers = 128 GSM8K + 128 MATH train problems x n=2; cells 10-11)
- Accuracy 17.4% (GSM8K 19.1%, MATH 15.6%). Failure counts: cut_off 430, correct 81, wrong_answer 1.
- **82 natural finishes, 81 correct** (answer tokens quartiles 718/1784/4064/6856/28369). The model almost never
  answers wrongly: it either finishes (and is right) or never finishes.
- MATH by level (n, correct, cut): L1 52/21/33, L2 52/8/45, L3 52/8/45, L4 50/3/47, L5 50/0/50.
- 416/430 cut-offs end in an exactly periodic token tail (token_loop, reps=3): GSM8K onset quartiles
  1757/3930/5163/6805/19961, period 11/34/62/108/390; MATH onset 1701/4596/7039/9418/26005, period 8/23/41/72/314.
- 381/416 loop periods contain "Wait": the loop is a self-doubt re-check ("Wait, is there any other interpretation...")
  whose wording is then copied verbatim. Per problem: 194 problems had both samples cut, 42 one, 20 none.
- Implication: anything that lets the model leave the loop and close </think> turns cut-offs into (mostly correct) answers.
### F3. Pass 2 (TF, original, 28 exact-token texts: 16 loops, 12 clean finishes; cells 14-18): the copy circuit
- Local CPU rebuild of the 6 full-attention layers (wsp/recon.py, from transformers 5.22 modeling_qwen3_5) reproduces the
  recorded next residual to 0.5-1% relative error (bf16 level): per-head numbers are exact, not estimates.
- Mean token logprob: loop region -0.007, pre-loop reasoning -0.457, clean finishes -0.333.
- Confidence grows with every repetition k: -0.175 (k=0), -0.10, -0.047, -0.026, -0.016, -0.010, -0.007 (k=6), -0.002 (k=11).
  Self-reinforcing copying. </think> is never in the top-20 at loop period starts; in clean finishes the model closes
  </think> as a sampled choice (logprob -0.6 to -2.7 at that point).
- Copy attention (sum over all earlier repetitions of attention to the token that followed the current context),
  control = same offsets shifted by P/2 (all controls <= 1.3%):
  L15H7 0.735, L23H5 0.581, L11H1 0.498, L19H5 0.358, L23H0 0.348, L23H4 0.320, L23H7 0.310, L19H7 0.250, L15H6 0.226.
- Direct logit attribution on the realised next token (loop / pre / natural): L15H7 1.75/0.45/0.25, L15H6 0.93/0.17/0.14,
  L19H0 0.53/0.15/0.07, L23H5 4.00/1.77/1.74, L23H0 1.16/0.45/0.34. Layer-level: the loop's extra logit comes from
  full-attention layers 15 (+2.0 vs pre), 19 (+1.1), 23 (+7.9); DeltaNet layers 12-14, 16-18 contribute LESS in loops.
- First-order removal (subtract head output from final residual, recompute softmax; ignores downstream heads):
  trio L15H7+L15H6+L19H0 changes logprob of the actual token by -0.10 (loop), -0.17 (pre), -0.06 (natural).
  The established loop is over-determined; an edit has to act through the copy circuit's composition / early repetitions.
### Harness fidelity checks
- vLLM keeps the GDN recurrent state in float32 for Qwen3.5 (config mamba_ssm_dtype; vllm v0.32.0
  model_executor/models/config.py:855-877), so the loops are not a low-precision-state artifact (source-verified).
- Remaining doubt: prefix caching on a hybrid model gave 0% hit rate (no benefit); disable it to remove a variable.
### Crash of attempt 1-a5 (resolved)
- Cause: SSM AWS-RunShellScript executionTimeout default 3600 s killed the nohup'd worker tree at exactly 1 h. Fix: worker runs as a systemd-run unit (wsp-job-<job>); command timeout 600 s.
- Silence at 20:54:44 UTC, shard log + orchestrator both stopped, no Python error, serial console has no OOM line
  (only the last 64 KB). vLLM at 100% KV cache, preempting (Running 128 -> 91). Root cause NOT yet established:
  needs `journalctl -b -1` on the instance disk.

### F4. E1's remaining cut-offs (pass 4 tokens, cell 24)
- E1 cut 131: 109 exact loops, onset quartiles 10530/15995/23378 (original on same problems: 4320/5825/8524),
  period 21/55/99 (orig 27/48/94). Same kind of "Wait, ..." re-check loops, entered later.
- 76/109 E1 loops have the gold answer string before onset (still answers lost to looping).
- 22 cut-offs without an exact loop: ~half near-loops (unique-line share < 0.1 in last 4k tokens), ~half long non-repetitive.
- Next: pass 6 = TF on E1 over 20 of these loops + 8 E1 clean finishes (work/req_pass6.json, 3.54 GB, hidden 11/15/19/23/24),
  to find which remaining heads carry the loops. Runs after pass 5 (confirmation).

### F5. Pass 6 (TF on E1, 20 of its own loops + 8 clean finishes, up to 12,155 tokens; cells 29-30)
- Rebuild with edited weights reproduces layer 24 to 0.55% median relative error.
- E1 loop logprob -0.0124 (pre -0.339, natural -0.462); by repetition k: -0.104 (k=0) -> -0.017 (k=11): still self-reinforcing, slower.
- L15H7 still ATTENDS to the earlier copy (copy-all 0.597) but writes nothing (o_proj columns zero): E1 cuts the write, not the read.
- Copy-all on E1 loops (orig pass-2 values in brackets, different texts): L23H5 0.49 (0.58), L23H4 0.29 (0.32), L23H0 0.26 (0.35),
  L23H7 0.26 (0.31), L19H5 0.28 (0.36), L11H1 0.38 (0.50), L15H5 0.18 (0.23). Controls <= 2%.
- DLA loop/natural: L23H5 5.13/1.75, L23H4 2.77/1.04, L23H6 4.29/2.85 (no copy attention), L19H5 0.69/0.01, L15H5 0.37/0.06.
- First-order removal (loop / pre / natural): L23H5 -0.457/-0.216/-0.091 (ratio 5.0), L23H4 -0.247/-0.124/-0.055 (4.5),
  L23H0 -0.089/-0.070/-0.060 (1.5), L19H5 -0.031/-0.112/-0.108 (0.29), L15H5 -0.006/-0.022/-0.017, L23H6 -0.028/-0.120/-0.118.
  So DLA alone misleads: L19H5 and L15H5 look loop-specific by DLA but removing them harms normal text as much.
- Choice: E2 = E1 + L23H5 + L23H4. Risk: natural harm (-0.09, -0.05) is larger than E1's trio had (-0.06 together).

## Edits tried
- E1: zero o_proj columns of L15H7 (1792-2047), L15H6 (1536-1791), L19H0 (0-255): 786,432 weights set to 0.
  * Pass 3 (TF, same 28 texts as pass 2): loop-token logprob -0.0073 -> -0.0468; pre-loop -0.458 -> -0.506;
    clean finishes -0.333 -> -0.350. By repetition k the E1 loop plateaus near -0.045 instead of -> -0.002.
  * Pass 4 (generate both, 128 FRESH problems = 64 GSM8K + 64 MATH L1-4 x16, n=2; E1 fixed before this pass):
    original 12.9% (GSM8K 15.6, MATH 10.2), cut 88.7%; E1 44.9% (GSM8K 50.8, MATH 39.1), cut 51.2%.
    Paired diff +32.0 pts, SE 3.75; wrong->right 92, right->wrong 10 (9 of the 10 are E1 cut-offs, 1 wrong answer).
    MATH by level orig->E1: L1 6->14, L2 1->14, L3 3->13, L4 3->9 (of 32 each).
    Finished answers: original 29 (100% correct), E1 125 (92% correct; 9 wrong, 1 no answer).
    E1 thinks LONGER, not shorter: thinking tokens before </think> among correct, median 3973 (orig) vs 10016 (E1);
    </think> closed in all finished answers. Not a skip-thinking effect.
    Remaining E1 cut-offs: 109/131 still end in exact loops, but later (onset quartiles 10530/15995/23378 vs ~4-7k orig).
  * Pass 5 = CONFIRMATION (generate both, 128 problems disjoint from passes 1 and 4: 64 GSM8K + 64 MATH L1-4, n=2; cells 26-28):
    original 10.9% (GSM8K 11.7, MATH 10.2), cut 91.0%; E1 45.7% (GSM8K 44.5, MATH 46.9), cut 49.2%.
    Paired per problem +34.8 pts, SE 3.6; wrong->right 93, right->wrong 4 (3 E1 cut-offs, 1 wrong answer).
    MATH by level orig->E1: L1 6->21, L2 5->11, L3 0->12, L4 2->16 (of 32 each).
    Finished: original 23 (100% correct); E1 130 (88.5% correct; 15 wrong, 13 of them GSM8K).
    Of the 15 wrong: 14 are real numeric errors; 1 is a grader false negative ('A5' vs gold '\text{A5}_{11}', base-11
    notation). Grader left unchanged (it treats both models the same). Full dump: work/pass5_wrong_answers.txt.
    Thinking before </think> among correct: median 3516 (orig) vs 10045 (E1). E1 cut-offs: 109/126 exact loops, onset q 12119/17750/24243.

- E2: E1 + zero o_proj columns of L23H5 (1280-1535), L23H4 (1024-1279). Pass 7: generate E2 on the 256 problems of passes 4+5, n=2,
  same seeds -> paired with E1 and original (cells 32-33).
  * E2 44.0% vs E1 45.3% vs original 11.9% (same 512 seeds; seeds verified identical 512/512).
    Paired per problem E2-E1 -1.4 pts (SE 2.4); wrong->right 66, right->wrong 73. Pass-4 half +4.7 (SE 3.4), pass-5 half -7.4 (SE 3.3).
    E2 breaks loops (cut 38% vs 50%; exact-loop cut-offs 72 vs ~218) but: wrong answers 67 (E1 24), no_answer 23 (E1 1).
    All 23 no_answer closed </think> and gave a final solution ending "Answer: X" with NO \boxed (19/23 numerically right;
    still failures under the protocol). Inference (not measured): L23H5/H4 also carry the prompt's \boxed instruction forward.
    Verdict: E2 rejected.
- E3: E1 + L23H5, L23H4 o_proj columns x0.5 (half strength). Pass 8: same 256 problems, n=2, same seeds (cell 35).
  * E3 51.0% vs E1 45.3% vs E2 44.0% vs original 11.9%. Paired E3-E1 +5.7 pts (SE 2.1); wrong->right 67, right->wrong 38.
    Halves: pass-4 problems +7.4 (SE 2.9), pass-5 problems +3.9 (SE 3.0). E3-orig +39.1 (SE 2.6), 205 vs 5 flips.
    no_answer 2 (formatting intact), wrong answers 42 (E1 24): accuracy among finished 85.6% (E1 90.2%).
    Exact-loop cut-offs 122 (E1 218). MATH of 64 per level orig/E1/E3: L1 12/35/48, L2 6/25/26, L3 3/25/27, L4 5/25/25.
    Caveat: E3 was selected over E2 on these problems. Pass 9 (E3) + pass 10 (E1) on 256 fresh problems = clean paired test.
  * FRESH test (passes 9 + 10; 256 problems never used before; 128 GSM8K + 128 MATH L1-4; same seeds; cell 38):
    E3 55.1% (GSM8K 54.7, MATH 55.5; wrong 46, no_answer 1, cut 35.9%) vs E1 47.7% (GSM8K 45.7, MATH 49.6; wrong 24, cut 47.3%).
    Paired E3-E1 +7.4 pts (SE 2.1); wrong->right 77, right->wrong 39; GSM8K +9.0 (SE 3.1), MATH +5.9 (SE 2.8).
    MATH of 64 per level E1/E3: L1 38/48, L2 42/44, L3 29/29, L4 18/21.
    Decision rule (written in journal before passes 9-10 were seen): E3 if > 0 -> E3.

## Current best edit
E3 = zero o_proj columns of L15H6, L15H7, L19H0 + halve those of L23H4, L23H5 (3 tensors; 786,432 weights zeroed, 524,288 halved).
vs E1: +5.7 +- 2.1 (selection problems), +7.4 +- 2.1 (fresh problems). E1 vs original: +32.0 +- 3.75, +34.8 +- 3.6.
Bundle: work/edits_archive/E3 (= work/edited).

## Held-out result (job heldout-a16, 147 min, done 05:21 IST; aggregate only, problem texts never left the worker)
64 held-out problems (32 MATH-500 + 32 GSM8K test), n=4, 256 generations per model, same seeds for both.
| | original | E3 |
|---|---|---|
| accuracy | 10.2% (26/256) | 43.8% (112/256) |
| GSM8K test | 17.2% | 51.6% |
| MATH-500 | 3.1% | 35.9% |
| hit 32k cap | 91.0% (233) | 47.3% (121) |
| cut while looping | 115 | 41 |
| wrong answer | 0 | 22 |
| mean tokens | 30,270 | 24,101 |
Paired E3 - original: +33.6 pts (SE 4.4); wrong->right 92, right->wrong 6.
(Original's 23 'correct' in failure_counts vs 0.1016*256=26: failure_counts seems to count per final state; accuracy is the
headline number. Not checked further: per-answer results were deliberately not returned.)

## Plain-English summary
The untouched model almost never finishes: on 91% of held-out attempts it was still "thinking" after 32,768 tokens,
usually going round in a loop. We found, by reading its internal activity during those loops, a few attention heads in
layers 15, 19 and 23 that keep copying earlier text back into the stream. Switching three of them off and turning two down
(editing 3 weight matrices by hand: no training) roughly halves the stuck runs and lifts held-out accuracy from 10% to 44%.
Cost of the edit: it now gives some wrong answers (22/256) where the original gave none, because it finishes more often;
nearly half of its attempts still hit the token cap, so loops are reduced, not cured.

## Post-hoc comparison with the published study (2026-11-07, after E3 was frozen)
Full write-up: work/comparison_with_article.md (evidence in work/comparison/).
- Task message identical to the authors' p10c prompt (word diff). Same diagnosis (loops), same copy heads (L15H7 top in both), same layers (15, 19).
- Authors' runs: p10c (10 passes) 30 rank-1 direction edits, +3.7 +-1.8; p25c (25) zero L15H7 only, +8.3 +-1.4; p50c (50) W-6u(u^T W) on 64 MLP neurons in L15/L19, +49.8 +-1.5.
- Same 64 held-out problems (authors' per-problem files): original 3.1/18.8 (mine 3.1/17.2), p10c 7.0/23.4, p25c 5.5/31.3, p50c 42.2/68.8, my E3 35.9/51.6 (MATH-500/GSM8K).
- Their checkpoint verified vs base: only L15/L19 down_proj changed, 130,188 values, hashes match manifest; alpha = 6.000 (reversal x5 on 64 neurons).
- Their direction separates loops from clean text in MY recorded data (d' 5.1 / 4.7 vs random 99th pct 1.7 / 1.9); E1 shrinks that gap only 10-14% -> likely complementary mechanisms (inference; untested).
- Caveat: my head selection used local recomputation of attention (rule b), which Astra did not use; the 10-pass comparison is not a fair win.
- Correction (2026-11-07): held-out per-answer files were NOT unavailable. The worker's 20 s partial backup copied the raw shards to S3 (jobs/heldout-a16/partial/). Astra's audit recovered and regraded all 512: aggregates match exactly; the 26 vs 23 gap = 3 original answers correct-but-cut. They never reached this laptop. Also: pass 1 had 89 graded correct = 81 finished-correct + 8 correct-but-cut (the "81 correct" in earlier summaries means finished-correct).
- Blog post (user's voice): work/blog_post.md; page https://claude.ai/artifact/Bq4D6qgFN3ybrUyqgWJa6n

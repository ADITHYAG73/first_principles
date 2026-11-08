# Experiment journal: weight-space-programming

## 2026-11-04: Preflight (rule 7), written BEFORE the machine exists

**Update 00:57:** g5.12xlarge launch failed with InsufficientInstanceCapacity, so nothing launched. The
user chose **g5.xlarge** (1x A10G 24 GB, 4 vCPU, 16 GB RAM, **$1.208/hr**, AWS Pricing API). On one
GPU, "both" passes run the two models one after the other. Host RAM peak: edited-model build
loads the 1.75 GB checkpoint, vLLM holds about 2-4 GB host-side and the records about 0.5 GB, so roughly
6 GB of 16 GB. Teacher-forced merge uses memmaps. New estimate (my inference): about 1 h setup, 10 passes
x 0.5-1.5 h, plus 1-2 h held-out, so roughly 10-20 instance-hours, about $12-25.
The text below was written for g5.12xlarge; the four answers are unchanged except where noted above.

Machine: 1x g5.12xlarge (4x A10G 24 GB, 48 vCPU, 192 GB RAM), ap-south-1, on-demand, **$6.81091/hr**
(AWS Pricing API, 2026-11-04). Name `weightspace-worker`, tag `Project=weight-space-programming`.
Spot G quota is 0, on-demand G quota is 48 vCPU (Service Quotas API + AWS support email of 2026-11-02).
Opening balance: Cost Explorer month-to-date (2026-11-01..03) UnblendedCost = $0.00 (-0.0000003).
Rough cost estimate (my inference, not measured): setup ~0.5 h + 10 passes x 0.25-0.5 h + held-out eval
~0.5 h, roughly 4-7 instance-hours, about $30-45. EBS 200 GB gp3 is extra while it exists.

Code paths that will run: `remote/setup.sh` (user-data), `remote/worker.py` (orchestrator plus per-GPU
shards: vLLM 0.32.0 generation, HF transformers teacher-forced/activations), `remote/prepare_heldout.py`.

1. **Timestamped progress line per unit of work?** Yes. The worker logs job start/end and, every 30 s,
   `[gen] done/total, elapsed, ETA`. The log is synced to S3 every 20 s and printed by `wsp/forward.py`
   while it waits. Setup logs to /var/log/wsp-setup.log.
2. **Results written incrementally?** Generation: each finished sample is appended and flushed to the
   per-GPU shard jsonl as soon as it finishes. Teacher-forced: one .npz per text. Activations: one .npz
   per GPU shard at shard end (a few minutes of work). Final merge and S3 upload happen at job end.
3. **Does working memory grow, and what is the peak vs the card?** vLLM preallocates the KV cache
   (gpu_memory_utilization 0.90 of 24 GB) and stays flat. Host: the record list grows with outputs,
   at most 512 x 32k tokens of text plus ids, about 0.5 GB, against 192 GB RAM. Teacher-forced worst
   case on the GPU: hooks hold all requested layers for ONE text, at most 24 MLP layers x 32k x 3584 x 2 B
   (5.5 GB) + 25 x 32k x 1024 x 2 B (1.6 GB) + 1.6 GB weights + 1 GB logits chunk, about 10 GB of 24 GB.
   The output merge uses on-disk memmaps (at most 6 GB per pass, enforced before running).
4. **If killed at 80%, what survives?** All finished generations sit in the shard jsonl files on the
   instance's EBS volume (they survive a stop, not a terminate). The pass is not marked done, so it
   does NOT count against the 10-pass budget; it costs re-run dollars only. The edits live locally in
   work/edited/ and are never lost by a remote failure.

Watchdog: on-box systemd timer every 2 min. It stops the instance after 15 idle minutes (no worker.py,
no setup running) or 8 h of uptime, and is installed first in user-data. A CloudWatch alarm is the
second line: if CPU stays under 1% for 30 min, it triggers the EC2 stop action. `wsp/forward.py` also
stops the instance right after each job. End of task: terminate, then confirm zero
`weightspace-worker` instances.

## Pass 2 preflight (2026-11-04 10:25 IST) — teacher-forced, original weights
Request: work/req_pass2.json (28 exact token-id texts: 16 loop windows = prompt + answer up to onset+1100; 12 clean correct finishes; max 4094 tok, total 84,011), return token_logprobs, topk 20, hidden_states 0..24. Meta: work/req_pass2_meta.json.
1. Progress lines: worker logs per shard start/end + Progress heartbeat every 20 s (log synced to jobs/2-*/partial/).
2. Incremental results: one npz per text in tf_parts/ (synced as partials); merged to memmaps at the end.
3. Memory: GPU holds 25 residual captures of one 4096-token text (25*4096*1024*2 B = 210 MB) + logits chunks of 1024x248320 fp32 (~1 GB); host merge uses np.memmap (no in-RAM growth), output ~5.9 GB on a 200 GB disk. 16 GB host RAM is not at risk.
4. Killed at 80%: per-text npz files already written survive in tf_parts and in S3 partials; failed request does not count.
Watchdog: systemd 900 s idle / 8 h cap + CloudWatch CPU alarm, unchanged.

## Pass 3 preflight (2026-11-04 ~11:00 IST) — teacher-forced, EDITED (E1: zero heads L15H7, L15H6, L19H0)
Same 28 texts as pass 2 (exact ids); return logprobs, topk 20, hidden 15/19/23/24 (~0.96 GB). Same code path as pass 2
(which ran cleanly in ~20 min): per-text npz incremental, memmap merge, watchdogs unchanged. Edit bundle merged on worker
(build_model_dir, cast to original dtype).

## Pass 4 preflight (2026-11-04 11:05 IST) — generate "both" (original vs E1), 128 fresh problems x n=2 = 256 per model
Same code path as pass 1 (job 1-a6, ran 125 min cleanly under systemd-run): models run one after the other on the A10G.
1. Progress: per-shard heartbeat lines (done/running/tokens/s, KV use, host MemAvailable, GPU mem) every 20 s, log synced.
2. Incremental: each finished answer appended to gen_<model>_gpu0.jsonl (FINAL_ONLY), synced to S3 partial/ every 20 s.
3. Memory: vLLM preallocates KV (gpu_memory_utilization fixed); host RAM flat in pass 1 (MemAvailable logged). No growth path.
4. Killed at 80%: finished answers survive in S3 partials; resume_from reuses them (same rid + seed), as done for 1-a6.
Expected wall time ~3.5-4 h (original side mostly 32k cut-offs). Under the 8 h watchdog cap. Launcher now retries
InsufficientInstanceCapacity for up to 60 min.

## Pass 5 preflight (2026-11-04 16:14 IST; launch time from the client log) — CONFIRMATION of E1: generate "both", 128 fresh problems x n=2
Problems disjoint from passes 1 and 4 (work/used_problems.json updated). Identical code path and request shape to
pass 4 (job 4-a9: 160 min, no incidents, host memory flat at ~10.9 GB available). Same four preflight answers as pass 4.
Launcher retries capacity errors up to 60 min. Watchdogs unchanged.

## Pass 5 result (2026-11-04 18:41 IST; job 5-a10, 145.9 min, instance stopped after)
Original 10.9% (GSM8K 11.7, MATH 10.2), cut 91.0%; E1 45.7% (GSM8K 44.5, MATH 46.9), cut 49.2%.
Paired per problem (128): +34.8 pts, SE 3.6; wrong->right 93, right->wrong 4. Problems disjoint from pass 4 (checked: overlap 0).
E1 confirmed. Analysis in cells 26-28.

## Pass 6 preflight (2026-11-04 18:44 IST; launch time from the client log) — teacher-forced, EDITED (E1), on E1's own remaining loops
Request: work/req_pass6.json (28 exact token-id texts: 20 E1 loop windows from pass 4 + 8 E1 clean finishes; max 12,155,
total 231,987 tokens), return token_logprobs, topk 20, hidden 11/15/19/23/24. Meta: work/req_pass6_meta.json.
Same code path as passes 2 and 3 (tf_shard: per-text npz, memmap merge; pass 3 ran in 1.6 min).
1. Progress: worker start/shard lines + 20 s heartbeat, log synced to S3.
2. Incremental: one npz per text in tf_parts/, synced as partials.
3. Memory: NEW longest text, 12,155 tokens (3x pass 2/3). Captures 6 x 12,155 x 1024 x 2 B = 150 MB; logits chunk 1024 x 248,320
   fp32 = 1 GB (+1 GB log_softmax). Attention uses the transformers default (sdpa), so no T^2 maps; GDN layers have no
   fla/causal-conv1d kernels installed, so the transformers torch fallback runs. I have NOT re-read that fallback in this session;
   my expectation that it is linear in T is inference. Padded output 28 x 12,155 x 5 x 1024 x 2 B = 3.5 GB (< 6 GB limit); local disk 47 GB free.
4. Killed at 80%: per-text npz survive in S3 partials; a failed pass does not count against the budget.
Watchdog: systemd 900 s idle / 8 h cap + CloudWatch CPU alarm, unchanged.

## Pass 6 result (job 6-a11, 4.8 min, 18:44-18:59 IST)
OK; 3.5 GB downloaded. Analysis cells 29-30: L23H5 and L23H4 carry the remaining loop-specific push (first-order loop harm 5.0x / 4.5x natural).

## Pass 7 preflight (written just before launch; launch time = first line of work/pass7_run.log) — generate EDITED only, E2
E2 = E1 + zero o_proj columns of L23H5, L23H4 (3 edited tensors). 256 problems = union of pass-4 and pass-5 problems, n=2 = 512
generations, seeds item_seed(problem, sample) identical to passes 4/5, so each E2 sample pairs with the E1 (and original) sample.
Same code path as pass 1 (job 1-a6: single model, 512 generations, 125 min, systemd unit, host memory flat).
1-4 as pass 4: per-shard 20 s heartbeat; each answer appended to jsonl and synced to S3 every 20 s; vLLM KV preallocated, host RAM
flat in pass 1; killed at 80% -> finished answers survive in S3 partials and resume_from reuses them; failed pass not counted.
Watchdogs unchanged (systemd 900 s idle / 8 h cap; CloudWatch CPU alarm). Expected 2-2.5 h.

## Pass 7 result (job 7-a12, 115.3 min, done 21:03 IST, instance stopped)
E2 44.0% vs E1 45.3% on the same 512 seeds: paired E2-E1 -1.4 pts (SE 2.4), wrong->right 66, right->wrong 73. Not better.
E2 cut 38% (E1 50%), but wrong 67 (E1 24) and no_answer 23 (E1 1): all 23 closed </think> and wrote "Answer: X" without \boxed
(19 numerically right; scored as failures per protocol). Cells 32-33; dump work/pass7_no_answer.txt.

## Pass 8 preflight (written just before launch; launch time = first line of work/pass8_run.log) — generate EDITED only, E3
E3 = E1 + L23H5, L23H4 o_proj columns x0.5. Request identical to pass 7 (work/req_pass8.json = req_pass7 problems, n=2, same seeds).
Code path, memory, checkpointing and watchdogs identical to pass 7 (ran 115 min, host memory flat at 10.9 GB free). Expected ~2 h.

## Pass 8 result (job 8-a13, 121 min incl. 9 min of capacity retries; done 23:08 IST, instance stopped)
E3 51.0% vs E1 45.3% (same 512 seeds): paired E3-E1 +5.7 pts (SE 2.1), wrong->right 67, right->wrong 38; both halves positive
(pass-4 problems +7.4 SE 2.9; pass-5 problems +3.9 SE 3.0). no_answer 2 (E2 had 23). Wrong answers 42 (E1 24). Cell 35.
Selection caveat: E3 was picked over E2 on these same problems -> needs fresh-problem confirmation.

## Pass 9 preflight (written just before launch; launch time = first line of work/pass9_run.log) — generate EDITED (E3), fresh problems
256 fresh problems (not in passes 1/4/5/7/8; 128 GSM8K train + 128 MATH train L1-4 x32; random.Random(9)), n=2 = 512 generations.
Plan: pass 10 runs E1 on the same problems + seeds -> paired E3 vs E1 on problems never used for any choice.
Code path, memory, checkpointing and watchdogs identical to passes 7/8 (115 / 107 min of generation; host memory flat).

## Decision rule for the final edit (written 2026-11-05 00:45 IST, BEFORE pass 9 or 10 results were seen)
Pass 9 = E3 and pass 10 = E1 on the same 256 fresh problems and seeds. Final edit = E3 if the paired per-problem
difference E3 - E1 on these fresh problems is > 0; otherwise E1. Then re-save the chosen edit with save_edit, verify the
edited tensors, and run `wsp/forward.py --final-heldout` (64 held-out problems, original vs chosen edit, n=4).
Teardown: terminate the g5 instance (wsp/infra.py down) after results are local and in S3; confirm zero instances.
User asleep from ~00:45 IST; this runs unattended (caffeinate on; on-box watchdog 900 s idle / 8 h; CloudWatch alarm).

## Pass 9 result (job 9-a14, 102.8 min, done 00:54 IST, instance stopped)
E3 on 256 fresh problems: 55.1% (GSM8K 54.7, MATH 55.5), cut 35.9%, wrong 46, no_answer 1. Not interpreted until pass 10.
E3 bundle archived to work/edits_archive/E3 (sha256 in cell log); saved edit reverted to exact E1 (cell 37, verified).

## Pass 10 preflight (written just before launch) — generate EDITED (E1), same problems + seeds as pass 9
Identical request to pass 9 except the edit bundle (E1: 2 tensors). Code path/memory/checkpointing/watchdogs as passes 7-9.

## Pass 10 result + DECISION (2026-11-05 ~02:50 IST)
Pass 10 (job 10-a15, 112.6 min): E1 on the pass-9 fresh problems 47.7% (wrong 24, cut 47.3%). E3 (pass 9) 55.1%.
Fresh paired E3 - E1 +7.4 pts (SE 2.1), wrong->right 77, right->wrong 39; GSM8K +9.0 (SE 3.1), MATH +5.9 (SE 2.8). Cell 38.
Pre-registered rule -> E3. E3 restored from work/edits_archive/E3, sha256 identical to the bundle of passes 8-9 (cell 40).

## Held-out preflight (written just before launch; launch time = first line of work/heldout_run.log)
`wsp/forward.py --final-heldout`: generate both (original vs E3), 64 held-out problems (32 MATH-500 + 32 GSM8K test, seeded
selection made on the worker by remote/prepare_heldout.py), n=4 -> 256 per model. Outside the 10-pass budget.
Read code path: worker.build_items appends heldout selection.jsonl; for heldout the per-answer results dir is deleted on the
worker before upload, so only aggregate numbers return (no problem texts reach this machine). Same generate code path as
pass 5 (both models, 256 each, 146 min, memory flat). Checkpointing/watchdogs unchanged. Expected ~2.5 h.

## Held-out result + teardown (2026-11-05 05:30 IST)
Job heldout-a16 done in 147.1 min (worker) / 151.3 min (client). Aggregate only (work/heldout_result.json).
original 10.16% (GSM8K 17.19, MATH-500 3.12; cut 91.0%) vs E3 43.75% (GSM8K 51.56, MATH-500 35.94; cut 47.3%, wrong 22).
Paired +33.59 pts (SE 4.39), wrong->right 92, right->wrong 6.
Teardown: wsp/infra.py down 05:26:21 terminated i-0f41ea7d4e01f90d4; DescribeInstances (ap-south-1) shows it terminated,
0 other weightspace-worker instances, 0 CloudWatch alarms. Other account instances (agilabs-web, tark-app, paperbuddy-neo4j)
left as found.

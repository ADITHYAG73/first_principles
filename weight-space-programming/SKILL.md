---
name: replicate-weight-space-programming
description: Replicate Experiment 8 of "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026) with a coding agent such as Claude Code or Codex. Two paths. (A) Verify an edited Qwen3.5-0.8B checkpoint on MATH-500 / GSM8K under the paper's exact protocol. (B) Re-run the blind experiment, where an agent edits the model's weights by hand with 10 forward passes. Use when asked to reproduce, verify, or extend weight-space programming on Qwen3.5-0.8B.
---

# Replicating weight-space programming on Qwen3.5-0.8B

The original work is the authors'. Read it first:

- **The post:** https://rikiyat.github.io/blog/weight-space-programming/ (Experiment 8 is the Qwen experiment).
- **Their logs:** https://github.com/RikiyaT/lm_understand-logs, in `exp8_math/`: every turn of GPT-6 Astra's three runs,
  the edits and the evaluations.
- **Their checkpoint:** https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- **Opus's checkpoint (this folder):** https://huggingface.co/TokenGod/Qwen3.5-0.8B-opus-blind-weight-edit

This folder adds one more data point: Claude Opus 5.5 given only the task prompt, blind (see `index.html`).
The authors did not release their agent harness. This folder contains the harness Opus built for itself, in `code/`.

## Path A: verify an edited checkpoint (about 1–3 GPU hours)

Use this to check a published number, whether the authors' or Opus's.

1. **The protocol must match the paper exactly.**
   - Model: Qwen3.5-0.8B, the text part, with thinking on (`enable_thinking=True`).
   - User message: the problem, then `"\nPlease reason step by step, and put your final answer within \boxed{}."`
   - Sampling: `temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768`.
   - Grading: the last `\boxed{}` answer, checked with `math-verify`.
   - Samples per problem: 4 for MATH-500, 2 for GSM8K. Report mean accuracy over all samples.
2. **Prefer the authors' evaluator.** It is `exp8_math/code/evaluate.py` with protocol `think35` from
   `exp8_math/code/profiles.py` in their logs repo. Their per-problem results for every model are in `exp8_math/eval/`,
   so you can compare problem by problem.
3. **Hardware.** One 24 GB GPU (an NVIDIA A10G, for example AWS `g5.xlarge`) runs vLLM for this model. Most original-model answers run to the
   32,768-token limit, so budget about 70 minutes per 256 answers.
4. **Expected numbers** on the full sets, from the authors' evaluation:

   | Model | MATH-500 | GSM8K |
   |---|---|---|
   | Original | 7.9% | 13.8% |
   | Astra's 50-pass edit | 48.1% | 67.1% |

   Opus's edit has only been measured on a 64-problem held-out subset. There it scored 35.9% on MATH-500 and 51.6% on GSM8K, against 3.1% and 18.8% for the original on the same problems.

## Path B: re-run the blind experiment (about one day, roughly 20 GPU hours)

1. **The task.** Give your agent `task/weight-space-programming.md`. It is the exact prompt the authors gave Astra, word for
   word, and nothing else.
2. **Keep it blind.** Do not let the agent open the post, the logs, either checkpoint, or this article. Block web search for
   them. If you also replicate with a second agent, keep each one's folders separate.
3. **The harness** (if you are not building your own). Copy `code/wsp/` and `code/remote/` into an empty project root
   containing `data/` and `work/`.
   - **Python 3.11+ locally:** `boto3`, `torch` (CPU), `safetensors`, `tokenizers`, `numpy`, `pandas`, `huggingface_hub`,
     `jupyter_client`, `ipykernel`.
   - **Configure `wsp/common.py`:** set `REGION`, `BUCKET` (your own S3 bucket) and `MODEL_DIR` (your local
     `Qwen/Qwen3.5-0.8B` snapshot at revision `2fc06364715b967f1860aea9cf38778875588b17`).
   - **Commands:**
     - `python wsp/prepare_practice.py` downloads the practice sets: GSM8K train, MATH train and AIME. These are not the test sets.
     - `python wsp/infra.py up` creates the worker: `g5.xlarge`, an idle watchdog and a CloudWatch stop alarm.
     - `python wsp/kx.py start`, then `python wsp/kx.py run < cell.py` runs the analysis kernel. It allows 300 runs and logs every run.
     - `python wsp/forward.py request.json` runs one forward pass. The budget of 10 is enforced, and failed requests don't count.
     - `python wsp/forward.py --final-heldout` runs the 64-problem held-out check after the edit is frozen.
     - `python wsp/infra.py down` terminates the worker. Confirm that zero instances remain.
4. **Rule (b) is the easy one to break.** The prompt forbids implementing or simulating the network's computation on any
   input in the kernel. Opus broke it with `code/wsp/recon.py`, which recomputes the attention layers from recorded
   activations. Delete that file before starting, tell the agent explicitly that recomputing any layer is out of
   bounds, and audit `work/cells/` afterwards.
5. **Rented-GPU practice.** Write a preflight before every billed run:
   - Is there a progress line per unit of work?
   - Are results saved incrementally?
   - What is peak memory against the card?
   - What survives a kill?

   Keep the watchdog running. Terminate the instance at the end.
6. **Held-out hygiene.** `code/remote/worker.py` was patched after Astra's audit. In the original run, held-out answer
   shards were backed up to S3 mid-run even though only aggregate numbers were meant to leave the machine. Held-out jobs now skip that backup.

## Reporting

Report against the authors' numbers, on the same problems where possible. Their per-problem files make that exact. State:
- the number of forward passes used;
- any rule exceptions;
- the paired gain with its standard error, over problems rather than samples.

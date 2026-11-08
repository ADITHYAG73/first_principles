# Opus E3 mechanism and tensor audit

Checked 7 November 2026 IST. Read-only inspection of the Claude project; NumPy comparisons only, no model/layer execution, GPU rental or cloud calls by this reviewer.

## Exact weights and job submission

The local base is `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`, as specified by [the harness](/Users/adithyagiridharan/Desktop/PythonProjects/claude/wsp/common.py:8). The full cached base file hashes to `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`, matching its content-addressed cache blob name. This audit did not independently refetch the original Hub file.

All three current saved overlays are byte-identical to their archived E3 files. Their complete tensor contents exactly equal the pinned local base after these operations; every entry outside the specified columns is unchanged:

| Matrix | Columns, inclusive | Operation | Changed values |
|---|---|---|---:|
| Layer 15 attention `o_proj` | 1536–2047, heads 6/7 | Zero | 524,288 |
| Layer 19 attention `o_proj` | 0–255, head 0 | Zero | 262,144 |
| Layer 23 attention `o_proj` | 1024–1535, heads 4/5 | Multiply by 0.5 | 524,288 |

There are exactly three overlay files and **1,310,720 changed values**. Construction code: [E1](/Users/adithyagiridharan/Desktop/PythonProjects/claude/work/cells/0019.py:1), [E3](/Users/adithyagiridharan/Desktop/PythonProjects/claude/work/cells/0034.py:1).

The original tensors are BF16; the saved overlays are F32. All overlay values are exactly BF16-representable. The recovered held-out job bundle `recovered-heldout/edits.safetensors` has SHA256 `5196674a5229e545afc677529a357a1566869c151558857bec45250192f54334`; it contains exactly these three tensors, whose F32 bytes are identical to the corresponding current/archive overlays. The worker [downloads this job bundle and casts edits to the original tensor dtype](/Users/adithyagiridharan/Desktop/PythonProjects/claude/remote/worker.py:154), then [runs vLLM in BF16](/Users/adithyagiridharan/Desktop/PythonProjects/claude/remote/worker.py:246).

This connects the saved E3 to the submitted held-out job input. No independent hash of the fully materialized checkpoint on the GPU was available in this audit, so it is not a separate runtime attestation. Complete machine-readable receipts: [mechanism-tensor-audit.json](mechanism-tensor-audit.json). Reproducible local verifier: [verify-opus-edits.py](verify-opus-edits.py).

## Protocol deviation

The written task [forbids implementing or simulating network computation on any input in the analysis kernel](/Users/adithyagiridharan/Desktop/PythonProjects/claude/weight-space-programming.md:29). The saved analysis does this:

- [Query/key/value and gate computation](/Users/adithyagiridharan/Desktop/PythonProjects/claude/wsp/recon.py:48), including RMSNorm and rotary positions.
- [Causal attention softmax, value aggregation, gating and output projection](/Users/adithyagiridharan/Desktop/PythonProjects/claude/wsp/recon.py:60).
- [MLP and whole-layer recomputation for fidelity checks](/Users/adithyagiridharan/Desktop/PythonProjects/claude/wsp/recon.py:83).
- [Final normalization, unembedding and full log-softmax on edited residuals](/Users/adithyagiridharan/Desktop/PythonProjects/claude/work/cells/0017.py:6).

These use recorded residual inputs rather than new tokens, but remain network computations outside the forward-pass tool. On the explicit written restriction, this is a protocol deviation, not merely a speculative possibility. The result can still be useful as an exploratory weight-editing experiment; it cannot be described as a compliant ten-pass replication of that restriction. This finding does not, by itself, invalidate actual GPU generation comparisons.

## Mechanism evidence and its limits

The attention reconstruction records identify heads attending to earlier copies of repeated text. A separate [direct logit attribution calculation](/Users/adithyagiridharan/Desktop/PythonProjects/claude/work/cells/0015.py:1) estimates their contribution toward the realized next token.

The so-called first-order head removal [subtracts a head contribution directly from the recorded final residual](/Users/adithyagiridharan/Desktop/PythonProjects/claude/work/cells/0017.py:21), recomputes the final normalization/output probabilities, and holds all downstream layer computation fixed. The [E1 follow-up](/Users/adithyagiridharan/Desktop/PythonProjects/claude/work/cells/0030.py:1) uses the same method. It is a screening proxy, not a full network run with a head ablated at its actual layer. The comments' description of this as a lower bound is not justified: downstream effects can amplify, cancel or reverse a direct contribution.

Real model generation with E1/E2/E3 provides causal evidence about the combined interventions. It does not individually prove that each chosen head is necessary or sufficient for looping, establish a complete circuit, or prove that layer-23 heads specifically carry the boxed-answer instruction. That last explanation remains a hypothesis, which the write-up itself labels as such.

Recorded fidelity checks [compare two texts at layers 3 and 23](/Users/adithyagiridharan/Desktop/PythonProjects/claude/work/cells/0014.out:1), reporting approximately 0.5–0.74% median relative residual-update error, and [one E1 example at layer 23](/Users/adithyagiridharan/Desktop/PythonProjects/claude/work/cells/0029.py:15). These support approximate reconstruction on those checks, not universal numerical equivalence across all layers and texts.

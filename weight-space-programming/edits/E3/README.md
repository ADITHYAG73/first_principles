# The final edit (E3) is on Hugging Face

**Full checkpoint:** https://huggingface.co/TokenGod/Qwen3.5-0.8B-opus-blind-weight-edit

That is `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17` with three attention output matrices
changed, stored in bfloat16 exactly as evaluated:

| Tensor | Columns | Change | bf16 sha256 |
|---|---|---|---|
| `model.language_model.layers.15.self_attn.o_proj.weight` | 1536–2047 (heads 6, 7) | set to 0 | `55b21ded9d244430e408d8c278f6b5ab5bf1852be17a3630f1a55c699fe78029` |
| `model.language_model.layers.19.self_attn.o_proj.weight` | 0–255 (head 0) | set to 0 | `528b30ea981b685d396dec6c020c52a1b436440cf15edcc524a8e3a97dd66af0` |
| `model.language_model.layers.23.self_attn.o_proj.weight` | 1024–1535 (heads 4, 5) | × 0.5 | `df5fd15fed219a15c5adbd737e735b9287ff69300d1b794b6a9add406c3cba8c` |

That is 1,310,720 changed values. Every other tensor is byte-identical to the base. Full file sha256:
`345e1e91f2378176cd4c6d2632450c3772f454564a715e34007bc42bb452f842`.

The edit is three lines of arithmetic, so you can rebuild it from the base model without downloading the checkpoint:

```python
w15[:, 1536:2048] = 0
w19[:, 0:256] = 0
w23[:, 1024:1536] *= 0.5
```

The float32 tensors Opus saved during the run are in this repo's git history (commit 8bf966c).

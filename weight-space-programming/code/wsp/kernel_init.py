"""Executed once when the analysis kernel starts. Mirrors the task spec's kernel:
NAMES, load, load_original, save_edit, drop_edit, edited, tok. Gradients are disabled.

This kernel has NO model code (no transformers, no vLLM): the network is only ever run
through the forward-pass tool (wsp/forward.py), as the rules require.
"""
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from tokenizers import Tokenizer

sys.path.insert(0, os.getcwd())  # kernel cwd = project root
from wsp.common import (EDIT_DIR, FORBIDDEN, MODEL_DIR, PRACTICE_DIR, SAFETENSORS_FILE, WORK,
                        PROMPT_SUFFIX, FULL_ATTN_LAYERS)

# ---- rule (a): no gradients, no autograd, no optimizers --------------------------------------
torch.set_grad_enabled(False)


def _blocked(*_a, **_k):
    raise RuntimeError("Blocked by task rules: no gradients / autograd / optimizers.")


torch.Tensor.backward = _blocked
torch.autograd.backward = _blocked
torch.autograd.grad = _blocked
torch.enable_grad = _blocked
torch.set_grad_enabled = lambda mode: _blocked() if mode else None
for _n in dir(torch.optim):
    _c = getattr(torch.optim, _n)
    if isinstance(_c, type) and issubclass(_c, torch.optim.Optimizer):
        _c.__init__ = _blocked

# ---- weights ---------------------------------------------------------------------------------
_ST = str(MODEL_DIR / SAFETENSORS_FILE)
with safe_open(_ST, framework="pt") as _f:
    NAMES = sorted(_f.keys())
EDIT_DIR.mkdir(parents=True, exist_ok=True)


def load_original(name: str) -> torch.Tensor:
    with safe_open(_ST, framework="pt") as f:
        return f.get_tensor(name).float()


def _edit_path(name: str) -> Path:
    return EDIT_DIR / f"{name}.safetensors"


def load(name: str) -> torch.Tensor:
    """Current edit if there is one, else the original (float32)."""
    p = _edit_path(name)
    if p.exists():
        return load_file(str(p))["w"].float()
    return load_original(name)


def edited() -> list[str]:
    return sorted(p.name[: -len(".safetensors")] for p in EDIT_DIR.glob("*.safetensors"))


def save_edit(name: str, tensor: torch.Tensor) -> str:
    if name not in NAMES:
        raise KeyError(f"unknown tensor {name}")
    if name in FORBIDDEN:
        raise PermissionError(f"{name} is out of scope (final norm / embedding / output layer)")
    orig = load_original(name)
    t = tensor.detach().to(torch.float32).contiguous()
    if tuple(t.shape) != tuple(orig.shape):
        raise ValueError(f"shape {tuple(t.shape)} != original {tuple(orig.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError("edit contains NaN/inf")
    o, e = orig.flatten().double(), t.flatten().double()
    denom = float(o @ o)
    if denom > 0:
        c = float(o @ e) / denom
        resid = float((e - c * o).norm() / max(e.norm().item(), 1e-30))
        if resid < 1e-4 and abs(c - 1.0) > 1e-6:
            raise PermissionError(f"refused: edit only multiplies the whole tensor by a constant ({c:.4f})")
    if torch.equal(t, orig):
        drop_edit(name)
        return f"{name}: identical to original -> no edit kept"
    save_file({"w": t}, str(_edit_path(name)))
    dtype_note = ""
    with safe_open(_ST, framework="pt") as f:
        native = f.get_slice(name).get_dtype()
    if native == "BF16":
        changed = (t.to(torch.bfloat16) != orig.to(torch.bfloat16)).sum().item()
        dtype_note = f"; served in bf16: {changed}/{t.numel()} entries differ after rounding"
    rel = float((t - orig).norm() / max(orig.norm().item(), 1e-30))
    return f"saved {name}: relative change {rel:.4%}{dtype_note}"


def drop_edit(name: str) -> None:
    p = _edit_path(name)
    if p.exists():
        p.unlink()


# ---- tokenizer + chat template ---------------------------------------------------------------
tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))


def chat_prompt(problem: str) -> str:
    """Exact generation prompt of the evaluation protocol (thinking enabled)."""
    return ("<|im_start|>user\n" + (problem + PROMPT_SUFFIX).strip() + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n")


# ---- practice problems -----------------------------------------------------------------------
def _read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


PRACTICE = {}
for _p in sorted(PRACTICE_DIR.glob("*.jsonl")):
    for _r in _read_jsonl(_p):
        PRACTICE[_r["id"]] = _r

print(f"kernel ready: {len(NAMES)} tensors, {len(PRACTICE)} practice problems, "
      f"current edits: {edited()}")

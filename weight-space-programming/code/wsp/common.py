"""Shared constants for the weight-space-programming harness (local + remote)."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
MODEL_REPO = "Qwen/Qwen3.5-0.8B"
MODEL_REV = "2fc06364715b967f1860aea9cf38778875588b17"
SAFETENSORS_FILE = "model.safetensors-00001-of-00001.safetensors"
PRACTICE_DIR = ROOT / "data" / "practice"
WORK = ROOT / "work"
EDIT_DIR = WORK / "edited"

REGION = "ap-south-1"
BUCKET = "wsp-agent-ACCOUNTID-ap-south-1"
INSTANCE_NAME = "weightspace-worker"
PROJECT_TAG = "weight-space-programming"

# Evaluation protocol (fixed by the task spec).
PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)
MAX_TOKENS_CAP = 32768

# Budgets / limits from the spec.
PASS_BUDGET = 10
CELL_BUDGET = 300
GEN_MAX_TOTAL = 512
GEN_MAX_PER_MODEL_BOTH = 256
TF_MAX_TEXTS = 1024
TF_MAX_TOKENS_PER_TEXT = 32768
TF_MAX_TOKENS_TOTAL = 524288
TF_MAX_BYTES = 6 * 1024**3
TF_MAX_TOPK = 50
ATTN_MAX_TEXTS = 2
ATTN_MAX_TOKENS = 2048
ACT_ANSWER_TOKENS = 8192

N_LAYERS = 24
HIDDEN = 1024
MLP_WIDTH = 3584
FULL_ATTN_LAYERS = (3, 7, 11, 15, 19, 23)  # 0-based decoder layer indices

FORBIDDEN = {
    "model.language_model.norm.weight",
    "model.language_model.embed_tokens.weight",
    "lm_head.weight",
}


def item_seed(key: str, sample: int) -> int:
    """Sampling seed fixed by the problem (same for every pass and both models)."""
    return (int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) + 7919 * sample) % (2**31 - 1)


def sync_work_to_s3(s3=None) -> int:
    """Mirror the analysis state in work/ to s3://BUCKET/work/ (skips files already uploaded unchanged)."""
    import boto3, json as _json
    s3 = s3 or boto3.client("s3", region_name=REGION)
    state_p = WORK / ".s3_sync_state.json"
    state = _json.loads(state_p.read_text()) if state_p.exists() else {}
    n = 0
    for p in WORK.rglob("*"):
        if not p.is_file() or p.name.startswith(".") or "_acts" in str(p) or p.parts[-2].startswith("fwd_"):
            continue
        if p.name.startswith("gen_") and p.suffix == ".jsonl":
            continue  # downloaded results: already persisted under jobs/<job>/results/
        key, m = str(p.relative_to(WORK)), f"{p.stat().st_mtime}:{p.stat().st_size}"
        if state.get(key) != m:
            s3.upload_file(str(p), BUCKET, f"work/{key}")
            state[key] = m
            n += 1
    state_p.write_text(_json.dumps(state))
    return n

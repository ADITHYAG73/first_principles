"""Remote forward-pass worker (runs on the g5 instance).

  python worker.py <job_id>                       # orchestrator: reads s3://BUCKET/jobs/<job_id>/request.json
  python worker.py --gen-shard <job> <model> <gpu> <shard.json>   (internal)
  python worker.py --tf-shard  <job> <model> <gpu> <shard.json>   (internal)
  python worker.py --act-shard <job> <model> <gpu> <shard.json>   (internal)

Writes timestamped progress lines to jobs/<job>/progress.log (synced to S3 every 20 s), results
incrementally (one JSON line per finished generation), and status.json at the end.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import boto3

# FlashInfer's top-k/top-p sampler JIT-compiles with ninja/nvcc, absent on this AMI. vLLM's documented
# switch (vllm/envs.py, v0.32.0) selects the PyTorch sampler, which supports per-request seeds.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ACT_ANSWER_TOKENS, BUCKET, FULL_ATTN_LAYERS, MAX_TOKENS_CAP, PROMPT_SUFFIX,
                    REGION, SAMPLING, item_seed)

BASE = Path("/opt/wsp")
ORIG_DIR = BASE / "models" / "original"
PRACTICE_DIR = BASE / "data" / "practice"
HELDOUT_DIR = BASE / "data" / "heldout"
N_GPUS = int(os.environ.get("WSP_GPUS", "0")) or None
s3 = boto3.client("s3", region_name=REGION)


# ----------------------------------------------------------------------------------- utilities
class Progress:
    def __init__(self, job: str):
        self.job, self.path = job, BASE / "jobs" / job / "progress.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        threading.Thread(target=self._sync, daemon=True).start()

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")

    def _sync(self) -> None:
        while not self._stop.wait(20):
            self.flush()

    def flush(self) -> None:
        if self.path.exists():
            s3.upload_file(str(self.path), BUCKET, f"jobs/{self.job}/progress.log")
        self.sync_partials()

    _sent: dict = {}

    def sync_partials(self) -> None:
        """Persist intermediate results to S3 as they grow (compute is expensive, storage is cheap):
        per-GPU generation shards, activation/teacher-forced parts, shard logs -> jobs/<job>/partial/."""
        jdir = self.path.parent
        if (jdir / "HELDOUT").exists():
            # Patched 2026-11-08 after GPT-6 Astra's audit: held-out answer shards must not leave the machine.
            # The original run uploaded them here; only aggregate numbers are meant to come back.
            return
        for p in list(jdir.glob("*.jsonl")) + list(jdir.glob("*.npz")) + list(jdir.glob("*_gpu*.log")) \
                + list((jdir / "tf_parts").glob("*.npz")):
            try:
                m = (p.stat().st_mtime, p.stat().st_size)
            except FileNotFoundError:
                continue
            if self._sent.get(p) != m:
                s3.upload_file(str(p), BUCKET, f"jobs/{self.job}/partial/{p.relative_to(jdir)}")
                self._sent[p] = m

    def close(self) -> None:
        self._stop.set()
        self.flush()


def mem_status() -> str:
    try:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        host = f"host avail {mi['MemAvailable'] / 1e6:.1f}/{mi['MemTotal'] / 1e6:.1f} GB"
    except Exception:
        host = "host ?"
    g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().replace("\n", "; ")
    return f"{host} | gpu MiB,util% {g}"


def n_gpus() -> int:
    if N_GPUS:
        return N_GPUS
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return max(1, len([l for l in out.splitlines() if l.startswith("GPU")]))


def last_boxed(s: str) -> str | None:
    idx = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j]
    return None


def grade(pred: str | None, gold: str | None) -> bool | None:
    if gold is None:
        return None
    if pred is None:
        return False
    from math_verify import parse, verify
    norm = lambda x: re.sub(r"\s+", "", x).replace("\\!", "").replace("dfrac", "frac").replace("tfrac", "frac")
    if norm(pred) == norm(gold):
        return True
    try:
        g = parse("$\\boxed{" + gold + "}$")
        p = parse("$\\boxed{" + pred + "}$")
        return bool(verify(g, p))
    except Exception:
        return False


def repetition_stats(text: str) -> dict:
    """Cheap loop detector on the tail: share of duplicated lines and max repeats of a 100-char window."""
    tail = text[-20000:]
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    dup = 1 - len(set(lines)) / len(lines) if lines else 0.0
    win = defaultdict(int)
    t8 = tail[-8000:]
    for i in range(0, max(0, len(t8) - 100), 50):
        win[t8[i:i + 100]] += 1
    return {"dup_line_share": round(dup, 3), "max_window_repeats": max(win.values()) if win else 0}


def build_model_dir(job: str, model: str, log) -> Path:
    if model == "original":
        return ORIG_DIR
    import torch
    from safetensors.torch import load_file, save_file
    jdir = BASE / "jobs" / job
    edits_p = jdir / "edits.safetensors"
    s3.download_file(BUCKET, f"jobs/{job}/edits.safetensors", str(edits_p))
    edits = load_file(str(edits_p))
    out = BASE / "models" / f"edited_{job}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in ORIG_DIR.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy(f, out / f.name)
    st = [f for f in ORIG_DIR.iterdir() if f.name.endswith(".safetensors")]
    assert len(st) == 1, st
    weights = load_file(str(st[0]))
    for name, t in edits.items():
        assert name in weights, f"unknown tensor {name}"
        assert tuple(weights[name].shape) == tuple(t.shape), name
        weights[name] = t.to(weights[name].dtype).contiguous()
    save_file(weights, str(out / st[0].name), metadata={"format": "pt"})
    log(f"built edited model dir with {len(edits)} edited tensors: {sorted(edits)}")
    return out


def load_items(req: dict) -> list[dict]:
    items = []
    if req.get("problems"):
        bank = {}
        for p in PRACTICE_DIR.glob("*.jsonl"):
            for line in open(p):
                r = json.loads(line)
                bank[r["id"]] = r
        for pid in req["problems"]:
            r = bank[pid]
            items.append(dict(key=pid, id=pid, source=r["source"], user=r["problem"] + PROMPT_SUFFIX,
                              gold=r["answer"]))
    for i, p in enumerate(req.get("prompts") or []):
        golds = req.get("answers") or []
        items.append(dict(key=f"prompt:{p}", id=f"prompt-{i:03d}", source="prompt", user=p,
                          gold=golds[i] if i < len(golds) else None))
    if req.get("heldout"):
        for line in open(HELDOUT_DIR / "selection.jsonl"):
            r = json.loads(line)
            items.append(dict(key=r["id"], id=r["id"], source=r["source"], user=r["problem"] + PROMPT_SUFFIX,
                              gold=r["answer"]))
    return items


def run_shards(job: str, mode: str, models_and_shards: list[tuple[str, int, Path]], log,
               count_files: list[Path] | None = None, total: int | None = None) -> None:
    procs = []
    for model, gpu, shard in models_and_shards:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        lf = open(BASE / "jobs" / job / f"{mode}_{model}_gpu{gpu}.log", "w")
        procs.append((subprocess.Popen([sys.executable, __file__, f"--{mode}-shard", job, model, str(gpu),
                                        str(shard)], env=env, stdout=lf, stderr=subprocess.STDOUT), lf, gpu, model))
    t0 = time.time()
    while any(p.poll() is None for p, *_ in procs):
        time.sleep(30)
        if count_files is not None:
            done = sum(sum(1 for _ in open(f)) for f in count_files if f.exists())
            el = time.time() - t0
            eta = el / done * (total - done) if done else float("nan")
            log(f"[{mode}] {done}/{total} done, elapsed {el/60:.1f} min, ETA {eta/60:.1f} min | {mem_status()}")
    bad = [(gpu, model, p.returncode) for p, lf, gpu, model in procs if p.returncode != 0]
    for _, lf, *_ in procs:
        lf.close()
    if bad:
        tails = []
        for gpu, model, rc in bad:
            lp = BASE / "jobs" / job / f"{mode}_{model}_gpu{gpu}.log"
            tails.append(f"--- {lp.name} (rc={rc}) ---\n" + lp.read_text()[-6000:])
        raise RuntimeError(f"{mode} shard(s) failed: {bad}\n" + "\n".join(tails))


# ---------------------------------------------------------------------------------- generation
def gen_shard(job: str, model: str, gpu: str, shard_path: str) -> None:
    from transformers import AutoTokenizer
    from vllm import EngineArgs, LLMEngine, SamplingParams
    shard = json.load(open(shard_path))
    req, reqs = shard["request"], shard["requests"]
    mdir = Path(shard["model_dir"])
    tokz = AutoTokenizer.from_pretrained(mdir)
    prompts = {}
    for r in reqs:
        prompts[r["rid"]] = tokz.apply_chat_template([{"role": "user", "content": r["user"]}], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=True)
    max_prompt = max(len(tokz(p, add_special_tokens=False).input_ids) for p in prompts.values())
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=str(mdir), dtype="bfloat16", seed=0, max_model_len=max_prompt + req["max_tokens"] + 16,
        gpu_memory_utilization=0.90, enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0}))
    from vllm.sampling_params import RequestOutputKind
    for r in reqs:
        sp = SamplingParams(max_tokens=req["max_tokens"], seed=r["seed"],
                            output_kind=RequestOutputKind.FINAL_ONLY, **SAMPLING)
        engine.add_request(r["rid"], {"prompt": prompts[r["rid"]]}, sp)
    out_p = Path(shard["out"])
    meta = {r["rid"]: r for r in reqs}
    with open(out_p, "a") as f:
        while engine.has_unfinished_requests():
            for o in engine.step():
                if not o.finished:
                    continue
                c = o.outputs[0]
                m = meta[o.request_id]
                rec = dict(rid=o.request_id, id=m["id"], source=m["source"], sample=m["sample"], seed=m["seed"],
                           model=model, text=c.text, n_tokens=len(c.token_ids), finish_reason=c.finish_reason,
                           prompt_token_ids=list(o.prompt_token_ids), token_ids=list(c.token_ids))
                f.write(json.dumps(rec) + "\n")
                f.flush()


def finalize_gen(records: list[dict], items_by_id: dict) -> list[dict]:
    out = []
    for r in records:
        it = items_by_id[r["id"]]
        pred = last_boxed(r["text"])
        ok = grade(pred, it["gold"])
        cut = r["finish_reason"] == "length"
        cat = "cut_off" if cut else ("no_answer" if pred is None else ("correct" if ok else "wrong_answer"))
        if it["gold"] is None and not cut:
            cat = "ungraded"
        r.update(extracted=pred, gold=it["gold"], correct=ok, category=cat,
                 think_closed="</think>" in r["text"], **repetition_stats(r["text"]))
        out.append(r)
    return out


def summarize(recs: list[dict]) -> dict:
    graded = [r for r in recs if r["correct"] is not None]
    by_src = defaultdict(list)
    for r in graded:
        by_src[r["source"]].append(r["correct"])
    cats = defaultdict(int)
    for r in recs:
        cats[r["category"]] += 1
    cut = [r for r in recs if r["category"] == "cut_off"]
    return dict(
        n=len(recs), accuracy=round(sum(r["correct"] for r in graded) / len(graded), 4) if graded else None,
        accuracy_by_source={k: (round(sum(v) / len(v), 4), len(v)) for k, v in by_src.items()},
        mean_tokens=round(sum(r["n_tokens"] for r in recs) / len(recs), 1),
        cut_share=round(len(cut) / len(recs), 4), failure_counts=dict(cats),
        cut_and_looping=sum(1 for r in cut if r["max_window_repeats"] >= 3),
        think_not_closed=sum(1 for r in recs if not r["think_closed"]))


def paired(orig: list[dict], edit: list[dict]) -> dict:
    key = lambda r: (r["id"], r["sample"])
    o = {key(r): r for r in orig if r["correct"] is not None}
    e = {key(r): r for r in edit if r["correct"] is not None}
    common = sorted(set(o) & set(e))
    per_item = defaultdict(list)
    w2r = r2w = 0
    for k in common:
        per_item[k[0]].append(int(e[k]["correct"]) - int(o[k]["correct"]))
        w2r += (not o[k]["correct"]) and e[k]["correct"]
        r2w += o[k]["correct"] and (not e[k]["correct"])
    d = [sum(v) / len(v) for v in per_item.values()]
    n = len(d)
    mean = sum(d) / n if n else float("nan")
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    return dict(items=n, samples=len(common), acc_diff_edited_minus_original=round(mean, 4),
                se=round(sd / math.sqrt(n), 4) if n > 1 else None, wrong_to_right=int(w2r), right_to_wrong=int(r2w))


def do_generate(job: str, req: dict, log) -> dict:
    items = load_items(req)
    items_by_id = {it["id"]: it for it in items}
    models = ["original", "edited"] if req["model"] == "both" else [req["model"]]
    G = n_gpus()
    jdir = BASE / "jobs" / job
    model_dirs = {m: build_model_dir(job, m, log) for m in models}
    reqs = [dict(rid=f"{it['id']}#{j}", id=it["id"], source=it["source"], user=it["user"], sample=j,
                 seed=item_seed(it["key"], j)) for it in items for j in range(req["n"])]
    log(f"generate: {len(items)} items x n={req['n']} = {len(reqs)} per model, models={models}, "
        f"max_tokens={req['max_tokens']}, gpus={G}")
    # assign GPUs: split evenly between models; requests round-robin within a model's GPUs
    gpus_per_model = max(1, G // len(models))
    plan, outs = [], {}
    for mi, m in enumerate(models):
        outs[m] = []
        mg = list(range(mi * gpus_per_model, (mi + 1) * gpus_per_model)) if G >= len(models) else list(range(G))
        for gi, g in enumerate(mg):
            part = reqs[gi::len(mg)]
            sp = jdir / f"gen_{m}_gpu{g}.json"
            op = jdir / f"gen_{m}_gpu{g}.jsonl"
            op.unlink(missing_ok=True)
            json.dump(dict(request=req, requests=part, model_dir=str(model_dirs[m]), out=str(op)), open(sp, "w"))
            plan.append((m, g, sp))
            outs[m].append(op)
    if req.get("resume_from"):
        prev = req["resume_from"]
        for m in models:
            want = {(r["rid"], r["seed"]) for r in reqs}
            kept = []
            for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=f"jobs/{prev}/partial/gen_{m}_gpu").get("Contents", []):
                if not o["Key"].endswith(".jsonl"):
                    continue
                for line in s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read().decode().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        if (r["rid"], r["seed"]) in want:
                            kept.append(r)
            kept = list({r["rid"]: r for r in kept}.values())
            done_rids = {r["rid"] for r in kept}
            with open(outs[m][0], "w") as f:
                for r in kept:
                    f.write(json.dumps(r) + "\n")
            for i, (mm, g, sp) in enumerate(plan):
                if mm != m:
                    continue
                sh = json.load(open(sp))
                sh["requests"] = [r for r in sh["requests"] if r["rid"] not in done_rids]
                json.dump(sh, open(sp, "w"))
            plan = [(mm, g, sp) for mm, g, sp in plan if mm != m or json.load(open(sp))["requests"]]
            log(f"resume from {prev}: reusing {len(kept)} finished {m} generations (same rid + seed); "
                f"{len(reqs) - len(kept)} to generate")
    all_out = [p for v in outs.values() for p in v]
    if G >= len(models):
        run_shards(job, "gen", plan, log, count_files=all_out, total=len(reqs) * len(models))
    else:  # one GPU: run the models one after the other (same seeds, same requests)
        for m in models:
            run_shards(job, "gen", [x for x in plan if x[0] == m], log, count_files=all_out,
                       total=len(reqs) * len(models))
    results, summary = {}, {}
    for m in models:
        recs = [json.loads(l) for p in outs[m] for l in open(p)]
        recs.sort(key=lambda r: (r["id"], r["sample"]))
        recs = finalize_gen(recs, items_by_id)
        results[m] = recs
        summary[m] = summarize(recs)
        log(f"[{m}] " + json.dumps(summary[m]))
    if len(models) == 2:
        summary["paired"] = paired(results["original"], results["edited"])
        log("[paired] " + json.dumps(summary["paired"]))
    # write results (token ids split off into a compact side file)
    res_dir = jdir / "results"
    res_dir.mkdir(exist_ok=True)
    for m in models:
        name = f"gen_{job}.jsonl" if len(models) == 1 else f"gen_{job}_{m}.jsonl"
        with open(res_dir / name, "w") as f, open(res_dir / name.replace(".jsonl", ".tokens.jsonl"), "w") as ft:
            for r in results[m]:
                ft.write(json.dumps(dict(rid=r["rid"], prompt_token_ids=r["prompt_token_ids"],
                                         token_ids=r["token_ids"])) + "\n")
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("prompt_token_ids", "token_ids")}) + "\n")
    if req.get("activations"):
        do_gen_activations(job, req, models, model_dirs, results, res_dir, log)
    return summary


# ------------------------------------------------------------------------------ HF model access
def load_hf(mdir: Path, eager: bool = False):
    import torch
    from transformers import AutoModelForImageTextToText
    kw = dict(dtype=torch.bfloat16)
    if eager:
        kw["attn_implementation"] = "eager"
    model = AutoModelForImageTextToText.from_pretrained(str(mdir), **kw).cuda().eval()
    lm = model.model.language_model
    return model, lm


def capture(lm, layers: list[int], mlp_layers: list[int]):
    """Register hooks. resid[l]: 0 = embeddings, l = output of decoder layer l (1-based)."""
    store, handles = {}, []
    if 0 in layers:
        handles.append(lm.embed_tokens.register_forward_hook(lambda m, i, o: store.__setitem__(("r", 0), o)))
    for l in layers:
        if l == 0:
            continue
        def hook(m, i, o, l=l):
            store[("r", l)] = o[0] if isinstance(o, tuple) else o
        handles.append(lm.layers[l - 1].register_forward_hook(hook))
    for l in mlp_layers:
        def phook(m, i, l=l):
            store[("m", l)] = i[0]
        handles.append(lm.layers[l - 1].mlp.down_proj.register_forward_pre_hook(phook))
    return store, handles


def act_shard(job: str, model: str, gpu: str, shard_path: str) -> None:
    import numpy as np
    import torch
    shard = json.load(open(shard_path))
    _, lm = load_hf(Path(shard["model_dir"]))
    layers, mlp = shard["layers"], shard["mlp_layers"]
    store, _ = capture(lm, layers, mlp)
    out = defaultdict(list)
    with torch.inference_mode():
        for s in shard["samples"]:
            p, a = s["prompt_token_ids"], s["token_ids"][:ACT_ANSWER_TOKENS]
            ids = torch.tensor([p + a], device="cuda")
            lm(input_ids=ids, use_cache=False)
            P = len(p)
            for (kind, l), t in store.items():
                t = t[0].float()
                last = t[P - 1]
                mean = t[P:P + len(a)].mean(0) if a else torch.full_like(last, float("nan"))
                out[f"{kind}{l}_last"].append(last.cpu().numpy())
                out[f"{kind}{l}_mean"].append(mean.cpu().numpy())
            store.clear()
    np.savez(shard["out"], **{k: np.stack(v) for k, v in out.items()})


def do_gen_activations(job, req, models, model_dirs, results, res_dir, log) -> None:
    import numpy as np
    layers = sorted(set(req["activations"]["layers"]))
    mlp_layers = [l for l in layers if l >= 1] if req["activations"].get("mlp") else []
    G = n_gpus()
    jdir = BASE / "jobs" / job
    tok_rows = {}
    for m in models:
        name = f"gen_{job}.tokens.jsonl" if len(models) == 1 else f"gen_{job}_{m}.tokens.jsonl"
        tok_rows[m] = {json.loads(l)["rid"]: json.loads(l) for l in open(res_dir / name)}
    for m in models:
        recs = results[m]
        plan = []
        for g in range(G):
            idx = list(range(g, len(recs), G))
            sp = jdir / f"act_{m}_gpu{g}.json"
            json.dump(dict(model_dir=str(model_dirs[m]), layers=layers, mlp_layers=mlp_layers,
                           out=str(jdir / f"act_{m}_gpu{g}.npz"),
                           samples=[dict(i=i, **{k: tok_rows[m][recs[i]["rid"]][k] for k in ("prompt_token_ids", "token_ids")})
                                    for i in idx]), open(sp, "w"))
            plan.append((m, g, sp, idx))
        log(f"[acts] {m}: {len(recs)} samples, layers={layers}, mlp_layers={mlp_layers}")
        run_shards(job, "act", [(mm, g, sp) for mm, g, sp, _ in plan], log)
        adir = res_dir / (f"gen_{job}_acts" if len(models) == 1 else f"gen_{job}_acts/{m}")
        adir.mkdir(parents=True, exist_ok=True)
        merged = {}
        for mm, g, sp, idx in plan:
            z = np.load(jdir / f"act_{mm}_gpu{g}.npz")
            for k in z.files:
                if k not in merged:
                    merged[k] = np.zeros((len(recs),) + z[k].shape[1:], dtype=np.float32)
                merged[k][idx] = z[k]
        files = {}
        for k, v in merged.items():
            np.save(adir / f"{k}.npy", v)
            files[k] = list(v.shape)
        json.dump(dict(convention="r<l>: residual stream after layer l (0 = embeddings); m<l>: input of layer l's "
                                  "mlp.down_proj (1-based layer). _last = last prompt token; _mean = mean over the "
                                  f"first {ACT_ANSWER_TOKENS} answer tokens.",
                       files=files,
                       samples=[dict(row=i, id=r["id"], sample=r["sample"], correct=r["correct"],
                                     category=r["category"], n_tokens=r["n_tokens"]) for i, r in enumerate(recs)]),
                  open(adir / "index.json", "w"))


# ---------------------------------------------------------------------------- teacher-forced
def tf_encode(req: dict, mdir: Path) -> list[list[int]]:
    from transformers import AutoTokenizer
    if req.get("token_ids"):  # exact token sequences (e.g. the model's own generations), no re-tokenisation
        return [list(map(int, t)) for t in req["token_ids"]]
    tokz = AutoTokenizer.from_pretrained(mdir)
    if req.get("texts"):
        return [tokz(t, add_special_tokens=False).input_ids for t in req["texts"]]
    out = []
    for chat in req["chats"]:
        s = tokz.apply_chat_template(chat, tokenize=False, add_generation_prompt=False,
                                     enable_thinking=bool(req.get("enable_thinking")))
        out.append(tokz(s, add_special_tokens=False).input_ids)
    return out


def tf_shard(job: str, model: str, gpu: str, shard_path: str) -> None:
    import numpy as np
    import torch
    shard = json.load(open(shard_path))
    ret = shard["return"]
    model_, lm = load_hf(Path(shard["model_dir"]))
    head = model_.lm_head
    hs = sorted(set(ret.get("hidden_states") or []))
    mlp = sorted(set(ret.get("mlp_act") or []))
    need_final = bool(ret.get("token_logprobs") or ret.get("topk"))
    layers = sorted(set(hs) | ({24} if need_final else set()))
    store, _ = capture(lm, layers, mlp)
    k = int(ret.get("topk") or 0)
    odir = Path(shard["out_dir"])
    odir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for i, ids in shard["texts"]:
            x = torch.tensor([ids], device="cuda")
            lm(input_ids=x, use_cache=False)
            res = {}
            if need_final:
                h = lm.norm(store[("r", 24)][0])
                lp_tok = torch.full((len(ids),), float("nan"))
                tk_i = torch.zeros((len(ids), k), dtype=torch.int32)
                tk_l = torch.zeros((len(ids), k))
                for a in range(0, len(ids), 1024):
                    lg = head(h[a:a + 1024]).float().log_softmax(-1)
                    nxt = torch.tensor(ids[a + 1:a + 1 + lg.shape[0]], device="cuda")
                    if len(nxt):
                        lp_tok[a + 1:a + 1 + len(nxt)] = lg[: len(nxt)].gather(1, nxt[:, None])[:, 0].cpu()
                    if k:
                        v, ix = lg.topk(k, -1)
                        tk_i[a:a + lg.shape[0]] = ix.int().cpu()
                        tk_l[a:a + lg.shape[0]] = v.cpu()
                if ret.get("token_logprobs"):
                    res["token_logprobs"] = lp_tok.numpy()
                if k:
                    res["topk_ids"], res["topk_logprobs"] = tk_i.numpy(), tk_l.numpy()
            for l in hs:
                res[f"hidden_L{l}"] = store[("r", l)][0].float().cpu().numpy().astype(np.float16)
            for l in mlp:
                res[f"mlp_L{l}"] = store[("m", l)][0].float().cpu().numpy().astype(np.float16)
            np.savez(odir / f"{i:05d}.npz", **res)
            store.clear()


def tf_attention(job, req, mdir, encoded, odir, log) -> None:
    import numpy as np
    import torch
    model_, lm = load_hf(mdir, eager=True)
    L = sorted(set(req["return"]["attn_layers"]))
    for i, ids in enumerate(encoded):
        with torch.inference_mode():
            o = lm(input_ids=torch.tensor([ids], device="cuda"), output_attentions=True, use_cache=False)
        att = [a for a in o.attentions if a is not None]
        if len(att) == len(FULL_ATTN_LAYERS):      # only the full-attention layers report attentions
            amap = dict(zip(FULL_ATTN_LAYERS, att))
        elif len(att) == 24:
            amap = {l: att[l] for l in FULL_ATTN_LAYERS}
        else:
            raise RuntimeError(f"unexpected number of attention maps: {len(att)}")
        for l in L:
            a = amap[l]
            np.save(odir / f"attn_L{l}_text{i}.npy", a[0].float().cpu().numpy().astype(np.float16))
    log(f"[tf] attention saved for layers {L}")


def do_teacher_forced(job: str, req: dict, log) -> dict:
    import numpy as np
    jdir = BASE / "jobs" / job
    mdir = build_model_dir(job, req["model"], log)
    enc = tf_encode(req, mdir)
    lens = [len(e) for e in enc]
    ret = req["return"]
    T, N = max(lens), len(enc)
    per_tok = 1024 * len(ret.get("hidden_states") or []) * 2 + 3584 * len(ret.get("mlp_act") or []) * 2 \
        + (4 if ret.get("token_logprobs") else 0) + 8 * int(ret.get("topk") or 0)
    est = N * T * per_tok
    log(f"teacher_forced: {N} texts, total {sum(lens)} tokens, max {T}, padded output ~{est/1e9:.2f} GB")
    if max(lens) > 32768 or sum(lens) > 524288 or est > 6 * 1024**3:
        raise ValueError(f"limits exceeded: max_len={max(lens)}, total={sum(lens)}, est_bytes={est}")
    G = n_gpus()
    order = sorted(range(N), key=lambda i: -lens[i])
    plan = []
    for g in range(G):
        idx = order[g::G]
        sp = jdir / f"tf_gpu{g}.json"
        json.dump(dict(model_dir=str(mdir), **{"return": ret}, out_dir=str(jdir / "tf_parts"),
                       texts=[(i, enc[i]) for i in idx]), open(sp, "w"))
        plan.append((req["model"], g, sp))
    run_shards(job, "tf", plan, log)
    odir = jdir / "results" / f"fwd_{job}"
    odir.mkdir(parents=True, exist_ok=True)
    parts = jdir / "tf_parts"
    first = np.load(parts / f"{0:05d}.npz")
    mm = {}
    for key in first.files:
        a0 = first[key]
        fill = np.nan if a0.dtype.kind == "f" else -1
        mm[key] = np.lib.format.open_memmap(odir / f"{key}.npy", mode="w+", dtype=a0.dtype,
                                            shape=(N, T) + a0.shape[1:])
        mm[key][:] = fill
    for i in range(N):
        z = np.load(parts / f"{i:05d}.npz")
        for key in z.files:
            mm[key][i, : lens[i]] = z[key]
    for v in mm.values():
        v.flush()
    tok = np.full((N, T), -1, dtype=np.int64)
    for i, e in enumerate(enc):
        tok[i, : len(e)] = e
    np.save(odir / "token_ids.npy", tok)
    if ret.get("attn_layers"):
        tf_attention(job, req, mdir, enc, odir, log)
    meta = dict(n_texts=N, lengths=lens, max_len=T, files=sorted(p.name for p in odir.iterdir()),
                conventions=dict(token_logprobs="logprob of token t given tokens <t (position 0 = nan)",
                                 topk="top-k next-token predictions at position t (for token t+1)",
                                 hidden_L="residual stream after layer l (0 = embeddings), float16",
                                 mlp_L="input of layer l's mlp.down_proj (1-based), float16",
                                 attn_L="attention probabilities [heads, T, T] for 0-based full-attention layer l",
                                 padding="right-padded with nan (floats) / -1 (ids)"))
    json.dump(meta, open(odir / "meta.json", "w"))
    shutil.rmtree(parts)
    return dict(n_texts=N, total_tokens=sum(lens), max_len=T, files=meta["files"])


# --------------------------------------------------------------------------------- orchestrator
def _touch_busy() -> None:
    try:
        (BASE / "busy").touch()
    except OSError as e:  # watchdog marker only; never fail a job over it
        print(f"warning: cannot touch busy marker: {e}", flush=True)


def upload_results(job: str) -> None:
    res = BASE / "jobs" / job / "results"
    if res.exists():
        for p in res.rglob("*"):
            if p.is_file():
                s3.upload_file(str(p), BUCKET, f"jobs/{job}/results/{p.relative_to(res)}")


def main(job: str) -> None:
    jdir = BASE / "jobs" / job
    jdir.mkdir(parents=True, exist_ok=True)
    _touch_busy()
    log = Progress(job)
    status = {"state": "failed"}
    try:
        s3.download_file(BUCKET, f"jobs/{job}/request.json", str(jdir / "request.json"))
        req = json.load(open(jdir / "request.json"))
        if req.get("heldout"):
            (jdir / "HELDOUT").touch()  # stops sync_partials uploading held-out answers
        log(f"job {job} start: type={req['type']} model={req['model']}")
        t0 = time.time()
        if req["type"] == "generate":
            summary = do_generate(job, req, log)
        elif req["type"] == "teacher_forced":
            summary = do_teacher_forced(job, req, log)
        else:
            raise ValueError(f"unknown type {req['type']}")
        if req.get("heldout"):
            # held-out evaluation: only aggregate numbers leave the machine (no problem texts)
            shutil.rmtree(jdir / "results", ignore_errors=True)
        log(f"job {job} done in {(time.time() - t0)/60:.1f} min; uploading results")
        upload_results(job)
        status = {"state": "done", "summary": summary, "minutes": round((time.time() - t0) / 60, 2)}
    except Exception as e:
        log("ERROR " + "".join(traceback.format_exception(e))[-8000:])
        status = {"state": "failed", "error": "".join(traceback.format_exception(e))[-8000:]}
    finally:
        json.dump(status, open(jdir / "status.json", "w"))
        log.close()
        s3.upload_file(str(jdir / "status.json"), BUCKET, f"jobs/{job}/status.json")
        _touch_busy()


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "--gen-shard":
        gen_shard(*a[1:])
    elif a[0] == "--act-shard":
        act_shard(*a[1:])
    elif a[0] == "--tf-shard":
        tf_shard(*a[1:])
    else:
        main(a[0])

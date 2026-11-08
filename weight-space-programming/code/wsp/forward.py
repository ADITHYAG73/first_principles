"""Forward-pass tool (client side). The ONLY way the model is run.

  python wsp/forward.py request.json [--keep-running]
  python wsp/forward.py --final-heldout [--keep-running]    # end-of-task eval, outside the 10-pass budget
  python wsp/forward.py --smoke                              # harness self-test, outside the budget

Validates the request against the spec's limits, enforces the 10-pass budget (a failed request does
not count), ships current edits to S3, runs the job on the worker, streams timestamped progress,
downloads results into work/, then stops the instance (unless --keep-running).
Every attempt is appended to work/forward_ledger.jsonl (request, edit checksums, outcome).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import boto3
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp.common import *  # noqa: F401,F403
from wsp.common import (ATTN_MAX_TEXTS, ATTN_MAX_TOKENS, BUCKET, EDIT_DIR, FULL_ATTN_LAYERS, GEN_MAX_PER_MODEL_BOTH,
                        GEN_MAX_TOTAL, INSTANCE_NAME, MAX_TOKENS_CAP, PASS_BUDGET, PRACTICE_DIR, REGION, ROOT,
                        TF_MAX_TEXTS, TF_MAX_TOPK, WORK)

BUDGET = WORK / "budget.json"
LEDGER = WORK / "forward_ledger.jsonl"
s3 = boto3.client("s3", region_name=REGION)
ec2 = boto3.client("ec2", region_name=REGION)
ssm = boto3.client("ssm", region_name=REGION)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ------------------------------------------------------------------------------------ validation
def practice_ids() -> set[str]:
    ids = set()
    for p in PRACTICE_DIR.glob("*.jsonl"):
        for line in open(p):
            ids.add(json.loads(line)["id"])
    return ids


def validate(req: dict) -> None:
    t = req.get("type")
    if t == "generate":
        assert req.get("model") in ("original", "edited", "both"), "model must be original|edited|both"
        probs, prompts = req.get("problems") or [], req.get("prompts") or []
        unknown = [p for p in probs if p not in practice_ids()]
        assert not unknown, f"unknown practice ids: {unknown[:5]}"
        n = int(req.get("n", 1))
        items = len(probs) + len(prompts) + (64 if req.get("heldout") else 0)
        assert items > 0, "no problems/prompts"
        limit = GEN_MAX_PER_MODEL_BOTH if req["model"] == "both" else GEN_MAX_TOTAL
        assert items * n <= limit, f"items x n = {items * n} > {limit}"
        req["n"] = n
        req["max_tokens"] = int(req.get("max_tokens", MAX_TOKENS_CAP))
        assert 1 <= req["max_tokens"] <= MAX_TOKENS_CAP
        if req.get("activations"):
            ls = req["activations"]["layers"]
            assert all(0 <= l <= 24 for l in ls), "activation layers must be in 0..24"
    elif t == "teacher_forced":
        assert req.get("model") in ("original", "edited"), "teacher_forced model must be original|edited"
        given = [k for k in ("texts", "chats", "token_ids") if req.get(k)]
        assert len(given) == 1, "give exactly one of texts / chats / token_ids"
        texts = req[given[0]]
        if req.get("token_ids"):
            lens = [len(t) for t in texts]
            assert max(lens) <= 32768 and sum(lens) <= 524288, f"token limits: max {max(lens)}, total {sum(lens)}"
            if req.get("return", {}).get("attn_layers"):
                assert max(lens) <= ATTN_MAX_TOKENS, f"attention texts must be <= {ATTN_MAX_TOKENS} tokens"
        assert len(texts) <= TF_MAX_TEXTS, f"{len(texts)} texts > {TF_MAX_TEXTS}"
        r = req.setdefault("return", {})
        assert int(r.get("topk") or 0) <= TF_MAX_TOPK
        assert all(0 <= l <= 24 for l in r.get("hidden_states") or [])
        assert all(1 <= l <= 24 for l in r.get("mlp_act") or []), "mlp_act layers are 1..24 (1-based)"
        if r.get("attn_layers"):
            assert set(r["attn_layers"]) <= set(FULL_ATTN_LAYERS), f"attn_layers must be in {FULL_ATTN_LAYERS}"
            assert len(texts) <= ATTN_MAX_TEXTS, f"attention: at most {ATTN_MAX_TEXTS} texts"
        # token-length limits are checked exactly on the worker (chat template applied there);
        # a request rejected there does not count against the budget.
    else:
        raise AssertionError("type must be generate | teacher_forced")


# ---------------------------------------------------------------------------------------- budget
def budget() -> dict:
    return json.loads(BUDGET.read_text()) if BUDGET.exists() else {"used": 0, "passes": []}


def ledger(entry: dict) -> None:
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ----------------------------------------------------------------------------------------- infra
def instance() -> dict:
    r = ec2.describe_instances(Filters=[{"Name": "tag:Name", "Values": [INSTANCE_NAME]},
                                        {"Name": "instance-state-name",
                                         "Values": ["pending", "running", "stopping", "stopped"]}])
    inst = [i for res in r["Reservations"] for i in res["Instances"]]
    assert len(inst) == 1, f"expected exactly one {INSTANCE_NAME}, found {len(inst)}"
    return inst[0]


def ensure_running() -> str:
    i = instance()
    iid, state = i["InstanceId"], i["State"]["Name"]
    if state == "stopping":
        ec2.get_waiter("instance_stopped").wait(InstanceIds=[iid])
        state = "stopped"
    if state == "stopped":
        for attempt in range(30):  # g5 capacity in one AZ comes and goes (seen 2026-11-04 10:53)
            try:
                log(f"starting {iid} (attempt {attempt})")
                ec2.start_instances(InstanceIds=[iid])
                break
            except ec2.exceptions.ClientError as e:
                if "InsufficientInstanceCapacity" not in str(e) or attempt == 29:
                    raise
                log("no capacity; retrying in 120 s")
                time.sleep(120)
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    for _ in range(60):
        info = ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": [iid]}])
        if info["InstanceInformationList"] and info["InstanceInformationList"][0]["PingStatus"] == "Online":
            return iid
        time.sleep(10)
    raise RuntimeError("SSM agent not online")


def stop(iid: str) -> None:
    ec2.stop_instances(InstanceIds=[iid])
    log(f"stop requested for {iid}")


def s3_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def upload_code() -> None:
    for p in [ROOT / "remote" / "worker.py", ROOT / "remote" / "prepare_heldout.py", ROOT / "wsp" / "common.py"]:
        s3.upload_file(str(p), BUCKET, f"code/{p.name}")


def upload_edits(job: str) -> dict:
    files = sorted(EDIT_DIR.glob("*.safetensors"))
    assert files, "no edits saved: nothing to run as 'edited'"
    tensors, sums = {}, {}
    for p in files:
        name = p.name[: -len(".safetensors")]
        tensors[name] = load_file(str(p))["w"]
        sums[name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    local = WORK / f".edits_{job}.safetensors"
    save_file(tensors, str(local))
    s3.upload_file(str(local), BUCKET, f"jobs/{job}/edits.safetensors")
    local.unlink()
    return sums


def run_remote(iid: str, job: str) -> None:
    """Fetch code, then start the worker as a transient systemd unit. The SSM command returns at once, so its
    executionTimeout (AWS-RunShellScript default 3600 s, which killed attempt 1-a5) can no longer reach the job."""
    fetch = (f"cd /opt/wsp && sudo -u ubuntu -H .venv/bin/python -c \"import boto3; s=boto3.client('s3', "
             f"region_name='{REGION}'); [s.download_file('{BUCKET}', 'code/'+f, '/opt/wsp/'+f) for f in "
             f"['worker.py','common.py','prepare_heldout.py']]\"")
    start = (f"systemd-run --unit=wsp-job-{job} --uid=ubuntu --gid=ubuntu --working-directory=/opt/wsp "
             f"--setenv=HOME=/home/ubuntu --property=StandardOutput=append:/opt/wsp/logs/{job}.log "
             f"--property=StandardError=append:/opt/wsp/logs/{job}.log "
             f"/opt/wsp/.venv/bin/python /opt/wsp/worker.py {job}")
    c = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                         Parameters={"commands": [fetch, start], "executionTimeout": ["600"]})
    cid = c["Command"]["CommandId"]
    for _ in range(60):
        time.sleep(3)
        try:
            o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        if o["Status"] not in ("Pending", "InProgress", "Delayed"):
            if o["Status"] != "Success":
                raise RuntimeError(f"launch failed: {o['StandardErrorContent']}")
            log("worker started: " + o["StandardOutputContent"].strip() + o["StandardErrorContent"].strip())
            return
    raise RuntimeError("launch command did not finish")


def worker_alive(iid: str, job: str) -> tuple[bool, str]:
    c = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript", Parameters={"commands": [
        f"systemctl is-active --quiet wsp-job-{job} && echo ALIVE || echo DEAD; "
        f"systemctl status wsp-job-{job} --no-pager 2>&1 | tail -5; tail -c 3000 /opt/wsp/logs/{job}.log"]})
    cid = c["Command"]["CommandId"]
    for _ in range(30):
        time.sleep(2)
        try:
            o = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        if o["Status"] not in ("Pending", "InProgress", "Delayed"):
            out = o["StandardOutputContent"]
            return out.startswith("ALIVE"), out
    return True, "liveness check timed out"


def wait(iid: str, job: str) -> dict:
    seen, polls = 0, 0
    while True:
        polls += 1
        try:
            body = s3.get_object(Bucket=BUCKET, Key=f"jobs/{job}/progress.log")["Body"].read().decode()
            lines = body.splitlines()
            for l in lines[seen:]:
                print("  [worker] " + l, flush=True)
            seen = len(lines)
        except s3.exceptions.NoSuchKey:
            pass
        if s3_exists(f"jobs/{job}/status.json"):
            return json.loads(s3.get_object(Bucket=BUCKET, Key=f"jobs/{job}/status.json")["Body"].read())
        if polls % 4 == 0:  # every ~2 min: is the worker process still alive?
            alive, out = worker_alive(iid, job)
            if not alive and not s3_exists(f"jobs/{job}/status.json"):
                return {"state": "failed", "error": "worker process died without status.json:\n" + out}
        time.sleep(30)


def download(job: str) -> list[str]:
    got = []
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET, Prefix=f"jobs/{job}/results/"):
        for o in page.get("Contents", []):
            rel = o["Key"][len(f"jobs/{job}/results/"):]
            dest = WORK / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, o["Key"], str(dest))
            got.append(rel)
    return got


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    args = sys.argv[1:]
    keep = "--keep-running" in args
    kind = "pass"
    if "--final-heldout" in args:
        kind = "heldout"
        req = {"type": "generate", "model": "both", "heldout": True, "n": 4, "max_tokens": MAX_TOKENS_CAP}
    elif "--smoke" in args:
        kind = "smoke"
        paths = [a for a in args if not a.startswith("--")]
        if paths:
            req = json.loads(Path(paths[0]).read_text())
        else:
            model = "both" if any(EDIT_DIR.glob("*.safetensors")) else "original"
            req = {"type": "generate", "model": model, "problems": ["gsm8k-train-0000"], "n": 2, "max_tokens": 256,
                   "activations": {"layers": [0, 12, 24], "mlp": True}}
    else:
        path = [a for a in args if not a.startswith("--")][0]
        req = json.loads(Path(path).read_text())
    validate(req)
    b = budget()
    if kind == "pass" and b["used"] >= PASS_BUDGET:
        sys.exit(f"forward-pass budget exhausted ({PASS_BUDGET}/{PASS_BUDGET})")
    k = b["used"] + 1
    attempt = sum(1 for _ in open(LEDGER)) + 1 if LEDGER.exists() else 1
    job = {"pass": f"{k}", "heldout": "heldout", "smoke": "smoke"}[kind] + f"-a{attempt}"
    log(f"{kind} {k if kind == 'pass' else ''} job={job}: {json.dumps({kk: vv for kk, vv in req.items() if kk not in ('texts', 'chats', 'problems', 'prompts')})}")
    sums = upload_edits(job) if req["model"] in ("edited", "both") else {}
    s3.put_object(Bucket=BUCKET, Key=f"jobs/{job}/request.json", Body=json.dumps(req).encode())
    upload_code()
    t0 = time.time()
    iid = ensure_running()
    log(f"instance {iid} online; launching job")
    run_remote(iid, job)
    status = wait(iid, job)
    entry = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), kind=kind, job=job, request=req, edit_sha256=sums,
                 state=status["state"], minutes=round((time.time() - t0) / 60, 1))
    if status["state"] == "done":
        files = download(job)
        # results are written under the pass number k (gen_<k>..., fwd_<k>/)
        if kind == "pass":
            for rel in files:
                src = WORK / rel
                dst = WORK / rel.replace(f"gen_{job}", f"gen_{k}").replace(f"fwd_{job}", f"fwd_{k}")
                if src != dst:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    src.replace(dst)
            for d in sorted(WORK.glob(f"*{job}*"), reverse=True):
                if d.is_dir() and not any(d.rglob("*.*")):
                    d.rmdir()
            b["used"] = k
            b["passes"].append(dict(k=k, job=job, type=req["type"], model=req["model"]))
            BUDGET.write_text(json.dumps(b, indent=1))
        entry["summary"] = status.get("summary")
        log(f"DONE in {entry['minutes']} min. passes used: {b['used']}/{PASS_BUDGET}")
        print(json.dumps(status.get("summary"), indent=1))
        if kind == "heldout":
            (WORK / "heldout_result.json").write_text(json.dumps(status, indent=1))
    else:
        log("FAILED (does not count against the budget):\n" + status.get("error", "")[-4000:])
    ledger(entry)
    from wsp.common import sync_work_to_s3
    log(f"work/ mirrored to s3://{BUCKET}/work/ ({sync_work_to_s3(s3)} files updated)")
    if not keep:
        stop(iid)


if __name__ == "__main__":
    main()

"""Persistent analysis kernel (like the spec's notebook kernel).

  python wsp/kx.py start          # start kernel, run kernel_init
  python wsp/kx.py run < cell.py  # execute code; variables persist between runs
  python wsp/kx.py stop

Every executed cell and its full output are logged to work/cells/NNNN.{py,out}
(audit trail for the reviewer). Budget: 300 executions, 900 s each.
"""
from __future__ import annotations

import json
import queue
import sys
import time
from pathlib import Path

from jupyter_client import BlockingKernelClient, KernelManager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp.common import CELL_BUDGET, ROOT, WORK

CONN = WORK / ".kernel_connection.json"
CELLS = WORK / "cells"
TIMEOUT = 900


def _client() -> BlockingKernelClient:
    if not CONN.exists():
        sys.exit("kernel not running: python wsp/kx.py start")
    c = BlockingKernelClient(connection_file=str(CONN))
    c.load_connection_file()
    c.start_channels()
    c.wait_for_ready(timeout=30)
    return c


def _execute(c: BlockingKernelClient, code: str) -> tuple[str, bool]:
    msg_id = c.execute(code, store_history=True)
    out, ok, deadline = [], True, time.time() + TIMEOUT
    while True:
        try:
            msg = c.get_iopub_msg(timeout=max(1, deadline - time.time()))
        except queue.Empty:
            out.append(f"\n[kx] TIMEOUT after {TIMEOUT}s; interrupt the kernel with: python wsp/kx.py restart\n")
            return "".join(out), False
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        t, ct = msg["msg_type"], msg["content"]
        if t == "stream":
            out.append(ct["text"])
        elif t in ("execute_result", "display_data"):
            out.append(ct["data"].get("text/plain", "") + "\n")
        elif t == "error":
            ok = False
            out.append("\n".join(ct["traceback"]) + "\n")
        elif t == "status" and ct["execution_state"] == "idle":
            return "".join(out), ok


def start() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    km = KernelManager(kernel_name="python3", connection_file=str(CONN))
    km.start_kernel(cwd=str(ROOT), independent=True,
                    extra_arguments=["--colors=NoColor"])
    info = km.get_connection_info(session=False)
    info = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in info.items()}
    tmp = CONN.with_suffix(".keep")
    tmp.write_text(json.dumps(info))
    (WORK / ".kernel_pid").write_text(str(km.provisioner.process.pid))
    km.cleanup_connection_file = lambda: None  # keep the file after this process exits
    tmp.replace(CONN)
    c = _client()
    init = (ROOT / "wsp" / "kernel_init.py").read_text()
    out, ok = _execute(c, init)
    print(out)
    if not ok:
        sys.exit(1)


def run() -> None:
    code = sys.stdin.read()
    CELLS.mkdir(parents=True, exist_ok=True)
    n = max([int(p.stem) for p in CELLS.glob("*.py")] or [0])
    if n >= CELL_BUDGET:
        sys.exit(f"cell budget exhausted ({CELL_BUDGET})")
    stem = CELLS / f"{n + 1:04d}"
    stem.with_suffix(".py").write_text(code)
    c = _client()
    t0 = time.time()
    out, ok = _execute(c, code)
    stem.with_suffix(".out").write_text(out)
    print(out, end="")
    try:
        from wsp.common import sync_work_to_s3
        synced = sync_work_to_s3()
    except Exception as e:  # never lose a cell's output because the network hiccupped
        synced = f"S3 sync failed: {e}"
    print(f"\n[kx] cell {n + 1}/{CELL_BUDGET} {'ok' if ok else 'ERROR'} in {time.time() - t0:.1f}s; s3 sync: {synced}")


def stop() -> None:
    if CONN.exists():
        c = _client()
        c.shutdown()
        CONN.unlink()
    print("kernel stopped")


if __name__ == "__main__":
    {"start": start, "run": run, "stop": stop,
     "restart": lambda: (stop(), start())}[sys.argv[1]]()

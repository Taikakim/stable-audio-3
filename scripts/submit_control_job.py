#!/usr/bin/env python
"""submit_control_job.py — STDLIB-ONLY submitter for the file-drop control-eval queue
served by control_eval_server.py.

No numpy/torch/onnxruntime — runs in ANY venv (or the system python). It writes one
job into <queue-root>/inbox atomically, then polls <queue-root>/outbox for the
server's `<job_id>.done` (success) or `<job_id>.err` (failure) sentinel, printing the
result.json on success and raising on error.

    python scripts/submit_control_job.py \
        --prompt "goa trance, 145 bpm" --onset-density 11 --gain 3 --steps 8
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_COUNTER = 0


def _new_job_id() -> str:
    """timestamp + pid + monotonic counter → unique without uuid4. The 6-hex suffix is
    derived from pid/counter/ns so concurrent submitters from the same process and
    across processes don't collide, and the timestamp lead keeps inbox name-sort FIFO."""
    global _COUNTER
    _COUNTER += 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = (os.getpid() ^ (time.time_ns() & 0xFFFFFF) ^ (_COUNTER << 8)) & 0xFFFFFF
    return f"{ts}_{suffix:06x}"


def wait_for_server_ready(queue_root: Path, timeout: float = 600.0, poll: float = 0.5) -> bool:
    """Block until <queue-root>/.server_ready exists (the server touches it once its
    ONNX sessions + text conditioner are resident). Returns True if ready, False on
    timeout."""
    ready = queue_root / ".server_ready"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ready.exists():
            return True
        time.sleep(poll)
    return ready.exists()


def submit(queue_root: Path, job: dict, timeout: float = 600.0, poll: float = 0.5) -> dict:
    """Drop the job atomically into inbox/, then poll outbox/ for .done or .err."""
    job_id = job["job_id"]
    inbox, outbox = queue_root / "inbox", queue_root / "outbox"
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)

    # Atomic drop so the server never claims a half-written job file.
    tmp = inbox / f"{job_id}.job.json.tmp"
    tmp.write_text(json.dumps(job, indent=2))
    os.replace(tmp, inbox / f"{job_id}.job.json")

    done = outbox / f"{job_id}.done"
    err = outbox / f"{job_id}.err"
    result = outbox / f"{job_id}.result.json"

    deadline = time.time() + timeout
    while time.time() < deadline:
        if err.exists():
            payload = json.loads(err.read_text())
            raise RuntimeError(f"job {job_id} failed: {payload.get('error')}\n"
                               f"{payload.get('traceback', '')}")
        if done.exists():
            return json.loads(result.read_text())
        time.sleep(poll)
    raise TimeoutError(f"job {job_id} did not complete within {timeout}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--onset-density", required=True, type=float)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--queue-root", type=Path, default=Path("/home/kim/Projects/SAO/control_eval_queue"))
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    if not wait_for_server_ready(args.queue_root, timeout=args.timeout):
        print(f"[fatal] server not ready (no {args.queue_root}/.server_ready) — is it running?",
              file=sys.stderr)
        sys.exit(2)

    job = {
        "job_id": _new_job_id(),
        "prompt": args.prompt,
        "onset_density": args.onset_density,
        "gain": args.gain,
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "seed": args.seed,
    }
    if args.frames is not None:
        job["frames"] = args.frames
    if args.out_name is not None:
        job["out_name"] = args.out_name

    print(f"[submit] {job['job_id']}  prompt={args.prompt!r} onset={args.onset_density} "
          f"gain={args.gain} steps={args.steps}", flush=True)
    result = submit(args.queue_root, job, timeout=args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

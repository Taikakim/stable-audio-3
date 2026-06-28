#!/usr/bin/env python
"""submit_latch_job.py — STDLIB-ONLY submitter for the file-drop LatCH-eval queue
served by latch_eval_server.py.

No numpy/torch/onnxruntime — runs in ANY venv (or the system python). It writes one
job into <queue-root>/inbox atomically, then polls <queue-root>/outbox for the
server's `<job_id>.done` (success) or `<job_id>.err` (failure) sentinel, printing the
result.json on success and raising on error.

    python scripts/submit_latch_job.py \
        --head-ckpt latch_weights_sa3_medium/latch_sa3_spectral_skewness_best.pt \
        --feature spectral_skewness --target-low -2 --target-high 6 \
        --gain 64 --prompts "psytrance, 140 bpm" --steps 30
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_COUNTER = 0


def _new_job_id() -> str:
    """timestamp + pid + monotonic counter → unique without uuid4; timestamp lead keeps
    inbox name-sort FIFO."""
    global _COUNTER
    _COUNTER += 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = (os.getpid() ^ (time.time_ns() & 0xFFFFFF) ^ (_COUNTER << 8)) & 0xFFFFFF
    return f"{ts}_{suffix:06x}"


def wait_for_server_ready(queue_root: Path, timeout: float = 600.0, poll: float = 0.5) -> bool:
    """Block until <queue-root>/.server_ready exists (the server touches it once its
    ONNX sessions + text conditioner are resident). Returns True if ready."""
    ready = queue_root / ".server_ready"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ready.exists():
            return True
        time.sleep(poll)
    return ready.exists()


def submit(queue_root: Path, job: dict, timeout: float = 1800.0, poll: float = 0.5) -> dict:
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


def _parse_prompts(values) -> list:
    """Each --prompts occurrence is ONE verbatim prompt — NO comma-splitting, because
    musical prompts contain commas (e.g. 'goa trance, psychedelic, 145 bpm'). Repeat
    --prompts for multiple prompts."""
    return [v.strip() for v in values if v.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head-ckpt", required=True)
    ap.add_argument("--feature", required=True)
    ap.add_argument("--target-low", required=True, type=float)
    ap.add_argument("--target-high", required=True, type=float)
    ap.add_argument("--gain", type=float, default=64)
    ap.add_argument("--prompts", action="append", required=True,
                    help="one verbatim prompt per occurrence; repeat for multiple "
                         "(commas are kept, not split)")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=7)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--n-iter", type=int, default=4)
    ap.add_argument("--start-pct", type=float, default=0.4)
    ap.add_argument("--end-pct", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--queue-root", type=Path, default=Path("/home/kim/Projects/SAO/latch_eval_queue"))
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    prompts = _parse_prompts(args.prompts)
    if not prompts:
        print("[fatal] no prompts parsed from --prompts", file=sys.stderr)
        sys.exit(2)

    if not wait_for_server_ready(args.queue_root, timeout=args.timeout):
        print(f"[fatal] server not ready (no {args.queue_root}/.server_ready) — is it running?",
              file=sys.stderr)
        sys.exit(2)

    job = {
        "job_id": _new_job_id(),
        "head_ckpt": args.head_ckpt,
        "feature": args.feature,
        "target_low": args.target_low,
        "target_high": args.target_high,
        "gain": args.gain,
        "prompts": prompts,
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "gamma": args.gamma,
        "n_iter": args.n_iter,
        "start_pct": args.start_pct,
        "end_pct": args.end_pct,
    }
    if args.frames is not None:
        job["frames"] = args.frames

    print(f"[submit] {job['job_id']}  feature={args.feature} "
          f"targets=({args.target_low},{args.target_high}) gain={args.gain} "
          f"steps={args.steps} prompts={prompts}", flush=True)
    result = submit(args.queue_root, job, timeout=args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

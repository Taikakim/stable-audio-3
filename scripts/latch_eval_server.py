#!/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python
"""latch_eval_server.py — long-lived, ALL-CPU LatCH-GUIDED eval render server over the
PLAIN SA3 DiT ONNX, driven by a FILE-DROP queue (no HTTP).

The LatCH sibling of control_eval_server.py. Where the control server folds a trained
forward adapter into the DiT graph, this one runs the *plain* DiT ONNX forward-only
(numpy/ORT) and applies guidance by torch autograd through a tiny LatCH **head**
(sa3_latch_onnx.generate_z0_latch_guided — the two-stage variance+mean Selective-TFG
sampler). Everything is CPU: ORT CPUExecutionProvider for the DiT + SAME decoder, the
resident T5-Gemma text conditioner, and the head + autograd in torch fp32. The GPU is
never touched.

It imports the validated LatCH generation core (sa3_latch_onnx.py) so z0 matches the
cos-validation oracle for the same seed — schedule, CFG two-call loop, target
standardization, and criterion are single-sourced.

Queue layout (all under --queue-root, stdlib-only file ops)
-----------------------------------------------------------
    inbox/<job_id>.job.json       submitter drops a job here (written atomically)
    processing/<job_id>.job.json  claimed (os.rename inbox->processing; FIFO)
    outbox/<name>.wav             the renders (lo + hi per prompt)
    outbox/<job_id>.result.json   per-clip provenance + timings
    outbox/<job_id>.done          touched LAST — the submitter's success sentinel
    outbox/<job_id>.err           json+traceback on failure
    .server_ready                 touched once sessions+conditioner are resident

A claimed job is published ATOMICALLY: each wav + the result.json are written to `.tmp`
siblings and os.replace'd into place, and only THEN is `<job_id>.done` touched.

Usage (CPU EP; MIGraphX absent in the SA3 venv is EXPECTED — CPU is the target):
    cd /home/kim/Projects/SAO/stable-audio-3 && \
    .venv/bin/python scripts/latch_eval_server.py \
        --dit-onnx dit_medium-base_L256.onnx \
        --decoder-onnx same_decoder_L128.onnx --threads 12
"""
import argparse
import json
import os
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np

# CPU-ONLY: make_text_cond.load_conditioner loads with device="cpu" so the 1.4B weights never
# touch cuda (no OOM on a busy GPU); the GPU stays visible (the flash_attn/aiter import probes a
# driver at import time) but is never allocated on, and the DiT/decoder run on the ORT CPU EP.
# Single-source the validated LatCH generation math + the resident text conditioner loader.
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
import make_text_cond  # noqa: E402
import sa3_latch_onnx  # noqa: E402
from decode_onnx import decode_chunked_onnx  # noqa: E402
from sa3_control_onnx import DS, SR  # noqa: E402

SEQ = 128  # cross-attn token budget (matches the DiT export)


def _frames_from_dit(dit_session) -> int:
    """The DiT 'x' input is [1, LATENT_DIM, T]; the last dim is the length rung T. The
    server derives frames from the graph so the rung the export was compiled at is
    authoritative (a job that disagrees is rejected, not silently mis-rendered)."""
    for inp in dit_session.get_inputs():
        if inp.name == "x":
            return int(inp.shape[-1])
    raise SystemExit("[fatal] DiT graph has no input named 'x' — cannot derive frames")


def claim_next_job(inbox: Path, processing: Path):
    """FIFO: try to claim the oldest *.job.json by os.rename into processing/. The
    rename is the atomic lock — a loser gets FileNotFoundError and we try the next."""
    candidates = sorted(inbox.glob("*.job.json"))  # timestamp-led name → FIFO
    for src in candidates:
        dst = processing / src.name
        try:
            os.rename(src, dst)
            return dst
        except FileNotFoundError:
            continue  # lost the race; try the next one
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit-onnx", required=True, type=Path,
                    help="the PLAIN DiT onnx (e.g. dit_medium-base_L256.onnx)")
    ap.add_argument("--decoder-onnx", required=True, type=Path, help="SAME decoder onnx")
    ap.add_argument("--model", default="medium-base", help="SA3 model id for the text conditioner")
    ap.add_argument("--queue-root", type=Path, default=Path("/home/kim/Projects/SAO/latch_eval_queue"))
    ap.add_argument("--decode-chunk", type=int, default=128, help="MUST match the decoder export L")
    ap.add_argument("--decode-overlap", type=int, default=16, help=">= decoder receptive field")
    ap.add_argument("--threads", type=int, default=12,
                    help="CPU intra-op threads (physical cores; 12 ~25%% faster than 24 SMT on 9900X)")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between inbox polls when idle")
    ap.add_argument("--seconds", type=float, default=None,
                    help="seconds_total for the text conditioner (default round(frames*4096/44100,1))")
    args = ap.parse_args()

    import onnxruntime as ort

    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
    providers = ["CPUExecutionProvider"]  # ALL CPU — never touches the GPU.

    print(f"[ort] compiling plain DiT + decoder on CPU EP (threads={args.threads}) ...", flush=True)
    t0 = time.time()
    dit = ort.InferenceSession(str(args.dit_onnx), sess_options=so, providers=providers)
    dec = ort.InferenceSession(str(args.decoder_onnx), sess_options=so, providers=providers)
    active_ep = dit.get_providers()[0]
    print(f"[ort] sessions ready in {time.time() - t0:.0f}s  (DiT EP {active_ep})", flush=True)

    frames = _frames_from_dit(dit)  # authoritative length rung from the graph
    seconds = args.seconds if args.seconds is not None else round(frames * DS / SR, 1)
    print(f"[latch] frames={frames}  seconds={seconds}  seq={SEQ}", flush=True)

    print(f"[load] {args.model} text conditioner (T5-Gemma) on CPU ...", flush=True)
    t0 = time.time()
    cdm = make_text_cond.load_conditioner(args.model)
    print(f"[load] conditioner resident in {time.time() - t0:.0f}s", flush=True)

    qroot = args.queue_root
    inbox, processing, outbox = qroot / "inbox", qroot / "processing", qroot / "outbox"
    for d in (inbox, processing, outbox):
        d.mkdir(parents=True, exist_ok=True)
    (qroot / ".server_ready").touch()
    print(f"[serve] queue={qroot}  ready — polling every {args.poll}s", flush=True)

    # ---- per-prompt text cond/uncond cache (T5-Gemma is the slow part) ----
    text_cache: dict = {}

    def get_text_cond(prompt: str):
        key = (prompt, seconds, SEQ)
        if key not in text_cache:
            cond = make_text_cond.build_text_cond(cdm, prompt, seconds, seq=SEQ, T=frames)
            uncond = make_text_cond.build_text_cond(cdm, "", seconds, seq=SEQ, T=frames)
            text_cache[key] = (cond, uncond)
        return text_cache[key]

    # ---- LRU cache of loaded LatCH heads keyed by checkpoint path ----
    HEAD_CACHE_MAX = 8
    head_cache: "OrderedDict[str, tuple]" = OrderedDict()

    def get_head(head_ckpt: str):
        if head_ckpt in head_cache:
            head_cache.move_to_end(head_ckpt)
            return head_cache[head_ckpt]
        head, metadata = sa3_latch_onnx.load_latch_head(head_ckpt)
        loss_type = metadata.get("loss_type", "mse")
        criterion = sa3_latch_onnx.make_criterion(loss_type, metadata.get("huber_beta", 1.0))
        entry = (head, metadata, criterion)
        head_cache[head_ckpt] = entry
        head_cache.move_to_end(head_ckpt)
        while len(head_cache) > HEAD_CACHE_MAX:
            head_cache.popitem(last=False)
        return entry

    def publish_error(job_id: str, message: str):
        payload = {"job_id": job_id, "error": message, "traceback": traceback.format_exc()}
        tmp = outbox / f"{job_id}.err.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, outbox / f"{job_id}.err")
        print(f"[err] {job_id}: {message}", flush=True)

    while True:
        claimed = claim_next_job(inbox, processing)
        if claimed is None:
            time.sleep(args.poll)
            continue

        job_id = claimed.stem.replace(".job", "") or claimed.name
        try:
            job = json.loads(claimed.read_text())
            job_id = job.get("job_id", job_id)

            missing = [f for f in ("head_ckpt", "feature", "target_low", "target_high", "prompts")
                       if f not in job]
            if missing:
                publish_error(job_id, f"missing required field(s): {missing}")
                continue

            if "frames" in job and int(job["frames"]) != frames:
                publish_error(job_id, f"frames mismatch: job asked {int(job['frames'])} "
                                      f"but server rung is {frames} (DiT export is authoritative)")
                continue

            head_ckpt = job["head_ckpt"]
            feature = job["feature"]
            target_low = float(job["target_low"])
            target_high = float(job["target_high"])
            prompts = job["prompts"]
            if isinstance(prompts, str):
                prompts = [prompts]
            gain = float(job.get("gain", 64))
            seed = int(job.get("seed", 777))
            steps = int(job.get("steps", 30))
            cfg = float(job.get("cfg", 7))
            gamma = float(job.get("gamma", 0.3))
            n_iter = int(job.get("n_iter", 4))
            start_pct = float(job.get("start_pct", 0.4))
            end_pct = float(job.get("end_pct", 1.0))

            print(f"[job] {job_id}  feature={feature} head={Path(head_ckpt).name} "
                  f"targets=({target_low},{target_high}) gain={gain} steps={steps} cfg={cfg} "
                  f"seed={seed} prompts={len(prompts)}", flush=True)

            head, metadata, criterion = get_head(head_ckpt)

            import soundfile as sf

            t_total = time.time()
            clips = []
            for p_idx, prompt in enumerate(prompts):
                cond, uncond = get_text_cond(prompt)
                for tag, raw in (("lo", target_low), ("hi", target_high)):
                    target_std = sa3_latch_onnx.make_latch_target(raw, metadata, frames)
                    t_clip = time.time()
                    gen = sa3_latch_onnx.generate_z0_latch_guided(
                        dit, cond=cond, uncond=uncond,
                        guides=[{"head": head, "target": target_std, "weight": 1.0,
                                 "criterion": criterion}],
                        frames=frames, steps=steps, cfg_scale=cfg, seed=seed,
                        rho=gain, mu=gain, gamma=gamma, n_iter=n_iter,
                        start_pct=start_pct, end_pct=end_pct)
                    z0 = gen["z0"]

                    t_dec = time.time()
                    audio = decode_chunked_onnx(dec, z0, args.decode_chunk, args.decode_overlap, DS)
                    decode_s = time.time() - t_dec

                    wav_name = f"{job_id}_{feature}_{p_idx}_{tag}.wav"
                    wav_tmp = outbox / (wav_name + ".tmp")
                    a = np.clip(audio[0], -1.0, 1.0).T
                    # format must be explicit: the .tmp suffix hides the .wav extension sf infers.
                    sf.write(str(wav_tmp), a, SR, subtype="PCM_16", format="WAV")
                    os.replace(wav_tmp, outbox / wav_name)

                    clips.append({
                        "prompt": prompt,
                        "prompt_idx": p_idx,
                        "tag": tag,
                        "target_raw": raw,
                        "target_std": float(target_std.flatten()[0]),
                        "gain": gain,
                        "wav": str(outbox / wav_name),
                        "timings": {
                            "dit_loop_s": round(gen["dit_loop_s"], 3),
                            "decode_s": round(decode_s, 3),
                        },
                    })
                    print(f"  [clip] p{p_idx} {tag} target_raw={raw} "
                          f"dit={gen['dit_loop_s']:.1f}s decode={decode_s:.1f}s "
                          f"-> {wav_name}", flush=True)

            total_s = time.time() - t_total
            result = {
                "job_id": job_id,
                "feature": feature,
                "head_ckpt": head_ckpt,
                "head_metadata": {
                    "out_channels": int(metadata.get("out_channels", 1)),
                    "loss_type": metadata.get("loss_type", "mse"),
                    "standardized": bool(metadata.get("standardized", False)),
                    "std_mean": float(metadata.get("std_mean", 0.0)),
                    "std_std": float(metadata.get("std_std", 1.0)),
                },
                "gain": gain,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "gamma": gamma,
                "n_iter": n_iter,
                "start_pct": start_pct,
                "end_pct": end_pct,
                "frames": frames,
                "seconds": seconds,
                "active_ep": active_ep,
                "clips": clips,
                "timings": {"total_s": round(total_s, 3)},
            }
            res_tmp = outbox / f"{job_id}.result.json.tmp"
            res_tmp.write_text(json.dumps(result, indent=2))

            # Atomic publish: result.json into place, THEN touch .done last.
            os.replace(res_tmp, outbox / f"{job_id}.result.json")
            (outbox / f"{job_id}.done").touch()
            print(f"[done] {job_id}  {len(clips)} clips  total={total_s:.1f}s", flush=True)
        except Exception as e:
            publish_error(job_id, f"{type(e).__name__}: {e}")
        finally:
            Path(claimed).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

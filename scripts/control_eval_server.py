#!/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python
"""control_eval_server.py — long-lived, ALL-CPU control-eval render server for the
SA3 control-DiT, driven by a FILE-DROP queue (no HTTP).

The sibling of latent_server_dit_onnx.py (HTTP + MIGraphX GPU). This server is the
**CPU eval path** recommended in MASTER.md §5 ("CPU-ONLY is the recommended eval
path — frees the GPU for training"): it compiles the control-DiT + SAME decoder
ONNX once on the CPU EP, loads the resident T5-Gemma text conditioner and the scalar
FiLM control conditioner, then watches a filesystem queue for jobs. Everything stays
resident between jobs so each render is just the sampler loop + decode.

It imports the *validated* generation core (sa3_control_onnx.py) so z0 is bit-exact
with dit_control_onnx_infer.py for the same seed — the schedule, CFG two-call loop,
host-side fractional PE, and the scalar→tokens FiLM port are single-sourced.

Queue layout (all under --queue-root, stdlib-only file ops)
-----------------------------------------------------------
    inbox/<job_id>.job.json       submitter drops a job here (written atomically)
    processing/<job_id>.job.json  claimed (os.rename inbox->processing; FIFO)
    outbox/<name>.wav             the render
    outbox/<job_id>.result.json   timings + provenance
    outbox/<job_id>.done          touched LAST — the submitter's success sentinel
    outbox/<job_id>.err           json+traceback on failure
    .server_ready                 touched once sessions+conditioner are resident

A claimed job is published ATOMICALLY: wav + result.json are written to `.tmp`
siblings and os.replace'd into place, and only THEN is `<job_id>.done` touched — so a
submitter that sees `.done` is guaranteed both files are complete.

Usage (CPU EP; MIGraphX absent in the SA3 venv is EXPECTED — CPU is the target):
    cd /home/kim/Projects/SAO/stable-audio-3 && \
    .venv/bin/python scripts/control_eval_server.py \
        --dit-onnx dit_medium-base_L256_ctrl_onset_density_fp16.onnx \
        --cond-npz dit_medium-base_L256_ctrl_onset_density.cond.npz \
        --decoder-onnx same_decoder_L128_fp16.onnx --threads 12
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Single-source the validated generation math + the resident text conditioner loader.
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
import make_text_cond  # noqa: E402
import sa3_control_onnx  # noqa: E402
from decode_onnx import decode_chunked_onnx  # noqa: E402
from sa3_control_onnx import (  # noqa: E402
    DS, SR, generate_z0, make_control_tokens, resolve_host_pe,
)


def _frames_from_dit(dit_session) -> int:
    """The DiT 'x' input is [1, LATENT_DIM, T]; the last dim is the length rung T. The
    server derives frames from the graph so the rung the export was compiled at is
    authoritative (a job that disagrees is rejected, not silently mis-rendered)."""
    for inp in dit_session.get_inputs():
        if inp.name == "x":
            return int(inp.shape[-1])
    raise SystemExit("[fatal] DiT graph has no input named 'x' — cannot derive frames")


def claim_next_job(inbox: Path, processing: Path) -> Path | None:
    """FIFO: try to claim the oldest *.job.json by os.rename into processing/. The
    rename is the atomic lock — a loser (someone else grabbed it first) gets
    FileNotFoundError and we move to the next candidate. Returns the claimed path in
    processing/, or None if the inbox is empty."""
    candidates = sorted(inbox.glob("*.job.json"))  # job_id is timestamp-led → name sort == FIFO
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
    ap.add_argument("--dit-onnx", required=True, type=Path, help="the CONTROL DiT onnx")
    ap.add_argument("--cond-npz", required=True, type=Path,
                    help="the scalar FiLM .cond.npz beside the onnx")
    ap.add_argument("--decoder-onnx", required=True, type=Path, help="SAME decoder onnx")
    ap.add_argument("--model", default="medium-base", help="SA3 model id for the text conditioner")
    ap.add_argument("--queue-root", type=Path, default=Path("/home/kim/Projects/SAO/control_eval_queue"))
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

    print(f"[ort] compiling control-DiT + decoder on CPU EP (threads={args.threads}) ...", flush=True)
    t0 = time.time()
    dit = ort.InferenceSession(str(args.dit_onnx), sess_options=so, providers=providers)
    dec = ort.InferenceSession(str(args.decoder_onnx), sess_options=so, providers=providers)
    active_ep = dit.get_providers()[0]
    print(f"[ort] sessions ready in {time.time() - t0:.0f}s  (DiT EP {active_ep})", flush=True)

    film_z = np.load(args.cond_npz)
    host_pe = resolve_host_pe(dit, film_z)              # asserts onnx/npz PE stamps agree
    frames = _frames_from_dit(dit)                       # authoritative length rung from the graph
    seconds = args.seconds if args.seconds is not None else round(frames * DS / SR, 1)
    print(f"[ctrl] field={str(film_z['field'])}  frames={frames}  seconds={seconds}  "
          f"host_pe={host_pe}  mean={float(film_z['mean']):.3f} std={float(film_z['std']):.3f}", flush=True)

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

    # Per-(prompt, seconds, seq) text cond/uncond cache: build_text_cond runs T5-Gemma,
    # so reuse across jobs that share a prompt.
    SEQ = 128
    text_cache: dict = {}

    def get_text_cond(prompt: str):
        key = (prompt, seconds, SEQ)
        if key not in text_cache:
            cond = make_text_cond.build_text_cond(cdm, prompt, seconds, seq=SEQ, T=frames)
            uncond = make_text_cond.build_text_cond(cdm, "", seconds, seq=SEQ, T=frames)
            text_cache[key] = (cond, uncond)
        return text_cache[key]

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

            missing = [f for f in ("job_id", "prompt", "onset_density") if f not in job]
            if missing:
                publish_error(job_id, f"missing required field(s): {missing}")
                continue

            if "frames" in job and int(job["frames"]) != frames:
                publish_error(job_id, f"frames mismatch: job asked {int(job['frames'])} "
                                      f"but server rung is {frames} (DiT export is authoritative)")
                continue

            prompt = job["prompt"]
            onset_density = float(job["onset_density"])
            steps = int(job.get("steps", 8))
            cfg_scale = float(job.get("cfg_scale", 6.0))
            seed = int(job.get("seed", 42))
            gain = float(job.get("gain", 1.0))
            out_name = job.get("out_name")

            print(f"[job] {job_id}  prompt={prompt!r}  onset={onset_density} gain={gain} "
                  f"steps={steps} cfg={cfg_scale} seed={seed}", flush=True)

            t_total = time.time()
            cond, uncond = get_text_cond(prompt)
            cond_tok, zero_tok = make_control_tokens(film_z, onset_density, host_pe)
            gen = generate_z0(dit, cond=cond, uncond=uncond, cond_tok=cond_tok,
                              zero_tok=zero_tok, frames=frames, steps=steps,
                              cfg_scale=cfg_scale, seed=seed, gain=gain)
            z0 = gen["z0"]

            t_dec = time.time()
            audio = decode_chunked_onnx(dec, z0, args.decode_chunk, args.decode_overlap, DS)
            decode_s = time.time() - t_dec
            total_s = time.time() - t_total

            import soundfile as sf
            wav_stem = out_name or job_id
            wav_name = wav_stem if str(wav_stem).endswith(".wav") else f"{wav_stem}.wav"
            wav_path = outbox / wav_name
            a = np.clip(audio[0], -1.0, 1.0).T
            wav_tmp = outbox / (wav_name + ".tmp")
            # format must be explicit: the .tmp suffix hides the .wav extension sf infers from.
            sf.write(str(wav_tmp), a, SR, subtype="PCM_16", format="WAV")

            result = {
                "job_id": job_id,
                "prompt": prompt,
                "onset_target": onset_density,
                "gain": gain,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "frames": frames,
                "seconds": round(audio.shape[-1] / SR, 2),
                "active_ep": active_ep,
                "timings": {
                    "dit_loop_s": round(gen["dit_loop_s"], 3),
                    "decode_s": round(decode_s, 3),
                    "total_s": round(total_s, 3),
                },
                "paths": {"wav": str(wav_path), "result": str(outbox / f"{job_id}.result.json")},
            }
            res_tmp = outbox / f"{job_id}.result.json.tmp"
            res_tmp.write_text(json.dumps(result, indent=2))

            # Atomic publish: move both artifacts into place, THEN touch .done last.
            os.replace(wav_tmp, wav_path)
            os.replace(res_tmp, outbox / f"{job_id}.result.json")
            (outbox / f"{job_id}.done").touch()
            print(f"[done] {job_id}  dit={result['timings']['dit_loop_s']}s "
                  f"decode={result['timings']['decode_s']}s total={result['timings']['total_s']}s "
                  f"-> {wav_path}", flush=True)
        except Exception as e:
            publish_error(job_id, f"{type(e).__name__}: {e}")
        finally:
            Path(claimed).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

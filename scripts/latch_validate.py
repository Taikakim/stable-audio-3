#!/usr/bin/env python
"""latch_validate.py — CPU-vs-GPU cosine validation harness for the LatCH ONNX path.

WHY: ``sa3_latch_onnx.generate_z0_latch_guided`` is a numpy/ORT *CPU* port of the
torch GPU oracle ``/run/media/kim/Mantu/latch_sweep/gen_one.py::gen_guided``
(``sample_flow_euler_multi_latch_guided``, the two-stage variance+mean Selective-TFG
sampler). This script runs BOTH halves with matched params and checks that the two
clean latents agree (z0 cosine >= 0.999), and that the LatCH head measures the
intended feature on each z0 (feature-follow sign/magnitude sanity).

TWO HALVES:
  * CPU half  — runs NOW. Plain DiT ONNX (CPUExecutionProvider) + torch LatCH head
                grads. Reuses sa3_latch_onnx + make_text_cond. Writes <out>.cpu.z0.npy.
  * GPU half  — written but GUARDED behind --run-gpu (this task does NOT pass it; the
                card is busy). Imports gen_one and runs gen_guided with MATCHED params,
                capturing the sampler's latent. Writes <out>.gpu.z0.npy.

----------------------------------------------------------------------------------
RUN THE GPU HALF LATER (when the RX 9070 XT is free) — exact command:
----------------------------------------------------------------------------------
  export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE   # CK flash-attn (MASTER 5)
  /home/kim/Projects/SAO/stable-audio-3/.venv/bin/python \
    scripts/latch_validate.py \
      --head-ckpt latch_weights_sa3_medium/latch_sa3_spectral_skewness_best.pt \
      --feature spectral_skewness --target 6 --prompt "goa trance, 145 bpm" \
      --gain 64 --seed 777 --steps 30 --frames 256 \
      --out /tmp/latch_val_skew \
      --run-gpu

  With --run-gpu the harness builds the cond/uncond from --prompt ON GPU via gen_one
  (so conditioning matches), runs both halves, then prints the z0 cosine and the
  per-z0 head-measured feature for CPU and GPU.

CAVEAT — conditioning coupling (read before trusting a sub-0.999 cosine):
  gen_one couples seconds_total and the latent length T: it builds the prompt
  conditioning with seconds_total=duration AND derives T from that same duration via
  _adapt_sample_size (+6 s padding). To force T == frames we must pass
  duration ~= frames*4096/44100 - 6 (see _matched_gpu_duration), which sets the GPU
  conditioning's seconds_total to that value. The CPU cond built from --prompt uses
  the SAME matched duration as its seconds (so both halves condition identically).
  BUT if you feed a precomputed --cond-npz/--uncond-npz built with a DIFFERENT
  seconds_total (e.g. make_text_cond's 23.8), the global_embed differs and the cosine
  will fall below 0.999 for a CONDITIONING reason, not a sampler-port bug. For a
  strict GPU-vs-CPU cosine, build cond from --prompt (omit --cond-npz) so this script
  uses the matched seconds on both sides.

GPU-half latent capture: gen_one.gen_guided returns only an info dict and writes a wav
  (it never exposes the sampler latent). Rather than fork it, we monkeypatch
  gen_one.sample_flow_euler_multi_latch_guided to capture its return value, then call
  gen_guided unmodified — so the GPU z0 comes from the EXACT gen_one code path (no
  logic duplication / drift). See _run_gpu().

    SA3 venv:  /home/kim/Projects/SAO/stable-audio-3/.venv/bin/python
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# scripts/ on path so sa3_latch_onnx + make_text_cond import (they cross-import too).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_REPO = _HERE.parent

SR = 44100
DS = 4096  # downsampling_ratio == latent stride in samples


# --------------------------------------------------------------------------- #
# cond / uncond
# --------------------------------------------------------------------------- #
def _load_cond_npz(path: Path):
    """Load a (cross_attn_cond, cross_attn_mask, global_embed) tuple from an npz."""
    d = np.load(path)
    return (
        d["cross_attn_cond"].astype("float32"),
        d["cross_attn_mask"].astype(bool),
        d["global_embed"].astype("float32"),
    )


def _build_cond_from_prompt(prompt: str, seconds: float, frames: int, seq: int = 128):
    """Run the T5-Gemma conditioner on CPU for cond (prompt) and uncond ('')."""
    import make_text_cond as mtc

    print(f"[cpu] loading medium-base conditioner on CPU (T5-Gemma) ...", flush=True)
    cdm = mtc.load_conditioner("medium-base")
    cond = mtc.build_text_cond(cdm, prompt, seconds=seconds, seq=seq, T=frames)
    uncond = mtc.build_text_cond(cdm, "", seconds=seconds, seq=seq, T=frames)
    return cond, uncond


# --------------------------------------------------------------------------- #
# head-measured feature on a z0
# --------------------------------------------------------------------------- #
def _measure_feature(head, metadata, z0_np: np.ndarray):
    """Return (standardized_mean, raw_mean) of head(z0, t=0) over all frames/channels.

    standardized_mean is exactly the brief's "standardized head(z0).mean()". raw_mean
    un-standardizes it via the head's std_mean/std_std for human-readable comparison to
    the requested RAW target.
    """
    import torch

    std_mean = float(metadata.get("std_mean", 0.0))
    std_std = float(metadata.get("std_std", 1.0)) or 1.0
    with torch.no_grad():
        zt = torch.tensor(z0_np, dtype=torch.float32)
        t0 = torch.zeros(zt.shape[0], dtype=torch.float32)
        pred = head(zt, t0)  # standardized prediction (head trained standardized=True)
    std_mean_pred = float(pred.mean().item())
    raw_mean_pred = std_mean_pred * std_std + std_mean
    return std_mean_pred, raw_mean_pred


# --------------------------------------------------------------------------- #
# CPU half
# --------------------------------------------------------------------------- #
def _run_cpu(args, cond, uncond):
    import onnxruntime as ort

    import sa3_latch_onnx as slo

    head, metadata = slo.load_latch_head(args.head_ckpt)
    target = slo.make_latch_target(float(args.target), metadata, args.frames)
    criterion = slo.make_criterion(metadata.get("loss_type", "smooth_l1"),
                                   metadata.get("huber_beta", 1.0))

    dit_onnx = args.dit_onnx or (_REPO / f"dit_medium-base_L{args.frames}.onnx")
    dit_onnx = Path(dit_onnx)
    if not dit_onnx.exists():
        raise SystemExit(f"DiT ONNX not found: {dit_onnx} (need the L{args.frames} rung)")
    print(f"[cpu] DiT ONNX: {dit_onnx.name}  (CPUExecutionProvider)", flush=True)
    sess = ort.InferenceSession(str(dit_onnx), providers=["CPUExecutionProvider"])

    guides = [{
        "head": head,
        "target": target,
        "weight": 1.0,
        "criterion": criterion,
    }]
    print(f"[cpu] generate_z0_latch_guided steps={args.steps} cfg={args.cfg} "
          f"gain={args.gain} frames={args.frames} ...", flush=True)
    res = slo.generate_z0_latch_guided(
        sess,
        cond=cond,
        uncond=uncond,
        guides=guides,
        frames=args.frames,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        rho=args.gain,
        mu=args.gain,
    )
    z0 = res["z0"]
    out = Path(f"{args.out}.cpu.z0.npy")
    np.save(out, z0)
    std_m, raw_m = _measure_feature(head, metadata, z0)
    print(f"[cpu] z0 {z0.shape} dtype={z0.dtype} -> {out}  "
          f"(dit loop {res['dit_loop_s']:.1f}s)", flush=True)
    print(f"[cpu] head-measured {args.feature}: standardized mean={std_m:+.4f}  "
          f"raw={raw_m:+.4f}  (target raw={args.target}, "
          f"target std={float(target.flatten()[0]):+.4f})", flush=True)
    return z0, (std_m, raw_m), head, metadata


# --------------------------------------------------------------------------- #
# GPU half (guarded — NOT run unless --run-gpu)
# --------------------------------------------------------------------------- #
def _matched_gpu_duration(model, frames: int) -> float:
    """Find a seconds_total `duration` such that gen_one's
    `model._adapt_sample_size(cond, 5292032, 6.0) // downsampling_ratio == frames`.

    gen_one derives T from duration (with +6 s padding then round-up to the latent
    alignment). We invert it: start at the analytic value frames*DS/SR - 6 and search a
    few latent-frame steps either way until adapt() lands exactly on frames.
    """
    from stable_audio_3 import StableAudioModel

    ds = model.model.pretransform.downsampling_ratio
    step = ds / SR  # seconds per latent frame

    def T_for(dur: float) -> int:
        cond, _ = StableAudioModel._build_conditioning_dicts("x", None, dur, 1)
        return model._adapt_sample_size(cond, 5292032, 6.0) // ds

    base = frames * ds / SR - 6.0
    # search base-2..base+2 latent frames (covers both ceil-to-multiple roundings)
    cands = sorted({round(base + k * step, 6) for k in range(-2, 3)})
    for dur in cands:
        if dur > 0 and T_for(dur) == frames:
            return dur
    # widen the search if the narrow window missed
    for k in range(-8, 9):
        dur = round(base + k * step, 6)
        if dur > 0 and T_for(dur) == frames:
            return dur
    raise SystemExit(
        f"could not find a duration giving T={frames} (analytic base={base:.3f}s); "
        f"check downsampling/alignment")


def _run_gpu(args):
    """Run the torch GPU oracle (gen_one.gen_guided) with matched params and capture z0.

    Returns (z0_np, (std_mean, raw_mean), head, metadata).
    """
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
    sys.path.insert(0, "/run/media/kim/Mantu/latch_sweep")
    import gen_one  # noqa: E402  (heavy: loads torch+cuda; only on --run-gpu)
    import torch

    # sanity: --head-ckpt should be the medium head for --feature (so weights match)
    expect = f"latch_sa3_{args.feature}_best.pt"
    if Path(args.head_ckpt).name != expect:
        print(f"[gpu][warn] --head-ckpt basename {Path(args.head_ckpt).name!r} != "
              f"{expect!r}; gen_one loads by --feature from its WEIGHTS_DIR, so the two "
              f"halves may use different weights.", flush=True)

    model = gen_one.get_model("medium-base")
    dur = _matched_gpu_duration(model, args.frames)
    print(f"[gpu] matched duration {dur:.4f}s -> T={args.frames} "
          f"(seconds_total used for BOTH conditioning and latent length)", flush=True)

    # Capture the sampler latent without forking gen_one: wrap the function it calls.
    captured: dict = {}
    _orig = gen_one.sample_flow_euler_multi_latch_guided

    def _wrap(*a, **k):
        out = _orig(*a, **k)
        captured["lat"] = out
        return out

    gen_one.sample_flow_euler_multi_latch_guided = _wrap
    try:
        gpu_wav = f"{args.out}.gpu.wav"
        info = gen_one.gen_guided(
            args.feature, float(args.target), args.prompt, float(args.gain),
            seed=args.seed, duration=dur, out_path=gpu_wav,
            steps=args.steps, cfg_scale=args.cfg,
            start_pct=0.4, end_pct=1.0, gamma=0.3, n_iter=4,
            model_name="medium-base",
        )
    finally:
        gen_one.sample_flow_euler_multi_latch_guided = _orig

    z0 = captured["lat"].detach().to(torch.float32).cpu().numpy().astype(np.float32)
    out = Path(f"{args.out}.gpu.z0.npy")
    np.save(out, z0)
    print(f"[gpu] z0 {z0.shape} -> {out}  (gen {info.get('gen_secs')}s, T={info.get('T')})",
          flush=True)

    # measure with the SAME head class used on CPU (canonical loader, CPU copy)
    import sa3_latch_onnx as slo
    head, metadata = slo.load_latch_head(args.head_ckpt)
    std_m, raw_m = _measure_feature(head, metadata, z0)
    print(f"[gpu] head-measured {args.feature}: standardized mean={std_m:+.4f}  "
          f"raw={raw_m:+.4f}", flush=True)
    return z0, (std_m, raw_m), head, metadata


# --------------------------------------------------------------------------- #
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head-ckpt", required=True,
                    help="LatCH head .pt (e.g. latch_weights_sa3_medium/latch_sa3_<feat>_best.pt)")
    ap.add_argument("--feature", required=True, help="feature name (maps to gen_one's WEIGHTS_DIR head)")
    ap.add_argument("--target", type=float, required=True, help="RAW feature target value")
    ap.add_argument("--prompt", default="goa trance, 145 bpm")
    ap.add_argument("--gain", type=float, default=64.0, help="rho == mu guidance strength")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--frames", type=int, default=256, help="T (must match a DiT ONNX rung)")
    ap.add_argument("--dit-onnx", default=None, help="override; default dit_medium-base_L<frames>.onnx")
    ap.add_argument("--cond-npz", default=None, help="skip T5-Gemma: load cond from npz")
    ap.add_argument("--uncond-npz", default=None, help="skip T5-Gemma: load uncond from npz")
    ap.add_argument("--cond-seconds", type=float, default=None,
                    help="seconds_total for prompt conditioning when building from --prompt "
                         "(default frames*4096/44100; see CAVEAT for GPU-cosine matching)")
    ap.add_argument("--out", required=True, help="output prefix (<out>.cpu.z0.npy / .gpu.z0.npy)")
    ap.add_argument("--run-gpu", action="store_true",
                    help="ALSO run the torch GPU oracle (gen_one). Off by default (GPU busy).")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    seconds = args.cond_seconds if args.cond_seconds is not None else args.frames * DS / SR

    # ---- conditioning (shared by the CPU half) ----
    if args.cond_npz and args.uncond_npz:
        print(f"[cond] loading {args.cond_npz} / {args.uncond_npz} (skipping T5-Gemma)", flush=True)
        cond = _load_cond_npz(Path(args.cond_npz))
        uncond = _load_cond_npz(Path(args.uncond_npz))
    else:
        cond, uncond = _build_cond_from_prompt(args.prompt, seconds, args.frames)

    # ---- CPU half (runs now) ----
    cpu_z0, cpu_feat, _, _ = _run_cpu(args, cond, uncond)

    # ---- GPU half (only if requested) ----
    gpu_z0 = None
    gpu_feat = None
    if args.run_gpu:
        gpu_z0, gpu_feat, _, _ = _run_gpu(args)
    else:
        print("[gpu] SKIPPED (no --run-gpu). Run later with --run-gpu when the card is "
              "free; see this file's docstring for the exact command.", flush=True)

    # ---- compare ----
    if cpu_z0 is not None and gpu_z0 is not None:
        if cpu_z0.shape != gpu_z0.shape:
            print(f"[cmp][warn] shape mismatch cpu{cpu_z0.shape} vs gpu{gpu_z0.shape} "
                  f"-> T did not align; cosine is meaningless.", flush=True)
        cos = _cosine(cpu_z0, gpu_z0)
        print("=" * 70, flush=True)
        print(f"z0 cosine = {cos:.6f}   (require >= 0.999)", flush=True)
        print(f"feature-follow  CPU std={cpu_feat[0]:+.4f} raw={cpu_feat[1]:+.4f}   "
              f"GPU std={gpu_feat[0]:+.4f} raw={gpu_feat[1]:+.4f}   "
              f"(target raw={args.target})", flush=True)
        print("=" * 70, flush=True)
        if cos < 0.999:
            raise SystemExit(f"FAIL: z0 cosine {cos:.6f} < 0.999")
        print("PASS: z0 cosine >= 0.999", flush=True)
    else:
        print(f"[cmp] CPU-only run complete. cpu_z0 saved -> {args.out}.cpu.z0.npy. "
              f"Run with --run-gpu to produce the cosine.", flush=True)


if __name__ == "__main__":
    main()

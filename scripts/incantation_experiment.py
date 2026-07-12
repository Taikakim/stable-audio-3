#!/usr/bin/env python3
"""Incantation-mask experiment: does barring the prompt's cross-attn from the clamped
history reduce the loop attractor in sliding-window continuation?

Baseline vs masked, same seed/prompt-arc/window geometry. Measures a self-similarity
"loopiness" on the assembled latents (high = frames repeat earlier content = looping) and
decodes both to audio for the ear. See incantation_mask.py + the long-form-coherence triage.

Run (SA3 venv, GPU):
  .venv/bin/python scripts/incantation_experiment.py --out-dir <dir> [--duration 120 --window-sec 30 --overlap-sec 5]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def loopiness(latents, fps, guard_sec=6.0):
    """Per-frame max cosine-similarity to any earlier NON-adjacent frame (high => repeats
    earlier material = loop). Returns (mean, p90, early_mean, late_mean) — early vs late
    detects a DEGENERATING loop (climbs over the rollout)."""
    x = latents.float().squeeze(0)                       # (C, T)
    x = x / (x.norm(dim=0, keepdim=True) + 1e-8)         # frame-normalize
    S = (x.T @ x).cpu().numpy()                          # (T, T) cosine sim
    T = S.shape[0]
    guard = int(guard_sec * fps)
    idx, vals = [], []
    for i in range(guard + 1, T):
        j_hi = i - guard
        if j_hi <= 0:
            continue
        idx.append(i); vals.append(float(S[i, :j_hi].max()))
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan")
    vals = np.array(vals)
    third = len(vals) // 3
    early = float(vals[:third].mean()) if third else float("nan")
    late = float(vals[-third:].mean()) if third else float("nan")
    return float(vals.mean()), float(np.percentile(vals, 90)), early, late


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--overlap-sec", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--single", action="store_true",
                    help="single static prompt (strongest loop-inducing case) instead of a 2-section arc")
    ap.add_argument("--intervention", choices=["incantation", "rope_jitter"], default="incantation",
                    help="which training-free fix to test against baseline")
    ap.add_argument("--jitter-scale", type=float, default=0.08)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from stable_audio_3 import StableAudioModel
    from stable_audio_3.inference.longform import (
        PromptSchedule, InpaintContinuationGenerator, LongFormRenderer)
    from stable_audio_3.inference.incantation_mask import incantation_mask
    from stable_audio_3.inference.rope_jitter import rope_jitter
    import contextlib

    m = StableAudioModel.from_pretrained(args.model, model_half=args.half)
    inner = m.model
    sr = inner.sample_rate
    fps = sr / inner.pretransform.downsampling_ratio
    dit = inner.model  # the DiT carrying the cross-attn blocks
    f = lambda s: int(round(s * fps))

    # single static prompt = strongest loop-inducing case; else a 2-section arc
    if args.single:
        arc = [(0.0, "dark hypnotic goa trance, rolling bassline, 145 bpm")]
    else:
        arc = [(0.0, "dark hypnotic goa trance, rolling bassline, 145 bpm"),
               (args.duration / 2, "euphoric melodic goa trance breakdown, bright leads, 145 bpm")]

    def run(intervention):
        torch.manual_seed(args.seed)
        gen = InpaintContinuationGenerator(m, steps=args.steps, cfg_scale=args.cfg)
        r = LongFormRenderer(gen, channels=inner.io_channels, fps=fps,
                             window_frames=f(args.window_sec), overlap_frames=f(args.overlap_sec))
        sched = PromptSchedule(arc)
        if intervention == "incantation":
            ctx = incantation_mask(dit, gen)
        elif intervention == "rope_jitter":
            ctx = rope_jitter(dit, scale=args.jitter_scale, seed=args.seed)
        else:
            ctx = contextlib.nullcontext()
        with ctx as info:
            if intervention:
                print(f"  [{intervention} ON] {info}", flush=True)
            lat = r.render_latents(sched, total_frames=f(args.duration), base_seed=args.seed)
        return lat, r.drift_log

    results = {}
    for intervention in (None, args.intervention):
        tag = intervention or "baseline"
        print(f"[run] {tag} ...", flush=True)
        lat, drift = run(intervention)
        mean_loop, p90_loop, early, late = loopiness(lat, fps)
        results[tag] = {"loopiness_mean": round(mean_loop, 4), "loopiness_p90": round(p90_loop, 4),
                        "loopiness_early": round(early, 4), "loopiness_late": round(late, 4),
                        "n_frames": int(lat.shape[-1]),
                        "rms_by_chunk": [round(d["rms"], 4) for d in drift]}
        print(f"  {tag}: loopiness mean {mean_loop:.4f} p90 {p90_loop:.4f} early {early:.4f} late {late:.4f}", flush=True)
        # decode to audio (chunked) -> wav
        with torch.no_grad():
            pt_dtype = next(inner.pretransform.parameters()).dtype
            audio = inner.pretransform.decode(lat.to(pt_dtype), chunked=True).float().cpu()
        audio = audio / (audio.abs().max() + 1e-6)       # peak-normalize (fp16 clip guard)
        import torchaudio
        torchaudio.save(f"{args.out_dir}/{args.intervention}_{tag}.wav",
                        audio.squeeze(0) if audio.dim() == 3 else audio, int(sr))
        print(f"  wrote {args.intervention}_{tag}.wav", flush=True)

    iv = args.intervention
    delta = results["baseline"]["loopiness_mean"] - results[iv]["loopiness_mean"]
    verdict = (f"{iv} REDUCES loopiness" if delta > 0.01
               else f"{iv} does NOT reduce loopiness (null/negative)")
    meta = {
        "purpose": f"{iv} (training-free long-form fix) vs baseline on longform.py sliding-window "
                   "continuation — does it reduce the loop attractor?",
        "intervention": iv, "single_prompt": args.single, "arc": arc,
        "window_sec": args.window_sec, "overlap_sec": args.overlap_sec,
        "duration": args.duration, "steps": args.steps, "cfg": args.cfg, "seed": args.seed,
        "jitter_scale": (args.jitter_scale if iv == "rope_jitter" else None),
        "fps": fps, "results": results,
        "verdict": verdict, "loopiness_delta_baseline_minus_intervention": round(delta, 4),
        "note": "loopiness = mean max-cosine-sim to any earlier non-adjacent frame; lower = less repetition. "
                "early vs late detects a degenerating loop (climbs over the rollout).",
    }
    json.dump(meta, open(f"{args.out_dir}/run_meta.json", "w"), indent=2)
    print(f"\n=== VERDICT: {verdict} (delta {delta:+.4f}) ===")
    print(f"  baseline loopiness {results['baseline']['loopiness_mean']:.4f} "
          f"(early {results['baseline']['loopiness_early']:.3f} late {results['baseline']['loopiness_late']:.3f})  "
          f"{iv} {results[iv]['loopiness_mean']:.4f} "
          f"(early {results[iv]['loopiness_early']:.3f} late {results[iv]['loopiness_late']:.3f})")
    print(f"wrote {args.out_dir}/run_meta.json")


if __name__ == "__main__":
    main()

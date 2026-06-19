# scripts/longform_render.py
"""Render long-form SA3 audio via sliding-window continuation. See
docs/superpowers/specs/2026-06-19-longform-sdedit-reanchor-crossfade-design.md."""
from __future__ import annotations
import argparse


def parse_schedule(arg: str) -> str | list[tuple[float, str]]:
    """Single prompt -> str; 't:prompt|t:prompt' -> [(float, str), ...].

    Treated as a schedule only if EVERY '|'-separated segment begins with a float
    timestamp followed by ':'. Otherwise returned as a single prompt, so prompts
    that contain a colon (e.g. '120bpm: deep techno') are handled safely.
    """
    parsed = []
    for seg in arg.split("|"):
        head, sep, prompt = seg.partition(":")
        if not sep:
            return arg
        try:
            t = float(head.strip())
        except ValueError:
            return arg
        parsed.append((t, prompt))
    return parsed


def main() -> None:
    import torch
    import soundfile as sf
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.inference.longform import (
        PromptSchedule,
        InpaintContinuationGenerator,
        LongFormRenderer,
    )

    ap = argparse.ArgumentParser(
        description="Render long-form SA3 audio via sliding-window inpaint continuation."
    )
    ap.add_argument("--model", default="small-music-base")
    ap.add_argument("--prompt", required=True, help="prompt, or '0:A|30:B' schedule")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--overlap-sec", type=float, default=5.0)
    ap.add_argument("--blend-frames", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("-o", "--out", default="longform.wav")
    args = ap.parse_args()

    m = StableAudioModel.from_pretrained(args.model, model_half=args.half)
    inner = m.model
    sr = inner.sample_rate
    fps = sr / inner.pretransform.downsampling_ratio

    def f(s: float) -> int:
        return int(round(s * fps))

    gen = InpaintContinuationGenerator(m, steps=args.steps, cfg_scale=args.cfg)
    r = LongFormRenderer(
        gen,
        channels=inner.io_channels,
        fps=fps,
        window_frames=f(args.window_sec),
        overlap_frames=f(args.overlap_sec),
        blend_frames=args.blend_frames,
    )
    sched = PromptSchedule(parse_schedule(args.prompt))
    lat = r.render_latents(sched, total_frames=f(args.duration))
    print(f"[longform] latents {tuple(lat.shape)} finite={bool(torch.isfinite(lat).all())}")
    print(f"[longform] drift_log rms: {[round(d['rms'], 3) for d in r.drift_log]}")

    # AutoencoderPretransform.decode accepts chunked=True (verified: pretransforms.py:23).
    # chunked=True delegates to model.decode_audio(chunked=True) which processes in chunks
    # internally, avoiding OOM on multi-minute latents.
    with torch.no_grad():
        pt_dtype = next(inner.pretransform.parameters()).dtype
        audio = inner.pretransform.decode(lat.to(pt_dtype), chunked=True).float().cpu()
    audio = audio.clamp(-1, 1)
    wav = audio[0] if audio.dim() == 3 else audio
    sf.write(args.out, wav.transpose(0, 1).numpy(), int(sr))
    print(f"[longform] saved -> {args.out}")


if __name__ == "__main__":
    main()

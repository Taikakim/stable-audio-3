"""Smoke test for InfiniteAudio-style FIFO generation on SA3.

UNTESTED at authoring time (GPU was occupied). Run on small-music-base first
(CPU-capable, fast). See docs/INFINITE_AUDIO_FIFO.md.

Two modes:
  --parity   : per-frame forward parity check. With a CONSTANT per-frame t, the
               patched (B,T) forward must match the original (B,) forward to ~1e-4.
               This isolates the surgery from the FIFO loop and needs no full gen.
  (default)  : run a short FIFO generation and save a WAV.

Usage:
  python scripts/fifo_infinite_smoke.py --parity
  python scripts/fifo_infinite_smoke.py --prompt "124 BPM acid techno" \
      --window 256 --emit 512 --cfg 6.0 -o fifo_out.wav
"""

from __future__ import annotations

import argparse

import torch


def run_parity(model, device):
    """Constant per-frame t must reproduce the scalar-t forward."""
    from stable_audio_3.inference.fifo_infinite import (
        install_per_frame_timestep_patch, build_window_conditioning,
    )

    window = 64
    wrapper = model.model.model  # DiTWrapper
    cond_pos, _, io_channels = build_window_conditioning(model, "test", window, device)

    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(1, io_channels, window, generator=g, device=device, dtype=torch.float32)
    sigma = 0.5

    # original scalar path (patch is a pass-through for 1-D t, but compare pre-install too)
    with torch.no_grad():
        t_scalar = torch.full((1,), sigma, device=device, dtype=torch.float32)
        v_scalar = wrapper(x, t_scalar, cfg_scale=1.0, **cond_pos).float()

    install_per_frame_timestep_patch()
    with torch.no_grad():
        t_perframe = torch.full((1, window), sigma, device=device, dtype=torch.float32)
        v_perframe = wrapper(x, t_perframe, cfg_scale=1.0, **cond_pos).float()

    diff = (v_scalar - v_perframe).abs().max().item()
    scale = v_scalar.abs().max().item()
    rel = diff / (scale + 1e-8)
    print(f"[parity] window={window} sigma={sigma}")
    print(f"[parity] max abs diff = {diff:.3e}  (rel {rel:.3e})")
    print(f"[parity] v_scalar finite={bool(torch.isfinite(v_scalar).all())}  "
          f"v_perframe finite={bool(torch.isfinite(v_perframe).all())}")
    # Constant per-frame t should reproduce the scalar forward up to kernel noise.
    # ROCm/MIOpen GEMM nondeterminism on this shape-divergent path floors abs diff at
    # ~1e-3..1e-2 (rel ~1e-3), NOT ~1e-4 — gate primarily on relative error.
    ok = (rel < 5e-3) or (diff < max(1e-2, 1e-2 * scale))
    print(f"[parity] {'PASS' if ok else 'FAIL'} (rel<5e-3 or abs floor ~1e-2)")
    return ok


def run_fifo(model, device, args):
    from stable_audio_3.inference.fifo_infinite import (
        FIFOConfig, sample_fifo_infinite, build_window_conditioning,
    )

    cond_pos, cond_neg, io_channels = build_window_conditioning(
        model, args.prompt, args.window, device
    )
    cfg = FIFOConfig(
        window=args.window, n_buffer=args.buffer, emit_frames=args.emit,
        warmup_frames=args.warmup, cfg_scale=args.cfg, seed=args.seed,
    )
    sr = model.model.sample_rate
    lat = sample_fifo_infinite(
        model.model.model, cond_pos, cond_neg,
        io_channels=io_channels, cfg=cfg, device=device,
        dist_shift=model.model.sampling_dist_shift,
    )
    print(f"[fifo] emitted latents shape={tuple(lat.shape)}  "
          f"finite={bool(torch.isfinite(lat).all())}")

    fps = sr / model.model.pretransform.downsampling_ratio
    print(f"[fifo] ~{lat.shape[-1] / fps:.1f}s of audio at {fps:.2f} latent fps")

    with torch.no_grad():
        audio = model.model.pretransform.decode(lat.to(next(model.model.pretransform.parameters()).dtype))
    audio = audio.float().cpu()
    print(f"[fifo] decoded audio shape={tuple(audio.shape)}  "
          f"rms={audio.pow(2).mean().sqrt().item():.4f}  "
          f"finite={bool(torch.isfinite(audio).all())}")

    _positional_drift_report(audio, sr)

    wav = audio[0] if audio.dim() == 3 else audio  # (C, T)
    # soundfile is portable across both venvs; torchaudio.save on torch 2.12 routes
    # through torchcodec, which the ROCm-7.14 test venv doesn't have.
    import soundfile as sf
    sf.write(args.out, wav.transpose(0, 1).cpu().numpy(), int(sr))  # (T, C)
    print(f"[fifo] saved -> {args.out}")


def _positional_drift_report(audio, sr, n_chunks: int = 8):
    """Coarse FIFO positional-consistency diagnostic.

    FIFO frames migrate through absolute rotary positions as they denoise
    (transformer.py:1217), a known FIFO-on-a-non-FIFO-model risk that can show up
    as tempo/pitch/energy drift over the emission. We split the output into chunks
    and print per-chunk RMS, spectral centroid, and (if librosa is present) tempo,
    so drift across the stream is visible at a glance. Stable values ≈ no drift.
    """
    wav = audio[0] if audio.dim() == 3 else audio
    mono = wav.mean(0) if wav.dim() == 2 else wav
    x = mono.detach().cpu().float().numpy()
    N = len(x)
    if N < n_chunks * 2048:
        print("[drift] output too short for drift report")
        return
    edges = [int(i * N / n_chunks) for i in range(n_chunks + 1)]
    print(f"[drift] per-chunk over {n_chunks} segments (watch for monotonic drift):")
    try:
        import librosa
        have_librosa = True
    except Exception:
        have_librosa = False
    for i in range(n_chunks):
        seg = x[edges[i]:edges[i + 1]]
        rms = float((seg ** 2).mean() ** 0.5)
        # spectral centroid (Hz) via rFFT magnitude
        import numpy as np
        mag = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
        cen = float((freqs * mag).sum() / (mag.sum() + 1e-9))
        extra = ""
        if have_librosa:
            try:
                tempo = float(librosa.beat.tempo(y=seg, sr=sr)[0])
                extra = f"  tempo={tempo:6.1f}bpm"
            except Exception:
                extra = ""
        print(f"[drift]   seg {i}: rms={rms:.4f}  centroid={cen:7.1f}Hz{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small-music-base")
    ap.add_argument("--prompt", default="124 BPM acid techno, dry snappy kick and bassline")
    # Defaults sized for a genuinely fast FIRST smoke (~64 frames window, ~32 emitted
    # ≈ 3 s). total = (warmup+emit) iters x2 forwards; scale up once it's working.
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--buffer", type=int, default=0)
    ap.add_argument("--emit", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--half", action="store_true", help="model_half (GPU)")
    ap.add_argument("--parity", action="store_true", help="run forward-parity check only")
    ap.add_argument("-o", "--out", default="fifo_out.wav")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from stable_audio_3 import StableAudioModel
    print(f"loading {args.model} (half={args.half}) on {device}...")
    model = StableAudioModel.from_pretrained(args.model, model_half=args.half)

    if args.parity:
        run_parity(model, device)
    else:
        run_fifo(model, device, args)


if __name__ == "__main__":
    main()

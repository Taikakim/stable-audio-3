#!/usr/bin/env python3
"""Asynchronous CPU VAE Decoder for SA3 Latents.

Decodes .z0.npy latent files to .wav using the SAME autoencoder on CPU.
Uses PyTorch SDPA by disabling flash-attention (SA3_DISABLE_FLASH_ATTN=1),
freeing all GPU VRAM for active training.

Usage:
  # Decode all pending latents in a directory:
  python decode_latents_cpu.py --latent_dir /path/to/demos

  # Watch mode (runs in background, decodes as latents arrive):
  python decode_latents_cpu.py --watch --latent_dir /path/to/demos
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""
os.environ["SA3_DISABLE_FLASH_ATTN"] = "1"
os.environ["SA3_DISABLE_FLASH_VARLEN"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import glob
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch
torch.set_num_threads(4)
from safetensors.torch import load_file

from stable_audio_3.factory import create_pretransform_from_config
from stable_audio_3.model_configs import base_models


def load_cpu_pretransform(model_name: str = "medium-base"):
    """Instantiate and load the SAME autoencoder pretransform on CPU."""
    cfg_path, ckpt = base_models[model_name].resolve()
    with open(cfg_path) as f:
        cfg = json.load(f)

    sample_rate = cfg.get("sample_rate", 44100)
    pretransform = create_pretransform_from_config(cfg["model"], sample_rate)

    sd = load_file(ckpt)
    pfx = "pretransform."
    ae_sd = {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)}
    pretransform.load_state_dict(ae_sd, strict=False)
    pretransform.eval()
    return pretransform, sample_rate


def decode_one(pretransform, latent_path: Path, sample_rate: int = 44100):
    """Decode a single .z0.npy latent file to .wav."""
    wav_path = latent_path.with_name(latent_path.name.replace(".z0.npy", ".wav"))
    if wav_path.exists():
        return False

    try:
        data = np.load(latent_path)
    except Exception as e:
        print(f"[CPU VAE] Error loading {latent_path.name}: {e}")
    if not np.isfinite(data).all():
        print(f"[CPU VAE] Error: latents in {latent_path.name} contain NaNs or Infs! Skipping decode to avoid empty audio.")
        return False

    z = torch.from_numpy(data)
    if z.ndim == 2:
        z = z.unsqueeze(0)
    if z.ndim != 3 or z.shape[1] != 256:
        print(f"[CPU VAE] Warning: unexpected latent shape {z.shape} in {latent_path.name}")

    z = z.to(torch.float32)
    t0 = time.time()
    with torch.no_grad():
        audio = pretransform.decode(z)

    # Audio shape: (B, C, N) -> (N, C) for soundfile
    if audio.ndim == 3:
        audio = audio[0]
    audio_np = audio.cpu().float().numpy().T

    # Peak normalization to -0.1 dB
    peak = np.abs(audio_np).max()
    if peak > 1e-6:
        audio_np = (audio_np / peak) * 0.988

    sf.write(str(wav_path), audio_np, sample_rate, subtype="PCM_16")
    dur = len(audio_np) / sample_rate
    elapsed = time.time() - t0
    print(f"[CPU VAE] Decoded: {wav_path.name} ({dur:.2f}s audio in {elapsed:.2f}s)", flush=True)
    return True


def scan_and_decode(pretransform, latent_dir: Path, sample_rate: int):
    """Scan latent_dir recursively for un-decoded .z0.npy files."""
    files = sorted(latent_dir.rglob("*.z0.npy"))
    decoded_count = 0
    for f in files:
        if decode_one(pretransform, f, sample_rate):
            decoded_count += 1
    return decoded_count


def main():
    parser = argparse.ArgumentParser(description="Decode SA3 latents to WAV on CPU")
    parser.add_argument("--latent_dir", type=str, required=True,
                        help="Directory to scan for .z0.npy files")
    parser.add_argument("--model", type=str, default="medium-base",
                        help="Model name for pretransform weights (default: medium-base)")
    parser.add_argument("--watch", action="store_true",
                        help="Watch directory continuously until stopped")
    parser.add_argument("--poll_interval", type=float, default=5.0,
                        help="Poll interval in seconds for watch mode")
    parser.add_argument("--idle_exit_after", type=float, default=None,
                        help="Exit watch mode after N seconds of no new files")
    args = parser.parse_args()

    latent_dir = Path(args.latent_dir)
    latent_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CPU VAE] Initializing CPU pretransform from {args.model}...")
    pretransform, sample_rate = load_cpu_pretransform(args.model)
    print(f"[CPU VAE] Ready. Scanning {latent_dir} (sr={sample_rate})...")

    if not args.watch:
        count = scan_and_decode(pretransform, latent_dir, sample_rate)
        print(f"[CPU VAE] Finished: decoded {count} files.")
        return

    last_active = time.time()
    try:
        while True:
            count = scan_and_decode(pretransform, latent_dir, sample_rate)
            if count > 0:
                last_active = time.time()
            elif args.idle_exit_after and (time.time() - last_active) > args.idle_exit_after:
                print(f"[CPU VAE] Idle timeout ({args.idle_exit_after}s) reached. Exiting.")
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("[CPU VAE] Stopped by user.")


if __name__ == "__main__":
    main()

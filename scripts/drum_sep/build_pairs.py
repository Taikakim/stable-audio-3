#!/usr/bin/env python3
"""Build paired training data for SA3 Drum Separator (Gate G2).
Encodes isolated drum stems, drum buses, and remix augmentations into the .npz contract.

Data contract per .npz sample:
  target_lat     (fp16, [256, T]) -- isolated target stem (e.g. kick)
  mix_lat        (fp16, [256, T]) -- drum bus (input mixture)
  prompt         (str)            -- caption (e.g. "solo kick drum, punchy goa trance kick")
  stem_tag       (str)            -- "kick", "snare", "hihat", etc.
  seconds_total  (float)          -- window length in seconds (e.g. 12.0)
  padding_mask   (uint8, [T])     -- all 1s for valid frames
  song           (str)            -- source song title
  window_idx     (int)            -- window index
  remix_type     (str)            -- "orig", "kick_boost", "kick_cut"
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio

from stable_audio_3.loading_utils import load_autoencoder
from stable_audio_3.model_configs import ae_models

SR = 44100
DOWNSAMPLING_RATIO = 4096

DEFAULT_HOLDOUTS = [
    "Aavepyörä - Goddess Guerrilla  goa morning bright energetic trance",
    "Summamutikka - Acid Alien aggressive powerful psychedelic melodic goa trance",
    "Aavepyörä - Dance of Tomorrow driving rhythmic hypnotic  goatrance",
    "Aavepyörä - Kääpiösaaren Salaisuus electro upbeat electro trance",
    "Aavepyörä - keväthuuto upbeat happy positive offbeat trance",
    "Summamutikka - Increase The Dose or Drop The Poison driving groovy upbeat goa trance",
    "Aavepyörä - Commodore Shiva impulse tracker soft adventure trance",
    "Aavepyörä - Kaikki-Alla driving upbeat energetic atmospheric lush melodic eclectic goa trance",
    "Aavepyörä - Kaleidoscillator colourful electronic eclectic dance music",
    "Summamutikka - Black Moon psychedelic goa trance",
]

def load_audio_stem(path: Path, target_sr=SR):
    """Load audio file as (C, T) float32 stereo at target_sr."""
    y, fs = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.T # (C, T)
    if fs != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sr)
        y_t = torch.from_numpy(y)
        y = resampler(y_t).numpy()
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    elif y.shape[0] > 2:
        y = y[:2]
    return y

def encode_batch(ae, wav_batch: torch.Tensor, device, dtype):
    """Encode (B, 2, T) audio batch into (B, 256, T_lat) fp16 latents."""
    wav_t = wav_batch.to(device=device, dtype=dtype)
    with torch.inference_mode():
        if hasattr(ae, "encode_audio"):
            latents = ae.encode_audio(wav_t)
        else:
            latents = ae.encode(wav_t)
    return latents.cpu().to(torch.float16).numpy()

def main():
    p = argparse.ArgumentParser(description="Encode paired drum separation dataset for SA3")
    p.add_argument("--stems_dir", type=str, default="/run/media/kim/Mantu/avp-stems-original-classified",
                   help="Root folder of classified stems")
    p.add_argument("--out_dir", type=str, default="/home/kim/sao_runs/drum_sep_data",
                   help="Output directory for .npz pairs")
    p.add_argument("--model", type=str, default="same-l", help="Model name for autoencoder")
    p.add_argument("--window_sec", type=float, default=12.0, help="Window length in seconds")
    p.add_argument("--hop_sec", type=float, default=6.0, help="Window hop in seconds (default 50%% overlap)")
    p.add_argument("--max_tracks", type=int, default=None, help="Max tracks to process (for testing)")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "all"])
    p.add_argument("--remixes", action="store_true", default=True, help="Include remix augmentations (3x pairs)")
    p.add_argument("--no_remixes", dest="remixes", action="store_false")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    split_dir = out_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading autoencoder '{args.model}'...")
    cfg_path, ckpt_path = ae_models[args.model].resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = load_autoencoder(cfg_path, ckpt_path, device=device)
    model_dtype = next(ae.parameters()).dtype
    print(f"Autoencoder ready on {device} ({model_dtype}).")

    stems_root = Path(args.stems_dir)
    track_dirs = sorted([d for d in stems_root.iterdir() if d.is_dir()])
    
    # Filter by holdout list
    if args.split == "val":
        selected_dirs = [d for d in track_dirs if d.name in DEFAULT_HOLDOUTS]
    elif args.split == "train":
        selected_dirs = [d for d in track_dirs if d.name not in DEFAULT_HOLDOUTS]
    else:
        selected_dirs = track_dirs

    if args.max_tracks is not None:
        selected_dirs = selected_dirs[:args.max_tracks]

    print(f"Processing {len(selected_dirs)} tracks for split '{args.split}'...")

    win_samples = int(args.window_sec * SR)
    hop_samples = int(args.hop_sec * SR)
    
    samples_manifest = []
    total_pairs = 0

    for t_idx, t_dir in enumerate(selected_dirs):
        song_name = t_dir.name
        print(f"\n[{t_idx+1}/{len(selected_dirs)}] {song_name}")
        
        kick_dir = t_dir / "kick"
        drums_dir = t_dir / "drums"
        
        if not kick_dir.exists() or not drums_dir.exists():
            print("  Skipping: missing kick/ or drums/ directory")
            continue
            
        # Find kick downmix or kick stem
        kick_files = list(kick_dir.glob("KICK_downmix.flac"))
        if not kick_files:
            kick_files = list(kick_dir.glob("*.flac"))
        if not kick_files:
            print("  Skipping: no kick audio file found")
            continue
            
        # Find drums downmix or drum stems
        drums_files = list(drums_dir.glob("DRUMS_downmix.flac"))
        if not drums_files:
            drums_files = list(drums_dir.glob("*.flac"))
        if not drums_files:
            print("  Skipping: no non-kick drum audio found")
            continue

        kick_wav = load_audio_stem(kick_files[0])
        drums_wav = load_audio_stem(drums_files[0])

        # Align lengths
        min_len = min(kick_wav.shape[1], drums_wav.shape[1])
        kick_wav = kick_wav[:, :min_len]
        drums_wav = drums_wav[:, :min_len]
        bus_wav = kick_wav + drums_wav

        # Slice into windows
        n_windows = max(0, (min_len - win_samples) // hop_samples + 1)
        print(f"  Audio duration: {min_len / SR:.1f}s -> {n_windows} windows of {args.window_sec:.1f}s")
        
        track_pairs = 0
        for w in range(n_windows):
            start = w * hop_samples
            end = start + win_samples
            
            k_win = kick_wav[:, start:end]
            d_win = drums_wav[:, start:end]
            b_win = bus_wav[:, start:end]

            # Activity gate: skip windows where kick is silent
            k_rms = np.sqrt(np.mean(k_win**2))
            if k_rms < 0.005:
                continue

            # Prepare batch of variants to encode together on GPU
            # 1. Base clean bus & clean kick
            # 2. Remix 1: kick boosted (+2.5 dB), other drums attenuated (-2.5 dB)
            # 3. Remix 2: kick attenuated (-2.5 dB), other drums boosted (+2.5 dB)
            remix1_b = k_win * 1.33 + d_win * 0.75
            remix1_k = k_win * 1.33
            remix2_b = k_win * 0.75 + d_win * 1.33
            remix2_k = k_win * 0.75

            if args.remixes:
                wav_batch = torch.from_numpy(np.stack([
                    k_win, b_win,
                    remix1_k, remix1_b,
                    remix2_k, remix2_b
                ], axis=0)) # (6, 2, win_samples)
                lat_batch = encode_batch(ae, wav_batch, device, model_dtype)
                
                variants = [
                    ("orig", lat_batch[0], lat_batch[1]),
                    ("kick_boost", lat_batch[2], lat_batch[3]),
                    ("kick_cut", lat_batch[4], lat_batch[5]),
                ]
            else:
                wav_batch = torch.from_numpy(np.stack([k_win, b_win], axis=0))
                lat_batch = encode_batch(ae, wav_batch, device, model_dtype)
                variants = [("orig", lat_batch[0], lat_batch[1])]

            for v_name, target_lat, mix_lat in variants:
                sample_id = f"{t_idx:03d}_{w:04d}_{v_name}"
                npz_fname = f"{sample_id}.npz"
                npz_path = split_dir / npz_fname
                
                T_lat = target_lat.shape[-1]
                np.savez_compressed(
                    npz_path,
                    target_lat=target_lat, # (256, T_lat) fp16
                    mix_lat=mix_lat,       # (256, T_lat) fp16
                    prompt="solo kick drum, punchy goa trance kick, dry isolated hit",
                    stem_tag="kick",
                    seconds_total=args.window_sec,
                    padding_mask=np.ones(T_lat, dtype=np.uint8),
                    song=song_name,
                    window_idx=w,
                    remix_type=v_name
                )
                
                samples_manifest.append({
                    "id": sample_id,
                    "file": npz_fname,
                    "song": song_name,
                    "window": w,
                    "remix": v_name,
                    "t_lat": T_lat
                })
                track_pairs += 1

        print(f"  -> Generated {track_pairs} paired samples")
        total_pairs += track_pairs

    manifest_path = split_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "split": args.split,
            "total_samples": len(samples_manifest),
            "window_sec": args.window_sec,
            "hop_sec": args.hop_sec,
            "samples": samples_manifest
        }, f, indent=2)

    print(f"\nDone! Successfully created {total_pairs} paired samples in {split_dir}")
    print(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    main()

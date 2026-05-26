# scripts/latch/encode_latch_dataset.py
"""Re-encode an audio corpus through SAME (SA3) into per-clip latents for LatCH training.

Output per clip <stem>:
  <out>/<stem>.npy   float32 latent of shape (256, T)
  <out>/<stem>.json  {"crop_key": <stem>, "latent_frames": T, "seconds": <float>}

Usage:
  uv run python scripts/latch/encode_latch_dataset.py \
    --model same-s --audio-dir /path/to/clips --out-dir /path/to/sa3_latents
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torchaudio

from stable_audio_3 import AutoencoderModel

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
STEM_SUFFIXES = ("_bass", "_drums", "_other", "_vocals")


def main(args):
    ae = AutoencoderModel.from_pretrained(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = [
        p for p in sorted(Path(args.audio_dir).rglob("*"))
        if p.suffix.lower() in AUDIO_EXTS
        and not any(p.stem.endswith(s) for s in STEM_SUFFIXES)
    ]
    if args.limit is not None:
        audio_paths = audio_paths[: args.limit]
    print(f"Found {len(audio_paths)} audio files to encode with {args.model}.")

    for i, path in enumerate(audio_paths):
        stem = path.stem
        npy_path = out_dir / f"{stem}.npy"
        if npy_path.exists() and not args.overwrite:
            continue
        try:
            wav, sr = torchaudio.load(str(path))           # (C, N)
            latent = ae.encode(wav, sr)                     # (1, 256, T)
            latent = latent.squeeze(0).float().cpu().numpy()  # (256, T)
            np.save(str(npy_path), latent)
            meta = {
                "crop_key": stem,
                "latent_frames": int(latent.shape[1]),
                "seconds": float(wav.shape[-1] / sr),
            }
            (out_dir / f"{stem}.json").write_text(json.dumps(meta))
        except Exception as e:
            print(f"  SKIP {stem}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(audio_paths)} encoded")

    print(f"Done. Latents in {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="same-s", choices=["same-s", "same-l"])
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Encode only the first N full-mix crops (after sort/filter).")
    main(p.parse_args())

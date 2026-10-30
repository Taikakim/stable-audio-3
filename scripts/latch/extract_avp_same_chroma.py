#!/usr/bin/env python3
"""
extract_avp_same_chroma.py -- per-crop 384-d SAME-chroma targets for the AVP corpus
(latents_avp), measured from each crop's separated STEM audio: the same recipe as the Goa
store latents_sa3_stem_chroma (compute_same_chroma on the crop's `other` / `bass` stem
window, WORKLOG 2026-06-25), so `--chroma-key other` means the same thing in both corpora.

KUANG 2026-10-30: the first version of this script (e9ce2b6) applied the linear latent
readout W[b] @ z + bias[b] (chroma_heads_real.npz) to the FULL-MIX latent and saved that one
array under both 'other' and 'full_mix'. That is the July production-head target, not the
stem chroma the E13 head learned, so the C3_avp_* heads learned a different quantity under
the same key and the Goa+AVP arm mixed two definitions. Do not reuse its output directory.

Per crop: <idx>.json gives source_path (the full_mix file) and timestamps [start, end] in
seconds. The stem <key>.<ext> is looked up in the same folder (mir get_stem_files naming),
read from `start` for CROP_SAMPLES/44100 s at the file's own rate, resampled to 44.1 kHz and
padded/trimmed to CROP_SAMPLES exactly as sa3_encode_from_manifest read the full mix, then
compute_same_chroma(align_to_latent=True) -> (3, 128, T). Saved fp16, one array per key.

Crops whose folder lacks a requested stem (augmentation variants, unless their stems were
rendered too) are skipped and counted; the trainer drops crops without an npz. A recipe
marker (_recipe.json) in --out-dir is checked on resume, so files from another recipe never
mix with these.

CPU only. Run on the desktop (any env with numpy/scipy/soundfile/torchaudio + mir-same-chroma):
    python3 scripts/latch/extract_avp_same_chroma.py --limit 4      # smoke: prints found/missing
    python3 scripts/latch/extract_avp_same_chroma.py                # full run, resumable
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CROP_SAMPLES = 16777216          # sa3_encode_from_manifest
SAMPLE_RATE = 44100
STEM_EXTS = (".flac", ".wav", ".mp3", ".ogg", ".aiff", ".aif")
RECIPE = {"recipe": "stem_audio_compute_same_chroma", "align_to_latent": True,
          "crop_samples": CROP_SAMPLES, "sample_rate": SAMPLE_RATE, "dtype": "float16"}
SAME_CHROMA_SRC = "/home/kim/Projects/mir-same-chroma/src"
BYTES_PER_KEY = 3 * 128 * 4096 * 2   # fp16 (3,128,4096), before compression

_sc = None
_keys = None
_out = None


def find_stem(folder: Path, key: str):
    for ext in STEM_EXTS:
        p = folder / f"{key}{ext}"
        if p.exists():
            return p
    return None


def crop_window(meta: dict):
    """(full_mix path, start seconds) from a crop sidecar, or raise KeyError."""
    src = meta.get("source_path") or meta.get("path")
    stamps = meta.get("timestamps")
    start = float(stamps[0]) if stamps else float(meta["seconds_start"])
    if not src:
        raise KeyError("source_path")
    return Path(src), start


def read_window(path: Path, start_s: float) -> np.ndarray:
    """(CROP_SAMPLES, C) float32 at 44.1 kHz, read like sa3_encode_from_manifest reads the
    full mix: native-rate window, fp32 resample, then pad/trim to CROP_SAMPLES. A stem that
    ends early is zero-padded (the encoder skipped short FULL MIXES, so the crop exists)."""
    import soundfile as sf
    native_sr = sf.info(str(path)).samplerate
    frames = int(round(CROP_SAMPLES / SAMPLE_RATE * native_sr))
    audio, sr = sf.read(str(path), start=int(round(start_s * native_sr)), frames=frames,
                        dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        import torch
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(audio.T).contiguous(), sr, SAMPLE_RATE)
        audio = wav.numpy().T
    n = audio.shape[0]
    if n < CROP_SAMPLES:
        audio = np.pad(audio, ((0, CROP_SAMPLES - n), (0, 0)))
    return np.ascontiguousarray(audio[:CROP_SAMPLES])


def init_worker(same_chroma_src, keys, out_dir):
    global _sc, _keys, _out
    if same_chroma_src and same_chroma_src not in sys.path:
        sys.path.insert(0, same_chroma_src)
    from harmonic import same_chroma
    _sc = same_chroma
    _keys = list(keys)
    _out = Path(out_dir)


def process_item(npy_str: str):
    """-> (status, message), status one of new / exists / missing / error."""
    npy = Path(npy_str)
    dst = _out / (npy.stem + ".npz")
    if dst.exists():
        return "exists", ""
    try:
        meta = json.loads(npy.with_suffix(".json").read_text())
        full_mix, start = crop_window(meta)
        stems = {k: (full_mix if k == "full_mix" else find_stem(full_mix.parent, k)) for k in _keys}
        missing = [k for k, p in stems.items() if p is None or not Path(p).exists()]
        if missing:
            return "missing", f"{npy.stem}: no {','.join(missing)} in {full_mix.parent}"
        arrays = {}
        for k, p in stems.items():
            wav = read_window(Path(p), start)
            arrays[k] = _sc.compute_same_chroma(wav, SAMPLE_RATE, align_to_latent=True, workers=1).astype(np.float16)
        tmp = dst.with_name(dst.stem + ".partial.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, dst)
        return "new", ""
    except Exception as e:  # one bad crop must not kill a long run; it is counted and printed
        return "error", f"{npy.stem}: {type(e).__name__}: {e}"


def check_recipe(out: Path, keys):
    """Refuse an out dir holding another recipe's files (e.g. the old readout targets)."""
    marker = out / "_recipe.json"
    want = dict(RECIPE, keys=sorted(keys))
    if marker.exists():
        have = json.loads(marker.read_text())
        if have != want:
            raise SystemExit(f"{out} holds targets from another recipe: {have}\nwanted {want}. Use a new --out-dir.")
    elif any(out.glob("*.npz")):
        raise SystemExit(f"{out} already has .npz files but no _recipe.json (old readout targets?). "
                         "Use a new --out-dir.")
    else:
        marker.write_text(json.dumps(want, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-dir", default="/run/media/kim/Kosmos/latents/latents_avp")
    ap.add_argument("--out-dir", default="/run/media/kim/Kosmos/latents/latents_avp_stem_chroma")
    ap.add_argument("--keys", default="other,bass", help="stems to measure (Goa store: other,bass); 'full_mix' reads the mix")
    ap.add_argument("--same-chroma-src", default=SAME_CHROMA_SRC, help="dir holding harmonic/same_chroma.py")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N crops only (smoke)")
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    args = ap.parse_args(argv)

    keys = [k for k in args.keys.split(",") if k]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    check_recipe(out, keys)

    items = sorted(p for p in Path(args.latent_dir).glob("*.npy") if p.stem != "silence")
    if args.limit:
        items = items[: args.limit]
    need_gb = len(items) * len(keys) * BYTES_PER_KEY / 2**30
    free_gb = shutil.disk_usage(out).free / 2**30
    print(f"[stem-chroma] {len(items)} crops in {args.latent_dir}, keys {keys}, {args.workers} workers; "
          f"<= {need_gb:.1f} GB uncompressed, {free_gb:.1f} GB free on {out}", flush=True)
    if free_gb - need_gb < args.min_free_gb:
        raise SystemExit(f"not enough space on {out}: need up to {need_gb:.1f} GB + {args.min_free_gb} GB margin")

    counts = {"new": 0, "exists": 0, "missing": 0, "error": 0}
    shown = {"missing": 0, "error": 0}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.same_chroma_src, keys, str(out))) as pool:
        for i, (status, msg) in enumerate(pool.map(process_item, [str(p) for p in items], chunksize=4)):
            counts[status] += 1
            if status in shown and shown[status] < 10:
                shown[status] += 1
                print(f"  [{status}] {msg}", flush=True)
            if (i + 1) % 100 == 0 or (i + 1) == len(items):
                print(f"[stem-chroma] {i + 1}/{len(items)} {counts} "
                      f"[{(i + 1) / max(1e-9, time.time() - t0):.2f} crops/s]", flush=True)
    print(f"[stem-chroma] DONE {counts} in {time.time() - t0:.0f}s -> {out}", flush=True)
    if counts["new"] + counts["exists"] == 0:
        raise SystemExit("no targets written: check the stem names/paths printed above")
    return counts


if __name__ == "__main__":
    main()

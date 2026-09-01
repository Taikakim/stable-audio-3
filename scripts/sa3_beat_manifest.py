"""
sa3_beat_manifest.py -- generate beat-aligned T=4096 crop windows for the SA3
latents_sa3 corpus, from mir-organized+separated track folders.

Reconstructed 2026-07-04: the original scripts (/tmp/sa3_beat_manifest.py ->
/tmp/sa3_encode_from_manifest.py, see docs/training-findings.md and WORKLOG
2026-06-2x) only ever lived in /tmp and were lost. This is a from-spec rewrite,
checked into the repo this time so it survives a session boundary.

Crop spec (unchanged from the original recipe):
  - Fixed window: CROP_SAMPLES = 16777216 samples @ 44.1kHz = 2**24 samples
    = 380.4357... s = exactly 4096 SAME-L latent frames (hop=4096, 10.767 Hz).
  - First crop starts LEAD_IN=1.0s into the track (skips a leading beat/silence).
  - Each subsequent crop's start snaps BACK to the last downbeat at-or-before
    the previous crop's end (musically-aligned overlap, not a plain stride).
  - The final crop is END-ANCHORED (start = duration - CROP_DURATION_S), so the
    tail of the track is always covered even if it overlaps the previous crop
    more than the standard downbeat-snap would.
  - Tracks shorter than CROP_DURATION_S are dropped (can't make even one crop).

Requires per track: <folder>/full_mix.<ext>, <folder>/<folder>.DOWNBEATS
(plain newline-separated timestamps in seconds, written by mir's rhythm stage).

2026-07-06 (WINTERMUTE spec, avp corpus): --include-augmentations treats each
<track>/augmentations/<variant>/full_mix.flac as a FIRST-CLASS track too, so
each Bungee pitch/tempo variant gets its own begin/mid/end crops. Variant
folders have no .DOWNBEATS of their own (no re-analysis has run -- Phase D
derive-vs-reanalysis validation is still open per avp-analyzed/_STATUS.md), so
downbeats are DERIVED from the parent's real downbeats by the exact duration
ratio actually measured between parent and variant (not a re-derived BPM
estimate): scale = variant_duration / parent_duration. This is mechanically
exact for both variant families -- Bungee's set_pitch() is a pure pitch shift
(scale ~= 1.0, timing untouched) and set_speed(x) stretches time by exactly x
(scale = 1/x) -- so ONE formula covers both, no pitch/tempo special-casing
needed. This derivation is about crop POSITIONING (deterministic by
construction of the render), not the separate open question of whether
derived SCALAR metadata (bpm/key/onset-density) matches a fresh re-analysis --
that's Phase D's concern, untouched by this.

Usage:
  mir/bin/python scripts/sa3_beat_manifest.py <dataset_dir> [<dataset_dir> ...] \
      --out /tmp/sa3_crop_manifest.csv [--include-augmentations]
"""
import argparse
import csv
import sys
from pathlib import Path

import soundfile as sf

CROP_SAMPLES = 16777216          # 2**24, exact -> 4096 latent frames at hop=4096
SAMPLE_RATE = 44100
CROP_DURATION_S = CROP_SAMPLES / SAMPLE_RATE   # 380.43573696145125
LEAD_IN_S = 1.0

AUDIO_EXTS = (".flac", ".wav", ".mp3", ".ogg", ".m4a")


def find_full_mix(folder: Path):
    for ext in AUDIO_EXTS:
        p = folder / f"full_mix{ext}"
        if p.exists():
            return p
    return None


def read_downbeats(folder: Path) -> list[float]:
    db_files = list(folder.glob("*.DOWNBEATS"))
    if not db_files:
        return []
    lines = db_files[0].read_text().strip().splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(float(ln))
        except ValueError:
            pass
    return out


def compute_crops(duration: float, downbeats: list[float]) -> list[tuple[float, float]]:
    if duration < CROP_DURATION_S:
        return []

    crops = []
    start = LEAD_IN_S
    while True:
        end = start + CROP_DURATION_S
        if end >= duration:
            break
        crops.append((start, end))
        candidates = [db for db in downbeats if db <= end]
        if not candidates:
            break
        next_start = candidates[-1]
        if next_start <= start:
            # Degenerate (duplicate/unsorted downbeats) -- avoid an infinite loop.
            break
        start = next_start

    final_start = duration - CROP_DURATION_S
    if final_start >= 0 and (not crops or abs(final_start - crops[-1][0]) > 1.0):
        crops.append((final_start, duration))

    return crops


def iter_track_folders(root: Path):
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and find_full_mix(folder) is not None:
            yield folder


def iter_augmentation_folders(track_folder: Path):
    aug_root = track_folder / "augmentations"
    if not aug_root.is_dir():
        return
    for variant_folder in sorted(aug_root.iterdir()):
        if variant_folder.is_dir() and find_full_mix(variant_folder) is not None:
            yield variant_folder


def crop_rows_for(dataset_dir: Path, folder: Path, duration: float, downbeats: list[float],
                   full_mix: Path, track_name: str, parent_track_name: str, variant_name: str) -> list[dict]:
    crops = compute_crops(duration, downbeats)
    return [{
        "dataset_dir": str(dataset_dir),
        "track_folder": str(folder),
        "track_name": track_name,
        "parent_track_name": parent_track_name,
        "variant_name": variant_name,
        "full_mix": str(full_mix),
        "crop_index": i,
        "start_time": f"{start:.6f}",
        "end_time": f"{end:.6f}",
        "duration": f"{duration:.6f}",
    } for i, (start, end) in enumerate(crops)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("/tmp/sa3_crop_manifest.csv"))
    ap.add_argument("--include-augmentations", action="store_true",
                     help="also crop each <track>/augmentations/<variant>/full_mix.* as its "
                          "own first-class track, deriving downbeats from the parent by the "
                          "measured duration ratio (see module docstring)")
    args = ap.parse_args()

    rows = []
    n_dropped_short = 0
    n_dropped_no_downbeats = 0
    n_aug_dropped_short = 0

    for dataset_dir in args.dataset_dirs:
        for folder in iter_track_folders(dataset_dir):
            full_mix = find_full_mix(folder)
            try:
                info = sf.info(str(full_mix))
                duration = info.frames / info.samplerate
            except Exception as e:
                print(f"  SKIP (unreadable): {folder.name}: {e}", file=sys.stderr)
                continue

            downbeats = read_downbeats(folder)
            if not downbeats:
                n_dropped_no_downbeats += 1
                continue

            crops = crop_rows_for(dataset_dir, folder, duration, downbeats, full_mix,
                                   folder.name, folder.name, "")
            if not crops:
                n_dropped_short += 1
            rows.extend(crops)

            if not args.include_augmentations:
                continue
            for variant_folder in iter_augmentation_folders(folder):
                variant_full_mix = find_full_mix(variant_folder)
                try:
                    v_info = sf.info(str(variant_full_mix))
                    v_duration = v_info.frames / v_info.samplerate
                except Exception as e:
                    print(f"  SKIP (unreadable variant): {variant_folder}: {e}", file=sys.stderr)
                    continue
                scale = v_duration / duration
                v_downbeats = [db * scale for db in downbeats]
                v_crops = crop_rows_for(dataset_dir, variant_folder, v_duration, v_downbeats,
                                         variant_full_mix, f"{folder.name}/{variant_folder.name}",
                                         folder.name, variant_folder.name)
                if not v_crops:
                    n_aug_dropped_short += 1
                rows.extend(v_crops)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset_dir", "track_folder", "track_name", "parent_track_name", "variant_name",
            "full_mix", "crop_index", "start_time", "end_time", "duration",
        ])
        writer.writeheader()
        writer.writerows(rows)

    n_tracks = len({r["track_folder"] for r in rows if not r["variant_name"]})
    n_variants = len({r["track_folder"] for r in rows if r["variant_name"]})
    print(f"Wrote {len(rows)} crops from {n_tracks} tracks + {n_variants} augmentation variants -> {args.out}")
    print(f"Dropped: {n_dropped_short} too-short tracks, {n_dropped_no_downbeats} no-downbeats, "
          f"{n_aug_dropped_short} too-short variants")


if __name__ == "__main__":
    main()

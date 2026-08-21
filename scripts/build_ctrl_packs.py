#!/usr/bin/env python3
"""build_ctrl_packs.py — precompute per-crop MIR control arrays for B7 conditioner
training (companion to mir_control.py; Kim direct 2026-08-21).

For every latent stem in a directory that has a .TIMESERIES.npz, writes
<latent_dir>_ctrl/<stem>.ctrl.npy — the (36, 4096) fp16 SUPERSET control array (all
CTRL_FIELDS, normalized with corpus-robust stats). ⚠️ The SIBLING _ctrl dir is
load-bearing, not cosmetic: PreEncodedDataset recursively globs *.npy inside the
latent dir, so co-located ctrl files would be consumed AS LATENTS. Plus, once per dir:
  ctrl_stats.json  the {field: {center, scale}} used (median / half-IQR from a sample)
  ctrl_meta.json   field list + channel offsets + version (consumers verify layout)

WHY PREBUILT FILES: LUMI trains from these with zero mir dependencies and no npz
field-drift risk — the ~250 KB/crop ctrl dir rsyncs in minutes. Packs (rhythm /
melody / ...) are CHANNEL SUBSETS selected at train time, so one build serves every
bracket arm. Missing fields (f0 on avp) zero-fill — same layout both corpora.

Run (any venv with numpy; ~10 min for 7.8k crops):
  python3 stable-audio-3/scripts/build_ctrl_packs.py \
      --dirs /home/kim/Projects/latents_sa3,/home/kim/Projects/latents_avp
Stats are estimated ONCE over a sample pooled from ALL --dirs (one normalization for
the mixed corpus), then applied everywhere.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mir_control import CTRL_FIELDS, N_CTRL_CHANNELS, _OFFSETS, build_ctrl_array  # noqa: E402


def estimate_stats(npz_paths, sample=500, rng_seed=0):
    """Robust per-field normalization stats: center = median, scale = half the
    16-84 percentile span (~=std for a Gaussian, outlier-immune). hpcp pools all
    12 bins into one field-level stat. Fields absent everywhere get identity."""
    rng = random.Random(rng_seed)
    paths = list(npz_paths)
    if len(paths) > sample:
        paths = rng.sample(paths, sample)
    vals = {f: [] for f in CTRL_FIELDS}
    for p in paths:
        try:
            z = np.load(p)
        except Exception:
            continue
        for f in CTRL_FIELDS:
            if f in z.files:
                a = np.asarray(z[f], np.float32).ravel()
                if a.size:
                    vals[f].append(a[:: max(1, a.size // 2048)])   # subsample per crop
    stats = {}
    for f in CTRL_FIELDS:
        if not vals[f]:
            stats[f] = {"center": 0.0, "scale": 1.0, "n_crops": 0}
            continue
        a = np.concatenate(vals[f])
        a = a[np.isfinite(a)]
        center = float(np.median(a))
        lo, hi = np.percentile(a, [16, 84])
        scale = float((hi - lo) / 2.0)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0
        stats[f] = {"center": center, "scale": scale, "n_crops": len(vals[f])}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="comma-separated latent dirs")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    dirs = [Path(d.strip()) for d in a.dirs.split(",") if d.strip()]

    all_npz = [p for d in dirs for p in sorted(d.glob("*.TIMESERIES.npz"))]
    print(f"[ctrl] {len(all_npz)} timeseries across {len(dirs)} dirs")
    stats = estimate_stats(all_npz, sample=a.sample)
    for f in CTRL_FIELDS:
        s = stats[f]
        print(f"[stats] {f:32s} center={s['center']:>10.4f} scale={s['scale']:>10.4f}")

    meta = {"version": 1, "fields": CTRL_FIELDS, "n_channels": N_CTRL_CHANNELS,
            "offsets": {f: list(_OFFSETS[f]) for f in CTRL_FIELDS},
            "built": time.strftime("%Y-%m-%d %H:%M")}
    t0 = time.time()
    for d in dirs:
        outdir = Path(str(d).rstrip("/") + "_ctrl")
        outdir.mkdir(exist_ok=True)
        (outdir / "ctrl_stats.json").write_text(json.dumps(stats, indent=1))
        (outdir / "ctrl_meta.json").write_text(json.dumps(meta, indent=1))
        (d / "ctrl_stats.json").write_text(json.dumps(stats, indent=1))  # for source=timeseries
        n_done = n_skip = 0
        for npz in sorted(d.glob("*.TIMESERIES.npz")):
            out = outdir / (npz.name.replace(".TIMESERIES.npz", ".ctrl.npy"))
            if out.exists() and not a.overwrite:
                n_skip += 1
                continue
            np.save(out, build_ctrl_array(npz, stats))
            n_done += 1
            if n_done % 1000 == 0:
                print(f"[ctrl] {d.name}: {n_done} written ({time.time()-t0:.0f}s)", flush=True)
        print(f"[ctrl] {d.name}: {n_done} written, {n_skip} skipped")
    print(f"[ctrl] DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

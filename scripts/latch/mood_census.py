#!/usr/bin/env python3
# scripts/latch/mood_census.py
"""Mood census over crop companions -> a vocabulary proposal and the LatCH target specs (KUANG 2026-10-30).

Step 1 of the mood-head plan. Reads the per-crop .TIMESERIES.npz companions (after the mood fields
were added with sa3_encode_from_manifest.py --companion-only --additive) of the SAME latent roots,
in the SAME order, with the SAME --val-* settings the training run will use, and writes:

  census.json          every tag: mean probability, p95 of crop means, logit mean/std over train
                       frames, between-song std of the song-mean logit, mean within-crop std over
                       time, the between-song share, and Spearman rho of the crop-mean logit against
                       loudness (mean of the 4 rms_energy_*_ts bands) and onset density
                       (onset_envelope_ts); valence/arousal the same, plus DEAM-vs-emoMusic agreement
  mood_spec.json       target spec for the proposed tags (transform logit_z), stats from TRAIN only
  va_spec.json         target spec for --va-field (transform z), if given

The proposal is a starting point, not a verdict: tags need mass (--min-mean-prob), must differ
between songs (--min-between-std, logit units), and are ranked by between-song std. Tags that track
loudness or onset density closely (|rho| >= --redundant-rho) are KEPT but flagged: the existing
dynamics/rhythm heads may already cover them. Edit mood_spec.json by hand (drop entries from
indices/names/mean/std together) if the census says otherwise.

The split is the trainer's own (split_indices over LatCHDataset.group_keys); the spec records a hash
of the train items, and train_latch --target-spec refuses a run whose split differs.

CPU only. Run on the desktop, SA3 venv:
  python -m scripts.latch.mood_census \
      --latent-dirs /run/media/kim/Kosmos/latents/latents_sa3 /run/media/kim/Kosmos/latents/latents_avp \
      --labels /home/kim/Projects/mir/models/essentia/mtg_jamendo_moodtheme-discogs-effnet-1.json \
      --val-frac 0.05 --val-seed 0 --val-group-by song --out-dir <dir>
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.latch.latch_dataset import LatCHDataset
from scripts.latch.mood_targets import LATENT_FPS, fill_nan, raw_transform, to_time_major, train_hash

MOOD_FIELD = "effnet_moodtheme_ts"
RMS_FIELDS = ("rms_energy_bass_ts", "rms_energy_body_ts", "rms_energy_mid_ts", "rms_energy_air_ts")
ONSET_FIELD = "onset_envelope_ts"


def split_indices(groups, val_frac, seed=0):
    """train_latch.split_indices, imported lazily so the census does not need the trainer's GPU stack
    at import time on machines without it."""
    from scripts.latch.train_latch import split_indices as _split
    return _split(groups, val_frac, seed=seed)


def spearman(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(np.float64)
    rb = np.argsort(np.argsort(b[ok])).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


class FieldStats:
    """Per-crop summaries of one multi-channel field, in transformed (logit or raw) units."""

    def __init__(self, n_ch, transform):
        self.n_ch, self.transform = n_ch, transform
        self.crop_mean, self.crop_std, self.crop_prob = [], [], []
        self.sum = np.zeros(n_ch)
        self.sumsq = np.zeros(n_ch)
        self.n = 0

    def add(self, arr, in_train):
        x = fill_nan(to_time_major(arr, self.n_ch))
        y = raw_transform(x, self.transform).astype(np.float64)
        self.crop_mean.append(y.mean(axis=0))
        self.crop_std.append(y.std(axis=0))
        self.crop_prob.append(x.mean(axis=0))
        if in_train:
            self.sum += y.sum(axis=0)
            self.sumsq += (y ** 2).sum(axis=0)
            self.n += y.shape[0]

    def frame_mean_std(self):
        m = self.sum / max(self.n, 1)
        v = np.maximum(self.sumsq / max(self.n, 1) - m ** 2, 0.0)
        return m, np.sqrt(v)


def summarize(fs: FieldStats, names, groups, train_mask, loud, onset):
    cm = np.stack(fs.crop_mean)                    # (n_crops, C) transformed
    cs = np.stack(fs.crop_std)
    cp = np.stack(fs.crop_prob)
    tr = np.asarray(train_mask)
    by_song = defaultdict(list)
    for i in np.flatnonzero(tr):
        by_song[groups[i]].append(cm[i])
    song_means = np.stack([np.mean(v, axis=0) for v in by_song.values()])
    between = song_means.std(axis=0)
    within = cs[tr].mean(axis=0)
    f_mean, f_std = fs.frame_mean_std()
    rows = []
    for c, name in enumerate(names):
        rows.append({
            "index": c, "name": name,
            "mean_prob": float(cp[tr, c].mean()),
            "p95_crop_prob": float(np.percentile(cp[tr, c], 95)),
            "frame_mean": float(f_mean[c]), "frame_std": float(f_std[c]),
            "between_song_std": float(between[c]),
            "within_crop_std": float(within[c]),
            "between_share": float(between[c] ** 2 / (between[c] ** 2 + within[c] ** 2 + 1e-12)),
            "rho_loudness": spearman(cm[tr, c], loud[tr]),
            "rho_onset": spearman(cm[tr, c], onset[tr]),
        })
    return rows, song_means.shape[0]


def make_spec(field, transform, rows, chosen, n_in, split, smooth_s, extra=None):
    sel = [r for r in rows if r["index"] in chosen]
    sel.sort(key=lambda r: chosen.index(r["index"]))
    spec = {
        "field": field, "transform": transform, "n_channels_in": n_in,
        "indices": [r["index"] for r in sel], "names": [r["name"] for r in sel],
        "mean": [r["frame_mean"] for r in sel],
        "std": [max(r["frame_std"], 1e-6) for r in sel],
        "smooth_s": smooth_s, "fps": LATENT_FPS, "split": split,
        "made_by": "scripts/latch/mood_census.py", "made_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    spec.update(extra or {})
    return spec


def propose(rows, min_mean_prob, min_between, max_tags, exclude):
    ok = [r for r in rows if r["name"] not in exclude and r["mean_prob"] >= min_mean_prob
          and r["between_song_std"] >= min_between]
    ok.sort(key=lambda r: -r["between_song_std"])
    return [r["index"] for r in ok[:max_tags]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latent-dirs", nargs="+", required=True, help="same roots, same order as the training run")
    ap.add_argument("--labels", required=True, help="mtg_jamendo_moodtheme-discogs-effnet-1.json")
    ap.add_argument("--field", default=MOOD_FIELD)
    ap.add_argument("--va-field", default="va_deam_ts", help="'' to skip valence/arousal")
    ap.add_argument("--va-compare", default="va_emomusic_ts", help="second V/A estimate for agreement; '' to skip")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--val-group-by", default="song")
    ap.add_argument("--min-mean-prob", type=float, default=0.01)
    ap.add_argument("--min-between-std", type=float, default=0.35, help="logit units")
    ap.add_argument("--max-tags", type=int, default=16)
    ap.add_argument("--redundant-rho", type=float, default=0.8)
    ap.add_argument("--exclude", default="", help="comma list of tags never proposed")
    ap.add_argument("--smooth-s", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0, help="first N crops only (smoke; the spec is then NOT valid for training)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    labels = json.loads(Path(args.labels).read_text())["classes"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roots = args.latent_dirs if len(args.latent_dirs) > 1 else args.latent_dirs[0]
    ds = LatCHDataset(roots, target_feature=args.field, target_source="npz")
    items = ds.items[: args.limit] if args.limit else ds.items
    groups_all = ds.group_keys(args.val_group_by)
    tr_idx, va_idx = split_indices(groups_all, args.val_frac, seed=args.val_seed)
    split = {"latent_dirs": [Path(d).name for d in args.latent_dirs], "n_items": len(ds.items),
             "val_frac": args.val_frac, "val_seed": args.val_seed, "val_group_by": args.val_group_by,
             "n_train_crops": len(tr_idx), "train_hash": train_hash(ds.items, tr_idx)}
    train_set = set(tr_idx)
    groups = groups_all[: len(items)]
    print(f"[census] {len(items)} crops ({len(tr_idx)} train / {len(va_idx)} val of {len(ds.items)}), "
          f"{len(set(groups_all))} {args.val_group_by} groups", flush=True)

    mood = FieldStats(len(labels), "logit_z")
    va = FieldStats(2, "z") if args.va_field else None
    va2_means, va_present = [], []
    loud, onset, missing = [], [], defaultdict(int)
    t0 = time.time()
    for i, p in enumerate(items):
        with np.load(str(p.with_suffix(".TIMESERIES.npz"))) as z:
            mood.add(z[args.field], i in train_set)
            if va is not None:
                present = args.va_field in z.files
                va_present.append(present)
                if present:
                    va.add(z[args.va_field], i in train_set)
                else:                                   # placeholder row keeps crops aligned; masked out below
                    missing[args.va_field] += 1
                    va.add(np.zeros((1, 2), np.float32), False)
            if args.va_compare:
                va2_means.append(np.nanmean(to_time_major(z[args.va_compare], 2), axis=0)
                                 if args.va_compare in z.files else np.full(2, np.nan))
            rms = [float(np.nanmean(z[f])) for f in RMS_FIELDS if f in z.files]
            loud.append(np.mean(rms) if rms else np.nan)
            onset.append(float(np.nanmean(z[ONSET_FIELD])) if ONSET_FIELD in z.files else np.nan)
        if (i + 1) % 500 == 0:
            print(f"[census] {i + 1}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    loud, onset = np.asarray(loud), np.asarray(onset)
    train_mask = [i in train_set for i in range(len(items))]

    rows, n_songs = summarize(mood, labels, groups, train_mask, loud, onset)
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    chosen = propose(rows, args.min_mean_prob, args.min_between_std, args.max_tags, exclude)
    for r in rows:
        r["proposed"] = r["index"] in chosen
        r["redundant"] = bool(max(abs(np.nan_to_num(r["rho_loudness"])), abs(np.nan_to_num(r["rho_onset"])))
                              >= args.redundant_rho)
    report = {"field": args.field, "n_crops": len(items), "n_train_songs": n_songs, "split": split,
              "missing_fields": dict(missing), "thresholds": vars(args), "tags": rows}

    if va is not None:
        pres = np.asarray(va_present, bool)
        va_rows, _ = summarize(va, ["valence", "arousal"], groups, list(np.asarray(train_mask) & pres), loud, onset)
        report["va"] = va_rows
        if args.va_compare and va2_means:
            a, b = np.stack(va.crop_mean)[pres], np.stack(va2_means)[pres]
            report["va_agreement_rho"] = {n: spearman(a[:, c], b[:, c]) for c, n in enumerate(["valence", "arousal"])}
        va_spec = make_spec(args.va_field, "z", va_rows, [0, 1], 2, split, args.smooth_s)
        if not args.limit:
            (out / "va_spec.json").write_text(json.dumps(va_spec, indent=1))

    (out / "census.json").write_text(json.dumps(report, indent=1, default=float))
    if not args.limit:
        spec = make_spec(args.field, "logit_z", rows, chosen, len(labels), split, args.smooth_s)
        (out / "mood_spec.json").write_text(json.dumps(spec, indent=1))

    print(f"\n| tag | mean p | between-song sd | within-crop sd | between share | rho loud | rho onset | |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in sorted(rows, key=lambda r: -r["between_song_std"]):
        if r["mean_prob"] < args.min_mean_prob / 2 and not r["proposed"]:
            continue
        flag = ("**proposed**" if r["proposed"] else "") + (" redundant?" if r["redundant"] else "")
        print(f"| {r['name']} | {r['mean_prob']:.3f} | {r['between_song_std']:.2f} | {r['within_crop_std']:.2f} | "
              f"{r['between_share']:.2f} | {r['rho_loudness']:+.2f} | {r['rho_onset']:+.2f} | {flag} |")
    if va is not None:
        for r in report["va"]:
            print(f"| {r['name']} (V/A) | — | {r['between_song_std']:.2f} | {r['within_crop_std']:.2f} | "
                  f"{r['between_share']:.2f} | {r['rho_loudness']:+.2f} | {r['rho_onset']:+.2f} | |")
        if "va_agreement_rho" in report:
            print(f"DEAM vs emoMusic crop-mean rho: {report['va_agreement_rho']}")
    print(f"\n[census] proposed {len(chosen)} tags: {[labels[i] for i in chosen]}")
    print(f"[census] wrote {out}/census.json" + ("" if args.limit else ", mood_spec.json" + (", va_spec.json" if va else "")))
    if args.limit:
        print("[census] --limit set: no spec written (stats from a partial split are not valid for training)")
    return report


if __name__ == "__main__":
    main()

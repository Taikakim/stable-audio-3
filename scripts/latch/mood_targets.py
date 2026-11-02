# scripts/latch/mood_targets.py
"""Mood / valence-arousal LatCH targets: a TARGET SPEC turns a stored multi-channel field into the
per-frame target the head learns (KUANG 2026-10-30).

Why a spec rather than the raw field: effnet_moodtheme_ts holds 56 MTG-Jamendo sigmoid scores that
are low and badly calibrated (on goa the strongest tag averages 0.23 and most sit near 0), and
train_latch's --standardize is ONE scalar mean/std across every channel. So the spec carries
  - which channels (tags) to keep,
  - a transform: "logit_z" for probabilities (log-odds, then per-channel z-score) or "z" for
    already-continuous fields such as va_deam_ts (1-9 valence/arousal),
  - per-channel mean/std computed by mood_census.py on the TRAIN split only,
  - a moving-average smoothing window in seconds (mood is a ~1 Hz, slowly varying field),
  - the split it was computed on, so the trainer can refuse a different split (leaked stats).
"+2" on a channel then means two standard deviations above the training corpus, and the same
spec (saved into the checkpoint) turns a UI value into a target at inference.

The resampled crop companions store 2-D fields as (T=4096, C); (C, T) is accepted too.
"""
import json
import warnings
from pathlib import Path

import numpy as np

LATENT_FPS = 44100 / 4096          # SAME latent frame rate, ~10.77 Hz
PROB_EPS = 1e-4                    # logit clip: sigmoid scores of exactly 0/1 occur after pooling


def train_hash(items, train_idx) -> str:
    """Identity of the training items (root folder name + crop stem), independent of where the
    drive is mounted. mood_census.py stores it in the spec; train_latch compares."""
    import hashlib
    h = hashlib.sha1()
    for i in train_idx:
        h.update(f"{Path(items[i]).parent.name}/{Path(items[i]).stem}\n".encode())
    return h.hexdigest()


def check_split(spec: dict, items, train_idx, val_frac, val_seed, val_group_by):
    """Refuse a training split other than the one the spec's mean/std came from: stats taken
    over crops that are now held out would leak val data into the target scale."""
    s = spec.get("split") or {}
    want = (s.get("val_frac"), s.get("val_seed"), s.get("val_group_by"), s.get("train_hash"))
    have = (val_frac, val_seed, val_group_by, train_hash(items, train_idx))
    if want != have:
        raise SystemExit(
            f"--target-spec stats were computed on another split: spec has val_frac={want[0]} "
            f"val_seed={want[1]} val_group_by={want[2]} ({s.get('n_train_crops')} train crops of "
            f"{s.get('latent_dirs')}); this run has val_frac={have[0]} val_seed={have[1]} "
            f"val_group_by={have[2]} ({len(train_idx)} train crops). Re-run mood_census.py with this "
            f"run's latent dirs and --val-* settings.")


def load_spec(path) -> dict:
    spec = json.loads(Path(path).read_text())
    for k in ("field", "transform", "indices", "names", "mean", "std", "n_channels_in"):
        if k not in spec:
            raise ValueError(f"target spec {path} has no {k!r}")
    if spec["transform"] not in ("logit_z", "z"):
        raise ValueError(f"unknown transform {spec['transform']!r}")
    n = len(spec["indices"])
    if not (len(spec["names"]) == len(spec["mean"]) == len(spec["std"]) == n):
        raise ValueError(f"target spec {path}: indices/names/mean/std lengths differ")
    return spec


def to_time_major(arr: np.ndarray, n_channels: int) -> np.ndarray:
    """(T, C) float32 from a stored (T, C) or (C, T) field with C == n_channels."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[1] == n_channels:
        return a
    if a.shape[0] == n_channels:
        return a.T
    raise ValueError(f"field shape {a.shape} has no axis of {n_channels} channels")


def raw_transform(x: np.ndarray, transform: str) -> np.ndarray:
    """The pre-standardization transform, (T, C) -> (T, C)."""
    if transform == "logit_z":
        p = np.clip(x, PROB_EPS, 1.0 - PROB_EPS)
        return np.log(p) - np.log1p(-p)
    return x


def fill_nan(x: np.ndarray) -> np.ndarray:
    """NaN frames -> that channel's mean over the crop (0 if the whole channel is NaN)."""
    if not np.isnan(x).any():
        return x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN channel -> NaN mean, set to 0 below
        m = np.nanmean(x, axis=0)
    m = np.where(np.isnan(m), 0.0, m)
    return np.where(np.isnan(x), m[None, :], x).astype(np.float32)


def smooth_time(x: np.ndarray, frames: int) -> np.ndarray:
    """Centered moving average over time with edge padding, (T, C) -> (T, C)."""
    if frames <= 1 or x.shape[0] <= 1:
        return x
    k = min(int(frames), x.shape[0])
    lo, hi = (k - 1) // 2, k // 2
    pad = np.pad(x, ((lo, hi), (0, 0)), mode="edge")
    c = np.cumsum(np.concatenate([np.zeros((1, x.shape[1]), np.float64), pad.astype(np.float64)]), axis=0)
    return ((c[k:] - c[:-k]) / k).astype(np.float32)


def apply_spec(arr: np.ndarray, spec: dict) -> np.ndarray:
    """Stored field -> (C_out, T) target in the spec's standardized units."""
    x = to_time_major(arr, int(spec["n_channels_in"]))[:, spec["indices"]]
    x = raw_transform(fill_nan(x), spec["transform"])
    x = (x - np.asarray(spec["mean"], np.float32)[None, :]) / np.asarray(spec["std"], np.float32)[None, :]
    frames = int(round(float(spec.get("smooth_s", 0.0)) * float(spec.get("fps", LATENT_FPS))))
    return np.ascontiguousarray(smooth_time(x.astype(np.float32), frames).T)


def target_from_values(spec: dict, values: dict, n_frames: int) -> np.ndarray:
    """Constant (C_out, n_frames) target from {name: standardized value}; unnamed channels = 0
    (the corpus mean). For inference: {"dark": 2.0} asks for +2 SD of dark, rest typical."""
    unknown = set(values) - set(spec["names"])
    if unknown:
        raise ValueError(f"not in this spec: {sorted(unknown)}; channels are {spec['names']}")
    v = np.array([float(values.get(n, 0.0)) for n in spec["names"]], np.float32)
    return np.repeat(v[:, None], n_frames, axis=1)

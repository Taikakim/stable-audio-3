# scripts/latch/latch_dataset.py
"""LatCH dataset for SA3: SAME latents (256xT) + MIR target resampled to each clip's T."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.latch.latch_targets import resample_target


def _open_default_db():
    sys.path.insert(0, "/home/kim/Projects/mir/src")
    from core.timeseries_db import TimeseriesDB, DEFAULT_DB_PATH
    return TimeseriesDB.open(DEFAULT_DB_PATH)


class LatCHDataset(Dataset):
    """Latent + per-frame MIR target pairs.

    target_source:
      "db"  — targets from the legacy per-crop TimeseriesDB, keyed by latent stem
              (small-music-base / phase-1 layout).
      "npz" — targets from a <stem>.TIMESERIES.npz companion next to each .npy
              (the beat-aligned latents_sa3 layout: SAME-L latents 256x4096 + a
              21-field timeseries companion already sliced/resampled to T=4096).
    """

    def __init__(self, latent_dir: str, target_feature: str = "rms_energy_bass",
                 db=None, db_path: Optional[str] = None, target_source: str = "db",
                 chroma_dir: Optional[str] = None, chroma_key: str = "other"):
        self.latent_dir = Path(latent_dir)
        self.bare_feature = target_feature.removesuffix("_ts")
        self.ts_feature = self.bare_feature + "_ts"
        self.target_source = target_source
        self.chroma_dir = Path(chroma_dir) if chroma_dir else None
        self.chroma_key = chroma_key   # which stem's SAME-chroma: other / bass / full_mix
        self.items = sorted(p for p in self.latent_dir.glob("*.npy")
                            if p.stem != "silence")
        if not self.items:
            raise RuntimeError(f"No .npy latents in {latent_dir}")
        self._db = None
        if target_source == "db":
            if db is not None:
                self._db = db
            elif db_path is not None:
                sys.path.insert(0, "/home/kim/Projects/mir/src")
                from core.timeseries_db import TimeseriesDB
                self._db = TimeseriesDB.open(db_path)
            else:
                self._db = _open_default_db()
        elif target_source == "npz":
            # Keep only items whose companion npz has the requested field.
            kept = []
            for p in self.items:
                npz = p.with_suffix(".TIMESERIES.npz")
                if npz.exists():
                    kept.append(p)
            if not kept:
                raise RuntimeError(
                    f"target_source='npz' but no *.TIMESERIES.npz companions in {latent_dir}")
            self.items = kept
        elif target_source == "chroma":
            # SAME-compatible (3,128,T) chroma from a separate per-crop npz (stem-resolved).
            if self.chroma_dir is None:
                raise ValueError("target_source='chroma' requires chroma_dir")
            kept = [p for p in self.items if (self.chroma_dir / (p.stem + ".npz")).exists()]
            if not kept:
                raise RuntimeError(
                    f"target_source='chroma' but no <stem>.npz in {self.chroma_dir}")
            self.items = kept
        else:
            raise ValueError(f"unknown target_source={target_source!r}")

    def __len__(self):
        return len(self.items)

    def _load_target(self, npy_path: Path, t_frames: int) -> np.ndarray:
        if self.target_source == "chroma":
            with np.load(str(self.chroma_dir / (npy_path.stem + ".npz"))) as z:
                if self.chroma_key not in z.files:
                    raise ValueError(f"{self.chroma_key} not in chroma npz for {npy_path.stem}")
                arr = z[self.chroma_key].astype(np.float32)              # (3, 128, T)
            arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])  # (384, T) band-major
            return resample_target(arr, t_frames)
        if self.target_source == "npz":
            with np.load(str(npy_path.with_suffix(".TIMESERIES.npz"))) as z:
                if self.ts_feature not in z.files:
                    raise ValueError(f"{self.ts_feature} not in {npy_path.stem}.TIMESERIES.npz")
                arr = z[self.ts_feature]
            return resample_target(arr, t_frames)
        arrays = self._db.get(npy_path.stem)
        if arrays is None or arrays.get(self.ts_feature) is None:
            raise ValueError(f"{self.ts_feature} missing for {npy_path.stem}")
        return resample_target(arrays[self.ts_feature], t_frames)

    def __getitem__(self, idx):
        start = idx
        while True:
            npy_path = self.items[idx]
            try:
                latent = np.load(str(npy_path)).astype(np.float32)  # (256, T)
                t_frames = latent.shape[1]
                target = self._load_target(npy_path, t_frames)  # (C, T)
                return torch.from_numpy(latent), torch.from_numpy(target)
            except Exception:
                idx = (idx + 1) % len(self.items)
                if idx == start:
                    raise RuntimeError("No valid items in dataset.")


def collate_varlen(batch):
    """Pad a list of (latent (256,T), target (C,T)) to the max T; return a length mask."""
    max_t = max(lat.shape[1] for lat, _ in batch)
    c_out = batch[0][1].shape[0]
    latents = torch.zeros(len(batch), 256, max_t)
    targets = torch.zeros(len(batch), c_out, max_t)
    mask = torch.zeros(len(batch), max_t, dtype=torch.bool)
    for i, (lat, tgt) in enumerate(batch):
        t = lat.shape[1]
        latents[i, :, :t] = lat
        targets[i, :, :t] = tgt
        mask[i, :t] = True
    return {"latents": latents, "targets": targets, "mask": mask}

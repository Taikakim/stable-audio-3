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
    def __init__(self, latent_dir: str, target_feature: str = "rms_energy_bass",
                 db=None, db_path: Optional[str] = None):
        self.latent_dir = Path(latent_dir)
        self.bare_feature = target_feature.removesuffix("_ts")
        self.ts_feature = self.bare_feature + "_ts"
        self.items = sorted(p for p in self.latent_dir.glob("*.npy"))
        if not self.items:
            raise RuntimeError(f"No .npy latents in {latent_dir}")
        if db is not None:
            self._db = db
        elif db_path is not None:
            sys.path.insert(0, "/home/kim/Projects/mir/src")
            from core.timeseries_db import TimeseriesDB
            self._db = TimeseriesDB.open(db_path)
        else:
            self._db = _open_default_db()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        start = idx
        while True:
            npy_path = self.items[idx]
            try:
                latent = np.load(str(npy_path)).astype(np.float32)  # (256, T)
                t_frames = latent.shape[1]
                arrays = self._db.get(npy_path.stem)
                if arrays is None or arrays.get(self.ts_feature) is None:
                    raise ValueError(f"{self.ts_feature} missing for {npy_path.stem}")
                target = resample_target(arrays[self.ts_feature], t_frames)  # (C, T)
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

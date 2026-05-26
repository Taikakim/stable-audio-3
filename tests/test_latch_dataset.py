# tests/test_latch_dataset.py
import json
import numpy as np
import torch

from scripts.latch.latch_dataset import LatCHDataset, collate_varlen


class FakeDB:
    """Stands in for mir TimeseriesDB: returns a 256-frame series per key."""
    def __init__(self, store):
        self._store = store
    def get(self, key):
        return self._store.get(key)


def _write_clip(tmp_path, stem, frames):
    np.save(tmp_path / f"{stem}.npy", np.zeros((256, frames), dtype=np.float32))
    (tmp_path / f"{stem}.json").write_text(
        json.dumps({"crop_key": stem, "latent_frames": frames, "seconds": frames * 0.0929})
    )


def test_item_resamples_target_to_latent_frames(tmp_path):
    _write_clip(tmp_path, "A - T_0", 128)
    db = FakeDB({"A - T_0": {"rms_energy_bass_ts": np.linspace(-60, -10, 256).astype(np.float32)}})
    ds = LatCHDataset(str(tmp_path), target_feature="rms_energy_bass", db=db)
    latent, target = ds[0]
    assert latent.shape == (256, 128)
    assert target.shape == (1, 128)          # resampled from 256 -> 128


def test_collate_pads_and_masks(tmp_path):
    _write_clip(tmp_path, "A - T_0", 100)
    _write_clip(tmp_path, "B - T_0", 150)
    db = FakeDB({
        "A - T_0": {"rms_energy_bass_ts": np.zeros(256, dtype=np.float32)},
        "B - T_0": {"rms_energy_bass_ts": np.ones(256, dtype=np.float32)},
    })
    ds = LatCHDataset(str(tmp_path), target_feature="rms_energy_bass", db=db)
    batch = collate_varlen([ds[0], ds[1]])
    assert batch["latents"].shape == (2, 256, 150)
    assert batch["targets"].shape == (2, 1, 150)
    assert batch["mask"].shape == (2, 150)
    # First item (len 100) has 100 valid then 50 padded frames.
    assert batch["mask"][0, :100].all() and not batch["mask"][0, 100:].any()
    assert batch["mask"][1].all()

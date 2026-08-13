# tests/test_latch_dataset.py
import json
import numpy as np
import pytest
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
    latent, target, weight = ds[0]
    assert latent.shape == (256, 128)
    assert target.shape == (1, 128)          # resampled from 256 -> 128
    assert weight.shape == (1, 128)
    assert torch.allclose(weight, torch.ones(1, 128))  # no voiced_field -> all-ones


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
    assert batch["weight"].shape == (2, 150)
    # First item (len 100) has 100 valid then 50 padded frames.
    assert batch["mask"][0, :100].all() and not batch["mask"][0, 100:].any()
    assert batch["mask"][1].all()
    # No voiced_field requested -> weight is all-ones over the valid region.
    assert torch.allclose(batch["weight"][0, :100], torch.ones(100))
    assert torch.allclose(batch["weight"][1], torch.ones(150))


def _write_npz_clip(tmp_path, stem, frames, f0, voiced):
    np.save(tmp_path / f"{stem}.npy", np.zeros((256, frames), dtype=np.float32))
    np.savez(tmp_path / f"{stem}.TIMESERIES.npz",
             f0_other_ts=f0.astype(np.float32), f0_other_voiced_ts=voiced.astype(np.float32))


def test_voiced_field_loads_as_soft_weight(tmp_path):
    # voiced_ts is the POOLED VOICED FRACTION per crop-frame (W's resampler convention,
    # 0.0-1.0 continuous, not boolean) -- the melody head's per-frame loss weight.
    f0 = np.linspace(50.0, 70.0, 64)
    voiced = np.array([1.0] * 32 + [0.0] * 32)   # second half unvoiced -> weight 0
    _write_npz_clip(tmp_path, "A - T_0", 64, f0, voiced)
    ds = LatCHDataset(str(tmp_path), target_feature="f0_other", target_source="npz",
                       voiced_field="f0_other_voiced_ts")
    latent, target, weight = ds[0]
    assert weight.shape == (1, 64)
    assert torch.allclose(weight[0, :32], torch.ones(32))
    assert torch.allclose(weight[0, 32:], torch.zeros(32))


def test_voiced_field_requires_presence_in_npz(tmp_path):
    # An item missing the requested voiced field must be dropped, same as a missing target
    # field -- silently falling back to all-ones would hide unweighted crops in a mixed corpus.
    np.save(tmp_path / "A - T_0.npy", np.zeros((256, 32), dtype=np.float32))
    np.savez(tmp_path / "A - T_0.TIMESERIES.npz", f0_other_ts=np.zeros(32, dtype=np.float32))
    with pytest.raises(RuntimeError):
        LatCHDataset(str(tmp_path), target_feature="f0_other", target_source="npz",
                     voiced_field="f0_other_voiced_ts")


def test_collate_weight_survives_padding(tmp_path):
    voiced_a = np.array([1.0] * 50 + [0.5] * 50)   # 100 frames, partial confidence in 2nd half
    _write_npz_clip(tmp_path, "A - T_0", 100, np.zeros(100), voiced_a)
    _write_npz_clip(tmp_path, "B - T_0", 150, np.zeros(150), np.ones(150))
    ds = LatCHDataset(str(tmp_path), target_feature="f0_other", target_source="npz",
                       voiced_field="f0_other_voiced_ts")
    batch = collate_varlen([ds[0], ds[1]])
    assert batch["weight"].shape == (2, 150)
    assert torch.allclose(batch["weight"][0, :50], torch.ones(50))
    assert torch.allclose(batch["weight"][0, 50:100], torch.full((50,), 0.5))
    assert torch.allclose(batch["weight"][1, :150], torch.ones(150))

# tests/test_latch_val_split.py
"""Held-out validation for LatCH heads — the split must be by SOURCE TRACK, not by crop.

Why this file exists (C, 2026-08-18): every LatCH head we have, including the f0 melody pilot,
was model-selected on TRAINING loss — train_latch.py had no validation path at all, and
`_best.pt` was whichever epoch had the lowest training loss. That is not a convergence claim.

And a naive random split cannot fix it here: in `latents_sa3` EVERY one of the 2676 source
tracks contributes >=2 crops (median 2, max 4). Splitting at crop level therefore puts a
sibling crop of essentially every val track into train. For an f0/melody target that is the
worst possible leak — sibling crops of one goa track share key, lead patch, and often literal
repeated loop material, so a leaked val loss would look excellent and mean nothing.
"""
import json

import numpy as np
import pytest

from scripts.latch.latch_dataset import LatCHDataset
from scripts.latch.train_latch import split_indices


def _write_clip(tmp_path, stem, frames, source_track=None):
    np.save(tmp_path / f"{stem}.npy", np.zeros((256, frames), dtype=np.float32))
    meta = {"crop_key": stem, "latent_frames": frames}
    if source_track is not None:
        meta["source_track"] = source_track
    (tmp_path / f"{stem}.json").write_text(json.dumps(meta))
    np.savez(tmp_path / f"{stem}.TIMESERIES.npz",
             f0_other_ts=np.linspace(100, 200, 256).astype(np.float32),
             f0_other_voiced_ts=np.ones(256, dtype=np.float32))


# --- the split itself -------------------------------------------------------------------

def test_no_group_appears_in_both_splits():
    """The whole point: a track's crops land entirely on one side."""
    groups = [f"track{i//3}" for i in range(60)]      # 20 tracks x 3 crops
    tr, va = split_indices(groups, val_frac=0.2, seed=0)
    assert set(tr).isdisjoint(va)
    tr_g = {groups[i] for i in tr}
    va_g = {groups[i] for i in va}
    assert tr_g.isdisjoint(va_g), "a source track was split across train and val — leak"


def test_every_item_is_assigned_exactly_once():
    groups = [f"track{i//3}" for i in range(60)]
    tr, va = split_indices(groups, val_frac=0.25, seed=0)
    assert sorted(tr + va) == list(range(60))


def test_val_fraction_is_approximately_honoured_by_group():
    groups = [f"track{i//2}" for i in range(200)]     # 100 tracks x 2 crops
    _tr, va = split_indices(groups, val_frac=0.2, seed=0)
    held = {groups[i] for i in va}
    assert 15 <= len(held) <= 25, f"expected ~20 of 100 tracks held out, got {len(held)}"


def test_split_is_deterministic_for_a_seed():
    groups = [f"track{i//3}" for i in range(60)]
    assert split_indices(groups, 0.2, seed=7) == split_indices(groups, 0.2, seed=7)


def test_different_seeds_give_different_splits():
    groups = [f"track{i//2}" for i in range(200)]
    a = split_indices(groups, 0.2, seed=1)[1]
    b = split_indices(groups, 0.2, seed=2)[1]
    assert a != b


def test_val_frac_zero_keeps_every_item_in_train():
    """Byte-identical behaviour for the 14 production heads that do not opt in."""
    groups = [f"track{i//3}" for i in range(60)]
    tr, va = split_indices(groups, val_frac=0.0, seed=0)
    assert tr == list(range(60))
    assert va == []


def test_raises_when_val_frac_would_hold_out_nothing():
    """Fail loud rather than silently reporting a val loss over an empty set."""
    groups = ["only_one_track"] * 4
    with pytest.raises(ValueError, match="val"):
        split_indices(groups, val_frac=0.2, seed=0)


# --- group keys off the real sidecar layout ---------------------------------------------

def test_dataset_exposes_source_track_as_group_key(tmp_path):
    _write_clip(tmp_path, "000000", 128, source_track="Artist - Alpha")
    _write_clip(tmp_path, "000001", 128, source_track="Artist - Alpha")
    _write_clip(tmp_path, "000002", 128, source_track="Artist - Beta")
    ds = LatCHDataset(str(tmp_path), target_feature="f0_other", target_source="npz")
    assert ds.group_keys("source_track") == ["Artist - Alpha", "Artist - Alpha", "Artist - Beta"]


def test_group_keys_none_gives_one_group_per_crop(tmp_path):
    """--val-group-by none is the documented, explicitly-opted-into leaky mode."""
    _write_clip(tmp_path, "000000", 128, source_track="Artist - Alpha")
    _write_clip(tmp_path, "000001", 128, source_track="Artist - Alpha")
    ds = LatCHDataset(str(tmp_path), target_feature="f0_other", target_source="npz")
    assert len(set(ds.group_keys("none"))) == 2


def test_missing_group_field_fails_loud(tmp_path):
    """A crop with no source_track must not silently become its own group — that is the
    leak this whole module exists to prevent, reintroduced quietly."""
    _write_clip(tmp_path, "000000", 128, source_track="Artist - Alpha")
    _write_clip(tmp_path, "000001", 128, source_track=None)
    ds = LatCHDataset(str(tmp_path), target_feature="f0_other", target_source="npz")
    with pytest.raises(RuntimeError, match="source_track"):
        ds.group_keys("source_track")

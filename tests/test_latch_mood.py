# tests/test_latch_mood.py -- KUANG 2026-10-30
# Mood / valence-arousal target specs, the dataset's npz field filter, the census, and the split check.
import json

import numpy as np
import pytest

from scripts.latch import mood_targets as MT
from scripts.latch.latch_dataset import LatCHDataset

T = 64
N_TAGS = 56
LABELS = [f"tag{i}" for i in range(N_TAGS)]
DARK, SPACE, NOISE = 3, 7, 11          # planted: dark differs between songs, space within songs


def _spec(indices=(DARK, SPACE), transform="logit_z", smooth_s=0.0):
    return {"field": "effnet_moodtheme_ts", "transform": transform, "n_channels_in": N_TAGS,
            "indices": list(indices), "names": [LABELS[i] for i in indices],
            "mean": [0.0] * len(indices), "std": [1.0] * len(indices), "smooth_s": smooth_s}


def test_apply_spec_selects_logits_and_standardizes():
    p = np.full((T, N_TAGS), 0.01, np.float32)
    p[:, DARK] = 0.5
    p[:, SPACE] = 0.9
    spec = _spec()
    spec["mean"], spec["std"] = [0.0, 1.0], [1.0, 2.0]
    out = MT.apply_spec(p, spec)
    assert out.shape == (2, T)
    assert np.allclose(out[0], 0.0, atol=1e-6)                         # logit(0.5) = 0
    assert np.allclose(out[1], (np.log(9.0) - 1.0) / 2.0, atol=1e-5)   # (logit(0.9) - 1) / 2
    assert np.allclose(MT.apply_spec(p.T, spec), out)                   # (C, T) storage accepted


def test_nan_fill_clip_and_smoothing():
    p = np.full((T, N_TAGS), 0.2, np.float32)
    p[10:20, DARK] = np.nan
    p[:, SPACE] = 0.0                                                   # clipped, not -inf
    out = MT.apply_spec(p, _spec(smooth_s=1.0))
    assert np.isfinite(out).all()
    assert np.allclose(out[0], np.log(0.25), atol=1e-5)                 # NaN -> the crop's mean prob
    step = np.zeros((T, 1), np.float32)
    step[T // 2:] = 1.0
    sm = MT.smooth_time(step, 9)
    naive = np.array([step[max(0, i - 4):i + 5, 0].mean() if 4 <= i < T - 4 else sm[i, 0] for i in range(T)])
    assert np.allclose(sm[:, 0], naive, atol=1e-6) and sm.shape == step.shape
    assert np.allclose(MT.smooth_time(np.ones((T, 2), np.float32), 9), 1.0)


def test_target_from_values():
    spec = _spec()
    t = MT.target_from_values(spec, {LABELS[DARK]: 2.0}, 5)
    assert t.shape == (2, 5) and np.all(t[0] == 2.0) and np.all(t[1] == 0.0)
    with pytest.raises(ValueError, match="not in this spec"):
        MT.target_from_values(spec, {"happy": 1.0}, 5)


def _corpus(root, songs, rng, va=True, skip_field=()):
    """Crops with mood (T, 56) probs, V/A, rms and onset companions. songs: {name: (dark_p, n_crops)}."""
    root.mkdir()
    i = 0
    for song, (dark_p, n) in songs.items():
        for _ in range(n):
            stem = f"{i:06d}"
            np.save(root / f"{stem}.npy", np.zeros((256, T), np.float32))
            (root / f"{stem}.json").write_text(json.dumps({"source_track": song}))
            comp = {"rms_energy_bass_ts": rng.normal(-20, 1, T).astype(np.float32),
                    "onset_envelope_ts": rng.random(T).astype(np.float32)}
            if stem not in skip_field:
                p = np.full((T, N_TAGS), 1e-3, np.float32)
                p[:, DARK] = dark_p * (1 + 0.05 * rng.standard_normal(T))
                p[:, SPACE] = np.where(np.arange(T) < T // 2, 0.05, 0.6)  # moves within every crop
                p[:, NOISE] = 0.02 + 0.001 * rng.random(T)
                comp["effnet_moodtheme_ts"] = np.clip(p, 0, 1)
                if va:
                    comp["va_deam_ts"] = np.stack([np.full(T, 5 + dark_p), np.full(T, 4.0)], 1).astype(np.float32)
                    comp["va_emomusic_ts"] = comp["va_deam_ts"] + 0.1
            np.savez(root / f"{stem}.TIMESERIES.npz", **comp)
            i += 1


def test_dataset_applies_spec_and_drops_crops_without_the_field(tmp_path, capsys):
    rng = np.random.default_rng(0)
    _corpus(tmp_path / "c", {"A": (0.3, 2), "B": (0.05, 2)}, rng, skip_field=("000001",))
    ds = LatCHDataset(str(tmp_path / "c"), target_feature="effnet_moodtheme", target_source="npz",
                      target_spec=_spec())
    assert [p.stem for p in ds.items] == ["000000", "000002", "000003"]
    assert "1 crops have a companion without effnet_moodtheme_ts" in capsys.readouterr().out
    lat, tgt, w = ds[0]
    assert tgt.shape == (2, T)
    with pytest.raises(ValueError, match="target spec is for"):
        LatCHDataset(str(tmp_path / "c"), target_feature="va_deam", target_source="npz", target_spec=_spec())


def test_census_proposes_the_planted_tag_and_writes_train_only_stats(tmp_path):
    from scripts.latch import mood_census
    rng = np.random.default_rng(1)
    songs = {f"S{k}": (0.02 + 0.08 * k, 3) for k in range(10)}       # dark rises song by song
    _corpus(tmp_path / "goa", songs, rng)
    _corpus(tmp_path / "avp", {f"V{k}": (0.05, 2) for k in range(4)}, rng)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"classes": LABELS}))
    out = tmp_path / "out"
    rep = mood_census.main(["--latent-dirs", str(tmp_path / "goa"), str(tmp_path / "avp"),
                            "--labels", str(labels), "--val-frac", "0.2", "--val-group-by", "song",
                            "--min-between-std", "0.35", "--out-dir", str(out)])
    by = {r["name"]: r for r in rep["tags"]}
    assert by[LABELS[DARK]]["proposed"]                              # mass + between-song spread
    assert not by[LABELS[SPACE]]["proposed"]                         # moves within crops, same per song
    assert by[LABELS[SPACE]]["within_crop_std"] > 1.0
    assert not by[LABELS[NOISE]]["proposed"]                         # mass but no spread
    spec = MT.load_spec(out / "mood_spec.json")
    assert spec["names"][0] == LABELS[DARK] and spec["transform"] == "logit_z"
    va = MT.load_spec(out / "va_spec.json")
    assert va["names"] == ["valence", "arousal"] and va["transform"] == "z"
    assert rep["va_agreement_rho"]["valence"] > 0.9

    # stats are over TRAIN frames only, and the trainer's split check accepts the same split
    from scripts.latch.train_latch import split_indices
    ds = LatCHDataset([str(tmp_path / "goa"), str(tmp_path / "avp")], target_feature="effnet_moodtheme",
                      target_source="npz")
    tr, va_idx = split_indices(ds.group_keys("song"), 0.2, seed=0)
    logits = []
    for i in tr:
        with np.load(ds.items[i].with_suffix(".TIMESERIES.npz")) as z:
            p = np.clip(z["effnet_moodtheme_ts"][:, DARK], MT.PROB_EPS, 1 - MT.PROB_EPS)
        logits.append(np.log(p / (1 - p)))
    assert np.isclose(spec["mean"][0], np.concatenate(logits).mean(), atol=1e-4)
    MT.check_split(spec, ds.items, tr, 0.2, 0, "song")
    with pytest.raises(SystemExit, match="another split"):
        MT.check_split(spec, ds.items, tr, 0.2, 1, "song")
    with pytest.raises(SystemExit, match="another split"):
        MT.check_split(spec, ds.items, tr[:-1], 0.2, 0, "song")


def test_companion_write_is_atomic(tmp_path):
    try:
        from scripts.sa3_encode_from_manifest import _savez_atomic
    except Exception as e:  # the encoder imports mir's resampler at import time; absent off the desktop
        pytest.skip(f"sa3_encode_from_manifest not importable here: {e}")
    p = tmp_path / "000000.TIMESERIES.npz"
    _savez_atomic(p, {"a": np.arange(3)})
    with np.load(p) as z:
        assert list(z["a"]) == [0, 1, 2]
    assert [f.name for f in tmp_path.iterdir()] == ["000000.TIMESERIES.npz"]

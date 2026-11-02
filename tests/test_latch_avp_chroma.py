# tests/test_latch_avp_chroma.py -- KUANG 2026-10-30
# Song-level val grouping, multi-root chroma pairing, and the AVP stem-chroma extractor.
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.latch.latch_dataset import LatCHDataset
import scripts.latch.extract_avp_same_chroma as X

T = 8
SAME_CHROMA_SRC = os.environ.get("SAME_CHROMA_SRC", r"C:\dev\private\mir-feature-extraction\src")


def _crop(root, chroot, stem, meta, fill, key="other"):
    np.save(root / f"{stem}.npy", np.zeros((256, T), np.float32))
    (root / f"{stem}.json").write_text(json.dumps(meta))
    if chroot is not None:
        np.savez(chroot / f"{stem}.npz", **{key: np.full((3, 128, T), fill, np.float16)})


@pytest.fixture
def two_corpora(tmp_path):
    goa, avp, gch, ach = (tmp_path / n for n in ("goa", "avp", "goa_ch", "avp_ch"))
    for d in (goa, avp, gch, ach):
        d.mkdir()
    # the same crop names in both corpora: the collision the multi-root pairing has to survive
    _crop(goa, gch, "000000", {"source_track": "A - One"}, 1.0)
    _crop(goa, gch, "000001", {"source_track": "A - One"}, 1.0)
    _crop(avp, ach, "000000", {"source_track": "Song"}, 2.0)
    _crop(avp, ach, "000001", {"source_track": "Song/pitch+1", "parent_track": "Song", "is_augmentation": True}, 2.0)
    _crop(avp, ach, "000002", {"source_track": "Song/tempo-5", "parent_track": "Song", "is_augmentation": True}, 2.0)
    return goa, avp, gch, ach


def test_song_grouping_collapses_variants(two_corpora):
    goa, avp, gch, ach = two_corpora
    ds = LatCHDataset([str(goa), str(avp)], target_feature="same_chroma", target_source="chroma",
                      chroma_dir=f"{gch},{ach}")
    assert ds.group_keys("song") == ["A - One", "A - One", "Song", "Song", "Song"]
    assert ds.group_keys("source_track")[3:] == ["Song/pitch+1", "Song/tempo-5"]


def test_multiroot_pairs_each_root_with_its_own_chroma_dir(two_corpora):
    goa, avp, gch, ach = two_corpora
    ds = LatCHDataset([str(goa), str(avp)], target_feature="same_chroma", target_source="chroma",
                      chroma_dir=f"{gch},{ach}")
    means = [(ds.items[i].parent.name, float(ds[i][1].mean())) for i in range(len(ds))]
    assert means == [("goa", 1.0), ("goa", 1.0), ("avp", 2.0), ("avp", 2.0), ("avp", 2.0)]


def test_one_chroma_dir_for_two_latent_dirs_is_refused(two_corpora):
    goa, avp, gch, _ = two_corpora
    with pytest.raises(ValueError, match="one chroma dir per latent dir"):
        LatCHDataset([str(goa), str(avp)], target_feature="same_chroma", target_source="chroma",
                     chroma_dir=str(gch))


def test_missing_chroma_key_is_refused(tmp_path):
    lat, ch = tmp_path / "lat", tmp_path / "ch"
    lat.mkdir()
    ch.mkdir()
    _crop(lat, ch, "000000", {"source_track": "x"}, 1.0, key="full_mix")
    with pytest.raises(ValueError, match="'other' not in"):
        LatCHDataset(str(lat), target_feature="same_chroma", target_source="chroma",
                     chroma_dir=str(ch), chroma_key="other")


# ---- extractor -------------------------------------------------------------------------

def test_recipe_marker_refuses_foreign_files(tmp_path):
    np.savez(tmp_path / "000000.npz", other=np.zeros(1))
    with pytest.raises(SystemExit, match="no _recipe.json"):
        X.check_recipe(tmp_path, ["other", "bass"])
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    X.check_recipe(fresh, ["other", "bass"])
    X.check_recipe(fresh, ["bass", "other"])          # same recipe, resume is fine
    with pytest.raises(SystemExit, match="another recipe"):
        X.check_recipe(fresh, ["other"])


@pytest.mark.skipif(not Path(SAME_CHROMA_SRC, "harmonic", "same_chroma.py").exists(),
                    reason="mir same_chroma not available")
def test_extractor_reads_the_stem_window(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    crop = 4096 * 32                                   # ~3 s instead of the real 380 s crop
    monkeypatch.setattr(X, "CROP_SAMPLES", crop)
    sr = 44100
    song = tmp_path / "Song"
    var = song / "augmentations" / "pitch+1"
    var.mkdir(parents=True)
    t = np.arange(sr * 8) / sr
    tone = lambda f: np.stack([0.3 * np.sin(2 * np.pi * f * t)] * 2, axis=1).astype(np.float32)
    sf.write(song / "full_mix.flac", tone(261.63), sr)  # C in the mix ...
    sf.write(song / "other.flac", tone(440.0), sr)      # ... A in the other stem
    sf.write(song / "bass.flac", tone(110.0), sr)
    sf.write(var / "full_mix.flac", tone(277.18), sr)   # variant without stems
    lat = tmp_path / "lat"
    lat.mkdir()
    for stem, src in (("000000", song / "full_mix.flac"), ("000001", var / "full_mix.flac")):
        np.save(lat / f"{stem}.npy", np.zeros((256, 32), np.float32))
        (lat / f"{stem}.json").write_text(json.dumps({"source_path": str(src), "timestamps": [2.0, 5.0]}))
    out = tmp_path / "out"
    out.mkdir()
    X.init_worker(SAME_CHROMA_SRC, ["other", "bass"], str(out))
    assert X.process_item(str(lat / "000000.npy")) == ("new", "")
    status, msg = X.process_item(str(lat / "000001.npy"))
    assert status == "missing" and "other,bass" in msg
    assert X.process_item(str(lat / "000000.npy"))[0] == "exists"
    with np.load(out / "000000.npz") as z:
        assert sorted(z.files) == ["bass", "other"]
        other = z["other"].astype(np.float32)
    assert other.shape == (3, 128, crop // 4096)
    pc = X._sc.fold_to_12(other)[1].mean(axis=1).argmax()   # mid band, C = 0
    assert pc == 9                                          # A, from the stem, not the C in the mix
    assert not list(out.glob("*.partial.npz"))

"""Tests for scripts/caption_tools.py — T1 template generation + tiered
caption sampling for the prompting-conditioning plan."""
import sys, os, json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from caption_tools import era_bucket, build_t1, make_caption_sampler, generate_sidecar


# ---------------------------------------------------------------- era

@pytest.mark.parametrize("year,era", [
    (1991, "early 90s"), (1994, "early 90s"),
    (1995, "mid 90s"), (1997, "mid 90s"),
    (1998, "late 90s"), (2000, "late 90s"),
    (2001, "early 2000s"), (2004, "early 2000s"),
    (2005, "mid 2000s"), (2007, "mid 2000s"),
    (2008, "late 2000s"), (2010, "late 2000s"),
    (2011, "2010s"), (2019, "2010s"),
    (2020, "2020s"), (2026, "2020s"),
])
def test_era_bucket(year, era):
    assert era_bucket(year) == era


def test_era_bucket_unknown():
    assert era_bucket(None) is None
    assert era_bucket(0) is None


# ---------------------------------------------------------------- T1

ROW = {
    "genres": ["Goa Trance", "Psy-Trance"],
    "moods": ["space", "energetic"],
    "bpm": 147.6,
    "year": 1996,
}


def test_build_t1_era_fronted_full():
    t1 = build_t1(**ROW)
    assert t1 == "mid 90s goa trance, psy-trance, space energetic mood, 148 bpm"


def test_build_t1_missing_year_and_moods():
    t1 = build_t1(genres=["Goa Trance"], moods=[], bpm=145.0, year=None)
    assert t1 == "goa trance, 145 bpm"


def test_build_t1_missing_bpm():
    t1 = build_t1(genres=["Ambient"], moods=["soundscape"], bpm=None, year=2003)
    assert t1 == "early 2000s ambient, soundscape mood"


# ---------------------------------------------------------------- sampler

def _sidecar(tmp_path, entries):
    p = tmp_path / "captions.json"
    p.write_text(json.dumps(entries))
    return str(p)


def test_sampler_returns_t1_when_no_t2_t3(tmp_path):
    sc = _sidecar(tmp_path, {"000123": {"t1": "goa trance", "t2": None, "t3": None}})
    fn = make_caption_sampler(sc, probs=(0.0, 0.7, 0.3))  # t1 prob 0 — must fall back
    out = fn({"latent_filename": "/x/y/000123.npy"}, None)
    assert out == {"prompt": "goa trance"}


def test_sampler_distribution_and_determinism(tmp_path):
    sc = _sidecar(tmp_path, {"000001": {"t1": "T1", "t2": "T2", "t3": "T3"}})
    fn = make_caption_sampler(sc, probs=(0.6, 0.3, 0.1), seed=7)
    got = [fn({"latent_filename": "000001.npy"}, None)["prompt"] for _ in range(400)]
    frac = {t: got.count(t) / len(got) for t in ("T1", "T2", "T3")}
    assert 0.5 < frac["T1"] < 0.7
    assert 0.2 < frac["T2"] < 0.4
    assert 0.04 < frac["T3"] < 0.17
    fn2 = make_caption_sampler(sc, probs=(0.6, 0.3, 0.1), seed=7)
    got2 = [fn2({"latent_filename": "000001.npy"}, None)["prompt"] for _ in range(400)]
    assert got == got2


def test_sampler_unknown_index_keeps_existing_prompt(tmp_path):
    sc = _sidecar(tmp_path, {"000001": {"t1": "T1"}})
    fn = make_caption_sampler(sc, probs=(1.0, 0.0, 0.0))
    out = fn({"latent_filename": "999999.npy", "prompt": "original"}, None)
    assert out == {}  # no override — dataset keeps its stored prompt


def test_sampler_survives_dill_roundtrip(tmp_path):
    import dill
    sc = _sidecar(tmp_path, {"000005": {"t1": "only"}})
    fn = dill.loads(dill.dumps(make_caption_sampler(sc, probs=(1.0, 0, 0))))
    assert fn({"latent_filename": "000005.npy"}, None) == {"prompt": "only"}


# ---------------------------------------------------------------- sidecar gen

def test_generate_sidecar(tmp_path):
    rows = [
        {"index_keys": ["000001", "000002"], "genres": ["Goa Trance"],
         "moods": ["dark"], "bpm": 150.0, "year": 1999},
    ]
    out = str(tmp_path / "cap.json")
    n = generate_sidecar(rows, out)
    assert n == 2
    d = json.loads(open(out).read())
    assert d["000001"]["t1"] == "late 90s goa trance, dark mood, 150 bpm"
    assert d["000002"]["t1"] == d["000001"]["t1"]
    assert d["000001"]["t2"] is None and d["000001"]["t3"] is None

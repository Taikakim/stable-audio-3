# Owner: KUANG (created 2026-10-23)
"""CPU tests for scripts/drum_sep/codec_ceiling.py (gate G0 of the SA3 drum separator plan), with fake codecs:
identity (everything kept), a 30 ms delay (onsets move), a smearing codec (attacks lost on the hats)."""

import json

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, lfilter

from scripts.drum_sep import codec_ceiling as cc

SR = cc.SR


def _kick(seconds=3.0):
    y = np.zeros(int(seconds * SR))
    t = np.arange(int(0.25 * SR)) / SR
    hit = np.sin(2 * np.pi * (55 + 120 * np.exp(-t / 0.02)) * t) * np.exp(-t / 0.12)
    for s in np.arange(0.1, seconds - 0.3, 0.5):
        a = int(s * SR)
        y[a:a + len(hit)] += hit
    return y


def _hats(seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    b, a = butter(4, 6000 / (SR / 2), btype="high")
    y = np.zeros(int(seconds * SR))
    n = int(0.03 * SR)
    for s in np.arange(0.17, seconds - 0.1, 0.25):
        i = int(s * SR)
        y[i:i + n] += lfilter(b, a, rng.normal(0, 0.4, n)) * np.exp(-np.arange(n) / 120.0)
    return y


def _stereo(y):
    return np.stack([y, y]).astype(np.float32)


class Identity:
    sample_rate = SR

    def encode(self, audio, sr):
        return np.asarray(audio, np.float32).copy()

    def decode(self, lat, n):
        return np.asarray(lat)[:, :n]


class Delay(Identity):
    def decode(self, lat, n):
        d = int(0.030 * SR)
        y = np.asarray(lat)
        return np.pad(y, ((0, 0), (d, 0)))[:, :n]


class Smear(Identity):
    """Lowpass at 2 kHz plus a 30 ms moving average: transients and hats are lost."""

    def decode(self, lat, n):
        b, a = butter(2, 2000 / (SR / 2))
        y = lfilter(b, a, np.asarray(lat), axis=1)
        k = int(0.03 * SR)
        y = np.stack([np.convolve(c, np.ones(k) / k, mode="same") for c in y])
        return y[:, :n].astype(np.float32)


def test_identity_keeps_everything():
    stems = {"kick": _stereo(_kick()), "hats": _stereo(_hats())}
    res, add = cc.run_song(Identity(), stems)
    for k in ("kick", "hats", "__bus__"):
        assert res[k]["lsd_db"] < 1e-3 and res[k]["onset_f1_20"] == pytest.approx(1.0)
        assert abs(res[k]["attack_db"]) < 1e-6 and res[k]["n_onsets"] > 3
    assert add["sum_vs_bus_lsd_db"] < 1e-3 and add["latent_sum_lsd_db"] < 1e-3
    assert cc.gate_verdict(cc.summarize([{"tracks": res}]))["transient_ok"]


def test_delay_moves_onsets_and_fails_the_20ms_window():
    res, _ = cc.run_song(Delay(), {"kick": _stereo(_kick())})
    m = res["kick"]
    assert m["onset_f1_20"] < 0.5 and m["onset_f1_50"] > 0.8 and 20.0 < m["onset_dt_ms"] < 40.0
    v = cc.gate_verdict(cc.summarize([{"tracks": res}]))
    assert not v["transient_ok"] and v["classes_failing"] == ["kick"]


def test_smearing_codec_loses_the_hats_attack():
    res, _ = cc.run_song(Smear(), {"kick": _stereo(_kick()), "hats": _stereo(_hats())})
    assert res["hats"]["attack_db"] < -6.0 and res["hats"]["lsd_db"] > res["kick"]["lsd_db"]
    assert "hats" in cc.gate_verdict(cc.summarize([{"tracks": res}]))["classes_failing"]


def test_onset_matching_is_one_to_one():
    p, r, f1, dts = cc.match_onsets(np.array([0.10, 0.11]), np.array([0.105]), tol=0.02)
    assert (p, r) == (1.0, 0.5) and len(dts) == 1
    assert cc.match_onsets(np.zeros(0), np.zeros(0), 0.02)[2] == 1.0              # nothing to find, nothing found
    assert cc.match_onsets(np.array([0.1]), np.zeros(0), 0.02)[2] == 0.0


def test_main_writes_results_and_report(tmp_path):
    root = tmp_path / "stems"
    for song in ("songA", "songB"):
        d = root / song
        d.mkdir(parents=True)
        sf.write(d / "kick.wav", _kick(2.0), SR)                                # mono file: duplicated to stereo
        sf.write(d / "hats.wav", _stereo(_hats(2.0, seed=len(song))).T, SR)
    out = tmp_path / "out"
    summary, verdict = cc.main(["--stems_dir", str(root), "--out_dir", str(out), "--crop_s", "2"], codec=Identity())
    assert set(summary) == {"kick", "hats", "__bus__"} and summary["kick"]["n"] == 2 and verdict["transient_ok"]
    data = json.loads((out / "results.json").read_text())
    assert len(data["songs"]) == 2 and "additivity" in data["songs"][0]
    assert (out / "report.md").exists() and (out / "listen" / "songA__kick__recon.wav").exists()

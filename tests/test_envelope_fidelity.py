"""Tests for eval/envelope_fidelity.py — the pad-fill / envelope-timing meter."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval"))
from envelope_fidelity import measure

SR = 44100


def _beat_track(dur=8.0, bpm=140, f=110.0):
    """Kick-ish pulse train + quiet second half."""
    n = int(dur * SR)
    y = np.zeros(n, dtype=np.float32)
    period = int(60 / bpm * SR)
    for i in range(0, n // 2, period):          # beats only in the first half
        t = np.arange(min(4000, n - i))
        y[i:i + len(t)] += np.sin(2 * np.pi * f * t / SR) * np.exp(-t / 800.0)
    return y


def test_identical_is_perfect():
    y = _beat_track()
    r = measure(y, y, SR)
    assert r["onset_corr"] > 0.99
    assert r["band_corr_mean"] > 0.99
    assert r["pad_fill_frac"] == 0.0


def test_pad_fill_detected_in_quiet_region():
    y = _beat_track()
    o = y.copy()
    half = len(y) // 2                           # source is quiet here
    t = np.arange(len(y) - half)
    o[half:] += 0.2 * np.sin(2 * np.pi * 220 * t / SR).astype(np.float32)  # drone
    r = measure(y, o, SR)
    assert r["pad_fill_frac"] > 0.15, r
    assert r["pad_fill_db"] > 6.0, r


def test_shifted_onsets_drop_timing_corr():
    y = _beat_track()
    shift = int(0.18 * SR)                       # ~half a beat at 140
    o = np.roll(y, shift)
    r_shift = measure(y, o, SR)
    r_same = measure(y, y, SR)
    assert r_shift["onset_corr"] < r_same["onset_corr"] - 0.25, r_shift


def test_region_restriction():
    y = _beat_track()
    o = y.copy()
    o[:len(y)//2] = _beat_track(bpm=97)[:len(y)//2]   # mangle first half only
    r_bad = measure(y, o, SR, region=(0.0, 4.0))
    r_good = measure(y, o, SR, region=(4.0, 8.0))
    assert r_good["band_corr_mean"] > r_bad["band_corr_mean"], (r_good, r_bad)


def test_v2_floor_delta_catches_layered_drone():
    """Pads BEHIND active content (Kim's actual case): busy source, output =
    source + quiet sustained drone. v1 pad_fill misses it (no quiet zones);
    the sustained-floor delta must catch it."""
    y = np.concatenate([_beat_track(4.0), _beat_track(4.0)])   # busy throughout
    t = np.arange(len(y))
    o = y + 0.12 * np.sin(2 * np.pi * 2000 * t / SR).astype(np.float32)  # mid-band drone
    r = measure(y, o, SR)
    assert r["pad_floor_db"] > 4.0, r
    r_id = measure(y, y, SR)
    assert abs(r_id["pad_floor_db"]) < 1.0, r_id

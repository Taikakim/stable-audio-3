#!/usr/bin/env python3
"""TDD for wiring W's per-field resampler into sa3_encode_from_manifest.build_crop_timeseries.

The OLD loop applies one rate (100 Hz) + one pooling rule (mean) to EVERY field. This test builds
a synthetic whole-track npz with the real spread the goa store now has — a 100 Hz continuous field,
a 0.2 Hz coarse field, a 100 Hz f0 SENTINEL field (0.0 = unvoiced) + its voiced mask, and a 100 Hz
CATEGORICAL chord field — then asserts a crop companion is built correctly. It FAILS on the old code:
  - f0 mean-pooled with the unvoiced 0.0s -> pulled toward silence (W bug #2, p95 +15.86 st)
  - chords_idx mean-pooled -> fractional class indices (W bug #3)
  - the 0.2 Hz field sliced at 100 Hz -> wrong region / dropped crop (W bug #1)

Run: /home/kim/Projects/mir/mir/bin/python stable-audio-3/scripts/test_encode_crop_wiring.py
(mir venv has numpy; the encoder module imports torch lazily only inside main(), not at import.)
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sa3enc", HERE / "sa3_encode_from_manifest.py")
enc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc)
LATENT_FRAMES = enc.LATENT_FRAMES


def _make_store(dirpath: Path, track_name: str, dur_s: float = 200.0):
    """Whole-track npz with mixed native rates, keyed for the colocated fallback."""
    tf = dirpath / track_name
    tf.mkdir(parents=True, exist_ok=True)
    # 100 Hz continuous field: a ramp 0..1 over the track (region-check target)
    n100 = int(dur_s * 100)
    ramp = np.linspace(0.0, 1.0, n100).astype(np.float32)
    # 0.2 Hz coarse field (like dyncomplexity_ts): 40 samples over 200 s
    n02 = int(dur_s * 0.2)
    coarse = np.linspace(10.0, 50.0, n02).astype(np.float32)
    # f0 SENTINEL @100 Hz: constant 60.0 MIDI where voiced, 0.0 unvoiced. Make the FIRST
    # half of our crop window unvoiced so mean-pooling would drag it toward 0.
    f0 = np.full(n100, 60.0, dtype=np.float32)
    voiced = np.ones(n100, dtype=np.float32)
    # crop window will be [100,110]s -> frames [10000:11000]; unvoice [10000:10500]
    f0[10000:10500] = 0.0
    voiced[10000:10500] = 0.0
    # CATEGORICAL chords @100 Hz: class 3 then class 10 within the crop window
    chords = np.full(n100, 3.0, dtype=np.float32)
    chords[10500:11000] = 10.0
    meta = {
        "frame_rate": 100, "n_frames": n100, "duration": dur_s, "sample_rate": 44100,
        "field_rates": {
            "ramp_ts": 100.0, "coarse_ts": 0.2, "f0_other_ts": 100.0,
            "f0_other_voiced_ts": 100.0, "chords_idx_ts": 100.0,
        },
        "fields": ["ramp_ts", "coarse_ts", "f0_other_ts", "f0_other_voiced_ts", "chords_idx_ts"],
    }
    np.savez(tf / f"{track_name}.TIMESERIES.npz",
             ramp_ts=ramp, coarse_ts=coarse, f0_other_ts=f0,
             f0_other_voiced_ts=voiced, chords_idx_ts=chords,
             __meta__=np.array(json.dumps(meta)))
    return tf


def main():
    fails = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        track = "Test Artist - Test Track"
        tf = _make_store(td, track)
        start, end = 100.0, 110.0            # 10 s crop, native frames [10000:11000]
        out = enc.build_crop_timeseries(None, tf, track, start, end)

        if out is None:
            print("FAIL: build_crop_timeseries returned None (coarse-field slice at wrong rate "
                  "dropped the whole crop — W bug #1)")
            sys.exit(1)

        # 1. every field present at LATENT_FRAMES
        for k in ["ramp_ts", "coarse_ts", "f0_other_ts", "chords_idx_ts"]:
            if k not in out:
                fails.append(f"field {k} missing from output")
            elif out[k].shape[0] != LATENT_FRAMES:
                fails.append(f"field {k} shape {out[k].shape} != ({LATENT_FRAMES},)")

        # 2. f0 SENTINEL: window is 50% unvoiced(0.0) + 50% voiced(60.0). Masked-mean of the
        #    VOICED frames is ~60.0. Naive mean-pool would give ~30.0. Assert we are near 60.
        if "f0_other_ts" in out:
            f0m = float(np.median(out["f0_other_ts"]))
            if not (55.0 <= f0m <= 60.5):
                fails.append(f"f0 median {f0m:.1f} — expected ~60 (masked-mean); "
                             f"~30 means the unvoiced 0.0s were smeared in (W bug #2)")
            # the voiced-weight companion must be emitted
            if "f0_other_voiced_ts" not in out:
                fails.append("f0_other_voiced_ts (pooled validity/loss-weight) not emitted")

        # 3. CATEGORICAL chords: must be integer class indices (3 or 10), never a fractional avg
        if "chords_idx_ts" in out:
            vals = np.unique(out["chords_idx_ts"])
            nonint = vals[(vals != np.round(vals))]
            if nonint.size > 0:
                fails.append(f"chords_idx has fractional classes {nonint[:4]} — averaged, "
                             f"not mode-pooled (W bug #3)")
            if not set(np.round(vals).astype(int)).issubset({3, 10}):
                fails.append(f"chords_idx classes {vals} outside the true {{3,10}}")

        # 4. coarse 0.2 Hz field sliced at ITS rate covers native seconds [100,110] -> value
        #    ~linspace(10,50) at 100/200 of the way ~= 30. At the WRONG 100 Hz slice it'd be
        #    read from frames [10000:11000] of a 40-length array -> out of range / garbage.
        if "coarse_ts" in out:
            cm = float(np.mean(out["coarse_ts"]))
            if not (26.0 <= cm <= 34.0):
                fails.append(f"coarse field mean {cm:.1f} — expected ~30 (0.2 Hz slice of "
                             f"[100,110]s); wrong value means it was sliced at 100 Hz (W bug #1)")

    if fails:
        print("RED — wiring not yet correct:")
        for f in fails:
            print("  FAIL:", f)
        sys.exit(1)
    print("GREEN — build_crop_timeseries resamples per-field correctly:")
    print("  f0 masked-mean ~60 (not smeared), chords mode-pooled to {3,10}, "
          "coarse sliced at 0.2 Hz, all fields at LATENT_FRAMES.")


if __name__ == "__main__":
    main()

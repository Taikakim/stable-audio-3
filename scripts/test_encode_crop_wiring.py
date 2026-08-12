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


def _make_wrongrate_store(dirpath: Path, track_name: str, dur_s: float = 1000.0):
    """A store whose maest-shaped coarse field has a 2x-WRONG stated rate (the real bug W caught
    2026-08-12, mir f10d007). True patch hop ~10 s -> ~0.1 Hz, but the producer hardcodes 0.2 Hz,
    so trusting the sidecar maps the FIRST HALF of the track onto the whole and every LATE crop
    slices out of range. W's fix derives the rate from n_frames/duration and overrides. This case
    only fires on a crop PAST the halfway mark -- the region a small early-crop sample never sees."""
    tf = dirpath / track_name
    tf.mkdir(parents=True, exist_ok=True)
    n100 = int(dur_s * 100)
    f0 = np.full(n100, 60.0, dtype=np.float32)
    voiced = np.ones(n100, dtype=np.float32)
    # maest-shaped: TRUE 0.1 Hz -> 100 samples over 1000 s, value = the 0..100 position ramp so a
    # crop's mean reveals WHICH region was read. Stated rate is 0.2 (2x wrong).
    n_maest = int(dur_s * 0.1)
    maest = np.linspace(0.0, 100.0, n_maest).astype(np.float32)
    meta = {
        "frame_rate": 100, "n_frames": n100, "duration": dur_s, "sample_rate": 44100,
        "field_rates": {"f0_other_ts": 100.0, "f0_other_voiced_ts": 100.0,
                        "maest_embed_ts": 0.2},          # <-- the wrong number
        "fields": ["f0_other_ts", "f0_other_voiced_ts", "maest_embed_ts"],
    }
    np.savez(tf / f"{track_name}.TIMESERIES.npz",
             f0_other_ts=f0, f0_other_voiced_ts=voiced, maest_embed_ts=maest,
             __meta__=np.array(json.dumps(meta)))
    return tf


def test_late_window_wrong_rate(fails):
    """Late crop [700,900]s of a 1000 s track. maest's TRUE 0.1 Hz puts that window at frames
    [70:90] (value ~80 on the 0..100 ramp). Trusting the stated 0.2 Hz would ask frames [140:180]
    of a 100-length array -> empty -> the whole crop dropped/raised. Passing REQUIRES W's f10d007
    rate-derivation."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        track = "Long Artist - Long Track"
        tf = _make_wrongrate_store(td, track)
        out = enc.build_crop_timeseries(None, tf, track, 700.0, 900.0)
        if out is None:
            fails.append("late-window crop DROPPED — maest sliced at the stated 2x-wrong 0.2 Hz "
                         "went out of range (W's f10d007 rate-derivation not active)")
            return
        if "maest_embed_ts" not in out:
            fails.append("maest_embed_ts missing from late-window crop")
        elif out["maest_embed_ts"].shape[0] != LATENT_FRAMES:
            fails.append(f"maest_embed_ts shape {out['maest_embed_ts'].shape} != ({LATENT_FRAMES},)")
        else:
            mm = float(np.mean(out["maest_embed_ts"]))
            if not (74.0 <= mm <= 86.0):
                fails.append(f"maest late-window mean {mm:.1f} — expected ~80 (frames [70:90] at "
                             f"the DERIVED 0.1 Hz); a different value means it read the wrong region")


def _make_f0_store(dirpath: Path, track_name: str, dur_s: float = 200.0):
    """Store with legacy + f0 + a maest-bloat field, for the additive-whitelist test."""
    tf = dirpath / track_name
    tf.mkdir(parents=True, exist_ok=True)
    n100 = int(dur_s * 100)
    spectral = np.linspace(0.0, 1.0, n100).astype(np.float32)          # a legacy 100 Hz field
    f0o = np.full(n100, 220.0, dtype=np.float32); vo = np.ones(n100, np.float32)
    f0b = np.full(n100, 55.0, dtype=np.float32);  vb = np.ones(n100, np.float32)
    f0o[:n100 // 2] = 0.0; vo[:n100 // 2] = 0.0                        # half unvoiced (masking check)
    maest = np.linspace(0, 1, int(dur_s * 0.1)).astype(np.float32)[:, None].repeat(768, 1)  # bloat
    meta = {"frame_rate": 100, "n_frames": n100, "duration": dur_s,
            "field_rates": {"spectral_flux_ts": 100.0, "f0_other_ts": 100.0, "f0_other_voiced_ts": 100.0,
                            "f0_bass_ts": 100.0, "f0_bass_voiced_ts": 100.0, "maest_embed_ts": 0.1}}
    np.savez(tf / f"{track_name}.TIMESERIES.npz", spectral_flux_ts=spectral,
             f0_other_ts=f0o, f0_other_voiced_ts=vo, f0_bass_ts=f0b, f0_bass_voiced_ts=vb,
             maest_embed_ts=maest, __meta__=np.array(json.dumps(meta)))
    return tf


def test_additive_f0_whitelist(fails):
    """The production path Kim authorized 2026-08-13: companion-only --fields <4 f0> --additive.
    A crop whose EXISTING companion carries only legacy fields must gain EXACTLY the 4 f0 fields,
    keep every legacy field (incl relative_position_ts), and NOT gain maest. Latent untouched."""
    F0 = {"f0_other_ts", "f0_other_voiced_ts", "f0_bass_ts", "f0_bass_voiced_ts"}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        track = "Add Artist - Add Track"
        tf = _make_f0_store(td, track)
        out = td / "corpus"; out.mkdir()
        idx = "000000"
        # existing companion = legacy fields only (the 21-field stand-in) + relative_position_ts
        legacy = {"spectral_flux_ts": np.linspace(0, 1, LATENT_FRAMES).astype(np.float32),
                  "relative_position_ts": np.linspace(0.0, 1.0, LATENT_FRAMES).astype(np.float32)}
        np.savez(out / f"{idx}.TIMESERIES.npz", **legacy)
        (out / f"{idx}.npy").write_bytes(b"LATENT")                    # sentinel: must stay byte-identical
        (out / f"{idx}.json").write_text(json.dumps({
            "source_track": track, "timestamps": [0.0, 100.0],
            "source_path": str(tf / "full_mix.flac"),
            "relative_position_start": 0.0, "relative_position_end": 0.5}))

        enc.rebuild_companions(out, None, fields=F0, additive=True)

        z = dict(np.load(out / f"{idx}.TIMESERIES.npz"))
        keys = set(z.keys())
        missing = F0 - keys
        if missing:
            fails.append(f"additive: f0 fields not added: {missing}")
        for legk in ("spectral_flux_ts", "relative_position_ts"):
            if legk not in keys:
                fails.append(f"additive: legacy field {legk} DROPPED (merge should preserve it)")
        if "maest_embed_ts" in keys:
            fails.append("additive: maest_embed_ts was added — whitelist did not exclude the bloat")
        if "f0_other_ts" in z:                                         # masking still correct
            vv = z["f0_other_ts"][z["f0_other_ts"] > 0]
            if vv.size and not (215.0 <= float(np.median(vv)) <= 225.0):
                fails.append(f"additive: f0_other median {np.median(vv):.0f} != ~220 (masking wrong)")
        if (out / f"{idx}.npy").read_bytes() != b"LATENT":
            fails.append("additive: latent .npy was modified — must be read-only")


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

    # Scenario 2: the maest 2x-wrong-rate late-window failure (W f10d007). A small early-crop
    # sample cannot reach this region — F's blind-spot lesson, now permanent in the suite.
    test_late_window_wrong_rate(fails)

    # Scenario 3: the production f0-only additive whitelist path (Kim GO 2026-08-13).
    test_additive_f0_whitelist(fails)

    if fails:
        print("RED — wiring not yet correct:")
        for f in fails:
            print("  FAIL:", f)
        sys.exit(1)
    print("GREEN — build_crop_timeseries resamples per-field correctly:")
    print("  [mixed-rate] f0 masked-mean ~60 (not smeared), chords mode-pooled to {3,10}, "
          "coarse sliced at 0.2 Hz, all fields at LATENT_FRAMES.")
    print("  [late-window] maest 2x-wrong stated rate overridden by n/duration derivation — "
          "crop [700,900]s of a 1000s track read the correct late region (~80), not dropped.")
    print("  [additive-f0] --fields whitelist + --additive: 4 f0 fields merged in, legacy fields + "
          "latent preserved, maest excluded.")


if __name__ == "__main__":
    main()

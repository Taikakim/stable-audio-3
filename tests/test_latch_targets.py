# tests/test_latch_targets.py
import numpy as np
import pytest

from scripts.latch.latch_targets import resample_target, build_target


def test_resample_1d_downsamples_to_target_frames():
    src = np.linspace(0.0, 1.0, 256, dtype=np.float32)  # (256,)
    out = resample_target(src, 128)
    assert out.shape == (1, 128)
    # Endpoints preserved by linear interpolation.
    assert out[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert out[0, -1] == pytest.approx(1.0, abs=1e-5)
    # Monotonic ramp stays monotonic.
    assert np.all(np.diff(out[0]) > 0)


def test_resample_multichannel_preserves_channels():
    src = np.stack([np.zeros(256), np.ones(256)]).astype(np.float32)  # (2, 256)
    out = resample_target(src, 100)
    assert out.shape == (2, 100)
    assert out[0].mean() == pytest.approx(0.0, abs=1e-5)
    assert out[1].mean() == pytest.approx(1.0, abs=1e-5)


def test_resample_channel_last_is_transposed():
    # hpcp natural storage is (T, C) with T > C; smaller dim is channels.
    src = np.zeros((256, 12), dtype=np.float32)
    out = resample_target(src, 64)
    assert out.shape == (12, 64)


def test_build_constant_target():
    out = build_target("constant", value=-30.0, n_frames=50, n_channels=1)
    assert out.shape == (1, 50)
    assert np.all(out == -30.0)


def test_build_ramp_up_target():
    out = build_target("ramp_up", value=-10.0, n_frames=10, n_channels=1)
    assert out.shape == (1, 10)
    assert out[0, 0] < out[0, -1]
    assert out[0, -1] == pytest.approx(-10.0, abs=1e-5)

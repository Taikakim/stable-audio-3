# scripts/latch/latch_targets.py
"""Resample MIR time-series targets to the SA3 latent frame grid, and build synthetic targets."""

import numpy as np


def _to_channel_first(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr[np.newaxis, :]  # (1, T)
    if arr.ndim == 2:
        # Treat the smaller dimension as channels (hpcp is (T, 12)).
        return arr.T if arr.shape[0] > arr.shape[1] else arr
    raise ValueError(f"Unexpected target ndim={arr.ndim}")


def resample_target(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Resample a target array to (C, n_frames) via per-channel linear interpolation."""
    cf = _to_channel_first(arr)  # (C, T_src)
    c, t_src = cf.shape
    if t_src == n_frames:
        return cf.astype(np.float32)
    src_x = np.linspace(0.0, 1.0, t_src, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    out = np.empty((c, n_frames), dtype=np.float32)
    for ch in range(c):
        out[ch] = np.interp(dst_x, src_x, cf[ch]).astype(np.float32)
    return out


def build_target(kind: str, value: float, n_frames: int, n_channels: int = 1) -> np.ndarray:
    """Build a synthetic control target of shape (n_channels, n_frames)."""
    if kind == "constant":
        return np.full((n_channels, n_frames), value, dtype=np.float32)
    if kind == "ramp_up":
        ramp = np.linspace(value - abs(value) - 1.0, value, n_frames, dtype=np.float32)
        return np.tile(ramp, (n_channels, 1))
    if kind == "ramp_down":
        ramp = np.linspace(value, value - abs(value) - 1.0, n_frames, dtype=np.float32)
        return np.tile(ramp, (n_channels, 1))
    raise ValueError(f"Unknown target kind: {kind!r}")

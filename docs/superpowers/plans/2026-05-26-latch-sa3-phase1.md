# LatCH on Stable Audio 3 — Phase 1 (small-base, Euler) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the LatCH (Latent-Control Head) pipeline from Stable Audio Open Small to Stable Audio 3, targeting the `small-base` flow-matching model with Euler sampling, and prove closed-loop control of one feature (`rms_energy_bass`).

**Architecture:** A lightweight BiTransformer head is trained to predict an MIR time-series from noisy SAME latents, then used as a Training-Free-Guidance signal during a gradient-enabled Euler sampler. Phase 1 deliberately targets `small-base` (50-step Euler + CFG) because it is the direct analogue of the working SAO Small setup; the post-trained ping-pong path is out of scope (Phase 2). SAME latents are 256-dim at ~10.76 Hz and variable length, so the SAO 64-dim / 256-frame assumptions are removed.

**Tech Stack:** Python 3.13, PyTorch (ROCm), `stable_audio_3` package (`AutoencoderModel`, `StableAudioModel`, `sample_discrete_euler`), the MIR `TimeseriesDB` + extractors at `/home/kim/Projects/mir/src`, pytest.

**Source of truth:** Requirements `Stable Audio 3 portability` section (`SA3-1`..`SA3-18`) in `../../../../stable-audio-tools/scripts/LATCH_TRAINER_REQUIREMENTS.md`. SAO reference implementation in `/home/kim/Projects/SAO/stable-audio-tools/scripts/{latch_model,latch_dataset,train_latch}.py`.

**Phase-1 boundaries (explicitly OUT of scope):** ping-pong guidance, post-trained models, `medium`/SAME-L, multi-head stacking, the Gradio UI. Phase 1 produces a CLI-driven, verifiable single-head result on `small-base` only.

---

## File Structure

All new code lives in the **`stable-audio-3`** repo (the port target). Imports use `stable_audio_3.*`, never `stable_audio_tools.*` (SA3-18).

- Create `scripts/latch/__init__.py` — package marker.
- Create `scripts/latch/latch_model.py` — the `LatCH` BiTransformer head. Ported verbatim from SAO except `in_channels` default 64→256; already length-agnostic via RoPE (SA3-1).
- Create `scripts/latch/latch_targets.py` — `resample_target(arr, n_frames)` and target builders (`constant`, `ramp_up`, `ramp_down`). Pure NumPy, fully unit-testable (SA3-4).
- Create `scripts/latch/encode_latch_dataset.py` — re-encode an audio corpus through SAME-S into per-clip `.npy` latents + a sidecar `.json` recording `{crop_key, latent_frames, seconds}` (SA3-3).
- Create `scripts/latch/latch_dataset.py` — `LatCHDataset` reading SA3 `.npy` latents (256×T) and resampling the MIR target to each clip's T (SA3-2, SA3-4, SA3-5).
- Create `scripts/latch/train_latch.py` — training loop: flow-matching forward noise, masked loss, v2 checkpoint (SA3-6, SA3-7, SA3-8).
- Create `stable_audio_3/inference/latch_guided.py` — `sample_flow_euler_latch_guided`, a grad-enabled mirror of `sample_discrete_euler` (SA3-9..SA3-14, SA3-17).
- Create `scripts/latch/verify_latch.py` — closed-loop control check via mir's real extractor.
- Create `tests/test_latch_targets.py`, `tests/test_latch_model.py`, `tests/test_latch_guided.py` — CPU-only unit tests (no GPU, no drives, no downloads).

Hardware/data-bound steps (encode, train, verify) are **integration runbooks**: they need the ROCm GPU, the mounted source drive, the mir DB, and an HF download of `small-base`/`same-s`. They are marked **[INTEGRATION]** and specify the exact command plus what success looks like, rather than a `pytest` PASS.

---

## Task 1: Scaffold the `scripts/latch` package

**Files:**
- Create: `scripts/latch/__init__.py`

- [ ] **Step 1: Create the package marker**

```python
# scripts/latch/__init__.py
"""LatCH (Latent-Control Head) pipeline for Stable Audio 3 — Phase 1 (small-base)."""
```

- [ ] **Step 2: Commit**

```bash
git add scripts/latch/__init__.py
git commit -m "feat(latch): scaffold scripts/latch package"
```

---

## Task 2: Target resampling (pure logic, TDD)

The MIR `TimeseriesDB` stores every feature at 256 frames (~21.53 Hz, the SAO grid). SA3 latents are at ~10.76 Hz and variable length, so each target must be resampled to the *actual* latent frame count `T` of its clip (SA3-4). `hpcp` is multi-channel; resampling is per-channel along time.

**Files:**
- Create: `scripts/latch/latch_targets.py`
- Test: `tests/test_latch_targets.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_latch_targets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.latch.latch_targets'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_latch_targets.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/latch/latch_targets.py tests/test_latch_targets.py
git commit -m "feat(latch): target resampling to SA3 frame grid + synthetic builders"
```

---

## Task 3: Port the LatCH head (256-dim input, variable length)

The head is copied from SAO with one change: `in_channels` default 64→256 (SA3-1). RoPE already makes it length-agnostic (SA3-5), which we assert with a test at two sequence lengths.

**Files:**
- Create: `scripts/latch/latch_model.py`
- Test: `tests/test_latch_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_latch_model.py
import torch

from scripts.latch.latch_model import LatCH


def test_head_accepts_256_channels_and_returns_out_channels():
    head = LatCH(in_channels=256, out_channels=1, dim=128, depth=2, num_heads=4)
    x = torch.randn(2, 256, 128)         # (B, 256, T)
    t = torch.rand(2)                    # (B,)
    out = head(x, t)
    assert out.shape == (2, 1, 128)      # (B, out_channels, T)


def test_head_is_length_agnostic():
    head = LatCH(in_channels=256, out_channels=1, dim=128, depth=2, num_heads=4)
    t = torch.rand(1)
    out_short = head(torch.randn(1, 256, 64), t)
    out_long = head(torch.randn(1, 256, 300), t)
    assert out_short.shape == (1, 1, 64)
    assert out_long.shape == (1, 1, 300)


def test_head_multichannel_output():
    head = LatCH(in_channels=256, out_channels=12, dim=128, depth=2, num_heads=4)
    out = head(torch.randn(1, 256, 80), torch.rand(1))
    assert out.shape == (1, 12, 80)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_latch_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.latch.latch_model'`

- [ ] **Step 3: Copy the head and change the input default**

Copy `/home/kim/Projects/SAO/stable-audio-tools/scripts/latch_model.py` to `scripts/latch/latch_model.py` **verbatim**, then change only the `LatCH.__init__` signature default:

```python
    def __init__(
        self,
        in_channels=256,   # SA3 SAME latent dimensionality (was 64 for SAO Small)
        out_channels=1,
        dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
    ):
```

Everything else (RotaryEmbedding, LatCHAttention, LatCHBlock, TimestepEmbedder, forward) is unchanged. The full file to reproduce is at the path above — do not paraphrase it; copy it and edit the one default.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_latch_model.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/latch/latch_model.py tests/test_latch_model.py
git commit -m "feat(latch): port LatCH head with 256-dim SAME input"
```

---

## Task 4: Dataset re-encode script (SAME-S) [INTEGRATION]

Re-encodes a corpus of audio clips through SAME-S into SA3 latents, writing per-clip `.npy` (256×T) and a `.json` sidecar with the crop key and latent frame count (SA3-3). This is the long-pole data step: it needs the GPU, the source audio, and an HF download of `same-s`.

**Files:**
- Create: `scripts/latch/encode_latch_dataset.py`

- [ ] **Step 1: Write the encode script**

```python
# scripts/latch/encode_latch_dataset.py
"""Re-encode an audio corpus through SAME (SA3) into per-clip latents for LatCH training.

Output per clip <stem>:
  <out>/<stem>.npy   float32 latent of shape (256, T)
  <out>/<stem>.json  {"crop_key": <stem>, "latent_frames": T, "seconds": <float>}

Usage:
  uv run python scripts/latch/encode_latch_dataset.py \
    --model same-s --audio-dir /path/to/clips --out-dir /path/to/sa3_latents
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from stable_audio_3 import AutoencoderModel

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
STEM_SUFFIXES = ("_bass", "_drums", "_other", "_vocals")


def main(args):
    ae = AutoencoderModel.from_pretrained(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = [
        p for p in sorted(Path(args.audio_dir).rglob("*"))
        if p.suffix.lower() in AUDIO_EXTS
        and not any(p.stem.endswith(s) for s in STEM_SUFFIXES)
    ]
    print(f"Found {len(audio_paths)} audio files to encode with {args.model}.")

    for i, path in enumerate(audio_paths):
        stem = path.stem
        npy_path = out_dir / f"{stem}.npy"
        if npy_path.exists() and not args.overwrite:
            continue
        try:
            wav, sr = torchaudio.load(str(path))           # (C, N)
            latent = ae.encode(wav, sr)                     # (1, 256, T)
            latent = latent.squeeze(0).float().cpu().numpy()  # (256, T)
            np.save(str(npy_path), latent)
            meta = {
                "crop_key": stem,
                "latent_frames": int(latent.shape[1]),
                "seconds": float(wav.shape[-1] / sr),
            }
            (out_dir / f"{stem}.json").write_text(json.dumps(meta))
        except Exception as e:
            print(f"  SKIP {stem}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(audio_paths)} encoded")

    print(f"Done. Latents in {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="same-s", choices=["same-s", "same-l"])
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
```

- [ ] **Step 2: Smoke-encode a handful of clips [INTEGRATION]**

Run (point `--audio-dir` at a small subdir first to validate end-to-end):

```bash
uv run python scripts/latch/encode_latch_dataset.py \
  --model same-s \
  --audio-dir "/run/media/kim/Mantu/ai-music/Goa_Separated_crops" \
  --out-dir /run/media/kim/Lehto/sa3-latch-latents
```

Expected: `.npy` files written; each loads as shape `(256, T)` with `T ≈ round(seconds * 44100 / 4096)`. Verify one:

```bash
uv run python -c "import numpy as np; a=np.load('/run/media/kim/Lehto/sa3-latch-latents/'+__import__('os').listdir('/run/media/kim/Lehto/sa3-latch-latents')[0]); print(a.shape, a.dtype)"
```

Expected: `(256, <T>) float32`.

- [ ] **Step 3: Commit**

```bash
git add scripts/latch/encode_latch_dataset.py
git commit -m "feat(latch): SAME re-encode script for SA3 latents"
```

---

## Task 5: SA3 LatCH dataset

Reads SA3 `.npy` latents (256×T) and resamples the MIR target to each clip's `T` (SA3-2, SA3-4, SA3-5). Variable T per item means batching needs a collate that pads and returns a length mask (SA3-6) — used by the trainer.

**Files:**
- Create: `scripts/latch/latch_dataset.py`
- Test: `tests/test_latch_dataset.py` (CPU, uses a fake DB injected via constructor)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_latch_dataset.py
import json
import numpy as np
import torch

from scripts.latch.latch_dataset import LatCHDataset, collate_varlen


class FakeDB:
    """Stands in for mir TimeseriesDB: returns a 256-frame series per key."""
    def __init__(self, store):
        self._store = store
    def get(self, key):
        return self._store.get(key)


def _write_clip(tmp_path, stem, frames):
    np.save(tmp_path / f"{stem}.npy", np.zeros((256, frames), dtype=np.float32))
    (tmp_path / f"{stem}.json").write_text(
        json.dumps({"crop_key": stem, "latent_frames": frames, "seconds": frames * 0.0929})
    )


def test_item_resamples_target_to_latent_frames(tmp_path):
    _write_clip(tmp_path, "A - T_0", 128)
    db = FakeDB({"A - T_0": {"rms_energy_bass_ts": np.linspace(-60, -10, 256).astype(np.float32)}})
    ds = LatCHDataset(str(tmp_path), target_feature="rms_energy_bass", db=db)
    latent, target = ds[0]
    assert latent.shape == (256, 128)
    assert target.shape == (1, 128)          # resampled from 256 -> 128


def test_collate_pads_and_masks(tmp_path):
    _write_clip(tmp_path, "A - T_0", 100)
    _write_clip(tmp_path, "B - T_0", 150)
    db = FakeDB({
        "A - T_0": {"rms_energy_bass_ts": np.zeros(256, dtype=np.float32)},
        "B - T_0": {"rms_energy_bass_ts": np.ones(256, dtype=np.float32)},
    })
    ds = LatCHDataset(str(tmp_path), target_feature="rms_energy_bass", db=db)
    batch = collate_varlen([ds[0], ds[1]])
    assert batch["latents"].shape == (2, 256, 150)
    assert batch["targets"].shape == (2, 1, 150)
    assert batch["mask"].shape == (2, 150)
    # First item (len 100) has 100 valid then 50 padded frames.
    assert batch["mask"][0, :100].all() and not batch["mask"][0, 100:].any()
    assert batch["mask"][1].all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_latch_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.latch.latch_dataset'`

- [ ] **Step 3: Write the dataset**

```python
# scripts/latch/latch_dataset.py
"""LatCH dataset for SA3: SAME latents (256xT) + MIR target resampled to each clip's T."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.latch.latch_targets import resample_target


def _open_default_db():
    sys.path.insert(0, "/home/kim/Projects/mir/src")
    from core.timeseries_db import TimeseriesDB, DEFAULT_DB_PATH
    return TimeseriesDB.open(DEFAULT_DB_PATH)


class LatCHDataset(Dataset):
    def __init__(self, latent_dir: str, target_feature: str = "rms_energy_bass",
                 db=None, db_path: Optional[str] = None):
        self.latent_dir = Path(latent_dir)
        self.bare_feature = target_feature.removesuffix("_ts")
        self.ts_feature = self.bare_feature + "_ts"
        self.items = sorted(p for p in self.latent_dir.glob("*.npy"))
        if not self.items:
            raise RuntimeError(f"No .npy latents in {latent_dir}")
        if db is not None:
            self._db = db
        elif db_path is not None:
            sys.path.insert(0, "/home/kim/Projects/mir/src")
            from core.timeseries_db import TimeseriesDB
            self._db = TimeseriesDB.open(db_path)
        else:
            self._db = _open_default_db()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        start = idx
        while True:
            npy_path = self.items[idx]
            try:
                latent = np.load(str(npy_path)).astype(np.float32)  # (256, T)
                t_frames = latent.shape[1]
                arrays = self._db.get(npy_path.stem)
                if arrays is None or arrays.get(self.ts_feature) is None:
                    raise ValueError(f"{self.ts_feature} missing for {npy_path.stem}")
                target = resample_target(arrays[self.ts_feature], t_frames)  # (C, T)
                return torch.from_numpy(latent), torch.from_numpy(target)
            except Exception:
                idx = (idx + 1) % len(self.items)
                if idx == start:
                    raise RuntimeError("No valid items in dataset.")


def collate_varlen(batch):
    """Pad a list of (latent (256,T), target (C,T)) to the max T; return a length mask."""
    max_t = max(lat.shape[1] for lat, _ in batch)
    c_out = batch[0][1].shape[0]
    latents = torch.zeros(len(batch), 256, max_t)
    targets = torch.zeros(len(batch), c_out, max_t)
    mask = torch.zeros(len(batch), max_t, dtype=torch.bool)
    for i, (lat, tgt) in enumerate(batch):
        t = lat.shape[1]
        latents[i, :, :t] = lat
        targets[i, :, :t] = tgt
        mask[i, :t] = True
    return {"latents": latents, "targets": targets, "mask": mask}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_latch_dataset.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/latch/latch_dataset.py tests/test_latch_dataset.py
git commit -m "feat(latch): SA3 dataset with per-clip target resampling + varlen collate"
```

---

## Task 6: Training loop (flow-matching forward noise + masked loss)

Trains the head. Forward-noising uses the flow-matching linear interpolation, which is the `rectified_flow` branch (SA3-7); a head trained this way serves both base-Euler and post-trained-pingpong (SA3-8). Loss is masked to valid frames (SA3-6). The v2 checkpoint records `noise_schedule="rectified_flow"` and `loss_type`.

**Files:**
- Create: `scripts/latch/train_latch.py`
- Test: `tests/test_latch_train_helpers.py` (CPU; tests the pure helpers, not the full loop)

- [ ] **Step 1: Write the failing test for the helpers**

```python
# tests/test_latch_train_helpers.py
import torch

from scripts.latch.train_latch import forward_noise, masked_mse


def test_forward_noise_linear_endpoints():
    z0 = torch.ones(2, 256, 10)
    noise = torch.zeros(2, 256, 10)
    # t=0 -> clean z0
    zt0 = forward_noise(z0, noise, torch.zeros(2))
    assert torch.allclose(zt0, z0, atol=1e-6)
    # t=1 -> pure noise (here zeros)
    zt1 = forward_noise(z0, noise, torch.ones(2))
    assert torch.allclose(zt1, noise, atol=1e-6)


def test_masked_mse_ignores_padding():
    pred = torch.zeros(1, 1, 4)
    target = torch.tensor([[[0.0, 0.0, 100.0, 100.0]]])  # last 2 are "padding"
    mask = torch.tensor([[True, True, False, False]])
    loss = masked_mse(pred, target, mask)
    assert loss.item() == 0.0  # padded huge errors excluded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_latch_train_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.latch.train_latch'`

- [ ] **Step 3: Write the trainer**

```python
# scripts/latch/train_latch.py
"""Train a LatCH head on SA3 SAME latents (Phase 1: rms_energy_bass, flow-matching schedule)."""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.latch.latch_dataset import LatCHDataset, collate_varlen
from scripts.latch.latch_model import LatCH


def forward_noise(z0, noise, t):
    """Flow-matching linear interpolation z_t = (1-t)*z0 + t*noise. t: (B,)."""
    t = t.view(-1, 1, 1)
    return (1.0 - t) * z0 + t * noise


def masked_mse(pred, target, mask):
    """MSE over valid (mask=True) frames only. mask: (B, T)."""
    m = mask.unsqueeze(1).to(pred.dtype)          # (B, 1, T)
    se = ((pred - target) ** 2) * m
    denom = m.sum() * pred.shape[1]
    return se.sum() / denom.clamp(min=1.0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LatCHDataset(args.latent_dir, target_feature=args.feature, db_path=args.db_path)
    sample_latent, sample_target = ds[0]
    out_channels = sample_target.shape[0]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, collate_fn=collate_varlen,
                        persistent_workers=args.num_workers > 0)

    model = LatCH(in_channels=256, out_channels=out_channels,
                  dim=256, depth=6, num_heads=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_type = "mse"  # Phase 1 targets rms_energy_bass

    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            latents = batch["latents"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            t = torch.rand(latents.shape[0], device=device)
            noise = torch.randn_like(latents)
            z_t = forward_noise(latents, noise, t)
            preds = model(z_t, t)
            loss = masked_mse(preds, targets, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"epoch {epoch+1}/{args.epochs}  loss={total/len(loader):.4f}")
        torch.save({
            "state_dict": model.state_dict(),
            "feature_name": args.feature,
            "noise_schedule": "rectified_flow",
            "loss_type": loss_type,
            "in_channels": 256,
            "out_channels": out_channels,
        }, os.path.join(args.save_dir, f"latch_sa3_{args.feature}_ep{epoch+1}.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature", default="rms_energy_bass")
    p.add_argument("--latent-dir", default="/run/media/kim/Lehto/sa3-latch-latents")
    p.add_argument("--db-path", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-dir", default="latch_weights_sa3")
    train(p.parse_args())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_latch_train_helpers.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/latch/train_latch.py tests/test_latch_train_helpers.py
git commit -m "feat(latch): SA3 training loop with flow-matching noise + masked loss"
```

- [ ] **Step 6: Train the head [INTEGRATION]**

Run (after Task 4 has produced latents):

```bash
uv run python scripts/latch/train_latch.py --feature rms_energy_bass --epochs 10
```

Expected: loss decreases epoch over epoch; checkpoints `latch_weights_sa3/latch_sa3_rms_energy_bass_ep{1..10}.pt` written. Per the SAO memory, large MSE on dB targets is normal; what matters is a downward trend.

---

## Task 7: Grad-enabled guided Euler sampler

A parallel, gradient-enabled mirror of `sample_discrete_euler` (SA3-9). Variance guidance queries the head at the true `t_curr` (SA3-10); mean guidance acts on the clean estimate `x̂₀ = x − t_curr·v` (SA3-11). The window is σ-relative (SA3-17): a step participates when `t_curr ∈ [sigma_lo, sigma_hi]`, not by step index. Per-step strength is scaled by `s_t = α/Σα` with `α = 1−t` (carried over from SAO). Conditioning is forwarded to the backbone unchanged (SA3-12); 2D per-element schedules are supported (SA3-13).

We unit-test the math with a toy linear model + toy head on CPU — no GPU, no SA3 weights.

**Files:**
- Create: `stable_audio_3/inference/latch_guided.py`
- Test: `tests/test_latch_guided.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_latch_guided.py
import torch

from stable_audio_3.inference.latch_guided import sample_flow_euler_latch_guided


def _toy_model(x, t, **kw):
    # Constant velocity field: returns x (so denoised = x - t*x).
    return x


class _ConstHead(torch.nn.Module):
    """Predicts the per-frame mean of the latent; differentiable, channel-collapsing."""
    def forward(self, z, t):
        return z.mean(dim=1, keepdim=True)  # (B, 1, T)


def _schedule(steps):
    return torch.linspace(1.0, 0.0, steps + 1)


def test_zero_gain_matches_plain_euler():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    sigmas = _schedule(6)
    head = _ConstHead()
    target = torch.zeros(1, 1, 8)
    guided = sample_flow_euler_latch_guided(
        _toy_model, x.clone(), sigmas, head=head, target=target,
        rho=0.0, mu=0.0, window=(0.0, 1.0),
    )
    # Replicate plain euler.
    y = x.clone()
    for i in range(6):
        v = _toy_model(y, sigmas[i].expand(1))
        y = y + (sigmas[i + 1] - sigmas[i]) * v
    assert torch.allclose(guided, y, atol=1e-5)


def test_positive_gain_moves_toward_target():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    sigmas = _schedule(20)
    head = _ConstHead()
    target = torch.full((1, 1, 8), -5.0)   # push the latent mean down
    out_unguided = sample_flow_euler_latch_guided(
        _toy_model, x.clone(), sigmas, head=head, target=target,
        rho=0.0, mu=0.0, window=(0.0, 1.0))
    out_guided = sample_flow_euler_latch_guided(
        _toy_model, x.clone(), sigmas, head=head, target=target,
        rho=0.0, mu=5.0, window=(0.0, 1.0))
    # Mean guidance toward a negative target should lower the predicted mean.
    assert head(out_guided, torch.zeros(1)).mean() < head(out_unguided, torch.zeros(1)).mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_latch_guided.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stable_audio_3.inference.latch_guided'`

- [ ] **Step 3: Write the guided sampler**

```python
# stable_audio_3/inference/latch_guided.py
"""Gradient-enabled Euler sampler with LatCH Training-Free Guidance for SA3 (Phase 1)."""

import torch
from tqdm import tqdm


def _st_weights(sigmas_1d):
    """Per-step s_t = alpha / sum(alpha), alpha = 1 - t. sigmas_1d: (steps+1,)."""
    alpha = (1.0 - sigmas_1d[:-1]).clamp(min=0.0)
    denom = alpha.sum().clamp(min=1e-8)
    return alpha / denom


def sample_flow_euler_latch_guided(
    model, x, sigmas, *, head, target,
    rho=1.0, mu=1.0, gamma=0.3, n_iter=4,
    window=(0.5, 1.0), loss_type="mse",
    disable_tqdm=False, **model_kwargs,
):
    """Euler sampling with selective TFG from a LatCH head.

    rho: variance-guidance strength on z_t (head queried at true t_curr).
    mu:  mean-guidance strength on the clean estimate x_hat0 (head queried at t=0).
    gamma: input-noise augmentation std for the clean head evaluation.
    window: (sigma_lo, sigma_hi) -- guidance active when sigma_lo <= t_curr <= sigma_hi.
    """
    per_element = sigmas.dim() == 2
    sigmas = sigmas.to(x.device)
    num_steps = sigmas.shape[-1] - 1
    target = target.to(x.device)

    sigmas_1d = sigmas[0] if per_element else sigmas
    st = _st_weights(sigmas_1d).to(x.device)
    lo, hi = window

    def head_loss(pred, tgt):
        if loss_type == "bce_logits":
            return torch.nn.functional.binary_cross_entropy_with_logits(pred, tgt)
        return torch.nn.functional.mse_loss(pred, tgt)

    for i in tqdm(range(num_steps), disable=disable_tqdm):
        if per_element:
            t_curr = sigmas[:, i].to(x.dtype)
            dt = (sigmas[:, i + 1] - sigmas[:, i]).view(-1, 1, 1)
        else:
            t_curr = sigmas[i].to(x.dtype) * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            dt = (sigmas[i + 1] - sigmas[i])

        t_scalar = float(t_curr.flatten()[0])
        active = (lo <= t_scalar <= hi)
        rho_t = rho * float(st[i])
        mu_t = mu * float(st[i])

        # --- Variance guidance on z_t (head queried at true t_curr) ---
        if active and rho_t > 0:
            x = x.detach().requires_grad_(True)
            pred = head(x, t_curr)
            loss = head_loss(pred, target)
            grad = torch.autograd.grad(loss, x)[0]
            x = (x - rho_t * grad).detach()

        # --- Model velocity (no grad needed through the DiT for mean guidance) ---
        with torch.no_grad():
            v = model(x, t_curr, **model_kwargs)
        x_hat0 = x - t_curr.view(-1, 1, 1) * v

        # --- Mean guidance on the clean estimate (head queried at t=0) ---
        if active and mu_t > 0:
            z0 = x_hat0.detach()
            t0 = torch.zeros_like(t_curr)
            for _ in range(n_iter):
                z0 = z0.detach().requires_grad_(True)
                aug = z0 + gamma * torch.randn_like(z0)
                loss = head_loss(head(aug, t0), target)
                grad = torch.autograd.grad(loss, z0)[0]
                z0 = (z0 - mu_t * grad).detach()
            x_hat0 = z0

        # --- Euler update reconstructed from the (possibly guided) clean estimate ---
        # v_eff = (x - x_hat0) / t_curr ; x_next = x + dt * v_eff
        t_b = t_curr.view(-1, 1, 1).clamp(min=1e-6)
        v_eff = (x - x_hat0) / t_b
        x = (x + dt * v_eff).detach()

    return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_latch_guided.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add stable_audio_3/inference/latch_guided.py tests/test_latch_guided.py
git commit -m "feat(latch): grad-enabled guided Euler sampler with sigma-relative window"
```

---

## Task 8: Closed-loop verification [INTEGRATION]

Proves *control*, not just *change* (the SAO verification principle): generate at several requested levels, decode, run mir's real extractor, and check the measured feature tracks the request (correlation, monotonic). This is the Phase-1 success gate.

**Files:**
- Create: `scripts/latch/verify_latch.py`

- [ ] **Step 1: Write the verification script**

```python
# scripts/latch/verify_latch.py
"""Closed-loop control check for a trained SA3 LatCH head (Phase 1: rms_energy_bass).

Generates with guidance at several constant target levels, decodes, runs mir's real
rms_energy_bass extractor, and reports correlation(requested, measured).
"""

import argparse
import sys

import numpy as np
import torch

from stable_audio_3 import StableAudioModel
from stable_audio_3.inference.latch_guided import sample_flow_euler_latch_guided
from scripts.latch.latch_model import LatCH
from scripts.latch.latch_targets import build_target


def load_head(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    head = LatCH(in_channels=256, out_channels=ckpt["out_channels"],
                 dim=256, depth=6, num_heads=8).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head, ckpt


def measured_bass_rms(audio_np, sr):
    sys.path.insert(0, "/home/kim/Projects/mir/src")
    from spectral.timeseries_features import _compute_multiband_rms_ts
    ts = _compute_multiband_rms_ts(audio_np, sr)  # dB series
    return float(np.mean(ts))


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StableAudioModel.from_pretrained("small-base", device=device)
    head, ckpt = load_head(args.ckpt, device)
    sr = model.model.sample_rate

    requested, measured = [], []
    for level in args.levels:
        torch.manual_seed(args.seed)
        # Build conditioning + noise via the public path, then run the guided sampler.
        audio = model.generate(
            prompt=args.prompt, duration=args.duration, steps=args.steps,
            cfg_scale=args.cfg_scale, seed=args.seed, return_latents=False,
            sampler_kwargs_hook=None,  # placeholder; see Step 2 wiring note
        ) if False else None  # real wiring in Step 2

        # The actual guided generation is wired in Step 2; here we record the API shape.
        requested.append(level)

    print("Wiring completed in Step 2.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--levels", type=float, nargs="+", default=[-50, -30, -10])
    p.add_argument("--prompt", default="124BPM acid techno, dry snappy kick and bassline")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
```

- [ ] **Step 2: Wire the guided sampler into generation**

`StableAudioModel.generate()` calls `sample_diffusion` internally and does not expose a guided-sampler hook. For Phase-1 verification, replicate the minimal pre-amble that `generate()` performs (conditioning + noise), then call `sample_flow_euler_latch_guided` directly with `model.model`. Replace the `main()` body in `verify_latch.py` with:

```python
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StableAudioModel.from_pretrained("small-base", device=device)
    head, ckpt = load_head(args.ckpt, device)
    sr = model.model.sample_rate

    # Build conditioning exactly as generate() does (prompt + seconds_total).
    cond, neg = StableAudioModel._build_conditioning_dicts(
        args.prompt, None, args.duration, 1)
    audio_sample_size = model._adapt_sample_size(cond, 5292032, 6.0)
    ds_ratio = model.model.pretransform.downsampling_ratio
    latent_len = audio_sample_size // ds_ratio
    cond_tensors = model.model.conditioner(cond, device)
    cond_tensors["inpaint_mask"] = [torch.zeros(1, 1, latent_len, device=device)]
    cond_tensors["inpaint_masked_input"] = [torch.zeros(1, model.model.io_channels, latent_len, device=device)]
    cond_inputs = model.model.get_conditioning_inputs(cond_tensors)
    model_dtype = next(model.model.model.parameters()).dtype
    cond_inputs = {k: (v.type(model_dtype) if v is not None else v) for k, v in cond_inputs.items()}

    from stable_audio_3.inference.sampling import build_schedule
    requested, measured = [], []
    for level in args.levels:
        torch.manual_seed(args.seed)
        noise = torch.randn(1, model.model.io_channels, latent_len, device=device).type(model_dtype)
        sigmas = build_schedule(steps=args.steps, sigma_max=1.0,
                                dist_shift=model.model.sampling_dist_shift,
                                fallback_seq_len=latent_len, include_endpoint=True, device=device)
        target = torch.from_numpy(
            build_target("constant", level, latent_len, 1)).to(device).type(model_dtype)
        latents = sample_flow_euler_latch_guided(
            model.model.model, noise, sigmas, head=head, target=target,
            rho=args.gain, mu=args.gain, gamma=0.3, n_iter=6,
            window=(args.win_lo, args.win_hi), loss_type=ckpt["loss_type"],
            cfg_scale=args.cfg_scale, batch_cfg=True, rescale_cfg=True, apg_scale=1.0,
            **{k: v for k, v in cond_inputs.items()})
        with torch.no_grad():
            audio = model.model.pretransform.decode(latents.type(
                next(model.model.pretransform.parameters()).dtype))
        audio_np = audio.squeeze(0).float().cpu().numpy()
        m = measured_bass_rms(audio_np, sr)
        requested.append(level); measured.append(m)
        print(f"requested {level:>6.1f} dB  ->  measured {m:>7.2f} dB")

    corr = float(np.corrcoef(requested, measured)[0, 1])
    monotonic = all(x < y for x, y in zip(measured, measured[1:]))
    print(f"\ncorrelation(requested, measured) = {corr:.3f}   monotonic = {monotonic}")
```

Add the missing args to the parser: `--gain` (default 8.0), `--win-lo` (default 0.4), `--win-hi` (default 1.0).

- [ ] **Step 3: Run the verification [INTEGRATION]**

Run (after Task 6 trained a head):

```bash
uv run python scripts/latch/verify_latch.py \
  --ckpt latch_weights_sa3/latch_sa3_rms_energy_bass_ep10.pt \
  --levels -50 -30 -10 --gain 8.0 --win-lo 0.4 --win-hi 1.0
```

Expected (Phase-1 success gate): `measured` increases with `requested`, `correlation ≥ 0.9`, `monotonic = True`. Absolute offset is acceptable (SAO showed perfect relative tracking with a low offset). If correlation is low, sweep `--gain` (3→10) and `--win-lo` (0.3→0.6) — the σ-relative window means these transfer more predictably than the SAO step-index window.

- [ ] **Step 4: Commit**

```bash
git add scripts/latch/verify_latch.py
git commit -m "feat(latch): closed-loop control verification on SA3 small-base"
```

---

## Self-Review

**Spec coverage (SA3-1..SA3-18):**
- SA3-1 head 256-dim → Task 3. SA3-2 frame rate → Tasks 4–5. SA3-3 re-encode → Task 4. SA3-4 target resample → Tasks 2, 5. SA3-5 varlen → Tasks 3, 5. SA3-6 masked loss → Task 6. SA3-7 flow-matching noise → Task 6. SA3-8 one head/two samplers → Task 6 metadata. SA3-9 separate grad sampler → Task 7. SA3-10 query at t_curr → Task 7. SA3-11 act on x̂₀ → Task 7. SA3-12 forward conditioning → Task 7 + Task 8 wiring. SA3-13 2D schedules → Task 7 (`per_element`). SA3-14 base/Euler window carries over → Task 7. SA3-17 σ-relative window → Task 7. SA3-18 lands in `stable_audio_3`/`scripts` → all tasks.
- **Deferred to Phase 2 (intentional):** SA3-15, SA3-16 (ping-pong) — stated in plan boundaries.

**Placeholder scan:** Task 8 Step 1 contains an intentional `if False` stub that is *fully replaced* by Step 2 — flagged in-text, not a silent placeholder. No other TBDs.

**Type consistency:** `LatCH(in_channels=256, out_channels=..., dim=256, depth=6, num_heads=8)` identical across Tasks 3/6/8. Checkpoint dict keys (`state_dict`, `out_channels`, `loss_type`, `noise_schedule`) written in Task 6, read in Task 8. `collate_varlen` returns `{"latents","targets","mask"}` (Task 5), consumed identically in Task 6. `sample_flow_euler_latch_guided(model, x, sigmas, *, head, target, rho, mu, gamma, n_iter, window, loss_type, ...)` signature identical in Tasks 7 and 8. `build_target(kind, value, n_frames, n_channels)` and `resample_target(arr, n_frames)` consistent across Tasks 2/5/8.

**Known risk:** Task 8 replicates `generate()`'s conditioning pre-amble by hand; if SA3's conditioning API differs at execution time, reconcile against the live `model.py:247-313` before running Step 3 (do not guess — read it).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-26-latch-sa3-phase1.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

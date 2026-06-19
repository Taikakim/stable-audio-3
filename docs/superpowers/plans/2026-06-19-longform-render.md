# Long-form SA3 Render (SDEdit re-anchor + crossfade) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render arbitrarily-long SA3 audio to one `.wav` with no drift/collapse, via sliding-window inpaint-continuation stitched by crossfades, with a swappable per-chunk generator seam.

**Architecture:** A `LongFormRenderer` walks a `PromptSchedule` and asks a `ChunkGenerator` for one window at a time, each window clamped (SA3 inpaint, in latent space) to the previous window's tail so generation stays in-distribution (no FIFO drift). `CrossfadeStitcher` joins chunks in latent space; `SDEditReanchor` morphs prompt transitions; `DriftMonitor` is telemetry. The `ChunkGenerator` interface is the seam: `InpaintContinuationGenerator` now, `BoundedFifoGenerator` (Approach C) later.

**Tech Stack:** Python 3.13, PyTorch (ROCm), `stable_audio_3` (`StableAudioModel`, `sample_diffusion`), `soundfile`, `pytest`.

## Global Constraints
- Python 3.13; run everything with `/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python` (alias `uv run` in repo).
- Branch: `latch-sa3-phase1`. Commit after every task.
- All sampler/guidance math in **fp32**; cast to model dtype only for the model forward.
- Latent grid: **10.767 fps** (`sample_rate / pretransform.downsampling_ratio`, = 44100/4096). Convert seconds→frames with `round(sec * fps)`.
- Audio writes go through **soundfile** (`sf.write(path, wav.T, sr)`), never `torchaudio.save` (torchcodec absent on the 7.14 venv).
- CPU unit tests must run with **no model and no GPU** (use the fake generator). GPU integration tests are marked `@pytest.mark.skipif(not torch.cuda.is_available())`.
- `ruff` clean: `uv run ruff check stable_audio_3/inference/longform.py scripts/longform_render.py`.

---

### Task 1: `PromptSchedule`

**Files:**
- Create: `stable_audio_3/inference/longform.py`
- Test: `tests/test_longform.py`

**Interfaces:**
- Produces: `PromptSchedule(spec)` where `spec` is `str | list[tuple[float, str]]`; method `resolve(t: float) -> tuple[str, bool, float]` returning `(prompt, is_transition, crossfade_sec)`. `is_transition` is True only on the first `resolve` at-or-after a boundary time. `total_entries() -> int`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_longform.py
import pytest
from stable_audio_3.inference.longform import PromptSchedule

def test_single_prompt_no_transitions():
    s = PromptSchedule("acid techno")
    assert s.total_entries() == 1
    p, is_tr, xf = s.resolve(0.0)
    assert p == "acid techno" and is_tr is False
    assert s.resolve(99.0)[0] == "acid techno"

def test_schedule_transitions_fire_once():
    s = PromptSchedule([(0.0, "A"), (10.0, "B")], crossfade_sec=4.0)
    assert s.resolve(0.0) == ("A", False, 4.0)
    assert s.resolve(5.0) == ("A", False, 4.0)
    assert s.resolve(10.0) == ("B", True, 4.0)   # boundary crossed -> transition
    assert s.resolve(12.0) == ("B", False, 4.0)  # already in B -> no repeat
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_longform.py -k PromptSchedule -v` (also `test_single_prompt`/`test_schedule`)
Expected: FAIL — `ImportError: cannot import name 'PromptSchedule'`.

- [ ] **Step 3: Write minimal implementation**
```python
# stable_audio_3/inference/longform.py
"""Long-form SA3 generation: sliding-window inpaint-continuation + crossfade.

See docs/superpowers/specs/2026-06-19-longform-sdedit-reanchor-crossfade-design.md.
Approach A (drift-free by clamping); the ChunkGenerator interface is the seam for
Approach C (bounded FIFO). slerp is reimplemented locally (ref: mir latent_crossfader).
"""
from __future__ import annotations

from dataclasses import dataclass


class PromptSchedule:
    def __init__(self, spec, crossfade_sec: float = 4.0):
        if isinstance(spec, str):
            self._entries = [(0.0, spec)]
        else:
            self._entries = sorted(((float(t), p) for t, p in spec), key=lambda e: e[0])
            if not self._entries or self._entries[0][0] > 0.0:
                raise ValueError("schedule must have an entry at t=0.0")
        self._crossfade_sec = float(crossfade_sec)
        self._last_index = -1  # which entry resolve() last reported, for transition edge

    def total_entries(self) -> int:
        return len(self._entries)

    def resolve(self, t: float) -> tuple[str, bool, float]:
        idx = 0
        for i, (start, _) in enumerate(self._entries):
            if t >= start:
                idx = i
        prompt = self._entries[idx][1]
        is_transition = idx != self._last_index and self._last_index != -1
        self._last_index = idx
        return prompt, is_transition, self._crossfade_sec
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_longform.py -k "single_prompt or schedule_transitions" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**
```bash
git add stable_audio_3/inference/longform.py tests/test_longform.py
git commit -m "feat(longform): PromptSchedule (single + scheduled prompts)"
```

---

### Task 2: `slerp` + `CrossfadeStitcher`

**Files:**
- Modify: `stable_audio_3/inference/longform.py`
- Test: `tests/test_longform.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `slerp(a, b, t) -> Tensor` (a,b same shape `(...,)`, t float or 1-D broadcast on last dim); `CrossfadeStitcher(blend_frames: int)` with `continuation_join(current_tail, new_region) -> Tensor` (crossfades the first `blend_frames` of `new_region` with `current_tail`, returns the full new region to append) and `transition_join(out_tail, in_head, n) -> Tensor` (length-`n` crossfade region). All operate on latents shaped `(1, C, T)`.

- [ ] **Step 1: Write the failing test**
```python
import torch
from stable_audio_3.inference.longform import slerp, CrossfadeStitcher

def test_slerp_endpoints():
    a = torch.randn(1, 8, 4); b = torch.randn(1, 8, 4)
    assert torch.allclose(slerp(a, b, 0.0), a, atol=1e-5)
    assert torch.allclose(slerp(a, b, 1.0), b, atol=1e-5)

def test_continuation_join_length_and_seam():
    st = CrossfadeStitcher(blend_frames=3)
    cur_tail = torch.zeros(1, 8, 3)          # end of current output
    new_region = torch.ones(1, 8, 10)        # start of next chunk's new region
    out = st.continuation_join(cur_tail, new_region)
    assert out.shape == (1, 8, 10)           # same length as new_region
    # first frame blended toward cur_tail (0), last frames untouched (1)
    assert out[..., 0].abs().mean() < 1.0
    assert torch.allclose(out[..., -1], torch.ones(1, 8))

def test_transition_join_length():
    st = CrossfadeStitcher(blend_frames=3)
    out = st.transition_join(torch.zeros(1, 8, 5), torch.ones(1, 8, 5), n=5)
    assert out.shape == (1, 8, 5)
    assert torch.allclose(out[..., 0], torch.zeros(1, 8), atol=1e-5)
    assert torch.allclose(out[..., -1], torch.ones(1, 8), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_longform.py -k "slerp or join" -v`
Expected: FAIL — `ImportError: cannot import name 'slerp'`.

- [ ] **Step 3: Write minimal implementation**
```python
# add to longform.py
import torch


def slerp(a: torch.Tensor, b: torch.Tensor, t):
    """Spherical interpolation per last-dim vector; falls back to lerp when near-collinear.
    Ref: mir scripts/latent_crossfader.py. a,b: (..., D)-ish; here used on (1,C,T) frame-wise."""
    t = torch.as_tensor(t, dtype=a.dtype, device=a.device)
    a32, b32 = a.float(), b.float()
    na = a32 / (a32.norm(dim=1, keepdim=True) + 1e-8)
    nb = b32 / (b32.norm(dim=1, keepdim=True) + 1e-8)
    dot = (na * nb).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    near = so.abs() < 1e-4
    w_a = torch.where(near, 1.0 - t, torch.sin((1.0 - t) * omega) / (so + 1e-8))
    w_b = torch.where(near, t, torch.sin(t * omega) / (so + 1e-8))
    return (w_a * a32 + w_b * b32).to(a.dtype)


class CrossfadeStitcher:
    def __init__(self, blend_frames: int = 3):
        self.blend_frames = int(blend_frames)

    def _ramp(self, n, device, dtype):
        # ramp t over n frames, shape (1,1,n) for broadcast over (1,C,n)
        return torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).view(1, 1, n)

    def continuation_join(self, current_tail, new_region):
        b = min(self.blend_frames, current_tail.shape[-1], new_region.shape[-1])
        if b == 0:
            return new_region
        t = self._ramp(b, new_region.device, new_region.dtype)
        head = slerp(current_tail[..., -b:], new_region[..., :b], t)
        return torch.cat([head, new_region[..., b:]], dim=-1)

    def transition_join(self, out_tail, in_head, n):
        n = min(n, out_tail.shape[-1], in_head.shape[-1])
        t = self._ramp(n, in_head.device, in_head.dtype)
        return slerp(out_tail[..., -n:], in_head[..., :n], t)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_longform.py -k "slerp or join" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**
```bash
git add stable_audio_3/inference/longform.py tests/test_longform.py
git commit -m "feat(longform): local slerp + CrossfadeStitcher (continuation + transition joins)"
```

---

### Task 3: `DriftMonitor`

**Files:**
- Modify: `stable_audio_3/inference/longform.py`
- Test: `tests/test_longform.py`

**Interfaces:**
- Produces: `DriftMonitor(rms_drop_frac=0.6)` with `observe(latents: Tensor) -> dict` (keys `rms`, `centroid_proxy`; updates running median of rms) and `should_reanchor(stats: dict) -> bool` (True when `stats['rms']` < `rms_drop_frac` × running median rms). Operates on latents (no decode) so it's CPU/GPU-agnostic.

- [ ] **Step 1: Write the failing test**
```python
from stable_audio_3.inference.longform import DriftMonitor

def test_drift_monitor_flags_collapse():
    m = DriftMonitor(rms_drop_frac=0.6)
    for _ in range(5):
        st = m.observe(torch.randn(1, 8, 16))   # ~unit RMS
        assert m.should_reanchor(st) is False
    collapsed = m.observe(torch.randn(1, 8, 16) * 0.05)  # RMS collapse
    assert m.should_reanchor(collapsed) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_longform.py -k drift -v`
Expected: FAIL — `cannot import name 'DriftMonitor'`.

- [ ] **Step 3: Write minimal implementation**
```python
# add to longform.py
import statistics


class DriftMonitor:
    def __init__(self, rms_drop_frac: float = 0.6):
        self.rms_drop_frac = float(rms_drop_frac)
        self._rms_history: list[float] = []

    def observe(self, latents: torch.Tensor) -> dict:
        x = latents.float()
        rms = float(x.pow(2).mean().sqrt())
        # cheap spectral proxy: mean abs first-difference along time / rms
        diff = (x[..., 1:] - x[..., :-1]).abs().mean()
        centroid_proxy = float(diff / (rms + 1e-8))
        self._rms_history.append(rms)
        return {"rms": rms, "centroid_proxy": centroid_proxy}

    def should_reanchor(self, stats: dict) -> bool:
        if len(self._rms_history) < 3:
            return False
        med = statistics.median(self._rms_history[:-1])
        return stats["rms"] < self.rms_drop_frac * med
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_longform.py -k drift -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add stable_audio_3/inference/longform.py tests/test_longform.py
git commit -m "feat(longform): DriftMonitor (latent RMS/centroid telemetry + reanchor signal)"
```

---

### Task 4: `ChunkGenerator` interface + `FakeChunkGenerator`

**Files:**
- Modify: `stable_audio_3/inference/longform.py`
- Test: `tests/test_longform.py`

**Interfaces:**
- Produces: abstract `ChunkGenerator` with `generate(prompt: str, prefix_latents: Tensor | None, prefix_frames: int, n_frames: int, seed: int) -> Tensor` returning `(1, C, n_frames)`. `FakeChunkGenerator(channels)` returns deterministic latents whose first `prefix_frames` **equal** `prefix_latents` (simulates a hard clamp) and whose remaining frames are a per-chunk constant, for renderer bookkeeping tests.

- [ ] **Step 1: Write the failing test**
```python
from stable_audio_3.inference.longform import ChunkGenerator, FakeChunkGenerator

def test_fake_generator_honors_prefix_and_shape():
    g = FakeChunkGenerator(channels=8)
    prefix = torch.full((1, 8, 4), 0.5)
    out = g.generate("p", prefix_latents=prefix, prefix_frames=4, n_frames=10, seed=0)
    assert out.shape == (1, 8, 10)
    assert torch.allclose(out[..., :4], prefix)        # clamp region preserved
    assert torch.allclose(out[..., 4:], out[..., 4:5].expand(1, 8, 6))  # constant tail

def test_chunkgenerator_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        ChunkGenerator()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_longform.py -k "fake_generator or abstract" -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**
```python
# add to longform.py
from abc import ABC, abstractmethod


class ChunkGenerator(ABC):
    @abstractmethod
    def generate(self, prompt, prefix_latents, prefix_frames, n_frames, seed) -> torch.Tensor:
        ...


class FakeChunkGenerator(ChunkGenerator):
    """Model-free generator for CPU bookkeeping tests. Honors the clamp exactly."""
    def __init__(self, channels: int):
        self.channels = channels
        self._call = 0

    def generate(self, prompt, prefix_latents, prefix_frames, n_frames, seed):
        self._call += 1
        out = torch.full((1, self.channels, n_frames), float(self._call))
        if prefix_latents is not None and prefix_frames > 0:
            out[..., :prefix_frames] = prefix_latents[..., :prefix_frames]
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_longform.py -k "fake_generator or abstract" -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add stable_audio_3/inference/longform.py tests/test_longform.py
git commit -m "feat(longform): ChunkGenerator interface + FakeChunkGenerator for CPU tests"
```

---

### Task 5: `LongFormRenderer` loop (CPU-tested with the fake generator)

**Files:**
- Modify: `stable_audio_3/inference/longform.py`
- Test: `tests/test_longform.py`

**Interfaces:**
- Consumes: `PromptSchedule`, `ChunkGenerator`, `CrossfadeStitcher`, `DriftMonitor`.
- Produces: `LongFormRenderer(generator, channels, fps, window_frames, overlap_frames, blend_frames, stitcher=None, monitor=None)` with `render_latents(schedule, total_frames, base_seed=0) -> Tensor` returning the assembled `(1, C, total_frames)` latent and logging drift stats per chunk (`.drift_log: list[dict]`).

- [ ] **Step 1: Write the failing test**
```python
from stable_audio_3.inference.longform import (
    LongFormRenderer, FakeChunkGenerator, PromptSchedule)

def test_renderer_length_and_continuity():
    g = FakeChunkGenerator(channels=8)
    r = LongFormRenderer(g, channels=8, fps=10.0, window_frames=20,
                         overlap_frames=5, blend_frames=2)
    lat = r.render_latents(PromptSchedule("x"), total_frames=50, base_seed=0)
    assert lat.shape == (1, 8, 50)                    # exact requested length
    assert torch.isfinite(lat).all()
    assert len(r.drift_log) >= 3                      # one entry per chunk
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_longform.py -k renderer -v`
Expected: FAIL — `cannot import name 'LongFormRenderer'`.

- [ ] **Step 3: Write minimal implementation**
```python
# add to longform.py
class LongFormRenderer:
    def __init__(self, generator, channels, fps, window_frames, overlap_frames,
                 blend_frames=3, stitcher=None, monitor=None):
        self.gen = generator
        self.channels = channels
        self.fps = float(fps)
        self.window = int(window_frames)
        self.overlap = int(overlap_frames)
        self.stitcher = stitcher or CrossfadeStitcher(blend_frames=blend_frames)
        self.monitor = monitor or DriftMonitor()
        self.drift_log: list[dict] = []
        if self.overlap >= self.window:
            raise ValueError("overlap_frames must be < window_frames")

    def render_latents(self, schedule, total_frames, base_seed=0):
        out = None
        prev_tail = None
        k = 0
        while out is None or out.shape[-1] < total_frames:
            t_sec = (out.shape[-1] / self.fps) if out is not None else 0.0
            prompt, is_transition, xf_sec = schedule.resolve(t_sec)
            prefix_frames = 0 if prev_tail is None else self.overlap
            chunk = self.gen.generate(
                prompt, prefix_latents=prev_tail, prefix_frames=prefix_frames,
                n_frames=self.window, seed=base_seed + k)
            self.drift_log.append(self.monitor.observe(chunk))
            if out is None:
                out = chunk
            elif is_transition:
                n = min(int(round(xf_sec * self.fps)), self.overlap)
                joined = self.stitcher.transition_join(out, chunk, n)
                out = torch.cat([out[..., :-n], joined, chunk[..., n:]], dim=-1)
            else:
                new_region = chunk[..., prefix_frames:]
                new_region = self.stitcher.continuation_join(out, new_region)
                out = torch.cat([out, new_region], dim=-1)
            prev_tail = out[..., -self.overlap:]
            k += 1
        return out[..., :total_frames]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_longform.py -k renderer -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add stable_audio_3/inference/longform.py tests/test_longform.py
git commit -m "feat(longform): LongFormRenderer loop (windowed continuation + transitions)"
```

---

### Task 6: `InpaintContinuationGenerator` (real SA3 generation; GPU test deferred)

**Files:**
- Modify: `stable_audio_3/inference/longform.py`
- Test: `tests/test_longform.py`

**Interfaces:**
- Consumes: a loaded `StableAudioModel`, `ChunkGenerator`.
- Produces: `InpaintContinuationGenerator(model)` implementing `generate(...)`. Builds latent-space inpaint conditioning (mask=1 on `[0,prefix_frames)`, masked_input=prefix) and calls `sample_diffusion(..., decode=False)` to return latents `(1, C, n_frames)`.

- [ ] **Step 1: Write the failing test** (GPU-gated integration + a CPU shape guard via monkeypatch)
```python
import pytest, torch
from stable_audio_3.inference.longform import InpaintContinuationGenerator

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU + model")
def test_inpaint_continuation_shape_and_clamp():
    from stable_audio_3 import StableAudioModel
    m = StableAudioModel.from_pretrained("small-music-base", model_half=False)
    g = InpaintContinuationGenerator(m)
    C = m.model.io_channels
    prefix = torch.randn(1, C, 16, device="cuda")
    out = g.generate("acid techno", prefix_latents=prefix, prefix_frames=16,
                     n_frames=128, seed=0)
    assert out.shape == (1, C, 128) and torch.isfinite(out).all()
    # clamp-hardness probe (informational): how close is the clamp region to the prefix?
    err = (out[..., :16].cpu() - prefix.cpu()).abs().mean().item()
    print(f"[clamp] mean abs err in clamp region = {err:.4e}")
```

- [ ] **Step 2: Run test to verify it fails / skips**

Run: `uv run pytest tests/test_longform.py -k inpaint_continuation -v`
Expected: SKIP on CPU (or FAIL `cannot import name` before Step 3). On a GPU box it must at least import.

- [ ] **Step 3: Write minimal implementation**
```python
# add to longform.py
class InpaintContinuationGenerator(ChunkGenerator):
    """Approach A: clamp the prefix in latent space, generate the rest in-distribution."""
    def __init__(self, model, steps: int = 50, cfg_scale: float = 6.0):
        self.model = model            # StableAudioModel
        self.inner = model.model      # ConditionedDiffusionModelWrapper
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.fps = model.model.sample_rate / model.model.pretransform.downsampling_ratio

    def _cond(self, prompt, prefix_latents, prefix_frames, n_frames, device, dtype, batch=1):
        win_seconds = n_frames / self.fps
        conditioning, _ = self.model._build_conditioning_dicts(prompt, None, win_seconds, batch)
        ct = self.inner.conditioner(conditioning, device)
        C = self.inner.io_channels
        mask = torch.zeros((batch, 1, n_frames), device=device)
        masked_input = torch.zeros((batch, C, n_frames), device=device)
        if prefix_latents is not None and prefix_frames > 0:
            mask[:, :, :prefix_frames] = 1.0
            masked_input[:, :, :prefix_frames] = prefix_latents[..., :prefix_frames].to(device)
        ct["inpaint_mask"] = [mask]
        ct["inpaint_masked_input"] = [masked_input]
        ci = self.inner.get_conditioning_inputs(ct)
        return {k: (v.type(dtype) if torch.is_tensor(v) else v) for k, v in ci.items()}, conditioning

    @torch.no_grad()
    def generate(self, prompt, prefix_latents, prefix_frames, n_frames, seed):
        from stable_audio_3.inference.sampling import sample_diffusion
        device = next(self.inner.model.parameters()).device
        dtype = next(self.inner.model.parameters()).dtype
        cond_inputs, conditioning = self._cond(
            prompt, prefix_latents, prefix_frames, n_frames, device, dtype)
        torch.manual_seed(seed)
        noise = torch.randn(1, self.inner.io_channels, n_frames, device=device, dtype=dtype)
        latents = sample_diffusion(
            model=self.inner.model, noise=noise, cond_inputs=cond_inputs,
            diffusion_objective=self.inner.diffusion_objective, steps=self.steps,
            cfg_scale=self.cfg_scale, conditioning=conditioning,
            sample_rate=self.inner.sample_rate, pretransform=self.inner.pretransform,
            mask_padding_attention=True, dist_shift=self.inner.sampling_dist_shift,
            decode=False)
        return latents.float()
```
> NOTE (verify-at-impl): if `sample_diffusion`'s inpaint only *conditions* (soft) rather than hard-replacing the known region, the clamp-region error printed in Step 1 will be > ~1e-2; that is expected and handled by `continuation_join`. If it is large enough that continuity suffers, add a hard replacement in a thin sampler callback (re-noise prefix to each step's σ and overwrite) — out of scope for v1.

- [ ] **Step 4: Run test (GPU box) / confirm import (CPU)**

Run: `uv run pytest tests/test_longform.py -k inpaint_continuation -v`
Expected: PASS on GPU (shape + finite; prints clamp err), SKIP on CPU.

- [ ] **Step 5: Commit**
```bash
git add stable_audio_3/inference/longform.py tests/test_longform.py
git commit -m "feat(longform): InpaintContinuationGenerator (latent-space clamp continuation)"
```

---

### Task 7: `SDEditReanchor` (transition morph / drift refresh; GPU test deferred)

**Files:**
- Modify: `stable_audio_3/inference/longform.py`
- Test: `tests/test_longform.py`

**Interfaces:**
- Consumes: a loaded `StableAudioModel`.
- Produces: `SDEditReanchor(model)` with `reanchor(latents, sigma_peak, prompt, seed) -> Tensor` — re-noise `latents` to `sigma_peak`, denoise back under `prompt`; returns same-shape latents.

- [ ] **Step 1: Write the failing test**
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU + model")
def test_sdedit_reanchor_preserves_shape():
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.inference.longform import SDEditReanchor
    m = StableAudioModel.from_pretrained("small-music-base", model_half=False)
    r = SDEditReanchor(m)
    C = m.model.io_channels
    z = torch.randn(1, C, 128, device="cuda")
    out = r.reanchor(z, sigma_peak=0.5, prompt="acid techno", seed=0)
    assert out.shape == z.shape and torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails / skips**

Run: `uv run pytest tests/test_longform.py -k sdedit -v`
Expected: SKIP on CPU; import error before Step 3.

- [ ] **Step 3: Write minimal implementation**
```python
# add to longform.py
class SDEditReanchor:
    """Triangular re-noise->denoise (audio2audio in latent space) to pull back on-manifold."""
    def __init__(self, model, steps: int = 50, cfg_scale: float = 6.0):
        self.model = model
        self.inner = model.model
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.fps = model.model.sample_rate / model.model.pretransform.downsampling_ratio

    @torch.no_grad()
    def reanchor(self, latents, sigma_peak, prompt, seed):
        from stable_audio_3.inference.sampling import sample_diffusion
        device = next(self.inner.model.parameters()).device
        dtype = next(self.inner.model.parameters()).dtype
        n_frames = latents.shape[-1]
        win_seconds = n_frames / self.fps
        conditioning, _ = self.model._build_conditioning_dicts(prompt, None, win_seconds, 1)
        ct = self.inner.conditioner(conditioning, device)
        ct["inpaint_mask"] = [torch.zeros((1, 1, n_frames), device=device)]
        ct["inpaint_masked_input"] = [torch.zeros((1, self.inner.io_channels, n_frames), device=device)]
        ci = self.inner.get_conditioning_inputs(ct)
        ci = {k: (v.type(dtype) if torch.is_tensor(v) else v) for k, v in ci.items()}
        torch.manual_seed(seed)
        noise = torch.randn_like(latents.to(device=device, dtype=dtype))
        out = sample_diffusion(
            model=self.inner.model, noise=noise, cond_inputs=ci,
            diffusion_objective=self.inner.diffusion_objective, steps=self.steps,
            cfg_scale=self.cfg_scale, conditioning=conditioning,
            sample_rate=self.inner.sample_rate, pretransform=self.inner.pretransform,
            mask_padding_attention=True, dist_shift=self.inner.sampling_dist_shift,
            init_data=latents.to(device=device, dtype=dtype), init_noise_level=float(sigma_peak),
            decode=False)
        return out.float()
```

- [ ] **Step 4: Run test (GPU) / confirm import (CPU)**

Run: `uv run pytest tests/test_longform.py -k sdedit -v`
Expected: PASS on GPU, SKIP on CPU.

- [ ] **Step 5: Commit**
```bash
git add stable_audio_3/inference/longform.py tests/test_longform.py
git commit -m "feat(longform): SDEditReanchor (latent audio2audio re-anchor)"
```

---

### Task 8: CLI `scripts/longform_render.py` (decode + soundfile write)

**Files:**
- Create: `scripts/longform_render.py`
- Test: `tests/test_longform.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `parse_schedule(arg: str) -> str | list[tuple[float,str]]` (CPU-testable) and a `main()` that loads the model, builds an `InpaintContinuationGenerator`, renders latents, decodes (chunked) and writes a `.wav`.

- [ ] **Step 1: Write the failing test** (CPU: schedule parsing only)
```python
from scripts.longform_render import parse_schedule  # noqa

def test_parse_schedule_single_and_list():
    assert parse_schedule("acid techno") == "acid techno"
    assert parse_schedule("0:acid techno|30:breakdown pad") == [(0.0, "acid techno"), (30.0, "breakdown pad")]
```
> If `scripts/` is not importable, add `tests/conftest.py` line `sys.path.insert(0, str(Path(__file__).parent.parent))` (fold into this step).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_longform.py -k parse_schedule -v`
Expected: FAIL — module/function missing.

- [ ] **Step 3: Write minimal implementation**
```python
# scripts/longform_render.py
"""Render long-form SA3 audio via sliding-window continuation. See
docs/superpowers/specs/2026-06-19-longform-sdedit-reanchor-crossfade-design.md."""
from __future__ import annotations
import argparse


def parse_schedule(arg: str):
    if "|" not in arg and ":" not in arg.split(" ")[0]:
        return arg
    entries = []
    for part in arg.split("|"):
        t, _, prompt = part.partition(":")
        entries.append((float(t), prompt))
    return entries


def main():
    import torch, soundfile as sf
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.inference.longform import (
        PromptSchedule, InpaintContinuationGenerator, LongFormRenderer)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small-music-base")
    ap.add_argument("--prompt", required=True, help="prompt, or '0:A|30:B' schedule")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--overlap-sec", type=float, default=5.0)
    ap.add_argument("--blend-frames", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("-o", "--out", default="longform.wav")
    args = ap.parse_args()

    m = StableAudioModel.from_pretrained(args.model, model_half=args.half)
    inner = m.model
    sr = inner.sample_rate
    fps = sr / inner.pretransform.downsampling_ratio
    f = lambda s: int(round(s * fps))

    gen = InpaintContinuationGenerator(m, steps=args.steps, cfg_scale=args.cfg)
    r = LongFormRenderer(gen, channels=inner.io_channels, fps=fps,
                         window_frames=f(args.window_sec), overlap_frames=f(args.overlap_sec),
                         blend_frames=args.blend_frames)
    sched = PromptSchedule(parse_schedule(args.prompt))
    lat = r.render_latents(sched, total_frames=f(args.duration))
    print(f"[longform] latents {tuple(lat.shape)} finite={bool(torch.isfinite(lat).all())}")
    print(f"[longform] drift_log rms: {[round(d['rms'],3) for d in r.drift_log]}")

    with torch.no_grad():
        pt_dtype = next(inner.pretransform.parameters()).dtype
        audio = inner.pretransform.decode(lat.to(pt_dtype)).float().cpu()
    wav = audio[0] if audio.dim() == 3 else audio
    sf.write(args.out, wav.transpose(0, 1).numpy(), int(sr))
    print(f"[longform] saved -> {args.out}")


if __name__ == "__main__":
    main()
```
> NOTE (verify-at-impl): for multi-minute renders, replace the single `pretransform.decode` with overlapped chunked decode (decode ~30 s latent chunks with a few-second overlap, crossfade the decoded audio) — the SAME decoder is convolutional and naïve chunk boundaries seam. Check `sample_diffusion(chunked_decode=...)` first.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_longform.py -k parse_schedule -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/longform_render.py tests/test_longform.py tests/conftest.py
git commit -m "feat(longform): CLI render script (schedule parse, decode, soundfile write)"
```

---

### Task 9: Full CPU suite green + ruff + GPU smoke (manual)

**Files:**
- Modify: none (verification task)

- [ ] **Step 1: Run the full CPU suite**

Run: `uv run pytest tests/test_longform.py -v`
Expected: all non-GPU tests PASS; the `inpaint_continuation`/`sdedit` tests SKIP on CPU.

- [ ] **Step 2: Lint**

Run: `uv run ruff check stable_audio_3/inference/longform.py scripts/longform_render.py`
Expected: All checks passed.

- [ ] **Step 3: GPU smoke (when card is free)**

Run: `uv run python scripts/longform_render.py --prompt "124 BPM acid techno" --duration 120 --window-sec 30 --overlap-sec 5 -o /home/kim/.claude/jobs/$CLAUDE_JOB_DIR/longform_2min.wav`
Expected: prints flat-ish `drift_log rms` across all chunks (the acceptance metric — contrast the FIFO run, which collapsed to ~0.03); saved wav; listen for seamless continuation. Also run the `-k "inpaint_continuation or sdedit"` GPU tests.

- [ ] **Step 4: Commit any fixes from the smoke**
```bash
git add -A && git commit -m "test(longform): CPU suite green; GPU smoke notes"
```

---

## Self-Review
- **Spec coverage:** PromptSchedule (§2)→T1; CrossfadeStitcher+slerp (§2/§3)→T2; DriftMonitor (§2)→T3; ChunkGenerator seam (§1)→T4; LongFormRenderer + clamp/transition mechanics (§1/§3)→T5; InpaintContinuationGenerator (§2)→T6; SDEditReanchor (§2)→T7; decode + soundfile + CLI + chunked-decode note (§4)→T8; testing (§4)→T1–T5 (CPU) + T6/T7/T9 (GPU). C-graft seam present (T4 interface). ✅
- **Placeholder scan:** the two `NOTE (verify-at-impl)` blocks are explicit, scoped known-unknowns from the spec (clamp hardness, chunked decode), not placeholders; all code steps contain full code. ✅
- **Type consistency:** `generate(prompt, prefix_latents, prefix_frames, n_frames, seed)` identical across ChunkGenerator/Fake/Inpaint (T4/T6); `reanchor(latents, sigma_peak, prompt, seed)` (T7); `continuation_join`/`transition_join` used in T5 match T2; `inner = model.model`, `inner.model` (DiTWrapper), `inner.io_channels`, `inner.sample_rate`, `inner.sampling_dist_shift`, `inner.diffusion_objective` consistent T6/T7/T8. ✅

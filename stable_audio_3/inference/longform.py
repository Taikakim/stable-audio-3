"""Long-form SA3 generation: sliding-window inpaint-continuation + crossfade.

See docs/superpowers/specs/2026-06-19-longform-sdedit-reanchor-crossfade-design.md.
Approach A (drift-free by clamping); the ChunkGenerator interface is the seam for
Approach C (bounded FIFO). slerp is reimplemented locally (ref: mir latent_crossfader).
"""
from __future__ import annotations

import statistics
from abc import ABC, abstractmethod

import torch


def slerp(a: torch.Tensor, b: torch.Tensor, t):
    """Spherical interpolation per last-dim vector; falls back to lerp when near-collinear.

    Ref: mir scripts/latent_crossfader.py. a,b: (..., D)-ish; here used on (1,C,T)
    frame-wise.
    """
    t = torch.as_tensor(t, dtype=torch.float32, device=a.device)
    a32, b32 = a.float(), b.float()
    na = a32 / (a32.norm(dim=1, keepdim=True) + 1e-8)
    nb = b32 / (b32.norm(dim=1, keepdim=True) + 1e-8)
    dot = (na * nb).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    near = so.abs() < 1e-4
    so_safe = torch.where(near, torch.ones_like(so), so)  # exact denom off the near path
    w_a = torch.where(near, 1.0 - t, torch.sin((1.0 - t) * omega) / so_safe)
    w_b = torch.where(near, t, torch.sin(t * omega) / so_safe)
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


class PromptSchedule:
    def __init__(self, spec: str | list[tuple[float, str]], crossfade_sec: float = 4.0):
        if isinstance(spec, str):
            self._entries = [(0.0, spec)]
        else:
            self._entries = sorted(((float(t), p) for t, p in spec), key=lambda e: e[0])
            if not self._entries or self._entries[0][0] != 0.0:
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


class ChunkGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, prefix_latents: torch.Tensor | None,
                 prefix_frames: int, n_frames: int, seed: int) -> torch.Tensor:
        """Generate latents for a chunk.

        Args:
            prompt: Text prompt (str)
            prefix_latents: Prior latents to clamp to, shape (1, C, prefix_frames) or None
            prefix_frames: Number of frames from prefix_latents to preserve
            n_frames: Total number of frames to generate
            seed: Random seed for reproducibility

        Returns:
            Latent tensor of shape (1, C, n_frames) where the first prefix_frames
            are clamped to prefix_latents if provided.
        """
        ...


class FakeChunkGenerator(ChunkGenerator):
    """Model-free generator for CPU bookkeeping tests. Honors the clamp exactly."""

    def __init__(self, channels: int):
        self.channels = channels
        self._call = 0

    def generate(self, prompt: str, prefix_latents: torch.Tensor | None,
                 prefix_frames: int, n_frames: int, seed: int) -> torch.Tensor:
        self._call += 1
        out = torch.full((1, self.channels, n_frames), float(self._call))
        if prefix_latents is not None and prefix_frames > 0:
            out[..., :prefix_frames] = prefix_latents[..., :prefix_frames]
        return out


class LongFormRenderer:
    """Walks a PromptSchedule, generating one window at a time via ChunkGenerator.

    Stitches windows via CrossfadeStitcher (continuation_join for same-prompt;
    transition_join for prompt changes), logs DriftMonitor.observe per chunk into
    drift_log, and returns assembled (1, C, total_frames) latent.
    """

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
                n = max(1, min(int(round(xf_sec * self.fps)), self.overlap))
                joined = self.stitcher.transition_join(out, chunk, n)
                out = torch.cat([out[..., :-n], joined, chunk[..., n:]], dim=-1)
            else:
                new_region = chunk[..., prefix_frames:]
                new_region = self.stitcher.continuation_join(out, new_region)
                out = torch.cat([out, new_region], dim=-1)
            prev_tail = out[..., -self.overlap:]
            k += 1
        return out[..., :total_frames]

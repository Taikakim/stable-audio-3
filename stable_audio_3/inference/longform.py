"""Long-form SA3 generation: sliding-window inpaint-continuation + crossfade.

See docs/superpowers/specs/2026-06-19-longform-sdedit-reanchor-crossfade-design.md.
Approach A (drift-free by clamping); the ChunkGenerator interface is the seam for
Approach C (bounded FIFO). slerp is reimplemented locally (ref: mir latent_crossfader).
"""
from __future__ import annotations

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

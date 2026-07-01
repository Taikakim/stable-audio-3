"""Target tensor builders for LatCH guidance (inference side).

Builds a [batch_size, channels, frames] control target on the SA3 latent frame
grid. The training/verify scripts under ``scripts/latch`` have their own numpy
helpers for resampling real MIR series; this module is the synthetic builder the
sampler/UI use to turn UI knobs (a value + a shape kind) into a target.
"""

import math

import torch

KINDS = ("constant", "ramp_up", "ramp_down", "beat_grid")


def build_target(kind, value, batch_size, channels, frames, *, fps, device, dtype):
    """Return a [batch_size, channels, frames] target tensor.

    kind:
        constant   -> every frame == value
        ramp_up     -> 0 .. value across time
        ramp_down   -> value .. 0 across time
        beat_grid   -> impulse train at `value` BPM (value is BPM, needs fps)
    """
    if kind not in KINDS:
        raise ValueError(f"unknown target kind '{kind}', expected one of {KINDS}")

    if kind == "constant":
        return torch.full((batch_size, channels, frames), float(value), device=device, dtype=dtype)

    if kind in ("ramp_up", "ramp_down"):
        ramp = torch.linspace(0.0, float(value), frames, device=device, dtype=dtype)
        if kind == "ramp_down":
            ramp = ramp.flip(0)
        return ramp.view(1, 1, frames).expand(batch_size, channels, frames).contiguous()

    # beat_grid
    bpm = float(value)
    if bpm <= 0:
        raise ValueError(f"beat_grid requires bpm > 0, got {bpm}")
    frames_per_beat = 60.0 * float(fps) / bpm
    n_beats = int(math.floor(frames / frames_per_beat)) + 2
    idx = torch.tensor([round(i * frames_per_beat) for i in range(n_beats)], dtype=torch.long)
    idx = idx[(idx >= 0) & (idx < frames)]
    out = torch.zeros((batch_size, channels, frames), device=device, dtype=dtype)
    out[:, :, idx] = 1.0
    return out

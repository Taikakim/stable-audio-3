"""Long-form SA3 generation: sliding-window inpaint-continuation + crossfade.

See docs/superpowers/specs/2026-06-19-longform-sdedit-reanchor-crossfade-design.md.
Approach A (drift-free by clamping); the ChunkGenerator interface is the seam for
Approach C (bounded FIFO). slerp is reimplemented locally (ref: mir latent_crossfader).
"""
from __future__ import annotations

import statistics
import warnings
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


def swap_join(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SaFa latent-swap join over an aligned overlap: frame-interleave a and b.

    Ref: "Latent Swap Joint Diffusion for 2D Long-Form Latent Generation"
    (arXiv:2502.05130, SaFa), Self-Loop Latent Swap, Eq. 9-10 with swap interval
    w=1: the overlap is merged with a BINARY operator — even time-frames verbatim
    from `a`, odd time-frames verbatim from `b` — instead of an averaging/slerp
    ramp. Every output frame is an exact copy of one input frame, so per-frame
    variance (incl. the high-frequency latent components that interpolation
    cancels) is fully preserved. Deterministic, seed-free.

    HONEST SCOPE NOTE — join-level, NOT the paper's per-step joint diffusion.
    SaFa applies the swap in the overlap of PARALLEL window trajectories at every
    denoising step, so each window's later steps are conditioned on the exchanged
    frames (step-wise differentiated trajectories keep the two sides mutually
    consistent). Our LongFormRenderer generates windows SEQUENTIALLY through
    ChunkGenerator.generate(), which runs the whole sample_diffusion loop as a
    closed call — there is no per-step hook, so we can only exchange frames in
    the fully-denoised latents post-hoc. What carries over: exchange-not-
    interpolate ⇒ no amplitude/HF-variance shrinkage at the seam (the failure
    mode of slerp joins). What is lost: the per-step mutual conditioning that
    lets the model smooth the interleave itself; adjacent frames here come from
    two independent trajectories, relying on the clamped shared prefix (Approach
    A) for their similarity. a, b: (..., n) with equal shapes.
    """
    if a.shape != b.shape:
        raise ValueError(f"swap_join: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    out = a.clone()
    out[..., 1::2] = b[..., 1::2]
    return out


class CrossfadeStitcher:
    """Joins window latents at the seam. join_mode: 'slerp' (default, the original
    ramped spherical crossfade) or 'swap' (SaFa w=1 frame-interleave — see
    swap_join for mechanism + scope caveats)."""

    JOIN_MODES = ("slerp", "swap")

    def __init__(self, blend_frames: int = 3, join_mode: str = "slerp"):
        self.blend_frames = int(blend_frames)
        if join_mode not in self.JOIN_MODES:
            raise ValueError(f"join_mode must be one of {self.JOIN_MODES}, got {join_mode!r}")
        self.join_mode = join_mode

    def _ramp(self, n, device, dtype):
        # ramp t over n frames, shape (1,1,n) for broadcast over (1,C,n)
        return torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).view(1, 1, n)

    def _join(self, a, b):
        # a: outgoing side, b: incoming side; aligned (..., n) overlap slices
        if self.join_mode == "swap":
            return swap_join(a, b)
        t = self._ramp(a.shape[-1], b.device, b.dtype)
        return slerp(a, b, t)

    def continuation_join(self, current_tail, new_region):
        b = min(self.blend_frames, current_tail.shape[-1], new_region.shape[-1])
        if b == 0:
            return new_region
        head = self._join(current_tail[..., -b:], new_region[..., :b])
        return torch.cat([head, new_region[..., b:]], dim=-1)

    def transition_join(self, out_tail, in_head, n):
        n = min(n, out_tail.shape[-1], in_head.shape[-1])
        return self._join(out_tail[..., -n:], in_head[..., :n])


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
                 blend_frames=3, stitcher=None, monitor=None, join_mode="slerp"):
        self.gen = generator
        self.channels = channels
        self.fps = float(fps)
        self.window = int(window_frames)
        self.overlap = int(overlap_frames)
        self.stitcher = stitcher or CrossfadeStitcher(blend_frames=blend_frames,
                                                      join_mode=join_mode)
        self.monitor = monitor or DriftMonitor()
        self.drift_log: list[dict] = []
        if self.overlap >= self.window:
            raise ValueError("overlap_frames must be < window_frames")

    def render_latents(self, schedule, total_frames, base_seed=0, init_latents=None):
        """init_latents: optional (1, C, T0) prefix to CONTINUE from — e.g. a prior
        render's own saved z0 latent — instead of starting from nothing. Only the
        last `overlap` frames condition the next window (InpaintContinuationGenerator's
        prefix_frames), matching the same seam used between any two windows; the full
        init_latents is kept in the returned tensor. total_frames is the FINAL total
        (existing + new), not the amount to add."""
        out = init_latents
        prev_tail = None if init_latents is None else init_latents[..., -self.overlap:]
        k = 0
        while out is None or out.shape[-1] < total_frames:
            t_sec = (out.shape[-1] / self.fps) if out is not None else 0.0
            prompt, is_transition, xf_sec = schedule.resolve(t_sec)
            prefix_frames = 0 if (prev_tail is None or is_transition) else self.overlap
            chunk = self.gen.generate(
                prompt, prefix_latents=prev_tail, prefix_frames=prefix_frames,
                n_frames=self.window, seed=base_seed + k)
            tries = 0
            while not torch.isfinite(chunk).all() and tries < 3:
                tries += 1
                chunk = self.gen.generate(
                    prompt, prefix_latents=prev_tail, prefix_frames=prefix_frames,
                    n_frames=self.window, seed=base_seed + k + 1000 * tries)
            if not torch.isfinite(chunk).all():
                raise RuntimeError(
                    f"LongFormRenderer: chunk {k} (prompt={prompt!r}) non-finite after {tries} retries")
            stats = self.monitor.observe(chunk)
            self.drift_log.append(stats)
            if self.monitor.should_reanchor(stats):
                warnings.warn(
                    f"DriftMonitor canary: RMS collapse at chunk {k} (t={t_sec:.1f}s, "
                    f"rms={stats['rms']:.4f}) — clamp may not be holding (Approach A)",
                    RuntimeWarning, stacklevel=2)
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


class SDEditReanchor:
    """Triangular re-noise->denoise (audio2audio in latent space) to pull back on-manifold.

    Note: Approach A ships slerp-only transitions; SDEditReanchor is available for an opt-in
    transition morph and is the drift-refresh used by Approach C — it is intentionally not
    wired into the default renderer path.
    """

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

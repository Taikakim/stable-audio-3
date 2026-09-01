"""Stereo-preservation auxiliary loss for SA3 DoRA/LoRA training.

WHY.  SA3 DoRA fine-tunes collapse the stereo image toward mono.  The
rectified-flow MSE operates in the 256-dim SAME latent and *under-weights* the
low-energy stereo/side (L-R) component, so a low-rank adapter can trample it at
near-zero RF-loss cost.  This module adds an AUDIO-SPACE term that forces the
gradient to "see" stereo width — the same meter-in-the-gradient logic that fixed
onset control (FusionCC / control_consistency_loss).

MECHANISM (reused from control/sa3_control/cc_probe.py).  Under the RF convention
in training/diffusion.py::training_step::

    noised = clean*(1-t) + noise*t ,  v = noise - clean
    =>  z0_hat = noised - t * v_pred          (rf_z0_hat; exact when v_pred exact,
                                               biased at high t -> hence the t-gate)

We reconstruct z0_hat, decode BOTH z0_hat and the ground-truth latent x1 through
the FROZEN pretransform to stereo audio (grad flows through decode into z0_hat;
the pretransform weights stay frozen -> only z0_hat / the DiT+adapters receive
gradient), take side = (L-R)/2, and match the PREDICTED side to the TARGET side
with an RMS-energy term + a multi-resolution STFT-magnitude term.

Meter-in-the-gradient scope rule (MASTER §4): this helps because RF-loss is blind
to the *distribution* of energy across the L/R sum vs difference — stereo width is
exactly the kind of fine, low-energy property the flat 256-d MSE cannot see.

Cost control: decode is expensive, so callers gate to LOW noise (t < tmax, where
z0_hat is meaningful) and decode only a sub-batch of K rows.  When no row passes
the gate the whole term is skipped (zero decode, zero VRAM).
"""
from __future__ import annotations

import contextlib
import warnings

import torch
import torch.nn.functional as F


def _nested_pretransforms(pretransform, _seen=None):
    """Yield every object in the pretransform chain (following `.model` and
    `.pretransform`) that carries an `enable_grad` control flag.  SAME's
    AudioAutoencoder nests a PatchedPretransform whose decode is wrapped in
    torch.no_grad() when enable_grad is False (autoencoders.py:480) — that DETACHES
    the decode output, so we must flip these on to let the graph reach z0_hat.
    Weights stay frozen (params keep requires_grad=False); only this flag changes.
    """
    if _seen is None:
        _seen = set()
    if pretransform is None or id(pretransform) in _seen:
        return
    _seen.add(id(pretransform))
    if hasattr(pretransform, "enable_grad"):
        yield pretransform
    for attr in ("model", "pretransform"):
        yield from _nested_pretransforms(getattr(pretransform, attr, None), _seen)


@contextlib.contextmanager
def _decode_with_grad(pretransform):
    """Temporarily enable_grad on the whole (frozen) pretransform chain so the decode
    graph is retained back to the latent input; restore the flags on exit."""
    toggled = list(_nested_pretransforms(pretransform))
    prev = [o.enable_grad for o in toggled]
    for o in toggled:
        o.enable_grad = True
    try:
        with torch.enable_grad():
            yield
    finally:
        for o, p in zip(toggled, prev):
            o.enable_grad = p


def rf_z0_hat(noised: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Differentiable clean-latent estimate under rectified flow (see cc_probe.rf_z0_hat).

    noised, v_pred: (B, C, T);  t: (B,) in [0,1] (0 clean, 1 noise).
    z0_hat = noised - t * v_pred.
    """
    tb = t.view(-1, *([1] * (noised.ndim - 1))).to(noised.dtype)
    return noised - tb * v_pred


def _mid_side(audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """audio: (B, 2, S) -> (mid, side), each (B, S).  side = (L-R)/2 (the stereo width)."""
    left, right = audio[:, 0], audio[:, 1]
    return 0.5 * (left + right), 0.5 * (left - right)


def _mr_stft_mag_loss(pred: torch.Tensor, target: torch.Tensor,
                      n_ffts=(512, 1024, 2048), eps: float = 1e-7) -> torch.Tensor:
    """Multi-resolution STFT-magnitude loss on 1-D (B, S) signals.

    L1 on linear magnitude + L1 on log magnitude, averaged over resolutions.
    STFT is computed in float32 (half is unsupported / unstable on ROCm).
    """
    pred = pred.float()
    target = target.float()
    total = pred.new_zeros(())
    for n_fft in n_ffts:
        if pred.shape[-1] < n_fft:
            continue
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=pred.device, dtype=torch.float32)
        pm = torch.stft(pred, n_fft, hop, window=win, return_complex=True).abs()
        tm = torch.stft(target, n_fft, hop, window=win, return_complex=True).abs()
        lin = F.l1_loss(pm, tm)
        log = F.l1_loss(torch.log(pm + eps), torch.log(tm + eps))
        total = total + lin + log
    return total / max(1, len(n_ffts))


def stereo_side_loss(pred_audio: torch.Tensor, target_audio: torch.Tensor,
                     n_ffts=(512, 1024, 2048), eps: float = 1e-7) -> torch.Tensor:
    """Match the PREDICTED side channel to the TARGET side channel.

    pred_audio, target_audio: (B, 2, S) stereo waveforms (any float dtype; cast
    to fp32 internally).  Returns a scalar = RMS-energy term + MR-STFT term on the
    side (L-R)/2 signal.  Penalizes destroying / collapsing stereo width.
    """
    _, pred_side = _mid_side(pred_audio.float())
    _, tgt_side = _mid_side(target_audio.float())

    # Energy / RMS term: match the loudness of the side channel (collapsing to
    # mono drives pred_side -> 0 while target side RMS stays > 0 -> large penalty).
    pred_rms = pred_side.pow(2).mean(dim=-1).clamp_min(eps).sqrt()
    tgt_rms = tgt_side.pow(2).mean(dim=-1).clamp_min(eps).sqrt()
    rms_loss = F.l1_loss(pred_rms, tgt_rms)

    spec_loss = _mr_stft_mag_loss(pred_side, tgt_side, n_ffts=n_ffts, eps=eps)
    return rms_loss + spec_loss


def compute_stereo_loss(pretransform, z0_hat: torch.Tensor, x1: torch.Tensor,
                        t: torch.Tensor, t_max: float = 0.3, subbatch: int = 2,
                        n_ffts=(512, 1024, 2048)) -> torch.Tensor:
    """Gated, sub-batched stereo-side loss.  Returns a scalar tensor (0 if gated out).

    pretransform : frozen AutoencoderPretransform (weights requires_grad=False).
                   decode() expects the SCALED latent (it multiplies by .scale
                   internally) -> pass z0_hat / x1 directly, same space they live
                   in inside training_step.
    z0_hat       : (B, C, T) differentiable clean-latent estimate (grad target).
    x1           : (B, C, T) ground-truth clean latent (no grad needed).
    t            : (B,) timesteps; only rows with t < t_max are decoded.
    subbatch     : max rows to decode (VRAM cap).

    Grad flows through decode INTO z0_hat only; the pretransform is frozen.
    Non-finite results are skipped (return 0 + warn) so a bad decode never NaNs
    the whole training step.
    """
    gate = t < t_max
    if not bool(gate.any()):
        return z0_hat.new_zeros(())
    idx = gate.nonzero(as_tuple=False).squeeze(-1)
    if subbatch and idx.numel() > subbatch:
        idx = idx[:subbatch]

    z_sel = z0_hat[idx]
    x_sel = x1[idx].detach().to(z_sel.dtype)  # target: no grad into ground truth

    # Decode z0_hat under grad. Frozen weights carry no grad; only z0_hat does.
    # _decode_with_grad flips the nested pretransform enable_grad flags so SAME's
    # inner no_grad wrapper (autoencoders.py:480) doesn't detach the output.
    with _decode_with_grad(pretransform):
        pred_audio = pretransform.decode(z_sel)
    with torch.no_grad():
        target_audio = pretransform.decode(x_sel)

    loss = stereo_side_loss(pred_audio, target_audio, n_ffts=n_ffts)

    if not torch.isfinite(loss):
        warnings.warn("stereo_loss: non-finite value, skipping this step's term", RuntimeWarning)
        return z0_hat.new_zeros(())
    return loss

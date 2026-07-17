"""recurrence_potential.py -- anti-loop energy potential as a latch_guided "head"
(longform validation plan 2026-07-15, E1; the FK-Flow/TFG tilt's differentiable form).

Wraps the validated whitened-patch recurrence statistic (mir
src/tools/recurrence_meter.py, v3) as a fixed nn.Module so it drops into
``sample_flow_euler_multi_latch_guided``'s guide list:

    head    = RecurrenceHead()                     # no parameters, fp32
    target  = torch.full((1, 1, 1), band_upper)    # corpus q90 (eval/corpus_bands.json)
    guide   = dict(head=head, target=target, weight=lam, loss_type="band_hinge",
                   start_pct=0.3, end_pct=0.8, huber_beta=0.05)   # huber_beta = hinge width

Design constraints baked in (do not undo silently):
  * STRIDE = 1 FRAME. The CPU meter's 1 s patch stride is provably fragile to
    loop periods incommensurate with the stride grid (E0 smoke 2026-07-15:
    period on-grid r_max=1.000, 0.4 s off-grid 0.404). Frame stride removes
    the grid entirely; the GPU absorbs the cost.
  * Soft max over the lookback band via temperature logsumexp -- C's
    gradient-informativeness fix (a hard max backprops through one pair;
    logsumexp spreads the gradient over every near-recurrence).
  * Per-channel whitening over the window before patching (the v3 correction:
    constant-energy channels must not drive similarity).
  * BAND-HINGE loss, not point matching (lens-B): zero gradient inside the
    corpus band -- the tilt only pushes when recurrence exceeds the band edge.

Cost at T=4096: P ~ 4054 patches, one (P,D)x(D,P) matmul (D=43*256) + masked
logsumexp ~ 0.4 GFLOP-scale per evaluation; ~40 evaluations per guided render
at the default selective window. Peak transient memory ~ 0.5 GB fp32.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class RecurrenceHead(torch.nn.Module):
    """(B, C, T) latent -> (B, 1, P) soft recurrence curve in [0, 1]-ish.

    rec[i] = tau * logsumexp(cos(patch_i, patch_j) / tau) over the strictly
    preceding lookback band j in [i - lb_max, i - lb_min] (frame units).
    Patches before the lookback horizon emit rec = 0 (never above any band
    edge -> no gradient, matching the CPU meter's NaN-skip convention).
    """

    def __init__(self, fps: float = 10.7666, patch_sec: float = 4.0,
                 lookback_min_sec: float = 8.0, lookback_max_sec: float = 40.0,
                 temp: float = 0.02, whiten: bool = True, chunk: int = 1024):
        super().__init__()
        self.pf = max(2, int(round(patch_sec * fps)))
        self.lb_min = max(self.pf, int(round(lookback_min_sec * fps)))
        self.lb_max = int(round(lookback_max_sec * fps))
        self.temp = float(temp)
        self.whiten = bool(whiten)
        self.chunk = int(chunk)

    def forward(self, z: torch.Tensor, t=None) -> torch.Tensor:
        # t accepted for sampler-API compatibility (heads are called head(x, t));
        # the statistic is time-embedding-free by design.
        if z.dim() != 3:
            raise ValueError(f"RecurrenceHead expects (B, C, T), got {tuple(z.shape)}")
        B, C, T = z.shape
        x = z.float()
        if self.whiten:
            mu = x.mean(dim=2, keepdim=True)
            sd = x.std(dim=2, keepdim=True).clamp_min(1e-6)
            x = (x - mu) / sd
        # patches: unfold time -> (B, P, C*pf), unit-normalized
        P = T - self.pf + 1
        if P < self.lb_min + 1:
            return z.new_zeros(B, 1, max(P, 1))
        u = x.unfold(2, self.pf, 1)                    # (B, C, P, pf)
        u = u.permute(0, 2, 1, 3).reshape(B, P, C * self.pf)
        u = u / u.norm(dim=2, keepdim=True).clamp_min(1e-8)

        rec = z.new_zeros(B, P)
        tau = self.temp
        # banded soft-max over strictly-preceding patches, row-chunked
        for s in range(self.lb_min, P, self.chunk):
            e = min(s + self.chunk, P)
            lo = max(0, s - self.lb_max)
            sim = torch.bmm(u[:, s:e], u[:, lo:e - self.lb_min].transpose(1, 2))  # (B, rows, cols)
            rows = torch.arange(s, e, device=z.device).unsqueeze(1)
            cols = torch.arange(lo, e - self.lb_min, device=z.device).unsqueeze(0)
            valid = (cols <= rows - self.lb_min) & (cols >= rows - self.lb_max)
            sim = sim.masked_fill(~valid.unsqueeze(0), -1e4)
            rec[:, s:e] = tau * torch.logsumexp(sim / tau, dim=2)
        return rec.clamp(max=1.5).unsqueeze(1)          # (B, 1, P)


def band_hinge_loss(pred: torch.Tensor, target: torch.Tensor,
                    width: float = 0.05) -> torch.Tensor:
    """Smooth one-sided hinge above the band edge; zero inside the band.

    pred (B, 1, P) recurrence curve; target broadcastable band UPPER edge
    (e.g. corpus r_max q90). width = softplus sharpness in recurrence units.
    """
    w = max(float(width), 1e-4)
    return (F.softplus((pred - target) / w) * w).mean()

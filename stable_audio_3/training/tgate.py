"""tgate.py — a noise-level gate for the subspace-weighted RF loss (task #59), derived from a MEASURED
R²(t) curve of how recoverable the melody subspace is at each noise level.

WHY (C, 2026-08-19). The #59 term upweights the flow-matching error inside a latent subspace (the
15-d melody-selective basis) by K at every t. But melody is not equally learnable at every noise
level: our own measurements say melody/harmony are degraded-and-abandoned at mid/high noise while
rhythm survives, and the E1 pre-test showed the v-trained base under-recovers the low-variance
directions 8.3x at sigma .2 vs 2.2x at .8. A flat K therefore spends most of its melody budget
where melody is not representable. Lemma A.2 of arXiv 2602.19512 says the isotropic special case
of a learned matrix schedule IS a per-noise-level loss weighting — their best FFHQ result was
exactly that; we hand-set ours from a measured curve instead of learning it.

USE. `eval/melody_r2_vs_t.py` writes {"t": [...], "r2_melody": [...], "r2_rest": [...],
"deficit": [...]} for a model. Here we pick one curve (--subspace-loss-tgate-mode):
    r2       gate ∝ R²_melody(t)          push where melody is recoverable            (default)
    r2sq     gate ∝ R²_melody(t)²         same, sharper
    deficit  gate ∝ (err_mel/var_mel)/(err_rest/var_rest)  push where melody is UNDER-recovered
                                          relative to the rest of the latent (the E1 quantity)
floor it (never zero — a zeroed t would remove the subspace term there entirely), and normalise
to MEAN 1 over the file's t grid so `--subspace-loss-weight K` keeps its meaning as the average
multiplier. At train time the gate is linearly interpolated at each sample's t (clamped at the
grid ends) and multiplies the per-sample subspace error energy.
"""
import json

import torch


def interp_gate(t: torch.Tensor, grid_t: torch.Tensor, grid_g: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of grid_g at t, clamped to the grid ends. t: any shape."""
    gt = grid_t.to(t.device, t.dtype)
    gg = grid_g.to(t.device, t.dtype)
    tc = t.clamp(gt[0], gt[-1])
    idx = torch.searchsorted(gt, tc, right=True).clamp(1, gt.numel() - 1)
    t0, t1 = gt[idx - 1], gt[idx]
    g0, g1 = gg[idx - 1], gg[idx]
    w = (tc - t0) / (t1 - t0).clamp_min(1e-12)
    return g0 + w * (g1 - g0)


def normalize_gate(g: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    g = g.clamp_min(float(floor))
    m = g.mean()
    return g / m if float(m) > 0 else torch.ones_like(g)


def load_tgate(path: str, mode: str = "r2", floor: float = 0.05):
    d = json.load(open(path))
    t = torch.tensor(d["t"], dtype=torch.float32)
    if mode == "r2":
        g = torch.tensor(d["r2_melody"], dtype=torch.float32)
    elif mode == "r2sq":
        g = torch.tensor(d["r2_melody"], dtype=torch.float32).clamp_min(0) ** 2
    elif mode == "deficit":
        g = torch.tensor(d["deficit"], dtype=torch.float32)
    else:
        raise ValueError(f"tgate mode must be r2|r2sq|deficit, got {mode!r}")
    if t.numel() < 2 or t.numel() != g.numel():
        raise ValueError(f"tgate file {path}: need matching t/gate arrays of length >= 2")
    order = torch.argsort(t)
    return t[order], normalize_gate(g[order], floor)

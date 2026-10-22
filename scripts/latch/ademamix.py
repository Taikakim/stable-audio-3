# Owner: WINTERMUTE (created 2026-10-22)
"""AdEMAMix (Pagliardini, Ablin, Grangier 2024, arXiv 2409.03137) for train_latch.py --optimizer ademamix.

Adam with a second, slow first moment mixed into the numerator:

    m1 <- b1 m1 + (1-b1) g          (fast, bias-corrected, as Adam)
    m2 <- b3 m2 + (1-b3) g          (slow, NOT bias-corrected)
    v  <- b2 v  + (1-b2) g^2
    p  <- p - lr * ( (m1/bc1 + alpha m2) / (sqrt(v/bc2) + eps) + wd p )

The slow average keeps gradients from thousands of steps back in play; the paper's point is that one EMA cannot weigh
both recent and old gradients well. alpha ramps linearly 0 -> alpha and b3 ramps from b1 to b3 on a half-life-linear
schedule over `warmup` steps (paper: the whole run), because a large slow term at the start destabilises training.
Follows Apple's reference (github.com/apple/ml-ademamix, pytorch/ademamix.py) and NVIDIA Emerging-Optimizers'
calculate_ademamix_update; written as a plain optimizer here because Emerging-Optimizers registers only the simplified
variant and wraps the full one in torch.compile. Decoupled weight decay as torch AdamW (p *= 1 - lr*wd before the step).

Horizon: b3 = 0.9999 (paper, LM pretraining) averages ~10k steps; a 40-epoch LatCH run at batch 64 is ~7.4k steps,
so b3 ~ 0.999 (~1k steps, ~5 epochs) is the scaled choice there.
"""
from __future__ import annotations

import math

import torch


def _beta3_at(step: int, b_start: float, b_end: float, warmup: int) -> float:
    """Half-life-linear ramp: the EMA's half-life (in steps) grows linearly from b_start's to b_end's."""
    if warmup <= 0 or step >= warmup:
        return b_end

    def half_life(b):
        return math.log(0.5) / math.log(b + 1e-8) - 1.0

    a = step / float(warmup)
    t = (1.0 - a) * half_life(b_start) + a * half_life(b_end)
    return math.pow(0.5, 1.0 / (t + 1.0))


class AdEMAMix(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.999), alpha=5.0, warmup=0, eps=1e-8, weight_decay=0.01):
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0 or not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"invalid betas {betas}")
        super().__init__(params, dict(lr=lr, betas=tuple(betas), alpha=alpha, warmup=int(warmup), eps=eps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2, b3), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m1"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["m2"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["step"] += 1
                t = st["step"]
                w = group["warmup"]
                alpha = group["alpha"] * min(1.0, t / w) if w > 0 else group["alpha"]
                b3_t = _beta3_at(t, b1, b3, w)
                if wd:
                    p.mul_(1.0 - lr * wd)
                st["m1"].lerp_(g, 1.0 - b1)
                st["m2"].lerp_(g, 1.0 - b3_t)
                st["v"].lerp_(g * g, 1.0 - b2)
                denom = (st["v"] / (1.0 - b2 ** t)).sqrt_().add_(eps)
                num = st["m1"] / (1.0 - b1 ** t) + alpha * st["m2"]
                p.addcdiv_(num, denom, value=-lr)
        return loss

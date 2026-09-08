"""DAdaptScaler — the D-Adaptation step-size estimator, factored out so a non-Fusion
optimizer can use it (CONTINUITY 2026-09-09, Kim's ask: "can we add D-Adaptation like
features for Lion too?").

This is the same mechanism FusionOpt runs as its "autoscale" component (fusion_opt.py
_update_autoscale, Kim 2026-09-01): `d` starts at d0 and only ever grows, driven by how
much the observed gradient correlates with the DISPLACEMENT FROM INIT, dot(g, p0 - p),
relative to an EMA of gradient magnitude (`s`). It returns d/d0 — a pure multiplier that
starts at 1.0 and rises as evidence accumulates — and never writes to p.data itself, so
it composes with any update rule.

MEMORY: like Fusion's copy (and Prodigy's own slice_p knob), `p0` and `s` are kept at
every slice_p-th coordinate only, and they ride in the HOST OPTIMIZER'S per-param state
dict rather than a second parallel one. Cost is O(numel / slice_p) in both state and
compute.

⚠ WHAT DOES NOT TRANSFER TO LION. D-Adaptation's guarantee is an online-convex-
optimization bound that assumes the update magnitude tracks the gradient — true of
AdamW/Muon-family steps, NOT of Lion, whose step is a unit sign vector whose magnitude
is exactly lr regardless of gradient scale. The `d` estimate still measures something
real on Lion (displacement from init against accumulated gradient mass), and it still
adapts an lr that would otherwise be a blind guess, but it is a HEURISTIC there, not the
bounded quantity the paper describes. Do not report a Lion+autoscale run as "Prodigy for
Lion".
"""

import math

import torch


class DAdaptScaler:
    """Global D-Adaptation multiplier over a set of param groups.

    Usage (once per optimizer step, BEFORE applying updates, while .grad is populated):

        mult = scaler.update(self.param_groups, lambda p: self.state[p])
        effective_lr = group["lr"] * mult
    """

    def __init__(self, d0: float = 1e-6, coef: float = 1.0,
                 growth_rate: float = float("inf"), slice_p: int = 16,
                 beta3: "float | None" = None):
        self.d0 = float(d0)
        self.coef = float(coef)
        self.growth_rate = float(growth_rate)
        self.slice_p = max(1, int(slice_p))
        self.beta3 = float(beta3) if beta3 is not None else math.sqrt(0.999)
        self.d = self.d0
        self.d_max = self.d0
        self.numerator = 0.0

    @torch.no_grad()
    def update(self, param_groups, state_fn) -> float:
        d, d0, beta3, slice_p = self.d, self.d0, self.beta3, self.slice_p
        factor = (d / d0) * d          # == (d/d0)*dlr with our own "lr" fixed at 1.0

        d_numerator = self.numerator * beta3
        delta_numerator = 0.0
        d_denom = 0.0
        any_grad = False

        for group in param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                any_grad = True
                state = state_fn(p)
                if "auto_p0" not in state:
                    state["auto_p0"] = p.detach().flatten()[::slice_p].clone().float()
                    state["auto_s"] = torch.zeros_like(state["auto_p0"])
                p0 = state["auto_p0"]
                s = state["auto_s"]

                g_sliced = p.grad.detach().flatten()[::slice_p].float()
                p_sliced = p.detach().flatten()[::slice_p].float()

                delta_numerator += float(factor * torch.dot(g_sliced, p0 - p_sliced))
                s.mul_(beta3).add_(g_sliced, alpha=factor)
                d_denom += float(s.abs().sum())

        if not any_grad or d_denom == 0.0:
            return d / d0

        d_numerator += delta_numerator
        d_hat = self.coef * d_numerator / d_denom
        if d == d0:
            d = max(d, d_hat)
        d_max = max(self.d_max, d_hat)
        d = min(d_max, d * self.growth_rate)

        self.numerator = d_numerator
        self.d = d
        self.d_max = d_max
        return d / d0

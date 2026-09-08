"""Lion optimizer with the same stochastic-rounding bf16 trick as AdamWSR.

Lion (Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms") keeps ONE
momentum buffer per param (vs AdamW's two: exp_avg + exp_avg_sq), so its state is
already half the size of AdamW's at equal precision. LionSR additionally stores that
buffer (and the master weight) in bf16 with stochastic-rounded writeback -- same
rationale as AdamWSR (stochastic_rounding.py): a sub-ULP update on a bf16 master is
silently truncated under deterministic round-to-nearest; SR makes it accumulate in
expectation instead. Net: ~1/4 the optimizer-state memory of a naive fp32 AdamW,
~1/2 of AdamWSR.

Update rule (decoupled weight decay, matching torch.optim.AdamW's convention):
    interp = beta1 * m + (1 - beta1) * g
    theta  = theta - lr * (sign(interp) + wd * theta)
    m      = beta2 * m + (1 - beta2) * g

Note Lion's update is a unit-sign step scaled only by lr, so it wants a smaller lr
and larger weight_decay than AdamW (paper: lr ~3-10x smaller, wd ~3-10x larger).

Self-contained, dependency-free (mirrors stochastic_rounding.py). NOT for LUMI
(FusionOpt-SF keeps an fp32 master there); this is the 16 GB desktop path.
"""

import torch

from .stochastic_rounding import stochastic_round_to_bf16


class LionSR(torch.optim.Optimizer):
    """Lion with a bf16 master weight + bf16 momentum buffer, fp32 update math,
    stochastic-rounded writeback. ONE state tensor per param (vs AdamWSR's two).
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0,
                 autoscale=False, autoscale_d0=1e-6, autoscale_coef=1.0,
                 autoscale_growth_rate=float("inf"), autoscale_slice_p=16):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        # D-Adaptation step-size estimator (dadapt_scale.DAdaptScaler) — the same
        # mechanism FusionOpt runs as its "autoscale" component. OFF by default and
        # byte-identical to plain LionSR when off. Read the module docstring before
        # trusting the number: the OCO bound does NOT transfer to Lion's unit-sign
        # step, so this is an adaptive-lr heuristic here, not Prodigy's guarantee.
        self._auto = None
        if autoscale:
            from .dadapt_scale import DAdaptScaler
            self._auto = DAdaptScaler(d0=autoscale_d0, coef=autoscale_coef,
                                      growth_rate=autoscale_growth_rate,
                                      slice_p=autoscale_slice_p)
        self._auto_mult = 1.0

    def autoscale_multiplier(self) -> float:
        """Current d/d0. 1.0 when autoscale is off, or before the first step."""
        return self._auto_mult

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # one global multiplier per step, computed while every .grad is still live
        if self._auto is not None:
            self._auto_mult = self._auto.update(self.param_groups, lambda q: self.state[q])

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"] * self._auto_mult
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LionSR does not support sparse gradients")

                state = self.state[p]
                if "exp_avg" not in state:      # not `len(state) == 0`: the autoscale
                    # scaler rides in this same dict and may have populated it already
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16)

                p_f = p.detach().float()
                g = grad.float()
                m = state["exp_avg"].float()

                interp = m * beta1 + g * (1.0 - beta1)
                update = interp.sign()
                p_f = p_f - lr * (update + wd * p_f)

                m.mul_(beta2).add_(g, alpha=1.0 - beta2)

                p.copy_(stochastic_round_to_bf16(p_f).to(p.dtype))
                state["exp_avg"].copy_(stochastic_round_to_bf16(m))

        return loss

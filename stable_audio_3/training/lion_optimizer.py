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

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LionSR does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
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

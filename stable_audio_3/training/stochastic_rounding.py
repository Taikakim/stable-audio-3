"""Stochastic-Rounding bf16 optimizer path — desktop-GPU full-finetune enabler.

Self-contained, dependency-free (mirrors how `SimpleEMA` was added to this fork:
no new pip deps, close to upstream, additive only). NOT used on LUMI — there we run
FusionOpt-SF, which keeps an fp32 master. This path is for **desktop GPUs (16 GB)**
where keeping an fp32 master of the 1.4 B DiT + fp32 Adam states does not fit.

The idea
--------
A bf16 weight `w` has only 7 mantissa bits. A gradient update `w <- w + delta` where
`delta` is smaller than half a bf16 ULP is *deterministically truncated back to w*:
the update vanishes. Over training this silently kills every sub-dominant signal (e.g.
melody, whose per-step contribution is small relative to the dominant beat/energy) —
the weights simply never move for those directions.

Stochastic rounding fixes this: instead of always truncating, round `w + delta` UP to
the next bf16 grid point with probability equal to the fractional position between the
two bracketing grid points. The *expected* stored value equals the true fp32 value, so
tiny updates ACCUMULATE in expectation across steps instead of being thrown away.

`AdamWSR` keeps params AND Adam states (exp_avg, exp_avg_sq) in bf16, upcasts to fp32
for the Adam math, and writes the new param back through `stochastic_round_to_bf16`
instead of a deterministic cast.
"""

import torch


@torch.no_grad()
def stochastic_round_to_bf16(x_fp32: torch.Tensor) -> torch.Tensor:
    """Stochastically round an fp32 tensor to bf16 via integer bit dithering.

    fp32 bit layout is [sign | 8 exp | 23 mantissa]; bf16 is the high 16 bits
    [sign | 8 exp | 7 mantissa]. Truncation drops the low 16 bits (the extra 16
    mantissa bits). We instead add a uniform random 16-bit dither to the *unsigned*
    32-bit pattern and then mask off the low 16 bits: with probability equal to the
    fractional position between the two bracketing bf16 grid points, the dither
    carries into the high half and rounds UP; otherwise it rounds down. The carry
    propagates correctly through mantissa->exponent, and because the fp32 pattern is
    magnitude-monotonic for a fixed sign, this is unbiased for both signs.

    E[stochastic_round_to_bf16(w)] == w (to within the bf16 grid), so sub-ULP updates
    accumulate in expectation instead of being deterministically truncated to zero.

    NaN/Inf pass through unchanged (cast to bf16).
    """
    x = x_fp32.float() if x_fp32.dtype != torch.float32 else x_fp32
    finite = torch.isfinite(x)

    # reinterpret fp32 bits as int32, promote to unsigned 32-bit held in int64 so the
    # dither add cannot overflow / wrap the sign bit.
    xi = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    dither = torch.randint(0, 1 << 16, x.shape, device=x.device, dtype=torch.int64)
    rounded = (xi + dither) & 0xFFFF0000  # add dither, then truncate low 16 bits

    # map unsigned-32 pattern back to signed two's-complement int32, then bit-cast to fp32
    rounded = torch.where(rounded >= (1 << 31), rounded - (1 << 32), rounded)
    out_f = rounded.to(torch.int32).view(torch.float32)

    # NaN/Inf (and any element the bit path shouldn't touch) pass through verbatim
    out_f = torch.where(finite, out_f, x)
    return out_f.to(torch.bfloat16)


class AdamWSR(torch.optim.Optimizer):
    """AdamW with bf16 master weights + bf16 states + stochastic-rounded param updates.

    Desktop-GPU full-finetune enabler: params and the Adam moments (`exp_avg`,
    `exp_avg_sq`) are stored in bf16 (half the memory of an fp32 master + fp32 states).
    The Adam update math is done in fp32 (states + grad upcast), then the new param is
    written back to the bf16 master through :func:`stochastic_round_to_bf16` instead of
    a deterministic cast, so sub-ULP updates accumulate. Dep-free; decoupled weight
    decay (AdamW), matching torch.optim.AdamW defaults.

    NOT for LUMI (there FusionOpt-SF keeps an fp32 master). This is the 16 GB path.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamWSR does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # bf16 states — half the memory of fp32 Adam moments
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.bfloat16)

                state["step"] += 1
                t = state["step"]

                # upcast everything to fp32 for the actual Adam math
                p_f = p.detach().float()
                g = grad.float()
                exp_avg = state["exp_avg"].float()
                exp_avg_sq = state["exp_avg_sq"].float()

                # decoupled weight decay (AdamW)
                if wd != 0.0:
                    p_f = p_f * (1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                bias_c1 = 1.0 - beta1 ** t
                bias_c2 = 1.0 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / (bias_c2 ** 0.5)).add_(eps)
                step_size = lr / bias_c1

                p_f = p_f - step_size * (exp_avg / denom)

                # write params back to the bf16 master via stochastic rounding, and
                # store the updated moments back down to bf16 (stochastically, so the
                # moment accumulation doesn't stall either).
                p.copy_(stochastic_round_to_bf16(p_f).to(p.dtype))
                state["exp_avg"].copy_(stochastic_round_to_bf16(exp_avg))
                state["exp_avg_sq"].copy_(stochastic_round_to_bf16(exp_avg_sq))

        return loss

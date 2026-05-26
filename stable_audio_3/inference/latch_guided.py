# stable_audio_3/inference/latch_guided.py
"""Gradient-enabled Euler sampler with LatCH Training-Free Guidance for SA3 (Phase 1)."""

import torch
from tqdm import tqdm


def _st_weights(sigmas_1d):
    """Per-step s_t = alpha / sum(alpha), alpha = 1 - t. sigmas_1d: (steps+1,)."""
    alpha = (1.0 - sigmas_1d[:-1]).clamp(min=0.0)
    denom = alpha.sum().clamp(min=1e-8)
    return alpha / denom


def sample_flow_euler_latch_guided(
    model, x, sigmas, *, head, target,
    rho=1.0, mu=1.0, gamma=0.3, n_iter=4,
    window=(0.5, 1.0), loss_type="mse",
    disable_tqdm=False, **model_kwargs,
):
    """Euler sampling with selective TFG from a LatCH head.

    rho: variance-guidance strength on z_t (head queried at true t_curr).
    mu:  mean-guidance strength on the clean estimate x_hat0 (head queried at t=0).
    gamma: input-noise augmentation std for the clean head evaluation.
    window: (sigma_lo, sigma_hi) -- guidance active when sigma_lo <= t_curr <= sigma_hi.
    Note: when gamma > 0 the mean-guidance evaluation is stochastic, so output is
    not bit-reproducible across calls even with a fixed seed; set gamma=0 for determinism.
    """
    per_element = sigmas.dim() == 2
    sigmas = sigmas.to(x.device)
    num_steps = sigmas.shape[-1] - 1
    target = target.to(x.device)

    sigmas_1d = sigmas[0] if per_element else sigmas
    st = _st_weights(sigmas_1d).to(x.device)
    lo, hi = window

    def head_loss(pred, tgt):
        if loss_type == "bce_logits":
            return torch.nn.functional.binary_cross_entropy_with_logits(pred, tgt)
        if loss_type == "mse":
            return torch.nn.functional.mse_loss(pred, tgt)
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    for i in tqdm(range(num_steps), disable=disable_tqdm):
        if per_element:
            t_curr = sigmas[:, i].to(x.dtype)
            dt = (sigmas[:, i + 1] - sigmas[:, i]).view(-1, 1, 1)
        else:
            t_curr = sigmas[i].to(x.dtype) * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            dt = (sigmas[i + 1] - sigmas[i])

        t_scalar = float(t_curr.flatten()[0])
        active = (lo <= t_scalar <= hi)
        rho_t = rho * float(st[i])
        mu_t = mu * float(st[i])

        # --- Variance guidance on z_t (head queried at true t_curr) ---
        if active and rho_t > 0:
            x = x.detach().requires_grad_(True)
            pred = head(x, t_curr)
            loss = head_loss(pred, target)
            grad = torch.autograd.grad(loss, x)[0]
            x = (x - rho_t * grad).detach()

        # --- Model velocity (no grad needed through the DiT for mean guidance) ---
        with torch.no_grad():
            v = model(x, t_curr, **model_kwargs)
        x_hat0 = x - t_curr.view(-1, 1, 1) * v

        # --- Mean guidance on the clean estimate (head queried at t=0) ---
        if active and mu_t > 0:
            z0 = x_hat0.detach()
            t0 = torch.zeros_like(t_curr)
            for _ in range(n_iter):
                z0 = z0.detach().requires_grad_(True)
                aug = z0 + gamma * torch.randn_like(z0)
                loss = head_loss(head(aug, t0), target)
                grad = torch.autograd.grad(loss, z0)[0]
                z0 = (z0 - mu_t * grad).detach()
            x_hat0 = z0

        # --- Euler update reconstructed from the (possibly guided) clean estimate ---
        # v_eff = (x - x_hat0) / t_curr ; x_next = x + dt * v_eff
        # clamp guards variation runs where t_curr can approach 0 in the final steps
        t_b = t_curr.view(-1, 1, 1).clamp(min=1e-6)
        v_eff = (x - x_hat0) / t_b
        x = (x + dt * v_eff).detach()

    return x

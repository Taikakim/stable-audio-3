# stable_audio_3/inference/latch_guided.py
"""Gradient-enabled Euler sampler with LatCH Training-Free Guidance for SA3 (Phase 1)."""

import torch
from tqdm import tqdm


def _ensure_time_cache(head, t_values, device) -> None:
    """Attach a TimeConditioningCache to an adaln_zero head (no-op otherwise).

    Per LATCH_RESULTS §17 / docs/FUSION_SHAREABLE.md: for an adaln_zero LatCH,
    t_emb and the per-block (γ,β,α) modulators are pure functions of t and
    weights → cacheable. The sampler queries a small fixed set of t values
    (per-step σ plus t=0 for mean guidance), so warming once removes ~5-10%
    inference latency at 40 steps. Falls back silently to live calc for
    non-adaln_zero heads or if SAT isn't importable.

    The cache attaches to the head and persists for the head's lifetime, so
    subsequent renders with the same head + schedule hit 100%.
    """
    if getattr(head, "t_injection", "concat") != "adaln_zero":
        return
    cache = getattr(head, "_time_cache", None)
    if cache is None:
        try:
            from stable_audio_tools.inference.time_cache import TimeConditioningCache
        except ImportError:
            return
        cache = TimeConditioningCache(head, device=device)
        head.attach_time_cache(cache)
    cache.warm(t_values)


def _make_latch_criterion(loss_type, huber_beta=1.0):
    """Guidance loss matching how the head was trained (see scripts/latch/train_latch.py)."""
    if loss_type == "bce_logits":
        return torch.nn.BCEWithLogitsLoss()
    if loss_type == "cosine":
        def _cos(pred, target):
            p = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
            t = target / (target.norm(dim=1, keepdim=True) + 1e-8)
            return (1.0 - (p * t).sum(dim=1)).mean()
        return _cos
    if loss_type == "smooth_l1":
        return torch.nn.SmoothL1Loss(beta=float(huber_beta or 1.0))
    if loss_type == "mse":
        return torch.nn.MSELoss()
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


@torch.enable_grad()
def sample_flow_euler_multi_latch_guided(
    model, x, sigmas, guides, *,
    rho=1.0, mu=1.0, gamma=0.3, n_iter=4,
    log_norms=False, disable_tqdm=False, **model_kwargs,
):
    """Flow-matching Euler sampler with multiple LatCH guides (Selective TFG).

    Mirrors the single-guide ``sample_flow_euler_latch_guided`` but applies a
    weighted sum of any number of heads, each with its own target and step-%
    window. This is the path the Gradio UI and ``StableAudioModel.generate()``
    drive; the single-guide function is kept for the verify script.

    guides: list of dicts, each with keys
        head       -- loaded LatCH nn.Module (fp32)
        target     -- target tensor [B, out_channels, T]
        weight     -- per-guide gradient multiplier
        start_pct  -- fraction of steps where this guide activates (0 = first)
        end_pct    -- fraction of steps where it deactivates
        loss_type  -- "mse" | "bce_logits" | "smooth_l1" | "cosine"
        huber_beta -- beta for smooth_l1 (optional)

    rho/mu: global variance/mean guidance strengths (scaled per-step by s_t).
    gamma:  input-noise augmentation std for the clean head evaluation.
    All guidance math runs in fp32 (heads are fp32); x is cast to the diffusion
    model's dtype only for the velocity forward, so this works whether the model
    is fp16 or fp32.
    """
    model_dtype = x.dtype  # the diffusion model's input dtype (e.g. fp16 in the GUI)
    x = x.detach().float()
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    num_steps = sigmas.shape[-1] - 1
    b = x.shape[0]

    alpha = (1.0 - sigmas[:-1]).clamp(min=0.0)
    sum_alphas = alpha.sum().clamp(min=1e-8)

    for g in guides:
        g["_start"] = int(num_steps * g["start_pct"])
        g["_end"] = int(num_steps * g["end_pct"])
        g["_criterion"] = _make_latch_criterion(g.get("loss_type", "mse"), g.get("huber_beta", 1.0))
        g["target"] = g["target"].to(device=x.device, dtype=torch.float32)

    # Pre-warm per-guide adaLN-zero time caches with the full schedule + t=0.
    # Heads that aren't adaln_zero are skipped silently (live calc).
    _t_values = sigmas[:-1].tolist() + [0.0]
    for g in guides:
        _ensure_time_cache(g["head"], _t_values, x.device)

    log_rows = [] if log_norms else None

    for i in tqdm(range(num_steps), disable=disable_tqdm):
        t_curr = sigmas[i]
        t_prev = sigmas[i + 1]
        t_b = t_curr * torch.ones(b, device=x.device, dtype=torch.float32)
        s_t = float(alpha[i]) / float(sum_alphas)
        rho_t = rho * s_t
        mu_t = mu * s_t

        active = [g for g in guides if g["_start"] <= i < g["_end"]]

        # --- Variance guidance on x at the true t_curr (head queried at t_curr) ---
        # Done before the model velocity and re-detached, mirroring the proven
        # single-guide sampler: SA3's DiT forward runs under inference_mode, so
        # the variance grad must be taken on a clean leaf before that call.
        if active and rho_t > 0:
            x = x.detach().requires_grad_(True)
            t_ten = torch.full((b,), float(t_curr), device=x.device)
            loss_var = sum(
                g["weight"] * g["_criterion"](g["head"](x, t_ten), g["target"])
                for g in active
            )
            grad_var = torch.autograd.grad(loss_var, x)[0]
            if log_rows is not None:
                log_rows.append({
                    "i": i, "sigma": float(t_curr),
                    "gv_norm": grad_var.detach().norm().item(),
                    "x_norm": x.detach().norm().item(), "rho_t": rho_t,
                })
            x = (x - rho_t * grad_var).detach()

        with torch.no_grad():
            v = model(x.to(model_dtype), t_b.to(model_dtype), **model_kwargs).float()
        z0 = x - t_curr * v

        # --- Mean guidance on the clean estimate (head queried at t=0) ---
        if active and mu_t > 0:
            t0 = torch.zeros(b, device=x.device)
            for _ in range(n_iter):
                z0 = z0.detach().requires_grad_(True)
                z_in = z0
                if gamma > 0:
                    z_in = z0 + gamma * torch.randn_like(z0)
                loss_mean = sum(
                    g["weight"] * g["_criterion"](g["head"](z_in, t0), g["target"])
                    for g in active
                )
                grad_mean = torch.autograd.grad(loss_mean, z0)[0]
                z0 = (z0 - mu_t * grad_mean).detach()

        t_b3 = t_curr.clamp(min=1e-6)
        d = (x - z0) / t_b3
        x = (x + d * (t_prev - t_curr)).detach()

    x = x.to(model_dtype)

    if log_rows:
        print("\n[LatCH] per-step grad norms:")
        for r in log_rows:
            print(f"  i={r['i']:>3} sigma={r['sigma']:.4f} "
                  f"||grad_var||={r['gv_norm']:.4e} ||x||={r['x_norm']:.4e} rho_t={r['rho_t']:.4e}")
    return x


def _st_weights(sigmas_1d):
    """Per-step s_t = alpha / sum(alpha), alpha = 1 - t. sigmas_1d: (steps+1,)."""
    alpha = (1.0 - sigmas_1d[:-1]).clamp(min=0.0)
    denom = alpha.sum().clamp(min=1e-8)
    return alpha / denom


def _scaled_step(grad, ref, normalize):
    """Guidance step direction. Raw gradient by default; when normalize=True, a unit-norm
    direction rescaled to ref's per-sample norm -> the gain becomes dimensionless (the fraction
    of the perturbed tensor's norm moved per step), so the useful range gravitates to O(0.01-1)
    and a single gain transfers across features/losses (removes cosine's 1/||pred|| scale)."""
    if not normalize:
        return grad
    gn = grad.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1)
    rn = ref.detach().flatten(1).norm(dim=1).view(-1, 1, 1)
    return grad / gn * rn


def sample_flow_euler_latch_guided(
    model, x, sigmas, *, head, target,
    rho=1.0, mu=1.0, gamma=0.3, n_iter=4,
    window=(0.5, 1.0), loss_type="mse", normalize=False,
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

    # Pre-warm the adaln_zero time cache if applicable. Only meaningful in the
    # non-per_element path — per_element batches use varying t per item and
    # don't hit the (uniform-t) cache anyway.
    if not per_element:
        _ensure_time_cache(head, sigmas_1d[:-1].tolist() + [0.0], x.device)

    def head_loss(pred, tgt):
        if loss_type == "bce_logits":
            return torch.nn.functional.binary_cross_entropy_with_logits(pred, tgt)
        if loss_type == "mse":
            return torch.nn.functional.mse_loss(pred, tgt)
        if loss_type in ("smooth_l1", "huber"):
            return torch.nn.functional.smooth_l1_loss(pred, tgt)
        if loss_type == "l1":
            return torch.nn.functional.l1_loss(pred, tgt)
        if loss_type == "cosine":
            # per-frame cosine distance over channels (chroma is a DIRECTION, not a magnitude)
            p = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
            t = tgt / (tgt.norm(dim=1, keepdim=True) + 1e-8)
            return (1.0 - (p * t).sum(dim=1)).mean()
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
            x = (x - rho_t * _scaled_step(grad, x, normalize)).detach()

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
                z0 = (z0 - mu_t * _scaled_step(grad, z0, normalize)).detach()
            x_hat0 = z0

        # --- Euler update reconstructed from the (possibly guided) clean estimate ---
        # v_eff = (x - x_hat0) / t_curr ; x_next = x + dt * v_eff
        # clamp guards variation runs where t_curr can approach 0 in the final steps
        t_b = t_curr.view(-1, 1, 1).clamp(min=1e-6)
        v_eff = (x - x_hat0) / t_b
        x = (x + dt * v_eff).detach()

    return x

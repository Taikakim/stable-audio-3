#!/usr/bin/env python
"""sa3_latch_onnx.py — CPU LatCH-GUIDED generator over the PLAIN DiT ONNX.

The LatCH sibling of ``sa3_control_onnx.py``. Where the control-DiT bakes a trained
forward adapter into the graph, LatCH guidance is a *gradient* method: the plain DiT
runs forward-only (numpy/ORT, no autograd), and guidance is applied by torch autograd
through a tiny LatCH **head** ONLY (a ~5-7M-param transformer that predicts an MIR
feature from noisy latents). Per step we convert the small (1,256,T) latent
numpy<->torch around the head grad calls; the DiT itself never sees torch.

This is a faithful port of
``stable_audio_3.inference.latch_guided.sample_flow_euler_multi_latch_guided``
(the TWO-stage variance+mean Selective-TFG sampler) and the schedule/target/seed
conventions in ``/run/media/kim/Mantu/latch_sweep/gen_one.py`` (the cos-validation
oracle). Everything runs fp32 on CPU; the ORT session is passed in (CPUExecutionProvider).

SCHEDULE (verified, NOT sa3_control_onnx.schedule):
  gen_one builds it with ``build_schedule(... dist_shift=inner.sampling_dist_shift ...)``.
  For SA3 medium-base ``sampling_distribution_shift_options`` is absent, so
  ``sampling_dist_shift = LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)``
  (seq-len-invariant). We replicate build_schedule + LogSNRShift exactly in torch fp32.
"""
import time

import numpy as np
import torch

# Per the brief: reuse constants from the control-DiT gen-core (do NOT edit it).
from sa3_control_onnx import (  # noqa: F401  (schedule imported for parity / callers)
    schedule,
    SR,
    LATENT_DIM,
    DS,
    LOCAL_ADD_DIM,
)

# LogSNRShift params for SA3 medium-base sampling_dist_shift (rate=0 => seq-len-invariant).
_LOGSNR_ANCHOR = -6.2     # anchor_logsnr; with rate=0 this IS logsnr_start for every seq_len
_LOGSNR_END = 2.0


def latch_schedule(steps: int, sigma_max: float = 1.0) -> torch.Tensor:
    """Replicate stable_audio_3.inference.sampling.build_schedule with the medium-base
    sampling_dist_shift (LogSNRShift rate=0, anchor_logsnr=-6.2, logsnr_end=2.0).

    Returns a torch fp32 tensor of shape (steps+1,) on CPU — bit-identical to what
    gen_one feeds the sampler (build_schedule on cpu, fp32, include_endpoint=True).
    """
    n_points = steps + 1
    t = torch.linspace(sigma_max, 0.0, n_points, dtype=torch.float32)  # t=1 noise .. t=0 data
    # LogSNRShift.shift: logsnr = logsnr_end - t*(logsnr_end - logsnr_start); t_out = sigmoid(-logsnr)
    logsnr_start = _LOGSNR_ANCHOR  # rate=0 => independent of seq_len
    logsnr = _LOGSNR_END - t * (_LOGSNR_END - logsnr_start)
    t_out = torch.sigmoid(-logsnr)
    # exact endpoint preservation (as in LogSNRShift.shift)
    t_out = torch.where(t <= 0, torch.zeros_like(t_out), t_out)
    t_out = torch.where(t >= 1, torch.ones_like(t_out), t_out)
    # build_schedule re-pins the first timestep to sigma_max after shifting
    t_out = t_out.clone()
    t_out[0] = float(sigma_max)
    return t_out


# ---------------------------------------------------------------------------
# Head / target / criterion helpers
# ---------------------------------------------------------------------------
def load_latch_head(ckpt_path: str):
    """Load a LatCH guidance head on CPU (fp32, eval) via the canonical loader.

    Returns (head, metadata). metadata carries std_mean/std_std/standardized/
    out_channels/loss_type (used to build the target + criterion).
    """
    from stable_audio_3.models.latch import load_latch_from_checkpoint

    head = load_latch_from_checkpoint(ckpt_path, device="cpu")
    head = head.float().eval()
    head.requires_grad_(False)  # we differentiate wrt the INPUT latent, not the weights
    return head, head.metadata


def make_criterion(loss_type: str, huber_beta: float = 1.0):
    """Guidance loss matching how the head was trained (mirrors latch_guided._make_latch_criterion).

    huber_beta is read from head metadata (meta.get('huber_beta')) for smooth_l1/huber heads;
    falls back to 1.0 when absent/None (matching latch_guided._make_latch_criterion).
    """
    if loss_type in ("smooth_l1", "huber"):
        return torch.nn.SmoothL1Loss(beta=float(huber_beta or 1.0))
    if loss_type == "mse":
        return torch.nn.MSELoss()
    if loss_type == "l1":
        return torch.nn.L1Loss()
    if loss_type == "bce_logits":
        return torch.nn.BCEWithLogitsLoss()
    if loss_type == "cosine":
        def _cos(pred, target):
            p = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
            t = target / (target.norm(dim=1, keepdim=True) + 1e-8)
            return (1.0 - (p * t).sum(dim=1)).mean()
        return _cos
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


def make_latch_target(raw_value_or_array, metadata, frames: int) -> torch.Tensor:
    """Build a (1, out_channels, frames) STANDARDIZED fp32 target from a RAW request.

    Ports gen_one._build_target on CPU:
      * scalar  -> constant (1, C, T)
      * (C,)    -> broadcast over T
      * (C, T)  -> nearest-resampled along time to fit T
    then standardized: (raw - std_mean) / std_std (if metadata['standardized']).
    """
    out_channels = int(metadata.get("out_channels", 1))
    std_mean = metadata.get("std_mean", 0.0)
    std_std = metadata.get("std_std", 1.0)
    standardized = bool(metadata.get("standardized", False))
    T = int(frames)

    if np.isscalar(raw_value_or_array):
        raw = torch.full((1, out_channels, T), float(raw_value_or_array), dtype=torch.float32)
    else:
        arr = np.asarray(raw_value_or_array, dtype=np.float32)
        if arr.ndim == 1:
            assert arr.shape[0] == out_channels, f"target {arr.shape} != C={out_channels}"
            raw = torch.from_numpy(arr).view(1, out_channels, 1).expand(1, out_channels, T).contiguous()
        elif arr.ndim == 2:
            assert arr.shape[0] == out_channels, f"target {arr.shape} != C={out_channels}"
            if arr.shape[1] != T:
                idx = np.clip((np.arange(T) * arr.shape[1] / T).astype(int), 0, arr.shape[1] - 1)
                arr = arr[:, idx]
            raw = torch.from_numpy(arr).unsqueeze(0)
        else:
            raise ValueError(f"unsupported target shape {arr.shape}")

    if standardized:
        raw = (raw - float(std_mean)) / (float(std_std) or 1.0)
    return raw.float()


# ---------------------------------------------------------------------------
# APG classifier-free guidance (host-side, faithful to dit.py at apg_scale=1.0)
# ---------------------------------------------------------------------------
def apg_cfg_velocity(x_np, v_cond_np, v_unc_np, t_cur: float, cfg_scale: float) -> np.ndarray:
    """Replicate dit.py's APG CFG combine (apg_scale=1.0) on the two ONNX velocity outputs.

    The plain DiT ONNX exports the CFG-free ``_forward`` core, so each ORT call returns a
    RAW per-condition velocity. The torch oracle (gen_one) instead lets the DiT do the CFG
    combine internally with apg_scale=1.0, cfg_norm_threshold=0, scale_phi=0. We reproduce
    exactly that branch here (dit.py lines 580-613, rectified_flow path, sigma == t_cur):

        cond_den = x - sigma*v_cond ;  unc_den = x - sigma*v_unc      (rectified_flow)
        diff     = cond_den - unc_den
        # apg_project(diff, cond_den) -> ORTHOGONAL component (dit.py lines 339-341):
        v1       = F.normalize(cond_den, dim=[-1,-2])                 # same eps=1e-12 default
        parallel = (diff*v1).sum(dim=[-1,-2], keepdim=True) * v1
        cfg_diff = diff - parallel                                    # orthogonal (apg_scale=1.0)
        cfg_den  = cond_den + (cfg_scale-1)*cfg_diff
        v        = (x - cfg_den) / sigma

    Done in torch fp32 (no grad) so F.normalize / reductions are bit-identical to dit.py's
    apg_project; returns numpy float32 velocity (1, C, T).
    """
    x = torch.from_numpy(np.ascontiguousarray(x_np)).float()
    v_cond = torch.from_numpy(np.ascontiguousarray(v_cond_np)).float()
    v_unc = torch.from_numpy(np.ascontiguousarray(v_unc_np)).float()

    cond_den = x - t_cur * v_cond
    unc_den = x - t_cur * v_unc
    diff = cond_den - unc_den

    # apg_project(v0=diff, v1=cond_den, padding_mask=None) — orthogonal component.
    v1 = torch.nn.functional.normalize(cond_den, dim=[-1, -2])
    parallel = (diff * v1).sum(dim=[-1, -2], keepdim=True) * v1
    cfg_diff = diff - parallel                       # apg_scale == 1.0 -> orthogonal only

    cfg_den = cond_den + (cfg_scale - 1.0) * cfg_diff
    v = (x - cfg_den) / t_cur
    return v.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# The guided generator
# ---------------------------------------------------------------------------
def generate_z0_latch_guided(
    dit_session,
    *,
    cond,
    uncond,
    guides,
    frames: int,
    steps: int = 30,
    cfg_scale: float = 7.0,
    seed: int = 777,
    rho: float = 64.0,
    mu: float = 64.0,
    gamma: float = 0.3,
    n_iter: int = 4,
    start_pct: float = 0.4,
    end_pct: float = 1.0,
):
    """LatCH-guided z0 over the PLAIN DiT ONNX (CPU). Faithful port of
    sample_flow_euler_multi_latch_guided (two guidance stages per step).

    cond/uncond: (cross_attn_cond, cross_attn_mask, global_embed) numpy tuples.
    guides: list of dicts with keys:
        head       -- torch LatCH (CPU, fp32, eval)
        target     -- torch [1, out_ch, frames] fp32, ALREADY standardized
        weight     -- float
        criterion  -- (optional) pre-built torch loss fn; if absent it is built from
                      loss_type + huber_beta below (mirrors latch_guided).
        loss_type  -- (optional) "smooth_l1"|"mse"|"l1"|"bce_logits"|"cosine" (default "mse")
        huber_beta -- (optional) beta for smooth_l1, from head metadata (FIX 3)
        start_pct  -- (optional) per-guide window start; defaults to the function-level start_pct
        end_pct    -- (optional) per-guide window end; defaults to the function-level end_pct
    Each guide is active for steps i in [int(num_steps*start_pct), int(num_steps*end_pct));
    the variance+mean losses sum ONLY over the guides active at that step (FIX 4).
    cfg_scale uses APG (apg_scale=1.0) host-side, faithful to dit.py (FIX 1).
    Returns {'z0': (1,256,frames) np.float32, 'dit_loop_s': float}.
    """
    c_cross, c_mask, c_glob = cond
    u_cross, u_mask, u_glob = uncond
    T = int(frames)
    b = 1
    local_add = np.zeros((1, LOCAL_ADD_DIM, T), np.float32)

    # ---- schedule (the gen_one / build_schedule LogSNR one) ----
    sigmas = latch_schedule(steps)                  # torch fp32 (steps+1,)
    num_steps = sigmas.shape[-1] - 1
    sig_np = sigmas.numpy().astype(np.float32)       # for the numpy euler/DiT math
    alpha = (1.0 - sigmas[:-1]).clamp(min=0.0)
    sum_alphas = float(alpha.sum().clamp(min=1e-8))

    # ---- per-guide windows + criteria (mirrors latch_guided.sample_flow_euler_multi_latch_guided) ----
    # Each guide gets its OWN [start_pct, end_pct] -> _start/_end = int(num_steps * pct).
    # Guides that don't specify their own fall back to the function-level start_pct/end_pct.
    for g in guides:
        g_start_pct = g.get("start_pct", start_pct)
        g_end_pct = g.get("end_pct", end_pct)
        g["_start"] = int(num_steps * g_start_pct)
        g["_end"] = int(num_steps * g_end_pct)
        # criterion: use a pre-built one if supplied, else build from loss_type + huber_beta
        # (huber_beta plumbed from head metadata, FIX 3).
        if g.get("criterion") is not None:
            g["_criterion"] = g["criterion"]
        else:
            g["_criterion"] = make_criterion(g.get("loss_type", "mse"), g.get("huber_beta", 1.0))

    # ---- RNG: draw the init latent the gen_one way (torch.manual_seed + torch.randn) ----
    # gen_one does: torch.manual_seed(seed); noise = torch.randn(1, io_channels=256, T).
    # Seeding ONCE then drawing the init from torch aligns BOTH the init latent AND the
    # subsequent gamma torch.randn_like stream with gen_one's generator order. (CPU torch
    # generator; bit-identical to gen_one's *logic* — gen_one's CUDA RNG stream differs by
    # device, but the method/seed-order is faithful.)
    torch.manual_seed(seed)
    x = torch.randn(1, LATENT_DIM, T, dtype=torch.float32).numpy().astype(np.float32)

    def dit_v(cross1, mask1, glob1, t_cur):
        """Plain-DiT forward (no control_tokens/gain). Returns velocity numpy (1,256,T)."""
        return dit_session.run(None, {
            "x": x,
            "t": np.full((1,), t_cur, np.float32),
            "cross_attn_cond": cross1[None],
            "cross_attn_cond_mask": mask1[None],
            "global_embed": glob1[None],
            "local_add_cond": local_add,
        })[0].astype(np.float32)

    t_loop = time.time()
    for i in range(num_steps):
        t_cur = float(sig_np[i])
        t_prev = float(sig_np[i + 1])
        s_t = float(alpha[i]) / sum_alphas
        rho_t = rho * s_t
        mu_t = mu * s_t
        # active set = guides whose own [_start, _end) window covers this step (FIX 4)
        active_guides = [g for g in guides if g["_start"] <= i < g["_end"]]

        # ---- (A) VARIANCE guidance on x at the true t_cur (head queried at t_cur) ----
        if active_guides and rho_t > 0.0:
            with torch.enable_grad():
                xt = torch.tensor(x, dtype=torch.float32, requires_grad=True)
                t_ten = torch.full((b,), t_cur, dtype=torch.float32)
                loss = sum(
                    g["weight"] * g["_criterion"](g["head"](xt, t_ten), g["target"])
                    for g in active_guides
                )
                grad = torch.autograd.grad(loss, xt)[0]
            x = (xt - rho_t * grad).detach().numpy().astype(np.float32)

        # ---- (B) DiT forward (numpy / ORT, NO grad). APG CFG on the PLAIN DiT (FIX 1) ----
        v_cond = dit_v(c_cross, c_mask, c_glob, t_cur)
        if cfg_scale == 1.0:
            v = v_cond
        else:
            v_unc = dit_v(u_cross, u_mask, u_glob, t_cur)
            # Faithful APG (apg_scale=1.0) CFG combine, exactly as dit.py does internally.
            v = apg_cfg_velocity(x, v_cond, v_unc, t_cur, cfg_scale)
        z0 = x - t_cur * v                          # clean estimate (numpy)

        # ---- (C) MEAN guidance on the clean estimate (head queried at t=0) ----
        if active_guides and mu_t > 0.0:
            with torch.enable_grad():
                z0t = torch.tensor(z0, dtype=torch.float32)
                t0 = torch.zeros(b, dtype=torch.float32)
                for _ in range(n_iter):
                    z0t = z0t.detach().requires_grad_(True)
                    z_in = z0t + gamma * torch.randn_like(z0t) if gamma > 0 else z0t
                    loss = sum(
                        g["weight"] * g["_criterion"](g["head"](z_in, t0), g["target"])
                        for g in active_guides
                    )
                    grad = torch.autograd.grad(loss, z0t)[0]
                    z0t = (z0t - mu_t * grad).detach()
            z0 = z0t.numpy().astype(np.float32)

        # ---- (D) Euler update reconstructed from the (possibly guided) clean estimate ----
        d = (x - z0) / max(t_cur, 1e-6)
        x = (x + d * (t_prev - t_cur)).astype(np.float32)

    dit_loop_s = time.time() - t_loop
    return {"z0": x.astype(np.float32), "dit_loop_s": dit_loop_s}

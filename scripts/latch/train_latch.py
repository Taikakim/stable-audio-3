# scripts/latch/train_latch.py
"""Train a LatCH head on SA3 SAME latents.

Phase 1 was AdamW + masked MSE on rectified_flow. This iteration adds the
FusionOpt + TemporalShapeLoss stack from SAT (LATCH_RESULTS.txt §17–18) while
keeping the minimal Phase-1 structure: flow-matching schedule, masked loss,
one optimizer step per batch, one checkpoint per epoch.

What's new vs Phase 1:
  --optimizer  {adamw, fusion}        FusionOpt = Muon(NS5)+MONA+KL-Shampoo+SF+
  --loss       {mse, smooth_l1, temporal}   temporal = TemporalShapeLoss
                                            (point + λ_d·deriv + λ_m·multi-scale)
  --hot-dtype  {fp32, bf16, fp16_safe}      NS5 quintic dtype on RDNA4 (bf16 default)
  --components ns5,normuon,sf               default is SF-NorMuon (production target
                                            per docs/FUSION_SHAREABLE.md); set to
                                            mona,shampoo,ns5,normuon,sf for full Fusion.
  --fp32-audit-period N                drift logging vs fp32 NS5 every N steps
  --seed       N                       bit-reproducible (LATCH_RESULTS §16)
  --save-best-only                     save the Schedule-Free averaged iterate
  --compile                            torch.compile the head (CRITICAL for FusionOpt
                                       per the doc — spectral-path overhead otherwise
                                       dominates the per-step cost)

ROCm env (mode 6, TunableOp, Triton/MIOpen cache paths) comes from rocm_env.yaml
via _apply_rocm_training_profile().
"""

import argparse
import os


def _apply_rocm_training_profile():
    """Apply rocm_env.yaml's `training` profile BEFORE torch is imported.

    Loaded standalone via importlib so the package __init__ (which applies the
    `inference` profile and pulls torch) does not run first. setdefault means the
    training keys we set here win over the later inference pass.
    """
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _re = _Path(__file__).resolve().parent.parent.parent / "stable_audio_3" / "rocm_env.py"
    if not _re.exists():
        return
    _spec = _ilu.spec_from_file_location("_sa3_rocm_env", _re)
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    _m.apply_profile("training")


_apply_rocm_training_profile()

import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts.latch.latch_dataset import LatCHDataset, collate_varlen  # noqa: E402
from scripts.latch.latch_model import LatCH  # noqa: E402


# ---------------------------------------------------------------------------
# Flow-matching schedule (unchanged from Phase 1)
# ---------------------------------------------------------------------------

def forward_noise(z0, noise, t):
    """Flow-matching linear interpolation z_t = (1-t)*z0 + t*noise. t: (B,)."""
    t = t.view(-1, 1, 1)
    return (1.0 - t) * z0 + t * noise


# ---------------------------------------------------------------------------
# Loss factories — all return (B, F, T) -> scalar tensor, mask-aware
# ---------------------------------------------------------------------------

def _masked_pointwise(pred, target, mask, fn):
    """Mean of fn(pred, target) over valid (mask=True) frames × channels."""
    m = mask.unsqueeze(1).to(pred.dtype)             # (B, 1, T)
    elem = fn(pred, target) * m
    denom = m.sum() * pred.shape[1]
    return elem.sum() / denom.clamp(min=1.0)


def masked_mse(pred, target, mask):
    return _masked_pointwise(pred, target, mask, lambda a, b: (a - b) ** 2)


def masked_smooth_l1(pred, target, mask, beta: float = 1.0):
    return _masked_pointwise(
        pred, target, mask,
        lambda a, b: F.smooth_l1_loss(a, b, beta=beta, reduction="none"),
    )


def make_criterion(args):
    """Return a callable (pred, target, mask) -> scalar loss tensor."""
    if args.loss == "mse":
        return masked_mse, "mse"
    if args.loss == "smooth_l1":
        beta = args.huber_beta
        return (lambda p, t, m: masked_smooth_l1(p, t, m, beta=beta)), "smooth_l1"
    if args.loss == "temporal":
        # TemporalShapeLoss = L_point + λ_d·L_deriv + λ_m·L_multi over (B, F, T).
        # We multiply pred/target by mask before the call: padded frames see
        # zero error on every component, contributing nothing to the gradient
        # (the averaging includes them with diluted weight — acceptable for the
        # Phase-1 fixed-length prototype; revisit if pad ratios get extreme).
        from stable_audio_tools.training.temporal_loss import TemporalShapeLoss
        crit = TemporalShapeLoss(
            huber_beta=args.huber_beta,
            lambda_deriv=args.lambda_deriv,
            lambda_multi=args.lambda_multi,
            curriculum_steps=args.curriculum_steps,
        )
        def _temporal(pred, target, mask):
            m = mask.unsqueeze(1).to(pred.dtype)
            return crit(pred * m, target * m)
        return _temporal, "temporal_shape"
    raise ValueError(f"Unknown --loss {args.loss!r}")


# ---------------------------------------------------------------------------
# Optimizer factories
# ---------------------------------------------------------------------------

def make_optimizer(args, model):
    """Return (optimizer, is_fusion_opt). FusionOpt routes 2D matrices with
    min(shape) >= 128 to the spectral Muon+MONA+KL-Shampoo path; everything
    else (1D, biases, LayerNorm, small/odd projections) to ScheduleFree-AdamW."""
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr), False
    if args.optimizer == "fusion":
        from stable_audio_tools.training.fusion_opt import FusionOpt
        from stable_audio_tools.training.fusion_groups import (
            build_fusion_param_groups, summarise_groups,
        )
        groups = build_fusion_param_groups(model, spectral_wd=0.01, scalar_wd=0.0)
        print("FusionOpt groups:")
        print(summarise_groups(groups))
        components = (
            None if not args.components else set(args.components.split(","))
        )
        opt = FusionOpt(
            groups,
            lr=args.lr,
            mona_alpha=args.mona_alpha,
            hot_dtype=args.hot_dtype,
            fp32_audit_period=args.fp32_audit_period,
            components=components,
        )
        print(f"FusionOpt components: {sorted(opt.components)}  "
              f"uses_sf_averaging={opt.uses_sf_averaging}  "
              f"hot_dtype={args.hot_dtype}")
        return opt, True
    raise ValueError(f"Unknown --optimizer {args.optimizer!r}")


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LatCHDataset(args.latent_dir, target_feature=args.feature,
                      db_path=args.db_path, target_source=args.target_source)
    print(f"Dataset: {len(ds)} crops, target_source={args.target_source}, feature={args.feature}")
    sample_latent, sample_target = ds[0]
    out_channels = sample_target.shape[0]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, collate_fn=collate_varlen,
                        persistent_workers=args.num_workers > 0)

    model = LatCH(in_channels=256, out_channels=out_channels,
                  dim=args.dim, depth=args.depth, num_heads=args.num_heads,
                  t_injection=args.t_injection).to(device)
    print(f"LatCH head: dim={args.dim} depth={args.depth} heads={args.num_heads} "
          f"t_injection={args.t_injection}, "
          f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if args.compile:
        # Build the optimiser on the un-compiled module so FusionOpt's
        # build_fusion_param_groups can introspect param names cleanly; the
        # underlying tensors are shared with the compiled callable.
        opt, is_fusion = make_optimizer(args, model)
        model = torch.compile(model)
        print("torch.compile: on (1st-iter Triton autotune will spike)")
    else:
        opt, is_fusion = make_optimizer(args, model)
    criterion, loss_type = make_criterion(args)

    print(f"Loss: {loss_type}; Optimizer: {args.optimizer}; Precision: {args.precision}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Target standardization stats (zero-mean/unit-std) from a sample of the dataset.
    std_mean, std_std = 0.0, 1.0
    if args.standardize:
        rng = np.random.RandomState(0)
        samp = [ds[int(i)][1].numpy().reshape(-1)
                for i in rng.randint(0, len(ds), size=min(256, len(ds)))]
        allv = np.concatenate(samp)
        std_mean = float(allv.mean())
        std_std = float(allv.std()) or 1.0
        print(f"Standardize: mean={std_mean:.4f} std={std_std:.4f}")
    use_bf16 = args.precision == "bf16"

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        if is_fusion:
            opt.train()  # SF eval point y = (1-β)·z + β·x in live params

        total, nbatches = 0.0, 0
        for batch in loader:
            latents = batch["latents"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            if args.standardize:
                targets = (targets - std_mean) / std_std
            t = torch.rand(latents.shape[0], device=device)
            noise = torch.randn_like(latents)
            z_t = forward_noise(latents, noise, t)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                preds = model(z_t, t)
            loss = criterion(preds.float(), targets, mask)
            opt.zero_grad()
            loss.backward()
            if is_fusion:
                # Polyak γ_t = γ_base·clamp(loss_ema / gnorm_ema) needs the loss tensor
                opt.set_loss(loss)
            opt.step()
            total += loss.item()
            nbatches += 1
        avg_loss = total / max(nbatches, 1)
        print(f"epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}")

        # For FusionOpt save the Schedule-Free averaged iterate x_t (deployable),
        # not the fast iterate z_t. opt.eval() swaps x into live params.
        if is_fusion:
            opt.eval()

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if args.save_best_only:
            if is_best:
                ckpt_path = os.path.join(args.save_dir, f"latch_sa3_{args.feature}_best.pt")
            else:
                ckpt_path = None
        else:
            ckpt_path = os.path.join(args.save_dir, f"latch_sa3_{args.feature}_ep{epoch+1}.pt")
        if ckpt_path is not None:
            torch.save({
                "state_dict": model.state_dict(),
                "feature_name": args.feature,
                "noise_schedule": "rectified_flow",
                "loss_type": loss_type,
                "optimizer": args.optimizer,
                "t_injection": args.t_injection,
                "in_channels": 256,
                "out_channels": out_channels,
                "standardized": args.standardize,
                "std_mean": std_mean,
                "std_std": std_std,
                "precision": args.precision,
                "seed": args.seed,
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
            }, ckpt_path)
            print(f"  -> saved {ckpt_path}{' (new best)' if is_best else ''}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Data / target
    p.add_argument("--feature", default="rms_energy_bass")
    p.add_argument("--latent-dir", default="/run/media/kim/Lehto/latents_sa3")
    p.add_argument("--db-path", default=None)
    p.add_argument("--target-source", choices=["db", "npz"], default="npz",
                   help="npz = <stem>.TIMESERIES.npz companions (latents_sa3, medium grid); "
                        "db = legacy per-crop TimeseriesDB (small-music-base / phase 1).")
    # Training loop
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16",
                   help="bf16 autocast on the head forward — ~6x throughput vs fp32 on "
                        "RDNA4 at T=4096. Loss is computed in fp32. fp32 only for debugging.")
    p.add_argument("--standardize", action="store_true",
                   help="Zero-mean/unit-std the target (stats from a 256-sample draw, stored "
                        "in the ckpt for inference de-standardization). Important for dB-scale "
                        "rms and low-variance spectral features (LATCH_RESULTS §18).")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save-dir", default="latch_weights_sa3")
    p.add_argument("--save-best-only", action="store_true",
                   help="Save only when train-loss improves (Schedule-Free averaged iterate)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the LatCH head. Per docs/FUSION_SHAREABLE.md this "
                        "is CRITICAL for FusionOpt — without it the spectral-path overhead "
                        "dominates the per-step cost; with it FusionOpt's wall-clock per step "
                        "matches AdamW's on the same model size. Pays a one-time Triton "
                        "autotune cost on the first epoch; the cache persists via rocm_env.")
    p.add_argument("--t-injection", choices=["concat", "film", "adaln_zero"], default="adaln_zero",
                   help="How t is injected into the LatCH head. adaln_zero = DiT-style "
                        "per-block (γ,β,α) modulators, T=256, the LATCH_RESULTS §17 winner "
                        "(val 0.1682) and the precondition for TimeConditioningCache "
                        "inference speedup. film = single (scale,shift) after latent_proj, "
                        "T=256, FA-aligned. concat = legacy prepend-token, T=257.")
    # Architecture: defaults match the production winner per LATCH_RESULTS §21/§22.
    p.add_argument("--dim", type=int, default=256,
                   help="LatCH transformer hidden size. Production target: 256. §22 "
                        "showed wider (d512) wastes throughput for no quality gain.")
    p.add_argument("--depth", type=int, default=4,
                   help="LatCH transformer depth. Production target: 4 (was 6 in Phase 1). "
                        "§22 confirmed d256/dp4 is the smallest sensible architecture: "
                        "matches d256/dp6 quality at ~67%% of the inference cost; depth past "
                        "6 stops paying off. Use 8 for fast-prototyping niche only.")
    p.add_argument("--num-heads", type=int, default=8)
    # Optimizer
    p.add_argument("--optimizer", choices=["adamw", "fusion"], default="adamw")
    p.add_argument("--hot-dtype", choices=["fp32", "bf16", "fp16_safe"], default="bf16",
                   help="FusionOpt NS5 dtype. bf16 = ~1.65x faster than fp32; "
                        "fp16_safe = ~1.3-1.5x faster than bf16 with fp32 polynomial accumulation. "
                        "fp16 plain will diverge — not exposed.")
    p.add_argument("--components", default="ns5,normuon,sf",
                   help="FusionOpt comma-separated subset of {mona,shampoo,ns5,normuon,sf}. "
                        "Default ns5,normuon,sf = SF-NorMuon, the production target per "
                        "docs/FUSION_SHAREABLE.md: captures ~95%% of the quality lift over "
                        "AdamW; adding mona+shampoo buys ~0.8%% for 50%% more wall-clock. "
                        "Empty string = all components (full Fusion).")
    p.add_argument("--fp32-audit-period", type=int, default=0,
                   help="Every N steps recompute NS5 in fp32 alongside hot_dtype and log "
                        "relative-error stats. 0 = off. Useful for verifying fp16_safe/bf16 "
                        "isn't quietly destabilising spectral updates.")
    p.add_argument("--mona-alpha", type=float, default=0.2,
                   help="FusionOpt MONA curvature-injection strength (default 0.2).")
    # Loss
    p.add_argument("--loss", choices=["mse", "smooth_l1", "temporal"], default="mse")
    p.add_argument("--huber-beta", type=float, default=1.0,
                   help="SmoothL1 knee. Also used by TemporalShapeLoss for deriv/multi-scale.")
    p.add_argument("--lambda-deriv", type=float, default=1.0,
                   help="TemporalShapeLoss: weight on the derivative term (default 1.0).")
    p.add_argument("--lambda-multi", type=float, default=0.5,
                   help="TemporalShapeLoss: weight on the multi-scale L1 term (default 0.5).")
    p.add_argument("--curriculum-steps", type=int, default=0,
                   help="TemporalShapeLoss: linear warmup of lambda_deriv/lambda_multi from "
                        "0 over N steps. 0 = constant from step 1.")
    train(p.parse_args())

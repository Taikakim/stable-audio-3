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
from torch.utils.data import DataLoader, Subset  # noqa: E402

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
# Held-out validation split — BY SOURCE TRACK, not by crop
# ---------------------------------------------------------------------------

def split_indices(groups, val_frac, seed=0):
    """Deterministic train/val split that never splits a GROUP across the two sides.

    `groups[i]` is item i's group key (for `latents_sa3`, its source track). Holding out
    whole groups is not a refinement here, it is the only honest option: in that corpus every
    one of the 2676 source tracks contributes >=2 crops, so a crop-level split would leak a
    sibling crop of essentially every val track into train. Sibling crops of one goa track
    share key, lead patch and often literal repeated loop material — for an f0/melody target
    that is close to training on the validation set.

    val_frac == 0 returns (everything, []) so runs that do not opt in are unchanged.
    """
    n = len(groups)
    if val_frac <= 0.0:
        return list(range(n)), []
    uniq = sorted(set(groups))
    shuffled = list(uniq)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(val_frac * len(uniq))))
    if n_val >= len(uniq):
        raise ValueError(
            f"val split would hold out {n_val} of {len(uniq)} group(s), leaving no training "
            f"data. The dataset has too few distinct groups for --val-frac {val_frac}.")
    held = set(shuffled[:n_val])
    train = [i for i, g in enumerate(groups) if g not in held]
    val = [i for i, g in enumerate(groups) if g in held]
    if not train or not val:
        raise ValueError(f"val split produced an empty side (train={len(train)}, val={len(val)})")
    return train, val


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
    if args.loss == "cosine":
        # Chroma is a DIRECTION, not a magnitude (handoff §5): per-frame cosine over the
        # 384 = 3×128 channels, masked-averaged. Scale-invariant ⟹ don't standardize chroma.
        def _cosine(pred, target, mask):
            num = (pred * target).sum(1)                                # (B, T)
            den = (pred.norm(dim=1) * target.norm(dim=1)).clamp_min(1e-8)
            cos = num / den
            m = mask.to(cos.dtype)
            return ((1.0 - cos) * m).sum() / m.sum().clamp_min(1.0)
        return _cosine, "cosine"
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
        # DAMPING (C 2026-08-19): in-optimizer decay over the run + optional SNR gate. train_latch's
        # steps are epochs*len(loader)/grad_accum; the loader exists by the time make_optimizer runs.
        _total = int(getattr(args, "_total_opt_steps", 0))
        if args.fusion_snr != "off":
            components = set(components or {"mona", "shampoo", "ns5", "normuon", "sf", "cautious"}) | {"snr"}
        opt = FusionOpt(
            groups,
            lr=args.lr,
            mona_alpha=args.mona_alpha,
            hot_dtype=args.hot_dtype,
            fp32_audit_period=args.fp32_audit_period,
            components=components,
            decay_schedule=args.fusion_decay, total_steps=_total, decay_min=args.fusion_decay_min,
            snr_mode=(args.fusion_snr if args.fusion_snr != "off" else "row"),
            snr_beta=args.fusion_snr_beta,
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
    # Resolve the (possibly multi-root) latent dir. --latent-dirs (list) wins; else split --latent-dir
    # on commas. A single dir stays a str (byte-identical single-root path); several -> a list.
    if args.latent_dirs:
        latent_dir = args.latent_dirs if len(args.latent_dirs) > 1 else args.latent_dirs[0]
    else:
        _parts = [s for s in args.latent_dir.split(",") if s]
        latent_dir = _parts if len(_parts) > 1 else args.latent_dir
    if isinstance(latent_dir, list):
        print(f"Multi-root dataset: {len(latent_dir)} corpora {latent_dir}")
    ds = LatCHDataset(latent_dir, target_feature=args.feature,
                      db_path=args.db_path, target_source=args.target_source,
                      chroma_dir=args.chroma_dir, chroma_key=args.chroma_key,
                      voiced_field=args.voiced_field)
    print(f"Dataset: {len(ds)} crops, target_source={args.target_source}, feature={args.feature}"
          + (f", voiced_field={args.voiced_field}" if args.voiced_field else ""))
    sample_latent, sample_target, _ = ds[0]
    out_channels = sample_target.shape[0]

    # --- Held-out validation (opt-in; --val-frac 0 = every prior run's behaviour) ----------
    # Without this, `_best.pt` is the epoch with the lowest TRAINING loss, which is not a
    # convergence claim — it is the most-memorised checkpoint. Split is BY SOURCE TRACK; see
    # split_indices for why a crop-level split cannot work on this corpus.
    val_loader = None
    val_meta = None
    if args.val_frac > 0.0:
        groups = ds.group_keys(args.val_group_by)
        tr_idx, va_idx = split_indices(groups, args.val_frac, seed=args.val_seed)
        n_groups = len(set(groups))
        n_val_groups = len({groups[i] for i in va_idx})
        val_meta = {"val_frac": args.val_frac, "val_seed": args.val_seed,
                    "val_group_by": args.val_group_by, "n_groups": n_groups,
                    "n_val_groups": n_val_groups, "n_train_crops": len(tr_idx),
                    "n_val_crops": len(va_idx)}
        print(f"Val split: {len(va_idx)} crops from {n_val_groups} held-out "
              f"{args.val_group_by}(s) / {n_groups} total; {len(tr_idx)} crops train"
              + ("  [LEAKY: crop-level split]" if args.val_group_by in (None, "none") else ""))
        val_ds = Subset(ds, va_idx)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                drop_last=False, num_workers=args.num_workers,
                                collate_fn=collate_varlen,
                                persistent_workers=args.num_workers > 0)
        train_ds = Subset(ds, tr_idx)
    else:
        train_ds, tr_idx = ds, list(range(len(ds)))

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, collate_fn=collate_varlen,
                        persistent_workers=args.num_workers > 0)

    args._total_opt_steps = (len(loader) // max(1, args.grad_accum)) * args.epochs   # for --fusion-decay
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
        # Sample TRAIN items only — target mean/std computed over the val crops too would leak
        # held-out statistics into the model's output scale, a quiet way to flatter val loss.
        rng = np.random.RandomState(0)
        samp = [ds[int(tr_idx[i])][1].numpy().reshape(-1)
                for i in rng.randint(0, len(tr_idx), size=min(256, len(tr_idx)))]
        allv = np.concatenate(samp)
        std_mean = float(allv.mean())
        std_std = float(allv.std()) or 1.0
        print(f"Standardize: mean={std_mean:.4f} std={std_std:.4f}")
    use_bf16 = args.precision == "bf16"

    # --- EMA of the HEAD's params (no DiT here). When --ema>0 we keep a param-wise
    #     exponential moving average updated on each OPTIMIZER step (every --grad-accum
    #     micro-batches), and the EMA iterate is what gets saved (it's what we evaluate). ---
    ema_params = None
    if args.ema > 0.0:
        ema_params = [p.detach().clone() for p in model.parameters()]
        print(f"EMA: on, decay={args.ema} (saved checkpoint will be the EMA weights)")

    def ema_state_dict():
        """state_dict with the EMA params swapped in (restores live params after)."""
        backup = [p.detach().clone() for p in model.parameters()]
        with torch.no_grad():
            for p, e in zip(model.parameters(), ema_params):
                p.copy_(e)
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            for p, b in zip(model.parameters(), backup):
                p.copy_(b)
        return sd

    # --- Full tiered telemetry -> wandb. STANDING REQUIREMENT (SAO/MASTER.md): every
    #     control-head run logs per-layer norms / dist-from-init / weight-space trajectory
    #     via avp_sa3's TrainTelemetry (RF loss is blind to control; the per-layer + trajectory
    #     signals are the real diagnostics, and the cross-run fingerprint builds the
    #     DiT-block x feature controllability map — only reconstructable if logged every run). ---
    import time as _time
    telem = wb = None
    step = 0
    _t_prev = _time.perf_counter()
    if args.wandb:
        import sys as _sys
        _sat = "/home/kim/Projects/SAO/stable-audio-tools"
        if _sat not in _sys.path:
            _sys.path.insert(0, _sat)
        import wandb as wb        # telemetry.py expects the wandb MODULE (wb.log / wb.Histogram), not the run
        from avp_sa3.sa3_control.telemetry import TrainTelemetry
        run_name = args.run_name or f"latch_{args.feature}_{args.t_injection}_d{args.depth}_{args.optimizer}"
        wb.init(project=args.wandb_project, name=run_name, config=vars(args))
        telem = TrainTelemetry(model, wb, scalar_every=args.log_every, layer_every=args.layer_every)
        print(f"[wandb] full telemetry -> project={args.wandb_project} run={run_name}", flush=True)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        if is_fusion:
            opt.train()  # SF eval point y = (1-β)·z + β·x in live params

        total, nbatches = 0.0, 0
        accum = max(1, args.grad_accum)   # effective batch = batch_size * accum
        n_micro = len(loader)
        opt.zero_grad()
        micro_in_group = 0
        for bi, batch in enumerate(loader):
            latents = batch["latents"].to(device)
            targets = batch["targets"].to(device)
            # mask = length (padding) mask; weight = per-frame loss weight (all-ones unless
            # --voiced-field is set — the melody head's voiced-fraction weighting, SAO
            # WORKLOG 2026-08-13). Folding weight into mask here is the ONLY change needed:
            # every masked_* loss already does mask.to(pred.dtype), so a continuous-valued
            # mask acts as a soft per-frame weight with zero changes to the loss functions.
            mask = batch["mask"].to(device).float() * batch["weight"].to(device)
            if args.standardize:
                targets = (targets - std_mean) / std_std
            t = torch.rand(latents.shape[0], device=device)
            noise = torch.randn_like(latents)
            z_t = forward_noise(latents, noise, t)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                preds = model(z_t, t)
            loss = criterion(preds.float(), targets, mask)
            (loss / accum).backward()    # scale so accumulated grads ≈ mean over the effective batch
            total += loss.item()         # report the unscaled per-micro-batch loss
            nbatches += 1
            micro_in_group += 1
            is_last = (bi == n_micro - 1)
            if micro_in_group == accum or is_last:   # optimizer step every `accum` micro-batches (+ final partial group)
                if is_fusion:
                    # Polyak γ_t = γ_base·clamp(loss_ema / gnorm_ema) needs the loss tensor
                    opt.set_loss(loss)
                if telem is not None:
                    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))  # measure only, no real clip
                    if hasattr(opt, "_telem_on"):       # FusionOpt: instrument the step we're about to log
                        opt._telem_on = (step % args.log_every == 0)
                opt.step()
                opt.zero_grad()
                if ema_params is not None:   # EMA on the optimizer step (every `accum` micro-batches), not per micro-batch
                    with torch.no_grad():
                        for e, p in zip(ema_params, model.parameters()):
                            e.lerp_(p.detach(), 1.0 - args.ema)
                if telem is not None:
                    _now = _time.perf_counter()
                    _rate = 1.0 / max(_now - _t_prev, 1e-6); _t_prev = _now
                    telem.log(step, loss=loss.item(), gnorm=gnorm, it_s=_rate,
                              lr=opt.param_groups[0]["lr"], epoch=epoch)
                step += 1
                micro_in_group = 0
        avg_loss = total / max(nbatches, 1)
        print(f"epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}")

        # For FusionOpt save the Schedule-Free averaged iterate x_t (deployable),
        # not the fast iterate z_t. opt.eval() swaps x into live params.
        if is_fusion:
            opt.eval()

        # --- validation ------------------------------------------------------------------
        # Runs AFTER opt.eval() so FusionOpt is measured on its deployable averaged iterate,
        # and with the EMA weights swapped in when EMA is on — i.e. we validate whatever this
        # epoch would actually SAVE, not a different set of weights.
        val_loss = None
        if val_loader is not None:
            swapped = None
            if ema_params is not None:
                swapped = [p.detach().clone() for p in model.parameters()]
                with torch.no_grad():
                    for p, e in zip(model.parameters(), ema_params):
                        p.copy_(e)
            model.eval()
            vtot, vb = 0.0, 0
            with torch.no_grad():
                for vi, batch in enumerate(val_loader):
                    latents = batch["latents"].to(device)
                    targets = batch["targets"].to(device)
                    mask = batch["mask"].to(device).float() * batch["weight"].to(device)
                    if args.standardize:
                        targets = (targets - std_mean) / std_std
                    # FIXED noise/t per batch index, identical every epoch: otherwise the
                    # epoch-to-epoch val delta is dominated by which t's happened to be drawn,
                    # and a "val improved" reading would be mostly resampling noise.
                    g = torch.Generator().manual_seed(args.val_seed * 1000003 + vi)
                    t = torch.rand(latents.shape[0], generator=g).to(device)
                    noise = torch.randn(latents.shape, generator=g).to(device)
                    z_t = forward_noise(latents, noise, t)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        preds = model(z_t, t)
                    vtot += criterion(preds.float(), targets, mask).item()
                    vb += 1
            val_loss = vtot / max(vb, 1)
            model.train()
            if swapped is not None:
                with torch.no_grad():
                    for p, b in zip(model.parameters(), swapped):
                        p.copy_(b)
            print(f"           val={val_loss:.4f}  (held-out {val_meta['n_val_groups']} "
                  f"{args.val_group_by}s)")

        # Select on VAL when we have it — training loss selects the most-memorised epoch.
        sel = val_loss if val_loss is not None else avg_loss
        is_best = sel < best_loss
        if is_best:
            best_loss = sel

        if args.save_best_only:
            if is_best:
                ckpt_path = os.path.join(args.save_dir, f"latch_sa3_{args.feature}_best.pt")
            else:
                ckpt_path = None
        else:
            ckpt_path = os.path.join(args.save_dir, f"latch_sa3_{args.feature}_ep{epoch+1}.pt")
        if ckpt_path is not None:
            torch.save({
                "state_dict": ema_state_dict() if ema_params is not None else model.state_dict(),
                "ema": args.ema,
                "grad_accum": args.grad_accum,
                "effective_batch": args.batch_size * max(1, args.grad_accum),
                "feature_name": args.feature,
                # Provenance for the TARGET itself. Before 2026-08-24 only
                # feature_name was saved, so a 12-d head could not be traced back
                # to WHICH chroma readout it was trained against (Kim, 2026-08-24) --
                # eval/head_meta.py reports readout_source="not-recorded" for those.
                "target_source": args.target_source,
                "chroma_dir": args.chroma_dir,
                "chroma_key": args.chroma_key,
                "db_path": args.db_path,
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
                # None when the run had no held-out set. A consumer can then tell a
                # val-selected head from a train-selected one instead of assuming.
                "val_loss": val_loss,
                "val_split": val_meta,
                "selected_on": "val_loss" if val_loader is not None else "train_loss",
            }, ckpt_path)
            print(f"  -> saved {ckpt_path}{' (new best)' if is_best else ''}")

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Data / target
    p.add_argument("--feature", default="rms_energy_bass")
    p.add_argument("--latent-dir", default="/home/kim/Projects/latents_sa3",
                   help="latent dir; ALSO accepts a comma-separated list (goa,avp) for a combined "
                        "multi-root dataset (full-path item lists concatenated -> stem collision sidestepped).")
    p.add_argument("--latent-dirs", nargs="+", default=None,
                   help="explicit multi-root form of --latent-dir: one or more latent dirs "
                        "(space-separated). Overrides --latent-dir when given.")
    p.add_argument("--db-path", default=None)
    p.add_argument("--chroma-dir", default=None,
                   help="dir of per-crop <stem>.npz SAME-chroma (3,128,T); enables --target-source chroma")
    p.add_argument("--chroma-key", default="other", help="which stem's chroma: other / bass / full_mix")
    p.add_argument("--voiced-field", default=None,
                   help="npz field carrying a per-frame [0,1] loss WEIGHT, pooled-voiced-fraction "
                        "convention (e.g. f0_other_voiced_ts for --feature f0_other). Melody/pitch "
                        "targets carry an unvoiced sentinel (0.0 Hz) that must not be regressed on "
                        "directly -- weighting by voiced fraction rather than hard-masking is the "
                        "settled design (SAO WORKLOG 2026-08-13). Only target_source=npz honors this; "
                        "default None = every other feature's loss is byte-identical to before.")
    p.add_argument("--target-source", choices=["db", "npz", "chroma", "scalar_json"], default="npz",
                   help="npz = <stem>.TIMESERIES.npz companions (latents_sa3, medium grid); "
                        "db = legacy per-crop TimeseriesDB (small-music-base / phase 1); "
                        "scalar_json = constant target from <stem>.TIMBRAL.json "
                        "(--feature hardness/depth/booming, pooled-readout head).")
    # Training loop
    p.add_argument("--val-frac", type=float, default=0.0,
                   help="fraction of GROUPS (source tracks) held out for validation. 0 (default) "
                        "= no val set, and `_best.pt` is then selected on TRAINING loss, which "
                        "is not a convergence signal. Set this for any run you intend to "
                        "believe.")
    p.add_argument("--val-seed", type=int, default=0,
                   help="seed for the group split AND for the fixed val noise/t draws")
    p.add_argument("--val-group-by", default="source_track",
                   help="crop-json field to group by so a track's crops never straddle the "
                        "split ('none' = crop-level split; LEAKY on latents_sa3, where every "
                        "track has >=2 crops)")
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
    p.add_argument("--ema", type=float, default=0.0,
                   help="EMA decay for the LatCH head's params (0 = off, e.g. 0.999). When >0, "
                        "a param-wise exponential moving average is updated on each optimizer step "
                        "and the SAVED checkpoint's state_dict is the EMA (the EMA iterate is what "
                        "we evaluate). Damping for the flat-RF drift (SAO/MASTER.md §4).")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Accumulate grads over N micro-batches before optimizer.step() "
                        "(loss scaled by 1/N, step+zero_grad every N, final partial group steps too). "
                        "Effective batch = batch_size * N. EMA updates on the optimizer step, not per micro-batch.")
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
    p.add_argument("--fusion-decay", dest="fusion_decay", default="none",
                   choices=("none", "cosine", "linear", "wsd"),
                   help="FusionOpt in-optimizer LR decay over the run (see train_lora --fusion-decay)")
    p.add_argument("--fusion-decay-min", dest="fusion_decay_min", type=float, default=0.0)
    p.add_argument("--fusion-snr", dest="fusion_snr", default="off", choices=("off", "row", "elem"),
                   help="FusionOpt SNR gate per row|elem (see train_lora --fusion-snr)")
    p.add_argument("--fusion-snr-beta", dest="fusion_snr_beta", type=float, default=0.9)
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
    p.add_argument("--loss", choices=["mse", "smooth_l1", "temporal", "cosine"], default="mse")
    p.add_argument("--huber-beta", type=float, default=1.0,
                   help="SmoothL1 knee. Also used by TemporalShapeLoss for deriv/multi-scale.")
    p.add_argument("--lambda-deriv", type=float, default=1.0,
                   help="TemporalShapeLoss: weight on the derivative term (default 1.0).")
    p.add_argument("--lambda-multi", type=float, default=0.5,
                   help="TemporalShapeLoss: weight on the multi-scale L1 term (default 0.5).")
    p.add_argument("--curriculum-steps", type=int, default=0,
                   help="TemporalShapeLoss: linear warmup of lambda_deriv/lambda_multi from "
                        "0 over N steps. 0 = constant from step 1.")
    p.add_argument("--wandb", action="store_true",
                   help="log full tiered telemetry (avp_sa3/sa3_control/telemetry.py) to wandb — "
                        "STANDING REQUIREMENT per SAO/MASTER.md for every control-head run")
    p.add_argument("--wandb-project", default="sa3-latch")
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-every", type=int, default=20, help="telemetry scalar_every (per-step scalars)")
    p.add_argument("--layer-every", type=int, default=200, help="telemetry layer_every (per-layer norms/trajectory)")
    train(p.parse_args())

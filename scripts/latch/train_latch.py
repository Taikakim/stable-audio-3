# scripts/latch/train_latch.py
"""Train a LatCH head on SA3 SAME latents (Phase 1: rms_energy_bass, flow-matching schedule)."""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from scripts.latch.latch_dataset import LatCHDataset, collate_varlen
from scripts.latch.latch_model import LatCH


def forward_noise(z0, noise, t):
    """Flow-matching linear interpolation z_t = (1-t)*z0 + t*noise. t: (B,)."""
    t = t.view(-1, 1, 1)
    return (1.0 - t) * z0 + t * noise


def masked_mse(pred, target, mask):
    """MSE over valid (mask=True) frames only. mask: (B, T)."""
    m = mask.unsqueeze(1).to(pred.dtype)          # (B, 1, T)
    se = ((pred - target) ** 2) * m
    denom = m.sum() * pred.shape[1]
    return se.sum() / denom.clamp(min=1.0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LatCHDataset(args.latent_dir, target_feature=args.feature, db_path=args.db_path)
    sample_latent, sample_target = ds[0]
    out_channels = sample_target.shape[0]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, collate_fn=collate_varlen,
                        persistent_workers=args.num_workers > 0)

    model = LatCH(in_channels=256, out_channels=out_channels,
                  dim=256, depth=6, num_heads=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_type = "mse"  # Phase 1 targets rms_energy_bass

    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            latents = batch["latents"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            t = torch.rand(latents.shape[0], device=device)
            noise = torch.randn_like(latents)
            z_t = forward_noise(latents, noise, t)
            preds = model(z_t, t)
            loss = masked_mse(preds, targets, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch+1}/{args.epochs}  loss={total/len(loader):.4f}")
        torch.save({
            "state_dict": model.state_dict(),
            "feature_name": args.feature,
            "noise_schedule": "rectified_flow",
            "loss_type": loss_type,
            "in_channels": 256,
            "out_channels": out_channels,
        }, os.path.join(args.save_dir, f"latch_sa3_{args.feature}_ep{epoch+1}.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature", default="rms_energy_bass")
    p.add_argument("--latent-dir", default="/run/media/kim/Lehto/sa3-latch-latents")
    p.add_argument("--db-path", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-dir", default="latch_weights_sa3")
    train(p.parse_args())

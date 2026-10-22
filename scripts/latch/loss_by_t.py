# Owner: WINTERMUTE (created 2026-10-22)
"""Held-out loss of LatCH heads split by noise level t: how much of the val loss is reducible at all?

The trainer's val loss averages over t ~ U(0,1). At high t the latent is mostly noise and the best any head can do
approaches the target's own variance (1.0 in standardized units, MSE), so part of every val number is a floor no
optimizer can move. This evaluates each checkpoint on the SAME held-out crops (the trainer's group split) at fixed
t-bin centres with fixed per-item noise, and prints per-bin loss, each bin's share of the total, and the gap between
checkpoints per bin. Also: the loss of the best constant predictor (the per-bin ceiling) for scale.

CPU is fine (fp32; reproduces the trainer's bf16 val to ~1e-4 on EB64). Run from anywhere:
  .venv-rocm10-amd/bin/python stable-audio-3/scripts/latch/loss_by_t.py --ckpt A.pt --ckpt B.pt --latent-dir ...
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from scripts.latch.latch_dataset import LatCHDataset, collate_varlen  # noqa: E402
from scripts.latch.latch_model import LatCH  # noqa: E402
from scripts.latch.train_latch import forward_noise, split_indices  # noqa: E402


def load_head(path, depth, dim, heads):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch") or {"dim": dim, "depth": depth, "num_heads": heads}
    m = LatCH(in_channels=ck.get("in_channels", 256), out_channels=ck["out_channels"], dim=arch["dim"],
              depth=arch["depth"], num_heads=arch["num_heads"], t_injection=ck.get("t_injection", "adaln_zero"))
    m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["state_dict"].items()}, strict=True)
    return m.eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--label", action="append", default=None)
    ap.add_argument("--latent-dir", required=True)
    ap.add_argument("--feature", default="onset_per_beat_ts")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--val-group-by", default="relpath")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="write the table as JSON here")
    a = ap.parse_args()
    labels = a.label or [os.path.basename(os.path.dirname(c)) for c in a.ckpt]

    ds = LatCHDataset(a.latent_dir, target_feature=a.feature, target_source="npz")   # train_latch's default source
    _, va_idx = split_indices(ds.group_keys(a.val_group_by), a.val_frac, seed=a.val_seed)
    loader = DataLoader(Subset(ds, va_idx), batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                        collate_fn=collate_varlen)
    heads = [load_head(c, a.depth, a.dim, a.num_heads) for c in a.ckpt]
    ts = [(i + 0.5) / a.bins for i in range(a.bins)]
    # per model, per bin: sum of masked squared error and mask weight; plus the constant-0 predictor (= target mean)
    se = torch.zeros(len(heads), a.bins, dtype=torch.float64)
    se0 = torch.zeros(a.bins, dtype=torch.float64)
    w = 0.0
    with torch.no_grad():
        for bi, b in enumerate(loader):
            lat, tgt = b["latents"].float(), b["targets"].float()
            mask = (b["mask"].float() * b["weight"].float()).unsqueeze(1).expand_as(tgt)
            g = torch.Generator().manual_seed(1000003 * a.val_seed + bi)
            noise = torch.randn(lat.shape, generator=g)
            for (m, ck), hi in zip(heads, range(len(heads))):
                tz = (tgt - ck["std_mean"]) / ck["std_std"] if ck.get("standardized") else tgt
                for k, t in enumerate(ts):
                    tt = torch.full((lat.shape[0],), t)
                    pred = m(forward_noise(lat, noise, tt), tt).float()
                    se[hi, k] += float((((pred - tz) ** 2) * mask).sum())
                    if hi == 0:
                        se0[k] += float(((tz ** 2) * mask).sum())
            w += float(mask.sum())
            print(f"  batch {bi + 1}/{len(loader)}", end="\r", flush=True)
    loss = se / w
    loss0 = se0 / w
    print()
    hdr = "t-bin   const(=target var)  " + "  ".join(f"{l:>12}" for l in labels)
    if len(heads) > 1:
        hdr += "   gap(" + labels[1] + "-" + labels[0] + ")"
    print(hdr)
    for k, t in enumerate(ts):
        row = f"{t:5.2f}   {float(loss0[k]):18.4f}  " + "  ".join(f"{float(loss[h, k]):12.4f}" for h in range(len(heads)))
        if len(heads) > 1:
            row += f"   {float(loss[1, k] - loss[0, k]):+.4f}"
        print(row)
    mean = loss.mean(1)
    print(f"mean    {float(loss0.mean()):18.4f}  " + "  ".join(f"{float(x):12.4f}" for x in mean))
    for h, l in enumerate(labels):
        share_hi = float(loss[h, ts.index(next(t for t in ts if t > 0.7)):].sum() / loss[h].sum())
        print(f"{l}: share of the val loss from t > 0.7: {share_hi:.0%}")
    if a.out:
        json.dump({"t": ts, "labels": labels, "const": loss0.tolist(), "loss": loss.tolist(), "ckpts": a.ckpt,
                   "n_val_crops": len(va_idx)}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

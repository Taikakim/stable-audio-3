#!/usr/bin/env python
# Owner: KUANG (created 2026-10-17)
# Last edited: WINTERMUTE 2026-10-21 -- mean_grad backprops per batch (holding all batch graphs OOMed at T=4096 on 16 GB)
"""Held-out spectral probe for a LatCH head (SAO docs/modular-opt-test-plan-2026-10-17.md section P).

Method after SAMuon (arXiv 2608.25990 section 4), adapted to our optimizer, precision and batch:
take the head's pre-polar momentum per spectral matrix, M = U S V^T. For each probe, every spectral matrix j
contributes  P_j = -s_j * sum_{i in probe} u_i v_i^T  (s_j = max(1, sqrt(d_out/d_in)), the per-matrix size our
row_unit NorMuon gives the full polar update, so alpha = 1 means "the step the optimizer takes at this lr").
On HELD-OUT batches, with the diffusion time and noise FIXED per batch, the loss L(theta + alpha*lr*P) is
evaluated on a grid of alpha; best alpha per probe = the step the landscape tolerates along that direction.
Also reported: the local-quadratic estimate (held-out gradient + one finite difference) for comparison with
the paper, and direction-resolved agreement of two independent training-gradient groups (A, B) on the same
fixed basis (stable_audio_tools.training.modular_opt.direction_agreement).

Probes: single ranks (--ranks), bands (r1, r2-4, r5-16, r17-32), "full" = the whole polar direction, and
"beyond_k" = full minus the top-k components (the residual bulk, labelled as such).

Point semantics (--point):
  y      the gradient point y = (1 - sf_beta) z + sf_beta x from the checkpoint's optimizer state
         (train_latch --save-opt-state); the momentum is the run's own. PRIMARY.
  x      the Schedule-Free average x from the same state (the deployable weights). SECONDARY: a perturbation of
         x is not the same experiment as one of y; read it separately.
  saved  the checkpoint's state_dict (EMA or x) when no optimizer state was saved; the momentum is then REBUILT
         as a bias-corrected EMA (--beta1) of gradients from a third group C of training batches, disjoint from
         A, B and from the held-out set.
Nothing is mutated: every evaluation is a torch.func.functional_call on a copy of the parameters; the script
checks that alpha = 0 reproduces the reference loss and that the module's parameters are unchanged at the end.

  python scripts/latch/spectral_probe.py --ckpt run/latch_sa3_onset_envelope_ts_best.pt --latent-dir ... \
      --feature onset_envelope_ts --val-frac 0.05 --val-group-by relpath --point y --out probe.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.latch.latch_model import LatCH  # noqa: E402
from scripts.latch.train_latch import forward_noise, masked_mse, masked_smooth_l1  # noqa: E402

DEFAULT_RANKS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
DEFAULT_BANDS = ((1, 1), (2, 4), (5, 16), (17, 32))
DEFAULT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


# ----------------------------------------------------------------------------------------------------------
# core (no data loading; unit-tested on synthetic tensors)
# ----------------------------------------------------------------------------------------------------------
def make_draws(batches, seed):
    """One fixed (t, noise) per batch, reused by every probe and every alpha."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for b in batches:
        lat = b["latents"]
        out.append((torch.rand(lat.shape[0], generator=g), torch.randn(lat.shape, generator=g)))
    return out


def batch_loss(model, params, batch, draw, criterion, std=None):
    lat, tgt = batch["latents"], batch["targets"]
    mask = batch["mask"].float() * batch.get("weight", torch.ones_like(batch["mask"].float()))
    if std is not None:
        tgt = (tgt - std[0]) / std[1]
    t, noise = draw
    dev = lat.device
    z_t = forward_noise(lat, noise.to(dev), t.to(dev))
    pred = torch.func.functional_call(model, params, (z_t, t.to(dev)), strict=False)
    return criterion(pred.float(), tgt, mask)


def mean_loss(model, params, batches, draws, criterion, std=None):
    with torch.no_grad():
        return float(sum(batch_loss(model, params, b, d, criterion, std) for b, d in zip(batches, draws)) / len(batches))


def mean_grad(model, params, names, batches, draws, criterion, std=None):
    """Mean gradient over `batches` with respect to params[names], at the point `params`."""
    leaf = {k: (v.detach().clone().requires_grad_(k in names)) for k, v in params.items()}
    acc = None
    for b, d in zip(batches, draws):   # one graph at a time: summing all batches first kept 8 graphs alive (OOM on 16 GB)
        gs = torch.autograd.grad(batch_loss(model, leaf, b, d, criterion, std), [leaf[n] for n in names])
        acc = [g.detach() for g in gs] if acc is None else [a + g.detach() for a, g in zip(acc, gs)]
    return {n: g / len(batches) for n, g in zip(names, acc)}


def spectral_bases(momenta):
    return {n: torch.linalg.svd(M.float(), full_matrices=False) for n, M in momenta.items()}


def probe_direction(bases, ranks=None, beyond_k=None):
    """{name: P_j}. ranks: iterable of 1-based ranks; beyond_k: everything after the top k (full if 0)."""
    out = {}
    for n, (U, S, Vh) in bases.items():
        m, nn_ = U.shape[0], Vh.shape[1]
        s = max(1.0, math.sqrt(m / nn_))
        # Newton-Schulz maps (numerically) zero singular values to zero, so do the same: a zero momentum
        # (e.g. behind a zero-initialised adaLN gate) or a rank-deficient tail contributes no direction.
        live = int((S > 1e-6 * S[0]).sum()) if S.numel() and float(S[0]) > 0 else 0
        r = live
        if beyond_k is not None:
            idx = list(range(min(beyond_k, r), r))
        else:
            idx = [i - 1 for i in ranks if i - 1 < r]
        if not idx:
            out[n] = torch.zeros(m, nn_)
            continue
        Ui, Vi = U[:, idx], Vh[idx]
        out[n] = -s * (Ui @ Vi)
    return out


def sweep(model, point, P, lr, grid, eval_sets, criterion, std=None):
    """Loss on each held-out set for theta + alpha*lr*P, alpha in grid. Returns per-set lists."""
    res = []
    for batches, draws in eval_sets:
        row = []
        for a in grid:
            pp = dict(point)
            for n, d in P.items():
                pp[n] = point[n] + (a * lr) * d.to(point[n].dtype)
            row.append(mean_loss(model, pp, batches, draws, criterion, std))
        res.append(row)
    return res


def lqa(model, point, P, lr, eval_set, criterion, std=None, delta=0.5):
    """SAMuon eq. (6)-(7): alpha* = -g.P / (P H P) with g the held-out gradient, curvature by finite difference."""
    batches, draws = eval_set
    names = list(P.keys())
    g = mean_grad(model, point, names, batches, draws, criterion, std)
    gp = sum(float((g[n] * P[n].to(g[n].dtype)).sum()) for n in names) * lr
    L0 = mean_loss(model, point, batches, draws, criterion, std)
    pp = dict(point)
    for n in names:
        pp[n] = point[n] + (delta * lr) * P[n].to(point[n].dtype)
    Ld = mean_loss(model, pp, batches, draws, criterion, std)
    curv = 2.0 * (Ld - L0 - delta * gp) / (delta * delta)
    return {"g_dot_p": gp, "curvature": curv, "descent": gp < 0, "alpha_lqa": (-gp / curv) if curv > 0 else float("nan")}


def run_probe(model, point, spectral_names, momenta, lr, eval_sets, criterion, std=None, grad_groups=None,
              ranks=DEFAULT_RANKS, bands=DEFAULT_BANDS, k_resid=32, grid=DEFAULT_GRID):
    from stable_audio_tools.training.modular_opt import direction_agreement
    ref = [mean_loss(model, point, b, d, criterion, std) for b, d in eval_sets]
    bases = spectral_bases(momenta)
    probes = {f"rank{r}": probe_direction(bases, ranks=[r]) for r in ranks}
    probes.update({f"band{lo}-{hi}": probe_direction(bases, ranks=range(lo, hi + 1)) for lo, hi in bands})
    probes["full"] = probe_direction(bases, beyond_k=0)
    probes[f"beyond_{k_resid}"] = probe_direction(bases, beyond_k=k_resid)
    out = {"reference_loss": ref, "grid": list(grid), "lr": lr, "probes": {}}
    for name, P in probes.items():
        losses = sweep(model, point, P, lr, grid, eval_sets, criterion, std)
        for row, r0 in zip(losses, ref):
            if abs(row[grid.index(0.0)] - r0) > 1e-6 * max(1.0, abs(r0)):
                raise RuntimeError(f"alpha=0 does not reproduce the reference loss ({row[0]} vs {r0})")
        best = [grid[min(range(len(row)), key=row.__getitem__)] for row in losses]
        out["probes"][name] = {"losses": losses, "best_alpha": best,
                               "lqa": lqa(model, point, P, lr, eval_sets[0], criterion, std)}
    if grad_groups is not None:
        (bA, dA), (bB, dB) = grad_groups
        gA = mean_grad(model, point, spectral_names, bA, dA, criterion, std)
        gB = mean_grad(model, point, spectral_names, bB, dB, criterion, std)
        agree = {}
        for n in spectral_names:
            U, S, Vh = bases[n]
            k = min(k_resid, S.numel())
            agree[n] = direction_agreement(gA[n], gB[n], {"U": U[:, :k], "S": S, "V": Vh[:k].T, "k": k},
                                           bands=tuple((lo - 1, hi) for lo, hi in bands))
        out["direction_agreement"] = agree
    return out


def point_from_state(model, groups, opt_state, which, spectral):
    """(point, momenta, lr) from a saved ModularOptimizer state. which = "y": the gradient point
    (1 - sf_beta) z + sf_beta x; "x": the Schedule-Free average. Parameters without Schedule-Free state keep
    the module's value. The optimizer is rebuilt on the same routing and only read."""
    from stable_audio_tools.training.modular_opt import ModularOptimizer
    opt = ModularOptimizer(groups, lr=1e-2, strict_group_lrs=False)  # analysis probe, not a run
    opt.load_state_dict(opt_state)
    point = {n: p.detach().clone() for n, p in model.named_parameters()}
    for g in opt.param_groups:
        for n, p in zip(g["param_names"], g["params"]):
            st = opt.state.get(p, {})
            if "z" in st:
                v = st["x"] if which == "x" else (1 - g["sf_beta"]) * st["z"] + g["sf_beta"] * st["x"]
                point[n] = v.to(p.device, p.dtype)
    momenta = {n: opt.state[p]["momentum"].float() for g in opt.param_groups
               for n, p in zip(g["param_names"], g["params"]) if n in spectral and "momentum" in opt.state.get(p, {})}
    return point, momenta, opt.param_groups[0]["lr"]


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------
def _infer_arch(sd):
    dim = sd["latent_proj.weight"].shape[0]
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    return dim, depth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latent-dir", required=True)
    ap.add_argument("--feature", required=True)
    ap.add_argument("--db-path", default=None)
    ap.add_argument("--target-source", default="npz", help="as train_latch (default npz)")
    ap.add_argument("--val-frac", type=float, default=None, help="default: the checkpoint's own val_split")
    ap.add_argument("--val-seed", type=int, default=None, help="default: the checkpoint's own val_split")
    ap.add_argument("--val-group-by", default=None, help="default: the checkpoint's own val_split")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-eval-batches", type=int, default=4, help="held-out batches per evaluation set")
    ap.add_argument("--repeats", type=int, default=2, help="independent evaluation sets (other batches + draws)")
    ap.add_argument("--n-grad-batches", type=int, default=8, help="training batches per agreement group (A, B)")
    ap.add_argument("--n-mom-batches", type=int, default=20, help="group C size when the momentum is rebuilt")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=None, help="default: the checkpoint's optimizer lr, else 1e-2")
    ap.add_argument("--point", choices=["y", "x", "saved"], default="y")
    ap.add_argument("--spectral-all", action="store_true", help="the run used --mod-muon-all")
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--t-injection", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from torch.utils.data import DataLoader, Subset
    from scripts.latch.latch_dataset import LatCHDataset, collate_varlen
    from scripts.latch.train_latch import split_indices
    from stable_audio_tools.training.modular_opt import build_modular_param_groups

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    arch = ck.get("arch") or {}
    dim, depth = _infer_arch(sd)
    model = LatCH(in_channels=256, out_channels=ck["out_channels"], dim=arch.get("dim", dim),
                  depth=arch.get("depth", depth), num_heads=a.num_heads or arch.get("num_heads", 8),
                  t_injection=a.t_injection or ck.get("t_injection", "adaln_zero")).to(dev)
    model.load_state_dict(sd)
    model.eval()
    groups = build_modular_param_groups(model, spectral_all=a.spectral_all)
    spectral = [n for g in groups if g["group_type"] == "spectral" for n, p in zip(g["param_names"], g["params"]) if p.ndim == 2]
    loss_type = ck.get("loss_type", "mse")
    criterion = masked_smooth_l1 if loss_type == "smooth_l1" else masked_mse
    std = (ck["std_mean"], ck["std_std"]) if ck.get("standardized") else None

    ds = LatCHDataset(a.latent_dir, target_feature=a.feature, db_path=a.db_path, target_source=a.target_source)
    vs = ck.get("val_split") or {}
    if not vs and a.val_frac is None:
        raise SystemExit("checkpoint has no val_split (trained without a held-out set): pass --val-frac/--val-seed/"
                         "--val-group-by explicitly and note that the probe set then overlaps training data")
    vfrac = a.val_frac if a.val_frac is not None else vs["val_frac"]
    vseed = a.val_seed if a.val_seed is not None else vs.get("val_seed", 0)
    vgroup = a.val_group_by if a.val_group_by is not None else vs.get("val_group_by", "source_track")
    tr, va = split_indices(ds.group_keys(vgroup), vfrac, seed=vseed)
    gen = torch.Generator().manual_seed(a.seed)
    tr = [tr[i] for i in torch.randperm(len(tr), generator=gen).tolist()]
    va = [va[i] for i in torch.randperm(len(va), generator=gen).tolist()]

    def take(idx, n_batches):
        dl = DataLoader(Subset(ds, idx), batch_size=a.batch_size, shuffle=False, drop_last=True, collate_fn=collate_varlen)
        out = []
        for b in dl:
            out.append({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()})
            if len(out) == n_batches:
                break
        return out

    bs = a.batch_size
    nA = a.n_grad_batches * bs
    gA, gB = take(tr[:nA], a.n_grad_batches), take(tr[nA:2 * nA], a.n_grad_batches)
    eval_sets = []
    for r in range(a.repeats):
        e = take(va[r * a.n_eval_batches * bs:(r + 1) * a.n_eval_batches * bs], a.n_eval_batches)
        eval_sets.append((e, make_draws(e, a.seed + 1000 + r)))

    point = {n: p.detach().clone() for n, p in model.named_parameters()}
    lr = a.lr
    if a.point in ("y", "x"):
        if "optimizer_state" not in ck:
            raise SystemExit("--point y/x needs a checkpoint written with train_latch --save-opt-state; use --point saved")
        point, momenta, ck_lr = point_from_state(model, groups, ck["optimizer_state"], a.point, spectral)
        lr = lr or ck_lr
        mom_source = "optimizer_state"
    else:
        lr = lr or 1e-2
        gC = take(tr[2 * nA:2 * nA + a.n_mom_batches * bs], a.n_mom_batches)
        dC = make_draws(gC, a.seed + 7)
        momenta = {n: torch.zeros_like(point[n]) for n in spectral}
        for i, (b, d) in enumerate(zip(gC, dC)):
            g = mean_grad(model, point, spectral, [b], [d], criterion, std)
            for n in spectral:
                momenta[n].mul_(a.beta1).add_(g[n], alpha=1 - a.beta1)
        bc = 1 - a.beta1 ** len(gC)
        momenta = {n: m / bc for n, m in momenta.items()}
        mom_source = f"rebuilt from {len(gC)} training batches (group C)"
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    res = run_probe(model, point, spectral, momenta, lr, eval_sets, criterion, std,
                    grad_groups=((gA, make_draws(gA, a.seed + 1)), (gB, make_draws(gB, a.seed + 2))))
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters()), "module parameters changed"
    res["meta"] = {"ckpt": a.ckpt, "point": a.point, "momentum_source": mom_source, "lr": lr,
                   "batch_size": bs, "n_eval_batches": a.n_eval_batches, "repeats": a.repeats,
                   "spectral_matrices": spectral, "loss_type": loss_type, "seed": a.seed,
                   "val_split": {"val_frac": vfrac, "val_seed": vseed, "val_group_by": vgroup}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for name, pr in res["probes"].items():
        print(f"{name:12s} best alpha {pr['best_alpha']}  lqa {pr['lqa']['alpha_lqa']:.3g}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

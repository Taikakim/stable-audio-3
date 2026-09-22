"""Measure the WIDE-side gradient covariance spectrum, before approximating it.

CONTINUITY, 2026-09-22.

THE QUESTION. Mousse whitens with Kronecker factors L = EMA(G G^T) and R = EMA(G^T G).
On a LoRA factor of shape (12288, 128), R is 128x128 and free; L is 12288x12288, which
measured 576 MiB and a 7.23 s eigendecomposition on this card. Every proposal for keeping
the wide side -- Frequent-Directions sketching, block-diagonal chunking, Nystrom -- is an
APPROXIMATION of L, and which one is right depends entirely on a property nobody has
measured: does L's spectrum actually decay?

  - top-k dominates      -> a rank-k sketch (Sketchy/EW-FD) is justified
  - spectrum is flat     -> low rank CANNOT work; blocks are the only option
  - L adds little over R -> the wide side is not paying rent; stay one-sided and stop

Choosing between approximations of an unmeasured object is how the modelled trajectory
table happened. This measures it first.

HOW, without ever forming a 12288x12288 matrix. L is a weighted sum of rank-128 outer
products, so stack the weighted gradients as columns:

    H = [ sqrt(w_1) G_1 , ... , sqrt(w_s) G_s ]   in R^(12288 x 128s),  w_i = beta^(s-i)

Then L = H H^T exactly (for the truncated horizon), and the nonzero eigenvalues of H H^T
are those of the much smaller H^T H, which is (128s x 128s). No wide matrix is built and
the spectrum is EXACT for the horizon covered, not a sketch of it.

The snapshots live on CPU (6 MiB each), so this costs no VRAM during training.

USE: add SA3_COV_PROBE=1 to a normal training run. It writes a JSON next to the run and
prints the verdict at the end.
"""

import json
import math
import os
from pathlib import Path

import torch
import pytorch_lightning as pl


class WideCovarianceProbe(pl.Callback):
    """Capture gradient snapshots for the widest LoRA factors, then report the spectrum."""

    def __init__(self, n_snapshots: int = 64, every: int = 4, beta: float = 0.95,
                 out_dir: str = ".", max_tensors: int = 3):
        self.n_snapshots = n_snapshots
        self.every = every
        self.beta = beta
        self.out_dir = Path(out_dir)
        self.max_tensors = max_tensors
        self.targets: list[str] = []
        self.snaps: dict[str, list[torch.Tensor]] = {}
        self.captured = 0
        self.done = False

    def _pick_targets(self, pl_module):
        """Widest 2-D LoRA factors first -- they are the ones we cannot afford exactly."""
        cands = [(n, tuple(p.shape)) for n, p in pl_module.named_parameters()
                 if p.requires_grad and p.ndim == 2 and ("lora_B" in n or "lora_A" in n)]
        cands.sort(key=lambda kv: -max(kv[1]))
        self.targets = [n for n, _ in cands[:self.max_tensors]]
        for n in self.targets:
            self.snaps[n] = []
        print(f"[COV PROBE] watching {len(self.targets)} tensors, "
              f"{self.n_snapshots} snapshots every {self.every} steps:", flush=True)
        for n, s in cands[:self.max_tensors]:
            print(f"[COV PROBE]   {s}  {n}", flush=True)

    def on_after_backward(self, trainer, pl_module):
        if self.done:
            return
        if not self.targets:
            self._pick_targets(pl_module)
            if not self.targets:
                print("[COV PROBE] no 2-D LoRA factors found; probe inert.", flush=True)
                self.done = True
                return
        if trainer.global_step % self.every:
            return
        named = dict(pl_module.named_parameters())
        for n in self.targets:
            g = named[n].grad
            if g is None:
                return
            self.snaps[n].append(g.detach().float().cpu().clone())
        self.captured += 1
        if self.captured >= self.n_snapshots:
            self.done = True
            print(f"[COV PROBE] captured {self.captured} snapshots; analysing.", flush=True)
            self.analyse()

    def on_train_end(self, trainer, pl_module):
        if not self.done and self.captured >= 8:
            print(f"[COV PROBE] run ended early with {self.captured} snapshots; analysing.", flush=True)
            self.analyse()

    def analyse(self):
        report = {}
        for name in self.targets:
            snaps = self.snaps.get(name) or []
            if len(snaps) < 4:
                continue
            s = len(snaps)
            # EMA weights: the most recent snapshot carries weight 1.
            w = [self.beta ** (self.every * (s - 1 - i)) for i in range(s)]
            H = torch.cat([snaps[i] * math.sqrt(w[i]) for i in range(s)], dim=1)
            d_wide, d_cols = H.shape
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                Hd = H.to(dev)                # ONE copy, not two
                gram = (Hd.t() @ Hd).cpu()
                del Hd
                if dev == "cuda":
                    torch.cuda.empty_cache()
            except RuntimeError:
                gram = H.t() @ H              # CPU fallback if VRAM is tight mid-run
            gram = 0.5 * (gram + gram.t())
            eig = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0)

            total = float(eig.sum())
            cum = torch.cumsum(eig, 0) / max(total, 1e-30)
            marks = [k for k in (32, 64, 128, 256, 512, 1024, 2048) if k <= eig.numel()]
            mass = {k: float(cum[k - 1]) for k in marks}
            p = eig / max(total, 1e-30)
            p = p[p > 0]
            eff_rank = float(torch.exp(-(p * p.log()).sum()))

            # The narrow side, for contrast: it is what one-sided preconditioning keeps.
            R = sum((snaps[i] * math.sqrt(w[i])).t() @ (snaps[i] * math.sqrt(w[i]))
                    for i in range(s))
            r_eig = torch.linalg.eigvalsh(0.5 * (R + R.t())).flip(0).clamp_min(0)
            r_p = r_eig / max(float(r_eig.sum()), 1e-30)
            r_p = r_p[r_p > 0]
            r_eff = float(torch.exp(-(r_p * r_p.log()).sum()))

            report[name] = {
                "shape": list(snaps[0].shape), "snapshots": s, "every": self.every,
                "beta": self.beta, "horizon_steps": self.every * s,
                "wide_dim": d_wide, "rank_available": int(eig.numel()),
                "wide_effective_rank": eff_rank, "narrow_effective_rank": r_eff,
                "cumulative_mass": mass,
                "top_eigs": [float(x) for x in eig[:8]],
                "eig_ratio_1_to_128": float(eig[0] / eig[min(127, eig.numel()-1)].clamp_min(1e-30)),
            }
            print(f"\n[COV PROBE] {name}  shape={tuple(snaps[0].shape)}  "
                  f"horizon={self.every*s} steps", flush=True)
            print(f"[COV PROBE]   wide side {d_wide}: effective rank {eff_rank:.1f} "
                  f"of {eig.numel()} available", flush=True)
            print(f"[COV PROBE]   narrow side {snaps[0].shape[1]}: effective rank {r_eff:.1f}", flush=True)
            for k in marks:
                print(f"[COV PROBE]   top-{k:<5d} captures {mass[k]*100:6.2f}% of variance", flush=True)
            v = self._verdict(mass, eff_rank)
            report[name]["verdict"] = v
            print(f"[COV PROBE]   => {v}", flush=True)

        if report:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            out = self.out_dir / "wide_covariance_spectrum.json"
            out.write_text(json.dumps(report, indent=2))
            print(f"\n[COV PROBE] written: {out}\n", flush=True)

    @staticmethod
    def _verdict(mass, eff_rank):
        m256 = mass.get(256, mass.get(128, 0.0))
        if m256 >= 0.90:
            return (f"SKETCH IS JUSTIFIED: top-256 holds {m256*100:.1f}% of the variance, so a "
                    f"rank-128/256 EW-FD sketch can carry the wide side at ~12 MiB.")
        if m256 >= 0.70:
            return (f"MARGINAL: top-256 holds {m256*100:.1f}%. A sketch keeps most but not all; "
                    f"consider block-diagonal plus a low-rank correction.")
        return (f"SPECTRUM IS FLAT: top-256 holds only {m256*100:.1f}% (effective rank "
                f"{eff_rank:.0f}). Low-rank CANNOT represent this -- use blocks, or stay "
                f"one-sided and spend the effort elsewhere.")


def maybe_build(out_dir: str = ".", args=None):
    """Armed by --cov-probe, or by SA3_COV_PROBE=1 for env-driven callers.

    The flag exists because the env route is quietly fragile: writing
    `SA3_COV_PROBE=1 && python ...` without `export` sets a SHELL variable, which the
    child process never sees, so the probe stays inert and the run produces no data with
    no error. That cost a real run on 2026-09-22.
    """
    flag = bool(getattr(args, "cov_probe", False)) if args is not None else False
    env = os.environ.get("SA3_COV_PROBE", "") in ("1", "true", "yes")
    if not (flag or env):
        return None
    n = int(getattr(args, "cov_probe_snaps", 0) or os.environ.get("SA3_COV_PROBE_SNAPS", "64"))
    every = int(getattr(args, "cov_probe_every", 0) or os.environ.get("SA3_COV_PROBE_EVERY", "4"))
    print(f"[COV PROBE] armed: {n} snapshots every {every} steps (CPU-resident, no VRAM cost).",
          flush=True)
    return WideCovarianceProbe(n_snapshots=n, every=every, out_dir=out_dir)

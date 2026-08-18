"""trajectory_sketch.py — record a training run's weight-space trajectory at STEP resolution in a
form small enough to keep every step for 10k+ steps, plus full adapter checkpoints on a coarse grid.

WHY (Kim, 2026-08-18): "train a small rank 8/16 adapter and save every step for 1000 or even
10,000 steps and see how the trajectory really is. And I bet there's a compressed way to store
movement like this." There is. Everything we read off a trajectory — successive-step cosines,
path efficiency at every window length, update/gradient autocorrelation, gradient signal-to-noise,
PCA of the path — is a function of INNER PRODUCTS between per-step vectors, i.e. of the Gram
matrix, never of individual coordinates. A CountSketch (Charikar et al.: hash each coordinate to
one of k buckets with a random sign; here s independent hashes concatenated) is a linear map that
preserves every inner product to ~1/sqrt(s·k). At s·k = 4096 that is ~1.6 % on a cosine — far
below any effect we care about — and turns a 12 M-parameter update into a 4096-float row. Ten
thousand steps of update AND gradient sketches = 330 MB. The rank-16 adapter itself is 46 MB per
checkpoint, so full checkpoints go on a coarse grid (default every 100 steps) for the questions
that need the actual matrices (singular spectra, ΔW_eff, repair experiments).

WHAT IS RECORDED PER OPTIMIZER STEP (rows aligned across files, row t = t-th optimizer step)
  upd_sketch.npy   [T, s·k] f32   sketch of  W_t − W_{t−1}  (the update the optimizer applied)
  grad_sketch.npy  [T, s·k] f32   sketch of  ∇L at step t   (raw, before clipping)
  upd_norms.npy    [T, n_tensors] f32   ‖update‖ per trainable tensor  (WHERE the walk happens)
  grad_norms.npy   [T, n_tensors] f32   ‖grad‖   per trainable tensor
  upd_sub.npy      [T, n_sub] f16       update on a fixed random coordinate subset (raw traces;
  grad_sub.npy     [T, n_sub] f16        an unbiased Gram estimator too, heavier-tailed)
  scalars.npy      [T, 4] f64           global_step, loss, lr, wall-clock
  params.json                           tensor names / numels / offsets, sketch seed & dims
  ckpt/step{N}.pt                       trainable tensors (bf16) + lora_config, every ckpt_every
The sketch seed is FIXED and stored, so sketches from different runs (bs1 vs bs8, AdamW vs
fusion) live in the same projected space and are directly comparable — a cross-run Gram at
step resolution needs no re-projection.

WHAT THE UPDATE IS. W_t − W_{t−1} of every requires_grad parameter, measured around
optimizer.step() (snapshot in on_before_optimizer_step, delta after the step). Under gradient
accumulation this is once per OPTIMIZER step, not per micro-batch. Under DDP only rank 0 writes
(grads are all-reduced before the step, so every rank sees the same update).

WIRING: train_lora.py --traj-sketch-dir DIR [--traj-sketch-k 1024 --traj-sketch-hashes 4
        --traj-subsample 16384 --traj-ckpt-every 100]
COST: two scatter_add passes over the flat parameter vector per step (~ms on GPU) and one
      46 MB param snapshot; negligible next to a DiT forward/backward.
"""
import json
import os
import time

import numpy as np
import torch

try:
    import pytorch_lightning as pl
except ImportError:                                  # allow importing CountSketch without PL
    class _Stub:                                     # pragma: no cover
        Callback = object
    pl = _Stub()


class CountSketch:
    """Linear map R^D -> R^{s·k}: s independent (bucket, sign) hashings, concatenated and scaled
    by 1/sqrt(s), so <sk(x), sk(y)> is an unbiased estimate of <x, y> with variance ~ 1/(s·k)."""

    def __init__(self, D, k=1024, hashes=4, seed=0, device="cpu"):
        self.D, self.k, self.s = int(D), int(k), int(hashes)
        g = torch.Generator().manual_seed(int(seed))
        idx = torch.randint(0, self.k, (self.s, self.D), generator=g, dtype=torch.int64)
        sgn = torch.randint(0, 2, (self.s, self.D), generator=g, dtype=torch.int8) * 2 - 1
        # bucket offset per hash so one scatter_add fills the concatenated vector
        self.idx = (idx + torch.arange(self.s).unsqueeze(1) * self.k).to(device)
        self.sgn = sgn.to(device)
        self.scale = 1.0 / (self.s ** 0.5)
        self.device = device

    def __call__(self, x):
        x = x.to(self.device, torch.float32).reshape(-1)
        assert x.numel() == self.D, (x.numel(), self.D)
        out = torch.zeros(self.s * self.k, dtype=torch.float32, device=self.device)
        for h in range(self.s):
            out.scatter_add_(0, self.idx[h], self.sgn[h].to(torch.float32) * x)
        return out * self.scale


class TrajectorySketch(pl.Callback):
    def __init__(self, out_dir, max_steps, k=1024, hashes=4, subsample=16384, ckpt_every=100,
                 seed=1234, lora_config=None, ckpt_dense_until=0):
        super().__init__()
        self.out_dir = out_dir
        self.max_steps = int(max_steps)
        self.k, self.s, self.n_sub, self.ckpt_every, self.seed = k, hashes, subsample, ckpt_every, seed
        # every step until this global_step, then every ckpt_every (Kim 2026-08-18: local runs may
        # use ~80% of the NVMe's free space; the spike forms in the first few hundred steps and
        # that is where per-step spectra are worth the disk)
        self.ckpt_dense_until = int(ckpt_dense_until)
        self.lora_config = lora_config or {}
        self._params = None
        self._prev = None
        self._last_gs = -1
        self._row = 0
        self._t0 = time.time()
        self._grad_row = None

    # -- helpers ------------------------------------------------------------------------------
    def _flat(self, tensors):
        return torch.cat([t.detach().reshape(-1).to(torch.float32) for t in tensors])

    def _per_tensor_norms(self, flat):
        return torch.stack([flat[o:o + n].norm() for o, n in zip(self._offsets, self._numels)])

    # -- lightning hooks ----------------------------------------------------------------------
    def on_train_start(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        named = [(n, p) for n, p in pl_module.named_parameters() if p.requires_grad]
        named.sort(key=lambda t: t[0])
        self._params = named
        self._numels = [p.numel() for _, p in named]
        self._offsets = np.concatenate([[0], np.cumsum(self._numels)[:-1]]).tolist()
        D = int(sum(self._numels))
        dev = named[0][1].device
        self.sketch = CountSketch(D, self.k, self.s, self.seed, device=dev)
        g = torch.Generator().manual_seed(self.seed + 1)
        self._sub_idx = torch.randperm(D, generator=g)[: self.n_sub].sort().values.to(dev)
        os.makedirs(os.path.join(self.out_dir, "ckpt"), exist_ok=True)
        T, SK = self.max_steps, self.s * self.k
        mm = lambda name, shape, dt: np.lib.format.open_memmap(
            os.path.join(self.out_dir, name), mode="w+", dtype=dt, shape=shape)
        self.f_upd = mm("upd_sketch.npy", (T, SK), np.float32)
        self.f_grad = mm("grad_sketch.npy", (T, SK), np.float32)
        self.f_upd_n = mm("upd_norms.npy", (T, len(named)), np.float32)
        self.f_grad_n = mm("grad_norms.npy", (T, len(named)), np.float32)
        self.f_upd_s = mm("upd_sub.npy", (T, self.n_sub), np.float16)
        self.f_grad_s = mm("grad_sub.npy", (T, self.n_sub), np.float16)
        self.f_scal = mm("scalars.npy", (T, 4), np.float64)
        json.dump({"names": [n for n, _ in named], "numels": self._numels, "offsets": self._offsets,
                   "D": D, "sketch_k": self.k, "sketch_hashes": self.s, "sketch_dim": SK,
                   "sketch_seed": self.seed, "subsample_seed": self.seed + 1, "n_sub": self.n_sub,
                   "max_steps": T, "ckpt_every": self.ckpt_every, "ckpt_dense_until": self.ckpt_dense_until,
                   "lora_config": self.lora_config,
                   "scalars_cols": ["global_step", "loss", "lr", "wall_s"],
                   "note": "row t = t-th optimizer step; sketch is a CountSketch, inner products "
                           "of rows estimate inner products of the true vectors"},
                  open(os.path.join(self.out_dir, "params.json"), "w"), indent=1)
        print(f"[traj] sketching {D:,} trainable params in {len(named)} tensors -> {SK}-d rows, "
              f"{T} steps max, ckpt every {self.ckpt_every} -> {self.out_dir}", flush=True)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self._params is None or self._row >= self.max_steps:
            return
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for _, p in self._params]
        g = self._flat(grads)
        self._grad_row = (self.sketch(g).cpu().numpy(), self._per_tensor_norms(g).cpu().numpy(),
                          g[self._sub_idx].cpu().numpy().astype(np.float16))
        self._prev = self._flat([p for _, p in self._params])
        self._lr = float(optimizer.param_groups[0]["lr"])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._params is None or self._prev is None or self._row >= self.max_steps:
            return
        gs = int(trainer.global_step)
        if gs <= self._last_gs:
            return                                   # accumulation micro-batch, no step yet
        self._last_gs = gs
        cur = self._flat([p for _, p in self._params])
        upd = cur - self._prev
        self._prev = None
        r = self._row
        self.f_upd[r] = self.sketch(upd).cpu().numpy()
        self.f_upd_n[r] = self._per_tensor_norms(upd).cpu().numpy()
        self.f_upd_s[r] = upd[self._sub_idx].cpu().numpy().astype(np.float16)
        gsk, gn, gsub = self._grad_row
        self.f_grad[r] = gsk; self.f_grad_n[r] = gn; self.f_grad_s[r] = gsub
        loss = float("nan")
        try:
            loss = float(outputs["loss"]) if isinstance(outputs, dict) else float(outputs)
        except Exception:
            pass
        self.f_scal[r] = (gs, loss, self._lr, time.time() - self._t0)
        self._row += 1
        if (self.ckpt_every and gs % self.ckpt_every == 0) or gs <= self.ckpt_dense_until:
            sd = {n: p.detach().to(torch.bfloat16).cpu() for n, p in self._params}
            torch.save({"state_dict": sd, "lora_config": self.lora_config, "global_step": gs,
                        "traj_row": r}, os.path.join(self.out_dir, "ckpt", f"step{gs:06d}.pt"))
        if r % 200 == 0:
            for f in (self.f_upd, self.f_grad, self.f_upd_n, self.f_grad_n, self.f_upd_s,
                      self.f_grad_s, self.f_scal):
                f.flush()

    def on_train_end(self, trainer, pl_module):
        if self._params is None:
            return
        for f in (self.f_upd, self.f_grad, self.f_upd_n, self.f_grad_n, self.f_upd_s,
                  self.f_grad_s, self.f_scal):
            f.flush()
        json.dump({"rows_written": self._row}, open(os.path.join(self.out_dir, "done.json"), "w"))
        print(f"[traj] wrote {self._row} steps -> {self.out_dir}", flush=True)

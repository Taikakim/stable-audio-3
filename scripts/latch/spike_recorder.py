# Owner: WINTERMUTE (created 2026-10-21)
"""Gradient-spike recorder for train_latch.py (--flight-recorder, --skip-spike-ratio).

The DoRA trainer has had a flight recorder since 2026-09-25 (stable_audio_3/training/flight_recorder.py); 83 incidents
from 21 of its runs were read together on 2026-10-21: arms with different optimizers spiked at the same steps on the
same 16 files (so a spike is a property of the batch), and the highest-loss item had t > 0.85 in 39 of 48 spike batches
(9% of draws). That recorder logs each item's LOSS only, so it could not name the item behind the one 42x spike. This
one also measures each item's own GRADIENT NORM when a spike fires, and can skip the optimizer step.

Per optimizer step the trainer passes the raw (pre-clip) gradient norm. A step is an incident when the norm is
non-finite or >= ratio x the median of the last `window` accepted steps (after `warmup` steps). With skip_ratio > 0,
a step at >= skip_ratio x median (or non-finite) is skipped: no optimizer step, gradient dropped. Skipped norms do not
enter the median. Keep skip_ratio well above ratio: most spikes are high-t batches, legitimate but noisy, and skipping
them would under-train the high-noise end.

Each incident is one JSON line in <save-dir>/incidents.jsonl: step, norm, median, ratio, skipped, the previous steps'
norms and losses, the top parameters by gradient norm, and per item: file, t, loss and gradient norm, recomputed alone
(same file and t, fresh seeded noise, the weights just after the step). The batch's file list comes from the sampler,
in order; the dataset's fallback (an unreadable file silently replaced by the next one) is not visible here.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import deque

import torch
from torch.utils.data import BatchSampler


class RecordingBatchSampler(BatchSampler):
    """BatchSampler that remembers each batch's indices in order. The DataLoader samples indices in the main process
    and yields batches in that same order (workers only load), so popping one entry per consumed batch gives the
    indices of the batch in hand, even with prefetching. Same RNG draws as DataLoader(shuffle=True)."""

    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.queue = deque()

    def __iter__(self):
        self.queue.clear()
        for b in super().__iter__():
            self.queue.append(list(b))
            yield b


def total_grad_norm(params) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads])))


class SpikeRecorder:
    def __init__(self, out_path, ratio=4.0, skip_ratio=0.0, window=200, warmup=50, keep_prev=8):
        self.out_path = out_path
        self.ratio = float(ratio)
        self.skip_ratio = float(skip_ratio)
        self.window = int(window)
        self.warmup = int(warmup)
        self.hist = deque(maxlen=self.window)
        self.prev = deque(maxlen=keep_prev)
        self.n_incidents = 0
        self.n_skipped = 0

    def judge(self, norm: float):
        """-> (incident, skip, median, ratio). Call once per optimizer step, before clipping."""
        finite = math.isfinite(norm)
        med = statistics.median(self.hist) if self.hist else float("nan")
        ready = len(self.hist) >= self.warmup and med > 0
        r = norm / med if (ready and finite) else float("nan")
        incident = (not finite) or (ready and r >= self.ratio)
        skip = self.skip_ratio > 0 and ((not finite) or (ready and r >= self.skip_ratio))
        if not skip and finite:
            self.hist.append(norm)
        return incident, skip, med, r

    def note(self, step, norm, loss):
        self.prev.append({"step": int(step), "norm": norm, "loss": loss})

    def write(self, rec: dict):
        self.n_incidents += 1
        self.n_skipped += int(bool(rec.get("skipped")))
        rec["previous_steps"] = list(self.prev)
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def top_params(model, k=6):
    out = [(n, float(p.grad.detach().float().norm())) for n, p in model.named_parameters() if p.grad is not None]
    out.sort(key=lambda x: -x[1] if math.isfinite(x[1]) else -float("inf"))
    return [{"param": n, "grad_norm": g} for n, g in out[:k]]


def attribute_items(model, criterion, ds, ds_indices, ts, collate, device, use_bf16, forward_noise,
                    std_mean=None, std_std=None, seed=0):
    """Recompute each item alone: its loss and the norm of its own gradient. Leaves .grad zeroed."""
    rows = []
    gen = torch.Generator().manual_seed(int(seed))
    params = [p for p in model.parameters() if p.requires_grad]
    for i, t in zip(ds_indices, ts):
        b = collate([ds[i]])
        lat = b["latents"].to(device)
        tgt = b["targets"].to(device)
        mask = b["mask"].to(device).float() * b["weight"].to(device)
        if std_mean is not None:
            tgt = (tgt - std_mean) / std_std
        tt = torch.tensor([float(t)], device=device)
        noise = torch.randn(lat.shape, generator=gen).to(device)
        z_t = forward_noise(lat, noise, tt)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            pred = model(z_t, tt)
        loss = criterion(pred.float(), tgt, mask)
        loss.backward()
        rows.append({"index": int(i), "file": str(ds.items[i]), "t": float(t), "loss": float(loss.item()),
                     "grad_norm": total_grad_norm(params),
                     "latent_std": float(lat.std()), "frames": int(lat.shape[-1])})
    model.zero_grad(set_to_none=True)
    return rows

"""Raw-gradient telemetry + the outlier flight recorder (CONTINUITY, 2026-09-25).

WHY THE RAW GRADIENT. Our optimizers normalise the step (Muon/NS -> fixed spectral size,
sign steps -> +-lr, LoRA-TSD -> msign), so a pathological batch does NOT show up as a big
update: the update norm is set by lr and layer shapes. Where a spike still does damage is
the MOMENTUM buffer -- one outsized gradient dominates it for ~1/(1-beta) steps and steers
their direction. So the thing to watch is the gradient BEFORE clipping and normalisation.
(Critique of the P95 step-governor proposal: SAO/docs/PROPOSAL_ADAPTIVE_P95_STEP_GOVERNOR.md.)

WHAT IT DOES, per optimizer step (called from `on_before_optimizer_step`, i.e. after
backward and before Lightning's gradient clipping):
  * per-kind raw grad norms (lora_A / lora_B / magnitude / other / total) + the count of
    non-finite grad tensors -- one host sync;
  * a robust rolling z-score of log(total norm): median / MAD over the last `window`
    steps (log because grad norms are heavy-tailed; median/MAD so the spikes being hunted
    don't inflate the yardstick);
  * on a trigger (z > z_thresh after warmup, or any non-finite grad), dumps an incident:
    `<run_dir>/incidents/step_<N>/incident.json` + `batch.pt`.

This is DIAGNOSTIC ONLY -- it never changes a gradient or a step. Whether spikes exist at
all on our runs is the first question it answers; clipping comes after, if they do.

LIMITS, stated so nobody over-reads a dump: with accumulate_grad_batches > 1 the stashed
batch is the LAST micro-batch only, while the gradient is the accumulated one. The batch
holds latents + noise + t, which is enough to replay the loss for a given checkpoint --
but the replay needs the weights at that step (nearest checkpoint), which are not dumped.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from typing import Any, Iterable

import torch

KINDS = ("lora_A", "lora_B", "magnitude", "other")


def param_kind(name: str) -> str:
    if name.endswith("lora_A"):
        return "lora_A"
    if name.endswith("lora_B"):
        return "lora_B"
    if name.endswith("magnitude"):
        return "magnitude"
    return "other"


class GradTelemetry:
    """Per-kind raw grad norms for a fixed list of (name, param)."""

    def __init__(self, named_params: Iterable[tuple[str, torch.nn.Parameter]]):
        self.names: list[str] = []
        self.params: list[torch.nn.Parameter] = []
        for n, p in named_params:
            if p.requires_grad:
                self.names.append(n)
                self.params.append(p)
        self.kind_idx = {k: [i for i, n in enumerate(self.names) if param_kind(n) == k] for k in KINDS}

    @torch.no_grad()
    def measure(self) -> tuple[dict[str, float], list[float | None] | None]:
        """Return (metrics, per-param norms). Per-param norms come back as a host list so
        an incident can name the top contributors without a second pass. One sync."""
        live = [(i, p.grad) for i, p in enumerate(self.params) if p.grad is not None]
        if not live:
            return {}, None
        idx = [i for i, _ in live]
        norms = torch.stack(torch._foreach_norm([g.float() for _, g in live]))  # (L,)
        per: list[float | None] = [None] * len(self.params)  # None = no grad this step
        vals = norms.tolist()  # the sync
        for i, v in zip(idx, vals):
            per[i] = v
        out: dict[str, float] = {}
        tot_sq = 0.0
        n_bad = 0
        for k in KINDS:
            sq = 0.0
            for i in self.kind_idx[k]:
                v = per[i]
                if v is None:
                    continue
                if not math.isfinite(v):
                    n_bad += 1
                    continue
                sq += v * v
            if self.kind_idx[k]:
                out[f"grad/raw_norm_{k}"] = math.sqrt(sq)
            tot_sq += sq
        out["grad/raw_norm_total"] = math.sqrt(tot_sq)
        out["grad/nonfinite_tensors"] = float(n_bad)
        finite = [v for v in vals if math.isfinite(v)]
        out["grad/raw_norm_max_tensor"] = max(finite) if finite else float("nan")
        return out, per


class FlightRecorder:
    def __init__(self, run_dir: str, named_params, window: int = 200, warmup: int = 50,
                 z_thresh: float = 6.0, max_dumps: int = 5, min_gap: int = 25,
                 context: int = 8, save_batch: bool = True):
        self.dir = os.path.join(run_dir, "incidents")
        self.tele = GradTelemetry(named_params)
        self.window = window
        self.warmup = warmup
        self.z_thresh = z_thresh
        self.max_dumps = max_dumps
        self.min_gap = min_gap
        self.save_batch = save_batch
        self.hist: deque[float] = deque(maxlen=window)  # log total norm, finite only
        self.ctx: deque[dict] = deque(maxlen=context)
        self.n_dumps = 0
        self.n_triggers = 0
        self.last_dump_step = -10**9
        self.stash: dict[str, Any] | None = None  # set by training_step

    # -- robust stats -------------------------------------------------------
    def _zscore(self, x: float) -> float | None:
        if len(self.hist) < self.warmup:
            return None
        s = sorted(self.hist)
        med = s[len(s) // 2]
        mad = sorted(abs(v - med) for v in s)[len(s) // 2]
        scale = 1.4826 * mad  # MAD -> sigma for a normal
        if scale <= 1e-12:
            return None
        return (x - med) / scale

    # -- per step -----------------------------------------------------------
    def observe(self, step: int) -> dict[str, float]:
        """Measure, maybe dump, return metrics to log. Call once per optimizer step."""
        m, per = self.tele.measure()
        if not m:
            return {}
        tot = m["grad/raw_norm_total"]
        bad = m["grad/nonfinite_tensors"] > 0
        z = None
        if math.isfinite(tot) and tot > 0:
            z = self._zscore(math.log(tot))
        if z is not None:
            m["grad/raw_norm_z"] = z
        trig = bad or (z is not None and z > self.z_thresh)
        m["grad/flight_trigger"] = 1.0 if trig else 0.0
        entry = {"step": step, "raw_norm_total": tot, "z": z,
                 "loss": self._stash_loss()}
        if trig:
            self.n_triggers += 1
            if self.n_dumps < self.max_dumps and step - self.last_dump_step >= self.min_gap:
                self._dump(step, m, per, z, bad)
        m["grad/flight_dumps"] = float(self.n_dumps)
        # a spike must not become part of its own yardstick
        if math.isfinite(tot) and tot > 0 and not trig:
            self.hist.append(math.log(tot))
        self.ctx.append(entry)
        return m

    def _stash_loss(self):
        if self.stash and self.stash.get("loss") is not None:
            try:
                return float(self.stash["loss"])
            except Exception:
                return None
        return None

    def _dump(self, step, metrics, per, z, bad):
        d = os.path.join(self.dir, f"step_{step:07d}")
        os.makedirs(d, exist_ok=True)
        ranked = sorted(((v, n) for v, n in zip(per, self.tele.names) if v is not None),
                        key=lambda t: (not math.isfinite(t[0]), t[0]), reverse=True)
        tot_sq = sum(v * v for v, _ in ranked if math.isfinite(v)) or 1.0
        top = [{"param": n, "grad_norm": v,
                "share_of_sq": (v * v / tot_sq) if math.isfinite(v) else None}
               for v, n in ranked[:15]]
        nonfinite = [n for v, n in zip(per, self.tele.names) if v is not None and not math.isfinite(v)]
        st = self.stash or {}
        t = st.get("t")
        rep = {
            "step": step,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "trigger": "nonfinite_grad" if bad else "raw_norm_z",
            "z": z, "z_thresh": self.z_thresh,
            "metrics": metrics,
            "rolling_median_norm": math.exp(sorted(self.hist)[len(self.hist) // 2]) if self.hist else None,
            "previous_steps": list(self.ctx),
            "top_params": top,
            "nonfinite_params": nonfinite[:50],
            "n_nonfinite_params": len(nonfinite),
            "batch": {
                "loss": self._stash_loss(),
                "t": [float(x) for x in t.flatten().tolist()] if t is not None else None,
                "prompts": st.get("prompts"),
                "files": st.get("files"),
                "per_item_loss": (st["per_item_loss"].float().tolist()
                                  if torch.is_tensor(st.get("per_item_loss")) else st.get("per_item_loss")),
                "note": "last micro-batch only if accumulate_grad_batches > 1",
            },
        }
        with open(os.path.join(d, "incident.json"), "w") as f:
            json.dump(rep, f, indent=2, default=str)
        if self.save_batch and st:
            # big float tensors (latents, noise) go to disk as fp16: the latents ARE fp16 on
            # disk already, and a fp16 noise copy is plenty for a replay. ~16 MB/dump at B=32 T1024.
            blob = {k: ((v.detach().cpu().half() if v.is_floating_point() and v.numel() > 100_000
                         else v.detach().cpu()) if torch.is_tensor(v) else v) for k, v in st.items()}
            torch.save(blob, os.path.join(d, "batch.pt"))
        self.n_dumps += 1
        self.last_dump_step = step
        print(f"[flight-recorder] incident at step {step} ({rep['trigger']}, z={z}) -> {d}", flush=True)

# Owner: WINTERMUTE (created 2026-10-23)
"""Local per-step training telemetry for train_latch.py (--step-log): JSON lines in the save dir, no wandb, no network.

steps.jsonl, one line per optimizer step:
  step, epoch, loss, lr (every param group), gnorm_raw (whole-step gradient before AdaGC / clipping: spikes show here),
  gnorm_after_adagc (the norm the global --grad-clip saw; null without --grad-clip), skipped (spike-skipped step),
  upd (||theta_t - theta_{t-1}||, the step actually taken), path (sum of upd since step 0 of this run), disp (||theta_t - theta_0||),
  eff (disp / path: 1 = straight line, low = wandering), wrms (global weight RMS)
layers.jsonl, every --layer-log steps, per tensor and per model part (the first two name components, e.g. blocks.3):
  w_std, w_rms, g_norm (raw), upd (movement since the previous layer line), dist0 (from theta_0), g_over_w
Under a Schedule-Free optimizer the live parameters are the training point y, so this is y's trajectory; the saved heads are
the averaged x. On resume, theta_0 / path are restored from the resume file, so the curves continue instead of restarting.
"""
import json
import os
import time

import torch


class StepTelemetry:
    def __init__(self, model, save_dir, layer_every=20, state=None):
        self.named = [(n.removeprefix("_orig_mod."), p) for n, p in model.named_parameters() if p.requires_grad]
        self.params = [p for _, p in self.named]
        self.layer_every = max(1, int(layer_every))
        self.f_step = open(os.path.join(save_dir, "steps.jsonl"), "a", encoding="utf-8")
        self.f_layer = open(os.path.join(save_dir, "layers.jsonl"), "a", encoding="utf-8")
        with torch.no_grad():
            if state is not None:
                self.theta0 = [t.to(p.device) for t, p in zip(state["theta0"], self.params)]
                self.path = float(state["path"])
            else:
                self.theta0 = [p.detach().float().clone() for p in self.params]
                self.path = 0.0
            self.prev = [p.detach().float().clone() for p in self.params]
            self.prev_layer = [p.detach().float().clone() for p in self.params]
        # On resume the loaded weights are the Schedule-Free average x and the first opt.train() moves them back to the
        # training point y; re-take the "previous" snapshot at the first gradient so that jump is not counted as a step.
        self._sync_prev = state is not None
        self.g_raw = None
        self.g_raw_per = None
        self.t_last = time.perf_counter()

    def state_dict(self):
        return {"theta0": [t.cpu() for t in self.theta0], "path": self.path}

    @torch.no_grad()
    def grads(self, step):
        """Call after backward, before AdaGC / clipping."""
        if self._sync_prev:
            cur = [p.detach().float() for p in self.params]
            torch._foreach_copy_(self.prev, cur)
            torch._foreach_copy_(self.prev_layer, cur)
            self._sync_prev = False
        gs = [p.grad.float() if p.grad is not None else torch.zeros((), device=p.device) for p in self.params]
        per = torch.stack(torch._foreach_norm(gs))
        self.g_raw = float(per.norm())
        self.g_raw_per = per if step % self.layer_every == 0 else None

    @torch.no_grad()
    def after_step(self, step, epoch, loss, lr, gnorm_after_adagc=None, skipped=False):
        cur = [p.detach().float() for p in self.params]
        upd = float(torch.stack(torch._foreach_norm(torch._foreach_sub(cur, self.prev))).norm())
        torch._foreach_copy_(self.prev, cur)
        self.path += upd
        disp = float(torch.stack(torch._foreach_norm(torch._foreach_sub(cur, self.theta0))).norm())
        n = sum(p.numel() for p in self.params)
        wrms = float(torch.stack(torch._foreach_norm(cur)).norm()) / n ** 0.5
        now = time.perf_counter()
        rec = {"step": step, "epoch": epoch, "loss": loss, "lr": lr, "gnorm_raw": self.g_raw, "gnorm_after_adagc": gnorm_after_adagc,
               "skipped": skipped,
               "upd": upd, "path": self.path, "disp": disp, "eff": disp / self.path if self.path > 0 else None, "wrms": wrms,
               "sec": round(now - self.t_last, 4)}
        self.t_last = now
        self.f_step.write(json.dumps(rec) + "\n")
        if step % self.layer_every == 0:
            self._layers(step, epoch, cur)
            self.f_step.flush()

    def _layers(self, step, epoch, cur):
        upd = torch._foreach_norm(torch._foreach_sub(cur, self.prev_layer))
        d0 = torch._foreach_norm(torch._foreach_sub(cur, self.theta0))
        torch._foreach_copy_(self.prev_layer, cur)
        tensors, parts = {}, {}
        for i, ((name, _), w) in enumerate(zip(self.named, cur)):
            g = float(self.g_raw_per[i]) if self.g_raw_per is not None else None
            wn = float(w.norm())
            r = {"w_std": float(w.std()) if w.numel() > 1 else 0.0, "w_rms": wn / w.numel() ** 0.5, "g_norm": g,
                 "upd": float(upd[i]), "dist0": float(d0[i]), "g_over_w": g / (wn + 1e-12) if g is not None else None}
            tensors[name] = r
            key = ".".join(name.split(".")[:2]) if name.split(".")[0] in ("blocks", "layers") else name.split(".")[0]
            a = parts.setdefault(key, {"n": 0, "sq": 0.0, "s": 0.0, "upd2": 0.0, "d02": 0.0, "g2": 0.0, "w2": 0.0})
            a["n"] += w.numel(); a["s"] += float(w.sum()); a["sq"] += float((w * w).sum())
            a["upd2"] += r["upd"] ** 2; a["d02"] += r["dist0"] ** 2; a["g2"] += (g or 0.0) ** 2; a["w2"] += wn ** 2
        part_out = {}
        for k, a in parts.items():
            mean = a["s"] / a["n"]
            part_out[k] = {"w_std": max(a["sq"] / a["n"] - mean * mean, 0.0) ** 0.5, "w_rms": (a["sq"] / a["n"]) ** 0.5,
                           "upd": a["upd2"] ** 0.5, "dist0": a["d02"] ** 0.5, "g_norm": a["g2"] ** 0.5,
                           "g_over_w": a["g2"] ** 0.5 / (a["w2"] ** 0.5 + 1e-12)}
        self.f_layer.write(json.dumps({"step": step, "epoch": epoch, "parts": part_out, "tensors": tensors}) + "\n")
        self.f_layer.flush()

    def close(self):
        self.f_step.close()
        self.f_layer.close()

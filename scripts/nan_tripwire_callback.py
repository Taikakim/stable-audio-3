"""NaN tripwire — names the first non-finite tensor in a training run.

CONTINUITY, 2026-09-21. Written for the modular-optimizer runs that go NaN inside a
~10-step window and then keep 'training' on poisoned weights until the GPU faults with
an out-of-bounds index. Nothing in stable_audio_tools/training/modular_opt/ checks
finiteness, so the first NaN is silent and every downstream symptom is a red herring.

Opt-in and inert unless SA3_NAN_TRIPWIRE=1 is exported, so it costs nothing when off.
It is a DIAGNOSTIC: it syncs the GPU on every check and will slow a run down.

Checks, in the order the corruption would actually appear:
  1. after backward   -> first non-finite GRADIENT   (blames the loss/backward)
  2. before optimizer -> gradients again, post-clip   (blames gradient clipping)
  3. after optimizer  -> first non-finite PARAMETER  (blames the optimizer step)

Whichever fires first localises the bug to one stage and names the tensor.
"""

import os

import torch
import pytorch_lightning as pl


def _first_nonfinite(named_tensors, kind):
    """Return (name, detail) for the first non-finite tensor, else None."""
    for name, t in named_tensors:
        if t is None:
            continue
        if not torch.isfinite(t).all():
            n_nan = int(torch.isnan(t).sum())
            n_inf = int(torch.isinf(t).sum())
            finite = t[torch.isfinite(t)]
            mx = float(finite.abs().max()) if finite.numel() else float("nan")
            return name, (f"{kind} non-finite: {n_nan} NaN, {n_inf} Inf of {t.numel()} "
                          f"elements; max |finite| = {mx:.4g}")
    return None


class NaNTripwireCallback(pl.Callback):
    """Abort at the first non-finite gradient or parameter, naming stage and tensor."""

    def __init__(self, check_params: bool = True, check_grads: bool = True):
        self.check_params = check_params
        self.check_grads = check_grads
        self.tripped = False
        self._announced = False

    def _announce(self, pl_module):
        """Say how many tensors are watched, so a silent run means 'clean', not 'inert'."""
        if self._announced:
            return
        self._announced = True
        n = sum(1 for _, p in pl_module.named_parameters() if p.requires_grad)
        print(f"[NaN TRIPWIRE] watching {n} trainable tensors.", flush=True)
        if n == 0:
            print("[NaN TRIPWIRE] WARNING: watching NOTHING — this tripwire cannot fire. "
                  "A silent run proves nothing.", flush=True)

    def _report(self, trainer, stage, hit):
        if hit is None or self.tripped:
            return False
        name, detail = hit
        self.tripped = True
        step = trainer.global_step
        print("\n" + "=" * 72, flush=True)
        print(f"[NaN TRIPWIRE] FIRST non-finite value at global_step={step}, "
              f"epoch={trainer.current_epoch}", flush=True)
        print(f"[NaN TRIPWIRE] stage : {stage}", flush=True)
        print(f"[NaN TRIPWIRE] tensor: {name}", flush=True)
        print(f"[NaN TRIPWIRE] {detail}", flush=True)
        print("[NaN TRIPWIRE] Stopping so the poisoned state is not overwritten.", flush=True)
        print("=" * 72 + "\n", flush=True)
        trainer.should_stop = True
        return True

    def on_after_backward(self, trainer, pl_module):
        self._announce(pl_module)
        if self.tripped or not self.check_grads:
            return
        grads = ((n, p.grad) for n, p in pl_module.named_parameters() if p.requires_grad)
        self._report(trainer, "after backward (loss/backward produced it)",
                     _first_nonfinite(grads, "grad"))

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.tripped or not self.check_grads:
            return
        grads = ((n, p.grad) for n, p in pl_module.named_parameters() if p.requires_grad)
        self._report(trainer, "before optimizer step (survived gradient clipping)",
                     _first_nonfinite(grads, "grad"))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.tripped or not self.check_params:
            return
        params = ((n, p.data) for n, p in pl_module.named_parameters() if p.requires_grad)
        self._report(trainer, "after optimizer step (the optimizer wrote it)",
                     _first_nonfinite(params, "param"))


def maybe_build():
    """Return a tripwire callback if SA3_NAN_TRIPWIRE=1, else None."""
    if os.environ.get("SA3_NAN_TRIPWIRE", "") not in ("1", "true", "yes"):
        return None
    print("[NaN TRIPWIRE] armed (SA3_NAN_TRIPWIRE=1) — this syncs the GPU every step "
          "and will slow training down.", flush=True)
    return NaNTripwireCallback()

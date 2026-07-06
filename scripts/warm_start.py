"""Optimizer/scheduler warm-start for continuing runs from OLD-format LoRA
checkpoints.

Checkpoints saved before the on_save_checkpoint Lightning-key fix carry
optimizer_states / lr_schedulers but NOT 'pytorch-lightning_version'/'loops',
so trainer.fit(ckpt_path=...) rejects them with a KeyError. Weights already
warm-start via --lora_checkpoint; this module restores the remaining pieces —
optimizer state (for FusionOpt that's the Schedule-Free z/x iterates, i.e.
most of what a true continuation needs) and LR-scheduler state. Only the loop
counters (epoch/global_step numbering, dataloader shuffle position) are lost,
which is cosmetic for constant-LR runs.

torch.optim.Optimizer.load_state_dict casts loaded state tensors to the
device/dtype of the matching params, so a map_location="cpu" load is fine.
"""
import torch
import pytorch_lightning as pl


def load_optimizer_state(trainer, ckpt):
    """Restore optimizer_states (+ lr_schedulers, if any) from an old-format
    checkpoint dict into a live trainer. Returns the number of optimizers
    restored. Fails loud on missing/mismatched state — a silent partial
    restore would masquerade as a fresh start."""
    opt_states = ckpt.get("optimizer_states")
    if not opt_states:
        raise ValueError(
            "checkpoint has no optimizer_states — nothing to warm-start "
            "(was it saved with checkpointing of optimizer state enabled?)"
        )
    if len(opt_states) != len(trainer.optimizers):
        raise ValueError(
            f"checkpoint has {len(opt_states)} optimizer state(s) but the "
            f"trainer has {len(trainer.optimizers)} optimizer(s)"
        )
    for opt, state in zip(trainer.optimizers, opt_states):
        opt.load_state_dict(state)

    sched_states = ckpt.get("lr_schedulers") or []
    for cfg, state in zip(trainer.lr_scheduler_configs, sched_states):
        cfg.scheduler.load_state_dict(state)
    return len(opt_states)


class OptimizerWarmStart(pl.Callback):
    """Load optimizer/scheduler state from `ckpt_path` at on_train_start —
    optimizers exist by then, and no gradient step has run yet."""

    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path

    def on_train_start(self, trainer, pl_module):
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        n = load_optimizer_state(trainer, ckpt)
        print(f"[warm_start] restored {n} optimizer state(s) from {self.ckpt_path}")

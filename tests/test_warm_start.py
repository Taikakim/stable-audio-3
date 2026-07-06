"""Tests for scripts/warm_start.py — optimizer/scheduler state warm-start for
continuing runs whose checkpoints predate the on_save_checkpoint Lightning-key
fix (no 'pytorch-lightning_version'/'loops', so trainer.fit(ckpt_path=...)
raises KeyError). Weights already warm-start via --lora_checkpoint; this
restores the missing optimizer_states / lr_schedulers."""
import sys, os

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from warm_start import load_optimizer_state, OptimizerWarmStart


def _stepped_adamw(model, n_steps=2, lr=1e-3):
    """Run a couple of real steps so the optimizer has exp_avg/step state."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
    return opt


class _TrainerStub:
    """Just the two attributes load_optimizer_state touches."""

    def __init__(self, optimizers, lr_scheduler_configs=()):
        self.optimizers = list(optimizers)
        self.lr_scheduler_configs = list(lr_scheduler_configs)


# ------------------------------------------------------- pure function

def test_restores_optimizer_state():
    torch.manual_seed(0)
    donor_model = nn.Linear(8, 2)
    donor_opt = _stepped_adamw(donor_model, n_steps=3)
    ckpt = {"optimizer_states": [donor_opt.state_dict()]}

    fresh_model = nn.Linear(8, 2)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    assert len(fresh_opt.state) == 0  # sanity: fresh optimizer has no state

    n = load_optimizer_state(_TrainerStub([fresh_opt]), ckpt)

    assert n == 1
    donor_state = donor_opt.state_dict()["state"]
    fresh_state = fresh_opt.state_dict()["state"]
    assert set(fresh_state.keys()) == set(donor_state.keys())
    for idx in donor_state:
        assert int(fresh_state[idx]["step"]) == int(donor_state[idx]["step"])
        assert torch.allclose(fresh_state[idx]["exp_avg"], donor_state[idx]["exp_avg"])
        assert torch.allclose(
            fresh_state[idx]["exp_avg_sq"], donor_state[idx]["exp_avg_sq"]
        )


def test_optimizer_count_mismatch_raises():
    model = nn.Linear(8, 2)
    opt = torch.optim.AdamW(model.parameters())
    ckpt = {"optimizer_states": []}  # ckpt has none, trainer has one
    with pytest.raises(ValueError, match="optimizer"):
        load_optimizer_state(_TrainerStub([opt]), ckpt)


def test_missing_optimizer_states_key_raises():
    model = nn.Linear(8, 2)
    opt = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="optimizer_states"):
        load_optimizer_state(_TrainerStub([opt]), {"state_dict": {}})


def test_restores_lr_scheduler_state():
    donor_model = nn.Linear(8, 2)
    donor_opt = torch.optim.AdamW(donor_model.parameters(), lr=1e-3)
    donor_sched = torch.optim.lr_scheduler.StepLR(donor_opt, step_size=10)
    for _ in range(4):
        donor_sched.step()
    ckpt = {
        "optimizer_states": [donor_opt.state_dict()],
        "lr_schedulers": [donor_sched.state_dict()],
    }

    fresh_model = nn.Linear(8, 2)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_sched = torch.optim.lr_scheduler.StepLR(fresh_opt, step_size=10)

    class _SchedCfg:
        scheduler = fresh_sched

    load_optimizer_state(_TrainerStub([fresh_opt], [_SchedCfg()]), ckpt)
    assert fresh_sched.state_dict()["_step_count"] == donor_sched.state_dict()["_step_count"]


# ------------------------------------------------- callback via real Trainer

def test_callback_loads_state_at_train_start(tmp_path):
    """Prove hook ordering: by on_train_start the optimizer exists, and the
    donor state survives into the first real step (AdamW step counts continue
    from the donor's, not from zero)."""
    import pytorch_lightning as pl

    class Tiny(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(8, 2)

        def training_step(self, batch, batch_idx):
            return self.net(batch[0]).sum()

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

    torch.manual_seed(0)
    donor = Tiny()
    donor_opt = _stepped_adamw(donor.net, n_steps=5)
    ckpt_path = tmp_path / "old_format.ckpt"
    torch.save(
        {"state_dict": donor.state_dict(), "optimizer_states": [donor_opt.state_dict()]},
        ckpt_path,
    )

    fresh = Tiny()
    ds = torch.utils.data.TensorDataset(torch.randn(4, 8))
    dl = torch.utils.data.DataLoader(ds, batch_size=4)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=1,
        limit_train_batches=1,
        callbacks=[OptimizerWarmStart(str(ckpt_path))],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(fresh, dl)

    steps = {int(s["step"]) for s in trainer.optimizers[0].state_dict()["state"].values()}
    # donor had 5 steps; one fit step on top -> 6. Fresh-start would be 1.
    assert steps == {6}

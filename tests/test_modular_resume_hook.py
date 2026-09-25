"""ModularTrainingWrapper.on_train_start restores the optimizer step count from global_step when a
pre-fix checkpoint carried none (training-findings 20a). Runs the REAL hook on a stand-in module."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_lora_modular import ModularTrainingWrapper  # noqa: E402
from stable_audio_tools.training.modular_opt import ModularOptimizer, build_modular_param_groups  # noqa: E402


def _opt():
    m = torch.nn.Sequential(torch.nn.Linear(64, 96))
    return ModularOptimizer(build_modular_param_groups(m), lr=1e-3)


def test_hook_sets_step_count_from_global_step():
    opt = _opt()
    lightning_wrapped = SimpleNamespace(optimizer=opt)  # LightningOptimizer exposes .optimizer
    fake = SimpleNamespace(optimizers=lambda: lightning_wrapped, global_step=6340)
    ModularTrainingWrapper.on_train_start(fake)
    assert opt.get_step_count() == 6340


def test_hook_leaves_a_restored_count_alone():
    opt = _opt()
    opt.set_step_count(7000)  # a post-fix checkpoint already restored it
    fake = SimpleNamespace(optimizers=lambda: [opt], global_step=6340)
    ModularTrainingWrapper.on_train_start(fake)
    assert opt.get_step_count() == 7000

# tests/test_latch_modular_c_series.py
# Owner: KUANG (created 2026-10-17)
"""train_latch --optimizer modular with the C-series flags (SAO docs/modular-opt-test-plan-2026-10-17.md):
the flags reach ModularOptimizer, and the half-batch gradient-noise reading leaves the real step untouched.
Needs stable_audio_tools with modular_opt.GradNoiseMeter (audio-tools-avp feature/modular-opt-c-series)."""
import argparse

import pytest
import torch

modular_opt = pytest.importorskip("stable_audio_tools.training.modular_opt")
if not hasattr(modular_opt, "GradNoiseMeter"):
    pytest.skip("stable_audio_tools without the C-series modular_opt", allow_module_level=True)

from scripts.latch.latch_model import LatCH  # noqa: E402
from scripts.latch.train_latch import _gn_measure, forward_noise, make_optimizer, masked_mse  # noqa: E402


def _args(**over):
    a = dict(optimizer="modular", lr=1e-2, mod_whitening="shampoo", mod_wd=0.01, mod_muon_all=False,
             mod_sign_lr=None, mod_sign_wd=0.0, mod_scalar_opt="sign", mod_scalar_sf_beta=None, mod_normuon=True,
             mod_sf=True, mod_warmup=0, mod_precond_freq=10, mod_radial_brake=1.0, mod_escape_velocity=False,
             mod_sf_c_warmup=None, mod_sf_r=1.0, mod_wd_overtraining=False, mod_snr_gate=False, mod_snr_floor=0.0,
             _spe=10, mod_momentum=0.9, mod_sf_beta=0.9, mod_normuon_mode="row_unit", mod_update_scale="native",
             mod_normuon_after_unwhiten=False, mod_ns_dtype="fp32", mod_late_from=0, mod_ns_batch=False)
    a.update(over)
    return argparse.Namespace(**a)


def _model():
    torch.manual_seed(0)
    return LatCH(in_channels=256, out_channels=1, dim=64, depth=1, num_heads=2, t_injection="adaln_zero")


def test_flags_reach_the_optimizer():
    m = _model()
    opt, is_fusion = make_optimizer(_args(mod_scalar_opt="adam", mod_sign_lr=3e-3, mod_scalar_sf_beta=0.95,
                                          mod_momentum=0.8, mod_update_scale="rms0.2", mod_ns_dtype="bf16",
                                          mod_normuon_mode="dion", mod_normuon_after_unwhiten=True), m)
    assert is_fusion
    types = {g["group_type"]: g for g in opt.param_groups}
    assert "adam" in types and types["adam"]["lr"] == 3e-3 and types["adam"]["sf_beta"] == 0.95
    g = types["spectral"]
    assert (g["beta1"], g["update_scale"], g["ns_dtype"], g["normuon_mode"], g["normuon_after_unwhiten"]) == \
        (0.8, "rms0.2", "bf16", "dion", True)


def test_grad_noise_reading_leaves_step_untouched(tmp_path):
    from stable_audio_tools.training.modular_opt import GradNoiseMeter
    m = _model()
    z0 = torch.randn(4, 256, 16)
    t = torch.rand(4)
    z_t = forward_noise(z0, torch.randn_like(z0), t)
    targets = torch.randn(4, 1, 16)
    mask = torch.ones(4, 16)
    loss = masked_mse(m(z_t, t).float(), targets, mask)
    loss.backward()
    before = [p.grad.clone() for p in m.parameters() if p.grad is not None]
    rng = torch.get_rng_state()
    meter = GradNoiseMeter(m.named_parameters(), every=1, out_path=str(tmp_path / "gn.jsonl"))
    out = _gn_measure(meter, m, masked_mse, z_t, t, targets, mask, 0, torch.device("cpu"), False)
    after = [p.grad for p in m.parameters() if p.grad is not None]
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert torch.equal(rng, torch.get_rng_state())
    assert -1.0 <= out["grad_noise/cos_pooled"] <= 1.0
    assert (tmp_path / "gn.jsonl").exists()

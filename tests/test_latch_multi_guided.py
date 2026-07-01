# tests/test_latch_multi_guided.py
import torch

from stable_audio_3.inference.latch_guided import (
    sample_flow_euler_multi_latch_guided,
    _make_latch_criterion,
)


def _toy_model(x, t, **kw):
    return x  # constant velocity field


class _ConstHead(torch.nn.Module):
    def forward(self, z, t):
        return z.mean(dim=1, keepdim=True)  # (B, 1, T)


def _schedule(steps):
    return torch.linspace(1.0, 0.0, steps + 1)


def _guide(target_val, weight=1.0, start=0.0, end=1.0):
    return {
        "head": _ConstHead(),
        "target": torch.full((1, 1, 8), float(target_val)),
        "weight": weight,
        "start_pct": start,
        "end_pct": end,
        "loss_type": "mse",
    }


def test_zero_weight_matches_plain_euler():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    sigmas = _schedule(6)
    out = sample_flow_euler_multi_latch_guided(
        _toy_model, x.clone(), sigmas, [_guide(0.0, weight=0.0)],
        rho=5.0, mu=5.0, gamma=0.0,
    )
    y = x.clone()
    for i in range(6):
        v = _toy_model(y, sigmas[i].expand(1))
        y = y + (sigmas[i + 1] - sigmas[i]) * v
    assert torch.allclose(out, y, atol=1e-5)


def test_no_guides_matches_plain_euler():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    sigmas = _schedule(6)
    out = sample_flow_euler_multi_latch_guided(_toy_model, x.clone(), sigmas, [])
    y = x.clone()
    for i in range(6):
        y = y + (sigmas[i + 1] - sigmas[i]) * _toy_model(y, sigmas[i].expand(1))
    assert torch.allclose(out, y, atol=1e-5)


def test_positive_weight_moves_toward_target():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    sigmas = _schedule(20)
    head_probe = _ConstHead()
    out_off = sample_flow_euler_multi_latch_guided(
        _toy_model, x.clone(), sigmas, [_guide(-5.0, weight=0.0)], rho=0.0, mu=5.0, gamma=0.0)
    out_on = sample_flow_euler_multi_latch_guided(
        _toy_model, x.clone(), sigmas, [_guide(-5.0, weight=1.0)], rho=0.0, mu=5.0, gamma=0.0)
    assert head_probe(out_on, torch.zeros(1)).mean() < head_probe(out_off, torch.zeros(1)).mean()


def test_two_guides_both_applied():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    sigmas = _schedule(20)
    probe = _ConstHead()
    out_one = sample_flow_euler_multi_latch_guided(
        _toy_model, x.clone(), sigmas, [_guide(-5.0)], rho=0.0, mu=5.0, gamma=0.0)
    out_two = sample_flow_euler_multi_latch_guided(
        _toy_model, x.clone(), sigmas, [_guide(-5.0), _guide(-5.0)], rho=0.0, mu=5.0, gamma=0.0)
    # Two equal guides exert stronger pull toward the negative target than one.
    assert probe(out_two, torch.zeros(1)).mean() < probe(out_one, torch.zeros(1)).mean()


def test_window_disables_guidance_outside_range():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    sigmas = _schedule(10)
    # Window [0,0) => never active => identical to plain euler.
    out = sample_flow_euler_multi_latch_guided(
        _toy_model, x.clone(), sigmas, [_guide(-5.0, start=0.0, end=0.0)], rho=5.0, mu=5.0, gamma=0.0)
    y = x.clone()
    for i in range(10):
        y = y + (sigmas[i + 1] - sigmas[i]) * _toy_model(y, sigmas[i].expand(1))
    assert torch.allclose(out, y, atol=1e-5)


def test_fp16_model_dtype_preserved():
    # The GUI loads the diffusion model in fp16. Guidance math runs in fp32
    # internally, but x must be cast back to the model's dtype for the velocity
    # forward and on return, or the next model call hits a dtype mismatch.
    torch.manual_seed(0)

    def fp16_model(x, t, **kw):
        assert x.dtype == torch.float16, f"model got {x.dtype}"
        return x * 0.5

    x = torch.randn(1, 4, 8, dtype=torch.float16)
    sigmas = _schedule(10)
    out = sample_flow_euler_multi_latch_guided(
        fp16_model, x, sigmas, [_guide(-5.0)], rho=8.0, mu=8.0, gamma=0.3, n_iter=2)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


def test_unknown_loss_type_raises():
    import pytest
    with pytest.raises(ValueError):
        _make_latch_criterion("nonsense")

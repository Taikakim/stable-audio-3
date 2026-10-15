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


# --- schedule="flat" (WINTERMUTE 2026-10-15) ---------------------------------------------
from stable_audio_3.inference.latch_guided import _window_scale  # noqa: E402


def test_window_scale_alpha_and_flat_sum_to_one_inside_window():
    alpha = 1.0 - _schedule(10)[:-1]          # 0.0, 0.1, ... 0.9
    for sched in ("alpha", "flat"):
        w = _window_scale(alpha, 0, 10, sched)
        assert torch.isclose(w.sum(), torch.tensor(1.0))
    a, f = _window_scale(alpha, 0, 10, "alpha"), _window_scale(alpha, 0, 10, "flat")
    assert a[0] == 0.0 and a[-1] > a[1]       # TFG alpha: nothing at the noisiest step
    assert torch.allclose(f, torch.full_like(f, 0.1))
    w = _window_scale(alpha, 2, 5, "flat")
    assert w[:2].sum() == 0 and w[5:].sum() == 0 and torch.allclose(w[2:5], torch.full((3,), 1 / 3))


def test_window_scale_rejects_unknown_schedule():
    import pytest
    with pytest.raises(ValueError):
        _window_scale(torch.ones(4), 0, 4, "cosine")


def test_flat_schedule_guides_the_first_step_alpha_does_not():
    # Window = first step only. alpha_0 = 0 there, so "alpha" is a no-op; "flat" must act.
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    sigmas = _schedule(10)
    plain = sample_flow_euler_multi_latch_guided(
        _toy_model, x.clone(), sigmas, [_guide(-5.0, start=0.0, end=0.1)], rho=5.0, mu=5.0, gamma=0.0)
    flat = sample_flow_euler_multi_latch_guided(
        _toy_model, x.clone(), sigmas, [_guide(-5.0, start=0.0, end=0.1)], rho=5.0, mu=5.0, gamma=0.0,
        guide_schedule="flat")
    probe = _ConstHead()
    assert probe(flat, torch.zeros(1)).mean() < probe(plain, torch.zeros(1)).mean() - 1e-3


# --- guide_rescale (WINTERMUTE 2026-10-15) -----------------------------------------------
from stable_audio_3.inference.latch_guided import (  # noqa: E402
    _rescale_like, sample_flow_pingpong_multi_latch_guided,
)


def test_rescale_like_phi_endpoints():
    torch.manual_seed(0)
    ref, new = torch.randn(2, 4, 16), 3.0 * torch.randn(2, 4, 16) + 1.0
    assert torch.equal(_rescale_like(new, ref, 0.0), new)
    out = _rescale_like(new, ref, 1.0)
    assert torch.allclose(out.std(dim=(1, 2)), ref.std(dim=(1, 2)), atol=1e-5)   # per sample
    half = _rescale_like(new, ref, 0.5).std(dim=(1, 2))
    assert ((half < new.std(dim=(1, 2))) & (half > ref.std(dim=(1, 2)))).all()


def test_rescale_zero_is_the_default_path():
    torch.manual_seed(0)
    x, sigmas = torch.randn(1, 4, 8), _schedule(8)
    a = sample_flow_euler_multi_latch_guided(_toy_model, x.clone(), sigmas, [_guide(-5.0)], rho=5.0, mu=5.0, gamma=0.0)
    b = sample_flow_euler_multi_latch_guided(_toy_model, x.clone(), sigmas, [_guide(-5.0)], rho=5.0, mu=5.0, gamma=0.0,
                                             guide_rescale=0.0)
    assert torch.equal(a, b)


def test_rescale_holds_energy_but_still_steers():
    # A strong pull to a far target inflates the latent; rescale must keep its std near the
    # unguided one while the head readout still moves toward the target.
    torch.manual_seed(0)
    x, sigmas = torch.randn(1, 4, 8), _schedule(8)
    run = lambda **kw: sample_flow_euler_multi_latch_guided(  # noqa: E731
        _toy_model, x.clone(), sigmas, [_guide(-5.0)], rho=5.0, mu=5.0, gamma=0.0, **kw)
    plain = sample_flow_euler_multi_latch_guided(_toy_model, x.clone(), sigmas, [])
    raw, resc = run(), run(guide_rescale=1.0)
    assert abs(resc.std() - plain.std()) < abs(raw.std() - plain.std())
    probe = _ConstHead()
    assert probe(resc, torch.zeros(1)).mean() < probe(plain, torch.zeros(1)).mean() - 1e-3


def test_pingpong_accepts_guide_rescale():
    torch.manual_seed(0)
    out = sample_flow_pingpong_multi_latch_guided(
        _toy_model, torch.randn(1, 4, 8), _schedule(6), [_guide(-5.0)], rho=5.0, mu=5.0, gamma=0.0, guide_rescale=1.0)
    assert torch.isfinite(out).all()


# --- guide_anchor (WINTERMUTE 2026-10-15) ------------------------------------------------
from stable_audio_3.inference.latch_guided import _anchor_std  # noqa: E402


def test_anchor_std_is_a_ceiling_at_the_healthy_level():
    torch.manual_seed(0)
    loud, quiet = 5.0 * torch.randn(2, 4, 64), 0.5 * torch.randn(2, 4, 64)
    assert torch.equal(_anchor_std(loud, 0.3, 0.0), loud)            # off
    assert torch.equal(_anchor_std(quiet, 0.3, 1.2), quiet)          # below the ceiling: untouched
    capped = _anchor_std(loud, 0.0, 1.2).std(dim=(1, 2))             # t=0: ceiling = s0
    assert torch.allclose(capped, torch.full_like(capped, 1.2), atol=1e-5)
    capped = _anchor_std(loud, 1.0, 1.2).std(dim=(1, 2))             # t=1: ceiling = unit noise
    assert torch.allclose(capped, torch.ones_like(capped), atol=1e-5)


def test_anchor_zero_is_the_default_path():
    torch.manual_seed(0)
    x, sigmas = torch.randn(1, 4, 8), _schedule(8)
    a = sample_flow_euler_multi_latch_guided(_toy_model, x.clone(), sigmas, [_guide(-5.0)], rho=5.0, mu=5.0, gamma=0.0)
    b = sample_flow_euler_multi_latch_guided(_toy_model, x.clone(), sigmas, [_guide(-5.0)], rho=5.0, mu=5.0, gamma=0.0,
                                             guide_anchor=0.0)
    assert torch.equal(a, b)


def test_anchor_caps_the_final_latent_and_still_steers():
    torch.manual_seed(0)
    # Target alternates +-5 over time, so steering toward it inflates the variance (a uniform
    # target would only shift the mean, which std does not see).
    torch.manual_seed(0)
    x, sigmas = torch.randn(1, 4, 8), _schedule(8)
    tgt = torch.tensor([5.0, -5.0] * 4).view(1, 1, 8)

    def run(**kw):
        g = {**_guide(0.0), "target": tgt.clone()}
        return sample_flow_euler_multi_latch_guided(_toy_model, x.clone(), sigmas, [g], rho=5.0, mu=5.0, gamma=0.0, **kw)

    raw, anch = run(), run(guide_anchor=1.2)
    assert raw.std() > 1.2 and anch.std() <= 1.2 + 1e-4
    probe = lambda z: (_ConstHead()(z, torch.zeros(1)) * tgt).mean()  # noqa: E731  alignment with the target
    plain = sample_flow_euler_multi_latch_guided(_toy_model, x.clone(), sigmas, [])
    assert probe(anch) > probe(plain) + 1e-3


def test_pingpong_accepts_guide_anchor():
    torch.manual_seed(0)
    out = sample_flow_pingpong_multi_latch_guided(
        _toy_model, torch.randn(1, 4, 8), _schedule(6),
        [{**_guide(0.0), "target": torch.tensor([5.0, -5.0] * 4).view(1, 1, 8)}], rho=5.0, mu=5.0, gamma=0.0, guide_anchor=1.2)
    assert torch.isfinite(out).all() and out.std() <= 1.2 + 1e-4

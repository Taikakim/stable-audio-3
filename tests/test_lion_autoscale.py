"""LionSR + D-Adaptation autoscale (Kim's ask 2026-09-09).

Two things must hold, and the first is the one that protects every existing run:
autoscale OFF must be byte-identical to the LionSR we already have. The second is that
the estimator we bolted onto Lion is the SAME estimator FusionOpt runs as its
"autoscale" component — factoring it out into dadapt_scale.py created a second copy of
that math, and a second copy drifts unless something checks it every time.
"""
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "lumi", "vendor"))

from stable_audio_3.training.dadapt_scale import DAdaptScaler  # noqa: E402
from stable_audio_3.training.lion_optimizer import LionSR  # noqa: E402
from stable_audio_tools.training.fusion_opt import FusionOpt  # noqa: E402


def _grads(shape, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=g) for _ in range(n)]


def _run(param, opt, grads, rng_seed=1234):
    """Run a whole trajectory under a fixed global RNG.

    LionSR's writeback is STOCHASTICALLY rounded to bf16, so it draws from the global
    RNG on every step. Two optimizers stepped alternately consume that stream in turns
    and diverge by design — each trajectory has to be run start-to-finish under its own
    identical seed, not interleaved.
    """
    torch.manual_seed(rng_seed)
    for g in grads:
        param.grad = g.clone()
        opt.step()
    return param


def test_autoscale_off_is_unchanged():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(16, 16))
    b = torch.nn.Parameter(a.detach().clone())
    gs = _grads((16, 16), 5)
    _run(a, LionSR([a], lr=1e-3), gs)
    _run(b, LionSR([b], lr=1e-3, autoscale=False), gs)
    assert torch.equal(a.data, b.data)


def test_scaler_matches_fusion_autoscale_exactly():
    """DAdaptScaler must track FusionOpt._update_autoscale step for step."""
    torch.manual_seed(0)
    p_f = torch.nn.Parameter(torch.randn(16, 16))
    p_s = torch.nn.Parameter(p_f.detach().clone())

    fus = FusionOpt([{"params": [p_f], "group_type": "spectral"}], lr=1e-3,
                    warmup_steps=0, components={"ns5", "normuon", "autoscale"},
                    autoscale_slice_p=4)
    scaler = DAdaptScaler(slice_p=4)
    state = {}

    for g in _grads((16, 16), 6, seed=1):
        p_f.grad, p_s.grad = g.clone(), g.clone()
        m_f = fus._update_autoscale()
        m_s = scaler.update([{"params": [p_s]}],
                            lambda q: state.setdefault(id(q), {}))
        assert m_f == pytest.approx(m_s, rel=1e-9), f"{m_f} != {m_s}"
        # keep the two parameter trajectories identical so the estimators see the
        # same displacement-from-init on the next step
        with torch.no_grad():
            p_f.add_(g, alpha=-1e-3)
            p_s.add_(g, alpha=-1e-3)
    assert scaler.d > scaler.d0, "d never grew — the test never exercised the mechanism"


def test_autoscale_on_actually_scales_the_step():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(16, 16))
    b = torch.nn.Parameter(a.detach().clone())
    gs = _grads((16, 16), 8, seed=2)
    _run(a, LionSR([a], lr=1e-3), gs)
    o2 = LionSR([b], lr=1e-3, autoscale=True, autoscale_d0=1e-6)
    _run(b, o2, gs)
    assert not torch.equal(a.data, b.data)
    assert o2.autoscale_multiplier() > 1.0


def test_growth_rate_caps_growth_after_the_first_estimate():
    """growth_rate bounds how fast d GROWS, not the initial jump.

    D-Adaptation's first step takes d = max(d0, d_hat) outright — d0 is a deliberately
    tiny placeholder (1e-6), so clamping that first estimate would just pin the run at
    a step size chosen by the placeholder. The rate limiter applies from step 2 on, and
    that is the property worth asserting; this mirrors prodigyopt and FusionOpt.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 16))
    o = LionSR([p], lr=1e-3, autoscale=True, autoscale_growth_rate=1.01)
    seen = []
    for g in _grads((16, 16), 10, seed=3):
        p.grad = g.clone()
        o.step()
        seen.append(o.autoscale_multiplier())
    for prev, cur in zip(seen[1:], seen[2:]):
        assert cur <= prev * 1.01 + 1e-9, f"{cur} grew faster than 1.01x from {prev}"

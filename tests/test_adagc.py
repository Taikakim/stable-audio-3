"""AdaGC per-tensor adaptive gradient clipping (stable_audio_3/training/adagc.py). CPU only.

AdaGC (arXiv 2502.11034): each tensor's gradient is scaled by min(1, lambda * gamma / ||g||), where
gamma is an EMA of that tensor's CLIPPED gradient norms, so a spike cannot raise its own threshold.
Here it runs before the optimizer step, i.e. before the gradient reaches momentum AND the Shampoo
accumulators (training-findings 13e/20a context, papers/deep-research/2026-09-26-*).
"""
import math

import torch

from stable_audio_3.training.adagc import AdaGC


def _params(n=3):
    torch.manual_seed(0)
    return [(f"blk.{i}.lora_B", torch.nn.Parameter(torch.zeros(4, 4))) for i in range(n)]


def _grads(named, scales):
    for (_, p), s in zip(named, scales):
        p.grad = torch.full_like(p, float(s))


def _norm(p):
    return float(p.grad.norm())


def test_warmup_only_observes():
    named = _params()
    ag = AdaGC(named, lam=1.04, beta=0.99, warmup=5)
    for _ in range(5):
        _grads(named, [1.0, 2.0, 3.0])
        ag.clip_()
        assert [_norm(p) for _, p in named] == [4.0, 8.0, 12.0]  # untouched during warmup
    assert ag.n_seen == 5


def test_spike_is_clipped_to_lambda_times_its_own_average():
    named = _params(1)
    ag = AdaGC(named, lam=2.0, beta=0.99, warmup=3)
    for _ in range(3):
        _grads(named, [1.0])  # norm 4
        ag.clip_()
    _grads(named, [42.0])  # norm 168, a 42x spike
    ag.clip_()
    assert math.isclose(_norm(named[0][1]), 2.0 * 4.0, rel_tol=1e-5)


def test_spike_does_not_raise_its_own_threshold():
    """The EMA is fed the CLIPPED norm, so after a spike the threshold barely moves."""
    named = _params(1)
    ag = AdaGC(named, lam=2.0, beta=0.9, warmup=3)
    for _ in range(3):
        _grads(named, [1.0])
        ag.clip_()
    before = float(ag.gamma[0])
    _grads(named, [1000.0])
    ag.clip_()
    after = float(ag.gamma[0])
    # clipped norm is 2*before, so gamma moves by (1-beta)*(2*before - before) at most
    assert after <= before * 1.1 + 1e-6


def test_normal_gradients_pass_untouched_with_rare_event_lambda():
    named = _params(2)
    ag = AdaGC(named, lam=3.0, beta=0.99, warmup=4)
    g = torch.Generator().manual_seed(1)
    for _ in range(4):
        _grads(named, [1.0, 1.0])
        ag.clip_()
    for _ in range(20):
        s = 1.0 + 0.2 * float(torch.randn(1, generator=g))
        _grads(named, [s, s])
        ag.clip_()
        assert math.isclose(_norm(named[0][1]), abs(s) * 4.0, rel_tol=1e-5)


def test_tensor_with_no_history_is_never_zeroed():
    """LoRA A gets an all-zero gradient at first (B starts at 0): its average stays 0, and a
    later real gradient must not be multiplied by lambda*0."""
    named = _params(2)
    ag = AdaGC(named, lam=1.04, beta=0.99, warmup=3)
    for _ in range(3):
        _grads(named, [0.0, 1.0])
        ag.clip_()
    _grads(named, [5.0, 1.0])
    ag.clip_()
    assert math.isclose(_norm(named[0][1]), 20.0, rel_tol=1e-5)


def test_nonfinite_grad_is_left_alone_and_not_averaged():
    named = _params(1)
    ag = AdaGC(named, lam=2.0, beta=0.99, warmup=2)
    for _ in range(2):
        _grads(named, [1.0])
        ag.clip_()
    before = float(ag.gamma[0])
    named[0][1].grad = torch.full((4, 4), float("nan"))
    ag.clip_()
    assert torch.isnan(named[0][1].grad).all()  # left for the loss guard, not silently rescaled
    assert float(ag.gamma[0]) == before


def test_missing_grad_is_skipped():
    named = _params(2)
    ag = AdaGC(named, lam=2.0, beta=0.99, warmup=1)
    _grads(named, [1.0, 1.0])
    ag.clip_()
    named[1][1].grad = None
    _grads(named[:1], [50.0])
    ag.clip_()  # must not raise
    assert _norm(named[0][1]) < 50.0 * 4.0


def test_stats_report_clip_fraction():
    named = _params(2)
    ag = AdaGC(named, lam=2.0, beta=0.99, warmup=2)
    for _ in range(2):
        _grads(named, [1.0, 1.0])
        ag.clip_()
    _grads(named, [10.0, 1.0])
    ag.clip_()
    s = ag.stats()
    assert s["adagc/clipped_frac"] == 0.5
    assert math.isclose(s["adagc/min_factor"], 0.2, rel_tol=1e-4)


def test_state_dict_roundtrip_keeps_averages():
    named = _params(2)
    ag = AdaGC(named, lam=2.0, beta=0.99, warmup=2)
    for _ in range(4):
        _grads(named, [1.0, 3.0])
        ag.clip_()
    sd = ag.state_dict()
    ag2 = AdaGC(_params(2), lam=2.0, beta=0.99, warmup=2)
    ag2.load_state_dict(sd)
    assert ag2.n_seen == ag.n_seen
    assert torch.allclose(ag2.gamma, ag.gamma)


def test_wrapper_hook_clips_in_adagc_mode(tmp_path):
    """The REAL DiffusionCondTrainingWrapper.configure_gradient_clipping, on a stand-in module."""
    from types import SimpleNamespace
    from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper

    diffusion = torch.nn.Module()
    diffusion.blk = torch.nn.Module()
    diffusion.blk.lora_B = torch.nn.Parameter(torch.zeros(4, 4))
    fake = SimpleNamespace(grad_clip_mode="adagc", adagc_cfg=dict(lam=2.0, beta=0.99, warmup=2),
                           diffusion=diffusion, global_step=0, logger=None,
                           _staggered_logger=SimpleNamespace(every_n_steps=1000))
    opt = torch.optim.SGD(diffusion.parameters(), lr=0.1)
    for _ in range(2):
        diffusion.blk.lora_B.grad = torch.ones(4, 4)
        DiffusionCondTrainingWrapper.configure_gradient_clipping(fake, opt)
    diffusion.blk.lora_B.grad = torch.full((4, 4), 42.0)
    DiffusionCondTrainingWrapper.configure_gradient_clipping(fake, opt)
    assert math.isclose(float(diffusion.blk.lora_B.grad.norm()), 8.0, rel_tol=1e-5)


def test_checkpoint_carries_adagc_state():
    """on_save_checkpoint rebuilds LoRA checkpoints from a fixed key list; AdaGC state must survive."""
    from types import SimpleNamespace
    from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper

    named = _params(1)
    ag = AdaGC(named, lam=2.0, beta=0.99, warmup=1)
    _grads(named, [1.0])
    ag.clip_()
    fake = SimpleNamespace(lora_config=None, _adagc=ag)
    ck = {}
    DiffusionCondTrainingWrapper.on_save_checkpoint(fake, ck)
    assert "adagc_state" in ck and ck["adagc_state"]["n_seen"] == 1

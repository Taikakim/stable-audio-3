# tests/test_latch_spectral_probe.py
# Owner: KUANG (created 2026-10-17)
"""scripts/latch/spectral_probe.py core on a tiny LatCH head and synthetic batches (CPU): alpha = 0 reproduces
the reference loss, the module is never mutated, probes are descent directions where expected, and the
direction-agreement block is produced. Needs the C-series stable_audio_tools.training.modular_opt."""
import pytest
import torch

modular_opt = pytest.importorskip("stable_audio_tools.training.modular_opt")
if not hasattr(modular_opt, "direction_agreement"):
    pytest.skip("stable_audio_tools without the C-series modular_opt", allow_module_level=True)

from scripts.latch.latch_model import LatCH  # noqa: E402
from scripts.latch.spectral_probe import (  # noqa: E402
    make_draws, mean_grad, mean_loss, probe_direction, run_probe, spectral_bases,
)
from scripts.latch.train_latch import masked_mse  # noqa: E402


def _batches(n, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        lat = torch.randn(4, 256, 16, generator=g)
        tgt = lat[:, :1] * 0.5 + 0.1 * torch.randn(4, 1, 16, generator=g)   # a learnable target
        out.append({"latents": lat, "targets": tgt, "mask": torch.ones(4, 16, dtype=torch.bool),
                    "weight": torch.ones(4, 16)})
    return out


def _setup():
    torch.manual_seed(0)
    model = LatCH(in_channels=256, out_channels=1, dim=64, depth=1, num_heads=2, t_injection="adaln_zero").eval()
    spectral = [n for n, p in model.named_parameters() if p.ndim == 2 and min(p.shape) >= 64]
    point = {n: p.detach().clone() for n, p in model.named_parameters()}
    tr = _batches(6, 1)
    momenta = {n: torch.zeros_like(point[n]) for n in spectral}
    for b, d in zip(tr, make_draws(tr, 3)):
        g = mean_grad(model, point, spectral, [b], [d], masked_mse)
        for n in spectral:
            momenta[n].mul_(0.9).add_(g[n], alpha=0.1)
    return model, point, spectral, momenta


def test_mean_grad_matches_module_autograd():
    model, point, spectral, _ = _setup()
    b = _batches(1, 9)
    d = make_draws(b, 4)
    g = mean_grad(model, point, spectral, b, d, masked_mse)
    from scripts.latch.spectral_probe import batch_loss
    model.zero_grad()
    batch_loss(model, dict(model.named_parameters()), b[0], d[0], masked_mse).backward()
    for n, p in model.named_parameters():
        if n in spectral:
            assert torch.allclose(g[n], p.grad, atol=1e-6)


def test_probe_direction_full_is_scaled_polar():
    _, _, _, momenta = _setup()
    bases = spectral_bases(momenta)
    P = probe_direction(bases, beyond_k=0)
    live = 0
    for n, M in momenta.items():
        if float(M.abs().max()) == 0.0:           # behind a zero-initialised adaLN gate: no direction
            assert float(P[n].abs().max()) == 0.0
            continue
        live += 1
        m, k = M.shape
        s = max(1.0, (m / k) ** 0.5)
        sv = torch.linalg.svdvals(P[n] / s)
        r = int((torch.linalg.svdvals(M) > 1e-6 * torch.linalg.svdvals(M)[0]).sum())
        assert torch.allclose(sv[:r], torch.ones(r), atol=1e-4)
        assert float((P[n] * M).sum()) < 0        # descent direction for the momentum
    assert live >= 1


def test_run_probe_reproduces_reference_and_never_mutates():
    model, point, spectral, momenta = _setup()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ev = _batches(2, 5)
    eval_sets = [(ev, make_draws(ev, 11))]
    grp = (( _batches(2, 6), make_draws(_batches(2, 6), 12)), (_batches(2, 7), make_draws(_batches(2, 7), 13)))
    out = run_probe(model, point, spectral, momenta, 1e-2, eval_sets, masked_mse, grad_groups=grp,
                    ranks=(1, 2), bands=((1, 1), (2, 4)), k_resid=4, grid=(0.0, 1.0, 4.0))
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    ref = out["reference_loss"][0]
    assert ref == pytest.approx(mean_loss(model, point, ev, eval_sets[0][1], masked_mse))
    for name, pr in out["probes"].items():
        assert pr["losses"][0][0] == pytest.approx(ref, rel=1e-6)
        assert pr["best_alpha"][0] in (0.0, 1.0, 4.0)
    assert set(out["probes"]) >= {"rank1", "rank2", "band1-1", "band2-4", "full", "beyond_4"}
    agree = out["direction_agreement"]
    assert set(agree) == set(spectral)
    assert "residual" in next(iter(agree.values()))


def test_point_from_state_gives_gradient_point_and_run_momentum():
    """A short real ModularOptimizer run; the probe must recover y = (1 - b) z + b x (the params the forward
    pass saw during training) and the optimizer's own momentum, without touching the model."""
    from stable_audio_tools.training.modular_opt import ModularOptimizer, build_modular_param_groups
    from scripts.latch.spectral_probe import batch_loss, point_from_state
    model, _, spectral, _ = _setup()
    model.train()
    groups = build_modular_param_groups(model)
    opt = ModularOptimizer(groups, lr=1e-2)
    for b, d in zip(_batches(4, 21), make_draws(_batches(4, 21), 22)):
        opt.zero_grad()
        batch_loss(model, dict(model.named_parameters()), b, d, masked_mse).backward()
        opt.step()
    y_live = {n: p.detach().clone() for n, p in model.named_parameters()}   # train mode: params hold y
    sd_opt = opt.state_dict()
    model2 = LatCH(in_channels=256, out_channels=1, dim=64, depth=1, num_heads=2, t_injection="adaln_zero")
    model2.load_state_dict(model.state_dict())
    point, momenta, lr = point_from_state(model2, build_modular_param_groups(model2), sd_opt, "y", spectral)
    assert lr == 1e-2
    for n in y_live:
        assert torch.allclose(point[n], y_live[n], atol=1e-6)
    for g in opt.param_groups:
        for n, p in zip(g["param_names"], g["params"]):
            if n in momenta:
                assert torch.equal(momenta[n], opt.state[p]["momentum"].float())

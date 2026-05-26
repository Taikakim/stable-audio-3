# tests/test_latch_guided.py
import torch

from stable_audio_3.inference.latch_guided import sample_flow_euler_latch_guided


def _toy_model(x, t, **kw):
    # Constant velocity field: returns x (so denoised = x - t*x).
    return x


class _ConstHead(torch.nn.Module):
    """Predicts the per-frame mean of the latent; differentiable, channel-collapsing."""
    def forward(self, z, t):
        return z.mean(dim=1, keepdim=True)  # (B, 1, T)


def _schedule(steps):
    return torch.linspace(1.0, 0.0, steps + 1)


def test_zero_gain_matches_plain_euler():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    sigmas = _schedule(6)
    head = _ConstHead()
    target = torch.zeros(1, 1, 8)
    guided = sample_flow_euler_latch_guided(
        _toy_model, x.clone(), sigmas, head=head, target=target,
        rho=0.0, mu=0.0, window=(0.0, 1.0),
    )
    # Replicate plain euler.
    y = x.clone()
    for i in range(6):
        v = _toy_model(y, sigmas[i].expand(1))
        y = y + (sigmas[i + 1] - sigmas[i]) * v
    assert torch.allclose(guided, y, atol=1e-5)


def test_positive_gain_moves_toward_target():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    sigmas = _schedule(20)
    head = _ConstHead()
    target = torch.full((1, 1, 8), -5.0)   # push the latent mean down
    out_unguided = sample_flow_euler_latch_guided(
        _toy_model, x.clone(), sigmas, head=head, target=target,
        rho=0.0, mu=0.0, window=(0.0, 1.0))
    out_guided = sample_flow_euler_latch_guided(
        _toy_model, x.clone(), sigmas, head=head, target=target,
        rho=0.0, mu=5.0, window=(0.0, 1.0))
    # Mean guidance toward a negative target should lower the predicted mean.
    assert head(out_guided, torch.zeros(1)).mean() < head(out_unguided, torch.zeros(1)).mean()


def test_variance_guidance_path_moves_toward_target():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    sigmas = _schedule(20)
    head = _ConstHead()
    target = torch.full((1, 1, 8), -5.0)
    out_unguided = sample_flow_euler_latch_guided(
        _toy_model, x.clone(), sigmas, head=head, target=target,
        rho=0.0, mu=0.0, window=(0.0, 1.0))
    out_var = sample_flow_euler_latch_guided(
        _toy_model, x.clone(), sigmas, head=head, target=target,
        rho=5.0, mu=0.0, window=(0.0, 1.0))
    # Variance guidance (rho path) toward a negative target should lower the predicted mean.
    assert head(out_var, torch.zeros(1)).mean() < head(out_unguided, torch.zeros(1)).mean()


def test_per_element_schedule_zero_gain_matches_plain_euler():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    sigmas_1d = _schedule(6)
    sigmas_2d = sigmas_1d.unsqueeze(0).expand(2, -1).contiguous()  # (2, steps+1)
    head = _ConstHead()
    target = torch.zeros(2, 1, 8)
    guided = sample_flow_euler_latch_guided(
        _toy_model, x.clone(), sigmas_2d, head=head, target=target,
        rho=0.0, mu=0.0, window=(0.0, 1.0))
    # Plain euler with per-element dt.
    y = x.clone()
    for i in range(6):
        v = _toy_model(y, sigmas_2d[:, i])
        dt = (sigmas_2d[:, i + 1] - sigmas_2d[:, i]).view(-1, 1, 1)
        y = y + dt * v
    assert torch.allclose(guided, y, atol=1e-5)

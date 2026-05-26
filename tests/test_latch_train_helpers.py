# tests/test_latch_train_helpers.py
import torch

from scripts.latch.train_latch import forward_noise, masked_mse


def test_forward_noise_linear_endpoints():
    z0 = torch.ones(2, 256, 10)
    noise = torch.zeros(2, 256, 10)
    # t=0 -> clean z0
    zt0 = forward_noise(z0, noise, torch.zeros(2))
    assert torch.allclose(zt0, z0, atol=1e-6)
    # t=1 -> pure noise (here zeros)
    zt1 = forward_noise(z0, noise, torch.ones(2))
    assert torch.allclose(zt1, noise, atol=1e-6)


def test_masked_mse_ignores_padding():
    pred = torch.zeros(1, 1, 4)
    target = torch.tensor([[[0.0, 0.0, 100.0, 100.0]]])  # last 2 are "padding"
    mask = torch.tensor([[True, True, False, False]])
    loss = masked_mse(pred, target, mask)
    assert loss.item() == 0.0  # padded huge errors excluded

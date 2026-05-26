# tests/test_latch_model.py
import torch

from scripts.latch.latch_model import LatCH


def test_head_accepts_256_channels_and_returns_out_channels():
    head = LatCH(in_channels=256, out_channels=1, dim=128, depth=2, num_heads=4)
    x = torch.randn(2, 256, 128)         # (B, 256, T)
    t = torch.rand(2)                    # (B,)
    out = head(x, t)
    assert out.shape == (2, 1, 128)      # (B, out_channels, T)


def test_head_is_length_agnostic():
    head = LatCH(in_channels=256, out_channels=1, dim=128, depth=2, num_heads=4)
    t = torch.rand(1)
    out_short = head(torch.randn(1, 256, 64), t)
    out_long = head(torch.randn(1, 256, 300), t)
    assert out_short.shape == (1, 1, 64)
    assert out_long.shape == (1, 1, 300)


def test_head_multichannel_output():
    head = LatCH(in_channels=256, out_channels=12, dim=128, depth=2, num_heads=4)
    out = head(torch.randn(1, 256, 80), torch.rand(1))
    assert out.shape == (1, 12, 80)

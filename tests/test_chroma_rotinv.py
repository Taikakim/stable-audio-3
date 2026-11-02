"""EXPERIMENTS E22: chroma_loss_rotinv is invariant to transposing the prediction, and still separates shapes."""
import torch

from stable_audio_3.inference.chroma_losses import _roll_target, chroma_loss_rotinv
from stable_audio_3.inference.latch_guided import _make_latch_criterion


def _shape12(pcs, T=8):
    v = torch.zeros(1, 12, T)
    v[:, pcs, :] = 1.0
    return v


def test_hpcp_layout_is_transposition_invariant():
    tgt = _shape12([0, 4, 7])                          # major triad on C
    for k in range(12):
        pred = torch.roll(tgt, k, dims=1)              # the same triad on any root
        assert chroma_loss_rotinv(pred, tgt, tau=0.01) < 0.05


def test_shape_still_matters():
    major, minor_ish = _shape12([0, 4, 7]), _shape12([0, 3, 6])   # diminished is not a rotation of major
    assert chroma_loss_rotinv(minor_ish, major, tau=0.01) > chroma_loss_rotinv(torch.roll(major, 5, 1), major, tau=0.01) + 0.2


def test_same_layout_roll_by_twelve_semitones_is_identity_and_invariance_holds():
    torch.manual_seed(0)
    tgt = torch.rand(1, 384, 6)
    assert torch.allclose(_roll_target(tgt, 12).reshape(1, 3, 128, 6), tgt.reshape(1, 3, 128, 6), atol=1e-5)
    pred = _roll_target(tgt, 3)                        # integer-semitone shift of the 128-bin bands (fractional bins)
    assert chroma_loss_rotinv(pred, tgt, tau=0.01) < chroma_loss_rotinv(torch.rand(1, 384, 6), tgt, tau=0.01)


def test_gradient_flows_and_criterion_is_registered():
    pred = torch.rand(2, 12, 5, requires_grad=True)
    crit = _make_latch_criterion("cosine_rotinv", huber_beta=0.05)
    crit(pred, _shape12([0, 4, 7], 5).repeat(2, 1, 1)).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0

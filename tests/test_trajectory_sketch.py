# tests/test_trajectory_sketch.py
"""The sketch is only useful if inner products survive it — that is the whole claim
("compressed way to store movement", Kim 2026-08-18): every trajectory statistic we compute
(cosines, path efficiency, autocorrelation, PCA) is a function of the Gram matrix, and a
CountSketch preserves the Gram to ~1/sqrt(k). Test that, not the plumbing."""
import math

import torch

from scripts.trajectory_sketch import CountSketch


def test_sketch_preserves_inner_products_within_tolerance():
    torch.manual_seed(0)
    D, k, s = 200_000, 1024, 4
    sk = CountSketch(D, k=k, hashes=s, seed=1, device="cpu")
    errs = []
    for _ in range(6):
        x = torch.randn(D); y = torch.randn(D) * 0.5 + 0.3 * x      # correlated pair
        est = float(sk(x) @ sk(y)); true = float(x @ y)
        errs.append(abs(est - true) / (x.norm() * y.norm()))
    # relative-to-norms error ~ 1/sqrt(k*s) = 1/64 ≈ 0.016 per estimate; allow 4 sigma
    assert max(errs) < 0.07, errs


def test_sketch_preserves_norm_and_is_linear():
    torch.manual_seed(1)
    D = 50_000
    sk = CountSketch(D, k=2048, hashes=2, seed=3, device="cpu")
    x = torch.randn(D); y = torch.randn(D)
    assert abs(sk(x).norm() / x.norm() - 1.0) < 0.05
    assert torch.allclose(sk(2 * x - y), 2 * sk(x) - sk(y), atol=1e-4)


def test_sketch_dim_is_hashes_times_buckets_and_deterministic():
    a = CountSketch(1000, k=64, hashes=3, seed=7, device="cpu")
    b = CountSketch(1000, k=64, hashes=3, seed=7, device="cpu")
    x = torch.randn(1000)
    assert a(x).shape == (192,)
    assert torch.equal(a(x), b(x))          # same seed -> same projection across processes


def test_orthogonal_steps_read_as_orthogonal_after_sketch():
    """The random-walk floor test again, through the sketch: three mutually orthogonal
    updates must come out near-orthogonal, so path efficiency reads ~1/sqrt(3)."""
    torch.manual_seed(2)
    D = 100_000
    sk = CountSketch(D, k=1024, hashes=4, seed=5, device="cpu")
    U = torch.zeros(3, D)
    U[0, :30000] = torch.randn(30000); U[1, 30000:60000] = torch.randn(30000); U[2, 60000:] = torch.randn(40000)
    S = torch.stack([sk(u) for u in U])
    C = (S @ S.T) / (S.norm(dim=1)[:, None] * S.norm(dim=1)[None, :])
    off = C - torch.eye(3)
    assert off.abs().max() < 0.06
    net = S.sum(0).norm(); path = S.norm(dim=1).sum()
    assert abs(net / path - 1 / math.sqrt(3)) < 0.05

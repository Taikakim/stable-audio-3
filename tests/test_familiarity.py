"""Tests for scripts/familiarity.py — per-crop familiarity-normalized loss
weighting (Kim 2026-07-07: down-weight updates for material the model already
renders well, let remote areas keep full gradient)."""
import sys, os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from familiarity import FamiliarityReweighter


def test_first_visit_all_ones():
    r = FamiliarityReweighter(beta=1.0)
    w = r.weights(["a", "b", "c"], [0.5, 1.0, 2.0])
    assert w == [1.0, 1.0, 1.0]


def test_familiar_down_remote_up_mean_one():
    r = FamiliarityReweighter(beta=1.0, decay=0.5)
    # crop "easy" consistently low relative loss, "hard" consistently high
    for _ in range(5):
        r.weights(["easy", "hard"], [0.5, 1.5])
    w = r.weights(["easy", "hard"], [0.5, 1.5])
    assert w[0] < 1.0 < w[1]
    assert sum(w) / len(w) == pytest.approx(1.0)


def test_beta_zero_disables():
    r = FamiliarityReweighter(beta=0.0)
    for _ in range(3):
        r.weights(["a", "b"], [0.1, 10.0])
    assert r.weights(["a", "b"], [0.1, 10.0]) == [1.0, 1.0]


def test_clipping_bounds_extremes():
    r = FamiliarityReweighter(beta=1.0, decay=0.0, clip=(0.25, 4.0))
    for _ in range(3):
        r.weights(["tiny", "huge"], [0.001, 100.0])
    w = r.weights(["tiny", "huge"], [0.001, 100.0])
    # pre-normalization values are clipped; ratio can't exceed clip range ratio
    assert max(w) / min(w) <= 4.0 / 0.25 + 1e-6


def test_new_crop_neutral_among_known():
    r = FamiliarityReweighter(beta=1.0, decay=0.5)
    for _ in range(4):
        r.weights(["a", "b"], [0.5, 1.5])
    w = r.weights(["a", "b", "new"], [0.5, 1.5, 1.0])
    # the unseen crop gets weight 1.0 pre-normalization (neutral, not extreme)
    assert w[0] < w[2] < w[1]


def test_mixed_batch_mean_stays_one():
    r = FamiliarityReweighter(beta=2.0, decay=0.3)
    import random
    rng = random.Random(0)
    ids = [f"c{i}" for i in range(8)]
    for _ in range(10):
        losses = [rng.uniform(0.2, 2.0) for _ in ids]
        w = r.weights(ids, losses)
        assert sum(w) / len(w) == pytest.approx(1.0)

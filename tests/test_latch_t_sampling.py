# Owner: WINTERMUTE (created 2026-10-25)
"""train_latch.sample_t: stratified / antithetic training t keep the U(0,1) marginal and cover [0,1) per micro-batch."""
import torch

from scripts.latch.train_latch import sample_t


def test_iid_shape_and_range():
    t = sample_t(32, "iid")
    assert t.shape == (32,) and float(t.min()) >= 0 and float(t.max()) < 1


def test_stratified_one_draw_per_bin():
    torch.manual_seed(0)
    for n in (1, 7, 32):
        t = sample_t(n, "stratified")
        assert t.shape == (n,)
        bins = torch.floor(t * n).long().sort().values
        assert torch.equal(bins, torch.arange(n))


def test_stratified_is_shuffled_and_uniform():
    torch.manual_seed(1)
    t = torch.stack([sample_t(32, "stratified") for _ in range(2000)])
    assert abs(float(t.mean()) - 0.5) < 0.01
    # position in the batch carries no information about t
    assert float((t[:, 0].mean() - t[:, -1].mean()).abs()) < 0.03


def test_antithetic_pairs():
    torch.manual_seed(2)
    t = sample_t(32, "antithetic")
    assert t.shape == (32,)
    assert abs(float(t.sum()) - 16.0) < 1e-5
    assert sample_t(5, "antithetic").shape == (5,)


def test_unknown_mode_raises():
    try:
        sample_t(4, "sobol")
    except ValueError:
        return
    raise AssertionError("no error")

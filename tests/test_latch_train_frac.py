# Owner: WINTERMUTE (created 2026-10-25)
"""--train-frac: subsample_groups keeps whole groups, a seeded fraction of them, in the original order."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.latch.train_latch import subsample_groups, split_indices  # noqa: E402


def _groups():
    return [f"track{i // 3}" for i in range(300)]   # 100 tracks x 3 crops


def test_full_frac_is_identity():
    g = _groups()
    idx = list(range(0, 300, 2))
    assert subsample_groups(g, idx, 1.0) == idx


def test_keeps_whole_groups_and_fraction():
    g = _groups()
    tr, _ = split_indices(g, 0.05, seed=0)
    kept = subsample_groups(g, tr, 0.2, seed=0)
    kg = {g[i] for i in kept}
    assert len(kg) == round(0.2 * len({g[i] for i in tr}))
    assert all(g[i] in kg for i in tr if g[i] in kg) and len(kept) == 3 * len(kg)
    assert kept == sorted(kept) and set(kept) <= set(tr)


def test_seeded_and_seed_dependent():
    g = _groups()
    tr = list(range(300))
    assert subsample_groups(g, tr, 0.2, seed=0) == subsample_groups(g, tr, 0.2, seed=0)
    assert subsample_groups(g, tr, 0.2, seed=0) != subsample_groups(g, tr, 0.2, seed=1)

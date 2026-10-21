# Owner: WINTERMUTE (created 2026-10-21)
"""CPU tests for scripts/latch/norm_watchdog.py (per-group weight-RMS lr watchdog)."""

import torch

from scripts.latch.norm_watchdog import GroupNormWatchdog


def _groups(*vals, lr=1e-2):
    return [{"group_type": f"g{i}", "lr": lr, "params": [torch.full((4, 4), float(v))]} for i, v in enumerate(vals)]


def _drive(wd, groups, growth, steps, first=1):
    """Grow group i's weights by growth[i] per step from step `first`; collect all cuts."""
    cuts = []
    for s in range(first, first + steps):
        for g, k in zip(groups, growth):
            g["params"][0].mul_(k)
        cuts += wd.step(s, groups)
    return cuts


def test_fast_growth_halves_only_that_group():
    groups = _groups(0.1, 0.1)
    wd = GroupNormWatchdog(2, ratio=2.0, window=100, start=0, check_every=10)
    # the first check with a full window behind it is step 110 (history starts at step 10)
    cuts = _drive(wd, groups, growth=(1.01, 1.001), steps=110)   # 1.01^100 = 2.7x vs 1.001^100 = 1.1x
    assert [c["group"] for c in cuts] == ["g0"]
    assert groups[0]["lr"] == 5e-3 and groups[1]["lr"] == 1e-2
    assert abs(cuts[0]["ratio"] - 1.01 ** 100) < 1e-3


def test_slow_growth_never_trips():
    groups = _groups(0.1)
    wd = GroupNormWatchdog(1, ratio=2.0, window=100, start=0, check_every=10)
    assert _drive(wd, groups, growth=(1.005,), steps=1000) == []       # 1.65x per window
    assert groups[0]["lr"] == 1e-2


def test_nothing_before_start():
    groups = _groups(0.1)
    wd = GroupNormWatchdog(1, ratio=2.0, window=10, start=500, check_every=5)
    assert _drive(wd, groups, growth=(1.2,), steps=499) == []


def test_history_restarts_after_a_cut():
    groups = _groups(0.1)
    wd = GroupNormWatchdog(1, ratio=2.0, window=100, start=0, check_every=10)
    cuts = _drive(wd, groups, growth=(1.01,), steps=200)
    assert [c["step"] for c in cuts] == [110]          # the next cut needs another full window
    cuts += _drive(wd, groups, growth=(1.01,), steps=20, first=201)
    assert [c["step"] for c in cuts] == [110, 210]
    assert abs(groups[0]["lr"] - 2.5e-3) < 1e-12


def test_min_scale_floor():
    groups = _groups(0.1)
    wd = GroupNormWatchdog(1, ratio=2.0, window=10, start=0, check_every=10, min_scale=0.25)
    _drive(wd, groups, growth=(1.2,), steps=200)
    assert abs(groups[0]["lr"] - 2.5e-3) < 1e-12        # two halvings, then the floor holds
    assert len(wd.events) == 2


def test_rejects_ratio_at_or_below_one():
    try:
        GroupNormWatchdog(1, ratio=1.0)
    except ValueError:
        return
    raise AssertionError("ratio 1.0 accepted")

"""Tests for scripts/weight_mutations.py — the pure mutation core behind
scripts/mutate_weights.py. CPU-only, no model checkpoints needed."""
import sys, os, re
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from weight_mutations import (
    decay_profile,
    classify_param,
    collect_targets,
    snapshot_targets,
    restore_targets,
    drift,
    shuffle,
    blur,
    contrast,
    life_step,
    spectral_tilt,
)


def g(seed=0):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def w0(seed=7, shape=(32, 48)):
    return torch.randn(*shape, generator=g(seed))


# ---------------------------------------------------------------- decay

def test_decay_late_direction_peaks_at_last_block():
    d = decay_profile(24, rate=0.5, direction="late")
    assert d[23] == pytest.approx(1.0)
    assert d[0] < d[12] < d[23]


def test_decay_early_direction_peaks_at_first_block():
    d = decay_profile(24, rate=0.5, direction="early")
    assert d[0] == pytest.approx(1.0)
    assert d[23] < d[12] < d[0]


def test_decay_flat():
    d = decay_profile(24, rate=0.5, direction="flat")
    assert all(x == pytest.approx(1.0) for x in d)


def test_decay_focus_bump_peaks_at_focus():
    d = decay_profile(24, rate=0.5, direction="focus", focus=10, width=3.0)
    assert d[10] == pytest.approx(1.0)
    assert d[10] > d[5] > d[0]
    assert d[10] > d[15] > d[23]


# ---------------------------------------------------------------- classify

@pytest.mark.parametrize("name,kind", [
    ("model.transformer.layers.3.self_attn.to_qkv.weight", "attn"),
    ("model.model.blocks.3.attn.to_qkv.weight", "attn"),
    ("model.model.blocks.3.attn.to_out.weight", "attn"),
    ("model.model.blocks.3.cross_attn.to_kv.weight", "attn"),
    ("model.model.blocks.3.ff.ff.0.0.weight", "mlp"),
    ("model.model.blocks.3.pre_norm.gamma", "norm"),
    ("model.model.blocks.3.ff_norm.gamma", "norm"),
])
def test_classify_param(name, kind):
    assert classify_param(name) == kind


# ---------------------------------------------------------------- targets

class FakeAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.to_qkv = nn.Linear(d, d * 3, bias=False)
        self.to_out = nn.Linear(d, d, bias=False)


class FakeBlock(nn.Module):
    """Mirrors the real TransformerBlock member names (self_attn/ff/pre_norm),
    inside a `layers` ModuleList like ContinuousTransformer."""
    def __init__(self, d):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d)
        self.self_attn = FakeAttn(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))


class FakeDiT(nn.Module):
    def __init__(self, d=8, n=4):
        super().__init__()
        self.project_in = nn.Linear(d, d)
        self.layers = nn.ModuleList(FakeBlock(d) for _ in range(n))
        self.project_out = nn.Linear(d, d)


def test_collect_targets_finds_only_block_linears_with_indices():
    m = FakeDiT(n=4)
    targets = collect_targets(m, target="all")
    names = [t.name for t in targets]
    assert all(re.match(r"layers\.\d+\.", n) for n in names)
    assert not any("project_in" in n or "project_out" in n for n in names)
    assert {t.block for t in targets} == {0, 1, 2, 3}
    assert max(t.block for t in targets) == 3


def test_collect_targets_attn_vs_mlp_filter():
    m = FakeDiT(n=2)
    attn = collect_targets(m, target="attn")
    mlp = collect_targets(m, target="mlp")
    norm = collect_targets(m, target="norm")
    assert attn and all("attn" in t.name for t in attn)
    assert mlp and all(".ff." in t.name for t in mlp)
    assert norm and all("norm" in t.name for t in norm)
    assert len(attn) + len(mlp) + len(norm) == len(collect_targets(m, target="all"))


def test_snapshot_restore_roundtrip():
    m = FakeDiT(n=2)
    targets = collect_targets(m, target="all")
    snap = snapshot_targets(targets)
    with torch.no_grad():
        for t in targets:
            t.param.add_(1.0)
    assert not torch.equal(targets[0].param, snap[targets[0].name])
    restore_targets(targets, snap)
    for t in targets:
        assert torch.equal(t.param, snap[t.name])


# ---------------------------------------------------------------- drift

def test_drift_zero_amount_is_identity():
    w = w0()
    assert torch.equal(drift(w, 0.0, g(1)), w)


def test_drift_seeded_reproducible():
    w = w0()
    a = drift(w, 0.1, g(42))
    b = drift(w, 0.1, g(42))
    c = drift(w, 0.1, g(43))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_drift_magnitude_relative_to_std():
    w = w0() * 3.0  # std ~3
    out = drift(w, 0.1, g(1))
    delta_std = (out - w).std().item()
    assert delta_std == pytest.approx(0.1 * w.std().item(), rel=0.15)


# ---------------------------------------------------------------- shuffle

def test_shuffle_preserves_values_moves_fraction():
    w = w0()
    out = shuffle(w, 0.2, g(5))
    assert torch.equal(out.flatten().sort().values, w.flatten().sort().values)
    moved = (out != w).float().mean().item()
    assert 0.05 < moved <= 0.45  # ~2*frac positions change, minus fixed points


def test_shuffle_zero_is_identity_and_seeded():
    w = w0()
    assert torch.equal(shuffle(w, 0.0, g(1)), w)
    assert torch.equal(shuffle(w, 0.3, g(9)), shuffle(w, 0.3, g(9)))


# ---------------------------------------------------------------- blur

def test_blur_zero_is_identity():
    w = w0()
    assert torch.equal(blur(w, 0.0, axis=1), w)


def test_blur_reduces_adjacent_variation_along_axis():
    w = w0()
    out = blur(w, 0.8, axis=1)
    rough_in = (w[:, 1:] - w[:, :-1]).abs().mean()
    rough_out = (out[:, 1:] - out[:, :-1]).abs().mean()
    assert rough_out < rough_in


def test_blur_axis0_differs_from_axis1():
    w = w0()
    assert not torch.equal(blur(w, 0.5, axis=0), blur(w, 0.5, axis=1))


# ---------------------------------------------------------------- contrast

def test_contrast_positive_stretches_negative_flattens():
    w = w0()
    hi = contrast(w, 0.5)
    lo = contrast(w, -0.5)
    assert hi.std() > w.std() > lo.std()
    assert hi.mean().item() == pytest.approx(w.mean().item(), abs=1e-5)
    assert lo.mean().item() == pytest.approx(w.mean().item(), abs=1e-5)


def test_contrast_minus_one_flattens_to_mean():
    w = w0()
    flat = contrast(w, -1.0)
    assert flat.std().item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- game of life

def test_life_step_kills_isolated_and_keeps_supported():
    # 5x5 grid, threshold at 0.5: strong=1.0 alive, weak=0.01 dead
    w = torch.full((5, 5), 0.01)
    w[0, 0] = 1.0                      # isolated: 0 alive neighbours -> dies
    w[2, 1] = w[2, 2] = w[2, 3] = 1.0  # blinker row: centre has 2 -> survives
    out = life_step(w, alive_thresh=0.5)
    assert out[0, 0].abs().item() < 0.5          # isolated died
    assert out[2, 2].abs().item() >= 0.5         # supported centre survived
    assert out[2, 1].abs().item() < 0.5          # row ends have 1 neighbour -> die


def test_life_step_birth_on_exactly_three():
    w = torch.full((5, 5), 0.01)
    w[2, 1] = w[2, 2] = w[2, 3] = 1.0
    out = life_step(w, alive_thresh=0.5)
    # blinker: cells above/below centre see exactly 3 alive -> born
    assert out[1, 2].abs().item() >= 0.5
    assert out[3, 2].abs().item() >= 0.5


def test_life_step_dead_grid_stays_dead():
    w = torch.full((4, 4), 0.01)
    out = life_step(w, alive_thresh=0.5)
    assert torch.equal(out, w)


# ---------------------------------------------------------------- spectral

def test_spectral_tilt_zero_is_near_identity():
    w = w0(shape=(16, 16))
    out = spectral_tilt(w, 0.0)
    assert torch.allclose(out, w, atol=1e-4)


def test_spectral_tilt_preserves_frobenius_norm():
    w = w0(shape=(16, 16))
    for tilt in (0.8, -0.8):
        out = spectral_tilt(w, tilt)
        assert out.norm().item() == pytest.approx(w.norm().item(), rel=1e-3)
        assert not torch.allclose(out, w, atol=1e-3)


def test_spectral_tilt_positive_boosts_tail():
    w = w0(shape=(16, 16))
    s_in = torch.linalg.svdvals(w)
    s_out = torch.linalg.svdvals(spectral_tilt(w, 0.8))
    # tail singular values gain relative share of the spectrum
    tail_share_in = (s_in[8:].sum() / s_in.sum()).item()
    tail_share_out = (s_out[8:].sum() / s_out.sum()).item()
    assert tail_share_out > tail_share_in


# ---------------------------------------------------------------- conditions

from weight_mutations import apply_condition, Condition


def test_apply_condition_mutates_and_is_seed_reproducible():
    torch.manual_seed(0)
    m1, m2, m3 = FakeDiT(n=4), FakeDiT(n=4), FakeDiT(n=4)
    for m in (m2, m3):
        m.load_state_dict(m1.state_dict())
    cond = Condition(name="t", ops=[{"op": "drift", "amount": 0.05}],
                     target="all", decay_rate=0.5, decay_direction="late",
                     mutation_seed=99)
    r1 = apply_condition(m1, cond)
    r2 = apply_condition(m2, cond)
    r3 = apply_condition(m3, cond._replace(mutation_seed=100))
    sd1, sd2, sd3 = m1.state_dict(), m2.state_dict(), m3.state_dict()
    assert any(not torch.equal(sd1[k], sd3[k]) for k in sd1)   # seed matters
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k])                     # same seed = same model
    assert r1["params_touched"] > 0
    assert r1["params_touched"] == r2["params_touched"]


def test_apply_condition_late_decay_leaves_early_blocks_alone():
    torch.manual_seed(0)
    m = FakeDiT(n=4)
    ref = {k: v.clone() for k, v in m.state_dict().items()}
    cond = Condition(name="t", ops=[{"op": "drift", "amount": 0.05}],
                     target="all", decay_rate=3.0, decay_direction="late",
                     mutation_seed=1)
    apply_condition(m, cond)
    sd = m.state_dict()
    assert all(torch.equal(sd[k], ref[k]) for k in sd if "layers.0." in k)
    assert any(not torch.equal(sd[k], ref[k]) for k in sd if "layers.3." in k)


def test_snapshot_restore_between_conditions():
    torch.manual_seed(0)
    m = FakeDiT(n=2)
    targets = collect_targets(m, target="all")
    snap = snapshot_targets(targets)
    cond = Condition(name="t", ops=[{"op": "contrast", "amount": -0.9}],
                     target="all", decay_rate=0.0, decay_direction="flat",
                     mutation_seed=5)
    apply_condition(m, cond)
    restore_targets(collect_targets(m, target="all"), snap)
    for t in collect_targets(m, target="all"):
        assert torch.equal(t.param, snap[t.name])


# ---------------------------------------------------------------- quantile

from weight_mutations import abs_quantile


def test_abs_quantile_matches_torch_on_small():
    w = w0()
    assert abs_quantile(w, 0.75) == pytest.approx(
        w.abs().float().quantile(0.75).item(), rel=1e-4)


def test_abs_quantile_works_past_torch_quantile_limit():
    w = torch.randn(1 << 24 | 7)  # torch.quantile raises above 2**24 elements
    with pytest.raises(RuntimeError):
        w.quantile(0.75)
    q = abs_quantile(w, 0.75)
    frac_below = (w.abs() <= q).float().mean().item()
    assert frac_below == pytest.approx(0.75, abs=0.01)

# tests/test_tgate.py — the t-gate for the subspace-weighted RF loss (R²(t)-derived weighting)
import json
import torch
import pytest

from stable_audio_3.training.tgate import interp_gate, load_tgate, normalize_gate


def test_interp_linear_and_clamped():
    gt = torch.tensor([0.0, 0.5, 1.0]); gg = torch.tensor([1.0, 3.0, 2.0])
    t = torch.tensor([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    g = interp_gate(t, gt, gg)
    assert torch.allclose(g, torch.tensor([1.0, 1.0, 2.0, 3.0, 2.5, 2.0, 2.0]))


def test_normalize_gate_mean_one_and_floor():
    g = torch.tensor([0.0, 0.0, 1.0, 3.0])
    gn = normalize_gate(g, floor=0.1)
    assert abs(float(gn.mean()) - 1.0) < 1e-6          # uniform-t mean == 1 -> K keeps its meaning
    assert float(gn.min()) > 0.0                        # floor applied before normalization
    gn2 = normalize_gate(torch.tensor([2.0, 2.0]), floor=0.0)
    assert torch.allclose(gn2, torch.ones(2))


def test_load_tgate_modes(tmp_path):
    p = tmp_path / "gate.json"
    p.write_text(json.dumps({"t": [0.1, 0.5, 0.9], "r2_melody": [0.9, 0.5, 0.1],
                             "r2_rest": [0.95, 0.9, 0.5], "deficit": [3.0, 2.0, 1.0]}))
    gt, gg = load_tgate(str(p), mode="r2", floor=0.05)
    assert torch.allclose(gt, torch.tensor([0.1, 0.5, 0.9])) and abs(float(gg.mean()) - 1.0) < 1e-6
    assert gg[0] > gg[-1]                               # r2 mode: weight where melody is recoverable
    _, gd = load_tgate(str(p), mode="deficit", floor=0.05)
    assert gd[0] > gd[-1]
    _, gs = load_tgate(str(p), mode="r2sq", floor=0.05)
    assert gs[0] / gs[-1] > gg[0] / gg[-1]              # sharper
    with pytest.raises(ValueError):
        load_tgate(str(p), mode="nope", floor=0.0)

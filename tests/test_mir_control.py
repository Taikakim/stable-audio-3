"""Tests for scripts/mir_control.py + scripts/build_ctrl_packs.py — the MIR-timeseries
conditioner lane (EXPERIMENTS B7, Kim direct 2026-08-21). TDD: written before the module.

Run: /home/kim/Projects/SAO/.venv/bin/python -m pytest stable-audio-3/tests/test_mir_control.py -x -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from mir_control import (  # noqa: E402
    PACKS, CTRL_FIELDS,
    build_ctrl_array, MirCtrlConditioner, make_ctrl_metadata_wrapper,
    install_mir_control, ControlAblationCallback, write_report, pack_channel_index,
)


# ---------------------------------------------------------------- fixtures
T = 4096


def synth_npz(tmp_path, stem="000001", tval=None):
    """A synthetic .TIMESERIES.npz with every CTRL_FIELDS entry present.
    Scalar fields get a known ramp (0..1); hpcp gets one-hot-ish rows."""
    d = {}
    for f in CTRL_FIELDS:
        if f == "hpcp_ts":
            a = np.zeros((T, 12), dtype=np.float32)
            a[np.arange(T), np.arange(T) % 12] = 1.0
            d[f] = a
        else:
            d[f] = np.linspace(0, 1, T, dtype=np.float32) if tval is None else np.full(T, tval, np.float32)
    p = tmp_path / f"{stem}.TIMESERIES.npz"
    np.savez(p, **d)
    return p


def synth_stats():
    return {f: {"center": 0.0, "scale": 1.0} for f in CTRL_FIELDS}


# ---------------------------------------------------------------- ctrl array
def test_build_ctrl_array_shape_and_order(tmp_path):
    npz = synth_npz(tmp_path)
    a = build_ctrl_array(npz, synth_stats())
    n_ch = sum(12 if f == "hpcp_ts" else 1 for f in CTRL_FIELDS)
    assert a.shape == (n_ch, T)
    assert a.dtype == np.float16


def test_build_ctrl_array_missing_field_zero_fills(tmp_path):
    npz_path = tmp_path / "x.TIMESERIES.npz"
    np.savez(npz_path, hpcp_ts=np.ones((T, 12), np.float32))  # everything else absent
    a = build_ctrl_array(npz_path, synth_stats())
    idx = pack_channel_index("melody")
    assert np.allclose(np.asarray(a, np.float32)[idx], 1.0)
    other = [i for i in range(a.shape[0]) if i not in idx]
    assert np.allclose(np.asarray(a, np.float32)[other], 0.0)


def test_build_ctrl_array_normalizes_and_guards_nan(tmp_path):
    npz = synth_npz(tmp_path, tval=5.0)
    stats = synth_stats()
    for f in stats:
        stats[f] = {"center": 5.0, "scale": 2.0}
    a = np.asarray(build_ctrl_array(npz, stats), np.float32)
    sc = [i for i, (f, _) in enumerate(_iter_field_channels()) if f != "hpcp_ts"]
    assert np.allclose(a[sc], 0.0, atol=1e-3)          # (5-5)/2
    stats["onset_envelope_ts"] = {"center": 0.0, "scale": 0.0}   # degenerate scale
    a2 = build_ctrl_array(npz, stats)
    assert np.isfinite(np.asarray(a2, np.float32)).all()


def _iter_field_channels():
    for f in CTRL_FIELDS:
        for c in range(12 if f == "hpcp_ts" else 1):
            yield f, c


def test_pack_channel_index_covers_all_and_disjoint_core():
    all_idx = pack_channel_index("all")
    assert sorted(all_idx) == list(range(len(list(_iter_field_channels()))))
    core = ["dynamics", "rhythm", "melody", "stems", "spectral"]
    seen = []
    for p in core:
        seen += list(pack_channel_index(p))
    assert len(seen) == len(set(seen)), "core packs must not overlap"
    for p in PACKS:
        assert len(pack_channel_index(p)) > 0


# ---------------------------------------------------------------- metadata wrapper
def test_metadata_wrapper_slices_exact_crop(tmp_path):
    npz = synth_npz(tmp_path)  # ramp 0..1 over 4096
    stem = str(npz).replace(".TIMESERIES.npz", "")
    fn = make_ctrl_metadata_wrapper(None, ctrl_source="timeseries",
                                    pack="rhythm", stats=synth_stats())
    info = {"latent_filename": stem + ".npy",
            "latent_crop_start": 1000, "latent_crop_length": 512}
    out = fn(info, None)
    ctrl = out["mir_ctrl"]
    assert ctrl.shape[1] == 512
    ramp = np.linspace(0, 1, T, dtype=np.float32)[1000:1512]
    assert np.allclose(np.asarray(ctrl[0], np.float32), ramp, atol=2e-3)  # fp16 tol


def test_metadata_wrapper_chains_base_fn(tmp_path):
    npz = synth_npz(tmp_path)
    stem = str(npz).replace(".TIMESERIES.npz", "")
    base = lambda info, _:  {"prompt": "goa"}
    fn = make_ctrl_metadata_wrapper(base, ctrl_source="timeseries",
                                    pack="melody", stats=synth_stats())
    out = fn({"latent_filename": stem + ".npy", "latent_crop_start": 0,
              "latent_crop_length": 256}, None)
    assert out["prompt"] == "goa" and "mir_ctrl" in out


def test_metadata_wrapper_reads_prebuilt_ctrl(tmp_path):
    lat_dir = tmp_path / "latents_x"
    lat_dir.mkdir()
    ctrl_dir = tmp_path / "latents_x_ctrl"       # SIBLING dir: co-located .ctrl.npy would
    ctrl_dir.mkdir()                             # be globbed as a latent by the dataset
    stem = lat_dir / "000009"
    n_ch = len(list(_iter_field_channels()))
    full = np.random.default_rng(0).standard_normal((n_ch, T)).astype(np.float16)
    np.save(str(ctrl_dir / "000009") + ".ctrl.npy", full)
    fn = make_ctrl_metadata_wrapper(None, ctrl_source="ctrl", pack="all", stats=None)
    out = fn({"latent_filename": str(stem) + ".npy", "latent_crop_start": 7,
              "latent_crop_length": 100}, None)
    assert np.array_equal(np.asarray(out["mir_ctrl"]), full[:, 7:107])


# ---------------------------------------------------------------- conditioner
def test_conditioner_stacks_and_masks():
    cond = MirCtrlConditioner(n_channels=3, dropout_prob=0.0)
    items = [np.ones((3, 64), np.float16), np.zeros((3, 64), np.float16)]
    t, mask = cond(items, "cpu")
    assert t.shape == (2, 3, 64) and t.dtype == torch.float32
    assert mask.shape[0] == 2 and bool(mask.all())


def test_conditioner_dropout_zeroes_whole_items():
    torch.manual_seed(0)
    cond = MirCtrlConditioner(n_channels=2, dropout_prob=1.0)
    cond.train()
    t, _ = cond([np.ones((2, 32), np.float16)] * 4, "cpu")
    assert torch.allclose(t, torch.zeros_like(t))
    cond.eval()   # eval: no dropout
    t2, _ = cond([np.ones((2, 32), np.float16)] * 4, "cpu")
    assert torch.allclose(t2, torch.ones_like(t2))


# ---------------------------------------------------------------- installer
class _TinyTransformer(torch.nn.Module):
    """Minimal stand-in exposing the modular_local_embeds contract of
    ContinuousTransformer (dict of per-id projections applied additively)."""
    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim
        self.modular_local_cond_configs = []
        self.modular_local_embeds = torch.nn.ModuleDict()

    def forward(self, x, modular_local_cond=None):
        if modular_local_cond:
            for cid, proj in self.modular_local_embeds.items():
                if cid in modular_local_cond:
                    x = x + proj(modular_local_cond[cid])
        return x


class _FakeWrapper:
    def __init__(self, tfm):
        self.modular_local_cond_ids = []
        self.model = type("M", (), {"model": type("N", (), {"transformer": tfm})()})()


def test_install_zero_init_is_noop_then_trains():
    torch.manual_seed(0)
    tfm = _TinyTransformer(dim=16)
    wrapper = _FakeWrapper(tfm)
    params = install_mir_control(wrapper, tfm, n_channels=5, cond_id="mir_ctrl", dim=16)
    assert "mir_ctrl" in tfm.modular_local_embeds
    assert wrapper.modular_local_cond_ids == ["mir_ctrl"]
    assert all(p.requires_grad for p in params)
    x = torch.randn(2, 40, 16)
    ctrl = {"mir_ctrl": torch.randn(2, 40, 5)}
    assert torch.allclose(tfm(x, ctrl), x), "zero-init must be a no-op"
    with torch.no_grad():   # perturb the zero-init out layer -> control must act
        tfm.modular_local_embeds["mir_ctrl"][-1].weight.add_(0.1)
    assert not torch.allclose(tfm(x, ctrl), x)


# ---------------------------------------------------------------- ablation + report
class _CtrlLossModel(torch.nn.Module):
    """Loss = mse(pred, ctrl-derived target): true ctrl => 0 loss, shuffled/zero => big."""
    def forward(self, latents, ctrl):
        return torch.nn.functional.mse_loss(ctrl.mean(1), latents.mean(1))


def test_ablation_separates_true_from_shuffled_and_zero(tmp_path):
    torch.manual_seed(0)
    lat = torch.randn(4, 2, 32)
    ctrl = lat.clone()                     # perfectly informative control
    cb = ControlAblationCallback(out_dir=str(tmp_path), every_n_steps=1)
    model = _CtrlLossModel()
    def loss_fn(c):
        return float(model(lat, c))
    rec = cb.measure(loss_fn, ctrl)
    assert rec["loss_true"] < rec["loss_shuffled"]
    assert rec["loss_true"] < rec["loss_zero"]
    cb.append_record(rec, step=1)
    assert (tmp_path / "control_ablation.jsonl").exists()


def test_write_report_produces_md_and_json(tmp_path):
    (tmp_path / "control_ablation.jsonl").write_text(
        '{"step": 1, "loss_true": 0.5, "loss_shuffled": 0.9, "loss_zero": 0.8}\n'
        '{"step": 2, "loss_true": 0.4, "loss_shuffled": 0.9, "loss_zero": 0.85}\n')
    csvdir = tmp_path / "lightning_logs" / "version_0"
    csvdir.mkdir(parents=True)
    (csvdir / "metrics.csv").write_text("step,train/loss\n1,1.0\n2,0.8\n")
    out = write_report(str(tmp_path), arm_meta={"pack": "rhythm", "rank": 32})
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["arm"]["pack"] == "rhythm"
    assert rep["ablation"]["final"]["control_gain"] > 0   # (shuffled - true) > 0
    md = (tmp_path / "report.md").read_text()
    assert "rhythm" in md and "control_gain" in md
    assert out.endswith("report.md")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))


# ---------------------------------------------------------------- builder stats
def test_estimate_stats_robust(tmp_path):
    from build_ctrl_packs import estimate_stats
    rng = np.random.default_rng(0)
    paths = []
    for i in range(6):
        d = {f: rng.normal(3.0, 2.0, T).astype(np.float32) for f in CTRL_FIELDS if f != "hpcp_ts"}
        d["hpcp_ts"] = rng.random((T, 12)).astype(np.float32)
        # one gross outlier crop must not wreck robust stats
        if i == 0:
            d["onset_envelope_ts"] = np.full(T, 1e6, np.float32)
        p = tmp_path / f"{i:06d}.TIMESERIES.npz"
        np.savez(p, **d)
        paths.append(p)
    stats = estimate_stats(paths, sample=6)
    st = stats["beat_activation_ts"]
    assert abs(st["center"] - 3.0) < 0.3 and 1.0 < st["scale"] < 4.0
    assert stats["onset_envelope_ts"]["center"] < 1e5   # median beat the outlier
    for f in CTRL_FIELDS:
        assert f in stats and stats[f]["scale"] > 0

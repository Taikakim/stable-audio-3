"""Flight recorder + raw grad telemetry (stable_audio_3/training/flight_recorder.py). CPU only."""
import json
import math
import os

import torch

from stable_audio_3.training.flight_recorder import FlightRecorder, GradTelemetry


def _params():
    torch.manual_seed(0)
    return [
        ("blk.0.lora_A", torch.nn.Parameter(torch.randn(4, 8))),
        ("blk.0.lora_B", torch.nn.Parameter(torch.randn(6, 4))),
        ("blk.0.magnitude", torch.nn.Parameter(torch.ones(6))),
        ("frozen", torch.nn.Parameter(torch.ones(3), requires_grad=False)),
    ]


def _set_grads(named, scale=1.0, nan_on=None):
    for n, p in named:
        if p.requires_grad:
            p.grad = torch.full_like(p, scale)
            if n == nan_on:
                p.grad[0] = float("nan")


def test_kind_norms_and_total():
    named = _params()
    _set_grads(named, 2.0)
    m, per = GradTelemetry(named).measure()
    assert math.isclose(m["grad/raw_norm_lora_A"], 2.0 * math.sqrt(32), rel_tol=1e-6)
    assert math.isclose(m["grad/raw_norm_lora_B"], 2.0 * math.sqrt(24), rel_tol=1e-6)
    assert math.isclose(m["grad/raw_norm_magnitude"], 2.0 * math.sqrt(6), rel_tol=1e-6)
    assert "grad/raw_norm_other" not in m  # frozen param excluded, no trainable "other"
    assert math.isclose(m["grad/raw_norm_total"], 2.0 * math.sqrt(62), rel_tol=1e-6)
    assert m["grad/nonfinite_tensors"] == 0
    assert len(per) == 3


def test_missing_grad_is_skipped_not_counted_bad():
    named = _params()
    _set_grads(named, 1.0)
    named[2][1].grad = None
    m, per = GradTelemetry(named).measure()
    assert per[2] is None and m["grad/nonfinite_tensors"] == 0
    assert math.isclose(m["grad/raw_norm_total"], math.sqrt(56), rel_tol=1e-6)


def test_spike_triggers_one_dump_with_batch(tmp_path):
    named = _params()
    fr = FlightRecorder(str(tmp_path), named, window=50, warmup=20, z_thresh=6.0, min_gap=1)
    g = torch.Generator().manual_seed(1)
    for step in range(40):  # calm: ~5% jitter
        _set_grads(named, 1.0 + 0.05 * float(torch.randn(1, generator=g)))
        m = fr.observe(step)
        assert m["grad/flight_trigger"] == 0.0, (step, m.get("grad/raw_norm_z"))
    fr.stash = {"t": torch.tensor([0.25, 0.75]), "prompts": ["a", "b"], "files": ["x", "y"],
                "latents": torch.zeros(2, 3, 5), "loss": torch.tensor(0.5)}
    _set_grads(named, 20.0)  # spike
    m = fr.observe(40)
    assert m["grad/flight_trigger"] == 1.0 and m["grad/raw_norm_z"] > 6.0
    d = tmp_path / "incidents" / "step_0000040"
    rep = json.loads((d / "incident.json").read_text())
    assert rep["trigger"] == "raw_norm_z"
    assert rep["batch"]["prompts"] == ["a", "b"] and rep["batch"]["t"] == [0.25, 0.75]
    assert rep["batch"]["loss"] == 0.5
    assert len(rep["previous_steps"]) == 8 and rep["previous_steps"][-1]["step"] == 39
    assert abs(sum(t["share_of_sq"] for t in rep["top_params"]) - 1.0) < 1e-6
    blob = torch.load(d / "batch.pt")
    assert blob["latents"].shape == (2, 3, 5)
    # the spike must not enter its own yardstick
    assert max(fr.hist) < math.log(20.0)


def test_nonfinite_grad_triggers_even_in_warmup(tmp_path):
    named = _params()
    fr = FlightRecorder(str(tmp_path), named, warmup=50)
    _set_grads(named, 1.0, nan_on="blk.0.lora_B")
    m = fr.observe(0)
    assert m["grad/nonfinite_tensors"] == 1 and m["grad/flight_trigger"] == 1.0
    rep = json.loads((tmp_path / "incidents" / "step_0000000" / "incident.json").read_text())
    assert rep["trigger"] == "nonfinite_grad" and rep["nonfinite_params"] == ["blk.0.lora_B"]


def test_max_dumps_and_min_gap(tmp_path):
    named = _params()
    fr = FlightRecorder(str(tmp_path), named, max_dumps=2, min_gap=5, save_batch=False)
    for step in range(10):
        _set_grads(named, 1.0, nan_on="blk.0.lora_A")
        fr.observe(step)
    assert fr.n_triggers == 10 and fr.n_dumps == 2
    assert sorted(os.listdir(tmp_path / "incidents")) == ["step_0000000", "step_0000005"]


def test_lightning_hook_wiring(tmp_path):
    """Run the REAL DiffusionCondTrainingWrapper.on_before_optimizer_step on a stand-in
    module: builds the recorder lazily from self.diffusion, feeds it the training_step
    stash, logs grad/* through the logger, and dumps on a non-finite grad."""
    from types import SimpleNamespace
    from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper

    diffusion = torch.nn.Module()
    diffusion.blk = torch.nn.Module()
    diffusion.blk.lora_A = torch.nn.Parameter(torch.randn(4, 8))
    diffusion.blk.lora_B = torch.nn.Parameter(torch.randn(6, 4))
    logged = {}

    class _Logger:
        def log_metrics(self, d, step=None):
            logged.update(d)

    fake = SimpleNamespace(
        flight_recorder_cfg=dict(run_dir=str(tmp_path), warmup=50),
        diffusion=diffusion, global_step=0, logger=_Logger(),
        _staggered_logger=SimpleNamespace(every_n_steps=1),
        _flight_stash={"t": torch.tensor([0.5]), "prompts": ["p"], "files": ["f"],
                       "loss": torch.tensor(1.0), "per_item_loss": torch.tensor([1.0])},
        _fusion_opt=lambda: None,
    )
    for p in diffusion.parameters():
        p.grad = torch.ones_like(p)
    diffusion.blk.lora_B.grad[0, 0] = float("inf")
    DiffusionCondTrainingWrapper.on_before_optimizer_step(fake, torch.optim.SGD(diffusion.parameters(), lr=0.1))
    assert fake._flight_recorder.n_dumps == 1
    assert logged["grad/nonfinite_tensors"] == 1.0 and logged["grad/flight_trigger"] == 1.0
    rep = json.loads((tmp_path / "incidents" / "step_0000000" / "incident.json").read_text())
    assert rep["nonfinite_params"] == ["blk.lora_B"] and rep["batch"]["per_item_loss"] == [1.0]

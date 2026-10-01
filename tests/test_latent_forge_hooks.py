"""Latent Forge hooks in the fork (avp-audio-craft spec 2026-09-15 §6.6)."""
import types

import torch

import stable_audio_3.model as m


def _stub():
    return types.SimpleNamespace(
        device="cpu",
        model=types.SimpleNamespace(pretransform=types.SimpleNamespace(downsampling_ratio=4096),
                                    sample_rate=44100, model=object(), diffusion_objective="rectified_flow"))


def _run(monkeypatch, **extra):
    seen = {}

    def fake_sampler(model, x, sigmas, guides, **kw):
        seen.update(kw)
        return x

    monkeypatch.setattr(m, "sample_flow_euler_multi_latch_guided", fake_sampler)
    m.StableAudioModel._latch_guided_generate(
        _stub(), noise=torch.zeros(1, 4, 8), cond_inputs={}, latch_configs=[], latch_hparams={},
        steps=4, cfg_scale=1.0, apg_scale=1.0, batch_size=1, latent_sample_size=8, dist_shift=None,
        return_latents=True, **extra)
    return seen


def test_latch_path_forwards_scale_phi(monkeypatch):
    assert _run(monkeypatch, scale_phi=0.7)["scale_phi"] == 0.7


def test_latch_path_omits_zero_scale_phi(monkeypatch):
    assert "scale_phi" not in _run(monkeypatch)

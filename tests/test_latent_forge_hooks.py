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


import pytest


def test_fit_latents_pads_crops_and_casts():
    z = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    out = m._fit_latents(z, 8, "cpu", torch.float16)
    assert out.shape == (1, 2, 8) and out.dtype == torch.float16
    assert float(out[0, 1, 5]) == 11.0 and float(out[0, 0, 7]) == 0.0
    assert m._fit_latents(z.unsqueeze(0), 4, "cpu", torch.float32).shape == (1, 2, 4)
    with pytest.raises(ValueError):
        m._fit_latents(torch.zeros(2, 2, 2, 2), 4, "cpu", torch.float32)


def test_generate_rejects_both_init_forms():
    with pytest.raises(ValueError, match="init_audio or init_latents"):
        m.StableAudioModel.generate(_stub(), prompt="x", init_audio=(44100, torch.zeros(2, 10)),
                                    init_latents=torch.zeros(1, 4, 2))


def test_generate_rejects_both_inpaint_forms():
    with pytest.raises(ValueError, match="inpaint_audio or inpaint_latents"):
        m.StableAudioModel.generate(_stub(), prompt="x", inpaint_audio=(44100, torch.zeros(2, 10)),
                                    inpaint_latents=torch.zeros(1, 4, 2))

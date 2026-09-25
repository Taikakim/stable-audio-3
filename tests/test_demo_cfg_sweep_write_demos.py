"""demo_cfg_sweep --write-demos: cfg-7 clips land in <run>/demos/step<N>/ under the demo callback's
names, cfg-1 clips do not (CPU only: model, sampler and decoder replaced by stand-ins)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import demo_cfg_sweep as S  # noqa: E402


def test_write_demos_copies_cfg7_with_callback_names(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "step=100.ckpt").write_text("x")

    class _PT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def decode(self, z):
            return torch.zeros(1, 2, 64)

    fake = SimpleNamespace(pretransform=_PT(), sample_rate=44100)
    monkeypatch.setattr(S, "adapter_state", lambda ckpt, w: ({}, {}, "x (test)"))
    monkeypatch.setattr(S, "load_model", lambda sd, cfg, tmp, base, merge=True: fake)
    monkeypatch.setattr(S, "generate_clip_latents",
                        lambda m, text, seed, total_frames, steps, cfg_scale, max_latent_std: torch.randn(1, 4, total_frames))
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self, raising=False)
    monkeypatch.setattr(sys, "argv", ["demo_cfg_sweep.py", "--run-dir", str(run), "--steps-ckpt", "100",
                                      "--cfgs", "1", "7", "--num-prompts", "1", "--write-demos"])
    S.main()

    names = sorted(p.name for p in (run / "demos" / "step100").iterdir())
    pid = S.CANONICAL_DEMO_PROMPTS[0]["id"]
    assert names == sorted([f"step100_{pid}_20s.wav", f"step100_{pid}_20s.z0.npy",
                            f"step100_{pid}_48s.wav", f"step100_{pid}_48s.z0.npy"])

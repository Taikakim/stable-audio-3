"""--no-inline-demos: milestones still save a checkpoint, but nothing renders (training-findings 13e)."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_demo_callback import ModularDemoAndLossGuardCallback  # noqa: E402


def _run(tmp_path, render_inline):
    cb = ModularDemoAndLossGuardCallback(save_dir=str(tmp_path), step_milestones=(5,), render_inline=render_inline)
    saved, rendered = [], []
    cb._render_demos = lambda *a, **k: rendered.append(k.get("tag"))
    trainer = SimpleNamespace(global_step=5, accumulate_grad_batches=1,
                              save_checkpoint=lambda p: (saved.append(p), Path(p).write_text("x")))
    cb.on_train_batch_end(trainer, None, None, None, 0)
    return saved, rendered


def test_no_inline_saves_but_does_not_render(tmp_path):
    saved, rendered = _run(tmp_path, render_inline=False)
    assert [Path(p).name for p in saved] == ["step=5.ckpt"] and rendered == []


def test_inline_default_still_renders(tmp_path):
    saved, rendered = _run(tmp_path, render_inline=True)
    assert len(saved) == 1 and rendered == ["step5"]

"""demo_cfg_sweep must render a warm-started adapter on the run's --init_state_ckpt base, and refuse
when that base is named but missing (e.g. drive unmounted) instead of falling back to medium-base."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from demo_cfg_sweep import warm_start_base  # noqa: E402


def _meta(run, args):
    (run / "run_meta.json").write_text(json.dumps({"args": args}))


def test_no_run_meta_means_stock_base(tmp_path):
    assert warm_start_base(tmp_path) is None


def test_run_without_init_ckpt_is_stock_base(tmp_path):
    _meta(tmp_path, {"lr": 3e-3})
    assert warm_start_base(tmp_path) is None


def test_existing_init_ckpt_is_returned(tmp_path):
    base = tmp_path / "epoch=7.weights.ckpt"
    base.write_bytes(b"x")
    _meta(tmp_path, {"init_state_ckpt": str(base)})
    assert warm_start_base(tmp_path) == base


def test_missing_init_ckpt_refuses(tmp_path):
    _meta(tmp_path, {"init_state_ckpt": str(tmp_path / "gone.ckpt")})
    with pytest.raises(SystemExit, match="Refusing"):
        warm_start_base(tmp_path)

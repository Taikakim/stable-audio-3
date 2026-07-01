"""ROCm/AMD environment loader for Stable Audio 3.

All values live in ``rocm_env.yaml`` (repo root) — this module just reads it and
applies a named profile to ``os.environ``. Nothing is hardcoded here.

Call sites, both BEFORE ``import torch``:
  - ``stable_audio_3/__init__.py``      -> ``apply_profile("inference")``
  - ``scripts/latch/train_latch.py``    -> ``apply_profile("training")``
    (loads this file standalone via importlib so the package ``__init__`` — and
    therefore torch — does not run before the training profile is set).

This module imports nothing heavier than PyYAML, so it is safe to run before
torch. Values are set with ``setdefault``: anything already exported in the
shell wins. Override the YAML location with ``SA3_ROCM_ENV_YAML``.
"""

import os
import sys
import warnings
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# Env vars whose value is a directory that must exist for the cache to work.
_DIR_KEYS = ("TRITON_CACHE_DIR", "MIOPEN_CUSTOM_CACHE_DIR", "MIOPEN_USER_DB_PATH")


def _yaml_path() -> Path:
    override = os.environ.get("SA3_ROCM_ENV_YAML")
    if override:
        return Path(override)
    repo_root = Path(__file__).resolve().parent.parent / "rocm_env.yaml"
    if repo_root.exists():
        return repo_root
    return Path.cwd() / "rocm_env.yaml"


def _resolve(value, tunings_root: str) -> str:
    return str(value).replace("${tunings_root}", tunings_root)


def _ensure_dirs():
    for key in _DIR_KEYS:
        path = os.environ.get(key)
        if path:
            try:
                os.makedirs(path, exist_ok=True)
            except OSError:
                pass
    tunable_file = os.environ.get("PYTORCH_TUNABLEOP_FILENAME")
    if tunable_file:
        parent = os.path.dirname(tunable_file)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                pass


# Marker in os.environ (shared across module instances — run_gradio loads this
# file standalone via importlib while the package imports it normally, so a
# module-level flag would not be seen by both). Lets the redundant second call
# (package __init__, after run_gradio/train_latch already applied) no-op silently.
_APPLIED_KEY = "SA3_ROCM_ENV_APPLIED"


def apply_profile(profile: str = "inference", verbose: bool = False) -> None:
    """Apply a profile from rocm_env.yaml to os.environ (idempotent, setdefault).

    Idempotent across the process: once any profile has been applied, later calls
    return immediately. This is why an early standalone apply (before torch) wins
    over the package-import apply, and why the latter does not re-warn.
    """
    if os.environ.get(_APPLIED_KEY):
        if verbose:
            print(f"[rocm_env] profile already applied ({os.environ[_APPLIED_KEY]}); skipping.")
        return
    if yaml is None:
        warnings.warn("PyYAML not installed; ROCm env profile not applied.", stacklevel=2)
        return
    path = _yaml_path()
    if not path.exists():
        warnings.warn(f"rocm_env.yaml not found at {path}; ROCm env not applied.", stacklevel=2)
        return
    if "torch" in sys.modules:
        warnings.warn(
            f"rocm_env.apply_profile('{profile}') ran after torch import; allocator, "
            "TunableOp, and MIOpen settings may be ignored.", stacklevel=2)

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    tunings_root = str(cfg.get("tunings_root", ""))
    merged = dict(cfg.get("common", {}))
    merged.update(cfg.get("profiles", {}).get(profile, {}))

    for key, value in merged.items():
        os.environ.setdefault(key, _resolve(value, tunings_root))

    _ensure_dirs()
    os.environ[_APPLIED_KEY] = profile

    if verbose:
        for key in merged:
            print(f"[rocm_env:{profile}] {key}={os.environ.get(key)}")

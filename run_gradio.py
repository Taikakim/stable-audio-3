import sys as _sys

# Honor --no-flash-attn before any import: the attention backend is chosen at
# transformer import time, so the env var must be set first. Useful when the
# local flash-attn build is broken/ABI-mismatched (falls back to SDPA).
if "--no-flash-attn" in _sys.argv:
    import os as _os0
    _os0.environ["SA3_DISABLE_FLASH_ATTN"] = "1"
# Disable only the varlen path (keep standard flash_attn_func). For ROCm where
# aiter's Triton varlen_fwd is version-skewed from the flash_attn wrapper.
if "--no-flash-varlen" in _sys.argv:
    import os as _os1
    _os1.environ["SA3_DISABLE_FLASH_VARLEN"] = "1"


# Apply the ROCm/AMD env profile from rocm_env.yaml BEFORE importing torch.
# Loaded standalone so this runs ahead of the line-1 torch import below.
def _apply_rocm_inference_profile():
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _re = _Path(__file__).resolve().parent / "stable_audio_3" / "rocm_env.py"
    if not _re.exists():
        return
    _spec = _ilu.spec_from_file_location("_sa3_rocm_env", _re)
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    _m.apply_profile("inference")


_apply_rocm_inference_profile()

import torch  # noqa: E402
from stable_audio_3.interface.diffusion_cond import create_diffusion_cond_ui  # noqa: E402
from stable_audio_3 import StableAudioModel  # noqa: E402
from stable_audio_3.verbose import set_verbose  # noqa: E402
import sys  # noqa: E402

# Silence Python warnings (FutureWarning, DeprecationWarning, etc.) unless --verbose.
# Must run before any ML library imports since most warnings fire at import time.
# We keep Gradio chatter, HF/torch progress bars, and generation tqdm intact.
if "--verbose" not in sys.argv:
    import os as _os

    _os.environ.setdefault("PYTHONWARNINGS", "ignore")
    import warnings as _warnings

    _warnings.filterwarnings("ignore")


def main(args):
    set_verbose(getattr(args, "verbose", False))
    torch.manual_seed(42)
    model_half = args.model_half
    model = StableAudioModel.from_pretrained(args.model, model_half=model_half)
    if args.lora_ckpt_path:
        model.load_lora(args.lora_ckpt_path)
    interface = create_diffusion_cond_ui(
        model,
        gradio_title=args.title if args.title is not None else "Stable Audio 3",
        default_prompt=args.default_prompt,
    )
    interface.queue()
    # Local-only by default (no public *.gradio.live tunnel). Pass --share to
    # explicitly open a public link.
    interface.launch(
        share=args.share,
        server_name=args.server_name,
        js=getattr(interface, "_sao_js", None),
        theme=getattr(interface, "_sao_theme", None),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run gradio interface")
    parser.add_argument(
        "--model", type=str, help="Name of pretrained model", required=True
    )
    parser.add_argument(
        "--model-config", type=str, help="Path to model config", required=False
    )
    parser.add_argument(
        "--ckpt-path", type=str, help="Path to model checkpoint", required=False
    )
    parser.add_argument(
        "--pretransform-ckpt-path",
        type=str,
        help="Optional to model pretransform checkpoint",
        required=False,
    )
    parser.add_argument("--username", type=str, help="Gradio username", required=False)
    parser.add_argument("--password", type=str, help="Gradio password", required=False)
    parser.add_argument(
        "--model-half",
        action="store_true",
        help="Whether to use half precision",
        required=False,
        default=True,
    )
    parser.add_argument(
        "--title", type=str, help="Display Title top of Gradio", required=False
    )
    parser.add_argument(
        "--lora-ckpt-path",
        type=str,
        nargs="*",
        help="Path(s) for LoRA(s) to apply. Can specify multiple.",
        required=False,
    )
    parser.add_argument(
        "--default-prompt",
        type=str,
        default=None,
        help="Default prompt to pre-fill in the textbox",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed load/generation progress",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Open a public *.gradio.live link (off by default; local-only).",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Bind address. Default 127.0.0.1 (localhost only); use 0.0.0.0 for LAN.",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        default=False,
        help="Force the SDPA attention fallback (handled before import; for broken flash-attn builds).",
    )
    parser.add_argument(
        "--no-flash-varlen",
        action="store_true",
        default=False,
        help="Disable only flash-attn's varlen path, keep standard flash attention "
             "(ROCm: aiter varlen_fwd skewed from the flash_attn wrapper).",
    )
    args = parser.parse_args()
    main(args)

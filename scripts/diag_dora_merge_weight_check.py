#!/usr/bin/env python3
"""Direct weight-space check: does merge_adapters() bake in the CURRENT set_lora_strength(w),
or does it silently merge as if w=1 regardless (WINTERMUTE's review point 1, 13e)?

No sampling involved (eliminates the diffusion trajectory's compounding-noise confound seen in
diag_dora_strength_equivalence.py's render-based comparison). For one parametrized module: read
its LIVE (parametrization-computed) weight at strength w, then merge and read the BAKED weight,
and diff them directly. CPU-only -- no GPU lock needed.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
import torch
import torch.nn.utils.parametrize as P

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_cfg_sweep import merge_adapters  # noqa: E402

ADAPTER = "/run/media/kim/Mantu/sa3_lora_runs/goa3_avp_r128_shampoo_b16_3e4_2026-09-25/slim_adapter_x_step3804.pt"


def check(strength: float):
    from stable_audio_3 import StableAudioModel
    m = StableAudioModel.from_pretrained("medium-base", device="cpu", model_half=False)
    m.load_lora([ADAPTER])
    m.set_lora_strength(strength)

    parametrized = [mod for mod in m.model.model.modules() if P.is_parametrized(mod)]
    print(f"[check] w={strength}: {len(parametrized)} parametrized modules")
    mod0 = parametrized[0]
    name0 = list(mod0.parametrizations.keys())[0]
    with torch.no_grad():
        live_w = getattr(mod0, name0).clone()

    n = merge_adapters(m.model)
    baked_w = getattr(mod0, name0).clone()

    diff = (live_w - baked_w).abs()
    print(f"[check] w={strength}: merged {n} params. live vs baked on module[0].{name0}: "
          f"max_abs_diff={float(diff.max()):.8f} mean_abs_diff={float(diff.mean()):.8f} "
          f"live_norm={float(live_w.norm()):.6f} baked_norm={float(baked_w.norm()):.6f}")
    return live_w, baked_w


def main():
    print("=== strength 1.0 ===")
    live1, baked1 = check(1.0)
    print("=== strength 0.5 ===")
    live05, baked05 = check(0.5)
    print("=== strength 0.5 vs strength 1.0 (must differ -- strength must reach the weight at all) ===")
    d = (baked05 - baked1).abs()
    print(f"[check] baked(w=0.5) vs baked(w=1.0): max_abs_diff={float(d.max()):.6f} mean_abs_diff={float(d.mean()):.6f}")


if __name__ == "__main__":
    main()

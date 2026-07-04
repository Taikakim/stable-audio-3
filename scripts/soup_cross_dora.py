#!/usr/bin/env python3
"""Build cross-optimiser DoRA soups: w_A*(AdamW ckpt) + w_F*(Fusion ckpt).

Reuses soup_dora.py's averaging (`soup`, fp64 accumulate) and the canonical
`load_lora_checkpoint`. Both sources are rank-128 -> identical LoRA shapes ->
averageable. Saved as {state_dict, lora_config (copied from AdamW), epoch:-1,
global_step:-1}. CPU-only, no GPU, no rendering.
"""
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))
sys.path.insert(0, str(SCRIPTS))
from stable_audio_3.models.lora.utils import load_lora_checkpoint  # noqa: E402
from soup_dora import soup  # noqa: E402

RUNS = Path("/run/media/kim/Mantu/sa3_lora_runs")
OUT = RUNS / "soups_dora"
OUT.mkdir(parents=True, exist_ok=True)

ADAMW = RUNS / "sa3-goa-dora-47s-r128-adamw/dq0egegi/checkpoints/epoch=0-step=1350.ckpt"
FUSION = RUNS / "sa3-goa-dora-47s-r128-fusion/mqe3ne49/checkpoints/epoch=3-step=5400.ckpt"

# (AdamW weight, Fusion weight)
RATIOS = {
    "cross_10A90F": (0.10, 0.90),
    "cross_15A85F": (0.15, 0.85),
    "cross_20A80F": (0.20, 0.80),
    "cross_25A75F": (0.25, 0.75),
    "cross_30A70F": (0.30, 0.70),
}


def main():
    assert ADAMW.exists(), f"missing AdamW: {ADAMW}"
    assert FUSION.exists(), f"missing Fusion: {FUSION}"

    sd_a, cfg_a = load_lora_checkpoint(str(ADAMW))
    sd_f, cfg_f = load_lora_checkpoint(str(FUSION))

    shapes_a = {k: tuple(v.shape) for k, v in sd_a.items()}
    shapes_f = {k: tuple(v.shape) for k, v in sd_f.items()}
    assert shapes_a == shapes_f, "shape mismatch between AdamW and Fusion ckpts"
    print(f"loaded AdamW ({len(sd_a)} tensors, rank={cfg_a.get('rank')}) + "
          f"Fusion ({len(sd_f)} tensors, rank={cfg_f.get('rank')}); shapes match")

    # idx 0 = AdamW, idx 1 = Fusion
    state_dicts = [(0, sd_a), (1, sd_f)]
    for name, (wa, wf) in RATIOS.items():
        souped = soup(state_dicts, {0: float(wa), 1: float(wf)})
        out_path = OUT / f"{name}.ckpt"
        torch.save(
            {"state_dict": souped, "lora_config": dict(cfg_a),
             "epoch": -1, "global_step": -1},
            str(out_path),
        )
        print(f"  wrote {out_path}  [AdamW={wa:.2f}, Fusion={wf:.2f}]")


if __name__ == "__main__":
    main()

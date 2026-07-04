#!/usr/bin/env python3
"""Build DoRA model soups by CPU weight-averaging Lightning .ckpt LoRA checkpoints.

Each source .ckpt stores LoRA tensors in ck['state_dict'] (lora_A/lora_B/magnitude,
687 tensors for these runs) plus a 'lora_config' dict. A soup is a weighted average
of N same-rank source state_dicts:

    souped[k] = sum_i w_i * source_i[k]     (accumulated in fp64, cast back to source dtype)

and is written as {'state_dict': souped, 'lora_config': <copied from a source>,
'epoch': -1, 'global_step': -1}. It loads via
stable_audio_3.models.lora.utils.load_lora_checkpoint.

CPU-only. No GPU, no rendering.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stable_audio_3.models.lora.utils import load_lora_checkpoint  # noqa: E402

RUNS = Path("/run/media/kim/Mantu/sa3_lora_runs")
OUT = RUNS / "soups_dora"
OUT.mkdir(parents=True, exist_ok=True)

# Epoch-ordered (ep0..ep7) source checkpoint paths per variant.
VARIANTS = {
    "r16": [
        RUNS / "sa3-goa-dora-47s-b4/vjnnndnu/checkpoints/epoch=0-step=1350.ckpt",
        RUNS / "sa3-goa-dora-47s-b4/vjnnndnu/checkpoints/epoch=1-step=2700.ckpt",
        RUNS / "sa3-goa-dora-47s-b4/vjnnndnu/checkpoints/epoch=2-step=4050.ckpt",
        RUNS / "sa3-goa-dora-47s-b4-cont/x20b3ygb/checkpoints/epoch=0-step=1350.ckpt",
        RUNS / "sa3-goa-dora-47s-b4-cont/x20b3ygb/checkpoints/epoch=1-step=2700.ckpt",
        RUNS / "sa3-goa-dora-47s-b4-cont/x20b3ygb/checkpoints/epoch=2-step=4050.ckpt",
        RUNS / "sa3-goa-dora-47s-b4-cont/x20b3ygb/checkpoints/epoch=3-step=5400.ckpt",
        RUNS / "sa3-goa-dora-47s-b4-cont/x20b3ygb/checkpoints/epoch=4-step=6750.ckpt",
    ],
    "r64": [
        RUNS / f"sa3-goa-dora-47s-r64/i8nygj4y/checkpoints/epoch={i}-step={(i+1)*1350}.ckpt"
        for i in range(8)
    ],
    "r128f": [
        RUNS / f"sa3-goa-dora-47s-r128-fusion/mqe3ne49/checkpoints/epoch={i}-step={(i+1)*1350}.ckpt"
        for i in range(8)
    ],
}

# eval-best 3 epochs per variant for the "goodearly" recipe (indices into ep0..ep7)
GOODEARLY_IDX = {"r16": [0, 1, 2], "r64": [0, 1, 2], "r128f": [0, 1, 3]}


def recipe_weights(variant):
    """Return {recipe_name: {epoch_idx: weight}} for one variant (8 epochs)."""
    i = np.arange(8, dtype=np.float64)
    expasc = np.exp(4.0 * i / 7.0)
    expasc /= expasc.sum()
    expdesc = np.exp(4.0 * (7 - i) / 7.0)
    expdesc /= expdesc.sum()
    ge_idx = GOODEARLY_IDX[variant]
    goodearly = np.zeros(8, dtype=np.float64)
    goodearly[ge_idx] = 1.0 / len(ge_idx)
    return {
        "expasc": {k: float(expasc[k]) for k in range(8)},
        "expdesc": {k: float(expdesc[k]) for k in range(8)},
        "goodearly": {k: float(goodearly[k]) for k in range(8) if goodearly[k] != 0.0},
    }


def soup(state_dicts, weights):
    """Weighted average. state_dicts: list of (idx, sd). weights: {idx: w}.

    fp64 accumulate, cast back to each tensor's source dtype.
    """
    keys = list(state_dicts[0][1].keys())
    ref = state_dicts[0][1]
    out = {}
    for k in keys:
        acc = None
        for idx, sd in state_dicts:
            w = weights.get(idx, 0.0)
            if w == 0.0:
                continue
            t = sd[k].to(torch.float64)
            acc = t * w if acc is None else acc + t * w
        out[k] = acc.to(ref[k].dtype)
    return out


def main():
    summary = {}
    for variant, paths in VARIANTS.items():
        # Load all 8 source ckpts once; verify shapes match across epochs.
        loaded = []
        ref_cfg = None
        ref_shapes = None
        for idx, p in enumerate(paths):
            assert p.exists(), f"missing source: {p}"
            sd, cfg = load_lora_checkpoint(str(p))
            shapes = {k: tuple(v.shape) for k, v in sd.items()}
            if ref_shapes is None:
                ref_shapes, ref_cfg = shapes, cfg
            else:
                assert shapes == ref_shapes, f"shape mismatch in {variant} at {p.name}"
            loaded.append((idx, sd))
        print(f"[{variant}] loaded {len(loaded)} ckpts, {len(ref_shapes)} tensors, "
              f"rank={ref_cfg.get('rank')}, adapter={ref_cfg.get('adapter_type')}")

        for recipe, weights in recipe_weights(variant).items():
            souped = soup(loaded, weights)
            out_path = OUT / f"{variant}_{recipe}.ckpt"
            torch.save(
                {"state_dict": souped, "lora_config": dict(ref_cfg),
                 "epoch": -1, "global_step": -1},
                str(out_path),
            )
            summary[out_path.name] = weights
            wstr = ", ".join(f"ep{k}={v:.4f}" for k, v in sorted(weights.items()))
            print(f"  wrote {out_path}  [{wstr}]")
    print("\n=== weights summary ===")
    for name, w in summary.items():
        print(name, {f"ep{k}": round(v, 6) for k, v in sorted(w.items())})


if __name__ == "__main__":
    main()

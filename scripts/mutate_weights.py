"""Weight garden — artistic weight mutation for the SA3 base model.

Mutates the DiT block weights (drift / shuffle / blur / contrast / spectral
tilt / Game-of-Life) with a per-block decay profile, renders audio per
condition, and writes a fully reproducible recipe next to every file.

Everything random is driven by --mutation-seed: a recipe (this file's
manifest entry) recreates the exact model state — no 4.6 GB checkpoints.

Modes:
  --tour                 curated audition set across the whole palette
  --conditions FILE      JSON list of condition dicts (same keys as Condition)
  (flags)                single condition from --drift/--shuffle/... flags

Examples:
  python scripts/mutate_weights.py --tour --out-dir /path/to/renders
  python scripts/mutate_weights.py --out-dir out --drift 0.03 \
      --target attn --decay-direction early --mutation-seed 7

Ops live in scripts/weight_mutations.py (tested: tests/test_weight_mutations.py).
"""

import os
# ROCm/RDNA4: must be set before torch is imported (MASTER.md §5)
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")

import argparse
import json
import re
import sys
import time

SA3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAO_ROOT = os.path.dirname(SA3_ROOT)
sys.path.append(SA3_ROOT)
sys.path.append(os.path.join(SA3_ROOT, "scripts"))
sys.path.append(os.path.join(SAO_ROOT, "control"))

import torch
from stable_audio_3 import StableAudioModel
from sa3_control.audio_io import save_audio
from weight_mutations import (
    Condition, apply_condition, collect_targets,
    snapshot_targets, restore_targets,
)

DEFAULT_PROMPT = "aggressive upbeat goa trance"


def tour_conditions(seed):
    """A curated walk across the palette. One knob per stop, plus combos and
    a Game-of-Life generation series ('life' entries render once per step)."""
    C = lambda name, ops, **kw: Condition(name=name, ops=ops, mutation_seed=seed, **kw)
    return [
        C("drift_late_lo",  [{"op": "drift", "amount": 0.01}]),
        C("drift_late_hi",  [{"op": "drift", "amount": 0.05}]),
        C("drift_early",    [{"op": "drift", "amount": 0.03}], decay_direction="early"),
        C("drift_attn",     [{"op": "drift", "amount": 0.05}], target="attn", decay_rate=0.3),
        C("drift_mlp",      [{"op": "drift", "amount": 0.05}], target="mlp", decay_rate=0.3),
        C("fader_drift",    [{"op": "drift", "amount": 0.15}], target="norm", decay_direction="flat"),
        C("shuffle",        [{"op": "shuffle", "amount": 0.02}]),
        C("blur_attn",      [{"op": "blur", "amount": 0.5}], target="attn", decay_rate=0.3),
        C("contrast_hi",    [{"op": "contrast", "amount": 0.5}]),
        C("contrast_wash",  [{"op": "contrast", "amount": -0.6}]),
        C("tilt_tail",      [{"op": "tilt", "amount": 0.8}], decay_rate=0.3),
        C("tilt_head",      [{"op": "tilt", "amount": -0.8}], decay_rate=0.3),
        C("combo_rot",      [{"op": "drift", "amount": 0.02},
                             {"op": "contrast", "amount": -0.3},
                             {"op": "tilt", "amount": 0.4}]),
        C("life",           [{"op": "life", "quantile": 0.75}], decay_rate=0.3),
    ]


def slug(text, n=32):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n]


def render(sam, sr, path, prompt, seed, args):
    with torch.inference_mode():
        audio = sam.generate(prompt=prompt, duration=args.duration, steps=args.steps,
                             cfg_scale=args.cfg, seed=seed, sampler_type="euler")
    save_audio(path, audio[0], sr)
    print(f"[render] {os.path.basename(path)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--tour", action="store_true", help="render the curated palette tour")
    ap.add_argument("--conditions", help="JSON file: list of condition dicts")
    # single-condition flags
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--shuffle", type=float, default=0.0)
    ap.add_argument("--blur", type=float, default=0.0)
    ap.add_argument("--contrast", type=float, default=0.0, help="negative flattens toward the mean")
    ap.add_argument("--tilt", type=float, default=0.0, help="spectral tilt: >0 tail, <0 head")
    ap.add_argument("--life", action="store_true", help="apply Game-of-Life steps (see --life-generations)")
    ap.add_argument("--life-quantile", type=float, default=0.75, help="|w| quantile that counts as alive")
    ap.add_argument("--target", default="all", choices=["all", "attn", "mlp", "norm"])
    ap.add_argument("--decay-rate", type=float, default=0.5)
    ap.add_argument("--decay-direction", default="late", choices=["late", "early", "flat", "focus"])
    ap.add_argument("--decay-focus", type=int, default=None, help="block index for --decay-direction focus")
    # reproducibility + series
    ap.add_argument("--mutation-seed", type=int, default=1234, help="seeds ALL mutation randomness — the recipe key")
    ap.add_argument("--life-generations", type=int, default=4, help="CA steps; one render per generation")
    # generation
    ap.add_argument("--prompt", action="append", default=None,
                    help=f"repeatable; default: {DEFAULT_PROMPT!r}")
    ap.add_argument("--seed", type=int, action="append", default=None, help="repeatable; default 1234")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--no-baseline", action="store_true", help="skip the unmutated A/B reference render")
    args = ap.parse_args()

    prompts = args.prompt or [DEFAULT_PROMPT]
    seeds = args.seed or [1234]

    if args.tour:
        conds = tour_conditions(args.mutation_seed)
    elif args.conditions:
        with open(args.conditions) as f:
            conds = [Condition(**c) for c in json.load(f)]
    else:
        ops = []
        for op in ("drift", "shuffle", "blur", "contrast", "tilt"):
            amt = getattr(args, op)
            if amt != 0.0:
                ops.append({"op": op, "amount": amt})
        if args.life:
            ops.append({"op": "life", "quantile": args.life_quantile})
        if not ops:
            ap.error("no mutation requested: use --tour, --conditions, or at least one op flag")
        name = "_".join(f"{o['op']}{o.get('amount', '')}" for o in ops)
        conds = [Condition(name=name, ops=ops, target=args.target,
                           decay_rate=args.decay_rate, decay_direction=args.decay_direction,
                           decay_focus=args.decay_focus, mutation_seed=args.mutation_seed)]

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] loading {args.model} on {device}...", flush=True)
    sam = StableAudioModel.from_pretrained(args.model, device=device)
    sr = sam.model.sample_rate

    # scope to the DiT ONLY: the T5-Gemma conditioner also has layers.N names,
    # and sam.model would expose them. sam.dit is the DiTWrapper the sampler uses.
    all_targets = collect_targets(sam.dit, target="all")
    print(f"[init] {len(all_targets)} mutable block params "
          f"(blocks 0..{max(t.block for t in all_targets)})", flush=True)
    snap = snapshot_targets(all_targets)

    manifest = {
        "purpose": "weight-mutation artifact exploration (weight garden)",
        "script": "stable-audio-3/scripts/mutate_weights.py",
        "library": "stable-audio-3/scripts/weight_mutations.py",
        "model": args.model,
        "generation": {"prompts": prompts, "seeds": seeds, "duration": args.duration,
                       "steps": args.steps, "cfg": args.cfg, "sampler": "euler"},
        "note": "every render is reproducible from its recipe (mutation_seed + ops); no ckpts saved",
        "renders": [],
    }

    def record(fname, cond, generation=None, summary=None, prompt=None, seed=None):
        entry = {"file": fname, "prompt": prompt, "gen_seed": seed}
        if cond is not None:
            entry["recipe"] = {**cond._asdict(), "life_generation": generation}
            entry["mutation"] = summary
        else:
            entry["recipe"] = "baseline (unmutated)"
        manifest["renders"].append(entry)

    t0 = time.time()
    if not args.no_baseline:
        for p in prompts:
            for s in seeds:
                fname = f"00_baseline_{slug(p)}_s{s}.wav"
                render(sam, sr, os.path.join(args.out_dir, fname), p, s, args)
                record(fname, None, prompt=p, seed=s)

    for ci, cond in enumerate(conds, start=1):
        restore_targets(all_targets, snap)
        is_life = any(o["op"] == "life" for o in cond.ops)
        generations = args.life_generations if is_life else 1
        for g in range(1, generations + 1):
            summary = apply_condition(sam.dit, cond)  # life: re-applying = next generation
            gtag = f"_gen{g}" if generations > 1 else ""
            for p in prompts:
                for s in seeds:
                    fname = f"{ci:02d}_{cond.name}{gtag}_{slug(p)}_s{s}.wav"
                    render(sam, sr, os.path.join(args.out_dir, fname), p, s, args)
                    record(fname, cond, generation=g if generations > 1 else None,
                           summary=summary, prompt=p, seed=s)
        print(f"[cond {ci}/{len(conds)}] {cond.name}: {summary['params_touched']} params, "
              f"blocks {summary['blocks_touched'][:4]}..", flush=True)

    restore_targets(all_targets, snap)
    manifest["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] {len(manifest['renders'])} renders + run_meta.json -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

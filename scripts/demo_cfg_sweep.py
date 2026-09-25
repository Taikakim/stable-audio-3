#!/usr/bin/env python3
"""Re-render a run's milestone demos at several cfg scales and report the PRE-CLAMP latent std.

WHY (CONTINUITY 2026-09-25): goa3_avp_r128_shampoo_b16_3e4_2026-09-25 produced all-NaN demos at
step 6340 and, earlier, 5 of 6 demo latents at step 3804 with pre-clamp std up to 1.7e10 -- which
--demo-latent-clamp scaled back to 1.25 before decode, so the clips looked survivable. Yet the
checkpoints' weights are finite and move smoothly. This asks the discriminating question:
  diverges at cfg 7 but clean at cfg 1-3  => guidance sensitivity; the checkpoint is usable.
  diverges at cfg 1 too                   => the model itself is broken.

FIDELITY. It calls the trainer's own `generate_clip_latents` (eval_demo_callback.py) with the same
prompts, seeds, 24 steps, bf16 base and frame counts, so cfg 7 reproduces the training-time demo.
SCHEDULE-FREE TRAP: a milestone .ckpt's state_dict holds the TRAINING point y; the demos render
the averaged iterate x (the callback swaps via the optimizer). x lives only in
optimizer_states[0]['state'][i]['x']. --weights x (default) rebuilds the adapter from x, keyed by
the optimizer's saved param_names; --weights y renders the state_dict as-is (what model_matrix_gen
and every other offline renderer see). No clamp is applied here: the raw latent is the measurement.

    cd /home/kim/Projects/SAO && export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE && \
      .venv/bin/python stable-audio-3/scripts/demo_cfg_sweep.py --run-dir <run> --steps-ckpt 3804 6340 --cfgs 1 3 7

Writes <out>/<tag>/<tag>_<prompt>_<len>_cfg<c>.z0.npy (+ .wav when finite and std < 5) and
<out>/sweep_results.jsonl, one line per clip.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_demo_callback import CANONICAL_DEMO_PROMPTS, generate_clip_latents  # noqa: E402


def adapter_state(ckpt_path: Path, weights: str):
    """Return (state_dict, lora_config, note) for the chosen Schedule-Free iterate."""
    ck = torch.load(str(ckpt_path), map_location="cpu", mmap=True, weights_only=False)
    sd = dict(ck["state_dict"])
    cfg = ck.get("lora_config", {})
    if weights == "y":
        return sd, cfg, "y (state_dict as saved)"
    opt = ck["optimizer_states"][0]
    swapped, missing, bnum, bden = 0, [], 0.0, 0.0
    for g in opt["param_groups"]:
        for name, idx in zip(g["param_names"], g["params"]):
            key = name.split(".", 1)[1]  # optimizer names are rooted one level above the state_dict
            st = opt["state"].get(idx)
            if key not in sd or st is None or "x" not in st:
                missing.append(name)
                continue
            x = st["x"].to(sd[key].dtype)
            if key.endswith("lora_B"):
                bnum += float((sd[key].float() - x.float()).pow(2).sum())
                bden += float(x.float().pow(2).sum())
            sd[key] = x
            swapped += 1
    if missing:
        raise SystemExit(f"{ckpt_path.name}: {len(missing)} adapter tensors have no x iterate, e.g. {missing[:3]}")
    rel = math.sqrt(bnum / bden) if bden else float("nan")
    return sd, cfg, f"x (averaged iterate, {swapped} tensors; |y-x|/|x| on lora_B = {rel:.3f})"


def merge_adapters(wrapper) -> int:
    """Bake every DoRA/LoRA parametrization into its weight, once.

    WHY (2026-09-25): with the parametrization live, the adapted weight is rebuilt on every
    forward, and on our ROCm 7.15-alpha stack that path returns HISTORY-DEPENDENT results --
    the first render in a process is bit-exact, later renders of the same input come back NaN or
    1e11 at random once shapes change between calls. Base model and merged adapter are
    bit-deterministic on the same sequence. Merging is exact up to bf16 rounding of the weight."""
    import torch.nn.utils.parametrize as P
    n = 0
    for mod in list(wrapper.model.modules()) + list(wrapper.conditioner.modules()):
        if P.is_parametrized(mod):
            for name in list(mod.parametrizations.keys()):
                with torch.no_grad():
                    P.remove_parametrizations(mod, name, leave_parametrized=True)
                n += 1
    return n


def load_model(sd, cfg, tmp_dir: Path, base: str, merge: bool = True):
    from stable_audio_3 import StableAudioModel
    slim = tmp_dir / "_sweep_adapter.ckpt"
    torch.save({"state_dict": sd or {}, "lora_config": cfg}, str(slim))
    m = StableAudioModel.from_pretrained(base, device="cuda", model_half=False)
    m.model.to(dtype=torch.bfloat16)  # the trainer's base precision (--base_precision bf16)
    if sd is not None:
        m.load_lora([str(slim)])
    slim.unlink()
    if sd is not None and merge:
        print(f"[sweep] merged {merge_adapters(m.model)} adapter parametrizations into the weights", flush=True)
    m.model.eval().requires_grad_(False)
    return m.model


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--steps-ckpt", nargs="+", type=int, required=True,
                    help="milestone steps, e.g. 3804 6340 (with --weights base, any one value; it only names the output)")
    ap.add_argument("--cfgs", nargs="+", type=float, default=[1.0, 3.0, 7.0])
    ap.add_argument("--weights", choices=("x", "y", "base"), default="x",
                    help="base = no adapter at all: the control arm that tells a sampler fault from a checkpoint fault")
    ap.add_argument("--num-prompts", type=int, default=3, help="the trainer's --eval_num_prompts (default 3)")
    ap.add_argument("--frames", nargs="+", type=int, default=[216, 256],
                    help="clip lengths in frames; the trainer's are 216 ('20s') and --frames ('48s', 256 on this run)")
    ap.add_argument("--only-prompts", nargs="*", default=None)
    ap.add_argument("--base", default="medium-base")
    ap.add_argument("--no-merge", dest="merge", action="store_false",
                    help="keep the adapter as a live parametrization (reproduces the ROCm nondeterminism; diagnostic only)")
    ap.add_argument("--out", type=Path, default=None, help="default <run-dir>/cfg_sweep")
    args = ap.parse_args()

    out = args.out or (args.run_dir / "cfg_sweep")
    out.mkdir(parents=True, exist_ok=True)
    results = out / "sweep_results.jsonl"
    prompts = CANONICAL_DEMO_PROMPTS[:args.num_prompts]
    if args.only_prompts:
        prompts = [p for p in prompts if p["id"] in args.only_prompts]

    import stable_audio_3.models.transformer as T  # the callback samples on SDPA (CK FMHA faults on gfx1201)
    T.flash_attn_func = None
    T.flash_attn_varlen_func = None

    for step in args.steps_ckpt:
        ckpt = args.run_dir / f"step={step}.ckpt"
        if args.weights == "base":
            sd, cfg, note = None, {}, "base (no adapter)"
        else:
            sd, cfg, note = adapter_state(ckpt, args.weights)
        print(f"[sweep] {ckpt.name}: weights {note}", flush=True)
        model = load_model(sd, cfg, out, args.base, merge=args.merge)
        del sd
        tag = f"step{step}_{args.weights}" + ("" if args.merge or args.weights == "base" else "_live")
        (out / tag).mkdir(exist_ok=True)
        for p in prompts:
            for frames in args.frames:
                for c in args.cfgs:
                    stem = f"{tag}_{p['id']}_{frames}f_cfg{c:g}"
                    t0 = time.time()
                    z = generate_clip_latents(model, p["text"], p["seed"], total_frames=frames,
                                              steps=24, cfg_scale=c, max_latent_std=None).float()
                    fin = torch.isfinite(z)
                    zf = z[fin]
                    rec = {"run": args.run_dir.name, "step": step, "weights": args.weights,
                           "prompt": p["id"], "frames": frames, "cfg": c,
                           "finite_frac": round(float(fin.float().mean()), 6),
                           "pre_clamp_std": float(zf.std()) if zf.numel() > 1 else None,
                           "max_abs": float(zf.abs().max()) if zf.numel() else None,
                           "secs": round(time.time() - t0, 1)}
                    np.save(out / tag / f"{stem}.z0.npy", z.numpy())
                    if rec["finite_frac"] == 1.0 and rec["pre_clamp_std"] < 5.0:
                        pt = model.pretransform
                        with torch.no_grad():
                            a = pt.decode(z.to(device="cuda", dtype=next(pt.parameters()).dtype))
                        a = a[0].float().cpu().numpy().T
                        pk = np.abs(a).max()
                        if pk > 1e-6:
                            a = a / pk * 0.988
                        sf.write(str(out / tag / f"{stem}.wav"), a, model.sample_rate, subtype="PCM_16")
                    with open(results, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"[sweep] {stem}: finite {rec['finite_frac']:.4f}  pre-clamp std {rec['pre_clamp_std']}  "
                          f"max|z| {rec['max_abs']}", flush=True)
        del model
        torch.cuda.empty_cache()
    print(f"[sweep] done -> {results}")


if __name__ == "__main__":
    main()

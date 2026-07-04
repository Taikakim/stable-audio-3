#!/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python
"""eval_dora_cpu.py — ALL-CPU audition-render harness for the SA3 DoRA finetune.

Auditions DoRA checkpoints from scripts/train_lora.py (adapter_type=dora-rows,
rank 16, base medium-base) OFF the GPU, so training can keep the card. It loads
the base SA3-medium DiT + conditioner on CPU (fp32), attaches the DoRA adapter and
loads the checkpoint into it (the run_gradio.py inference path — `model.load_lora`),
then generates 3 fixed prompts x 1 fixed seed, decodes, and writes 3 named clips.

NO GPU is touched (device="cpu", model_half=False) and nothing the running training
job depends on is modified — it only reads checkpoint files.

Why "load" not "merge": dora-rows changes the DiT Linear WEIGHTS as
    W' = magnitude * V / ||V||_row ,  V = W + (alpha/rank) * B @ A
(per-output-row column-... per-row L2 norm). The control-adapter ONNX graph can NOT
be reused (DoRA is a weight edit, not a forward-only cross-attn add). The correct,
already-validated application of that exact formula lives in
`stable_audio_3.models.lora.model.LoRAParametrization.lora_forward` (adapter_type
"dora-rows"). `StableAudioModel.load_lora()` -> `load_and_apply_loras()` attaches
that parametrization and loads {lora_A, lora_B, magnitude} from the checkpoint, so
the parametrization recomputes W' on every forward. We therefore "load + attach"
(parametrized) rather than hand-merging — same math, no risk of a wrong norm axis.
(If you ever need a true static merge, use
`stable_audio_3.models.lora.utils.merge_loras_into_base_model`, which calls the same
`lora_p(original)` forward — never a plain `W += BA` add.)

CPU device gotcha (MASTER.md §5): load via from_pretrained(device="cpu",
model_half=False) so the 1.4B weights never touch the GPU. Do NOT set
HIP_VISIBLE_DEVICES="" — flash_attn/aiter probes a Triton driver at import and
crashes with no visible device. We just never allocate on the GPU. SA3_DISABLE_FLASH_ATTN=1
forces the math-SDPA attention path (CPU-safe).

Usage
-----
    cd /home/kim/Projects/SAO/stable-audio-3
    VENV=.venv/bin/python

    # one checkpoint
    $VENV scripts/eval_dora_cpu.py --ckpt <run>/checkpoints/epoch=9-step=50.ckpt \
        --out-dir renders_dora/ep9

    # loop every checkpoint in a run (skips ones already rendered); --watch to poll
    $VENV scripts/eval_dora_cpu.py \
        --run-dir /run/media/kim/Mantu/sa3_lora_runs/sa3-goa-dora-47s \
        --out-dir renders_dora --watch

    # base model sanity (no DoRA)
    $VENV scripts/eval_dora_cpu.py --base --out-dir renders_dora/base
"""
# --- device-aware env MUST precede torch import ----------------------------------
# We peek at --device in argv (default cuda) BEFORE importing torch so the flash-attn
# / thread-cap envs are set correctly for the chosen path:
#   * cpu  -> SA3_DISABLE_FLASH_ATTN=1 (math-SDPA, CPU-safe) + 12-thread caps
#   * cuda -> FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE to activate the CK flash-attn
#             build (MASTER §5: this venv has aiter, FALSE switches Triton->CK; 30-100% faster)
import os
import sys as _sys


def _peek_device(argv) -> str:
    dev = "cuda"
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            dev = argv[i + 1]
        elif a.startswith("--device="):
            dev = a.split("=", 1)[1]
    return dev.strip().lower()


_DEVICE = _peek_device(_sys.argv)
if _DEVICE == "cpu":
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "12")
    os.environ.setdefault("SA3_DISABLE_FLASH_ATTN", "1")
else:
    # GPU: activate the CK flash-attn build for speed (MASTER §5).
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# Do NOT set HIP_VISIBLE_DEVICES="" — flash_attn/aiter import probes a driver.

import argparse
import re
import sys
import time
from pathlib import Path

import torch

# --- audio writer: prefer the shared no-clip helper, fall back to an identical copy
try:
    sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools/avp_sa3")
    from sa3_control.audio_io import save_audio  # float32->peak-norm->clamp->PCM16
except Exception:  # pragma: no cover - keep the harness self-contained
    import soundfile as sf

    def save_audio(path, audio, sr, normalize=True):
        a = audio.detach().to(torch.float32).cpu()
        if a.dim() == 1:
            a = a.unsqueeze(0)
        peak = torch.max(torch.abs(a))
        if normalize and peak > 1e-6:
            a = a / peak
        a = a.clamp(-1.0, 1.0)
        arr = a.transpose(0, 1).contiguous().numpy()
        sf.write(str(path), arr, int(sr), subtype="PCM_16")

from stable_audio_3 import StableAudioModel  # noqa: E402

# The familiar three prompts + the FiLM control-eval seed (onset_eval.py / multi_eval.py
# default = 1234) so these renders are directly comparable to the control evals.
PROMPTS = [
    "aggressive upbeat goa trance",
    "energetic acid techno, 130 BPM, driving analog bassline, crisp drum machine",
    "psytrance, 140 bpm",
]


def _step_of(ckpt: Path) -> int:
    """Sort key from 'epoch=E-step=S.ckpt'; falls back to mtime-ish 0."""
    m = re.search(r"step=(\d+)", ckpt.name)
    return int(m.group(1)) if m else 0


def find_checkpoints(run_dir: Path):
    """Every .ckpt / .safetensors under run_dir (recursively), ordered by step."""
    cks = []
    for pat in ("*.ckpt", "*.safetensors"):
        cks += list(run_dir.rglob(pat))
    # drop obvious non-adapter safetensors (base weights) — keep only files that look
    # like adapter checkpoints (have a step= or live under a checkpoints/ dir)
    cks = [c for c in cks if "step=" in c.name or c.parent.name == "checkpoints"]
    return sorted(set(cks), key=_step_of)


def ckpt_tag(ckpt: Path) -> str:
    """Stable, filesystem-safe tag for output names: '<run_id>_<ckpt_stem>'."""
    stem = ckpt.stem.replace("=", "").replace(".", "")
    # parent of 'checkpoints' is usually the run id (wandb/comet) or run name
    run_id = ckpt.parent.parent.name if ckpt.parent.name == "checkpoints" else ckpt.parent.name
    return f"{run_id}_{stem}"


def load_model(ckpt: Path | None, device: str = "cuda"):
    if device == "cpu":
        print("[load] medium-base on CPU (fp32, flash-attn off) — this is the slow part ...", flush=True)
        model_half = False
    else:
        print("[load] medium-base on GPU (fp16, CK flash-attn) ...", flush=True)
        model_half = True
    t0 = time.time()
    model = StableAudioModel.from_pretrained("medium-base", device=device, model_half=model_half)
    print(f"[load] base resident in {time.time() - t0:.0f}s", flush=True)
    if ckpt is not None:
        print(f"[lora] attaching DoRA + loading {ckpt}", flush=True)
        t0 = time.time()
        model.load_lora([str(ckpt)])
        n = len(getattr(model.model, "_lora_layers", []) or [])
        try:
            from stable_audio_3.models.lora.utils import get_lora_layers
            n = len(get_lora_layers(model.model))
        except Exception:
            pass
        print(f"[lora] applied in {time.time() - t0:.0f}s  ({n} parametrized layers)", flush=True)
    return model


def render(model, out_dir: Path, tag: str, steps: int, duration: float,
           seeds: list[int], cfg_scale: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = model.model.sample_rate
    times = []
    for seed in seeds:
        for i, prompt in enumerate(PROMPTS):
            out_path = out_dir / f"{tag}__p{i}_seed{seed}.wav"
            if out_path.exists():
                print(f"[skip] {out_path.name} exists", flush=True)
                continue
            t0 = time.time()
            audio = model.generate(
                prompt=prompt,
                duration=duration,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                batch_size=1,
            )  # (1, C, samples) float32 clamped [-1,1]
            dt = time.time() - t0
            times.append(dt)
            a = audio[0]
            finite = bool(torch.isfinite(a).all())
            peak = float(torch.max(torch.abs(a)))
            save_audio(out_path, a, sr, normalize=True)
            print(f"[clip] p{i} seed{seed} {dt:6.1f}s  finite={finite} peak={peak:.3f}  "
                  f"prompt={prompt!r} -> {out_path}", flush=True)
            if not finite:
                print(f"[WARN] p{i} seed{seed} produced non-finite samples!", flush=True)
    if times:
        print(f"[time] {tag}: {len(times)} clips, "
              f"mean {sum(times)/len(times):.1f}s/clip "
              f"(min {min(times):.1f}, max {max(times):.1f})", flush=True)
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=Path, help="single DoRA checkpoint (.ckpt/.safetensors)")
    g.add_argument("--run-dir", type=Path, help="loop every checkpoint under this dir")
    g.add_argument("--base", action="store_true", help="render the BASE model (no DoRA)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=16, help="RF sampler steps (16-24 typical)")
    ap.add_argument("--duration", type=float, default=47.0, help="clip seconds (match training crop)")
    ap.add_argument("--seed", type=int, default=1234, help="fixed seed (control-eval default)")
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma list of seeds, e.g. '1234,42' — renders each ckpt x prompt "
                         "once PER seed (overrides --seed)")
    ap.add_argument("--device", type=str, default="cuda",
                    help="cuda (GPU, fp16, CK flash-attn — fast) or cpu (fp32, math-SDPA)")
    ap.add_argument("--cfg-scale", type=float, default=6.0, help="CFG scale (control-eval default)")
    ap.add_argument("--threads", type=int, default=12, help="torch CPU threads (physical cores)")
    ap.add_argument("--watch", action="store_true",
                    help="with --run-dir: keep polling for new checkpoints")
    ap.add_argument("--poll", type=float, default=60.0, help="--watch poll seconds")
    args = ap.parse_args()

    device = args.device.strip().lower()
    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else [args.seed])

    torch.set_num_threads(max(1, args.threads))
    print(f"[{device}] torch threads={torch.get_num_threads()}  "
          f"steps={args.steps} duration={args.duration}s seeds={seeds} cfg={args.cfg_scale}",
          flush=True)
    if device == "cuda" and not torch.cuda.is_available():
        print("[error] --device cuda but torch.cuda.is_available() is False", flush=True)
        return

    if args.base:
        model = load_model(None, device)
        render(model, args.out_dir, "base", args.steps, args.duration, seeds, args.cfg_scale)
        return

    if args.ckpt:
        model = load_model(args.ckpt, device)
        render(model, args.out_dir, ckpt_tag(args.ckpt), args.steps, args.duration,
               seeds, args.cfg_scale)
        return

    # --run-dir: load base ONCE, then load each adapter in turn. NOTE: load_lora stacks
    # (lora_index increments) — to keep each render isolated we reload the base per ckpt.
    seen: set[str] = set()
    while True:
        cks = [c for c in find_checkpoints(args.run_dir)]
        todo = []
        for c in cks:
            tag = ckpt_tag(c)
            done = all((args.out_dir / f"{tag}__p{i}_seed{s}.wav").exists()
                       for s in seeds for i in range(len(PROMPTS)))
            if not done and tag not in seen:
                todo.append(c)
        if not todo and not args.watch:
            if not cks:
                print(f"[warn] no checkpoints found under {args.run_dir}", flush=True)
            else:
                print("[done] all checkpoints already rendered", flush=True)
            return
        for c in todo:
            tag = ckpt_tag(c)
            print(f"\n=== {tag} ({c}) ===", flush=True)
            model = load_model(c, device)    # fresh base each time -> isolated adapter
            render(model, args.out_dir, tag, args.steps, args.duration, seeds, args.cfg_scale)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            seen.add(tag)
        if not args.watch:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()

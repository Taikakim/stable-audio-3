#!/usr/bin/env python3
"""Repro for the live-DoRA render fault (SAO docs/training-findings.md 13e).

Renders ONE prompt at a fixed seed several times in one process, interleaving calls of different
shapes (cfg 1 = batch 1, cfg 7 = batch 2, plus a SAME decode). A correct stack returns the same
latent every time: the cfg-1 renders must all print the SAME std, bit for bit.

What we saw 2026-09-25 (RX 9070 XT, gfx1201): the first render in a process is always exact
(std 0.9581015706 on the default checkpoint), later ones come back NaN, 1e7-1e11, or slightly
off, and some runs die with a GPU memory fault in an unrelated kernel. Reproduced on BOTH the
ROCm 7.15-alpha venv and the ROCm 7.2.3 venv, under linux 7.2.6-zen2. Exact with
PYTORCH_NO_CUDA_MEMORY_CACHING=1, with --mode merged, and on the base model (--mode base).

    cd /home/kim/Projects/SAO && export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE ROCR_VISIBLE_DEVICES=0 && \
      .venv/bin/python stable-audio-3/scripts/diag_dora_render_determinism.py --mode live

Exit code 0 = every cfg-1 render matched the first one; 1 = mismatch (the fault is present).
WARNING: a failing run can GPU-fault; on a card that also drives the display this has taken the
desktop compositor down twice. Run it on a compute-only card.
"""
import argparse
import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_cfg_sweep import adapter_state, load_model  # noqa: E402
from eval_demo_callback import CANONICAL_DEMO_PROMPTS, generate_clip_latents  # noqa: E402

RUN = "/run/media/kim/Mantu/sa3_lora_runs/goa3_avp_r128_shampoo_b16_3e4_2026-09-25"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("live", "merged", "base"), default="live")
    ap.add_argument("--ckpt", default=f"{RUN}/step=3804.ckpt")
    ap.add_argument("--sdpa", action="store_true", help="route the DiT through SDPA like the demo callback (default: CK)")
    ap.add_argument("--no-flex", action="store_true",
                    help="disable FlexAttention (compiled Triton): windowed attention falls back to masked SDPA")
    a = ap.parse_args()
    if a.no_flex:
        import stable_audio_3.models.transformer as T
        T.flex_attention_available = False
        T.flex_attention_compiled = None

    if a.sdpa:
        import stable_audio_3.models.transformer as T
        T.flash_attn_func = None
        T.flash_attn_varlen_func = None
    if a.mode == "base":
        sd, cfg = None, {}
    else:
        sd, cfg, _ = adapter_state(Path(a.ckpt), "x")
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    m = load_model(sd, cfg, tmp, "medium-base", merge=(a.mode == "merged"))
    p = CANONICAL_DEMO_PROMPTS[0]
    pt = m.pretransform
    stds = []

    def render(tag, c):
        z = generate_clip_latents(m, p["text"], p["seed"], total_frames=216, steps=24, cfg_scale=c).float()
        fin = torch.isfinite(z)
        s = float(z[fin].std()) if fin.any() else None
        print(f"{a.mode} {tag}: finite {float(fin.float().mean()):.4f} std {s!r}", flush=True)
        return z, s

    def render_decode(tag):
        z, _ = render(tag + " (cfg7)", 7.0)
        with torch.no_grad():
            pt.decode(z.to(device="cuda", dtype=next(pt.parameters()).dtype))

    stds.append(render("fresh cfg1", 1.0)[1])
    render_decode("render+decode #1")
    stds.append(render("cfg1 after decode", 1.0)[1])
    stds.append(render("cfg1 again", 1.0)[1])
    torch.cuda.empty_cache()
    stds.append(render("cfg1 after empty_cache", 1.0)[1])
    render_decode("render+decode #2")
    stds.append(render("cfg1 after decode #2", 1.0)[1])
    ok = all(s == stds[0] for s in stds)
    print(f"{a.mode}: {'DETERMINISTIC' if ok else 'MISMATCH'} over {len(stds)} cfg-1 renders", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

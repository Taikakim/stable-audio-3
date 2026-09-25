#!/usr/bin/env python3
"""Strength-equivalence check for model_matrix_gen.py's merge-before-render fix (13e).

WINTERMUTE's review of commit 662b4af (2026-09-25), point 1: a w=1 equivalence check alone
cannot see set_lora_strength(w) silently dropped at merge time -- only a w!=1 check can, because
at w=1 a bug that always merges "the raw adapter, ignoring w" would happen to look correct too.

Compares the LIVE path (set_lora_strength(w), no merge -- what model_matrix_gen.py did for every
w before the fix, via a shared model's live parametrization) against the MERGED path
(set_lora_strength(w) THEN merge_adapters() -- the new per-w adapter branch) on the SAME cell,
same seed, at w=1.0 and w=0.5. Each invocation is a fresh process with exactly one render call,
so this never crosses the render-fault's own history-dependent nondeterminism (training-findings
13e; that needs shape-interleaving WITHIN one process, which a single render call can't trigger).

Run all four combinations (one process each), then compare the saved latents:

    cd /home/kim/Projects/SAO && export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE ROCR_VISIBLE_DEVICES=0
    .venv/bin/python stable-audio-3/scripts/diag_dora_strength_equivalence.py --mode live   --strength 1.0
    .venv/bin/python stable-audio-3/scripts/diag_dora_strength_equivalence.py --mode merged --strength 1.0
    .venv/bin/python stable-audio-3/scripts/diag_dora_strength_equivalence.py --mode live   --strength 0.5
    .venv/bin/python stable-audio-3/scripts/diag_dora_strength_equivalence.py --mode merged --strength 0.5
    .venv/bin/python stable-audio-3/scripts/diag_dora_strength_equivalence.py --compare
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_cfg_sweep import merge_adapters  # noqa: E402
from eval_demo_callback import CANONICAL_DEMO_PROMPTS, generate_clip_latents  # noqa: E402

ADAPTER = "/run/media/kim/Mantu/sa3_lora_runs/goa3_avp_r128_shampoo_b16_3e4_2026-09-25/slim_adapter_x_step3804.pt"
OUT_DIR = Path(os.environ.get("TMPDIR", "/tmp"))


def out_path(mode: str, strength: float) -> Path:
    return OUT_DIR / f"strength_equiv_{mode}_w{strength}.pt"


def render(mode: str, strength: float, ckpt: str, sdpa: bool) -> None:
    if sdpa:
        import stable_audio_3.models.transformer as T
        T.flash_attn_func = None
        T.flash_attn_varlen_func = None

    from stable_audio_3 import StableAudioModel
    m = StableAudioModel.from_pretrained("medium-base", device="cuda", model_half=False)
    m.model.to(dtype=torch.bfloat16)
    m.load_lora([ckpt])
    m.set_lora_strength(strength)
    if mode == "merged":
        n = merge_adapters(m.model)
        print(f"[equiv] merged {n} adapter parametrizations at strength {strength}", flush=True)
    m.model.eval().requires_grad_(False)

    p = CANONICAL_DEMO_PROMPTS[0]
    z = generate_clip_latents(m.model, p["text"], p["seed"], total_frames=216, steps=24, cfg_scale=7.0).float()
    fin = torch.isfinite(z)
    s = float(z[fin].std()) if fin.any() else None
    op = out_path(mode, strength)
    torch.save(z.cpu(), str(op))
    print(f"[equiv] mode={mode} strength={strength}: finite {float(fin.float().mean()):.4f} std {s!r} saved {op}", flush=True)


def compare() -> int:
    ok = True
    for w in (1.0, 0.5):
        lp, mp = out_path("live", w), out_path("merged", w)
        if not (lp.exists() and mp.exists()):
            print(f"[equiv] w={w}: MISSING ({lp.exists()=} {mp.exists()=})", flush=True)
            ok = False
            continue
        zl, zm = torch.load(lp), torch.load(mp)
        fin = torch.isfinite(zl) & torch.isfinite(zm)
        if not fin.any():
            print(f"[equiv] w={w}: NO FINITE OVERLAP", flush=True)
            ok = False
            continue
        a, b = zl[fin], zm[fin]
        cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
        maxdiff = float((a - b).abs().max())
        print(f"[equiv] w={w}: cos={cos:.6f} max_abs_diff={maxdiff:.6f} "
              f"live_std={float(zl[torch.isfinite(zl)].std()):.6f} merged_std={float(zm[torch.isfinite(zm)].std()):.6f}",
              flush=True)
        if cos < 0.999 or maxdiff > 0.05:
            ok = False
    print(f"[equiv] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("live", "merged"))
    ap.add_argument("--strength", type=float)
    ap.add_argument("--ckpt", default=ADAPTER)
    ap.add_argument("--sdpa", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare())
    if a.mode is None or a.strength is None:
        ap.error("--mode and --strength are required unless --compare")
    render(a.mode, a.strength, a.ckpt, a.sdpa)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""dit_onnx_infer.py — end-to-end SA3 text→audio on AMD via the ONNX DiT + ONNX
decoder (ORT + MIGraphX). The host loop here; the GPU only runs the two ONNX
sessions.

Pipeline (text precached offline in the SA3 venv → pure numpy/ORT at runtime):

    cond/uncond .npz {cross_attn_cond[seq,768], cross_attn_mask[seq], global_embed[768]}
        │
        ▼  rectified-flow Euler, CFG (two batch-1 DiT calls/step)
    DiT-onnx[L]  ×steps   →  z0[1,256,T]  →  same_decoder ONNX (chunk-loop)  →  wav

CFG simplifies (rectified-flow, vanilla): per step a cond + uncond DiT call, then
v = v_uncond + cfg·(v_cond − v_uncond);  x += dt·v. (The DiT is exported static
batch=1; a batch=2 export would let CFG be one call/step — an efficiency option.)

Runs in the **mir venv** (onnxruntime_migraphx). Reuses decode_onnx's validated
decoder chunk-loop. `--frames` (T) MUST match the DiT export's length rung.

Usage
-----
    python scripts/dit_onnx_infer.py \\
        --dit-onnx dit_medium-base_L256.onnx --decoder-onnx same_decoder_L128.onnx \\
        --cond cond.npz --uncond uncond.npz \\
        --frames 256 --steps 8 --cfg-scale 6.0 --provider migraphx --out gen.wav
"""
import argparse
import time
from pathlib import Path

import numpy as np

# Reuse the validated decoder chunk-loop + provider selection.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_onnx import decode_chunked_onnx, pick_providers  # noqa: E402

SR = 44100
LATENT_DIM = 256
DS = 4096



def schedule(steps: int, T: int, alpha_min: float = 1.0, alpha_max: float = 1.0,
             min_len: int = 256, max_len: int = 4096) -> np.ndarray:
    """Rectified-flow sigma schedule = linspace(1,0,steps+1), warped by the SD3/
    Flux time-shift. medium-base uses alpha_min=alpha_max=1.0 → identity."""
    import math
    t = np.linspace(1.0, 0.0, steps + 1).astype(np.float64)
    log_amin, log_amax = math.log(max(alpha_min, 1e-8)), math.log(max(alpha_max, 1e-8))
    log_lo, log_hi = math.log(min_len), math.log(max(max_len, min_len + 1))
    seqc = max(min(T, max_len), min_len)
    frac = (math.log(seqc) - log_lo) / (log_hi - log_lo)
    alpha = math.exp(log_amin + frac * (log_amax - log_amin))
    if abs(alpha - 1.0) > 1e-9:
        t = alpha * t / (1.0 + (alpha - 1.0) * t)
        t[0] = 1.0      # keep first step aligned with sigma_max
    return t


def load_cond(path: Path):
    z = np.load(path)
    return (z["cross_attn_cond"].astype(np.float32),
            z["cross_attn_mask"],
            z["global_embed"].astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit-onnx", required=True, type=Path)
    ap.add_argument("--decoder-onnx", required=True, type=Path)
    ap.add_argument("--cond", required=True, type=Path, help="npz: cross_attn_cond,cross_attn_mask,global_embed")
    ap.add_argument("--uncond", required=True, type=Path, help="npz for the negative/empty prompt")
    ap.add_argument("--frames", type=int, required=True, help="T — MUST match the DiT export rung")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode-chunk", type=int, default=128, help="MUST match the decoder export L")
    ap.add_argument("--decode-overlap", type=int, default=16)
    ap.add_argument("--provider", default="migraphx")
    ap.add_argument("--ep-fp16", action="store_true",
                    help="run the MIGraphX EP in fp16 — halves VRAM so the fp32 DiT (~5.8GB) "
                         "and decoder fit together on 16GB without thrashing, and is faster")
    ap.add_argument("--alpha-min", type=float, default=1.0)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("dit_gen.wav"))
    args = ap.parse_args()

    import onnxruntime as ort
    import soundfile as sf

    providers = pick_providers(args.provider)
    if args.ep_fp16:
        providers = [("MIGraphXExecutionProvider", {"migraphx_fp16_enable": "1"})
                     if p == "MIGraphXExecutionProvider" else p for p in providers]
    print(f"[ort] providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}"
          + (" (fp16 EP)" if args.ep_fp16 else ""))
    print("[ort] compiling DiT + decoder (one-time MIGraphX AOT here) ...")
    t0 = time.time()
    dit = ort.InferenceSession(str(args.dit_onnx), providers=providers)
    dec = ort.InferenceSession(str(args.decoder_onnx), providers=providers)
    print(f"[ort] sessions ready in {time.time() - t0:.0f}s  "
          f"(DiT EP {dit.get_providers()[0]}, decoder EP {dec.get_providers()[0]})")

    c_cross, c_mask, c_glob = load_cond(args.cond)
    u_cross, u_mask, u_glob = load_cond(args.uncond)
    T = args.frames

    # CFG = two batch-1 DiT calls/step (cond, then uncond). The DiT is exported
    # static batch=1; batched CFG would need a batch=2 export (one call/step) — an
    # efficiency optimization, not required for correctness.
    # local_add_cond = cat(inpaint_mask[1], inpaint_masked_input[256]) = 257ch; all
    # zeros for text-to-audio (no inpaint). NOT a no-op (the DiT projects it with a
    # bias) so it must be fed, not omitted.
    LOCAL_ADD_DIM = 257
    local_add = np.zeros((1, LOCAL_ADD_DIM, T), np.float32)

    def dit_v(cross1, mask1, glob1, t_cur):
        return dit.run(None, {
            "x": x, "t": np.full((1,), t_cur, np.float32),
            "cross_attn_cond": cross1[None], "cross_attn_cond_mask": mask1[None],
            "global_embed": glob1[None], "local_add_cond": local_add})[0]

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((1, LATENT_DIM, T)).astype(np.float32)         # sigma_max=1.0
    sig = schedule(args.steps, T, args.alpha_min, args.alpha_max)

    print(f"[gen] T={T}  steps={args.steps}  cfg={args.cfg_scale}")
    t0 = time.time()
    for i in range(args.steps):
        t_cur, dt = float(sig[i]), float(sig[i + 1] - sig[i])
        v_cond = dit_v(c_cross, c_mask, c_glob, t_cur)
        if args.cfg_scale == 1.0:
            v = v_cond
        else:
            v_unc = dit_v(u_cross, u_mask, u_glob, t_cur)
            v = v_unc + args.cfg_scale * (v_cond - v_unc)                  # CFG (velocity space)
        x = x + dt * v
    print(f"[gen] {args.steps} steps in {time.time() - t0:.2f}s -> z0 {x.shape} "
          f"range[{x.min():.2f},{x.max():.2f}]")
    np.save(str(args.out) + ".z0.npy", x)

    audio = decode_chunked_onnx(dec, x, args.decode_chunk, args.decode_overlap, DS)
    a = np.clip(audio[0], -1.0, 1.0).T
    sf.write(str(args.out), a, SR, subtype="PCM_16")
    print(f"[out] wrote {args.out}  ({audio.shape[-1] / SR:.1f}s)")


if __name__ == "__main__":
    main()

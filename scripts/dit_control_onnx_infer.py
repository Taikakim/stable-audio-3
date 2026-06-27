#!/usr/bin/env python
"""dit_control_onnx_infer.py — ONSET-STEERED text→audio via the control-DiT ONNX
(ORT + MIGraphX). Sibling of dit_onnx_infer.py; the DiT graph here has the scalar
control adapter baked in (export_dit_control_onnx.py), so generation can dial the
output's onset density.

Per CFG step, exactly as the trained sa3_control/onset_eval.py:
    cond pass   -> control_tokens = conditioner((onset_target - mean)/std)
    uncond pass -> control_tokens = zeros  (the trained null)
    v = v_uncond + cfg * (v_cond - v_uncond)
so the control rides the CFG axis; `--gain` dials strength (1=as trained, ~2-4 strong).

The scalar conditioner is the numpy port of the FiLM saved in the .cond.npz next to
the .onnx — no torch at runtime. mir venv.

    python scripts/dit_control_onnx_infer.py \
        --dit-onnx dit_medium-base_L256_ctrl_onset_density.onnx \
        --cond-npz dit_medium-base_L256_ctrl_onset_density.cond.npz \
        --decoder-onnx same_decoder_L128.onnx \
        --cond real.cond.npz --uncond real.uncond.npz \
        --frames 256 --onset-density 8 --gain 3 --out steered.wav
"""
import argparse
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_onnx import decode_chunked_onnx, pick_providers  # noqa: E402
from sa3_control_onnx import (  # noqa: E402
    SR, DS, load_cond, resolve_host_pe, make_control_tokens, generate_z0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit-onnx", required=True, type=Path, help="the CONTROL DiT onnx")
    ap.add_argument("--cond-npz", required=True, type=Path, help="the .cond.npz beside it")
    ap.add_argument("--decoder-onnx", required=True, type=Path)
    ap.add_argument("--cond", required=True, type=Path, help="text cond npz (cross/mask/global)")
    ap.add_argument("--uncond", required=True, type=Path, help="text uncond npz")
    ap.add_argument("--frames", type=int, required=True, help="T — MUST match the DiT export rung")
    ap.add_argument("--onset-density", type=float, default=8.0, help="RAW target onset density (onsets/sec)")
    ap.add_argument("--gain", type=float, default=3.0, help="control strength (1=as trained, ~2-4 strong)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode-chunk", type=int, default=128)
    ap.add_argument("--decode-overlap", type=int, default=16)
    ap.add_argument("--provider", default="migraphx")
    ap.add_argument("--ep-fp16", action="store_true")
    ap.add_argument("--alpha-min", type=float, default=1.0)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU intra-op threads (0=ORT default). On the 12-core/24-thread 9900X, 12 "
                         "(physical cores) is ~25%% faster than 24 — SMT/bandwidth bound. Ignored on GPU.")
    ap.add_argument("--out", type=Path, default=Path("dit_ctrl_gen.wav"))
    args = ap.parse_args()

    import onnxruntime as ort
    import soundfile as sf

    providers = pick_providers(args.provider)
    if args.ep_fp16:
        providers = [("MIGraphXExecutionProvider", {"migraphx_fp16_enable": "1"})
                     if p == "MIGraphXExecutionProvider" else p for p in providers]
    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
    print(f"[ort] compiling control-DiT + decoder ... (provider={args.provider}, threads="
          f"{args.threads or 'default'})")
    t0 = time.time()
    dit = ort.InferenceSession(str(args.dit_onnx), sess_options=so, providers=providers)
    dec = ort.InferenceSession(str(args.decoder_onnx), sess_options=so, providers=providers)
    print(f"[ort] sessions ready in {time.time() - t0:.0f}s  (DiT EP {dit.get_providers()[0]})")

    cond = load_cond(args.cond)
    uncond = load_cond(args.uncond)
    T = args.frames

    cz = np.load(args.cond_npz)
    # The export stamps host_pe into the onnx metadata; resolve_host_pe asserts the
    # .cond.npz agrees (catches a mismatched pair → silent double/zero PE).
    host_pe = resolve_host_pe(dit, cz)
    # control ON (requested onset) + OFF (trained null); host PE applied to both if needed.
    cond_tok, zero_tok = make_control_tokens(cz, args.onset_density, host_pe)
    print(f"[ctrl] field={str(cz['field'])}  onset_density={args.onset_density} "
          f"(norm {(args.onset_density - float(cz['mean'])) / float(cz['std']):+.2f})  gain={args.gain}  "
          f"host_pe={host_pe}  ||cond_tok||={np.linalg.norm(cond_tok):.1f}")

    print(f"[gen] T={T}  steps={args.steps}  cfg={args.cfg_scale}")
    res = generate_z0(dit, cond=cond, uncond=uncond, cond_tok=cond_tok, zero_tok=zero_tok,
                      frames=T, steps=args.steps, cfg_scale=args.cfg_scale, seed=args.seed,
                      gain=args.gain, alpha_min=args.alpha_min, alpha_max=args.alpha_max)
    x = res["z0"]
    print(f"[gen] {args.steps} steps in {res['dit_loop_s']:.2f}s -> z0 {x.shape} "
          f"range[{x.min():.2f},{x.max():.2f}]")
    np.save(str(args.out) + ".z0.npy", x)

    audio = decode_chunked_onnx(dec, x, args.decode_chunk, args.decode_overlap, DS)
    a = np.clip(audio[0], -1.0, 1.0).T
    sf.write(str(args.out), a, SR, subtype="PCM_16")
    print(f"[out] wrote {args.out}  ({audio.shape[-1] / SR:.1f}s)")


if __name__ == "__main__":
    main()

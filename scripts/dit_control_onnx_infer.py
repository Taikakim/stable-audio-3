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

SR = 44100
LATENT_DIM = 256
DS = 4096
LOCAL_ADD_DIM = 257


def schedule(steps: int, T: int, alpha_min: float = 1.0, alpha_max: float = 1.0,
             min_len: int = 256, max_len: int = 4096) -> np.ndarray:
    import math
    t = np.linspace(1.0, 0.0, steps + 1).astype(np.float64)
    log_amin, log_amax = math.log(max(alpha_min, 1e-8)), math.log(max(alpha_max, 1e-8))
    log_lo, log_hi = math.log(min_len), math.log(max(max_len, min_len + 1))
    seqc = max(min(T, max_len), min_len)
    frac = (math.log(seqc) - log_lo) / (log_hi - log_lo)
    alpha = math.exp(log_amin + frac * (log_amax - log_amin))
    if abs(alpha - 1.0) > 1e-9:
        t = alpha * t / (1.0 + (alpha - 1.0) * t)
        t[0] = 1.0
    return t


def load_cond(path: Path):
    z = np.load(path)
    return (z["cross_attn_cond"].astype(np.float32), z["cross_attn_mask"],
            z["global_embed"].astype(np.float32))


def control_tokens_from_npz(z, raw_value: float) -> np.ndarray:
    """ScalarAttributeEncoder forward in numpy: scalar -> (1, n_tokens, control_dim)."""
    s = np.float32((raw_value - float(z["mean"])) / float(z["std"]))
    x = np.array([[s]], np.float32)
    h = x @ z["film0_w"].T + z["film0_b"]
    h = h * (1.0 / (1.0 + np.exp(-h)))                                  # SiLU
    nt, cd = int(z["n_tokens"]), int(z["control_dim"])
    gb = (h @ z["film2_w"].T + z["film2_b"]).reshape(1, nt, cd, 2)
    return (z["tokens"][None] * (1.0 + gb[..., 0]) + gb[..., 1]).astype(np.float32)


def add_fractional_positions_np(tokens: np.ndarray) -> np.ndarray:
    """numpy port of sa3_control.adapters.add_fractional_positions. Identical sin/cos
    layout. Applied HOST-SIDE when the control-DiT was exported with the adapter PE
    disabled (so the fp16 graph has no ConstantOfShape) — the result is the same as the
    in-adapter PE because PE is added to the tokens before the K/V projection."""
    b, s, d = tokens.shape
    if s <= 1:
        return tokens
    half = d // 2
    if half == 0:
        return tokens
    pos = np.linspace(0.0, 1.0, s, dtype=np.float32)
    freq = np.exp(np.linspace(0.0, 8.0, half, dtype=np.float32))
    ang = pos[:, None] * freq[None, :] * np.float64(np.pi)   # fp64 pi (matches torch's pi)
    pe = np.zeros((s, d), np.float32)
    pe[:, :half] = np.sin(ang)
    pe[:, half:half + half] = np.cos(ang)
    return (tokens + pe[None]).astype(np.float32)


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

    c_cross, c_mask, c_glob = load_cond(args.cond)
    u_cross, u_mask, u_glob = load_cond(args.uncond)
    T = args.frames
    local_add = np.zeros((1, LOCAL_ADD_DIM, T), np.float32)
    gain = np.array([args.gain], np.float32)

    cz = np.load(args.cond_npz)
    cond_tok = control_tokens_from_npz(cz, args.onset_density)          # control ON (requested onset)
    zero_tok = np.zeros_like(cond_tok)                                  # control OFF (trained null)
    # If the adapter PE was moved host-side at export (fp16-friendly graph), apply it to
    # BOTH passes here — exactly as the in-adapter PE did (uncond's zeros become PE, not 0).
    host_pe = bool(cz["host_pe"]) if "host_pe" in cz.files else False
    # Guard against a mismatched .cond.npz/.onnx pair (would double-apply or drop the PE → wrong
    # CFG, silently). The export stamps host_pe into the onnx metadata; assert it agrees.
    onnx_meta = dit.get_modelmeta().custom_metadata_map
    if "host_pe" in onnx_meta:
        onnx_host_pe = str(onnx_meta["host_pe"]).lower() == "true"
        if onnx_host_pe != host_pe:
            raise SystemExit(
                f"[fatal] PE mismatch — onnx host_pe={onnx_host_pe} but .cond.npz host_pe={host_pe}: "
                f"wrong .cond.npz/.onnx pair (PE would be applied "
                f"{'twice' if host_pe else 'zero times'}). Use the matching pair.")
    if host_pe:
        cond_tok = add_fractional_positions_np(cond_tok)
        zero_tok = add_fractional_positions_np(zero_tok)
    print(f"[ctrl] field={str(cz['field'])}  onset_density={args.onset_density} "
          f"(norm {(args.onset_density - float(cz['mean'])) / float(cz['std']):+.2f})  gain={args.gain}  "
          f"host_pe={host_pe}  ||cond_tok||={np.linalg.norm(cond_tok):.1f}")

    def dit_v(cross1, mask1, glob1, t_cur, ctrl_tok):
        return dit.run(None, {
            "x": x, "t": np.full((1,), t_cur, np.float32),
            "cross_attn_cond": cross1[None], "cross_attn_cond_mask": mask1[None],
            "global_embed": glob1[None], "local_add_cond": local_add,
            "control_tokens": ctrl_tok, "gain": gain})[0]

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((1, LATENT_DIM, T)).astype(np.float32)
    sig = schedule(args.steps, T, args.alpha_min, args.alpha_max)

    print(f"[gen] T={T}  steps={args.steps}  cfg={args.cfg_scale}")
    t0 = time.time()
    for i in range(args.steps):
        t_cur, dt = float(sig[i]), float(sig[i + 1] - sig[i])
        v_cond = dit_v(c_cross, c_mask, c_glob, t_cur, cond_tok)        # cond + control
        if args.cfg_scale == 1.0:
            v = v_cond
        else:
            v_unc = dit_v(u_cross, u_mask, u_glob, t_cur, zero_tok)     # uncond + null control
            v = v_unc + args.cfg_scale * (v_cond - v_unc)
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

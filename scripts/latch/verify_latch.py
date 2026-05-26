# scripts/latch/verify_latch.py
"""Closed-loop control check for a trained SA3 LatCH head (Phase 1: rms_energy_bass).

Generates with guidance at several constant target levels, decodes, runs mir's real
rms_energy_bass extractor, and reports correlation(requested, measured).

INTEGRATION: requires a GPU, the small-music-base model, a trained head checkpoint, and the
mir extractor at /home/kim/Projects/mir/src. The conditioning preamble below mirrors
StableAudioModel.generate(); reconcile against the live model.py if SA3's API changes.
"""

import argparse
import sys

import numpy as np
import torch

from stable_audio_3 import StableAudioModel
from stable_audio_3.inference.latch_guided import sample_flow_euler_latch_guided
from stable_audio_3.inference.sampling import build_schedule
from scripts.latch.latch_model import LatCH
from scripts.latch.latch_targets import build_target


def load_head(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    head = LatCH(in_channels=256, out_channels=ckpt["out_channels"],
                 dim=256, depth=6, num_heads=8).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head, ckpt


def measured_bass_rms(audio_np, sr, n_steps=256):
    """Mean bass-band RMS (dB) of a clip via mir's real extractor.

    Requires scipy + the mir source on path. `_compute_multiband_rms_ts` expects a
    1-D mono signal, takes n_steps, and returns a dict of 4 bands; we downmix to mono
    and read the bass band.
    """
    sys.path.insert(0, "/home/kim/Projects/mir/src")
    from spectral.timeseries_features import _compute_multiband_rms_ts
    mono = audio_np.mean(axis=0) if audio_np.ndim == 2 else audio_np
    bands = _compute_multiband_rms_ts(mono, sr, n_steps)  # {band: [dB per step]}
    return float(np.mean(bands["rms_energy_bass_ts"]))


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp32: TFG backprops through the model/head; fp16 grad dtypes clash and add noise.
    model = StableAudioModel.from_pretrained(args.model, device=device, model_half=False)
    head, ckpt = load_head(args.ckpt, device)
    sr = model.model.sample_rate

    cond, _ = StableAudioModel._build_conditioning_dicts(
        args.prompt, None, args.duration, 1)
    audio_sample_size = model._adapt_sample_size(cond, 5292032, 6.0)
    ds_ratio = model.model.pretransform.downsampling_ratio
    latent_len = audio_sample_size // ds_ratio
    cond_tensors = model.model.conditioner(cond, device)
    cond_tensors["inpaint_mask"] = [torch.zeros(1, 1, latent_len, device=device)]
    cond_tensors["inpaint_masked_input"] = [
        torch.zeros(1, model.model.io_channels, latent_len, device=device)]
    cond_inputs = model.model.get_conditioning_inputs(cond_tensors)
    model_dtype = next(model.model.model.parameters()).dtype
    cond_inputs = {k: (v.type(model_dtype) if v is not None else v)
                   for k, v in cond_inputs.items()}

    requested, measured = [], []
    for level in args.levels:
        torch.manual_seed(args.seed)
        noise = torch.randn(1, model.model.io_channels, latent_len,
                            device=device).type(model_dtype)
        sigmas = build_schedule(steps=args.steps, sigma_max=1.0,
                                dist_shift=model.model.sampling_dist_shift,
                                fallback_seq_len=latent_len, include_endpoint=True,
                                device=device)
        # (1, T) -> (1, 1, T) to match head output (B, out_channels, T)
        target = torch.from_numpy(
            build_target("constant", level, latent_len, 1)).unsqueeze(0).to(device).type(model_dtype)
        latents = sample_flow_euler_latch_guided(
            model.model.model, noise, sigmas, head=head, target=target,
            rho=args.gain, mu=args.gain, gamma=0.3, n_iter=6,
            window=(args.win_lo, args.win_hi), loss_type=ckpt["loss_type"],
            cfg_scale=args.cfg_scale, batch_cfg=True, rescale_cfg=True, apg_scale=1.0,
            **cond_inputs)
        with torch.no_grad():
            audio = model.model.pretransform.decode(
                latents.type(next(model.model.pretransform.parameters()).dtype))
        audio_np = audio.squeeze(0).float().cpu().numpy()
        m = measured_bass_rms(audio_np, sr)
        requested.append(level)
        measured.append(m)
        print(f"requested {level:>6.1f} dB  ->  measured {m:>7.2f} dB")

    corr = float(np.corrcoef(requested, measured)[0, 1])
    monotonic = all(x < y for x, y in zip(measured, measured[1:]))
    print(f"\ncorrelation(requested, measured) = {corr:.3f}   monotonic = {monotonic}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="small-music-base",
                   help="SA3 base model to guide (flow-matching/Euler). same-s latent space.")
    p.add_argument("--levels", type=float, nargs="+", default=[-50, -30, -10])
    p.add_argument("--prompt", default="124BPM acid techno, dry snappy kick and bassline")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gain", type=float, default=8.0)
    p.add_argument("--win-lo", type=float, default=0.4)
    p.add_argument("--win-hi", type=float, default=1.0)
    main(p.parse_args())

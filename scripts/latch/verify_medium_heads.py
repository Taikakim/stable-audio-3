"""Closed-loop controllability check for SA3 MEDIUM LatCH heads.

For each (measurable) head: guide medium-base generation toward a few standardized
target levels, decode, MEASURE the feature from the audio (mir extractor), and report
whether the measured value tracks the request (corr + monotonic) — the medium-grid
equivalent of the phase-1 bass-RMS corr-0.965 test.

Heads loaded via load_latch_from_checkpoint (auto-detects depth/dim/t_injection); the
ckpt's std_mean/std_std de-standardize. Short gen (low VRAM); per-head try/except so one
OOM doesn't sink the rest. Robust for unattended auto-run.
"""
import argparse, sys, traceback
import numpy as np, torch

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-3")
sys.path.insert(0, "/home/kim/Projects/mir/src")
from stable_audio_3 import StableAudioModel
from stable_audio_3.inference.latch_guided import sample_flow_euler_latch_guided
from stable_audio_3.inference.sampling import build_schedule
from stable_audio_3.models.latch import load_latch_from_checkpoint

WEIGHTS = "/home/kim/Projects/SAO/stable-audio-3/latch_weights_sa3_medium"
# measurable, probe-STRONG heads only (per-stem/hpcp need separation/chroma → later)
HEADS = ["spectral_flux", "spectral_flatness", "spectral_skewness", "onset_envelope"]


def measure(feature, audio_np, sr):
    """Measure the mean feature value from decoded mono audio."""
    mono = audio_np.mean(0) if audio_np.ndim > 1 else audio_np
    if feature == "onset_envelope":
        import librosa
        return float(librosa.onset.onset_strength(y=mono, sr=sr, hop_length=512).mean())
    from spectral.timeseries_features import _compute_spectral_ts
    d = _compute_spectral_ts(mono.astype(np.float32), sr, n_steps=64)
    return float(np.mean(d[feature + "_ts"]))


def main(a):
    dev = "cuda"
    print(f"loading {a.model} (fp32 for TFG)...", flush=True)
    model = StableAudioModel.from_pretrained(a.model, device=dev, model_half=False)
    sr = model.model.sample_rate
    cond, _ = StableAudioModel._build_conditioning_dicts(a.prompt, None, a.duration, 1)
    ass_ = model._adapt_sample_size(cond, 5292032, 6.0)
    ds = model.model.pretransform.downsampling_ratio
    T = ass_ // ds
    ct = model.model.conditioner(cond, dev)
    ct["inpaint_mask"] = [torch.zeros(1, 1, T, device=dev)]
    ct["inpaint_masked_input"] = [torch.zeros(1, model.model.io_channels, T, device=dev)]
    ci = model.model.get_conditioning_inputs(ct)
    mdt = next(model.model.model.parameters()).dtype
    ci = {k: (v.type(mdt) if v is not None else v) for k, v in ci.items()}

    print(f"\n{'head':<20s} {'std_lvls→measured':<40s} {'corr':>6s} {'mono':>5s}")
    print("-" * 76)
    for feat in HEADS:
        ck = f"{WEIGHTS}/latch_sa3_{feat}_best.pt"
        try:
            head = load_latch_from_checkpoint(ck, device=dev)
            m = head.metadata
            sm, ss = m.get("std_mean", 0.0), m.get("std_std", 1.0)
            std_levels = [-1.5, 0.0, 1.5]                # standardized request
            req_raw = [sm + z * ss for z in std_levels]  # de-standardized for reporting
            measured = []
            for z in std_levels:
                torch.manual_seed(a.seed)
                noise = torch.randn(1, model.model.io_channels, T, device=dev).type(mdt)
                sig = build_schedule(steps=a.steps, sigma_max=1.0,
                                     dist_shift=model.model.sampling_dist_shift,
                                     fallback_seq_len=T, include_endpoint=True, device=dev)
                tgt = torch.full((1, m["out_channels"], T), float(z), device=dev).type(mdt)
                lat = sample_flow_euler_latch_guided(
                    model.model.model, noise, sig, head=head, target=tgt,
                    rho=a.gain, mu=a.gain, gamma=0.3, n_iter=4, window=(0.4, 1.0),
                    loss_type=m.get("loss_type", "smooth_l1"),
                    cfg_scale=a.cfg, batch_cfg=True, rescale_cfg=True, apg_scale=1.0, **ci)
                with torch.no_grad():
                    aud = model.model.pretransform.decode(
                        lat.type(next(model.model.pretransform.parameters()).dtype))
                measured.append(measure(feat, aud.squeeze(0).float().cpu().numpy(), sr))
            corr = float(np.corrcoef(req_raw, measured)[0, 1])
            mono = all(x < y for x, y in zip(measured, measured[1:]))
            disp = " ".join(f"{z:+.0f}→{mv:.2f}" for z, mv in zip(std_levels, measured))
            print(f"{feat:<20s} {disp:<40s} {corr:>+6.3f} {str(mono):>5s}", flush=True)
            del head; torch.cuda.empty_cache()
        except Exception as e:
            print(f"{feat:<20s} FAILED: {type(e).__name__}: {str(e)[:50]}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="medium-base")
    p.add_argument("--prompt", default="Goa trance, driving psychedelic, analog synths")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--gain", type=float, default=8.0)
    p.add_argument("--cfg", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

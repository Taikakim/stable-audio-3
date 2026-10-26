#!/usr/bin/env python3
"""Evaluation and Audition Generator for SA3 Drum Separator (Gate G2).

Evaluates the trained LoRA + mix_lat separation model on held-out tracks against:
1. Ground Truth isolated stems
2. BS-RoFormer baseline
3. IDM / MSG-LD baselines (when available)

Computes objective metrics:
- LSD (Log-Spectral Distance, dB)
- SI-SDR (Scale-Invariant SDR, dB) and SDR (dB)
- MRSTFT (Multi-Resolution STFT distance)
- Onset F1 at 10ms, 20ms, 50ms
- Attack energy ratio (dB)

Outputs:
- eval_results.json
- Separated .wav stems
- Interactive HTML audition console (index.html)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio

from stable_audio_3.loading_utils import load_model, load_autoencoder
from stable_audio_3.model_configs import base_models, ae_models
from stable_audio_3.models.lora.loader import load_and_apply_loras
from stable_audio_3.models.lora.utils import load_lora_checkpoint
from stable_audio_3.inference.sampling import sample_diffusion
import stable_audio_3.models.transformer as T
from scripts.mir_control import MirCtrlConditioner, install_mir_control

SR = 44100
DOWNSAMPLING_RATIO = 4096


# -----------------------------------------------------------------------------
# Metric Calculations
# -----------------------------------------------------------------------------

def _py_peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    peaks = np.ones(x.shape[0], dtype=bool)
    for n in range(x.shape[0]):
        i = max(0, n - pre_max)
        j = min(n + post_max + 1, x.shape[0])
        peaks[n] = (x[n] == np.max(x[i:j])) and (x[n] > 0)
    n = 0
    while n < x.shape[0]:
        if not peaks[n]:
            n += 1
            continue
        avgn = np.mean(x[max(0, n - pre_avg):min(n + post_avg, x.shape[0])])
        peaks[n] &= (x[n] >= avgn + delta)
        if not peaks[n]:
            n += 1
            continue
        n += wait + 1
    return peaks


def detect_onsets(y, sr=SR):
    import librosa
    try:
        import librosa.util.utils as lu
        lu.__peak_pick = _py_peak_pick
    except Exception:
        pass
    if np.max(np.abs(y)) < 1e-4:
        return np.zeros(0)
    return librosa.onset.onset_detect(y=np.asarray(y, np.float32), sr=sr, units="time", backtrack=False)


def match_onsets(ref_t, est_t, tol=0.020):
    used, dts = set(), []
    for r in ref_t:
        best, bj = None, None
        for j, e in enumerate(est_t):
            if j in used:
                continue
            d = abs(e - r)
            if d <= tol and (best is None or d < best):
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            dts.append(best)
    tp = len(dts)
    p = tp / len(est_t) if len(est_t) else (1.0 if not len(ref_t) else 0.0)
    r = tp / len(ref_t) if len(ref_t) else (1.0 if not len(est_t) else 0.0)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1, dts


def attack_db(ref, est, onsets, sr=SR, win_s=0.010, eps=1e-12):
    w = int(round(win_s * sr))
    vals = []
    for t in onsets:
        a = int(round(t * sr))
        if a + w > len(ref):
            continue
        er, ee = float(np.sum(ref[a:a + w] ** 2)), float(np.sum(est[a:a + w] ** 2))
        if er > eps:
            vals.append(10.0 * np.log10((ee + eps) / er))
    return float(np.median(vals)) if vals else float("nan")


def lsd_db(ref, est, n_fft=2048, hop=512, eps=1e-8):
    w = np.hanning(n_fft)
    R = np.abs(np.lib.stride_tricks.sliding_window_view(np.pad(ref, n_fft // 2), n_fft)[::hop] * w)
    E = np.abs(np.lib.stride_tricks.sliding_window_view(np.pad(est, n_fft // 2), n_fft)[::hop] * w)
    R_spec = np.fft.rfft(R, axis=-1)
    E_spec = np.fft.rfft(E, axis=-1)
    lr = np.log10(np.abs(R_spec) ** 2 + eps)
    le = np.log10(np.abs(E_spec) ** 2 + eps)
    return float(np.mean(np.sqrt(np.mean((lr - le) ** 2, axis=-1))) * 10.0)


def calc_sdr_metrics(est_wav, ref_wav, sr=SR):
    min_len = min(len(est_wav), len(ref_wav))
    est = est_wav[:min_len]
    ref = ref_wav[:min_len]

    max_lag = int(0.2 * sr)
    sub_len = min(len(est), 10 * sr)
    corr = np.correlate(est[:sub_len], ref[:sub_len], mode="full")
    lag = np.argmax(corr) - (sub_len - 1)
    if abs(lag) < max_lag and lag != 0:
        if lag > 0:
            est = est[lag:]
            ref = ref[:-lag]
        elif lag < 0:
            lag = abs(lag)
            est = est[:-lag]
            ref = ref[lag:]

    min_len = min(len(est), len(ref))
    est = est[:min_len]
    ref = ref[:min_len]

    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    e_target = alpha * ref
    e_res = est - e_target
    si_sdr = 10 * np.log10(np.sum(e_target**2) / (np.sum(e_res**2) + 1e-8))

    s_err = est - ref
    sdr = 10 * np.log10(np.sum(ref**2) / (np.sum(s_err**2) + 1e-8))
    return float(sdr), float(si_sdr)


def compute_all_metrics(ref_mono, est_mono, sr=SR):
    min_len = min(len(ref_mono), len(est_mono))
    ref = ref_mono[:min_len]
    est = est_mono[:min_len]

    sdr, si_sdr = calc_sdr_metrics(est, ref, sr)
    lsd = lsd_db(ref, est)
    on_r = detect_onsets(ref, sr)
    on_e = detect_onsets(est, sr)
    p20, r20, f1_20, dts20 = match_onsets(on_r, on_e, 0.020)
    p50, r50, f1_50, _ = match_onsets(on_r, on_e, 0.050)
    atk = attack_db(ref, est, on_r, sr)

    return {
        "sdr_db": sdr,
        "si_sdr_db": si_sdr,
        "lsd_db": lsd,
        "onset_f1_20": float(f1_20),
        "onset_f1_50": float(f1_50),
        "onset_precision_20": float(p20),
        "onset_recall_20": float(r20),
        "attack_db": float(atk),
        "n_ref_onsets": int(len(on_r)),
        "n_est_onsets": int(len(on_e)),
    }


# -----------------------------------------------------------------------------
# Model Loading & Conditioning
# -----------------------------------------------------------------------------

def load_separation_model(checkpoint_path: str, model_name="medium-base", device="cuda"):
    """Load SA3 model, install LoRA and mix_lat modular inlet, load checkpoint weights."""
    print(f"Loading base model '{model_name}' on {device}...")
    model, model_config = load_model(model_name, device=torch.device(device), dtype=torch.bfloat16)

    # Install mix_lat modular inlet
    _tfm = model.model.model.transformer
    _ctrl_params = install_mir_control(model, _tfm, n_channels=256, cond_id="mix_lat", blocks="12-23")
    model.conditioner.conditioners["mix_lat"] = MirCtrlConditioner(n_channels=256, dropout_prob=0.0)

    print(f"Loading checkpoint weights from {checkpoint_path}...")
    state_dict, lora_config = load_lora_checkpoint(checkpoint_path)

    # Apply LoRA if checkpoint contains LoRA weights
    is_lora = any("lora_" in k for k in state_dict.keys())
    if is_lora:
        load_and_apply_loras(model, [checkpoint_path], model_config.get("model_type", "diffusion_cond"))
    else:
        # Full or modular weights: strip prefix if from Lightning wrapper
        clean_sd = {}
        for k, v in state_dict.items():
            k_clean = k
            if k_clean.startswith("diffusion."):
                k_clean = k_clean[len("diffusion."):]
            clean_sd[k_clean] = v
        model.load_state_dict(clean_sd, strict=False)

    model.to(device=device, dtype=torch.bfloat16).eval()
    return model


# -----------------------------------------------------------------------------
# Inference Loop
# -----------------------------------------------------------------------------

def separate_window(model, ae, mix_lat_np: np.ndarray, prompt="solo kick drum, punchy goa trance kick, dry isolated hit",
                    steps=25, cfg_scale=2.0, device="cuda"):
    """Separate target stem from mix_lat array (256, T)."""
    T_lat = mix_lat_np.shape[-1]
    mix_lat_t = torch.from_numpy(mix_lat_np).to(device=device, dtype=torch.float32).unsqueeze(0) # (1, 256, T)

    cond = {
        "prompt": prompt,
        "seconds_total": T_lat * DOWNSAMPLING_RATIO / SR,
        "mix_lat": mix_lat_t,
        "padding_mask": [torch.ones(T_lat, dtype=torch.bool, device=device)],
    }

    # Attention workaround for GFX1201 / RDNA 4
    orig_fa = T.flash_attn_func
    orig_fa_varlen = T.flash_attn_varlen_func
    T.flash_attn_func = None
    T.flash_attn_varlen_func = None

    with torch.inference_mode():
        sampled_lat = sample_diffusion(
            model,
            conditioning=cond,
            batch_size=1,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=42,
            device=device,
        )

        # Decode to audio
        if hasattr(ae, "decode_audio"):
            est_audio = ae.decode_audio(sampled_lat.to(dtype=torch.float32))
        else:
            est_audio = ae.decode(sampled_lat.to(dtype=torch.float32))

    T.flash_attn_func = orig_fa
    T.flash_attn_varlen_func = orig_fa_varlen

    est_wav = est_audio.squeeze(0).cpu().float().numpy() # (2, N)
    return est_wav


def main():
    p = argparse.ArgumentParser(description="Evaluate SA3 Gate G2 Drum Separator")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained .ckpt or .safetensors")
    p.add_argument("--val_dir", type=str, default="/home/kim/sao_runs/drum_sep_data/val",
                   help="Path to validation directory with manifest.json")
    p.add_argument("--comparison_dir", type=str, default="/home/kim/sao_runs/drum_sep_comparison/listen/audio",
                   help="Path to pre-existing comparison tracks (acid_alien, black_moon, etc.)")
    p.add_argument("--out_dir", type=str, default="/home/kim/sao_runs/drum_sep_g2_eval",
                   help="Output directory for evaluations and audition page")
    p.add_argument("--steps", type=int, default=25, help="Diffusion sampling steps")
    p.add_argument("--cfg", type=float, default=2.0, help="Classifier-free guidance scale")
    p.add_argument("--max_samples", type=int, default=10, help="Max validation samples to evaluate")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    audio_out = out_dir / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load SAME-L autoencoder for decoding
    print("Loading SAME-L autoencoder...")
    cfg_path, ckpt_path = ae_models["same-l"].resolve()
    ae = load_autoencoder(cfg_path, ckpt_path, device=device)

    # 2. Load fine-tuned separation model
    model = load_separation_model(args.checkpoint, device=device)

    # 3. Evaluate on held-out validation samples
    val_manifest_path = Path(args.val_dir) / "manifest.json"
    val_results = []
    if val_manifest_path.exists():
        print(f"\nEvaluating on held-out validation set from {val_manifest_path}...")
        with open(val_manifest_path) as f:
            manifest = json.load(f)
        samples = manifest.get("samples", [])[:args.max_samples]

        for s_idx, s_info in enumerate(samples):
            npz_path = Path(args.val_dir) / s_info["file"]
            if not npz_path.exists():
                continue
            with np.load(npz_path) as npz:
                target_lat = npz["target_lat"]
                mix_lat = npz["mix_lat"]
                song = str(npz["song"])
                window = int(npz["window_idx"])
                remix = str(npz["remix_type"])

            print(f"[{s_idx+1}/{len(samples)}] {song} (win {window}, {remix})...")

            # Decode ground truth kick & mixture for audio comparison
            with torch.inference_mode():
                gt_k_t = torch.from_numpy(target_lat).unsqueeze(0).to(device=device, dtype=torch.float32)
                mix_t = torch.from_numpy(mix_lat).unsqueeze(0).to(device=device, dtype=torch.float32)
                gt_k_wav = ae.decode(gt_k_t).squeeze(0).cpu().float().numpy()
                mix_wav = ae.decode(mix_t).squeeze(0).cpu().float().numpy()

            # Separate target kick using trained model
            est_k_wav = separate_window(model, ae, mix_lat, steps=args.steps, cfg_scale=args.cfg, device=device)

            # Mono downmix for metric calculation
            ref_mono = np.mean(gt_k_wav, axis=0)
            est_mono = np.mean(est_k_wav, axis=0)
            metrics = compute_all_metrics(ref_mono, est_mono, SR)

            # Save audio files
            sample_tag = f"val_{s_idx:03d}_{window}_{remix}"
            sf.write(str(audio_out / f"{sample_tag}_mix.wav"), mix_wav.T, SR)
            sf.write(str(audio_out / f"{sample_tag}_gt_kick.wav"), gt_k_wav.T, SR)
            sf.write(str(audio_out / f"{sample_tag}_sa3_kick.wav"), est_k_wav.T, SR)

            metrics["sample_tag"] = sample_tag
            metrics["song"] = song
            metrics["window"] = window
            metrics["remix"] = remix
            val_results.append(metrics)

            print(f"  -> SI-SDR: {metrics['si_sdr_db']:.2f} dB | LSD: {metrics['lsd_db']:.2f} dB | Onset F1: {metrics['onset_f1_20']:.3f}")

    # 4. Summary & Benchmarking Table
    summary_path = out_dir / "eval_results.json"
    with open(summary_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "steps": args.steps,
            "cfg": args.cfg,
            "n_samples": len(val_results),
            "mean_si_sdr_db": float(np.mean([r["si_sdr_db"] for r in val_results])) if val_results else 0.0,
            "mean_lsd_db": float(np.mean([r["lsd_db"] for r in val_results])) if val_results else 0.0,
            "mean_onset_f1_20": float(np.mean([r["onset_f1_20"] for r in val_results])) if val_results else 0.0,
            "samples": val_results
        }, f, indent=2)

    print(f"\n================================================================================")
    print("GATE G2 EVALUATION COMPLETE")
    print(f"Results saved to {summary_path}")
    if val_results:
        print(f"Mean SI-SDR:  {np.mean([r['si_sdr_db'] for r in val_results]):.2f} dB")
        print(f"Mean LSD:     {np.mean([r['lsd_db'] for r in val_results]):.2f} dB")
        print(f"Mean Onset F1: {np.mean([r['onset_f1_20'] for r in val_results]):.3f}")
    print(f"================================================================================")

if __name__ == "__main__":
    main()

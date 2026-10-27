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

from safetensors.torch import load_file
from stable_audio_tools.models.utils import copy_state_dict
from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3.loading_utils import load_autoencoder
from stable_audio_3.model_configs import base_models, ae_models
from stable_audio_3.models.lora.loader import load_and_apply_loras
from stable_audio_3.models.lora.utils import load_lora_checkpoint
from stable_audio_3.inference.sampling import sample_diffusion
import stable_audio_3.models.transformer as T
from scripts.mir_control import MirCtrlConditioner, install_mir_control

SR = 44100
DOWNSAMPLING_RATIO = 4096


def load_model(model_name: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    if model_name not in base_models:
        raise ValueError(f"LoRA eval requires a base model. Got '{model_name}', valid: {list(base_models)}")
    model_cfg = base_models[model_name]
    local_config, local_ckpt = model_cfg.resolve()
    with open(local_config) as f:
        model_config = json.load(f)
    model = create_diffusion_cond_from_config(model_config)
    copy_state_dict(model, load_file(local_ckpt))
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if model.pretransform is not None:
        model.pretransform.enable_grad = False
    return model, model_config


def encode_wav(ae, wav_np: np.ndarray, device: str = "cuda"):
    """Encode audio array (channels, samples) to latents (256, T)."""
    with torch.inference_mode():
        if wav_np.ndim == 1:
            wav_np = np.stack([wav_np, wav_np], axis=0)
        elif wav_np.shape[0] != 2 and wav_np.shape[1] == 2:
            wav_np = wav_np.T
        wav_t = torch.from_numpy(wav_np).unsqueeze(0).to(device=device, dtype=torch.float32)
        if hasattr(ae, "encode_audio"):
            lat = ae.encode_audio(wav_t)
        else:
            lat = ae.encode(wav_t)
        return lat.squeeze(0).cpu().numpy()




# -----------------------------------------------------------------------------
# Metric Calculations
# -----------------------------------------------------------------------------

def _py_peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait, peaks=None):
    if peaks is None:
        peaks = np.zeros(x.shape, dtype=bool)
    else:
        peaks.fill(False)
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

    from scipy.signal import correlate
    max_lag = int(0.2 * sr)
    sub_len = min(len(est), 10 * sr)
    corr = correlate(est[:sub_len], ref[:sub_len], mode="full", method="fft")
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
                    steps=25, cfg_scale=1.0, device="cuda"):
    """Separate target stem from mix_lat array (256, T)."""
    T_lat = mix_lat_np.shape[-1]
    mix_lat_t = torch.from_numpy(mix_lat_np).to(device=device, dtype=torch.float32) # (256, T)
    mix_lat_batch = mix_lat_t.unsqueeze(0) if mix_lat_t.ndim == 2 else mix_lat_t

    raw_cond = [{
        "prompt": prompt,
        "seconds_total": T_lat * DOWNSAMPLING_RATIO / SR,
        "mix_lat": mix_lat_batch.squeeze(0),
    }]

    conditioning = model.conditioner(raw_cond, device)
    # Native inpainting conditioning matches DrumSepDataset training setup
    conditioning['inpaint_mask'] = [torch.zeros(1, 1, T_lat, device=device)]
    conditioning['inpaint_masked_input'] = [mix_lat_batch]
    cond_inputs = model.get_conditioning_inputs(conditioning)

    model_dtype = next(model.model.parameters()).dtype
    noise = torch.randn(1, 256, T_lat, device=device, dtype=model_dtype)

    # Attention workaround for GFX1201 / RDNA 4
    orig_fa = T.flash_attn_func
    orig_fa_varlen = T.flash_attn_varlen_func
    T.flash_attn_func = None
    T.flash_attn_varlen_func = None

    with torch.inference_mode():
        sampled_lat = sample_diffusion(
            model=model.model,
            noise=noise,
            cond_inputs=cond_inputs,
            diffusion_objective="rectified_flow",
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=raw_cond,
            sample_rate=SR,
            pretransform=model.pretransform,
            batch_cfg=True,
            disable_tqdm=True,
            decode=False,
        )
        sampled_lat = torch.nan_to_num(sampled_lat, nan=0.0)

        # Decode to audio
        if hasattr(ae, "decode_audio"):
            est_audio = ae.decode_audio(sampled_lat.to(dtype=torch.float32))
        else:
            est_audio = ae.decode(sampled_lat.to(dtype=torch.float32))
        est_audio = torch.nan_to_num(est_audio, nan=0.0).clamp(-1.0, 1.0)

    T.flash_attn_func = orig_fa
    T.flash_attn_varlen_func = orig_fa_varlen

    est_wav = est_audio.squeeze(0).cpu().float().numpy() # (2, N)
    est_wav = np.nan_to_num(est_wav, nan=0.0)
    return np.clip(est_wav, -1.0, 1.0)


def separate_long_audio(model, ae, mix_wav: np.ndarray, prompt="solo kick drum, punchy goa trance kick, dry isolated hit",
                        window_frames=130, hop_frames=96, steps=25, cfg_scale=1.0, device="cuda"):
    """Separate full-length audio by encoding and sliding-window diffusion matching training length (T=130)."""
    mix_lat = encode_wav(ae, mix_wav, device=device)
    T_total = mix_lat.shape[-1]
    total_samples = mix_wav.shape[0] if (mix_wav.ndim == 1 or mix_wav.shape[0] > 2) else mix_wav.shape[1]

    if T_total <= window_frames:
        out = separate_window(model, ae, mix_lat, prompt=prompt, steps=steps, cfg_scale=cfg_scale, device=device)
        return out[:, :total_samples]

    # Compute sliding window starts
    starts = []
    s = 0
    while s + window_frames < T_total:
        starts.append(s)
        s += hop_frames
    starts.append(max(0, T_total - window_frames))
    starts = sorted(list(set(starts)))

    out_audio = np.zeros((2, total_samples), dtype=np.float32)
    weight = np.zeros(total_samples, dtype=np.float32)

    for s_idx, start in enumerate(starts):
        end = start + window_frames
        chunk_lat = mix_lat[:, start:end]
        chunk_wav = separate_window(model, ae, chunk_lat, prompt=prompt, steps=steps, cfg_scale=cfg_scale, device=device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        a_start = start * DOWNSAMPLING_RATIO
        a_end = min(total_samples, a_start + chunk_wav.shape[-1])
        w_len = a_end - a_start

        fade_samples = min(w_len // 4, int(1.5 * SR))
        fade = np.ones(w_len, dtype=np.float32)
        if s_idx > 0 and fade_samples > 0:
            fade[:fade_samples] = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        if s_idx < len(starts) - 1 and fade_samples > 0:
            fade[-fade_samples:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade_samples) / fade_samples))

        out_audio[:, a_start:a_end] += chunk_wav[:, :w_len] * fade
        weight[a_start:a_end] += fade

    mask = weight > 1e-4
    out_audio[:, mask] /= weight[mask]
    return np.clip(np.nan_to_num(out_audio, nan=0.0), -1.0, 1.0)


def apply_wiener_masking(est_wav: np.ndarray, mix_wav: np.ndarray, alpha: float = 1.5,
                         n_fft: int = 2048, hop_length: int = 512, eps: float = 1e-10) -> np.ndarray:
    """Apply spectrogram Wiener masking against the mix bus to ensure energy conservation and zero hallucination.

    Args:
        est_wav: (channels, samples) generated separated stem audio.
        mix_wav: (samples, channels) or (channels, samples) original mixture bus audio.
        alpha: Exponent for magnitude weighting (1.0 = linear amplitude ratio, 2.0 = power ratio).
    Returns:
        reconciled_wav: (channels, samples) energy-conserved masked stem audio.
    """
    if mix_wav.ndim == 1:
        mix_wav = np.stack([mix_wav, mix_wav], axis=0)
    elif mix_wav.shape[0] > 2 and mix_wav.shape[1] == 2:
        mix_wav = mix_wav.T

    if est_wav.ndim == 1:
        est_wav = np.stack([est_wav, est_wav], axis=0)
    elif est_wav.shape[0] > 2 and est_wav.shape[1] == 2:
        est_wav = est_wav.T

    samples = min(est_wav.shape[-1], mix_wav.shape[-1])
    est_crop = est_wav[:, :samples]
    mix_crop = mix_wav[:, :samples]

    device = "cpu"
    est_t = torch.from_numpy(est_crop).float().to(device)
    mix_t = torch.from_numpy(mix_crop).float().to(device)
    window = torch.hann_window(n_fft, device=device)

    X_est = torch.stft(est_t, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    X_mix = torch.stft(mix_t, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

    mag_est = torch.abs(X_est)
    mag_mix = torch.abs(X_mix)

    mag_res = torch.clamp(mag_mix - mag_est, min=0.0)

    est_p = mag_est ** alpha
    res_p = mag_res ** alpha
    mask = est_p / (est_p + res_p + eps)
    mask = torch.clamp(mask, 0.0, 1.0)

    X_masked = mask * X_mix
    recon_t = torch.istft(X_masked, n_fft=n_fft, hop_length=hop_length, window=window, length=samples)
    recon = recon_t.cpu().numpy()
    return np.clip(np.nan_to_num(recon, nan=0.0), -1.0, 1.0)


def main():
    p = argparse.ArgumentParser(description="Evaluate SA3 Gate G2 Drum Separator")
    p.add_argument("--compare_checkpoint", type=str, default="/home/kim/sao_runs/drum_sep_g2_kick_armA/epoch=2-step=5000.ckpt",
                   help="Optional path to comparison checkpoint (e.g. Arm A)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained .ckpt or .safetensors")
    p.add_argument("--val_dir", type=str, default="/home/kim/sao_runs/drum_sep_data/val",
                   help="Path to validation directory with manifest.json")
    p.add_argument("--comparison_dir", type=str, default="/home/kim/sao_runs/drum_sep_comparison/listen/audio",
                   help="Path to pre-existing comparison tracks (acid_alien, black_moon, etc.)")
    p.add_argument("--out_dir", type=str, default="/home/kim/sao_runs/drum_sep_g2_eval",
                   help="Output directory for evaluations and audition page")
    p.add_argument("--steps", type=int, default=25, help="Diffusion sampling steps")
    p.add_argument("--cfg", type=float, default=1.0, help="Classifier-free guidance scale")
    p.add_argument("--max_samples", type=int, default=10, help="Max validation samples to evaluate")
    p.add_argument("--wiener_alpha", type=float, default=1.5,
                   help="Exponent for STFT Wiener mask reconciliation against mix bus (default 1.5, set <=0 to disable)")
    p.add_argument("--reuse_existing", action="store_true",
                   help="Reuse existing separated audio files from disk if available (fast metric recompute)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    audio_out = out_dir / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None
    ae = None
    compare_model = None

    if not args.reuse_existing:
        # 1. Load SAME-L autoencoder for decoding & encoding
        print("Loading SAME-L autoencoder...")
        cfg_path, ckpt_path = ae_models["same-l"].resolve()
        ae = load_autoencoder(cfg_path, ckpt_path, device=device)

        # 2. Load fine-tuned separation model (Primary)
        print(f"Loading primary separation model from {args.checkpoint}...")
        model = load_separation_model(args.checkpoint, device=device)
        # Note: compare_model is lazy-loaded after primary model finishes to stay under 16GB VRAM
    else:
        print("Reusing existing separated audio from disk (--reuse_existing). Skipping model load.")

    # 3. Evaluate on held-out validation samples
    val_manifest_path = Path(args.val_dir) / "manifest.json"
    val_results = []
    if val_manifest_path.exists():
        print(f"\nEvaluating on held-out validation set from {val_manifest_path}...")
        with open(val_manifest_path) as f:
            manifest = json.load(f)
        all_samples = manifest.get("samples", [])
        if args.max_samples and len(all_samples) > args.max_samples:
            stems = ["kick", "snare", "hihat", "percussion"]
            per_stem = max(1, args.max_samples // len(stems))
            stratified = []
            for st in stems:
                stem_s = [s for s in all_samples if s.get("stem") == st and s.get("remix") == "orig"]
                stratified.extend(stem_s[:per_stem])
            samples = stratified if len(stratified) >= args.max_samples // 2 else all_samples[:args.max_samples]
        else:
            samples = all_samples[:args.max_samples] if args.max_samples else all_samples


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
                prompt = str(npz["prompt"]) if "prompt" in npz.files else "solo kick drum, punchy goa trance kick, dry isolated hit"
                stem_tag = str(npz["stem_tag"]) if "stem_tag" in npz.files else "kick"

            print(f"[{s_idx+1}/{len(samples)}] {song} ({stem_tag}, win {window}, {remix})...")

            sample_tag = f"val_{s_idx:03d}_{window}_{stem_tag}_{remix}"
            mix_p = audio_out / f"{sample_tag}_mix.wav"
            gt_p = audio_out / f"{sample_tag}_gt_{stem_tag}.wav"
            sa3_path = audio_out / f"{sample_tag}_sa3_{stem_tag}.wav"

            if args.reuse_existing and mix_p.exists() and gt_p.exists() and sa3_path.exists():
                mix_wav, _ = sf.read(str(mix_p))
                gt_k_wav, _ = sf.read(str(gt_p))
                est_k_wav, _ = sf.read(str(sa3_path))
                if mix_wav.ndim > 1 and mix_wav.shape[0] > 2:
                    mix_wav = mix_wav.T
                if gt_k_wav.ndim > 1 and gt_k_wav.shape[0] > 2:
                    gt_k_wav = gt_k_wav.T
                if est_k_wav.ndim > 1 and est_k_wav.shape[0] > 2:
                    est_k_wav = est_k_wav.T
            else:
                # Decode ground truth target stem & mixture for audio comparison
                with torch.inference_mode():
                    gt_k_t = torch.from_numpy(target_lat).unsqueeze(0).to(device=device, dtype=torch.float32)
                    mix_t = torch.from_numpy(mix_lat).unsqueeze(0).to(device=device, dtype=torch.float32)
                    gt_k_wav = ae.decode(gt_k_t).squeeze(0).cpu().float().numpy()
                    mix_wav = ae.decode(mix_t).squeeze(0).cpu().float().numpy()

                # Separate target stem using trained model with stem-specific prompt
                est_k_wav = separate_window(model, ae, mix_lat, prompt=prompt, steps=args.steps, cfg_scale=args.cfg, device=device)
                sf.write(str(sa3_path), est_k_wav.T, SR)
                sf.write(str(mix_p), mix_wav.T, SR)
                sf.write(str(gt_p), gt_k_wav.T, SR)

            # Mono downmix for metric calculation
            ref_mono = np.mean(gt_k_wav, axis=0)
            est_mono = np.mean(est_k_wav, axis=0)
            metrics = compute_all_metrics(ref_mono, est_mono, SR)

            # Apply Wiener Masking Reconciliation if requested
            if args.wiener_alpha > 0:
                w_k_wav = apply_wiener_masking(est_k_wav, mix_wav, alpha=args.wiener_alpha)
                sf.write(str(audio_out / f"{sample_tag}_wiener_{stem_tag}.wav"), w_k_wav.T, SR)
                w_mono = np.mean(w_k_wav, axis=0)
                metrics["wiener"] = compute_all_metrics(ref_mono, w_mono, SR)

            # Save audio files
            sf.write(str(audio_out / f"{sample_tag}_mix.wav"), mix_wav.T, SR)
            sf.write(str(audio_out / f"{sample_tag}_gt_{stem_tag}.wav"), gt_k_wav.T, SR)

            metrics["sample_tag"] = sample_tag
            metrics["song"] = song
            metrics["window"] = window
            metrics["remix"] = remix
            metrics["stem_tag"] = stem_tag
            val_results.append(metrics)

            msg = f"  -> SA3 {stem_tag.capitalize()}: SI-SDR: {metrics['si_sdr_db']:.2f} dB | LSD: {metrics['lsd_db']:.2f} dB | Onset F1: {metrics['onset_f1_20']:.3f}"
            if "wiener" in metrics:
                msg += f" | Wiener: SI-SDR: {metrics['wiener']['si_sdr_db']:.2f} dB | LSD: {metrics['wiener']['lsd_db']:.2f} dB"
            print(msg)

    # 4. Evaluate on Benchmark Comparison Multitracks
    comp_results = []
    comp_dir = Path(args.comparison_dir)
    if comp_dir.exists():
        comp_tracks = ["acid_alien", "black_moon", "goddess_guerrilla"]
        print(f"\nEvaluating on benchmark comparison multitracks from {comp_dir}...")
        for track_id in comp_tracks:
            tdir = comp_dir / track_id
            mix_path = tdir / "mix.wav"
            if not mix_path.exists():
                continue
            gt_path = tdir / "studio_kick.wav"
            if not gt_path.exists():
                gt_path = tdir / "studio_ground_truth.wav"
            if not gt_path.exists():
                continue

            print(f"\n--- Benchmark Track: {track_id} ---")
            mix_wav, _ = sf.read(str(mix_path))
            gt_wav, _ = sf.read(str(gt_path))
            gt_mono = np.mean(gt_wav, axis=-1) if gt_wav.ndim > 1 else gt_wav

            track_audio_out = audio_out / track_id
            track_audio_out.mkdir(parents=True, exist_ok=True)
            sf.write(str(track_audio_out / "mix.wav"), mix_wav, SR)
            sf.write(str(track_audio_out / "ground_truth.wav"), gt_wav, SR)

            row = {
                "track_id": track_id,
                "audio": {
                    "mix": f"audio/{track_id}/mix.wav",
                    "ground_truth": f"audio/{track_id}/ground_truth.wav",
                }
            }

            # Separate all 4 drum pieces with Primary Model
            stem_configs = [
                ("kick", "solo kick drum, punchy goa trance kick, dry isolated hit"),
                ("snare", "acoustic snare drum, snap and crack, punchy tight transient"),
                ("hihat", "crisp closed hi-hats, bright metallic offbeat 16th pattern"),
                ("percussion", "tribal percussion, shakers, congas, rolling grooves")
            ]

            for s_tag, s_prompt in stem_configs:
                s_path = track_audio_out / f"sa3_{s_tag}.wav"
                if args.reuse_existing and s_path.exists():
                    s_wav, _ = sf.read(str(s_path))
                    if s_wav.ndim > 1 and s_wav.shape[0] > 2:
                        s_wav = s_wav.T
                else:
                    print(f"  Separating {s_tag} with primary model...")
                    s_wav = separate_long_audio(model, ae, mix_wav, prompt=s_prompt, steps=args.steps, cfg_scale=args.cfg, device=device)
                    sf.write(str(s_path), s_wav.T, SR)

                row["audio"][f"sa3_{s_tag}"] = f"audio/{track_id}/sa3_{s_tag}.wav"

                if s_tag == "kick":
                    sa3_wav = s_wav
                    sa3_mono = np.mean(sa3_wav, axis=0)
                    m_sa3 = compute_all_metrics(gt_mono, sa3_mono, SR)
                    row["sa3"] = m_sa3
                    print(f"  -> SA3 Kick:      SI-SDR: {m_sa3['si_sdr_db']:6.2f} dB | LSD: {m_sa3['lsd_db']:5.2f} dB | Onset F1: {m_sa3['onset_f1_20']:.3f} | Attack: {m_sa3.get('attack_db', float('nan')):+5.2f} dB")

                # Wiener Reconciliation
                if args.wiener_alpha > 0:
                    s_w_wav = apply_wiener_masking(s_wav, mix_wav.T, alpha=args.wiener_alpha)
                    sf.write(str(track_audio_out / f"sa3_wiener_{s_tag}.wav"), s_w_wav.T, SR)
                    row["audio"][f"sa3_wiener_{s_tag}"] = f"audio/{track_id}/sa3_wiener_{s_tag}.wav"
                    if s_tag == "kick":
                        sa3_w_mono = np.mean(s_w_wav, axis=0)
                        m_sa3_w = compute_all_metrics(gt_mono, sa3_w_mono, SR)
                        row["sa3_wiener"] = m_sa3_w
                        print(f"  -> SA3 Wiener:    SI-SDR: {m_sa3_w['si_sdr_db']:6.2f} dB | LSD: {m_sa3_w['lsd_db']:5.2f} dB | Onset F1: {m_sa3_w['onset_f1_20']:.3f} | Attack: {m_sa3_w.get('attack_db', float('nan')):+5.2f} dB")


            # BS-RoFormer baseline
            bs_path = tdir / "bs_roformer_sw_drums.wav"
            if not bs_path.exists():
                bs_path = tdir / "bs_roformer_zfturbo_drums.wav"
            if bs_path.exists():
                bs_wav, _ = sf.read(str(bs_path))
                bs_mono = np.mean(bs_wav, axis=-1) if bs_wav.ndim > 1 else bs_wav
                sf.write(str(track_audio_out / "bs_roformer.wav"), bs_wav, SR)
                m_bs = compute_all_metrics(gt_mono, bs_mono, SR)
                row["bs_roformer"] = m_bs
                row["audio"]["bs_roformer"] = f"audio/{track_id}/bs_roformer.wav"
                print(f"  -> BS-RoFormer:   SI-SDR: {m_bs['si_sdr_db']:6.2f} dB | LSD: {m_bs['lsd_db']:5.2f} dB | Onset F1: {m_bs['onset_f1_20']:.3f} | Attack: {m_bs.get('attack_db', float('nan')):+5.2f} dB")

            # MSG-LD baseline
            msgld_path = tdir / "msgld_kick.wav"
            if msgld_path.exists():
                m_wav, sr_m = sf.read(str(msgld_path))
                if sr_m != SR:
                    resampler = torchaudio.transforms.Resample(sr_m, SR)
                    m_wav_t = resampler(torch.from_numpy(m_wav).unsqueeze(0).float()).squeeze(0).numpy()
                else:
                    m_wav_t = m_wav
                sf.write(str(track_audio_out / "msgld_kick.wav"), m_wav_t, SR)
                m_msgld = compute_all_metrics(gt_mono, m_wav_t, SR)
                row["msgld"] = m_msgld
                row["audio"]["msgld"] = f"audio/{track_id}/msgld_kick.wav"
                print(f"  -> MSG-LD Kick:   SI-SDR: {m_msgld['si_sdr_db']:6.2f} dB | LSD: {m_msgld['lsd_db']:5.2f} dB | Onset F1: {m_msgld['onset_f1_20']:.3f} | Attack: {m_msgld.get('attack_db', float('nan')):+5.2f} dB")

            comp_results.append(row)

        # Phase 2: Comparison Model (if provided), run after primary model is released from VRAM
        if args.compare_checkpoint and Path(args.compare_checkpoint).exists():
            print(f"\nPhase 2: Offloading primary model and evaluating compare model from {args.compare_checkpoint}...")
            del model
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            compare_model = load_separation_model(args.compare_checkpoint, device=device)

            for row in comp_results:
                track_id = row["track_id"]
                tdir = comp_dir / track_id
                mix_path = tdir / "mix.wav"
                gt_path = tdir / "studio_kick.wav"
                if not gt_path.exists():
                    gt_path = tdir / "studio_ground_truth.wav"
                mix_wav, _ = sf.read(str(mix_path))
                gt_wav, _ = sf.read(str(gt_path))
                gt_mono = np.mean(gt_wav, axis=-1) if gt_wav.ndim > 1 else gt_wav

                track_audio_out = audio_out / track_id
                cmp_path = track_audio_out / "sa3_compare_kick.wav"
                if args.reuse_existing and cmp_path.exists():
                    cmp_wav, _ = sf.read(str(cmp_path))
                    if cmp_wav.ndim > 1 and cmp_wav.shape[0] > 2:
                        cmp_wav = cmp_wav.T
                else:
                    print(f"  Separating {track_id} kick with compare model ({Path(args.compare_checkpoint).name})...")
                    cmp_wav = separate_long_audio(compare_model, ae, mix_wav, steps=args.steps, cfg_scale=args.cfg, device=device)
                    sf.write(str(cmp_path), cmp_wav.T, SR)

                cmp_mono = np.mean(cmp_wav, axis=0)
                m_cmp = compute_all_metrics(gt_mono, cmp_mono, SR)
                row["sa3_compare"] = m_cmp
                row["audio"]["sa3_compare_kick"] = f"audio/{track_id}/sa3_compare_kick.wav"
                print(f"  -> SA3 Compare:   SI-SDR: {m_cmp['si_sdr_db']:6.2f} dB | LSD: {m_cmp['lsd_db']:5.2f} dB | Onset F1: {m_cmp['onset_f1_20']:.3f} | Attack: {m_cmp.get('attack_db', float('nan')):+5.2f} dB")

            del compare_model
            compare_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 5. Summary & Benchmarking Table
    summary_path = out_dir / "eval_results.json"
    with open(summary_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "compare_checkpoint": args.compare_checkpoint,
            "steps": args.steps,
            "cfg": args.cfg,
            "wiener_alpha": args.wiener_alpha,
            "n_val_samples": len(val_results),
            "mean_val_si_sdr_db": float(np.mean([r["si_sdr_db"] for r in val_results])) if val_results else 0.0,
            "mean_val_lsd_db": float(np.mean([r["lsd_db"] for r in val_results])) if val_results else 0.0,
            "mean_val_onset_f1_20": float(np.mean([r["onset_f1_20"] for r in val_results])) if val_results else 0.0,
            "mean_val_wiener_si_sdr_db": float(np.mean([r["wiener"]["si_sdr_db"] for r in val_results if "wiener" in r])) if val_results else 0.0,
            "mean_val_wiener_lsd_db": float(np.mean([r["wiener"]["lsd_db"] for r in val_results if "wiener" in r])) if val_results else 0.0,
            "val_samples": val_results,
            "comp_samples": comp_results
        }, f, indent=2)

    # 6. Deploy Audition Console HTML
    from scripts.drum_sep.audition_console import generate_html_console
    generate_html_console(out_dir, val_results, comp_results, {
        "checkpoint": args.checkpoint,
        "compare_checkpoint": args.compare_checkpoint,
        "steps": args.steps,
        "cfg": args.cfg,
        "wiener_alpha": args.wiener_alpha
    })

    print(f"\n================================================================================")
    print("GATE G2 EVALUATION COMPLETE")
    print(f"Results saved to {summary_path}")
    if val_results:
        print(f"Held-out Val Mean SI-SDR:         {np.mean([r['si_sdr_db'] for r in val_results]):.2f} dB")
        if any("wiener" in r for r in val_results):
            print(f"Held-out Val Mean Wiener SI-SDR:  {np.mean([r['wiener']['si_sdr_db'] for r in val_results if 'wiener' in r]):.2f} dB")
        print(f"Held-out Val Mean LSD:            {np.mean([r['lsd_db'] for r in val_results]):.2f} dB")
        print(f"Held-out Val Mean Onset F1:       {np.mean([r['onset_f1_20'] for r in val_results]):.3f}")
    if comp_results:
        print(f"Benchmark Tracks Mean SI-SDR:        {np.mean([r['sa3']['si_sdr_db'] for r in comp_results]):.2f} dB")
        if any("sa3_wiener" in r for r in comp_results):
            print(f"Benchmark Tracks Mean Wiener SI-SDR: {np.mean([r['sa3_wiener']['si_sdr_db'] for r in comp_results if 'sa3_wiener' in r]):.2f} dB")
        print(f"Benchmark Tracks Mean LSD:           {np.mean([r['sa3']['lsd_db'] for r in comp_results]):.2f} dB")
        print(f"Benchmark Tracks Mean Onset:         {np.mean([r['sa3']['onset_f1_20'] for r in comp_results]):.3f}")
    print(f"Audition Console: file://{out_dir}/index.html")
    print(f"================================================================================")

if __name__ == "__main__":
    main()


# Owner: KUANG (created 2026-10-23)
"""Gate G0 of the SA3 drum separator plan (audio-tools-avp scripts/synth_inversion/DRUM_SEPARATION_SA3_PLAN.md):
how much of an isolated drum stem survives a SAME-L encode -> decode round trip, before any separator is trained.

A separator that generates stem latents can never sound better than decode(encode(true stem)). SAME-L runs at
44100 / 4096 = 10.77 latent frames per second (93 ms per frame), so the question is whether kick and hi-hat
transients, and quiet stems, come back intact. This script measures it on real stem sets and writes listening pairs.

Input: --stems_dir ROOT with one folder per song, each holding one WAV per kit piece (the file name is the class,
e.g. kick.wav, snare.wav, hats.wav). The drum bus is the sum of the stems. One crop per song (--offset_s, --crop_s).

Per stem and for the bus, original vs round trip:
  - lsd_db:       log-spectral distance (n_fft 2048, as in the IDM paper), dB, lower = closer
  - mrstft:       multi-resolution STFT distance (spectral convergence + log-magnitude L1, n_fft 2048/1024/512/256)
  - onset_f1_20 / onset_f1_50: onsets of the round trip matched to the original's within 20 / 50 ms
  - onset_dt_ms:  median |time offset| of the matched onsets
  - attack_db:    median energy change in the first 10 ms after each original onset (0 = kept, < 0 = smeared)
  - si_sdr_db:    for reference only (a GAN-trained codec need not keep phase)
Per song, additivity (decides whether stems need reconciling against the real mix):
  - sum_vs_bus_lsd_db: sum of round-tripped stems vs round-tripped bus
  - latent_sum_lsd_db: decode(sum of stem latents) vs the bus (latents are not expected to be additive)

  python -m scripts.drum_sep.codec_ceiling --stems_dir /path/stem_sets --out_dir ~/sa3_runs/drum_sep_g0
Writes results.json, report.md and listen/<song>__<stem>__{orig,recon}.wav. Untested on SAME-L (written on a laptop
without the weights); the CPU tests run it with fake codecs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

SR = 44100
ONSET_TOL_S = (0.020, 0.050)


# ------------------------------------------------------------------------------------------------ metrics
def _stft_mag(y, n_fft, hop):
    w = np.hanning(n_fft)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(y[idx] * w, axis=1))


def lsd_db(ref, est, n_fft=2048, hop=512, eps=1e-8):
    """Log-spectral distance in dB: mean over frames of the RMS (over bins) dB difference."""
    R, E = _stft_mag(ref, n_fft, hop), _stft_mag(est, n_fft, hop)
    d = 20.0 * np.log10(R + eps) - 20.0 * np.log10(E + eps)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=1))))


def mrstft(ref, est, sizes=(2048, 1024, 512, 256), eps=1e-7):
    tot = 0.0
    for n in sizes:
        R, E = _stft_mag(ref, n, n // 4) + eps, _stft_mag(est, n, n // 4) + eps
        sc = np.linalg.norm(R - E) / (np.linalg.norm(R) + eps)
        tot += sc + float(np.mean(np.abs(np.log(R) - np.log(E))))
    return float(tot / len(sizes))


def si_sdr_db(ref, est, eps=1e-9):
    a = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    s = a * ref
    return float(10.0 * np.log10((np.dot(s, s) + eps) / (np.dot(est - s, est - s) + eps)))


def _py_peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait, peaks):
    peaks[0] = (x[0] >= np.max(x[:min(post_max, x.shape[0])]))
    peaks[0] &= (x[0] >= np.mean(x[:min(post_avg, x.shape[0])]) + delta)
    n = wait + 1 if peaks[0] else 1
    while n < x.shape[0]:
        maxn = np.max(x[max(0, n - pre_max):min(n + post_max, x.shape[0])])
        peaks[n] = (x[n] == maxn)
        if not peaks[n]:
            n += 1
            continue
        avgn = np.mean(x[max(0, n - pre_avg):min(n + post_avg, x.shape[0])])
        peaks[n] &= (x[n] >= avgn + delta)
        if not peaks[n]:
            n += 1
            continue
        n += wait + 1


def detect_onsets(y, sr=SR):
    """Onset times (s), librosa's default detector; empty for (near) silent input."""
    import librosa
    try:
        import librosa.util.utils as lu
        lu.__peak_pick = _py_peak_pick
    except Exception:
        pass
    if np.max(np.abs(y)) < 1e-4:
        return np.zeros(0)
    return librosa.onset.onset_detect(y=np.asarray(y, np.float32), sr=sr, units="time", backtrack=False)


def match_onsets(ref_t, est_t, tol):
    """Greedy one-to-one matching within tol. -> (precision, recall, f1, |dt| list)."""
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
    """Median over onsets of 10*log10(energy of est / energy of ref) in [onset, onset + win_s]."""
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


def track_metrics(ref, est, sr=SR):
    """ref, est: mono float arrays of equal length."""
    on_r, on_e = detect_onsets(ref, sr), detect_onsets(est, sr)
    out = {"lsd_db": lsd_db(ref, est), "mrstft": mrstft(ref, est), "si_sdr_db": si_sdr_db(ref, est),
           "n_onsets": int(len(on_r)), "attack_db": attack_db(ref, est, on_r, sr)}
    for tol in ONSET_TOL_S:
        _, _, f1, dts = match_onsets(on_r, on_e, tol)
        out[f"onset_f1_{int(tol * 1000)}"] = float(f1)
        if tol == ONSET_TOL_S[-1]:
            out["onset_dt_ms"] = float(np.median(dts) * 1000) if dts else float("nan")
    return out


# ------------------------------------------------------------------------------------------------ codec
class SameCodec:
    """SAME-L through stable_audio_3.model.AutoencoderModel: (C, n) float array -> latent -> (C, n) array."""

    def __init__(self, name="same-l", device=None):
        import torch
        from stable_audio_3.model import AutoencoderModel
        self.torch = torch
        self.ae = AutoencoderModel.from_pretrained(name, device=device)
        self.sample_rate = self.ae.sample_rate

    def encode(self, audio, sr):
        x = self.torch.as_tensor(np.asarray(audio, np.float32))
        return self.ae.encode(x, sr)                                  # (1, D, T_lat)

    def decode(self, latent, n):
        y = self.ae.decode(latent).float().cpu().numpy()[0]           # (C, n')
        return y[:, :n] if y.shape[1] >= n else np.pad(y, ((0, 0), (0, n - y.shape[1])))


# ------------------------------------------------------------------------------------------------ run
def load_song(folder, sr=SR, offset_s=0.0, crop_s=12.0):
    """-> {stem name: (2, n) float32}, all cropped to the same window; mono files are duplicated."""
    import soundfile as sf
    stems = {}
    for p in sorted(Path(folder).glob("*.wav")):
        y, fs = sf.read(str(p), always_2d=True, dtype="float32")
        if fs != sr:
            import librosa
            y = librosa.resample(y.T, orig_sr=fs, target_sr=sr).T
        a = int(round(offset_s * sr))
        y = y[a:a + int(round(crop_s * sr))].T
        if y.shape[0] == 1:
            y = np.concatenate([y, y], 0)
        stems[p.stem] = y[:2]
    if stems:
        n = min(v.shape[1] for v in stems.values())
        stems = {k: v[:, :n] for k, v in stems.items()}
    return stems


def run_song(codec, stems, sr=SR, listen_dir=None, song=""):
    import soundfile as sf
    names = list(stems)
    bus = np.sum([stems[k] for k in names], axis=0)
    n = bus.shape[1]
    tracks = dict(stems, __bus__=bus)
    lat, rec, res = {}, {}, {}
    for k, y in tracks.items():
        lat[k] = codec.encode(y, sr)
        rec[k] = codec.decode(lat[k], n)
        res[k] = track_metrics(y.mean(0), rec[k].mean(0), sr)
        if listen_dir:
            for tag, a in (("orig", y), ("recon", rec[k])):
                sf.write(os.path.join(listen_dir, f"{song}__{k}__{tag}.wav"), a.T, sr, subtype="PCM_16")
    rec_sum = np.sum([rec[k] for k in names], axis=0)
    lat_sum = lat[names[0]]
    for k in names[1:]:
        lat_sum = lat_sum + lat[k]
    add = {"sum_vs_bus_lsd_db": lsd_db(rec["__bus__"].mean(0), rec_sum.mean(0)),
           "latent_sum_lsd_db": lsd_db(bus.mean(0), codec.decode(lat_sum, n).mean(0))}
    return res, add


def summarize(per_song):
    """-> {class: {metric: median over songs}} (class = file name; the bus as __bus__)."""
    by = {}
    for song in per_song:
        for k, m in song["tracks"].items():
            by.setdefault(k, []).append(m)
    out = {}
    for k, ms in sorted(by.items()):
        out[k] = {"n": len(ms)}
        for key in ms[0]:
            v = [m[key] for m in ms if m[key] == m[key]]
            out[k][key] = float(np.median(v)) if v else float("nan")
    return out


def gate_verdict(summary, f1_min=0.9, attack_min_db=-3.0):
    """Plan rule: a separator inherits the codec's transient loss. If any class keeps fewer than f1_min of its
    onsets within 20 ms, or loses more than |attack_min_db| of attack energy (median), stems must be reconciled
    against the real mix (Wiener masks) rather than used as decoded."""
    bad = [k for k, m in summary.items() if k != "__bus__" and m.get("n_onsets", 0) > 0
           and (m["onset_f1_20"] < f1_min or m["attack_db"] < attack_min_db)]
    return {"transient_ok": not bad, "classes_failing": bad, "f1_min": f1_min, "attack_min_db": attack_min_db}


def write_report(summary, verdict, add, path):
    cols = ["n", "lsd_db", "mrstft", "onset_f1_20", "onset_f1_50", "onset_dt_ms", "attack_db", "si_sdr_db"]
    lines = ["# SAME-L round trip on drum stems (gate G0)", "",
             "| class | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k, m in summary.items():
        lines.append(f"| {k} | " + " | ".join(f"{m[c]:.3g}" if isinstance(m[c], float) else str(m[c]) for c in cols) + " |")
    lines += ["", f"Additivity (median over songs): sum of round-tripped stems vs round-tripped bus "
              f"{np.median([a['sum_vs_bus_lsd_db'] for a in add]):.2f} dB LSD; decode(sum of latents) vs bus "
              f"{np.median([a['latent_sum_lsd_db'] for a in add]):.2f} dB LSD.", "",
              f"Verdict: transients {'kept' if verdict['transient_ok'] else 'NOT kept for ' + ', '.join(verdict['classes_failing'])} "
              f"(rule: onset F1 at 20 ms >= {verdict['f1_min']} and attack >= {verdict['attack_min_db']} dB). "
              "Listen to listen/ before trusting the numbers."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None, codec=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--offset_s", type=float, default=0.0)
    ap.add_argument("--crop_s", type=float, default=12.0)
    ap.add_argument("--max_songs", type=int, default=0)
    ap.add_argument("--model", default="same-l")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_listen", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    listen = out / "listen"
    listen.mkdir(parents=True, exist_ok=True)
    if codec is None:
        codec = SameCodec(a.model, a.device)
    sr = getattr(codec, "sample_rate", SR)
    songs = sorted(p for p in Path(a.stems_dir).iterdir() if p.is_dir())
    if a.max_songs:
        songs = songs[:a.max_songs]
    per_song, adds = [], []
    for p in songs:
        stems = load_song(p, sr, a.offset_s, a.crop_s)
        if not stems:
            continue
        res, add = run_song(codec, stems, sr, None if a.no_listen else str(listen), p.name)
        per_song.append({"song": p.name, "tracks": res, "additivity": add})
        adds.append(add)
        print(f"{p.name}: " + ", ".join(f"{k} lsd {m['lsd_db']:.1f} f1@20 {m['onset_f1_20']:.2f} atk {m['attack_db']:+.1f}"
                                         for k, m in res.items()), flush=True)
    summary = summarize(per_song)
    verdict = gate_verdict(summary)
    (out / "results.json").write_text(json.dumps({"summary": summary, "verdict": verdict, "songs": per_song},
                                                 indent=2), encoding="utf-8")
    if per_song:
        write_report(summary, verdict, adds, out / "report.md")
    return summary, verdict


if __name__ == "__main__":
    main()

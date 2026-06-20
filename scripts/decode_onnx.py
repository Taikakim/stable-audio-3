#!/usr/bin/env python
"""decode_onnx.py — decode SAME (Stable Audio 3) latents to audio with the
ONNX decoder (from export_same_onnx.py) via ONNX Runtime, using a host-side
fixed-chunk loop + overlap-add stitch.

The exported graph is a *single fixed-size chunk* (`[1, C, L] -> [1, 2, L*ds]`).
Arbitrary-length latents are decoded by looping the graph over chunks and
stitching with overlap-add — a faithful port of
`stable_audio_3.models.autoencoders.AudioAutoencoder.decode_audio(chunked=True)`,
so the seam handling matches torch exactly. `--chunk-latents`/`--overlap` MUST
match the export's chunk size (`L`); `overlap` should be >= the decoder's
receptive field or you get seam artefacts.

No torch needed for decoding — only numpy + onnxruntime + soundfile. Pass
`--compare-torch` to diff against the real `AudioAutoencoder.decode_audio`
(that branch needs the SA3 model + venv).

Usage
-----
    # decode one latents_sa3 crop with the exported chunk-128 decoder, on AMD:
    python scripts/decode_onnx.py --onnx same_decoder_L128.onnx \\
        --crop 000000 --chunk-latents 128 --overlap 16 --out out.wav

    # or a direct .npy, forcing CPU, and measure the seam vs torch:
    python scripts/decode_onnx.py --onnx same_decoder_L128.onnx \\
        --npy /path/to/latent.npy --chunk-latents 128 --overlap 16 \\
        --provider cpu --compare-torch
"""
import argparse
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100
DOWNSAMPLING_RATIO = 4096   # SAME-L: 256 (patch) * 16 (encoder stride)
LATENT_DIM = 256
DEFAULT_LATENT_DIR = Path("/home/kim/Projects/latents_sa3")

_PROVIDER_ALIASES = {
    "migraphx": "MIGraphXExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def load_latents(args) -> np.ndarray:
    """Return latents as float32 [1, C, T]."""
    if args.npy:
        path = Path(args.npy)
    elif args.crop:
        path = Path(args.latent_dir) / f"{args.crop}.npy"
    else:
        raise SystemExit("provide --crop or --npy")
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]            # [C, T] -> [1, C, T]
    if arr.ndim != 3 or arr.shape[1] != LATENT_DIM:
        raise SystemExit(f"expected [1,{LATENT_DIM},T], got {arr.shape} from {path}")
    return arr


def pick_providers(choice: str | None) -> list[str]:
    import onnxruntime as ort
    available = set(ort.get_available_providers())
    if choice:
        want = _PROVIDER_ALIASES.get(choice.lower(), choice)
        if want not in available:
            raise SystemExit(f"provider {want} not available; have {sorted(available)}")
        return [want, "CPUExecutionProvider"] if want != "CPUExecutionProvider" else [want]
    # Auto: prefer AMD GPU EPs, fall back to CPU.
    order = ["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in available] or ["CPUExecutionProvider"]


def _augment_migraphx(providers: list[str], args) -> list:
    """Attach MIGraphX provider options for compile-caching / fp16 when requested.
    Caching makes the multi-minute AOT compile a one-time cost: first run compiles
    and saves the program, later runs load it. Returns a providers list where the
    MIGraphX entry may be a (name, options) tuple."""
    if not (args.cache_dir or args.ep_fp16):
        return providers
    out = []
    for p in providers:
        if p != "MIGraphXExecutionProvider":
            out.append(p)
            continue
        opts: dict[str, str] = {}
        if args.ep_fp16:
            opts["migraphx_fp16_enable"] = "1"
        if args.cache_dir:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            tag = "fp16" if args.ep_fp16 else "fp32"
            cache = args.cache_dir / f"{Path(args.onnx).stem}_L{args.chunk_latents}_{tag}.migx"
            opts.update({
                "migraphx_save_compiled_model": "1",
                "migraphx_save_model_name": str(cache),
                "migraphx_load_compiled_model": "1",
                "migraphx_load_model_name": str(cache),
            })
            print(f"[ort] MIGraphX compile cache: {cache}"
                  + ("  (exists — will load)" if cache.exists() else "  (will compile+save)"))
        out.append(("MIGraphXExecutionProvider", opts))
    return out


def decode_chunked_onnx(sess, latents: np.ndarray, chunk_size: int,
                        overlap: int, ds: int = DOWNSAMPLING_RATIO) -> np.ndarray:
    """Port of AudioAutoencoder.decode_audio(chunked=True) using an ORT session
    for each fixed-size chunk. latents [1, C, T] -> audio [1, 2, T*ds]."""
    in_name = sess.get_inputs()[0].name
    B, C, total_latents = latents.shape

    def _decode(chunk: np.ndarray) -> np.ndarray:
        # chunk must be exactly [B, C, chunk_size] to match the static graph.
        return sess.run(None, {in_name: chunk.astype(np.float32)})[0]

    # Short input: pad to one chunk, decode once, trim back.
    if total_latents <= chunk_size:
        pad = chunk_size - total_latents
        padded = np.pad(latents, ((0, 0), (0, 0), (0, pad)))
        out = _decode(padded)
        return out[..., : total_latents * ds]

    hop = chunk_size - overlap
    chunk_starts = list(range(0, total_latents - chunk_size + 1, hop))
    if chunk_starts[-1] != total_latents - chunk_size:
        chunk_starts.append(total_latents - chunk_size)   # anchor final chunk to the end

    decoded = [_decode(latents[..., s:s + chunk_size]) for s in chunk_starts]

    total_samples = total_latents * ds
    chunk_size_samples = chunk_size * ds
    half_overlap_samples = (overlap // 2) * ds
    out = np.zeros((*decoded[0].shape[:-1], total_samples), dtype=np.float32)
    n = len(chunk_starts)
    for i, (start_latent, chunk) in enumerate(zip(chunk_starts, decoded)):
        is_first, is_last = i == 0, i == n - 1
        out_start = (total_samples - chunk_size_samples) if is_last else start_latent * ds
        left = 0 if is_first else half_overlap_samples
        right = chunk_size_samples if is_last else chunk_size_samples - half_overlap_samples
        out[..., out_start + left: out_start + right] = chunk[..., left:right]
    return out


def summarize_placement(profile_path: str, requested: list[str]) -> None:
    """Parse an ORT profiling JSON and report which execution provider actually
    ran each node — the definitive check for silent CPU fallback. ORT emits one
    `cat:"Node"` kernel event per executed node with `args.provider`."""
    import json
    from collections import defaultdict
    with open(profile_path) as f:
        events = json.load(f)

    by_provider_dur = defaultdict(float)     # provider -> total kernel us
    by_provider_ops = defaultdict(set)       # provider -> {op_name}
    cpu_op_dur = defaultdict(float)          # op_name -> us on CPU
    for e in events:
        if e.get("cat") != "Node":
            continue
        args = e.get("args") or {}
        prov, op = args.get("provider"), args.get("op_name")
        if not prov or not op:
            continue
        dur = float(e.get("dur", 0))
        by_provider_dur[prov] += dur
        by_provider_ops[prov].add(op)
        if "CPU" in prov:
            cpu_op_dur[op] += dur

    total = sum(by_provider_dur.values()) or 1.0
    print("\n[placement] node execution time by provider:")
    for prov in sorted(by_provider_dur, key=by_provider_dur.get, reverse=True):
        pct = 100 * by_provider_dur[prov] / total
        print(f"  {prov:32s} {by_provider_dur[prov]/1000:9.2f} ms  {pct:5.1f}%  "
              f"({len(by_provider_ops[prov])} op types)")

    gpu_eps = [p for p in requested if "CPU" not in p]
    if gpu_eps and cpu_op_dur:
        print(f"\n[placement] ⚠ ops that FELL BACK to CPU (coverage gaps on {gpu_eps[0]}):")
        for op, dur in sorted(cpu_op_dur.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {op:28s} {dur/1000:8.2f} ms")
        cpu_share = 100 * sum(cpu_op_dur.values()) / total
        print(f"[placement] {cpu_share:.1f}% of node time ran on CPU — "
              + ("mostly GPU, minor fallback" if cpu_share < 15 else
                 "SIGNIFICANT fallback; MIGraphX op coverage is the bottleneck"))
    elif gpu_eps:
        print(f"[placement] ✅ no CPU fallback — every node ran on {gpu_eps[0]}")


def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """audio [1, 2, samples] or [2, samples] -> int16 WAV via soundfile PCM_16
    (never torchaudio.save — MASTER §5: torchcodec clips fp16)."""
    import soundfile as sf
    a = audio[0] if audio.ndim == 3 else audio       # [2, samples]
    a = np.clip(a, -1.0, 1.0).T                       # [samples, 2]
    sf.write(str(path), a, sr, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, type=Path, help="exported decoder .onnx")
    ap.add_argument("--crop", help="crop id in --latent-dir (e.g. 000000)")
    ap.add_argument("--npy", help="direct path to a latent .npy")
    ap.add_argument("--latent-dir", default=str(DEFAULT_LATENT_DIR))
    ap.add_argument("--chunk-latents", type=int, default=128,
                    help="MUST match the export's chunk size L")
    ap.add_argument("--overlap", type=int, default=16,
                    help="latent-frame overlap for the stitch; >= receptive field")
    ap.add_argument("--provider", default=None,
                    help="migraphx | rocm | cpu | <full EP name> (default: auto, GPU-first)")
    ap.add_argument("--out", type=Path, default=Path("decode_onnx.wav"))
    ap.add_argument("--compare-torch", action="store_true",
                    help="also run AudioAutoencoder.decode_audio and diff (needs torch+model)")
    ap.add_argument("--model", default="same-l")
    ap.add_argument("--report-placement", action="store_true",
                    help="dump which EP actually ran each node (detects silent CPU fallback)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="cache the compiled MIGraphX program here so the ~min AOT "
                         "compile is one-time (first run compiles+saves, later runs load)")
    ap.add_argument("--ep-fp16", action="store_true",
                    help="run the MIGraphX EP in fp16 (faster/less VRAM; slight precision loss)")
    args = ap.parse_args()

    import onnxruntime as ort

    latents = load_latents(args)
    providers = pick_providers(args.provider)
    providers = _augment_migraphx(providers, args)
    print(f"[ort] providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
    so = ort.SessionOptions()
    so.enable_profiling = args.report_placement
    req = args.provider and _PROVIDER_ALIASES.get(args.provider.lower(), args.provider)
    plain = [p[0] if isinstance(p, tuple) else p for p in providers]

    sess = ort.InferenceSession(str(args.onnx), sess_options=so, providers=providers)
    # ORT silently falls back to CPU when a requested GPU EP errors (e.g. an
    # unsupported provider option in this build) — so a "CPU pretending to be the
    # GPU run" never gets mistaken for verified AMD inference. If we asked for a
    # GPU EP, got CPU, and had passed extra options, retry on the bare GPU EP.
    if req and "CPU" not in req and "CPU" in sess.get_providers()[0]:
        if providers != plain:
            print(f"[ort] ⚠ provider options rejected by this build → silent CPU fallback; "
                  f"retrying on bare {req} (compile caching unavailable here)")
            sess = ort.InferenceSession(str(args.onnx), sess_options=so, providers=plain)
        if "CPU" in sess.get_providers()[0]:
            print(f"[ort] ⚠ WARNING: requested {req} but ACTIVE EP is CPU — "
                  f"this is NOT an AMD-GPU run (check EP availability).")
    print(f"[ort] active: {sess.get_providers()[0]}")

    L = latents.shape[-1]
    print(f"[run] latents {latents.shape}  chunk={args.chunk_latents}  overlap={args.overlap}")
    t0 = time.time()
    audio = decode_chunked_onnx(sess, latents, args.chunk_latents, args.overlap)
    dt = time.time() - t0
    secs = audio.shape[-1] / SAMPLE_RATE
    print(f"[run] -> audio {audio.shape}  ({secs:.1f}s)  in {dt:.2f}s  "
          f"RTF={secs / dt:.1f}x")

    save_wav(args.out, audio, SAMPLE_RATE)
    print(f"[out] wrote {args.out}")

    if args.report_placement:
        prof = sess.end_profiling()
        summarize_placement(prof, providers)

    if args.compare_torch:
        print("[compare] decoding with torch AudioAutoencoder.decode_audio ...")
        import os
        os.environ.setdefault("SA3_DISABLE_FLASH_ATTN", "1")
        import torch
        from stable_audio_3 import AutoencoderModel
        model = AutoencoderModel.from_pretrained(args.model, device="cpu")
        ae = model.autoencoder.eval()
        with torch.no_grad():
            ref = ae.decode_audio(torch.from_numpy(latents).float(),
                                  chunked=True, chunk_size=args.chunk_latents,
                                  overlap=args.overlap).cpu().numpy()
        m = min(ref.shape[-1], audio.shape[-1])
        d = np.abs(ref[..., :m] - audio[..., :m])
        denom = (np.linalg.norm(ref[..., :m]) * np.linalg.norm(audio[..., :m])) or 1.0
        cos = float((ref[..., :m] * audio[..., :m]).sum() / denom)
        print(f"[compare] vs torch decode_audio: max|Δ|={d.max():.3e}  "
              f"mean|Δ|={d.mean():.3e}  cos={cos:.6f}")
        print("[compare] seam ok — overlap is sufficient" if d.max() < 5e-2 else
              "[compare] WARNING: large diff — raise --overlap (>= receptive field)")


if __name__ == "__main__":
    main()

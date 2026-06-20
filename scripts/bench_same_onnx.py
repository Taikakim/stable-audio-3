#!/usr/bin/env python
"""bench_same_onnx.py — benchmark the ONNX/AMD SAME decode against the stock
torch SAME-L decode: latency / RTF, peak VRAM, and reconstruction quality.

"Stock model" = the torch `AudioAutoencoder.decode_audio(chunked=True)` path
(what mir's /decode uses today). We can't compare against the cgisky MNN models
(different autoencoder, CUDA/Windows-only), so torch is the reference both for
speed and as the gold output for the quality diff.

Backends (select with --backends):
  torch                stock fp32 torch decode (the reference)
  onnx-cpu             ORT CPUExecutionProvider (sanity baseline)
  onnx-migraphx        ORT MIGraphX EP, fp32 onnx
  onnx-migraphx-fp16   ORT MIGraphX EP, fp16 onnx (the low-VRAM target)

Lengths are sliced from one real latents_sa3 crop to show the RTF curve and the
constant-VRAM claim. Run with nothing else on the GPU for clean VRAM numbers.

Requires: numpy, onnxruntime(-rocm/migraphx for GPU EPs); torch + stable_audio_3
only if 'torch' is in --backends or for the quality reference. Needs `rocm-smi`
on PATH for device VRAM sampling.

Usage
-----
    python scripts/bench_same_onnx.py --crop 000000 \\
        --onnx-fp32 same_decoder_L128.onnx --onnx-fp16 same_decoder_L128_fp16.onnx \\
        --backends torch,onnx-migraphx,onnx-migraphx-fp16 \\
        --lengths 128,512,1024,4096 --chunk-latents 128 --overlap 16 --runs 3
"""
import argparse
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

# Reuse the validated chunk-loop + provider selection.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_onnx import (  # noqa: E402
    DEFAULT_LATENT_DIR, SAMPLE_RATE, decode_chunked_onnx, pick_providers,
)

_EP = {
    "onnx-cpu": "cpu",
    "onnx-migraphx": "migraphx",
    "onnx-migraphx-fp16": "migraphx",
}


# ---------------------------------------------------------------------------
# device VRAM sampling (rocm-smi) — device-level, fair across torch and ORT
# ---------------------------------------------------------------------------
def _rocm_used_bytes() -> int | None:
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            text=True, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return None
    # CSV has a 'VRAM Total Used Memory (B)' column; grab the first numeric cell.
    for line in out.splitlines():
        for cell in line.split(","):
            cell = cell.strip()
            if cell.isdigit():
                return int(cell)
    return None


class VramSampler:
    """Background thread sampling device VRAM; reports peak delta over baseline."""
    def __init__(self, period=0.05):
        self.period, self._stop, self.peak, self.base = period, False, 0, 0

    def __enter__(self):
        self.base = _rocm_used_bytes() or 0
        self.peak = self.base
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        while not self._stop:
            u = _rocm_used_bytes()
            if u and u > self.peak:
                self.peak = u
            time.sleep(self.period)

    def __exit__(self, *a):
        self._stop = True
        self._t.join(timeout=1)

    @property
    def peak_delta_gb(self) -> float:
        return max(0, self.peak - self.base) / 1e9


# ---------------------------------------------------------------------------
# backends: each returns a callable latents[1,C,T] -> audio[1,2,T*ds] (numpy)
# ---------------------------------------------------------------------------
def make_torch_backend(model_id, chunk, overlap, fp16=False):
    import os
    os.environ.setdefault("SA3_DISABLE_FLASH_ATTN", "1")
    import torch
    from stable_audio_3 import AutoencoderModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoencoderModel.from_pretrained(model_id, device=dev)
    ae = model.autoencoder.eval().to(dev)
    if fp16:
        ae = ae.half()

    def run(latents: np.ndarray) -> np.ndarray:
        z = torch.from_numpy(latents).to(dev)
        z = z.half() if fp16 else z.float()
        with torch.no_grad():
            a = ae.decode_audio(z, chunked=True, chunk_size=chunk, overlap=overlap)
        return a.float().cpu().numpy()
    return run


def make_onnx_backend(onnx_path, ep, chunk, overlap):
    import onnxruntime as ort
    providers = pick_providers(ep)
    so = ort.SessionOptions()
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    active = sess.get_providers()[0]

    def run(latents: np.ndarray) -> np.ndarray:
        return decode_chunked_onnx(sess, latents, chunk, overlap)
    return run, active


# ---------------------------------------------------------------------------
def time_backend(run, latents, n_runs):
    """Warmup (excluded — MIGraphX compiles kernels on first call), then median
    of n_runs. Returns (cold_s, median_s, output)."""
    t0 = time.time()
    out = run(latents)            # cold: includes kernel compile
    cold = time.time() - t0
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        out = run(latents)
        times.append(time.time() - t0)
    return cold, float(np.median(times)), out


def quality(ref: np.ndarray, out: np.ndarray) -> tuple[float, float]:
    m = min(ref.shape[-1], out.shape[-1])
    r, o = ref[..., :m].ravel(), out[..., :m].ravel()
    denom = (np.linalg.norm(r) * np.linalg.norm(o)) or 1.0
    cos = float((r * o).sum() / denom)
    rel = float(np.abs(r - o).max() / (np.abs(r).max() or 1.0))
    return cos, rel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crop", default="000000")
    ap.add_argument("--latent-dir", default=str(DEFAULT_LATENT_DIR))
    ap.add_argument("--onnx-fp32", type=Path, help="fp32 decoder onnx")
    ap.add_argument("--onnx-fp16", type=Path, help="fp16 decoder onnx")
    ap.add_argument("--backends", default="torch,onnx-migraphx,onnx-migraphx-fp16")
    ap.add_argument("--lengths", default="128,512,1024,4096",
                    help="latent lengths to decode (sliced from the crop)")
    ap.add_argument("--chunk-latents", type=int, default=128, help="MUST match the onnx export L")
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default="same-l")
    ap.add_argument("--json", type=Path, help="also dump raw results as JSON")
    args = ap.parse_args()

    backends = args.backends.split(",")
    lengths = [int(x) for x in args.lengths.split(",")]
    full = np.load(Path(args.latent_dir) / f"{args.crop}.npy").astype(np.float32)
    if full.ndim == 2:
        full = full[None]
    max_len = full.shape[-1]
    lengths = [L for L in lengths if L <= max_len] or [max_len]

    # Build backend callables once.
    runners = {}
    for b in backends:
        if b == "torch":
            runners[b] = make_torch_backend(args.model, args.chunk_latents, args.overlap)
        elif b == "torch-fp16":
            runners[b] = make_torch_backend(args.model, args.chunk_latents, args.overlap, fp16=True)
        elif b in _EP:
            path = args.onnx_fp16 if "fp16" in b else args.onnx_fp32
            if not path:
                print(f"[skip] {b}: no onnx path provided"); continue
            run, active = make_onnx_backend(path, _EP[b], args.chunk_latents, args.overlap)
            runners[b] = run
            print(f"[init] {b} -> EP {active}")
        else:
            print(f"[skip] unknown backend {b}")

    results = []
    refs = {}   # length -> torch fp32 reference output (gold)
    for L in lengths:
        latents = full[..., :L].copy()
        audio_s = L * 4096 / SAMPLE_RATE
        # Reference = torch if available, else first backend.
        ref_backend = "torch" if "torch" in runners else next(iter(runners))
        for b, run in runners.items():
            with VramSampler() as vram:
                cold, med, out = time_backend(run, latents, args.runs)
            if b == ref_backend and L not in refs:
                refs[L] = out
            cos, rel = quality(refs.get(L, out), out)
            rtf = audio_s / med if med > 0 else float("inf")
            results.append(dict(backend=b, latents=L, audio_s=round(audio_s, 1),
                                cold_s=round(cold, 3), med_s=round(med, 3),
                                rtf=round(rtf, 1), vram_gb=round(vram.peak_delta_gb, 2),
                                cos=round(cos, 6), max_rel=round(rel, 4)))
            print(f"  [{b:20s} L={L:5d}] {med:7.3f}s  RTF {rtf:6.1f}x  "
                  f"VRAM {vram.peak_delta_gb:4.2f}GB  cos {cos:.5f}  (cold {cold:.2f}s)")

    # Markdown table
    print("\n| backend | latents | audio | median s | RTF | VRAM GB | cos | max-rel | cold s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['backend']} | {r['latents']} | {r['audio_s']}s | {r['med_s']} | "
              f"{r['rtf']}x | {r['vram_gb']} | {r['cos']} | {r['max_rel']} | {r['cold_s']} |")
    print("\nNotes: VRAM = device-level rocm-smi delta (run with an idle GPU); cos/max-rel "
          "vs the torch-fp32 output; cold = first-call incl. MIGraphX kernel compile.")

    if args.json:
        import json
        args.json.write_text(json.dumps(results, indent=2))
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    main()

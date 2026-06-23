#!/usr/bin/env python
"""bench_dit_onnx.py — fair head-to-head benchmark of SA3 DiT *generation*
(rectified-flow Euler loop + CFG + SAME decode) on ONNX (ORT + MIGraphX) vs
native torch, on byte-for-byte identical inputs.

One backend per invocation (`--backend`); the caller runs each backend in its
own venv and diffs the saved z0 across runs (cosine) externally.

    --backend torch               native fp16 CUDA path (the reference)
    --backend onnx-migraphx       ORT MIGraphX EP, fp32 onnx
    --backend onnx-migraphx-fp16  ORT MIGraphX EP, migraphx_fp16_enable='1'

WHICH VENV RUNS WHICH BACKEND (deps are imported lazily, so this module loads in
either venv; a backend fails with a clear message if its deps are absent):

    torch backend          → SA3 venv:  /home/kim/Projects/SAO/stable-audio-3/.venv/bin/python
                             (imports torch + stable_audio_3; runs the DiT in fp16 on cuda)
    onnx-migraphx[-fp16]    → mir venv:  /home/kim/Projects/mir/mir/bin/python
                             (imports onnxruntime with the MIGraphX EP)

FAIRNESS (the whole point — every backend is identical except torch-vs-ONNX):
  * SAME seed→noise   : x0 = np.random.default_rng(seed).standard_normal((1,256,T)) (fp32)
  * SAME schedule     : sig = np.linspace(1.0, 0.0, steps+1)
  * SAME cond/uncond  : the precached cond/uncond .npz (cross_attn_cond[seq,768],
                        cross_attn_mask[seq], global_embed[768])
  * SAME local_add    : zeros[1,257,T] (NOT omitted — the DiT projects it with a bias)
  * SAME CFG formula  : two batch-1 DiT evals/step, v = v_unc + cfg*(v_cond - v_unc)
  * SAME Euler step   : x += (sig[i+1]-sig[i]) * v
  * SAME decode       : decode_chunked_onnx (ONNX decoder) for BOTH backends, so the
                        only measured difference is the DiT path (torch vs ONNX).
We deliberately do NOT call model.generate() (it uses a different sampler) —
both backends replicate dit_onnx_infer.py's exact math.

METRICS (emitted to --json + a one-line summary):
  * cold_s          : first full generation, incl. MIGraphX AOT compile (onnx) or model
                      load (torch). Reported separately from the warm timing.
  * gen_s           : median over --runs of the WARM full generation (DiT loop + decode).
  * rtf             : audio_seconds / gen_s.
  * vram_warm_gb    : device-level peak during warm runs − process-start baseline (rocm-smi).
                      This is the steady-state, comparable headline number.
  * vram_peak_gb    : device-level peak across the whole process − baseline (incl. compile/load).
  * torch_max_alloc_gb : additionally torch.cuda.max_memory_allocated (torch only).
Saves the final latent z0 to <out>.z0.npy and the decoded wav to <out>.

NOTE on cross-backend dtype: per-backend dtype differs by design (ONNX fp32 EP = fp32 loop;
fp16 EP = fp16 DiT internally, fp32 host accumulation; torch = fp16 incl host accumulation
and fp16 t). The cross-backend z0 cosine is a 3-way *agreement* check, not an fp16==fp16 proof.

NOTE on schedule: the hardcoded linspace(1,0,steps+1) is the identity time-shift schedule;
it is correct for medium-base only (which uses an identity time-shift in training).

Usage
-----
    # ONNX (mir venv):
    /home/kim/Projects/mir/mir/bin/python scripts/bench_dit_onnx.py \\
        --backend onnx-migraphx-fp16 \\
        --dit-onnx dit_medium-base_L256.onnx --decoder-onnx same_decoder_L128.onnx \\
        --cond cond.npz --uncond uncond.npz --frames 256 --steps 8 \\
        --out gen_onnx.wav --json gen_onnx.json

    # torch (SA3 venv):
    /home/kim/Projects/SAO/stable-audio-3/.venv/bin/python scripts/bench_dit_onnx.py \\
        --backend torch \\
        --dit-onnx dit_medium-base_L256.onnx --decoder-onnx same_decoder_L128.onnx \\
        --cond cond.npz --uncond uncond.npz --frames 256 --steps 8 \\
        --out gen_torch.wav --json gen_torch.json
"""
import argparse
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Reuse the validated decoder chunk-loop + provider selection (importable; pulls
# in onnxruntime only lazily, inside the functions that use it).
import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_onnx import decode_chunked_onnx, pick_providers  # noqa: E402

SR = 44100
LATENT_DIM = 256
LOCAL_ADD_DIM = 257
DS = 4096

# Capture the device VRAM baseline at process start — before any InferenceSession or
# torch model is built — so the warm/peak deltas exclude weights already resident from
# a prior run and are comparable across backends.
_PROCESS_BASELINE_BYTES: int = 0


# ---------------------------------------------------------------------------
# device VRAM sampling (rocm-smi) — device-level, fair across torch and ORT.
# Copied from bench_same_onnx.py so this benchmark stands alone.
# ---------------------------------------------------------------------------
def _rocm_used_bytes() -> int | None:
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            text=True, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return None
    for line in out.splitlines():
        for cell in line.split(","):
            cell = cell.strip()
            if cell.isdigit():
                return int(cell)
    return None


class VramSampler:
    """Background thread sampling device VRAM; reports peak delta over an external baseline.

    Pass ``baseline_bytes`` to fix the reference point to the process-start measurement
    (captured before any session/model is built). If omitted, the baseline is sampled at
    __enter__ time (within-context baseline, matches the old behaviour).
    """
    def __init__(self, period=0.05, baseline_bytes: int | None = None):
        self.period, self._stop, self.peak = period, False, 0
        self._ext_baseline = baseline_bytes

    def __enter__(self):
        self.base = self._ext_baseline if self._ext_baseline is not None else (_rocm_used_bytes() or 0)
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
# shared inputs (identical for every backend) — the fairness contract.
# ---------------------------------------------------------------------------
def load_cond(path: Path):
    """npz -> (cross_attn_cond[seq,768] fp32, cross_attn_mask[seq], global_embed[768] fp32)."""
    z = np.load(path)
    return (z["cross_attn_cond"].astype(np.float32),
            z["cross_attn_mask"],
            z["global_embed"].astype(np.float32))


def make_inputs(args):
    """Build the SAME inputs every backend must consume: seed->noise, schedule,
    cond/uncond, zero local_add_cond. Returns a dict of numpy arrays."""
    T = args.frames
    rng = np.random.default_rng(args.seed)
    x0 = rng.standard_normal((1, LATENT_DIM, T)).astype(np.float32)     # sigma_max=1.0
    sig = np.linspace(1.0, 0.0, args.steps + 1).astype(np.float64)      # identity schedule
    c_cross, c_mask, c_glob = load_cond(args.cond)
    u_cross, u_mask, u_glob = load_cond(args.uncond)
    local_add = np.zeros((1, LOCAL_ADD_DIM, T), np.float32)             # non-inpaint, fed (has bias)
    return dict(x0=x0, sig=sig, local_add=local_add,
                c_cross=c_cross, c_mask=c_mask, c_glob=c_glob,
                u_cross=u_cross, u_mask=u_mask, u_glob=u_glob)


# ---------------------------------------------------------------------------
# ONNX backend (mir venv): ORT + MIGraphX EP, fp32 or migraphx_fp16_enable.
# ---------------------------------------------------------------------------
def make_onnx_backend(args, inp, fp16: bool):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise SystemExit(
            "[onnx] onnxruntime not importable — run the onnx backends in the mir venv: "
            "/home/kim/Projects/mir/mir/bin/python") from e

    providers = pick_providers(args.provider)
    if fp16:
        providers = [("MIGraphXExecutionProvider", {"migraphx_fp16_enable": "1"})
                     if p == "MIGraphXExecutionProvider" else p for p in providers]
    plain = [p[0] if isinstance(p, tuple) else p for p in providers]
    print(f"[onnx] providers: {plain}" + (" (fp16 EP)" if fp16 else " (fp32)"))

    dit = ort.InferenceSession(str(args.dit_onnx), providers=providers)
    dec = ort.InferenceSession(str(args.decoder_onnx), providers=providers)
    print(f"[onnx] DiT EP {dit.get_providers()[0]}  decoder EP {dec.get_providers()[0]}")

    sig, local_add = inp["sig"], inp["local_add"]
    cfg = args.cfg_scale

    def dit_v(x, cross1, mask1, glob1, t_cur):
        return dit.run(None, {
            "x": x, "t": np.full((1,), t_cur, np.float32),
            "cross_attn_cond": cross1[None], "cross_attn_cond_mask": mask1[None],
            "global_embed": glob1[None], "local_add_cond": local_add})[0]

    def generate():
        """One full generation: rectified-flow Euler loop (CFG) + decode -> (z0, audio)."""
        x = inp["x0"].copy()
        for i in range(args.steps):
            t_cur, dt = float(sig[i]), float(sig[i + 1] - sig[i])
            v_cond = dit_v(x, inp["c_cross"], inp["c_mask"], inp["c_glob"], t_cur)
            if cfg == 1.0:
                v = v_cond
            else:
                v_unc = dit_v(x, inp["u_cross"], inp["u_mask"], inp["u_glob"], t_cur)
                v = v_unc + cfg * (v_cond - v_unc)                     # CFG (velocity space)
            x = x + dt * v
        audio = decode_chunked_onnx(dec, x, args.decode_chunk, args.decode_overlap, DS)
        return x.astype(np.float32), audio

    return generate


# ---------------------------------------------------------------------------
# torch backend (SA3 venv): native fp16 CUDA path; SAME math, SAME ONNX decoder.
# ---------------------------------------------------------------------------
def make_torch_backend(args, inp):
    import os
    os.environ.setdefault("SA3_DISABLE_FLASH_ATTN", "1")
    try:
        import torch
        from stable_audio_3 import StableAudioModel
    except ImportError as e:
        raise SystemExit(
            "[torch] torch / stable_audio_3 not importable — run the torch backend in the "
            "SA3 venv: /home/kim/Projects/SAO/stable-audio-3/.venv/bin/python") from e
    if not torch.cuda.is_available():
        raise SystemExit("[torch] cuda not available — the torch backend is the native GPU path")

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise SystemExit(
            "[torch] onnxruntime not importable, but the decoder must be the SAME ONNX decoder "
            "as the onnx backends for a fair DiT-only comparison. Install onnxruntime in this "
            "venv, or run decode in a separate process.") from e

    dev = "cuda"
    DTYPE = torch.float16                                              # native inference dtype

    print(f"[torch] loading {args.model} (DiT) on {dev} fp16 ...")
    model = StableAudioModel.from_pretrained(args.model)
    cdm = model.model.to(dev)
    dit = next(m for m in cdm.modules()
               if type(m).__name__ == "DiffusionTransformer").eval().to(DTYPE)

    # The decoder is the SAME ONNX decoder both backends use → only the DiT differs.
    dec = ort.InferenceSession(str(args.decoder_onnx), providers=pick_providers(args.provider))
    print(f"[torch] decoder EP {dec.get_providers()[0]} (shared ONNX decoder)")

    sig, cfg = inp["sig"], args.cfg_scale
    local_add = torch.from_numpy(inp["local_add"]).to(dev, DTYPE)

    def to_t(a, dtype=DTYPE):
        return torch.from_numpy(np.asarray(a)).to(dev, dtype)

    c_cross, c_mask, c_glob = to_t(inp["c_cross"])[None], to_t(inp["c_mask"], torch.bool)[None], to_t(inp["c_glob"])[None]
    u_cross, u_mask, u_glob = to_t(inp["u_cross"])[None], to_t(inp["u_mask"], torch.bool)[None], to_t(inp["u_glob"])[None]

    def dit_v(x, cross, mask, glob, t_cur):
        return dit._forward(x, torch.full((1,), t_cur, device=dev, dtype=DTYPE),
                            cross_attn_cond=cross, cross_attn_cond_mask=mask,
                            global_embed=glob, local_add_cond=local_add)

    def generate():
        """SAME loop as ONNX, in torch fp16 on cuda; decode via the shared ONNX decoder."""
        x = torch.from_numpy(inp["x0"]).to(dev, DTYPE)
        with torch.no_grad():
            for i in range(args.steps):
                t_cur, dt = float(sig[i]), float(sig[i + 1] - sig[i])
                v_cond = dit_v(x, c_cross, c_mask, c_glob, t_cur)
                if cfg == 1.0:
                    v = v_cond
                else:
                    v_unc = dit_v(x, u_cross, u_mask, u_glob, t_cur)
                    v = v_unc + cfg * (v_cond - v_unc)                # CFG (velocity space)
                x = x + dt * v
        z0 = x.float().cpu().numpy()
        audio = decode_chunked_onnx(dec, z0, args.decode_chunk, args.decode_overlap, DS)
        return z0.astype(np.float32), audio

    return generate


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True,
                    choices=["torch", "onnx-migraphx", "onnx-migraphx-fp16"])
    ap.add_argument("--dit-onnx", required=True, type=Path)
    ap.add_argument("--decoder-onnx", required=True, type=Path)
    ap.add_argument("--cond", required=True, type=Path,
                    help="npz: cross_attn_cond[seq,768], cross_attn_mask[seq], global_embed[768]")
    ap.add_argument("--uncond", required=True, type=Path, help="npz for the negative/empty prompt")
    ap.add_argument("--frames", type=int, required=True, help="T — MUST match the DiT export rung")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode-chunk", type=int, default=128, help="MUST match the decoder export L")
    ap.add_argument("--decode-overlap", type=int, default=16)
    ap.add_argument("--runs", type=int, default=3, help="warm runs to take the median over")
    ap.add_argument("--provider", default="migraphx")
    ap.add_argument("--out", type=Path, default=Path("bench_dit.wav"))
    ap.add_argument("--json", type=Path, help="results path (one JSON dict)")
    ap.add_argument("--model", default="medium-base")
    args = ap.parse_args()

    # Capture the device VRAM baseline NOW — before building any InferenceSession or
    # loading the torch model — so warm/peak deltas are steady-state and comparable
    # across backends (compile scratch and weight-load transients are excluded).
    global _PROCESS_BASELINE_BYTES
    _PROCESS_BASELINE_BYTES = _rocm_used_bytes() or 0

    inp = make_inputs(args)
    T = args.frames
    audio_s = T * DS / SR

    fp16 = args.backend == "onnx-migraphx-fp16"
    if args.backend == "torch":
        generate = make_torch_backend(args, inp)
    else:
        generate = make_onnx_backend(args, inp, fp16=fp16)

    # torch.cuda peak (torch backend only): reset before the cold call.
    torch_max_alloc_gb = None
    _torch = None
    if args.backend == "torch":
        import torch as _torch
        _torch.cuda.reset_peak_memory_stats()

    print(f"[gen] backend={args.backend}  T={T}  steps={args.steps}  cfg={args.cfg_scale}  "
          f"audio={audio_s:.1f}s  runs={args.runs}")

    # COLD: first full generation — incl. MIGraphX AOT compile (onnx) / model load
    # warmup of kernels (torch). Track peak across the whole process (incl. compile).
    with VramSampler(baseline_bytes=_PROCESS_BASELINE_BYTES) as vram_full:
        t0 = time.time()
        z0, audio = generate()                                        # cold
        if _torch is not None:
            _torch.cuda.synchronize()                                 # sync before timing cold_s
        cold_s = time.time() - t0

        # WARM: median over --runs of the full generation (DiT loop + decode).
        # Track VRAM separately so the steady-state number excludes compile transients.
        with VramSampler(baseline_bytes=_PROCESS_BASELINE_BYTES) as vram_warm:
            times = []
            for _ in range(args.runs):
                t0 = time.time()
                z0, audio = generate()
                if _torch is not None:
                    _torch.cuda.synchronize()
                times.append(time.time() - t0)

    gen_s = float(np.median(times))
    rtf = audio_s / gen_s if gen_s > 0 else float("inf")

    if _torch is not None:
        torch_max_alloc_gb = round(_torch.cuda.max_memory_allocated() / 1e9, 3)

    # Save z0 reliably (caller diffs z0 across backends with cosine) + the wav.
    z0_path = str(args.out) + ".z0.npy"
    np.save(z0_path, z0)
    a = np.clip(audio[0], -1.0, 1.0).T
    sf.write(str(args.out), a, SR, subtype="PCM_16")

    result = dict(
        backend=args.backend, model=args.model, frames=T, steps=args.steps,
        cfg_scale=args.cfg_scale, seed=args.seed, runs=args.runs,
        audio_s=round(audio_s, 2), cold_s=round(cold_s, 3), gen_s=round(gen_s, 3),
        gen_runs_s=[round(t, 3) for t in times], rtf=round(rtf, 2),
        vram_warm_gb=round(vram_warm.peak_delta_gb, 3),   # steady-state: warm runs only
        vram_peak_gb=round(vram_full.peak_delta_gb, 3),   # incl. compile/load transients
        torch_max_alloc_gb=torch_max_alloc_gb,
        z0_path=z0_path, z0_shape=list(z0.shape),
        z0_range=[round(float(z0.min()), 3), round(float(z0.max()), 3)],
        wav=str(args.out),
    )

    if args.json:
        import json
        args.json.write_text(json.dumps(result, indent=2))
        print(f"[json] wrote {args.json}")

    extra = f"  torch_max_alloc {torch_max_alloc_gb}GB" if torch_max_alloc_gb is not None else ""
    print(f"[done] {args.backend}: gen {gen_s:.3f}s (median/{args.runs})  RTF {rtf:.2f}x  "
          f"VRAM(warm) {vram_warm.peak_delta_gb:.2f}GB  VRAM(peak/incl-compile) {vram_full.peak_delta_gb:.2f}GB"
          f"{extra}  cold {cold_s:.2f}s  -> {args.out}, {z0_path}")


if __name__ == "__main__":
    main()

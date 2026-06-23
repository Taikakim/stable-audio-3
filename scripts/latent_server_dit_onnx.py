#!/home/kim/Projects/mir/mir/bin/python
"""latent_server_dit_onnx.py — low-VRAM SA3 *generation* (text→audio) server on
ONNX Runtime + MIGraphX (AMD GPU). The generative sibling of
`latent_server_onnx.py` (which is decode-only): this one runs the full ONNX DiT
rectified-flow sampler **and** the ONNX decoder, both compiled once at boot.

Pipeline (text precached offline → pure numpy/ORT at runtime)
------------------------------------------------------------
Text → embeddings is the *only* part that needs the SA3 venv + T5-Gemma, and it
is done **offline** by `precache_dit_cond.py`, which writes a per-prompt `.npz`:

    cond/uncond .npz {cross_attn_cond[seq,768], cross_attn_mask[seq], global_embed[768]}

At runtime the server is torch-free. `/generate?cond=<npz>&uncond=<npz>` runs:

    DiT-onnx[L] ×steps  (rectified-flow Euler, CFG = two batch-1 calls/step)
        → z0[1,256,T]  →  ONNX decoder (chunk-loop)  →  WAV

CFG is vanilla rectified-flow in velocity space: per step a cond + uncond DiT
call, then v = v_uncond + cfg·(v_cond − v_uncond); x += dt·v. The DiT is exported
static batch=1, so CFG costs two calls/step (a batch=2 export would halve it).
`local_add_cond` = zeros[1,257,T] (inpaint_mask + masked_input; zeros for
text-to-audio, but the DiT projects it with a bias so it MUST be fed, not omitted).

Boot
----
Runs in the **mir venv** (`onnxruntime_migraphx`; the SA3 venv's ORT is CPU-only).
At startup it AOT-compiles BOTH the DiT (~5.8 GB fp32) and the decoder on the
MIGraphX **fp16 EP** by default (`migraphx_fp16_enable`) — critical so the two fit
together on the 16 GB card (fp32-both saturates VRAM and the decoder compile
thrashes). The compile is a one-time per-session cost (~min); it prints the active
EP and warns loudly on silent CPU fallback. Pass `--no-fp16` to force fp32.

`--frames` (T) MUST match the DiT export's length rung (the DiT can't be chunked —
full-sequence attention — so each length is its own compiled graph).

`/steer`-style LatCH gradient guidance is **N/A** here: it needs torch backprop
through the DiT, so it stays on the torch server (`latent_server_sa3.py`). This
server is the low-VRAM, torch-free generation path.

Usage
-----
    /home/kim/Projects/mir/mir/bin/python scripts/latent_server_dit_onnx.py \\
        --dit-onnx dit_medium-base_L256.onnx --decoder-onnx same_decoder_L128.onnx \\
        --frames 256 --provider migraphx --port 7894

    # then (cond/uncond precached by precache_dit_cond.py):
    curl 'http://localhost:7894/generate?cond=acid.npz&uncond=empty.npz&steps=8&cfg=6.0&seed=42' -o gen.wav
"""
import argparse
import io
import json
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

# Reuse the *validated* sampler math + decoder chunk-loop, single-sourced so the
# rectified-flow schedule, the cond/uncond npz layout, and the overlap-add stitch
# can't drift from what we verified.
_SCRIPTS_DIR = Path("/home/kim/Projects/SAO/stable-audio-3/scripts")
sys.path.insert(0, str(_SCRIPTS_DIR))
from decode_onnx import decode_chunked_onnx, pick_providers  # noqa: E402
from dit_onnx_infer import schedule, load_cond, SR, LATENT_DIM, DS  # noqa: E402

# local_add_cond = cat(inpaint_mask[1], inpaint_masked_input[256]) = 257 ch; all
# zeros for text-to-audio. Reused EXACTLY from dit_onnx_infer.py.
LOCAL_ADD_DIM = 257

# Globals set in main(), read-only afterwards.
_dit = None
_dec = None
_cond_dir: Path | None = None
_frames = 256
_chunk = 128
_overlap = 16
_fp16 = True
_model_name = "?"
_active_ep = "?"
_lock = threading.Lock()


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """audio [1,2,N] or [2,N] float32 → stereo int16 WAV bytes."""
    a = audio[0] if audio.ndim == 3 else audio
    a = np.clip(a, -1.0, 1.0)
    i16 = (a * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.T.flatten().tobytes())
    return buf.getvalue()


def _resolve_npz(name: str) -> Path:
    """Accept either a bare filename (looked up in --cond-dir) or an absolute/
    relative path. Raises FileNotFoundError so the handler returns 404."""
    p = Path(name)
    if not p.is_absolute() and _cond_dir is not None and not p.exists():
        p = _cond_dir / name
    if not p.exists():
        raise FileNotFoundError(f"npz not found: {name}")
    return p


def _generate(cond_npz: str, uncond_npz: str, steps: int, cfg: float, seed: int) -> bytes:
    """Run the DiT rectified-flow sampler with the precached cond/uncond npz, then
    decode → WAV. Replicates the CFG two-call loop from dit_onnx_infer.py exactly."""
    c_cross, c_mask, c_glob = load_cond(_resolve_npz(cond_npz))
    u_cross, u_mask, u_glob = load_cond(_resolve_npz(uncond_npz))
    T = _frames

    local_add = np.zeros((1, LOCAL_ADD_DIM, T), np.float32)

    def dit_v(x, cross1, mask1, glob1, t_cur):
        return _dit.run(None, {
            "x": x, "t": np.full((1,), t_cur, np.float32),
            "cross_attn_cond": cross1[None], "cross_attn_cond_mask": mask1[None],
            "global_embed": glob1[None], "local_add_cond": local_add})[0]

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, LATENT_DIM, T)).astype(np.float32)     # sigma_max=1.0
    sig = schedule(steps, T)

    for i in range(steps):
        t_cur, dt = float(sig[i]), float(sig[i + 1] - sig[i])
        v_cond = dit_v(x, c_cross, c_mask, c_glob, t_cur)
        if cfg == 1.0:
            v = v_cond
        else:
            v_unc = dit_v(x, u_cross, u_mask, u_glob, t_cur)
            v = v_unc + cfg * (v_cond - v_unc)                        # CFG (velocity space)
        x = x + dt * v

    audio = decode_chunked_onnx(_dec, x, _chunk, _overlap, DS)
    return wav_bytes(audio, SR)


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _wav(self, raw: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/status":
            return self._json({"ok": True, "backend": "onnx-migraphx-dit",
                               "model": _model_name, "active_ep": _active_ep,
                               "frames": _frames, "fp16": _fp16,
                               "sample_rate": SR, "cond_dir": str(_cond_dir),
                               "decode_chunk": _chunk, "decode_overlap": _overlap})
        if u.path == "/generate":
            # Hold the lock around the WHOLE generate (DiT loop + decode share the
            # GPU); try/finally so the lock is always released, even on error.
            _lock.acquire()
            try:
                raw = _generate(q["cond"], q["uncond"],
                                int(q.get("steps", 8)), float(q.get("cfg", 6.0)),
                                int(q.get("seed", 42)))
                try:
                    return self._wav(raw)
                except BrokenPipeError:
                    pass  # client disconnected during the long write — not an error
            except (FileNotFoundError, KeyError) as e:
                return self._json({"error": str(e)}, 404)
            except Exception as e:
                return self._json({"error": str(e)}, 500)
            finally:
                _lock.release()
        return self._json({"error": "unknown endpoint"}, 404)

    def log_message(self, *a):
        pass


def _make_providers(provider: str, fp16: bool):
    """pick_providers() + the fp16 MIGraphX EP option when requested."""
    providers = pick_providers(provider)
    if fp16:
        providers = [("MIGraphXExecutionProvider", {"migraphx_fp16_enable": "1"})
                     if p == "MIGraphXExecutionProvider" else p for p in providers]
    return providers


def main():
    global _dit, _dec, _cond_dir, _frames, _chunk, _overlap, _fp16, _model_name, _active_ep
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit-onnx", required=True, type=Path, help="exported SA3 DiT .onnx")
    ap.add_argument("--decoder-onnx", required=True, type=Path, help="exported SAME decoder .onnx")
    ap.add_argument("--frames", type=int, required=True,
                    help="T — MUST match the DiT export's length rung")
    ap.add_argument("--cond-dir", default="/home/kim/Projects/latents_sa3",
                    help="dir for resolving bare cond/uncond npz filenames")
    ap.add_argument("--decode-chunk", type=int, default=128, help="MUST match the decoder export L")
    ap.add_argument("--decode-overlap", type=int, default=16, help=">= decoder receptive field")
    ap.add_argument("--provider", default="migraphx", help="migraphx | rocm | cpu")
    ap.add_argument("--no-fp16", action="store_true",
                    help="force the fp32 MIGraphX EP (default is fp16 so the DiT + decoder "
                         "co-resident on the 16 GB card; fp32-both thrashes VRAM)")
    ap.add_argument("--port", type=int, default=7894)
    args = ap.parse_args()

    _cond_dir = Path(args.cond_dir)
    _frames = args.frames
    _chunk, _overlap = args.decode_chunk, args.decode_overlap
    _fp16 = not args.no_fp16
    _model_name = args.dit_onnx.name

    import onnxruntime as ort
    providers = _make_providers(args.provider, _fp16)
    plain = [p[0] if isinstance(p, tuple) else p for p in providers]
    print(f"[ort] providers: {plain}" + ("  (fp16 EP)" if _fp16 else "  (fp32 EP)"))
    print(f"[ort] compiling DiT + decoder (one-time MIGraphX AOT here, ~min; "
          f"fp16 so both fit on 16 GB) ...")
    t0 = time.time()
    _dit = ort.InferenceSession(str(args.dit_onnx), providers=providers)
    _dec = ort.InferenceSession(str(args.decoder_onnx), providers=providers)
    dt = time.time() - t0
    dit_ep, dec_ep = _dit.get_providers()[0], _dec.get_providers()[0]

    # If a GPU EP was requested with fp16 options and either session fell back to CPU,
    # retry on the bare GPU EP (no provider options) — catches builds that reject
    # migraphx_fp16_enable while still supporting the MIGraphX EP itself.
    if _fp16 and "migraphx" in args.provider.lower() and ("CPU" in dit_ep or "CPU" in dec_ep):
        print("[ort] WARNING: fp16 EP yielded a CPU fallback (DiT EP={dit_ep}, decoder EP={dec_ep}). "
              "Retrying both sessions on the bare GPU EP (no fp16 option) ...")
        bare_providers = pick_providers(args.provider)
        if "CPU" in dit_ep:
            _dit = ort.InferenceSession(str(args.dit_onnx), providers=bare_providers)
            dit_ep = _dit.get_providers()[0]
        if "CPU" in dec_ep:
            _dec = ort.InferenceSession(str(args.decoder_onnx), providers=bare_providers)
            dec_ep = _dec.get_providers()[0]
        print(f"[ort] after retry: DiT EP {dit_ep}, decoder EP {dec_ep}")

    _active_ep = dit_ep
    print(f"[ort] sessions ready in {dt:.0f}s  (DiT EP {dit_ep}, decoder EP {dec_ep})")
    if "migraphx" in args.provider.lower() and ("CPU" in dit_ep or "CPU" in dec_ep):
        print("WARNING: requested MIGraphX but DiT EP={dit_ep} and decoder EP={dec_ep} — "
              "at least one session is still on CPU. This is NOT a GPU run. "
              "Check that onnxruntime_migraphx is installed in this venv.")
    print(f"[serve] http://localhost:{args.port}  frames={_frames} fp16={_fp16}  "
          f"(cond npz: {_cond_dir})")
    ThreadingHTTPServer(("localhost", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

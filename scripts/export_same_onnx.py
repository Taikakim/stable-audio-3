#!/usr/bin/env python
"""export_same_onnx.py — export the SAME (Stable Audio 3) autoencoder to ONNX,
one *fixed-size latent chunk* at a time, for AMD inference via ONNX Runtime +
the MIGraphX execution provider.

Why fixed-chunk (read this before changing it)
-----------------------------------------------
SAME's encoder/decoder are transformer-based (`TransformerResamplingBlock`) and
fold the sequence into chunks whose *count depends on the input length*
(`x.shape[1] // chunk_size`, modulo padding, sliding-window attention). A single
dynamic-axis ONNX graph over arbitrary T is therefore fragile. But the model is
*built* to be chunked: `AudioAutoencoder.decode_audio(chunked=True)` already
calls `decode()` on fixed-size chunks and stitches the outputs with overlap-add.
So we export the unit the model already runs internally — one fixed chunk — and
keep the chunk loop + overlap stitching on the host (Python/Rust). At a fixed
length every length-dependent reshape collapses to a constant and the graph is
clean.

What it does
------------
- Loads `AutoencoderModel.from_pretrained(<model>)` (default `same-l`) on CPU.
- Folds `weight_norm` into the conv weights (matches the cgisky/MNN "WeightNorm
  pre-fusion" that keeps fp16/int8 decode numerically stable).
- Forces SDPA attention (`SA3_DISABLE_FLASH_ATTN=1`) — flash-attn is not
  ONNX-exportable; SDPA is.
- Exports either the decoder (`latents[1,C,L] -> audio[1,2,L*ds]`) or the
  encoder (`audio[1,2,L*ds] -> latents[1,C,L]`) at a fixed chunk length L.
- Optionally validates the ONNX graph against torch on CPU (max-abs-diff +
  correlation) and converts to fp16.

Runs entirely on CPU — no GPU needed for export or validation. The only GPU
step is *deploying* the resulting .onnx with the MIGraphX EP (printed at the
end).

Usage
-----
    # in the SA3 venv:
    python scripts/export_same_onnx.py --component decoder --chunk-latents 128 --validate
    python scripts/export_same_onnx.py --component encoder --chunk-latents 128 --validate
    # quick CPU smoke (small chunk, fast):
    python scripts/export_same_onnx.py --component decoder --chunk-latents 32 --validate
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Must be set before importing torch / stable_audio_3 so attention takes the
# SDPA path (exportable) instead of flash-attn (not exportable).
os.environ.setdefault("SA3_DISABLE_FLASH_ATTN", "1")
# Keep export/validation off the GPU regardless of what's reserved on the box.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# weight_norm folding
# ---------------------------------------------------------------------------
def fold_weight_norm(module: nn.Module) -> int:
    """Remove weight_norm parametrization from every submodule, folding g·v/‖v‖
    into a single `weight`. Handles both the legacy (`torch.nn.utils.weight_norm`,
    which SAME uses via `WNConv1d`) and the new `parametrize` API. Returns the
    number of modules folded."""
    from torch.nn.utils import remove_weight_norm
    try:
        from torch.nn.utils.parametrize import (
            is_parametrized,
            remove_parametrizations,
        )
    except Exception:  # very old torch
        is_parametrized = None

    folded = 0
    for m in module.modules():
        # New parametrize API
        if is_parametrized is not None and is_parametrized(m, "weight"):
            try:
                remove_parametrizations(m, "weight", leave_parametrized=True)
                folded += 1
                continue
            except Exception:
                pass
        # Legacy weight_norm hook (leaves a `weight_g`/`weight_v` pair)
        if hasattr(m, "weight_g") or hasattr(m, "weight_v"):
            try:
                remove_weight_norm(m)
                folded += 1
            except Exception:
                pass
    return folded


def disable_checkpointing(module: nn.Module) -> None:
    """Gradient checkpointing must be off for a clean export (and is pointless at
    inference)."""
    for m in module.modules():
        if hasattr(m, "checkpointing"):
            m.checkpointing = False


def force_exportable_attention() -> None:
    """Route sliding-window attention away from FlexAttention.

    With flash-attn off, SAME's sliding-window layers default to FlexAttention
    (a torch.compile-only HOP). It traces but the ONNX exporter cannot translate
    its mask_mod subgraph (fails on the `bitwise_and` graph output). Disabling
    Flex at the module level drops the cascade into `_sliding_window_chunked_halo_sdpa`
    — math-equivalent windowed attention built from plain masked SDPA, which
    exports cleanly. `apply_attn` reads these module globals at call time, so
    patching the module object is enough."""
    import stable_audio_3.models.transformer as T
    T.flex_attention_available = False
    T.flex_attention_compiled = None
    print("[prep] FlexAttention disabled -> windowed attention uses masked-SDPA fallback")


# ---------------------------------------------------------------------------
# Export wrappers — one fixed chunk, no host-side chunk loop inside the graph
# ---------------------------------------------------------------------------
class DecodeChunk(nn.Module):
    """latents [B, C, L] -> audio [B, 2, L*downsampling_ratio]."""
    def __init__(self, autoencoder):
        super().__init__()
        self.ae = autoencoder

    def forward(self, latents):
        return self.ae.decode(latents)


class EncodeChunk(nn.Module):
    """audio [B, 2, L*downsampling_ratio] -> latents [B, C, L].

    NOTE: parity here is against `AudioAutoencoder.encode` (the raw single-pass
    encode incl. the SoftNormBottleneck `x / running_std`). Cross-check against
    `AutoencoderModel.encode` / the values in `latents_sa3/*.npy` before trusting
    it for re-encoding — the wrapper used to produce the stored latents adds
    automatic resample + length padding that we deliberately skip here (a fixed
    chunk is already a whole multiple of the downsampling ratio)."""
    def __init__(self, autoencoder):
        super().__init__()
        self.ae = autoencoder

    def forward(self, audio):
        return self.ae.encode(audio, return_info=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--component", choices=["decoder", "encoder"], default="decoder")
    ap.add_argument("--model", default="same-l",
                    help="AutoencoderModel id (default: same-l, the latents_sa3 AE)")
    ap.add_argument("--chunk-latents", type=int, default=128,
                    help="fixed chunk length in latent frames (decode/encode unit). "
                         "Match this at runtime. 128 ≈ 11.9 s; 256 ≈ 23.8 s.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .onnx path (default: same_<component>_L<chunk>.onnx)")
    ap.add_argument("--opset", type=int, default=18,
                    help="ONNX opset. The dynamo exporter implements these ops at "
                         ">=18; requesting 17 forces a lossy down-convert that emits "
                         "an invalid Split(num_outputs). Keep >=18.")
    ap.add_argument("--validate", action="store_true",
                    help="run the exported graph in ORT (CPU) and diff vs torch")
    ap.add_argument("--fp16", action="store_true",
                    help="also write an fp16 copy (needs onnxconverter_common)")
    ap.add_argument("--device", default="cpu",
                    help="torch device for tracing/validation (default cpu — no GPU needed)")
    args = ap.parse_args()

    from stable_audio_3 import AutoencoderModel

    print(f"[load] AutoencoderModel.from_pretrained({args.model!r}) on {args.device} ...")
    model = AutoencoderModel.from_pretrained(args.model, device=args.device)
    ae = model.autoencoder.eval().to(args.device).float().requires_grad_(False)

    sr = int(model.sample_rate)
    ds = int(ae.downsampling_ratio)
    latent_dim = int(getattr(ae, "latent_dim", 256))
    L = args.chunk_latents
    n_audio = L * ds
    print(f"[info] sample_rate={sr}  downsampling_ratio={ds}  latent_dim={latent_dim}")
    print(f"[info] chunk: {L} latents  <->  {n_audio} samples  ({n_audio / sr:.2f} s)")

    n_folded = fold_weight_norm(ae)
    disable_checkpointing(ae)
    force_exportable_attention()
    print(f"[prep] folded weight_norm in {n_folded} modules; flash-attn disabled "
          f"(SA3_DISABLE_FLASH_ATTN={os.environ.get('SA3_DISABLE_FLASH_ATTN')})")

    if args.component == "decoder":
        wrap = DecodeChunk(ae).eval()
        dummy = torch.randn(1, latent_dim, L, device=args.device)
        input_names, output_names = ["latents"], ["audio"]
        default_name = f"same_decoder_L{L}.onnx"
    else:
        wrap = EncodeChunk(ae).eval()
        dummy = torch.randn(1, 2, n_audio, device=args.device)
        input_names, output_names = ["audio"], ["latents"]
        default_name = f"same_encoder_L{L}.onnx"

    out = args.out or Path(default_name)

    # Reference forward (also confirms the model runs at this shape before export)
    print(f"[torch] reference forward {tuple(dummy.shape)} ...")
    t0 = time.time()
    with torch.no_grad():
        ref = wrap(dummy)
    print(f"[torch] -> {tuple(ref.shape)} in {time.time() - t0:.1f}s")

    # Fully static shapes — MIGraphX prefers static; we loop fixed chunks on host.
    print(f"[onnx] exporting opset {args.opset} -> {out}")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrap, (dummy,), str(out),
            input_names=input_names, output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,   # static: batch=1, fixed chunk length
        )
    print(f"[onnx] exported in {time.time() - t0:.1f}s  ({out.stat().st_size / 1e6:.1f} MB)")

    if args.validate:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[validate] onnxruntime not installed — skipping. "
                  "pip install onnxruntime")
        else:
            print("[validate] running ORT CPUExecutionProvider ...")
            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            ort_out = sess.run(None, {input_names[0]: dummy.cpu().numpy().astype(np.float32)})[0]
            r = ref.cpu().numpy().astype(np.float32)
            o = ort_out.astype(np.float32)
            max_abs = float(np.abs(r - o).max())
            mean_abs = float(np.abs(r - o).mean())
            # Scale-aware: decoder output is audio in [-1,1] but encoder output is
            # latents with a wider range, so judge by relative error + cosine, not
            # an absolute audio-calibrated threshold.
            scale = float(np.abs(r).max()) or 1.0
            rel_max = max_abs / scale
            denom = (np.linalg.norm(r.ravel()) * np.linalg.norm(o.ravel())) or 1.0
            corr = float(np.dot(r.ravel(), o.ravel()) / denom)
            print(f"[validate] max|Δ|={max_abs:.3e} (rel {rel_max:.2%})  "
                  f"mean|Δ|={mean_abs:.3e}  cos={corr:.6f}")
            ok = corr > 0.9999 and rel_max < 0.05
            print("[validate] ok" if ok else
                  "[validate] WARNING: larger diff than expected — inspect before trusting")

    if args.fp16:
        try:
            from onnxconverter_common import float16
            import onnx
        except ImportError:
            print("[fp16] onnxconverter_common/onnx not installed — skipping. "
                  "pip install onnxconverter-common onnx")
        else:
            fp16_out = out.with_name(out.stem + "_fp16.onnx")
            m = float16.convert_float_to_float16(onnx.load(str(out)), keep_io_types=True)
            onnx.save(m, str(fp16_out))
            print(f"[fp16] wrote {fp16_out} ({fp16_out.stat().st_size / 1e6:.1f} MB)")

    print("\n[next] deploy on the 9070 XT with ONNX Runtime + MIGraphX EP:")
    print("  sess = ort.InferenceSession(")
    print(f"      '{out.name}',")
    print("      providers=['MIGraphXExecutionProvider', 'CPUExecutionProvider'])")
    print("  # host-side chunk loop = stable_audio_3 AudioAutoencoder.decode_audio():")
    print(f"  #   slice latents into [1,{latent_dim},{L}] chunks (hop = {L}-overlap),")
    print("  #   run sess per chunk, overlap-add stitch. overlap >= receptive field.")
    print("  # diff the stitched audio vs the torch /decode to set overlap.")


if __name__ == "__main__":
    main()

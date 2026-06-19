# SAME → ONNX for AMD inference (ORT + MIGraphX)

Export the SAME autoencoder (the SA3 `same-l` / `same-s` VAE) to ONNX so the
decoder/encoder can run on AMD via **ONNX Runtime + the MIGraphX EP** — a
low-VRAM inference path that doesn't need the full torch/ROCm stack and can run
alongside a training job. Motivated by the cgisky `stable-audio-3-rs` MNN port
(which proves SAME exports cleanly, but is CUDA/Windows-only).

**Status (2026-06-20):** decoder **and** encoder export + CPU validation pass.
GPU/MIGraphX deployment + the overlap-stitch seam test are the remaining items.

Scripts: `scripts/export_same_onnx.py` (export+validate, CPU-only),
`scripts/decode_onnx.py` (host chunk-loop runner over the exported decoder).

## Why fixed-chunk, not a dynamic-length graph

SAME's encoder/decoder are transformer-based (`TransformerResamplingBlock`) and
fold the sequence into chunks whose count depends on input length
(`x.shape[1] // chunk_size`, modulo padding, sliding-window attention). A single
dynamic-axis ONNX graph over arbitrary T is therefore fragile. But the model is
*built* to be chunked: `AudioAutoencoder.decode_audio(chunked=True)` already
calls `decode()` on fixed-size chunks and stitches with overlap-add. So we export
**one fixed chunk** (`decode`: `[1,256,L] → [1,2,L·4096]`) and keep the chunk
loop + overlap-add on the host. At a fixed length every length-dependent reshape
collapses to a constant and the graph is clean. The varlen "problem" disappears
because we never export varlen.

## The two gotchas that cost iterations

1. **`SA3_DISABLE_FLASH_ATTN=1` is necessary but NOT sufficient.** With flash off,
   SAME's sliding-window layers fall back to **FlexAttention** (a `torch.compile`
   higher-order op). It *traces* fine, but the dynamo ONNX exporter cannot
   translate its `mask_mod` subgraph — it dies with
   `ConversionError: ... translating node return bitwise_and` /
   `TypeError: 'Node' object is not iterable`. Fix: disable Flex at the module
   level so the cascade drops to `_sliding_window_chunked_halo_sdpa`
   (math-equivalent windowed attention from plain masked SDPA, which exports):
   ```python
   import stable_audio_3.models.transformer as T
   T.flex_attention_available = False
   T.flex_attention_compiled = None
   ```
   (`apply_attn` reads these module globals at call time — patching the module is
   enough. `export_same_onnx.py:force_exportable_attention()` does this.)

2. **opset must be ≥ 18.** The dynamo exporter implements these ops at opset 18;
   requesting 17 forces a lossy down-convert that emits an invalid
   `Split(num_outputs)` (`num_outputs` is an opset-18 attribute) and ORT rejects
   the model with `INVALID_GRAPH: Unrecognized attribute: num_outputs for
   operator Split`. Export at 18 directly.

Other prep: fold `weight_norm` into the conv weights (`remove_weight_norm` /
`remove_parametrizations` — only 2 modules in SAME, it's transformer-heavy);
disable gradient checkpointing; export static shapes (MIGraphX prefers static).

## Dependencies

The modern `torch.onnx.export` (dynamo) needs `onnxscript` + `onnx`. On the SA3
venv this is purely additive — it does **not** bump the pinned ROCm torch or
numpy (verified via `uv pip install --dry-run`). For validation/runtime add
`onnxruntime`; for `--fp16`, `onnxconverter-common`.

```bash
uv pip install onnxscript onnxruntime          # export + validate
uv pip install onnxconverter-common onnx       # only for --fp16
```

## Validation results (CPU, ORT vs torch)

| Component | Chunk | mean\|Δ\| | cos | note |
|---|---|---|---|---|
| decoder | 32  | 7.2e-4 | 0.99988 | max\|Δ\| 3.0e-2 (single-sample outlier) |
| decoder | 128 | 4.8e-4 | 0.99998 | max\|Δ\| 1.5e-2 |
| encoder | 128 | 8.9e-3 | 0.999996 | max\|Δ\| 0.167 = ~1-2% rel (latents have wider range than audio; judge by cos / relative error, not an audio-calibrated absolute threshold) |

## Usage

```bash
# export (CPU; runs while the GPU is busy)
python scripts/export_same_onnx.py --component decoder --chunk-latents 128 --validate
python scripts/export_same_onnx.py --component encoder --chunk-latents 128 --validate
python scripts/export_same_onnx.py --component decoder --chunk-latents 128 --fp16   # optional

# decode latents with the ONNX decoder (auto-picks MIGraphX > ROCm > CPU)
python scripts/decode_onnx.py --onnx same_decoder_L128.onnx \
    --crop 000000 --chunk-latents 128 --overlap 16 --out out.wav
# verify the overlap-add seam vs torch (needs the model; sets a safe --overlap):
python scripts/decode_onnx.py --onnx same_decoder_L128.onnx --npy <latent.npy> \
    --chunk-latents 128 --overlap 16 --provider cpu --compare-torch
```

## Caveats / open items

- **Seam not yet validated.** Single-chunk graphs match torch; the overlap-add
  stitch across chunks (`decode_onnx.py`) needs a `--compare-torch` run to set
  `--overlap` ≥ the decoder's receptive field (raise it until the seam diff is
  negligible).
- **Encoder parity.** The export wrapper calls `AudioAutoencoder.encode` directly
  and skips the auto-resample/length-pad that `AutoencoderModel.encode` does to
  produce the stored `latents_sa3/*.npy`. Cross-check encoder output against a
  real `.npy` before using it for re-encoding.
- **INT8 is a stretch on MIGraphX** (coverage is uneven). FP16 is the safe target.
- These are `same-l` exports (the `latents_sa3` AE). `--model same-s` for the
  Small/CPU variant.

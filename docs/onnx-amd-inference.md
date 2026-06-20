# SAME → ONNX for AMD inference (ORT + MIGraphX)

Export the SAME autoencoder (the SA3 `same-l` / `same-s` VAE) to ONNX so the
decoder/encoder can run on AMD via **ONNX Runtime + the MIGraphX EP** — a
low-VRAM inference path that doesn't need the full torch/ROCm stack and can run
alongside a training job. Motivated by the cgisky `stable-audio-3-rs` MNN port
(which proves SAME exports cleanly, but is CUDA/Windows-only).

**Status (2026-06-20):** **GPU-VERIFIED.** Decoder runs 100% on the MIGraphX EP
(zero CPU fallback), numerically identical to torch (cos=0.999998), RTF ~39×
post-compile → ONNX inference on AMD works. Operational catch: a ~9-min MIGraphX
AOT compile per session (caching not exposed in this ORT build — compile once in a
long-lived server). Multi-chunk overlap-add **seam** test still TODO.

## GPU verification (RX 9070 XT, mir venv onnxruntime_migraphx 1.23.2)

```
[ort] active: MIGraphXExecutionProvider
[run] -> audio (1,2,131072)  in 0.08s   RTF=38.6x          # L32, post-compile
[placement] MIGraphXExecutionProvider 100.0%  ✅ no CPU fallback
MIGraphX-GPU vs torch-CPU:  max|Δ|=2.8e-4  mean|Δ|=4.5e-5  cos=0.999998
```

- **EP availability:** only the **mir venv** has the MIGraphX EP (`onnxruntime_migraphx`
  1.23.2). The SA3 venv's `onnxruntime` (1.27, PyPI) is **CPU-only** (`['Azure','CPU']`).
  `decode_onnx.py` needs no torch (unless `--compare-torch`), so run it from the mir venv.
- **The ~9-min AOT compile is the only real catch.** It is CPU-bound (pegs all cores;
  GPU just holds the 1.7 GB of weights until kernels are built). It is NOT exhaustive
  tuning (tried `MIGRAPHX_DISABLE_EXHAUSTIVE_TUNE=1`) and NOT chunk size (same at L32 and
  L128) — MIGraphX is simply slow to compile the masked-SDPA-heavy graph. **Caching:** ORT's
  compiled-model cache options (`migraphx_save/load_compiled_model`, `_model_name`,
  `_model_path`) are **all rejected by this `onnxruntime_migraphx` 1.23.2 build** (`Unknown
  provider option`), and ORT then silently falls back to CPU — `decode_onnx.py --cache-dir`
  detects that and retries on the bare GPU EP, so caching is effectively a no-op here. The
  practical mitigation is **a long-lived server that compiles once at boot** (the
  `latent_server_sa3.py` pattern — per-request cost is then nil), or a newer ORT-ROCm build
  that exposes the cache options. `--ep-fp16` (`migraphx_fp16_enable`) IS supported.
- **Placement reports "1 op type" at 100%** because MIGraphX fuses the entire graph into a
  single compiled program — that IS full-graph GPU execution, not a one-op model.

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
# AMD coverage check: which EP actually ran each node (detects silent CPU fallback):
python scripts/decode_onnx.py --onnx same_decoder_L128.onnx --crop 000000 \
    --provider migraphx --report-placement --compare-torch

# benchmark vs the stock torch decode (latency/RTF, VRAM, quality):
python scripts/bench_same_onnx.py --crop 000000 \
    --onnx-fp32 same_decoder_L128.onnx --onnx-fp16 same_decoder_L128_fp16.onnx \
    --backends torch,onnx-migraphx,onnx-migraphx-fp16 \
    --lengths 128,512,1024,4096 --chunk-latents 128 --overlap 16
```

**EP availability:** the plain PyPI `onnxruntime` is CPU-only. For the MIGraphX/ROCm
EP run `decode_onnx.py`/`bench_same_onnx.py` from a venv that has it (mir's
`onnxruntime_migraphx`) or `uv pip install onnxruntime-rocm`. `decode_onnx.py` is
pure numpy/ORT/soundfile, so it runs in the mir venv (only `--compare-torch` needs
the model). "ONNX inference on AMD" is verified only when `--report-placement` shows
the heavy ops on the GPU EP (not CPU fallback) AND `--compare-torch` cos ≈ 0.9999.

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

# SAME → ONNX for AMD inference (ORT + MIGraphX)

Export the SAME autoencoder (the SA3 `same-l` / `same-s` VAE) to ONNX so the
decoder/encoder can run on AMD via **ONNX Runtime + the MIGraphX EP** — a
low-VRAM inference path that doesn't need the full torch/ROCm stack and can run
alongside a training job. Motivated by the cgisky `stable-audio-3-rs` MNN port
(which proves SAME exports cleanly, but is CUDA/Windows-only).

**Status (2026-06-20):** **GPU-VERIFIED.** Decoder runs 100% on the MIGraphX EP
(zero CPU fallback), numerically identical to torch (cos=0.999998), RTF ~39×
post-compile → ONNX inference on AMD works. Multi-chunk overlap-add **seam test
passes** (overlap=16: ONNX-chunked vs torch-unchunked cos=0.999997, no boundary
spikes). Operational catch: a ~9-min MIGraphX AOT compile per session (caching not
exposed in this ORT build — compile once in a long-lived server).

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
**Deployment:** `mir/scripts/latent_server_onnx.py` is a low-VRAM decode server (mir
venv, ~2 GB GPU) that compiles the ONNX once at boot then serves `/decode /mix /source`
— a drop-in alternative to the torch player `latent_server_sa3.py` for the explorer
(point the viewer at it with `SA3_PLAYER_PORT=7893`).

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

- **Seam validated (2026-06-20, overlap=16, 512-latent/4-chunk, CPU).** torch's own
  chunked≈unchunked (cos=1.000004, max|Δ|=6.7e-4) confirms overlap=16 ≥ receptive field;
  ONNX-chunked vs torch-unchunked cos=0.999997; per-boundary local max|Δ| (1.5e-4…3.5e-4)
  is the same order as everywhere else → no seam artefacts. The stitch is EP-independent,
  so this also covers the MIGraphX path (per-chunk already cos=0.999998). Raise `--overlap`
  only if a future export changes the decoder's receptive field.
- **Encoder parity.** The export wrapper calls `AudioAutoencoder.encode` directly
  and skips the auto-resample/length-pad that `AutoencoderModel.encode` does to
  produce the stored `latents_sa3/*.npy`. Cross-check encoder output against a
  real `.npy` before using it for re-encoding.
- **INT8 is a stretch on MIGraphX** (coverage is uneven). FP16 is the safe target.
- These are `same-l` exports (the `latents_sa3` AE). `--model same-s` for the
  Small/CPU variant.

---

# DiT → ONNX (full text→audio on AMD)

The autoencoder is the cheap end; the **DiT** is the per-step hot loop. With text
**precached** (T5-Gemma run offline) the DiT export is clean and the whole
generation can run on MIGraphX. Tooling: `scripts/export_dit_onnx.py` (export) +
`scripts/dit_onnx_infer.py` (host sampler/runner, mir venv).

**Status (2026-06-23):** DiT **GPU-VERIFIED** on MIGraphX. `medium-base` L256:
export vs torch `_forward` cos=1.000000; **MIGraphX vs CPU/torch cos=1.000000, 100%
on the MIGraphX EP (zero CPU fallback), 233 ms/call** (warm, batch=1); ~12-min one-time
AOT compile. The full pipeline (sampler + CFG + decode → WAV) runs end-to-end. The 1.4B
DiT exports with the *same* recipe as the AE (flash-off + opset 18) and was structurally
friendlier (no chunk-folding, global self-attn → plain SDPA). **Only a real-prompt
audio-vs-torch comparison remains — it needs the cached T5-Gemma embeddings** (t5gemma
not downloaded; prompts ARE in the `latents_sa3` json). The DiT is exported static
**batch=1** → CFG runs as two batch-1 calls/step; a batch=2 export would make CFG one
call/step (efficiency option).

## Why a ladder of fixed lengths

The DiT does full-sequence self-attention every step → can't be chunked; the whole
`T` is one graph. It has no length-dependent control flow, so a *dynamic-axis*
export would trace — but MIGraphX compiles per static shape and this build can't
cache, so dynamic = a fresh ~min compile per distinct T. So: export a **ladder**
`T ∈ {256, 512, 1024, 2048, 4096}` (= 23.8/47.5/95/190/380 s, power-of-2, matches
the `latents_sa3` grid), compile each once, pad requests up to the smallest rung.

## Export target + recipe

Export `DiffusionTransformer._forward` — the **CFG-free core**. The export builds
the **DiT only** (`DiTWrapper(**diffusion.config)` + the `model.model.*` weights
from the cached `medium-base` safetensors) — **no T5-Gemma** (precached, and it
isn't even downloaded). Inputs (B=1, two fixed dims `T` and text `seq`):
`x[1,256,T]`, `t[1]`, `cross_attn_cond[1,seq,768]`, `cross_attn_cond_mask[1,seq]`,
`global_embed[1,768]`. Output: velocity `[1,256,T]`. Runs on CPU (RAM);
artefact ≈ 8 MB graph + ~5.5 GB `.onnx.data` fp32 per rung (**export fp16 to halve**).

```bash
for L in 256 512 1024 2048 4096; do
  python scripts/export_dit_onnx.py --model medium-base --frames $L --text-seq 128 --validate
done
```

## Host loop (rectified-flow, CFG)

CFG collapses to velocity space (rectified-flow, vanilla): per step, one DiT call on
the stacked `[cond; uncond]` batch, then `v = v_uncond + cfg·(v_cond − v_uncond)`;
`x += dt·v`. Schedule = `linspace(1,0,steps+1)` warped by the SD3/Flux time-shift —
**identity for medium-base** (`alpha_min=alpha_max=1.0`). Final `z0 → same_decoder
ONNX (chunk-loop) → wav`. All numpy + ORT; no `stable_audio_3` at runtime.

```bash
python scripts/dit_onnx_infer.py --dit-onnx dit_medium-base_L256.onnx \
    --decoder-onnx same_decoder_L128.onnx --cond cond.npz --uncond uncond.npz \
    --frames 256 --steps 8 --cfg-scale 6.0 --provider migraphx --out gen.wav
```

## Precache contract (produced offline in the SA3 venv)

Per (prompt, duration), emit a `cond.npz` and an `uncond.npz` (negative/empty prompt)
each holding the **assembled DiT conditioning** so the runtime is pure numpy/ORT:
- `cross_attn_cond` `float32 [seq,768]` — T5-Gemma `b-b-ul2` embeddings, **padded to a
  fixed `seq`** (matching `--text-seq`)
- `cross_attn_mask` `[seq]` (bool/int)
- `global_embed` `float32 [768]` — NumberConditioner over `seconds_start`/`seconds_total`

## Open items
- GPU: MIGraphX compile (likely > the AE's 9 min at 1.4B) + end-to-end **audio vs torch
  `generate()`** correctness (the real check). Per-length compile = use a long-lived server.
- **fp16 export** to cut the 5.5 GB/rung (and the ladder is 5×).
- Confirm `medium-base` has no active `prepend_cond`/`input_concat` (the wrapper only feeds
  cross_attn + global; if a model adds them, extend `DiTCore`).

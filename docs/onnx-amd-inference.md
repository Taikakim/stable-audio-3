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

**Status (2026-06-23):** DiT **GPU-VERIFIED end-to-end** for a real prompt.
`medium-base` L256: ONNX vs torch `_forward` cos=1.000000; **corrected DiT on MIGraphX
vs CPU cos=1.000000, 100% on-EP (no CPU fallback), 191 ms/call, ~13-min compile**;
**full 8-step generation (real t5gemma conditioning) ONNX z0 vs torch z0 cos=0.999944**
→ the ONNX text→audio pipeline reproduces torch on the AMD GPU. Real audio produced
(structured/low-heavy, not noise). Ladder exported: L∈{256,512,1024,2048,4096} (each
8.8 MB graph + 5.8 GB weights — they don't share weights yet, ~29 GB total; dedup TODO).

**Critical correctness fix — `local_add_cond` must be fed, not omitted.** medium-base's
DiT takes a 257-ch `local_add_cond` = cat(`inpaint_mask`[1], `inpaint_masked_input`[256]);
`local_add_cond_ids` in the config. For text-to-audio (no inpaint) it's all-zeros — but
the DiT **projects it with a bias**, so `local_add_cond=None ≠ zeros` (measured cos 0.98,
max|Δ| 0.74). The first export omitted it (and falsely validated cos=1.0 None-vs-None);
the export now takes `local_add_cond[1,257,T]` as an input and the runner feeds zeros.

**T5-Gemma:** `google/t5gemma-b-b-ul2` (0.6B, gated). It's referenced by SA3 (not bundled)
and downloaded on demand; the HF Xet protocol stalled, so fetch via plain HTTPS
(`HF_HUB_DISABLE_XET=1`) or `curl -C -` the resolve URL. Precache tool: `scripts/precache_dit_cond.py`
(prompt+duration → cond/uncond npz). Build the DiT itself with only the cached safetensors
(no t5gemma needed for export).

DiT exported static **batch=1** → CFG runs as two batch-1 calls/step; a batch=2 export
would make CFG one call/step (efficiency option).

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

## Benchmark — ONNX (MIGraphX) vs native torch (2026-06-24, GPU free)

L256, 8 steps, CFG 6, medium-base, identical seed/conditioning. `dit_loop_s` = the DiT
sampling loop only (decode excluded — the shared ONNX decode runs on CPU-ORT in the torch
SA3 venv vs MIGraphX in the onnx mir venv, so full `gen_s` isn't comparable; the loop is).

| Metric | torch (cuda fp16, eager/SDPA) | ONNX fp16 (MIGraphX) |
|---|---|---|
| **DiT loop (16 calls)** | **0.707 s** (44 ms/call, RTF 33.6×) | **2.314 s** (144 ms/call, RTF 10.3×) |
| Resident VRAM | 3.1 GB working set (full StableAudioModel load ~9.6 GB) | **3.8 GB, no torch stack** |
| z0 vs torch | — | **cos 0.999308** |
| AOT compile | none | ~14 min DiT + ~24 min decoder (one-time/session) |

**Verdict: the ONNX→MIGraphX port is a VRAM / deployment win, NOT a speed win.** Eager torch
is **~3.3× faster** per DiT call (MIGraphX's compiled graph doesn't beat torch's tuned
rocBLAS/MIOpen kernels here; torch + CK flash-attn would widen it). Quality is identical
(z0 cos 0.9993). What ONNX buys: **3.8 GB in a lightweight ORT process with zero torch/
ROCm-torch dependency** — low-VRAM inference that coexists with a training run, one portable
graph. fp16 MIGraphX is ~25 % faster than fp32 (144 vs 191 ms/call). (The bench's auto VRAM
column reads 0.00 — `_rocm_used_bytes` CSV-parse quirk, reviewer Minor; the resident figures
above are from `torch.cuda.max_memory_allocated` and the fp16 file sizes.)

## Performance + open items
- **DiT-only RTF ≈ 7.8×** (L256: 16 calls × 191 ms fp32 / 144 ms fp16 MIGraphX; torch eager 44 ms,
  RTF 33.6×). A **batch=2 export** (one call/step, cos 1.0) halves the call count — `export_dit_onnx.py --batch 2`.
- **⚠ VRAM: don't co-resident fp32 DiT + fp32 decoder on the 16 GB card.** The DiT weights are ~5.8 GB;
  with it resident, compiling the decoder pushes VRAM to ~16.5 GB (spills to GTT) and the decoder compile
  **thrashes — 31 min+ vs 9 min standalone**.
- **The fix is fp16-EXPORTED onnx files, NOT the `migraphx_fp16_enable` EP option.** Export fp16 with
  `export_dit_onnx.py --fp16` (uses `convert_float_to_float16_model_path` + external-data save — the
  >2 GB fp16 DiT overflows the in-memory protobuf path). fp16 files load directly: **DiT 2.9 GB + decoder
  0.9 GB = 3.8 GB resident**, both co-reside easily. Validated: fp16 DiT vs fp32 cos = **0.999992** (0.5 %
  rel). The `migraphx_fp16_enable` EP option does the **opposite** of helping — it loads the fp32 weights
  then quantizes at session init, so DiT+decoder co-residency **OOMs harder** (HIP out-of-memory, measured).
  Use it only for single-model fp16 speed where VRAM is ample. (Or run the two as **separate processes**.)
- **Tools added:** `bench_dit_onnx.py` (ONNX-vs-torch generation benchmark — same seed/schedule/cond/cfg,
  warm-median latency, device VRAM, z0 cosine; fairness-reviewed) and `latent_server_dit_onnx.py`
  (low-VRAM gen server: compile DiT+decoder once at boot, `/generate?cond=&uncond=` → WAV; pass fp16 files).
  Server **end-to-end smoke-tested on the CPU EP** (boot → `/status` → `/generate` → valid 23.8 s WAV).

  **Running the benchmark (needs a FREE GPU — each onnx backend AOT-compiles ~13 min and the torch
  baseline needs VRAM; co-residency with a training run OOMs):** one backend per run, in its venv —
  ```
  # torch baseline (SA3 venv):
  .venv/bin/python scripts/bench_dit_onnx.py --backend torch \
    --dit-onnx dit_medium-base_L256.onnx --decoder-onnx same_decoder_L128.onnx \
    --cond /tmp/real.cond.npz --uncond /tmp/real.uncond.npz --frames 256 --json /tmp/bench_torch.json
  # ONNX fp16 on MIGraphX (mir venv) — point at the fp16 FILES, plain EP (NOT --backend onnx-migraphx-fp16,
  # whose migraphx_fp16_enable OOMs): use the corrected runner / bench onnx-migraphx backend with fp16 onnx.
  /home/kim/Projects/mir/mir/bin/python scripts/bench_dit_onnx.py --backend onnx-migraphx \
    --dit-onnx dit_medium-base_L256_fp16.onnx --decoder-onnx same_decoder_L128_fp16.onnx \
    --cond /tmp/real.cond.npz --uncond /tmp/real.uncond.npz --frames 256 --json /tmp/bench_onnx_fp16.json
  ```
  Then compare the `vram_warm_gb` / `gen_s` / `rtf` rows and the saved `*.z0.npy` cosine. (z0-vs-torch is
  already cos 0.9999 on CPU; the benchmark adds the device latency/VRAM head-to-head.)

## Control adapters (steering) baked into the ONNX (2026-06-27)

A trained **control adapter** (`avp_sa3/sa3_control`) — decoupled cross-attention added to every
DiT block, driven by a control signal — is a pure **forward** modification (no autograd/guidance),
so unlike a LatCH guidance head it folds straight into the DiT ONNX. `export_dit_control_onnx.py`
loads an adapter checkpoint (e.g. `onset_FUSION_lr2e5_40epoch/soup_exppeak.pt`, `scalar_field=onset_density`),
wraps the 24 cross-attns with `ControlledCrossAttention`, and exports a DiT graph with two extra inputs:

    control_tokens [1, n_tokens, control_dim]   # from the scalar conditioner (host, numpy)
    gain           [1]                           # control strength (1=as trained, ~2-4 strong)

The scalar→tokens FiLM conditioner is saved as a `.cond.npz` (a 5-line numpy port — no torch at
runtime). `dit_control_onnx_infer.py` runs the CFG-control loop exactly as `sa3_control/onset_eval.py`:
**cond pass gets `enc((target−mean)/std)`, uncond pass gets zeros (the trained null)**, so the control
rides the CFG axis. The adapter reaches the wrapped modules via its module-global; threaded as explicit
forward inputs it traces cleanly (`torch.export`).

**Validated (CPU):** ONNX vs controlled-torch **cos = 1.000000** (the adapter is faithfully in the graph),
and the graph differs from the plain DiT (control is non-trivial). **End-to-end steering works:** 8-step
generation, requested onset density **3 → measured 4.88 onsets/sec, 11 → 11.15** (monotonic, calibrated;
same prompt/seed, librosa onset detection).

**MIGraphX GPU run (2026-06-27, NOT YET CONFIRMED THIS SESSION).** The control-DiT MIGraphX path was
launched but **not numerically verified** in this session — the EP is present and bound, but no on-GPU cos,
placement, steering, or RTF number was produced. What is established vs. what remains:
- **EP available + selected, node-level placement NOT observed.** mir venv `onnxruntime 1.23.2`,
  `get_available_providers() = ['MIGraphXExecutionProvider','CPUExecutionProvider']`. The runner binds
  MIGraphX (`[ort] sessions ready … (DiT EP …)`, `dit_control_onnx_infer.py:105`) but that line had not
  printed — the **~13–15 min AOT compile** (CPU-bound, ~106 % CPU / GPU idle, ~1 GB VRAM held) was still
  running when the turn ended. This runner has **no compile cache** (ORT 1.23.2 rejects the caching opts),
  so every invocation pays the full compile. **Do not kill it.**
- **cos vs torch: NOT MEASURED, and not measurable from the control runner.** `dit_control_onnx_infer.py`
  emits **no** torch-parity metric (its only `cos` is a positional-encoding `np.cos`). The sole script that
  prints cos-vs-torch is `decode_onnx.py --compare-torch`, and only on the **decoder**. That decoder check
  is runnable from the **mir venv** (it has torch 2.9.1+rocm7.2 and can import `stable_audio_3` when
  `PYTHONPATH=/home/kim/Projects/SAO/stable-audio-3` is set — the package is not installed there). Queued:
  ```bash
  PYTHONPATH=/home/kim/Projects/SAO/stable-audio-3 \
  /home/kim/Projects/mir/mir/bin/python scripts/decode_onnx.py \
    --onnx same_decoder_L128.onnx --crop 000000 --provider migraphx \
    --report-placement --compare-torch     # expect decoder cos≈0.999998, max|Δ|≈2.8e-4, 100% on-EP
  ```
  Full **DiT**-vs-torch parity is **not reproducible from the shipped runtime scripts** (the control runner
  self-reports nothing); it rests on export-time validation (CPU cos = 1.000000 above; MASTER.md §5). The
  DiT runner also has **no node-level placement check** — even a completed DiT run would prove "ran on
  MIGraphX" only via `get_providers()[0]`, which does NOT rule out per-node CPU fallback.
- **Steering: NOT MEASURED.** Plan: lo (`--onset-density 3`) vs hi (`--onset-density 12`), identical
  `--gain 3 --seed 42`, density read post-hoc with `librosa.onset.onset_detect` (onsets/sec). Control
  `.cond.npz` verified (`onset_density` mean 7.107, std 1.419, n_tokens 16, control_dim 768 → density 3
  normalizes to −2.89, density 12 to +3.45, strong span). Neither generation completed (no `steered_*.wav`).
- **RTF: NOT MEASURED.** Reference (export-time / runbook, L256 fp32): ~191 ms/call, DiT-only RTF ≈ 7.8×,
  full-pipeline MIGraphX ≈ 10.3× — **a VRAM/deployment win, not a speed win** (eager torch is ~3.3× faster
  per call). Record the runner's `[gen] {steps} steps in {X}s`; do not expect MIGraphX to beat torch.

The only reason these are unverified is wall-clock: three sequential uncached MIGraphX AOT compiles (lo +
hi + decoder, ~10–15 min each) exceed one turn. Re-run the queued commands to finish — no hard blocker
remains. (Step-0 precache needed a documented workaround: the current SA3 `precache_dit_cond.py` hits a
`cuda:0 vs cpu` device split and a `KeyError: 'inpaint_mask'` from `local_add_cond_ids`; a text-only
assemble of `cross_attn_cond_ids=['prompt','seconds_total']` + `global_cond_ids=['seconds_total']` produces
a correct `.cond.npz` — cross `(128,768)`, mask sum 14 (cond) / 0 (uncond).)

The adapter is length-agnostic — export any ladder rung with `--frames`.

**fp16 control-DiT — FIXED (host-side PE).** The bad `ConstantOfShape` came from the adapter's
`add_fractional_positions` PE (`tokens.new_zeros`). Moving the PE **host-side** removes it: export the adapter
with `position_encoding=False` (`export_dit_control_onnx.py` default) and apply the PE in numpy in the runner.
Mathematically exact (PE adds to the tokens before the K/V projection): host-PE vs trained in-graph PE
**cos=1.000000** (gated in `--validate`). `--fp16` now converts cleanly → **3.1 GB** (stamped). The npz carries
`host_pe=True` and the onnx is metadata-stamped; the runner asserts they match (guards a mismatched pair →
silent double/zero PE). The numpy PE isn't byte-equal to torch (max|Δ| ~1.2e-3 from `np.exp`/`np.float64(pi)`
vs torch — below fp16 resolution, washes out of the velocity cos).

**CPU-ONLY is the recommended eval path** (frees the GPU for training — and avoids the GPU contention that
stalls a co-resident MIGraphX compile). On the Ryzen 9 9900X: onset-steered gen **~10 s (8-step) + decode,
≈2× realtime**, steers identically to GPU (onset 11→11.19). **Pin `--threads 12`** (physical cores; ~25%
faster than 24 SMT, bandwidth-bound). **INT8 (CPU):** 1.4–1.7× faster but **cos 0.95** (real quality cost,
audition first); needs a `value_info` strip + `op_types_to_quantize=['MatMul']` (this ORT build lacks a
`ConvInteger` kernel). Since fp32 is already ~2× realtime, prefer fp32 for evals; reserve INT8 for VST latency.

    # GPU (free card) or CPU — same script:
    python scripts/export_dit_control_onnx.py --ckpt <adapter.pt> --model medium-base --frames 256 --fp16 --validate
    python scripts/dit_control_onnx_infer.py --provider cpu --threads 12 \
        --dit-onnx dit_..._ctrl_onset_density.onnx --cond-npz dit_..._ctrl_onset_density.cond.npz \
        --decoder-onnx same_decoder_L128.onnx --cond cond.npz --uncond uncond.npz \
        --frames 256 --onset-density 8 --gain 3 --out steered.wav
- The fp16 ladder halves the 5.8 GB/rung; weights also aren't shared across rungs (dedup TODO).
- `medium-base` conditioning confirmed = cross_attn + global + **local_add_cond (257, must be fed)**; no
  active prepend/input_concat. If a variant adds them, extend `DiTCore`.

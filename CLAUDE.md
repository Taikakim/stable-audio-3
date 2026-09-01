# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Cross-project coordination (read first).** This repo is one of three in the
> mir + Stable Audio pipeline. Shared facts — data paths, which venv for which
> task, gotchas that span repos — live in `/home/kim/Projects/SAO/MASTER.md`.
> Read it before cross-cutting work, and append to
> `/home/kim/Projects/SAO/WORKLOG.md` when you finish something another repo's
> agent would want to know.
>
> ⚠️ **`WORKLOG.md` and the `AGENT_DIALOGUE.md` cross-instance channel are PUBLIC** (the
> dialogue log auto-mirrors to a public URL for remote review). **Never write secrets** —
> passwords, API keys/tokens, SSH creds, `.netrc` contents, or credential-revealing paths —
> into either; keep secrets in the shell/env. (See MASTER §4.)
>
> 📓 **Journal as you go.** The moment you land a finding or a **negative result** — however small —
> drop a few lines in your instance journal (`SAO/profiles/<handle>.journal.md`, per
> `SAO/profiles/SPEC-agent-profiles-journals.md`): a sentence + a link to the real doc. Negative
> results are first-class — a logged dead end stops the next instance re-deriving it. (MASTER §4.)
>
> 🗂️ **Self-describing outputs.** When you create an eval / render / test output dir, write a short
> sidecar beside the files: the test's **purpose** (one line), **paths to the related files** (config /
> script / spec), and the **checkpoint id + location** if it lives elsewhere. Manual tracking no longer
> scales — an anonymous dump of `.wav`/`.m4a` no one can place is a dead end. (MASTER §4.)
@/home/kim/Projects/SAO/MASTER.md

## Project Overview

**Stable Audio 3** is a state-of-the-art audio generation platform for fast, high-quality generated audio and music. It provides three inference modes: text-to-audio, audio-to-audio editing, and inpainting/continuation. The project also supports LoRA fine-tuning for model personalization.

- **Python version**: 3.13 (strictly enforced in pyproject.toml)
- **Package manager**: uv (required for dependency management)
- **Hardware**: CPU (Small models), CUDA (Medium models), Apple Silicon (via CoreML)

## ⚠️ Fork ↔ upstream: Stability shipped SA3 with features GUTTED (2026-08-03)

This is a thin fork. **Authoritative upstream = the `upstream` remote → `https://github.com/Stability-AI/stable-audio-3.git`**
(added 2026-08-03; `git fetch upstream` for fresh Stability commits). `origin`/`fork` are Kim's *mirror*
(`Taikakim/stable-audio-3`). As of 2026-08-03 our working branch is **127 commits behind `upstream/main`**
(their recent work is mostly TensorRT/NVIDIA quant tiers — scan before pulling). The gutted features below are
**still present in current `upstream/main`** (e.g. `rescale_cfg: bool = False` at diffusion.py:230), so they're
upstream-vestigial, not our regressions, and reportable against their live tree.
Audit OUR changes with `git diff origin/main...HEAD` (vs the mirror point we forked from: 114 commits, mostly
*new* files; real source deletions ~90 lines of refactor-*replacements*; the "−2250" is 96% regenerated `uv.lock`).

But the bigger gotcha is **upstream itself ships features half-wired** — hooks/flags present, the
actual build/wiring stripped, so they silently no-op:
- **EMA was gutted**: `DiffusionCondTrainingWrapper` had `use_ema=True`, `self.diffusion_ema = None`,
  and defensive `diffusion_ema.ema_model if not None` reads — but the `EMA(...)` build + `ema_pytorch`
  dep were **absent from the entire repo history** (pickaxe-confirmed: never ours to strip). So every
  run silently had no EMA (and `train_lora` also hardcoded `use_ema=False`). **FIXED 2026-08-03**:
  added a self-contained `SimpleEMA` (no dep) + build + `on_train_batch_end` update + `train_lora`
  `--use-ema/--ema-beta/--ema-warmup-steps` flags — **full-finetune only** (LoRA force-disables EMA).
  The full-FT ckpt now carries BOTH online (`diffusion.model.*`) and EMA (`diffusion_ema.ema_model.*`)
  weights → **render/audition must load the EMA set** (TODO in the render path).
- **`generate()` no longer clamps output**: upstream did `result.to(float32).clamp(-1, 1)`; we kept only
  `.to(float32)` (model.py ~L395). Intentional — the clamp moved to peak-normalize-at-save
  (`sa3_control.audio_io.save_audio()`, better: preserves transients). **Any path doing raw
  `torchaudio.save()` on `generate()`'s output will clip** — always go through the save helper.

**Two more confirmed by the 2026-08-03 package-wide sweep (24 gutted features found; these two matter):**
- **`rescale_cfg` is a silent no-op — and `generate()` passes it `True`.** The boolean is threaded through
  `sample_diffusion` (documented) and set `True` by `model.py:381` (generate) + `:568` (LatCH), but
  `DiTWrapper.forward` (diffusion.py:230) forwards only `scale_phi`, dropping `rescale_cfg`; the rescale
  math (dit.py:616) fires ONLY on `scale_phi != 0`. So **`model.generate()` / all batch eval renderers ran
  PLAIN CFG, not rescaled** — the whole eval corpus is plain-CFG. Rescaled CFG DOES work via `scale_phi`
  (the Gradio "cfg_rescale" slider → diffusion_cond.py:271). Upstream-vestigial (since initial commit).
  **Decision pending:** wire `rescale_cfg=True → scale_phi≈0.7` (changes output vs corpus) or drop it and
  pass `scale_phi` directly. Reportable to Stability.
- **`"v"` diffusion objective (the factory DEFAULT) used to crash training — FIXED.** `training_step` had no
  `"v"` schedule branch (used `alphas` unconditionally → NameError) and `validation_step` called
  `get_alphas_sigmas`, which was defined/imported NOWHERE (upstream-gutted). **Resolved**: `get_alphas_sigmas`
  is now defined locally (`training/diffusion.py:40`) and both `training_step`/`validation_step` branch on
  `diffusion_objective == "v"` correctly. No live run hits this path (all real ckpts are `rectified_flow`),
  but #65/#66 (v-pred + ZTSNR HF recovery) can now build on it without re-deriving the fix. Still worth a
  report to Stability since their upstream never got the fix, but it is no longer blocking on our side.

**Semantic-gutting sweep (2026-08-03, wired-but-INERT values) found 2 that matter:**
- **`cross_attn_cond_mask` is unconditionally NULLED (dit.py:414)** — `cross_attn_cond_mask = None` right after
  it's computed, comment "Temporarily disabling conditioning masks due to kernel issue for flash attention."
  So text (T5Gemma) cross-attention attends over PADDED tokens on every gen; all our auditions/evals ran unmasked.
  **RESOLVED 2026-08-04 — do NOT re-enable, it is intentional + redundant.** We built a gated re-enable
  (`SA3_ENABLE_CROSS_ATTN_MASK=1`, off by default) + A/B'd it (`eval/cross_attn_mask_ab.py`): the masked arm
  **DISINTEGRATES to spectral artifacts** — masking the padding rebalances the softmax and over-amplifies the
  real-token conditioning ~N× (the DiT was trained WITHOUT the mask, so it's OOD). **Zach @ Stability confirmed:**
  "we don't really use the cross-attention mask — turned it off when I switched to flash attention (didn't support
  masking), never turned it back on. The learned padding token for T5Gemma is kind of doing a similar thing." So
  the learned pad token IS the built-in soft-mask; explicit masking is OOD + redundant. The gated flag stays as a
  documented OFF toggle; do not enable it at inference OR bake it into training. (Negative-result autopsy logged.)
- **`use_effective_length_for_schedule` reads True but is INERT** — shipped `LogSNRShift` has `rate=0`, which
  zero-multiplies the only seq_len term → noise schedule is byte-identical for every length. Length-adaptive
  scheduling LOOKS on, does nothing. **Null-result trap for #50/#54 long-context work** — want it real? need
  `rate>0` / distinct-min/max Flux shift, not just the flag. (Minor also-rans: negative-GLOBAL cond dead
  end-to-end [near-zero effect, negative cross-attn works]; self-attn `mask` param vestigial [padding_mask is live].)

**Lesson: before relying on any SA3 `use_*`/`enable_*` flag or config key, confirm it's actually
WIRED *and* that its value actually reaches the math (rescale_cfg & cross_attn_cond_mask both passed the
"declared" test but were inert).** Two sweeps done 2026-08-03 (structural + semantic; 24+9 findings, tiered
in the workflow outputs). NOT yet swept: sampler solver internals, pretransform/VAE chain, LatCH/sa3_control,
training-side loss/schedule configs — a third pass would close those.

## Common Commands

### Setup & Installation

```bash
# Install base dependencies (Python API only)
uv sync

# With Gradio UI
uv sync --extra ui

# With LoRA training support
uv sync --extra lora

# Everything (UI + LoRA)
uv sync --extra ui --extra lora

# Development dependencies (includes pytest, ruff)
uv sync --group dev
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_inference.py

# Run a specific test function
uv run pytest tests/test_inference.py::test_text_to_audio

# Save generated audio outputs to test_audio_outputs/ for manual inspection
uv run pytest --save-audio

# Run tests with verbose output
uv run pytest -v
```

### Development & Linting

```bash
# Lint with ruff (excludes models/, inference/, interface/, data/, training/)
uv run ruff check .

# Format with ruff
uv run ruff format .
```

### Running Models

```bash
# Launch Gradio web UI with the medium model
uv run python run_gradio.py --model medium

# Launch with a LoRA checkpoint
uv run python run_gradio.py --model medium --lora-ckpt-path path/to/lora.ckpt

# CLI: text-to-audio
stable-audio --model small-music -p "lo-fi hip hop beat, 90 BPM" --duration 30 -o beat.wav

# CLI: audio-to-audio (restyle)
stable-audio -p "bossa nova bassline" --init-audio input.wav --init-noise-level 0.8 -o out.wav

# CLI: inpainting (regenerate region)
stable-audio -p "punchy kick drum fill" --inpaint-audio input.wav --inpaint-start 4 --inpaint-end 8 -o out.wav

# CLI: continuation (extend beyond original length)
stable-audio -p "dreamy synth outro" --inpaint-audio input.wav --inpaint-start 10 --inpaint-end 30 --duration 30 -o out.wav
```

### LoRA Fine-Tuning

```bash
# Pre-encode audio dataset to latents (faster training)
uv run python scripts/pre_encode_dataset.py --model same-s --data_dir ./my_data --output_path ./latents_out

# Train LoRA with raw audio + captions
uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data --save_dir ./lora_out

# Train LoRA with pre-encoded latents
uv run python scripts/train_lora.py --model medium-base --encoded_dir ./latents_out --save_dir ./lora_out

# Training with custom hyperparameters
uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data --steps 500 --rank 8 --save_dir ./lora_out
```

## Architecture Overview

### Core Model Classes

**StableAudioModel** (`stable_audio_3/model.py`)
- Main inference wrapper that combines DIT (Diffusion Transformer) + SAME autoencoder
- Methods:
  - `from_pretrained(model_name)` — loads a checkpoint from HuggingFace
  - `generate()` — generates audio (supports text-to-audio, audio-to-audio, inpainting)
  - `load_lora()` — loads LoRA checkpoints for fine-tuned models
  - `set_lora_strength()` — controls the blend of LoRA adapters at runtime

**AutoencoderModel** (`stable_audio_3/model.py`)
- Standalone SAME (Semantic-Acoustic Music Encoder) autoencoder for encoding/decoding
- Methods:
  - `from_pretrained()` — loads SAME-S or SAME-L variants
  - `encode()` — converts audio waveforms to latents
  - `decode()` — reconstructs audio from latents

### Model Variants

Models are defined in `stable_audio_3/model_configs.py`:

| Model ID | Type | Hardware | Size | Max Duration |
|----------|------|----------|------|-------------|
| `small-music` | Full | CPU | 433M | 120s |
| `small-sfx` | Full | CPU | 433M | 120s |
| `medium` | Full | CUDA GPU | 1.4B | 380s |
| `small-music-base` | Base (unfinetuned) | CPU | 433M | 120s |
| `small-sfx-base` | Base (unfinetuned) | CPU | 433M | 120s |
| `medium-base` | Base (unfinetuned) | CUDA GPU | 1.4B | 380s |
| `same-s` | Autoencoder | CPU | — | — |
| `same-l` | Autoencoder | CUDA GPU | — | — |

**Base models** (`-base` suffix) are un-fine-tuned checkpoints used for LoRA training.

### Package Structure

```
stable_audio_3/
├── model.py              # StableAudioModel, AutoencoderModel (public API)
├── model_configs.py      # Model definitions and HuggingFace repo mappings
├── factory.py            # Model construction from config dictionaries
├── cli.py                # Command-line interface entry point
├── models/               # (excluded from ruff)
│   ├── diffusion.py      # DiTWrapper, ConditionedDiffusionModelWrapper
│   ├── dit.py            # Diffusion Transformer implementation
│   ├── autoencoders.py   # SAME encoder/decoder implementations
│   ├── conditioners.py   # Text/prompt encoding (T5-Gemma conditioner)
│   ├── transformer.py    # Attention blocks and transformer layers
│   └── lora/
│       ├── model.py      # LoRA layer implementations
│       ├── loader.py     # LoRA checkpoint loading
│       └── utils.py      # LoRA utilities (strength scaling, etc.)
├── inference/            # (excluded from ruff)
│   ├── sampling.py       # Diffusion sampling loop and scheduler
│   ├── audio_utils.py    # Audio preprocessing/postprocessing
│   └── distribution_shift.py  # Perplexity-based latent shifting
├── training/             # LoRA training
│   ├── diffusion.py      # Training loop (PyTorch Lightning)
│   └── utils.py          # Training utilities
├── data/                 # Dataset classes
│   ├── dataset.py        # LocalDataset, PreEncodedDataset, LatentDataset
│   └── utils.py          # Audio loading, augmentation, metadata handling
├── interface/
│   └── diffusion_cond.py # Gradio UI definition
└── loading_utils.py      # Checkpoint loading utilities
```

### Inference Pipeline

The `generate()` method orchestrates:

1. **Prompt encoding** — converts text prompts to embeddings via T5-Gemma conditioner
2. **Latent initialization** — initializes latent space (random or from input audio)
3. **Diffusion sampling** — iterative denoising via the DIT model
4. **Audio reconstruction** — decodes latents back to waveform via SAME decoder

Key parameters:
- `steps` — number of diffusion steps (8 typical for fast inference, 50+ for higher quality)
- `cfg_scale` — classifier-free guidance strength (0 = no guidance, higher = more prompt adherence)
- `duration` — target audio length in seconds
- `seed` — random seed for reproducibility

### Model Configuration Format

Models are loaded from JSON config files (hosted on HuggingFace). Key sections:

- `sample_rate` — audio sample rate (44.1 kHz)
- `io_channels` — mono (1) or stereo (2)
- `model.diffusion` — DIT architecture config (transformer depth, width, attention heads, etc.)
- `model.pretransform` — SAME autoencoder architecture (encoder, decoder, bottleneck)
- `conditioning` — how prompts and other signals are encoded and injected

## Testing

### Test Organization

```
tests/
├── conftest.py           # Fixtures: sa3_model, sa3_base_model, autoencoder, device, maybe_save_audio
├── test_inference.py     # Text-to-audio, audio-to-audio, inpainting tests
├── test_autoencoder.py   # SAME encoder/decoder tests
├── test_cli.py           # CLI tests
└── test_lora.py          # LoRA loading and inference tests
```

### Hardware-Conditional Testing

The test suite automatically detects available hardware (CUDA, MPS) and gates tests:

- **small-music, small-sfx** — run on CPU or any accelerator
- **medium** — requires CUDA; skipped on CPU-only systems
- **medium-base** — requires CUDA; skipped on CPU-only systems
- **same-l** — requires CUDA; skipped otherwise

Tests are parametrized via pytest fixtures so they run across all applicable model variants.

### Running a Subset of Tests

```bash
# Only CPU-friendly models
uv run pytest tests/test_inference.py -k "small"

# Skip base models (faster, but less comprehensive)
uv run pytest tests/test_inference.py -k "not base"

# Test autoencoder only
uv run pytest tests/test_autoencoder.py
```

## Hardware & Dependencies

### CUDA Support

By default, `uv sync` installs PyTorch 2.10.0 built against CUDA 12.6. To use a different CUDA version:

```bash
# Example: CUDA 11.8
uv pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu118
uv sync --no-install-package torch --no-install-package torchaudio
```

See the README for available CUDA variants and their requirements.

### Flash Attention 2 — RDNA4 / Composable-Kernel (NOT the CUDA wheel)

**Required for the Medium model.** On our hardware (AMD RDNA4, gfx1201, RX 9070 XT) the CUDA prebuilt
wheels are useless — we use the **CK-backend `flash_attn 2.8.4`** built for gfx1201, already present in
`.venv` (torch 2.10/2.12 ROCm).

> **FAST venv — ROCm 7.14, use it (Kim 2026-08-02).** `SAO/.venv` (py3.13, HIP runtime 7.14.60850,
> torch 2.14.0a0, source-built native-CK `flash_attn 2.8.4`) runs **100–200% faster** than this repo's
> 7.2.3 `.venv` and is now the **default for SA3 render/inference** — `/home/kim/Projects/SAO/.venv/bin/python`.
> Same activation rule below. Keep the 7.2.3 `.venv` as the stable reference (the existing eval corpus
> was rendered on it — a same-config A/B across backends carries a small confound). Full note: `SAO/MASTER.md` §3/§5.

**Activation — do it for every run; it's 30–100% faster than the fallback:**

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE   # MUST be set before `import torch` / `flash_attn`
```

Without it the wrapper auto-routes to the `aiter` Triton-AMD backend → `No module named 'aiter'` +
`flash_attn not installed, disabling Flash Attention` → SDPA/flex fallback (the slow path). FA
**training** (LoRA / control-adapter backprop through the DiT) is safe — the `FlashAttnFunc.backward`
13-gradient patch is applied. Build + verify recipe: **`../docs/flash-attn-ck-rdna4.md`** (canonical
cross-repo note: `SAO/MASTER.md` §5). `uv sync --inexact` preserves the built flash-attn.

### ROCm Support

The repo includes local ROCm wheels in `pyproject.toml` under `[tool.uv.sources]`. These are referenced with relative paths and require the wheel files to be present in the repo root.

**SA3-medium ROCm/RDNA4 gotchas (full list: `SAO/MASTER.md` §5):**
- **`MIOPEN_FIND_MODE=6` CRASHES the SA3-medium DiT** (MIOpen `std::vector` assertion / coredump) — use
  `MIOPEN_FIND_MODE=2` for medium training/inference (mode 6 is fine for the tiny LatCH heads).
- **`PYTORCH_TUNABLEOP_ENABLED=0`** — TunableOp freezes on RDNA4 / torch-2.12 (kernel-selection path trips
  when the validator passes). With CK flash-attn, GEMM tuning is marginal anyway.
- **Fixed `T=4096` beat-aligned crops** (the `latents_sa3` corpus) — batch=1 + variable-length training
  thrashes the GEMM/Triton kernel cache (each unique sequence length is a new kernel shape).
- **TFG / LatCH guidance must run fp32** — fp16 (model_half default) clashes with backprop grad dtypes.
- INT8/INT4 quantization is non-functional on ROCm (use bf16 + FA2).

## Key Files & Patterns

### Adding a New Inference Mode

1. Define the inference logic in `stable_audio_3/inference/sampling.py`
2. Add a method to `StableAudioModel` that calls it (e.g., `generate()` already wraps `sample_diffusion()`)
3. Add CLI flag to `stable_audio_3/cli.py`
4. Add test to `tests/test_inference.py`

### Adding a New Model Variant

1. Add a new `ModelConfig` entry to `stable_audio_3/model_configs.py` with the HuggingFace repo ID and paths
2. Update `models` dict and `all_models` dict
3. Add parametrized test fixture (already done in `conftest.py`)

### Training Custom LoRAs

Workflow:
1. Prepare dataset: audio clips + text captions (one .txt per .wav)
2. Pre-encode: `python scripts/pre_encode_dataset.py --model same-l --data_dir ./clips --output_path ./latents`
3. Train: `python scripts/train_lora.py --model medium-base --encoded_dir ./latents --save_dir ./output`
4. Infer: `run_gradio.py --model medium --lora-ckpt-path output/lora.safetensors`

LoRA is implemented via adapters in the DIT and conditioner; apply via `model.load_lora()` and adjust strength with `model.set_lora_strength()`.

## Excluded from Linting

The following directories are excluded from ruff checks (see `pyproject.toml`):
- `stable_audio_3/models/` — complex model implementations
- `stable_audio_3/inference/` — numerical stability-critical code
- `stable_audio_3/interface/` — Gradio UI
- `stable_audio_3/data/` — dataset loading
- `stable_audio_3/training/` — training loop

This is intentional to avoid false positives on complex numerical code.

## Useful Documentation

- **Inference methods**: `docs/workflows/inference.md` — detailed guide to all generation modes
- **LoRA training**: `docs/workflows/lora.md` — fine-tuning setup and best practices
- **Autoencoder workflows**: `docs/workflows/autoencoder.md` — encoding/decoding and batch processing
- **ONNX / AMD inference**: `docs/onnx-amd-inference.md` — exporting the SAME autoencoder to ONNX
  for low-VRAM AMD decode (ORT + MIGraphX); the FlexAttention/opset-18 export gotchas, GPU-verified
  results (cos 0.999998, RTF ~39×), and the low-VRAM decode server (`mir/scripts/latent_server_onnx.py`)
- **Prompting guide**: `docs/guides/prompting.md` — prompt engineering tips
- **Model overview**: `docs/guides/model-overview.md` — architecture deep dive

## Python & Import Patterns

- **Type hints**: The codebase uses Python 3.13 type hints (e.g., `list[T]` instead of `List[T]`)
- **Public API**: Exported via `stable_audio_3/__init__.py` — only `StableAudioModel` and `AutoencoderModel`
- **Model loading**: Always use `from_pretrained()` class method, never load checkpoints manually
- **LoRA paths**: Can be single path or list of paths; applied in order and can be blended with `set_lora_strength()`

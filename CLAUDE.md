# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Cross-project coordination (read first).** This repo is one of three in the
> mir + Stable Audio pipeline. Shared facts — data paths, which venv for which
> task, gotchas that span repos — live in `/home/kim/Projects/SAO/MASTER.md`.
> Read it before cross-cutting work, and append to
> `/home/kim/Projects/SAO/WORKLOG.md` when you finish something another repo's
> agent would want to know.
@/home/kim/Projects/SAO/MASTER.md

## Project Overview

**Stable Audio 3** is a state-of-the-art audio generation platform for fast, high-quality generated audio and music. It provides three inference modes: text-to-audio, audio-to-audio editing, and inpainting/continuation. The project also supports LoRA fine-tuning for model personalization.

- **Python version**: 3.13 (strictly enforced in pyproject.toml)
- **Package manager**: uv (required for dependency management)
- **Hardware**: CPU (Small models), CUDA (Medium models), Apple Silicon (via CoreML)

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

### Flash Attention 2

**Required for the Medium model** (automatically detected in tests via `test_flash_attention_available`).

Flash Attention is **not** listed in `pyproject.toml` because it's not published on PyPI with wide platform coverage. Install manually:

```bash
# Pre-built wheel (recommended)
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu126torch2.7-cp310-cp310-linux_x86_64.whl

# Or build from source (slow)
uv pip install ninja
FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 uv pip install flash-attn --no-build-isolation ...
```

Note: `uv sync --inexact` preserves flash-attn without the lockfile.

### ROCm Support

The repo includes local ROCm wheels in `pyproject.toml` under `[tool.uv.sources]`. These are referenced with relative paths and require the wheel files to be present in the repo root.

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
- **Prompting guide**: `docs/guides/prompting.md` — prompt engineering tips
- **Model overview**: `docs/guides/model-overview.md` — architecture deep dive

## Python & Import Patterns

- **Type hints**: The codebase uses Python 3.13 type hints (e.g., `list[T]` instead of `List[T]`)
- **Public API**: Exported via `stable_audio_3/__init__.py` — only `StableAudioModel` and `AutoencoderModel`
- **Model loading**: Always use `from_pretrained()` class method, never load checkpoints manually
- **LoRA paths**: Can be single path or list of paths; applied in order and can be blended with `set_lora_strength()`

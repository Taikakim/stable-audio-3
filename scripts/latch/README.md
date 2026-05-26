# LatCH for Stable Audio 3 — Phase 1 (small-music-base)

Latent-Control Heads: a lightweight transformer head predicts an MIR time-series
(e.g. bass RMS) from noisy SAME latents and is used as Training-Free Guidance in a
gradient-enabled Euler sampler to steer generation. Phase 1 targets the
**`small-music-base`** flow-matching model (50-step Euler + CFG). Ping-pong / post-trained
models and the Gradio UI are out of scope.

See the plan at `docs/superpowers/plans/2026-05-26-latch-sa3-phase1.md`.

## Prerequisites for the integration runs

The unit tests run anywhere, but the three commands below need:

- A ROCm GPU.
- Mounted drives for the audio corpus and latent output.
- HuggingFace access to download `same-s` (encode) and `small-music-base` (verify).
- `scipy` and `librosa` installed in this venv — required by the mir extractor used
  in `verify` (`uv pip install scipy librosa`). Note: a later `uv sync` may remove
  them since they aren't in `pyproject.toml`.
- The MIR project at `/home/kim/Projects/mir` with `timeseries.db` (training targets
  + verification extractor). The DB is keyed by clip name, so each encoded `.npy`
  stem must equal its mir crop key.

## The three commands

### 1. Re-encode a corpus through SAME → SA3 latents

```bash
uv run python scripts/latch/encode_latch_dataset.py \
  --model same-s \
  --audio-dir "/run/media/kim/Mantu/ai-music/Goa_Separated_crops" \
  --out-dir /run/media/kim/Lehto/sa3-latch-latents
```

Writes per clip: `<stem>.npy` (256×T latent) + `<stem>.json` (crop key, frame count,
seconds). Start with a small `--audio-dir` subset to validate end-to-end first.

### 2. Train the head

```bash
uv run python scripts/latch/train_latch.py \
  --feature rms_energy_bass --epochs 10 \
  --latent-dir /run/media/kim/Lehto/sa3-latch-latents
```

Saves `latch_weights_sa3/latch_sa3_rms_energy_bass_ep{1..10}.pt`. Watch for a
downward loss trend (large absolute MSE on dB targets is normal).

### 3. Closed-loop control verification (the Phase-1 success gate)

```bash
uv run python scripts/latch/verify_latch.py \
  --ckpt latch_weights_sa3/latch_sa3_rms_energy_bass_ep10.pt \
  --levels -50 -30 -10 --gain 8.0 --win-lo 0.4 --win-hi 1.0
```

Generates at each requested bass-RMS level, decodes, measures with mir's real
extractor, and prints `correlation(requested, measured)` + monotonicity.
**Pass criterion: correlation ≥ 0.9 and monotonic.** If control is weak, sweep
`--gain` (3→10) and `--win-lo` (0.3→0.6); the window is σ-relative so operating
points transfer more predictably than the old step-index window.

## Known risks for the runs

- `verify_latch.py` hand-replicates `StableAudioModel.generate()`'s conditioning
  preamble via internal `model.model.*` APIs. If SA3's `model.py` has changed,
  reconcile the preamble against the live `generate()` before trusting the run.
- The head is latent-space specific: a head trained on `same-s` latents is only
  valid for `small`/`small-music-base`. `medium`/`large` use `same-l` and need their own
  head.

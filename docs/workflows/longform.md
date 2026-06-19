# Long-form Generation (sliding-window render)

Render audio **longer than the model's native window** as a single `.wav`, by generating
overlapping windows that are each clamped to the previous window's tail and stitched with
crossfades. Offline (quality over speed); optimised for seamlessness and **no drift/collapse**.

> **Status (2026-06-19):** CPU-side logic is implemented and tested (18/18); the on-GPU
> render path is reviewed-by-construction but **not yet runtime-validated** (the dev box's
> VRAM was held by a training run). Treat outputs as unverified until the validation runs
> below have been done. Design/plan: `docs/superpowers/specs/2026-06-19-longform-sdedit-reanchor-crossfade-design.md`.

## Why this instead of the FIFO prototype

A pure FIFO/InfiniteAudio stream (`stable_audio_3/inference/fifo_infinite.py`) is *out of
distribution* on SA3 and **drifts** — energy collapses after ~18 s (measured). This
sliding-window approach **deletes drift by construction**: every window is a fresh,
in-distribution generation clamped (via SA3 inpainting, in latent space) to the previous
tail, so there is nothing to accumulate and collapse. The FIFO surgery is kept for later
(see "Approach C" below).

## CLI

```bash
# single prompt, 2 minutes
uv run python scripts/longform_render.py \
  --prompt "124 BPM acid techno, dry snappy kick and bassline" \
  --duration 120 --window-sec 30 --overlap-sec 5 -o out.wav

# prompt schedule (energy arc / scene changes): "t0:promptA|t1:promptB|..."
uv run python scripts/longform_render.py \
  --prompt "0:ambient drone pad|60:124 BPM acid techno|150:breakdown, filtered" \
  --duration 210 -o set.wav
```

Flags: `--model` (default `small-music-base`), `--duration`, `--window-sec`, `--overlap-sec`,
`--blend-frames`, `--steps`, `--cfg`, `--half`, `-o/--out`. A single prompt is a 1-entry
schedule; a `t:prompt|t:prompt` string is parsed into a timeline (a prompt may contain
colons — only a leading `float:` makes a segment a schedule entry).

## How it works

```
PromptSchedule ─► LongFormRenderer ─ per window ─►
   ChunkGenerator.generate(prompt, prefix_latents, prefix_frames, n_frames, seed)
      └ InpaintContinuationGenerator: clamp first `overlap` frames to the previous tail
        (latent-space inpaint, no decode→encode round-trip), generate the rest
   ─► CrossfadeStitcher: continuation = short slerp over the overlap; transition = slerp crossfade
   ─► DriftMonitor: per-chunk RMS/centroid telemetry + a collapse canary (see below)
   ─► accumulate latents ─► chunked decode ─► clamp [-1,1] ─► soundfile .wav
```

- **Continuation** (same prompt): window *k* is clamped to window *k−1*'s tail, so the join
  is continuous by construction; a short slerp smooths any residual seam.
- **Transition** (prompt change): the new section is generated **fresh** (no clamp to the old
  prompt) and slerp-crossfaded over `crossfade-sec`.
- **Decode** is chunked (`pretransform.decode(..., chunked=True)`) so multi-minute latents
  don't OOM, and audio is **clamped to [-1, 1]** before writing (SA3 decoder output can
  exceed 1.0; WAV writers clip — see `MASTER.md` §5).
- **Drift canary:** if `DriftMonitor.should_reanchor` fires (RMS collapse vs the running
  median), the renderer emits a `RuntimeWarning`. In this design that should *never* fire —
  it is the signal that the clamp isn't holding (the FIFO failure mode). Watch the printed
  `drift_log rms` list: flat ⇒ healthy; a late drop ⇒ investigate.

## Approach C (later): grafting FIFO back in

The renderer depends only on the `ChunkGenerator` interface. A future `BoundedFifoGenerator`
(per-frame FIFO within its ~15 s good horizon, clamped to the prefix) drops in behind the
same signature with no renderer changes — combining frame-level continuity *inside* chunks
with the drift-bounded re-seeding *between* them. `SDEditReanchor` (latent audio2audio
re-noise→denoise) is implemented and reserved for an opt-in transition morph and for C's
drift refresh; the default Approach-A path is slerp-only.

## Validate the GPU path (run when VRAM is free)

```bash
# 1) the gated generation tests (skip cleanly when VRAM < ~6 GB free; should PASS when free)
uv run pytest tests/test_longform.py -v

# 2) a real 2-minute render — confirm drift_log rms stays flat, then listen
uv run python scripts/longform_render.py --prompt "124 BPM acid techno" --duration 120 -o /tmp/longform_2min.wav
```
Acceptance: the per-chunk `drift_log` RMS stays roughly flat across the whole render (the
FIFO prototype failed exactly this — it collapsed to ~0.03), and the seams are inaudible.

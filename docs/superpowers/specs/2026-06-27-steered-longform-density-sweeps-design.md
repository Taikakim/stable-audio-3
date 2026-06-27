# Steered Long-Form Generation with Density Sweeps — Design

**Date:** 2026-06-27  **Status:** design, awaiting approval

## Goal

Generate ~6-minute SA3 audio whose onset density follows a shaped time schedule
(linear-descending, triangular, bi-modal, sinewave), to (a) demonstrate time-varying
control over a long canvas and (b) test **disentanglement-at-scale**: does density
follow the requested shape while BPM stays flat?

## Two workstreams (staged)

### A — FIFO mechanism smoke (CPU, de-risk, runs first)
- Run existing `scripts/fifo_infinite_smoke.py` on `small-music-base`: `--parity`
  (patched per-frame forward must match the scalar forward, rel-err < 5e-3), then a
  short FIFO gen (window 64, emit 32 ≈ 3 s).
- **Answers:** does the UNTESTED per-frame-timestep FIFO produce coherent audio, or
  does it drift (spec risk #3, "the most likely failure mode")? Check finite / RMS /
  the per-segment drift report.
- No new code, no GPU. Runs parallel to the GPU work. Purely informational — its
  result decides whether FIFO-native steering is ever worth pursuing (vs the
  in-distribution sliding-window path, which is what workstream B uses regardless).

### B — Steered 6-min longform (GPU, after the card frees)
Builds on `scripts/longform_render.py` — sliding-window inpaint-continuation that
**already supports a time-varying `PromptSchedule`** (`'0:A|30:B'`). The density sweep
is the same pattern on a new axis.

- **New module `density_schedule.py`** — pure function: (shape, duration, span) → S(t)
  array. Shapes: linear-descending, triangular, bi-modal, sinewave. Standalone-testable.
- **New (thin) renderer hook** — a `ControlSchedule` analogous to `PromptSchedule`:
  per window, load the control adapter (`load_adapter_state`) and set the onset scalar
  = S(window-midpoint) via `use_control_context`, exactly as `onset_eval.py` drives the
  head. Reuses the existing `LongFormRenderer`; no change to its continuation logic.
- **Heads (the A/B):** the **triangular** shape on BOTH heads —
  `onset_per_beat` (disentangled) vs `onset_density` (literal/entangled) — the headline
  side-by-side. The other three shapes run on `onset_per_beat`.
- **Params:** window 30 s / overlap 5 s (~12 control points across 360 s, smooth for all
  shapes); gain ≈ 6 (ep25's best-authority point); span 2→14 onsets/s; best-sounding
  checkpoint per head (`cross_25A75F` soup / Fusion-40ep for density; opb-ep15+ for opb).
- **Measure (the payoff):** windowed output onset-density vs S(t) (control tracking)
  **and** BPM-over-time (tempo flatness) — via mir's essentia/librosa measure. Per run:
  a 6-min wav + a plot (requested vs measured density, BPM overlaid).
- **Deliverable:** the wavs + plots + a small viewer page (eval-page style, same-playhead
  playback).

## Components & boundaries
- `density_schedule.py` — pure, no deps on the model; unit-testable in isolation.
- `ControlSchedule` hook — thin adapter over the existing renderer; injects a scalar
  per window; does not touch continuation/blending.
- Measurement — reuses the existing mir onset/BPM eval measure (no new analysis code).

## Staging
- **A:** CPU, starts now (parallel to the fp16 GPU verify).
- **B:** GPU, after the fp16 verify frees the card. ~5 runs × 6 min ≈ 1–1.5 h GPU.

## Out of scope (YAGNI)
- **FIFO-native steered longform** (per-frame control on the untested FIFO) — deferred
  until workstream A proves the FIFO coherent. Workstream B does NOT depend on FIFO.
- Curved denoising; the FIFO buffer-zone clean anchor (v2 per the FIFO spec).
- Per-frame (sub-window) control resolution — window granularity is plenty for 6-min sweeps.

## Success criteria
- A: a coherence verdict (coherent / drifts) with the drift report attached.
- B: for the triangular A/B, the disentangled head shows density tracking S(t) with BPM
  flat; the entangled head shows density tracking with measurable BPM drift — the plots
  make the contrast legible.

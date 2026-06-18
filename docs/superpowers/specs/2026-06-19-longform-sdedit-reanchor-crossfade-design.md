# Long-form SA3 generation — SDEdit re-anchor + crossfade (Approach A, C-ready)

**Status:** design checkpoint. Sections 1–3 approved by Kim in brainstorming; Section 4
(decode/error/testing) written here for review. **Next step on resume: user reviews this
spec → `writing-plans` skill → implement.** GPU was busy (~6 h control-head run) at
authoring; all design/spec/code is GPU-free, validation deferred.

## Goal & decisions (locked)
Offline long render (2–30 min) of SA3 audio to one `.wav`, optimizing for **seamlessness,
no drift/collapse, and evolution over time**; latency irrelevant. Configurable prompt:
single prompt OR a `(time, prompt)` schedule (single = 1-entry schedule).

**Approach A** (sliding-window outpainting + beat-aligned crossfade) **now**, architected
so **Approach C** (bounded-FIFO chunks) grafts in later behind one interface. Why A: it
deletes drift instead of managing it — each window is a fresh *in-distribution* generation
clamped to the previous tail, so there is no OOD accumulation to collapse (the FIFO 24 s
run collapsed RMS→0.03 after ~18 s; A has no such failure mode). Reuses `generate()`'s
inpaint path, mir `latent_crossfader` slerp, and the drift metrics already written.

The validated FIFO surgery (`stable_audio_3/inference/fifo_infinite.py`) is **not used by
A** but is not wasted: it's the engine C drops in once a finetune makes per-frame
conditioning in-distribution.

## Section 1 — Architecture & the C-graft seam (APPROVED)
The orchestrator only ever talks to a **`ChunkGenerator` interface**. Swap the impl to
move A↔C; nothing else changes.
```
LongFormRenderer ── walks PromptSchedule, requests one chunk at a time,
                    stitches, monitors, decodes, writes the long .wav
   └─► ChunkGenerator (interface)
        ├─ A now : InpaintContinuationGenerator   (clamp prefix → generate rest, in-distribution)
        └─ C later: BoundedFifoGenerator          (same signature; FIFO within ~15 s horizon)
   supporting units (built now; A vs C lean on them differently):
   • CrossfadeStitcher — clamp-and-blend for continuation; slerp/SDEdit for transitions
   • SDEditReanchor    — noise→denoise under prompt (audio2audio); A morphs transitions, C refreshes drift
   • DriftMonitor      — RMS/centroid/tempo telemetry + should_reanchor(); A logs, C gates
```

## Section 2 — Components (APPROVED)
- **`PromptSchedule`** — from a string or `[(start_sec, prompt, opts?)]`; `resolve(t) →
  (prompt, is_transition, crossfade_sec)`. Single prompt = 1 entry, 0 transitions. Pure data.
- **`ChunkGenerator`** interface: `generate(prompt, prefix_latents, prefix_frames,
  n_frames, seed) → latents (1,C,n_frames)`.
  - **`InpaintContinuationGenerator` (A):** builds **latent-space** inpaint conditioning
    (`inpaint_mask=1` on `[0,prefix_frames)`, `inpaint_masked_input=prefix_latents` there),
    runs `sample_diffusion`. Clamp is on **latents** — no decode→encode round-trip. Reuses
    the cond plumbing from `fifo_infinite.build_window_conditioning` (non-zero mask now).
- **`SDEditReanchor`** — `reanchor(latents, sigma_peak, prompt, seed)`: the triangular
  sweep = `sample_diffusion(init_data=latents, init_noise_level=sigma_peak, …)` (SA3 mixes
  init_data at `sampling.py:464`). `sigma_peak ≈ 0.4–0.6` (zerosep η sweet spot). A: morph
  transitions; C: drift refresh.
- **`CrossfadeStitcher`** — latent-space (reuse mir `latent_crossfader.slerp`):
  `continuation_join` (short slerp over the overlap — smooths soft-clamp seam, near-identity
  if hard); `transition_join` (longer slerp over `crossfade_sec`, or `SDEditReanchor` morph
  under a blended prompt).
- **`DriftMonitor`** — `observe(chunk)→stats` (RMS/centroid/tempo) + `should_reanchor(stats)`.
  A: telemetry (drift not expected — a canary if it fires). C: gates chunk boundaries.
- **`LongFormRenderer`** — the loop; holds `prev_tail`, accumulates latents, chunked-decodes,
  writes one `.wav` via soundfile.

## Section 3 — Clamp & transition mechanics (APPROVED)
Latent grid 10.77 fps. Defaults: window ≈30 s (~323 fr), overlap ≈5 s (~54 fr), transition
crossfade ≈4 s (~43 fr).

**Continuation (same prompt) — drift-free by clamping:**
```
window k: inpaint_mask = [1]*overlap + [0]*(n-overlap);  masked_input[:,:, :overlap]=prev_tail
       → generate frames [overlap:n];  continuation_join: short slerp over [0:overlap]
       → append new region [overlap:n] to the long latent;  prev_tail = chunk[..., -overlap:]
```
**Transition (prompt change):** generate new section under `prompt_k`; `transition_join` =
latent slerp crossfade over the region, OR SDEdit morph under `blend(prompt_{k-1}, prompt_k)`.

**Verify at implementation (design robust either way):** (1) clamp hardness — does SA3
inpaint hard-replace the known region during sampling or only condition on it (it rides as
`local_add_cond`)? (2) continuation seam by ear; (3) decode chunking (§4).

## Section 4 — Decode, errors, testing (NEW — for review)
**Decode.** Stitch entirely in latent space → one long latent `(1,256,T)`. Decode via
`pretransform.decode`; for multi-minute latents use **overlapped chunked decode** (~30 s
latent chunks, few-second overlap, crossfade the decoded audio — the SAME decoder is
convolutional, so naïve chunk boundaries seam). Check `sample_diffusion(chunked_decode=…)`
first; if it lacks overlap-crossfade, implement overlapped decode in the renderer. Write
with **soundfile** (portable across the torchcodec-less 7.14 venv).

**Errors.** Drive-unmounted / model-load → fail fast, clear message. NaN/inf chunk → retry
with a new seed (bounded), then abort with context. DriftMonitor firing in A → loud warning
(canary: the clamp isn't holding). Schedule edges: transition at t=0, total_duration not a
window multiple, final partial window.

**Testing.** *Unit (CPU, no model):* `PromptSchedule.resolve` (single/multi/edge times);
`CrossfadeStitcher` slerp shapes + endpoints (α=0/1 return endpoints); `DriftMonitor` on
synthetic signals; **renderer loop with a FAKE ChunkGenerator** (ramp latents) → verify
overlap/concat bookkeeping, total length, `prev_tail` handoff — no GPU. *Integration (GPU,
deferred):* clamp region matches prefix; 2-min single-prompt render stays flat on the drift
report (the FIFO run failed this — the acceptance metric); prompt-schedule render →
audible-smooth transitions; A/B vs the FIFO collapse.

**C-graft validation (future):** drop `BoundedFifoGenerator` behind the interface;
renderer + tests unchanged.

## File plan (proposed)
`stable_audio_3/inference/longform.py` (units + renderer) · `scripts/longform_render.py`
(CLI: prompt or `--schedule`, duration, window/overlap/crossfade, out) · tests under
`tests/` for the CPU-unit pieces.

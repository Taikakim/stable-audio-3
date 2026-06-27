# Long-form Development System — Design Spec

**Date:** 2026-06-28
**Status:** Design (buildable; GPU-bound parts flagged as stub-this-pass)
**Owner repos:** `stable-audio-3` (longform engine), `stable-audio-tools/avp_sa3` (control + new code)
**Cross-refs:**
- `stable-audio-3/docs/superpowers/specs/2026-06-19-longform-sdedit-reanchor-crossfade-design.md` (the original longform design)
- `SAO/MASTER.md` §3 (venvs), §4 (control-eval / LatCH), §5 (gotchas)
- memory: `riffer-disentanglement-recipe`, `sa3-riffer-findings`

---

## 0. Problem statement (validated this session)

The steered long-form renderer **works** — no-adapter baseline A0 scores Audiobox
**PQ 7.85 / CE 7.19**, above the real-Goa floor (PQ 7.60 / CE 6.11). The constant
control-gain bug is fixed by the density-dependent `ridge_gain` (`density_schedule.py`).

The **remaining defect is LOOPINESS**: the inpaint *clamp* continuation
(`InpaintContinuationGenerator`, Approach A) preserves the prefix so faithfully that
each window **repeats the same idea** instead of **developing**. We need a *development*
system, not just a *coherence* system.

**Metric discipline (do not violate):**
- **Audiobox Content Enjoyment (CE)** is the quality north-star. Real-Goa floor CE 6.11.
- **Onset-authority is GAMED by high gain** — never optimize it alone; it is a constraint
  the `ridge_gain` already satisfies, not an objective.
- The new development metrics (MERT rhythm-sim / melody-sim) are **band targets**, not
  maxima — maximizing melody-sim *is* the loop.

This spec defines five parts. Each part lists: **file path**, **public interface
(typed)**, **deps on existing code**, **venv**, and **how it wires into
`steered_longform.py`**. Cross-venv (best-of-N) and the trained-MERT-model plan are
specified in full; GPU-only pieces are flagged "STUB THIS PASS".

---

## 1. Architecture map

### 1.1 Existing code this builds on (read before editing)

| Symbol | File | Role |
|---|---|---|
| `LongFormRenderer.render_latents` | `stable-audio-3/stable_audio_3/inference/longform.py` | window loop, prefix/stitch decisions, drift log |
| `ChunkGenerator` (ABC) | same | `.generate(prompt, prefix_latents, prefix_frames, n_frames, seed) -> (1,C,T)` |
| `InpaintContinuationGenerator` | same | Approach-A clamp generator (uses `sample_diffusion` + inpaint mask) |
| `SDEditReanchor.reanchor(latents, sigma_peak, prompt, seed)` | same | re-noise→denoise via `sample_diffusion(init_data=…, init_noise_level=…)` — **UNWIRED** |
| `CrossfadeStitcher.{continuation_join,transition_join}` | same | slerp stitches; `transition_join` currently fires **only on prompt change** |
| `PromptSchedule(spec, crossfade_sec)` / `.resolve(t)->(prompt,is_transition,xf_sec)` | same | multi-prompt arc (already supports `[(t,prompt),…]`) |
| `parse_schedule(arg)` | `stable-audio-3/scripts/longform_render.py` | `'0:A|30:B' -> [(0.0,'A'),(30.0,'B')]`; single prompt -> str |
| `SteeredGenerator` | `avp_sa3/sa3_control/steered_longform.py` | wraps inner generate, applies onset scalar via `use_control_context` |
| `ControlSchedule(shape,duration,lo,hi).resolve(t)` / `ridge_gain(density)` | `avp_sa3/sa3_control/density_schedule.py` | density curve + inverse-ridge gain |
| `use_control_context(ControlContext(tokens, gain))` | `avp_sa3/sa3_control/adapters.py` | module-global control channel (survives grad-checkpoint recompute) |
| `ScalarAttributeEncoder(control_dim,n_tokens)` | `avp_sa3/sa3_control/conditioner.py` | scalar -> `(1,n_tokens,control_dim)` control tokens |
| `install_adapters(sam, control_dim)` / `load_adapter_state(state, wrappers, enc)` | `avp_sa3/sa3_control/{inject,generate}.py` | bake the onset adapter into the DiT |
| `load_latch_from_checkpoint(path, device) -> LatCH` | `stable-audio-3/stable_audio_3/models/latch.py` | auto-detects arch; attaches `.metadata` (`std_mean`/`std_std`/`loss_type`/`t_injection`) |
| `sample_flow_euler_multi_latch_guided(model, x, sigmas, guides, **cond_inputs)` | `stable-audio-3/stable_audio_3/inference/latch_guided.py` | TFG sampler; `guides=[{head,target,weight,start_pct,end_pct,loss_type,huber_beta}]` |
| `build_target(kind,value,B,C,frames,*,fps,device,dtype)` | `stable-audio-3/stable_audio_3/inference/latch_targets.py` | synthetic LatCH target builder (we add a chroma builder beside it) |
| `analyze_audiobox_aesthetics(path) -> {content_enjoyment,…}` | `mir/src/timbral/audiobox_aesthetics.py` | CE scorer (mir venv) |
| MERT embed pattern (layers `(3,4,5,6,23)`, `m-a-p/MERT-v1-330M`, 24 kHz) | `/home/kim/.claude/jobs/b1ee18e7/tmp/mert_embed.py` | reference for the MERT scorer |

### 1.2 New code (all under `avp_sa3/sa3_control/`, tests under `avp_sa3/sa3_control/tests/`)

| File | Part | Venv |
|---|---|---|
| `development_renderer.py` | 1 (continuation modes) | render: `sa3-rocm7.13-test/.venv` |
| `prompt_arc.py` | 2 (prompt arc) | render (pure-python, importable anywhere) |
| `chroma_guided_generator.py` | 3 (chroma guidance) | render |
| `bestof_selector.py` | 4 (best-of-N orchestration, render side) | render |
| `mert_score.py` | 4 (scorer, mir side) | **mir** `mir/bin/python` |
| `mert_reward/{reward.py,dataset.py,model.py,train.py}` | 5 (trained re-ranker) | mir (train), render (inference swap-in) |
| `tests/test_*.py` | all | pure-python (any venv with torch CPU) |

### 1.3 Venv & cross-repo rules

- **Render** (everything that touches the SA3 model / DiT / decode): `sa3-rocm7.13-test/.venv/bin/python`,
  `FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE` before `import torch`.
- **Score** (MERT + Audiobox): `/home/kim/Projects/mir/mir/bin/python` — has essentia/madmom, MERT via
  `transformers` 5.0, and `src/timbral/audiobox_aesthetics`. Runs as a **separate subprocess** from the render.
- **Never edit `stable-audio-3/stable_audio_3/inference/longform.py`.** Subclass `LongFormRenderer`
  and wrap `ChunkGenerator` from `avp_sa3`. (The one allowed cross-repo edit is `steered_longform.py`,
  which already lives in `avp_sa3`.)
- `avp_sa3/sa3_control` is importable as `sa3_control` (its parent `avp_sa3` is on `sys.path`); the mir
  subprocess gets `PYTHONPATH=<avp_sa3>:<mir>` so it can `import sa3_control.mert_score` **and**
  `from timbral.audiobox_aesthetics import analyze_audiobox_aesthetics`.

### 1.4 Generator/renderer stack (outer → inner)

```
DevelopmentRenderer(LongFormRenderer)            # Part 1: continuation mode + stitch
  └─ BestOfNChunkGenerator (optional)            # Part 4: N candidates -> subprocess score -> argmax
       └─ SteeredGenerator                       # existing: onset scalar via use_control_context
            └─ ChromaGuidedGenerator             # Part 3 (optional) — else InpaintContinuationGenerator
                 └─ (TFG sampler / sample_diffusion over the DiT)
PromptSchedule  <-  prompt_arc.parse_prompt_arc  # Part 2: drives DevelopmentRenderer.render_latents
```

Each layer keeps the `ChunkGenerator.generate(prompt, prefix_latents, prefix_frames, n_frames, seed)`
signature, so any layer is optional and the renderer is agnostic.

---

## 2. Part 1 — Continuation modes

**Goal:** replace the single loopy clamp with `--continuation {clamp,sdedit,crossfade}`.

- **clamp** — current behaviour. `prefix_frames=overlap`, `InpaintContinuationGenerator` clamps the
  prefix. Coherent, loopy. (default, regression-safe.)
- **sdedit** — decaying audio-init: seed each window **from the previous tail** and re-noise→denoise at a
  **decaying `init_noise_level`** so early windows develop freely and later windows stay anchored. Uses
  `SDEditReanchor` (already exists, just unwired).
- **crossfade** — **fresh seed per window** (`prefix_frames=0`) + **slerp the full overlap**
  (`CrossfadeStitcher.transition_join` over the whole `overlap`, not just `blend_frames`). Maximum
  development, coherence comes only from the slerp.

### 2.1 File: `avp_sa3/sa3_control/development_renderer.py`  (render venv)

```python
from __future__ import annotations
from typing import Literal, Optional
import torch
from stable_audio_3.inference.longform import (
    LongFormRenderer, ChunkGenerator, InpaintContinuationGenerator,
    SDEditReanchor, CrossfadeStitcher,
)

ContinuationMode = Literal["clamp", "sdedit", "crossfade"]


class SDEditContinuationGenerator(ChunkGenerator):
    """sdedit mode generator: init each window FROM the prefix and denoise at a
    decaying init_noise_level (high early -> low late). Wraps a clamp generator for
    the cond build and SDEditReanchor for the re-noise->denoise pass.

    init_noise_level(k) = init_noise_end + (init_noise_start - init_noise_end) *
                          max(0, 1 - k / max(1, total_windows - 1))
    Window 0 (prefix_frames==0) falls back to the inner clamp generate (pure fresh).
    """
    def __init__(self, model, *, steps: int = 50, cfg_scale: float = 6.0,
                 init_noise_start: float = 0.85, init_noise_end: float = 0.55,
                 total_windows: Optional[int] = None) -> None: ...

    def set_total_windows(self, n: int) -> None: ...   # renderer calls this once it knows the count

    def generate(self, prompt: str, prefix_latents: Optional[torch.Tensor],
                 prefix_frames: int, n_frames: int, seed: int) -> torch.Tensor:
        """If prefix_latents is None -> fresh clamp generate. Else build init_data by
        tiling/continuing prefix_latents to n_frames, then SDEditReanchor.reanchor at
        sigma_peak=init_noise_level(self._call). Returns (1, C, n_frames)."""
        ...


class DevelopmentRenderer(LongFormRenderer):
    """LongFormRenderer subclass that selects prefix/stitch behaviour by continuation_mode.

    clamp     -> base behaviour (prefix=overlap, continuation_join blend_frames).
    sdedit    -> prefix=overlap passed to SDEditContinuationGenerator (init), stitch via
                 continuation_join (slerp blend_frames) — anchoring comes from the init, not a clamp.
    crossfade -> prefix_frames forced to 0 (fresh seed each window); stitch via
                 transition_join over crossfade_overlap_frac * overlap frames every window.
    """
    def __init__(self, generator, channels: int, fps: float,
                 window_frames: int, overlap_frames: int, *,
                 continuation_mode: ContinuationMode = "clamp",
                 crossfade_overlap_frac: float = 1.0,
                 blend_frames: int = 3, stitcher=None, monitor=None) -> None: ...

    def render_latents(self, schedule, total_frames: int, base_seed: int = 0) -> torch.Tensor:
        """Same window loop as the base, but:
          - prefix_frames per window is chosen by self.continuation_mode
            (crossfade -> 0; clamp/sdedit -> overlap unless is_transition);
          - every-window slerp join in crossfade mode (n = round(crossfade_overlap_frac*overlap));
          - if the generator exposes set_total_windows, call it before the loop
            (ceil((total_frames - window) / (window - overlap)) + 1).
        Keeps drift_log + non-finite retry from the base."""
        ...
```

**Deps:** `LongFormRenderer`, `ChunkGenerator`, `SDEditReanchor`, `CrossfadeStitcher` (all in
`longform.py`). No edits to `longform.py`.

**Wiring:** `steered_longform.main` builds `DevelopmentRenderer(...)` instead of `LongFormRenderer`,
passing `continuation_mode=args.continuation`. For `sdedit`, the **inner** generator under
`SteeredGenerator` becomes `SDEditContinuationGenerator` instead of `InpaintContinuationGenerator`
(chosen in `main` by `args.continuation`).

**Stub note:** fully buildable + unit-testable now with `FakeChunkGenerator` (no GPU). The
`SDEditContinuationGenerator` numeric init-construction (`tile/continue` prefix) is testable with the
fake; the *quality* of sdedit needs a GPU launch session.

---

## 3. Part 2 — Prompt arc

**Goal:** `steered_longform.py` currently throws away the arc (`--prompt` single string ->
`PromptSchedule(args.prompt)`). Wire `'0:intro|90:build|180:peak'` through the existing
`PromptSchedule` so the prompt **develops** across the render.

### 3.1 File: `avp_sa3/sa3_control/prompt_arc.py`  (pure python)

```python
from __future__ import annotations

def parse_prompt_arc(arg: str) -> "str | list[tuple[float, str]]":
    """Mirror of stable_audio_3/scripts/longform_render.parse_schedule (kept local to
    avoid importing a script module). Single prompt -> str; '0:A|90:B' -> [(0.0,'A'),(90.0,'B')].
    Treated as a schedule ONLY if every '|'-segment starts with 'float:' (so '120bpm: x'
    stays a single prompt). Raises nothing; falls back to single-prompt on any parse miss."""
    ...
```

**Deps:** none (string parse). `PromptSchedule` (in `longform.py`) consumes the result.

**Wiring (`steered_longform.main`):**
- replace `ap.add_argument("--prompt", …)` semantics: keep `--prompt` but pass it through
  `parse_prompt_arc`, then `PromptSchedule(parse_prompt_arc(args.prompt), crossfade_sec=args.xfade_sec)`.
- add `ap.add_argument("--xfade-sec", type=float, default=4.0)`.
- `SteeredGenerator` is **unaffected** — it tracks density time from its own `_frames_before`
  accounting, independent of the prompt; the prompt simply flows from `schedule.resolve(t)` →
  `SteeredGenerator.generate(prompt=…)` → inner. On a prompt transition the renderer sets
  `prefix_frames=0`; `SteeredGenerator` already advances `_frames_before += (n_frames - prefix_frames)`
  correctly (full window after a transition).
- the schedule JSON sidecar (`args.out + ".schedule.json"`) gains `"prompt_arc": entries`.

**Stub note:** fully buildable + testable now (string parse + a renderer dry-run with `FakeChunkGenerator`).

---

## 4. Part 3 — Chroma (HPCP) guidance: the development axis

**Goal:** a chroma target that **MOVES per window** = a chord progression = melodic/harmonic
development. Use the trained head `latch_weights_sa3_medium/latch_sa3_hpcp_best.pt` (a 12-channel
HPCP LatCH **guidance** head — backprop through the DiT, fp32, gain heavier than the riffer adapter:
SA3-medium needs gain ≈ 48–96, MASTER §5).

This is **separate** from the onset adapter: the adapter is a forward mod (`use_control_context`); the
chroma head is **TFG guidance** (`sample_flow_euler_multi_latch_guided`). They compose: the guided
sampler runs the DiT forward under the onset control context, and the adapter is differentiable + reads
the module-global, so guidance gradients flow through it cleanly.

### 4.1 File: `avp_sa3/sa3_control/chroma_guided_generator.py`  (render venv)

```python
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import torch
from stable_audio_3.inference.longform import ChunkGenerator
from stable_audio_3.models.latch import load_latch_from_checkpoint, LatCH


# 12-pitch-class chroma templates. name -> root pc + chord-tone set.
def chord_to_chroma(name: str, *, sharpness: float = 1.0) -> np.ndarray:
    """'Am', 'C', 'Dm7', 'F#', or a raw '0,4,7' pc-set -> np.ndarray(12) in [0,1],
    L2-normalized. sharpness scales non-chord-tone leakage (1.0 = hard template)."""
    ...


class ChromaSchedule:
    """Time-varying 12-d chroma target = a chord progression on the latent frame grid."""
    def __init__(self, progression: Sequence[tuple[float, str]], *, fps: float,
                 head_metadata: Optional[dict] = None) -> None:
        """progression: [(t_sec, chord_name), ...], sorted, first t must be 0.0.
        head_metadata: LatCH .metadata; if it has standardized/std_mean/std_std the target
        is standardized to match how the head was trained."""
        ...

    def target(self, t_start_sec: float, n_frames: int,
               device, dtype=torch.float32) -> torch.Tensor:
        """Return (1, 12, n_frames): per-frame chroma template for the window covering
        [t_start_sec, t_start_sec + n_frames/fps), resolving chord changes within the window
        (step or short linear cross-blend), standardized if head_metadata says so."""
        ...


class ChromaGuidedGenerator(ChunkGenerator):
    """Inpaint-continuation + HPCP TFG. Builds the same inpaint cond_inputs as
    InpaintContinuationGenerator, then runs the gradient-enabled sampler with one hpcp guide
    whose target moves per window (the development axis)."""
    def __init__(self, model, hpcp_head: LatCH, chroma_schedule: ChromaSchedule, *,
                 steps: int = 50, cfg_scale: float = 6.0,
                 rho: float = 64.0, mu: float = 64.0, gamma: float = 0.3, n_iter: int = 4,
                 start_pct: float = 0.0, end_pct: float = 1.0,
                 loss_type: str = "cosine") -> None:
        """hpcp_head: load_latch_from_checkpoint(latch_sa3_hpcp_best.pt) (fp32, on cuda).
        rho/mu are the variance/mean guidance strengths (the 'gain ~48-96' knob; chroma is a
        DIRECTION so loss_type defaults to 'cosine'). The generator tracks its own output time
        (frames emitted minus prefix) to index chroma_schedule.target()."""
        ...

    @torch.no_grad()   # the sampler internally escapes inference_mode/enable_grad for TFG
    def generate(self, prompt: str, prefix_latents: Optional[torch.Tensor],
                 prefix_frames: int, n_frames: int, seed: int) -> torch.Tensor:
        """1. cond_inputs = inpaint cond build (mask+masked_input from prefix), like
              InpaintContinuationGenerator._cond.
           2. target = chroma_schedule.target(self._t_emitted_sec, n_frames, device).
           3. guides = [{head: hpcp_head, target, weight: 1.0, start_pct, end_pct,
                         loss_type, huber_beta: 1.0}].
           4. latents = sample_flow_euler_multi_latch_guided(inner.model.model, noise, sigmas,
                         guides, rho=rho, mu=mu, gamma=gamma, n_iter=n_iter, **cond_inputs).
           5. advance self._t_emitted_sec by (n_frames - prefix_frames)/fps; return latents.float()."""
        ...
```

**Deps:** `load_latch_from_checkpoint`, `sample_flow_euler_multi_latch_guided`,
`build_schedule` (sampling), the inpaint cond build (copy the 6 lines from
`InpaintContinuationGenerator._cond`; do not subclass it — that class hardcodes `sample_diffusion`).

**Wiring (`steered_longform.main`):**
- `--chroma-head PATH`, `--chroma-progression '0:Am|32:F|64:C|96:G'`, `--chroma-rho 64`,
  `--chroma-mu 64`, `--chroma-key A` (used only if a progression entry is a degree, not a name).
- if `--chroma-head` given: `inner = ChromaGuidedGenerator(sam, head, ChromaSchedule(parse_progression(...), fps=fps, head_metadata=head.metadata), rho=…, mu=…)`,
  **wrapped by `SteeredGenerator`** exactly as today. The onset adapter still applies because
  `SteeredGenerator.generate` opens `use_control_context` around `inner.generate`, and the guided
  sampler's DiT forwards run inside that context.
- progression parsing: reuse `prompt_arc.parse_prompt_arc` shape (`t:chord`) → `[(t, chord)]`.

**Stub / risk flags (THIS PASS):**
- `ChromaSchedule` + `chord_to_chroma` are **pure** → fully unit-testable now (shape, normalization,
  standardization, chord-change resolution).
- `ChromaGuidedGenerator.generate` needs the **GPU + the model**. MASTER §5: **TFG/LatCH guidance must
  run fp32 on SA3** (fp16 clashes with backprop grad dtypes) → the model used for chroma windows must be
  loaded fp32, which **raises VRAM**. Co-resident fp32 DiT (~5.8 GB) + decode + the onset adapter on 16 GB
  is tight; mitigations to validate in a launch session: smaller `--window-sec`, fewer `--steps`, or
  CPU-offload of the head. **The guided-generate path is STUB-THIS-PASS for runtime; the interface and
  schedule are final.**
- `rho/mu` defaults (64) are the MASTER §5 starting point, not a tuned value — sweep in a launch session,
  judge by **spread/CE**, never by rank-corr (the `corr=1.0` mirage).

---

## 5. Part 4 — MERT best-of-N selector (cross-venv)

**Goal:** generate **N candidate next-windows** at different seeds, **score each**, pick the best.
Reward steers toward **development without losing identity**:

```
reward = w_ce * CE
       + w_rhythm * rhythm_sim                       # MERT MID layers (3,4,5,6); want HIGH (groove continuity)
       - w_melody * |melody_sim - band_center|       # MERT UPPER layer (23); want a BAND, not max (max = loop)
```

- **CE** = Audiobox `content_enjoyment` (mir `analyze_audiobox_aesthetics`).
- **rhythm_sim** = cosine(MERT-mid(candidate), MERT-mid(reference=prev window)) — high = same groove.
- **melody_sim** = cosine(MERT-upper(candidate), MERT-upper(reference)) — a **target band**
  (e.g. 0.55–0.80): below = unrelated/incoherent, above = literal repeat (the loop).

The selector orchestrates in the **render venv** (it has the model + decoder); scoring runs in a **mir
venv subprocess** (MERT + Audiobox). Communication = a job JSON + a scores JSON in a shared workdir.

### 5.1 Render side — `avp_sa3/sa3_control/bestof_selector.py`  (render venv)

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import torch
from stable_audio_3.inference.longform import ChunkGenerator


@dataclass
class RewardWeights:
    w_ce: float = 1.0
    w_rhythm: float = 1.0
    w_melody: float = 1.0
    band_center: float = 0.675     # midpoint of the melody-sim band
    band_lo: float = 0.55
    band_hi: float = 0.80


class BestOfNChunkGenerator(ChunkGenerator):
    """Wraps an inner ChunkGenerator. On each generate(): produce N candidate windows at
    distinct seeds, decode each to wav in a workdir, score them via a mir-venv subprocess,
    keep the argmax-reward candidate's LATENTS. Logs every candidate's scores to
    self.log (for Part 5 bootstrapping)."""
    def __init__(self, inner: ChunkGenerator, *, pretransform, sample_rate: int, fps: float,
                 n_candidates: int = 4, seeds: Optional[Sequence[int]] = None,
                 weights: RewardWeights = RewardWeights(),
                 mert_python: str = "/home/kim/Projects/mir/mir/bin/python",
                 avp_sa3_path: str = "...avp_sa3", mir_path: str = "/home/kim/Projects/mir/src",
                 workdir: Optional[str] = None) -> None: ...

    def generate(self, prompt: str, prefix_latents, prefix_frames: int,
                 n_frames: int, seed: int) -> torch.Tensor:
        """For c in range(n_candidates):
             lat_c = self.inner.generate(prompt, prefix_latents, prefix_frames, n_frames,
                                         seed = (seeds[c] if seeds else seed + 9173*c))
             wav_c = decode(lat_c) -> workdir/win{k}_cand{c}.wav  (peak-norm, int16, mono ok)
           reference = workdir/win{k-1}_best.wav (None for k==0)
           scores = _run_scorer(job)   # subprocess -> scores.json
           best = argmax(reward); copy its wav -> win{k}_best.wav; self.log.append({...})
           return latents[best]"""
        ...

    def _run_scorer(self, job: dict) -> list[dict]:
        """Write job.json; subprocess.run([mert_python, '-m', 'sa3_control.mert_score',
        job.json, scores.json], env={PYTHONPATH: avp_sa3_path:mir_path, ...}); read scores.json."""
        ...
```

**Decode-per-candidate note:** decoding N windows per step is the cost. Use
`pretransform.decode(lat.to(pt_dtype), chunked=True)` (same as `steered_longform.main`), peak-normalize
+ clamp to `[-1,1]` before write (MASTER §5 clip gotcha).

### 5.2 Job / scores JSON contract (the cross-venv seam)

`job.json` (written by render side):
```json
{
  "candidates": ["/work/win3_cand0.wav", "/work/win3_cand1.wav", "..."],
  "reference":  "/work/win2_best.wav",        // null for the first window
  "sample_rate": 44100,
  "weights": {"w_ce":1.0,"w_rhythm":1.0,"w_melody":1.0,
              "band_center":0.675,"band_lo":0.55,"band_hi":0.80},
  "mert_layers_mid": [3,4,5,6],
  "mert_layers_upper": [23]
}
```
`scores.json` (written by mir side): list aligned to `candidates`:
```json
[{"path":"/work/win3_cand0.wav","ce":7.21,"rhythm_sim":0.83,"melody_sim":0.71,"reward":8.93}, ...]
```

### 5.3 Score side — `avp_sa3/sa3_control/mert_score.py`  (mir venv)

```python
"""MERT + Audiobox candidate scorer. Run via the MIR venv subprocess:
   mir/bin/python -m sa3_control.mert_score job.json scores.json
Imports: transformers (MERT-v1-330M), and timbral.audiobox_aesthetics (PYTHONPATH includes mir/src)."""
from __future__ import annotations
import numpy as np

MERT_ID = "m-a-p/MERT-v1-330M"
MERT_SR = 24000

class MERTEmbedder:
    """Loads MERT once; embeds a mono waveform -> {'mid': (D,), 'upper': (D,)} mean-pooled
    over the requested hidden-state layer groups (ref: jobs/b1ee18e7/tmp/mert_embed.py)."""
    def __init__(self, layers_mid=(3,4,5,6), layers_upper=(23,), device="cuda") -> None: ...
    def embed(self, wav: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        """resample to 24 kHz, mono, AutoProcessor -> model(output_hidden_states=True);
        return mean-pooled, L2-normalized 'mid' (mean over layers_mid) and 'upper'."""
        ...

def composite_reward(ce: float, rhythm_sim: float, melody_sim: float, *,
                     w_ce: float, w_rhythm: float, w_melody: float,
                     band_center: float) -> float:
    return w_ce*ce + w_rhythm*rhythm_sim - w_melody*abs(melody_sim - band_center)

def score_candidates(candidate_paths: list[str], reference_path: "str | None", *,
                     weights: dict, layers_mid, layers_upper) -> list[dict]:
    """For each candidate: ce = analyze_audiobox_aesthetics(path)['content_enjoyment'];
    emb = MERTEmbedder.embed(wav); rhythm_sim/melody_sim = cosine vs reference emb (1.0 if
    reference is None -> falls to band edge so the first window is CE-only);
    reward = composite_reward(...). Returns list of dicts (see scores.json schema)."""
    ...

def main() -> None:
    """argv: job.json scores.json -> json.dump(score_candidates(...))."""
    ...
```

**Deps:** `transformers>=5.0` (mir venv), `mir/src/timbral/audiobox_aesthetics.analyze_audiobox_aesthetics`,
`torchaudio` for load/resample. Model load is **once per subprocess** — so call the subprocess **once per
window** with all N candidates (not once per candidate).

**Wiring (`steered_longform.main`):**
- `--best-of-N` (int, default 1 = disabled), `--mert-weights w_ce,w_rhythm,w_melody`,
  `--melody-band lo,hi`, `--mert-python /home/kim/Projects/mir/mir/bin/python`.
- if `args.best_of_N > 1`: wrap the `SteeredGenerator` in `BestOfNChunkGenerator(steered, pretransform=…,
  sample_rate=sr, fps=fps, n_candidates=args.best_of_N, weights=RewardWeights(...), mert_python=…)` and
  hand **that** to `DevelopmentRenderer`.
- dump `bestof_log = bestofn.log` into the `.schedule.json` sidecar (this is the Part-5 training data).

**Cost / risk flags (THIS PASS):**
- N× generation + N× decode per window → N× wall-clock. Pair with `--best-of-N 3..4` and short windows.
- subprocess MERT load (~few s) amortized per window, acceptable.
- **Buildable + unit-testable now**: `composite_reward` (pure math), `BestOfNChunkGenerator` orchestration
  with a `FakeChunkGenerator` + a **monkeypatched `_run_scorer`** (returns canned scores → assert argmax
  pick + log). The real MERT/Audiobox path is an **integration test requiring the mir venv** (mark
  `@pytest.mark.integration`, skip in CI). The job/scores schema is final and independently testable
  (write job.json, run `mir/bin/python -m sa3_control.mert_score` by hand).

---

## 6. Part 5 — Trained MERT continuation model (SCAFFOLD ONLY — no GPU run this pass)

**Goal:** replace the per-window Audiobox+MERT subprocess (slow, N decodes) with a **learned
re-ranker** that predicts the reward (or the good-continuation band) directly from MERT features of
(prev window, candidate). Bootstrapped from the Part-4 `bestof_log`. Optionally a later **RL fine-tune**
of the generator against this reward (deferred).

### 6.1 Package `avp_sa3/sa3_control/mert_reward/`

`reward.py` (mir + render):
```python
def composite_reward(ce, rhythm_sim, melody_sim, *, w_ce, w_rhythm, w_melody, band_center) -> float: ...
# single source of truth, imported by mert_score.py AND the trainer (no divergence).
```

`dataset.py` (mir venv):
```python
def build_reward_dataset(bestof_logs: list[str], out_npz: str) -> None:
    """Read Part-4 bestof_log entries ({prev_wav, cand_wav, ce, rhythm_sim, melody_sim, reward}),
    MERT-embed prev+cand (mid+upper), write npz: X=concat(prev_emb, cand_emb), y=reward,
    plus the component columns (ce, rhythm_sim, melody_sim) for auxiliary heads."""
    ...
```

`model.py` (render + mir):
```python
import torch, torch.nn as nn
class MERTRewardModel(nn.Module):
    """Small MLP over [prev_mid, prev_upper, cand_mid, cand_upper] -> predicted reward
    (+ optional aux heads for ce/rhythm_sim/melody_sim). ~1-2 M params."""
    def __init__(self, emb_dim: int = 1024, hidden: int = 512, aux: bool = True) -> None: ...
    def forward(self, prev_emb: torch.Tensor, cand_emb: torch.Tensor) -> torch.Tensor: ...
```

`train.py` (mir venv):
```python
def main() -> None:
    """Fit MERTRewardModel on build_reward_dataset output. Loss = MSE(pred_reward, reward) [+ aux MSE].
    The TARGET reward uses composite_reward (Part-4 weights) so the learned model reproduces the
    selector, then can be evaluated as a drop-in re-ranker. Save .pt for the render-side swap-in."""
    ...
```

### 6.2 Swap-in point

In `BestOfNChunkGenerator`, gate the scorer:
- **subprocess mode** (default, Part 4): decode all N → mir subprocess → Audiobox+MERT.
- **learned mode** (Part 5, once trained): decode all N → embed with MERT (still mir-side, cheap) →
  `MERTRewardModel(prev_emb, cand_emb)` → reward; **skips Audiobox** per candidate (the model has
  internalized CE), cutting the per-window cost. Same `_run_scorer` JSON contract, different mir-side
  entry (`--scorer model --model PATH`).

**STUB THIS PASS:** define `reward.py` (real, shared, unit-tested), `model.py` forward-shape (unit-tested
CPU), `dataset.py` + `train.py` skeletons. **No training run** (needs a corpus of `bestof_log`s from
Part-4 runs first). **RL fine-tune of the generator is explicitly deferred** (named, not designed here).

---

## 7. `steered_longform.py` — consolidated changes

New args (all additive; defaults reproduce today's behaviour):

| arg | default | part |
|---|---|---|
| `--continuation {clamp,sdedit,crossfade}` | `clamp` | 1 |
| `--xfade-sec` | `4.0` | 2 |
| (`--prompt` now parsed via `parse_prompt_arc`) | — | 2 |
| `--chroma-head PATH` | none | 3 |
| `--chroma-progression '0:Am|32:F|…'` | none | 3 |
| `--chroma-rho` / `--chroma-mu` | `64` | 3 |
| `--best-of-N` | `1` (off) | 4 |
| `--mert-weights w_ce,w_rhythm,w_melody` | `1,1,1` | 4 |
| `--melody-band lo,hi` | `0.55,0.80` | 4 |
| `--mert-python` | `mir/bin/python` | 4 |

`main()` build order:
```python
sched_density = ControlSchedule(args.shape, args.duration, args.lo, args.hi)
prompt_sched  = PromptSchedule(parse_prompt_arc(args.prompt), crossfade_sec=args.xfade_sec)

if args.chroma_head:
    head = load_latch_from_checkpoint(args.chroma_head, device="cuda")  # fp32
    inner = ChromaGuidedGenerator(sam, head, ChromaSchedule(parse_prompt_arc(args.chroma_progression),
                                  fps=fps, head_metadata=head.metadata),
                                  steps=args.steps, cfg_scale=args.cfg,
                                  rho=args.chroma_rho, mu=args.chroma_mu)
elif args.continuation == "sdedit":
    inner = SDEditContinuationGenerator(sam, steps=args.steps, cfg_scale=args.cfg)
else:
    inner = InpaintContinuationGenerator(sam, steps=args.steps, cfg_scale=args.cfg)

steered = SteeredGenerator(inner, sched_density, enc, mean, std, args.gain, fps,
                           args.cfg, "cuda", md, ridge=args.ridge)

gen = steered
if args.best_of_N > 1:
    gen = BestOfNChunkGenerator(steered, pretransform=sam.model.pretransform,
                                sample_rate=sr, fps=fps, n_candidates=args.best_of_N,
                                weights=RewardWeights(*parse_weights(args)), mert_python=args.mert_python)

r = DevelopmentRenderer(gen, channels=sam.model.io_channels, fps=fps,
                        window_frames=f(args.window_sec), overlap_frames=f(args.overlap_sec),
                        continuation_mode=args.continuation)
lat = r.render_latents(prompt_sched, total_frames=f(args.duration), base_seed=args.seed)
```
Sidecar `.schedule.json` gains: `continuation`, `prompt_arc`, `chroma_progression`,
`best_of_N`, `mert_weights`, `bestof_log`.

**Composition guarantee:** the onset adapter (`use_control_context` in `SteeredGenerator`) wraps
`inner.generate`. Whether inner is `Inpaint`, `SDEdit`, or `ChromaGuided`, the DiT forwards happen inside
that context, so onset steering + chroma guidance + sdedit all stack. `BestOfN` sits **outside**
`SteeredGenerator` (it calls the fully-steered generate N times).

---

## 8. Tests (`avp_sa3/sa3_control/tests/`) — all CPU/model-free unless marked integration

| file | asserts |
|---|---|
| `test_development_renderer.py` | `FakeChunkGenerator`: clamp == base output; crossfade forces `prefix_frames==0` every window + slerp join length; sdedit calls `set_total_windows` and the init path; all modes hit `total_frames`, finite, drift_log length |
| `test_prompt_arc.py` | `parse_prompt_arc`: single→str, `'0:A|30:B'`→list, `'120bpm: x'`→str (colon-safe), missing-t→str |
| `test_chroma_schedule.py` | `chord_to_chroma('Am')` pc-set + L2-norm; `ChromaSchedule.target` shape `(1,12,n)`, chord change resolves at the right frame, standardization applied when `head_metadata['standardized']` |
| `test_bestof_selector.py` | `FakeChunkGenerator` + monkeypatched `_run_scorer`: returns N latents, picks argmax reward, log has N rows/window, reference threading (None at window 0) |
| `test_mert_reward.py` | `composite_reward` math (band penalty sign + zero at center); `MERTRewardModel.forward` output shape; `build_reward_dataset` schema on a tiny fake log |
| `test_mert_score_contract.py` (integration, skip w/o mir venv) | job.json→scores.json round-trip via `subprocess.run([mir_python, -m sa3_control.mert_score])` on 2 short wavs |

Follow the existing `tests/test_steered_generator.py` style (Fake encoder/inner, pure asserts, no GPU).

---

## 9. Build order & what is stub-this-pass

**Buildable + unit-tested now (no GPU):**
1. Part 2 `prompt_arc.py` (+ test) — trivial, unblocks the arc.
2. Part 1 `development_renderer.py` `DevelopmentRenderer` + `SDEditContinuationGenerator` bookkeeping
   (+ test with `FakeChunkGenerator`).
3. Part 3 `ChromaSchedule` + `chord_to_chroma` (pure) (+ test).
4. Part 4 `composite_reward` + `BestOfNChunkGenerator` orchestration with a monkeypatched scorer (+ test),
   and the job/scores JSON schema.
5. Part 5 `reward.py`, `model.py` forward, `dataset.py`/`train.py` skeletons (+ test of reward/model shapes).
6. `steered_longform.py` arg wiring (all defaults reproduce current behaviour; a `--continuation clamp`
   run must byte-match today's output as a regression gate).

**STUB THIS PASS (needs a GPU launch session / mir venv corpus):**
- Part 3 `ChromaGuidedGenerator.generate` runtime — **fp32 guidance VRAM** on 16 GB is the open risk;
  tune `rho/mu` (start 64), window length, steps by **CE/spread**, not rank-corr.
- Part 4 real MERT+Audiobox scoring path and the **melody-band calibration** (0.55–0.80 is a guess; set it
  from a sweep of real Goa window-to-window MERT-upper cosines).
- Part 5 actual training — blocked until Part-4 runs have produced a `bestof_log` corpus; RL fine-tune
  of the generator deferred (named only).

**Hard invariants to preserve:**
- never edit `stable-audio-3/stable_audio_3/inference/longform.py` (subclass/wrap only);
- guidance fp32 (MASTER §5); peak-normalize→clamp→int16 before any wav write (MASTER §5);
- judge quality by **Audiobox CE**, never onset-authority alone; MERT melody-sim is a **band**, not a max.

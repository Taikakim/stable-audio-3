# Steered Long-Form Density Sweeps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate ~6-minute SA3 audio whose onset density follows a shaped schedule (linear-descending, triangular, bi-modal, sinewave), driven by the trained control adapter, and measure whether output density tracks the shape while BPM stays flat.

**Architecture:** Reuse the existing sliding-window inpaint-continuation renderer (`stable_audio_3/inference/longform.py`) unchanged. Inject per-window control by wrapping its `ChunkGenerator` seam: a `SteeredGenerator` resolves a `ControlSchedule` by output-time (tracked from its own frame accounting — valid because single-prompt sweeps have no prompt transitions) and applies the onset scalar via `use_control_context` exactly as `onset_eval.py` does. Measurement and plotting run separately in the mir venv.

**Tech Stack:** PyTorch ROCm, `stable_audio_3` (medium-base), `sa3_control` (adapters/encoder), librosa + essentia (measurement).

## Global Constraints

- **Venv split (never mix):** generation/render → `/home/kim/Projects/SAO/sa3-rocm7.13-test/.venv/bin/python` (has `stable_audio_3` + `sa3_control` + torch). Measurement/plots → `/home/kim/Projects/mir/mir/bin/python` (essentia + librosa).
- **Set `FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE` before torch import** (activates CK flash-attn).
- **Control wiring (verbatim pattern from `onset_eval.py`):** `wrappers = install_adapters(sam, control_dim)`; `enc = ScalarAttributeEncoder(control_dim, n_tokens)`; `load_adapter_state(ck["state"], wrappers, enc)`; per window `s = torch.tensor([(raw-mean)/std])`, `ctrl = enc(s)`, `cc = torch.cat([ctrl, zeros],0)` when `cfg!=1` else `ctrl`, `with use_control_context(ControlContext(cc, gain=gain)): inner.generate(...)`. `mean,std = ck["scalar_norm"]`; `n_tokens=ck["args"]["n_tokens"]`, `control_dim=ck["args"]["control_dim"]`.
- **Latent fps** = `model.model.sample_rate / model.model.pretransform.downsampling_ratio` (~10.767 Hz, medium-base).
- **No edits to `stable_audio_3/inference/longform.py`** — integrate only through the `ChunkGenerator` interface.
- **Checkpoints:** density head — best-sounding `cross_25A75F` soup or Fusion-40ep; disentangled — opb `riffer_step81000.pt` (ep15). gain ≈ 6, span 2→14.
- New code lives in `stable-audio-tools/avp_sa3/sa3_control/`; tests in `stable-audio-tools/avp_sa3/sa3_control/tests/`.

---

### Task 0: FIFO mechanism smoke (workstream A, CPU, de-risk)

**Files:** none created — runs existing `stable-audio-3/scripts/fifo_infinite_smoke.py`.

- [ ] **Step 1: Parity check.** Run: `cd /home/kim/Projects/SAO/stable-audio-3 && HIP_VISIBLE_DEVICES="" .venv/bin/python scripts/fifo_infinite_smoke.py --parity`
  Expected: prints a rel-err; **PASS if rel-err < 5e-3** (the doc's gate).
- [ ] **Step 2: Short FIFO gen.** Run: `HIP_VISIBLE_DEVICES="" .venv/bin/python scripts/fifo_infinite_smoke.py --model small-music-base --window 64 --emit 32 --out /home/kim/fifo_smoke.wav`
  Expected: finite, sane RMS, a per-segment drift report, a WAV written.
- [ ] **Step 3: Record verdict.** Note in the run log whether output is coherent or drifts (the doc's risk #3). This is informational — it decides if FIFO-native steering is ever worth pursuing. **No commit (no code).**

---

### Task 1: `density_schedule.py` — the four shapes

**Files:**
- Create: `stable-audio-tools/avp_sa3/sa3_control/density_schedule.py`
- Test: `stable-audio-tools/avp_sa3/sa3_control/tests/test_density_schedule.py`

**Interfaces:**
- Produces: `ControlSchedule(shape:str, duration:float, lo:float, hi:float)` with `.resolve(t:float)->float` and module const `SHAPES`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_density_schedule.py
from sa3_control.density_schedule import ControlSchedule, SHAPES

def test_linear_descending_endpoints():
    s = ControlSchedule("linear_descending", 100.0, 2.0, 14.0)
    assert abs(s.resolve(0.0) - 14.0) < 1e-6
    assert abs(s.resolve(100.0) - 2.0) < 1e-6

def test_triangular_peaks_at_mid():
    s = ControlSchedule("triangular", 100.0, 2.0, 14.0)
    assert abs(s.resolve(0.0) - 2.0) < 1e-6
    assert abs(s.resolve(50.0) - 14.0) < 1e-6

def test_bimodal_two_peaks():
    s = ControlSchedule("bimodal", 100.0, 2.0, 14.0)
    assert abs(s.resolve(25.0) - 14.0) < 1e-6   # first peak
    assert abs(s.resolve(75.0) - 14.0) < 1e-6   # second peak
    assert abs(s.resolve(50.0) - 2.0) < 1e-6    # dip between

def test_clamps_outside_duration_and_validates_shape():
    s = ControlSchedule("linear_descending", 100.0, 2.0, 14.0)
    assert abs(s.resolve(500.0) - 2.0) < 1e-6
    assert set(SHAPES) == {"linear_descending","triangular","bimodal","sinewave"}
    try:
        ControlSchedule("bogus", 1.0, 0.0, 1.0); assert False
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kim/Projects/SAO/stable-audio-tools/avp_sa3 && /home/kim/Projects/SAO/sa3-rocm7.13-test/.venv/bin/python -m pytest sa3_control/tests/test_density_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError: sa3_control.density_schedule`.

- [ ] **Step 3: Write minimal implementation**

```python
# sa3_control/density_schedule.py
"""Time-varying control schedules for steered long-form generation.
Pure: (shape, duration, lo, hi) -> raw onset-density value at time t. No model deps."""
import math

SHAPES = ("linear_descending", "triangular", "bimodal", "sinewave")

class ControlSchedule:
    def __init__(self, shape, duration, lo, hi):
        if shape not in SHAPES:
            raise ValueError(f"unknown shape {shape!r}; expected one of {SHAPES}")
        self.shape = shape
        self.duration = float(duration)
        self.lo = float(lo)
        self.hi = float(hi)

    def resolve(self, t):
        u = min(max(t / self.duration, 0.0), 1.0)   # normalized 0..1
        if self.shape == "linear_descending":
            f = 1.0 - u
        elif self.shape == "triangular":
            f = 1.0 - abs(2.0 * u - 1.0)
        elif self.shape == "bimodal":
            f = 0.5 * (1.0 - math.cos(4.0 * math.pi * u))   # peaks at u=.25,.75
        else:  # sinewave: 3 smooth oscillations
            f = 0.5 * (1.0 - math.cos(6.0 * math.pi * u))
        return self.lo + (self.hi - self.lo) * f
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kim/Projects/SAO/sa3-rocm7.13-test/.venv/bin/python -m pytest sa3_control/tests/test_density_schedule.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add sa3_control/density_schedule.py sa3_control/tests/test_density_schedule.py
git commit -m "feat(sa3_control): density-sweep control schedules (4 shapes)"
```

---

### Task 2: `SteeredGenerator` — per-window control injection

**Files:**
- Create: `stable-audio-tools/avp_sa3/sa3_control/steered_longform.py`
- Test: `stable-audio-tools/avp_sa3/sa3_control/tests/test_steered_generator.py`

**Interfaces:**
- Consumes: `ControlSchedule` (Task 1); a `ChunkGenerator` (`.generate(prompt, prefix_latents, prefix_frames, n_frames, seed)`); an encoder callable `enc(s)->(1,n,d)`.
- Produces: `SteeredGenerator(inner, schedule, encoder, mean, std, gain, fps, cfg_scale, device, dtype)` implementing `.generate(...)` and exposing `.applied: list[tuple[float,float]]` (per-window `(t_sec, raw_density)`).

- [ ] **Step 1: Write the failing test** (model-free: fake inner + fake encoder)

```python
# tests/test_steered_generator.py
import torch
from sa3_control.density_schedule import ControlSchedule
from sa3_control.steered_longform import SteeredGenerator

class FakeEnc:
    def __call__(self, s): return s.view(1, 1, 1)

class FakeInner:
    def __init__(self): self.seen = []
    def generate(self, prompt, prefix_latents, prefix_frames, n_frames, seed):
        self.seen.append((prefix_frames, n_frames))
        return torch.zeros(1, 4, n_frames)

def test_window_time_and_scalar_tracking():
    sched = ControlSchedule("linear_descending", 100.0, 2.0, 14.0)
    inner = FakeInner()
    g = SteeredGenerator(inner, sched, FakeEnc(), mean=0.0, std=1.0, gain=1.0,
                         fps=10.0, cfg_scale=1.0, device="cpu", dtype=torch.float32)
    # window=300 frames (30s @10fps), overlap=50; window0 prefix=0, then prefix=50
    g.generate("p", None, 0, 300, 0)        # t=0   -> 14.0
    g.generate("p", torch.zeros(1,4,50), 50, 300, 1)  # t=300/10=30 -> resolve(30)
    g.generate("p", torch.zeros(1,4,50), 50, 300, 2)  # t=(300+250)/10=55 -> resolve(55)
    ts = [t for t, _ in g.applied]
    assert ts == [0.0, 30.0, 55.0]
    assert abs(g.applied[0][1] - 14.0) < 1e-6
    assert abs(g.applied[1][1] - sched.resolve(30.0)) < 1e-6
    assert abs(g.applied[2][1] - sched.resolve(55.0)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kim/Projects/SAO/stable-audio-tools/avp_sa3 && /home/kim/Projects/SAO/sa3-rocm7.13-test/.venv/bin/python -m pytest sa3_control/tests/test_steered_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: sa3_control.steered_longform`.

- [ ] **Step 3: Write minimal implementation** (the class only; `main()` is Task 3)

```python
# sa3_control/steered_longform.py
"""Steered long-form generation: wrap the longform ChunkGenerator with a per-window
onset-density control schedule. Single-prompt sweeps only (no prompt transitions)."""
import torch
from sa3_control.adapters import ControlContext, use_control_context


class SteeredGenerator:
    """Tracks output time from its own frame accounting and applies the scheduled
    onset scalar via use_control_context around each window's inner generate."""

    def __init__(self, inner, schedule, encoder, mean, std, gain, fps, cfg_scale, device, dtype):
        self.inner = inner
        self.sched = schedule
        self.enc = encoder
        self.mean = float(mean); self.std = float(std); self.gain = float(gain)
        self.fps = float(fps); self.cfg = float(cfg_scale)
        self.device = device; self.dtype = dtype
        self._frames_before = 0
        self.applied = []  # (t_sec, raw_density) per window

    def generate(self, prompt, prefix_latents, prefix_frames, n_frames, seed):
        t = self._frames_before / self.fps
        raw = self.sched.resolve(t)
        self.applied.append((t, raw))
        s = torch.tensor([(raw - self.mean) / self.std], device=self.device, dtype=self.dtype)
        ctrl = self.enc(s)
        cc = torch.cat([ctrl, torch.zeros_like(ctrl)], 0) if self.cfg != 1.0 else ctrl
        with use_control_context(ControlContext(cc, gain=self.gain)):
            out = self.inner.generate(prompt, prefix_latents, prefix_frames, n_frames, seed)
        self._frames_before += (n_frames - prefix_frames)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kim/Projects/SAO/sa3-rocm7.13-test/.venv/bin/python -m pytest sa3_control/tests/test_steered_generator.py -v`
Expected: 1 passed. (If `ControlContext`/`use_control_context` import-time-fail outside a model, wrap the import inside `generate` — but they are plain context objects and import cleanly.)

- [ ] **Step 5: Commit**

```bash
git add sa3_control/steered_longform.py sa3_control/tests/test_steered_generator.py
git commit -m "feat(sa3_control): SteeredGenerator — per-window control injection over longform"
```

---

### Task 3: render entry — `steered_longform.py:main()`

**Files:**
- Modify: `stable-audio-tools/avp_sa3/sa3_control/steered_longform.py` (add `main()` + `if __name__`)

**Interfaces:**
- Consumes: `SteeredGenerator` (Task 2), `ControlSchedule` (Task 1), `stable_audio_3.inference.longform.{LongFormRenderer,InpaintContinuationGenerator,PromptSchedule}`, the control wiring from Global Constraints.

- [ ] **Step 1: Add `main()`** (no unit test — verified by a real short render in Step 2)

```python
def main():
    import os, json, argparse
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
    import torch
    import soundfile as sf
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.inference.longform import (
        LongFormRenderer, InpaintContinuationGenerator, PromptSchedule)
    from sa3_control.inject import install_adapters
    from sa3_control.conditioner import ScalarAttributeEncoder
    from sa3_control.generate import load_adapter_state
    from sa3_control.density_schedule import ControlSchedule

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--shape", required=True)
    ap.add_argument("--prompt", default="goa trance, psychedelic, driving, 145 bpm")
    ap.add_argument("--duration", type=float, default=360.0)
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--overlap-sec", type=float, default=5.0)
    ap.add_argument("--lo", type=float, default=2.0); ap.add_argument("--hi", type=float, default=14.0)
    ap.add_argument("--gain", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=777); ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mean, std = ck.get("scalar_norm", [7.219, 1.424])
    n_tokens = min(int(ck["args"].get("n_tokens", 16)), 16)
    control_dim = int(ck["args"].get("control_dim", 768))
    sam = StableAudioModel.from_pretrained("medium-base", device="cuda")
    md = next(sam.model.model.parameters()).dtype
    fps = sam.model.sample_rate / sam.model.pretransform.downsampling_ratio
    wrappers = install_adapters(sam, control_dim=control_dim)
    enc = ScalarAttributeEncoder(control_dim=control_dim, n_tokens=n_tokens)
    load_adapter_state(ck["state"], wrappers, enc)
    for w in wrappers: w.adapter.to(device="cuda", dtype=md)
    enc.to(device="cuda", dtype=md).eval()

    inner = InpaintContinuationGenerator(sam, steps=args.steps, cfg_scale=args.cfg)
    sched = ControlSchedule(args.shape, args.duration, args.lo, args.hi)
    steered = SteeredGenerator(inner, sched, enc, mean, std, args.gain, fps, args.cfg, "cuda", md)
    f = lambda s: int(round(s * fps))
    r = LongFormRenderer(steered, channels=sam.model.io_channels, fps=fps,
                         window_frames=f(args.window_sec), overlap_frames=f(args.overlap_sec))
    lat = r.render_latents(PromptSchedule(args.prompt), total_frames=f(args.duration), base_seed=args.seed)
    with torch.no_grad():
        pt_dtype = next(sam.model.pretransform.parameters()).dtype
        audio = sam.model.pretransform.decode(lat.to(pt_dtype), chunked=True).float().cpu()
    wav = audio[0] if audio.dim() == 3 else audio
    sf.write(args.out, wav.clamp(-1, 1).transpose(0, 1).numpy(), int(sam.model.sample_rate))
    json.dump({"shape": args.shape, "duration": args.duration, "gain": args.gain,
               "scalar_field": ck.get("scalar_field"), "fps": fps,
               "applied": steered.applied}, open(args.out + ".schedule.json", "w"), indent=1)
    print(f"[steered] wrote {args.out}  windows={len(steered.applied)}  "
          f"density {steered.applied[0][1]:.1f}..{min(r[1] for r in steered.applied):.1f}.."
          f"{steered.applied[-1][1]:.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify with a SHORT real render** (GPU; proves wiring end-to-end before a 6-min run)

Run: `cd /home/kim/Projects/SAO/stable-audio-tools/avp_sa3 && FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE /home/kim/Projects/SAO/sa3-rocm7.13-test/.venv/bin/python sa3_control/steered_longform.py <opb_ckpt> --shape triangular --duration 60 --out /home/kim/steer_smoke.wav`
Expected: `[steered] wrote ... windows=3 density 2.0..2.0..` plus a `.schedule.json`; WAV is finite and ~60 s.

- [ ] **Step 3: Commit**

```bash
git add sa3_control/steered_longform.py
git commit -m "feat(sa3_control): steered-longform render entry (medium-base + control schedule)"
```

---

### Task 4: `measure_longform.py` — output density + BPM over time

**Files:**
- Create: `stable-audio-tools/avp_sa3/sa3_control/measure_longform.py` (run in **mir venv**)
- Test: `stable-audio-tools/avp_sa3/sa3_control/tests/test_measure_longform.py`

**Interfaces:**
- Produces: `windowed_metrics(wav_path, win_sec=10.0, hop_sec=5.0) -> list[dict]` with keys `t, onset_density, bpm`.

- [ ] **Step 1: Write the failing test** (synthetic click train → known density)

```python
# tests/test_measure_longform.py
import numpy as np, soundfile as sf, tempfile, os
from sa3_control.measure_longform import windowed_metrics

def test_density_of_click_train(tmp_path):
    sr = 44100; dur = 20.0; rate = 4.0   # 4 clicks/sec
    y = np.zeros(int(sr*dur), np.float32)
    for i in range(int(dur*rate)):
        y[int(i*sr/rate)] = 1.0
    p = str(tmp_path/"clicks.wav"); sf.write(p, y, sr)
    rows = windowed_metrics(p, win_sec=10.0, hop_sec=10.0)
    assert len(rows) >= 1
    assert abs(rows[0]["onset_density"] - rate) < 1.0   # within 1/s of 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kim/Projects/SAO/stable-audio-tools/avp_sa3 && /home/kim/Projects/mir/mir/bin/python -m pytest sa3_control/tests/test_measure_longform.py -v`
Expected: FAIL with `ModuleNotFoundError: sa3_control.measure_longform`.

- [ ] **Step 3: Write minimal implementation**

```python
# sa3_control/measure_longform.py
"""Windowed output metrics for steered long-form audio (mir venv: librosa + essentia)."""
import numpy as np, librosa

def windowed_metrics(wav_path, win_sec=10.0, hop_sec=5.0):
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    win = int(win_sec * sr); hop = int(hop_sec * sr); rows = []
    for start in range(0, max(1, len(y) - win + 1), hop):
        seg = y[start:start + win]
        if len(seg) < win // 2: break
        on = librosa.onset.onset_detect(y=seg, sr=sr, units="time")
        dens = len(on) / (len(seg) / sr)
        tempo = librosa.feature.rhythm.tempo(y=seg, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        rows.append({"t": start / sr, "onset_density": dens, "bpm": bpm})
    return rows

def main():
    import sys, json
    rows = windowed_metrics(sys.argv[1])
    json.dump(rows, open(sys.argv[1] + ".metrics.json", "w"), indent=1)
    print(f"[measure] {len(rows)} windows -> {sys.argv[1]}.metrics.json")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest sa3_control/tests/test_measure_longform.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add sa3_control/measure_longform.py sa3_control/tests/test_measure_longform.py
git commit -m "feat(sa3_control): windowed density+BPM measurement for steered longform"
```

---

### Task 5: plot/viewer

**Files:**
- Create: `~/build_steered_longform_page.py` (mir venv)

**Interfaces:**
- Consumes: each `<out>.schedule.json` (requested S(t)) + `<out>.metrics.json` (measured density+BPM).

- [ ] **Step 1: Implement the page builder**

```python
# ~/build_steered_longform_page.py  — overlays requested S(t) vs measured density + BPM per run
import json, glob, os
runs = []
for sj in sorted(glob.glob("/home/kim/steered_runs/*.wav.schedule.json")):
    base = sj[:-len(".schedule.json")]
    mj = base + ".metrics.json"
    if not os.path.exists(mj): continue
    sched = json.load(open(sj)); metr = json.load(open(mj))
    runs.append({"name": os.path.basename(base), "shape": sched["shape"],
                 "field": sched.get("scalar_field"),
                 "req": [[t, d] for t, d in sched["applied"]],
                 "meas": [[r["t"], r["onset_density"], r["bpm"]] for r in metr]})
html = "<!doctype html><meta charset=utf-8><title>steered longform</title>" + \
       "<body style='font:13px system-ui;background:#0e0e10;color:#ddd'>" + \
       "<h1>Steered long-form density sweeps</h1><div id=app></div>" + \
       "<script src='https://cdn.plot.ly/plotly-2.35.0.min.js'></script><script>" + \
       f"const R={json.dumps(runs)};" + r"""
for(const r of R){const d=document.createElement('div');document.getElementById('app').appendChild(d);
Plotly.newPlot(d,[
 {x:r.req.map(p=>p[0]),y:r.req.map(p=>p[1]),name:'requested',line:{dash:'dot',color:'#7cf'}},
 {x:r.meas.map(p=>p[0]),y:r.meas.map(p=>p[1]),name:'measured density',line:{color:'#5d5'}},
 {x:r.meas.map(p=>p[0]),y:r.meas.map(p=>p[2]),name:'BPM',yaxis:'y2',line:{color:'#e66'}}],
 {title:r.name+' ('+r.field+')',paper_bgcolor:'#0e0e10',plot_bgcolor:'#16181c',font:{color:'#ddd'},
  yaxis:{title:'onset/s'},yaxis2:{title:'BPM',overlaying:'y',side:'right'}});}
"""+"</script></body>"
open("/home/kim/riffer-evals/steered_longform.html","w").write(html)
print("wrote steered_longform.html")
```

- [ ] **Step 2: Run it** (after at least one full render+measure exists)

Run: `/home/kim/Projects/mir/mir/bin/python ~/build_steered_longform_page.py`
Expected: `wrote steered_longform.html`; opening it shows requested-vs-measured density + BPM overlay per run.

- [ ] **Step 3: Commit** (in `riffer-evals`)

```bash
cd /home/kim/riffer-evals && git add steered_longform.html && git commit -m "steered longform viewer: requested vs measured density + BPM"
```

---

## Full-run procedure (after Tasks 1–5 land; GPU)

1. **Triangular A/B:** render triangular on the opb head AND a density head → 2 runs.
2. **Shape variety:** render linear_descending, bimodal, sinewave on the opb head → 3 runs.
3. Measure each (Task 4), build the page (Task 5).
4. Read the page: disentangled (opb) → density tracks S(t), BPM flat; entangled (density) → density tracks, BPM drifts.

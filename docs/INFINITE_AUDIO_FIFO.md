# InfiniteAudio FIFO on Stable Audio 3 — Spec (Phase: prototype)

Status: **design + code written, UNTESTED** (GPU was occupied at authoring time).
Target validation hardware: RX 9070 XT (gfx1201), ROCm. First validation runs on
`small-music-base` (CPU-capable, fast iteration) per `MASTER.md` §3.

Source paper: *InfiniteAudio: Infinite-Length Audio Generation with Consistency*
(Jung et al., 2025, arXiv:2506.03020). Lineage: FIFO-Diffusion (Kim et al., NeurIPS 2024)
for video → adapted to audio LDMs (AudioLDM / VoiceLDM).

---

## 1. What the paper does

Three ideas, in decreasing order of how cleanly they port to SA3:

1. **FIFO / diagonal denoising.** Hold a fixed-size window of `n` latent frames where
   **each frame sits at a different noise level** along the time axis (monotonically
   increasing from front→back). One model forward per step advances every frame one
   notch toward clean; the fully-denoised front frame is **popped** (emitted), and a
   fresh pure-noise frame is **pushed** at the back. Output length is unbounded;
   memory is constant (the window). This is the core mechanism.

2. **Buffer zone.** The first `f` frames of the window are kept clean / unperturbed and
   act as an anchoring context, mitigating the train↔inference gap (the model never
   saw "clean prefix + noisy tail" during training).

3. **Curved denoising.** Skip diffusion steps in σ-regions the model attends to least
   (found via self-attention analysis). Pure efficiency; orthogonal to correctness.
   **Out of scope for this prototype.**

The paper's models (AudioLDM/VoiceLDM) are mel-spectrogram U-Nets on a DDPM schedule.
SA3 is a **rectified-flow DiT** on raw-audio latents. The math carries over; the
empirical "which σ matters" finding (curved denoising) does not and would be re-derived.

---

## 2. The blocker, and why we patch the model

True FIFO requires the network to accept a **per-frame timestep** `(B, T)`. SA3's DiT
does **not**: at `models/dit.py:239` the timestep is embedded once per batch element as
a single global vector `(B, embed_dim)` and folded into the adaLN global conditioning
(`timestep_cond_type="global"`, `global_cond_type="adaLN"` for *all* SA3 variants —
verified by live introspection of `small-music-base`). The samplers' existing
`per_element_schedule` path varies σ **per batch item**, not **per frame** — it does not
help.

So FIFO on SA3 is **not a sampler swap**; it needs the timestep conditioning to become
per-token. Because the model was never trained this way, the result is **out of
distribution** — expect artifacts. This is a research probe, not production.

### Live model facts (introspected, not guessed)

`small-music-base`, fp32, CPU:

| Property | Value | Consequence |
|---|---|---|
| `dit.patch_size` | **1** | 1 latent frame = 1 transformer token. Per-frame σ == per-token σ. Clean. |
| `dit.timestep_cond_type` | `global` | timestep added into the adaLN global cond (`dit.py:243-247`). |
| `dit.global_cond_type` | `adaLN` | global cond drives per-block scale/shift/gate (`transformer.py:1039`). |
| `transformer.num_memory_tokens` | **64** | 64 learned tokens prepended to x (`transformer.py:1212-1214`). Per-token cond must be padded for them. |
| `pretransform.downsampling_ratio` | **4096** | ~10.77 latent fps @ 44.1 kHz. Window of `n` frames ≈ `n/10.77` s. |
| `embed_dim` | 1024 (small) / 1536 (medium) | adaLN block param `to_scale_shift_gate` is `(6*dim,)`. |

---

## 3. Surgery — per-frame timestep conditioning

All changes are **guarded monkey-patches** installed by `install_per_frame_timestep_patch()`.
The original `(B,)` scalar-timestep path is byte-for-byte preserved; the new path
triggers only when a 2-D timestep `(B, T)` is passed. Installing the patch is therefore
safe for normal generation.

Three methods are replaced (originals saved, idempotent install):

### 3a. `DiffusionTransformer._forward` (`models/dit.py:179`)
At the timestep-embed line (`dit.py:239`):
```python
# original:  timestep_embed = self.to_timestep_embed(self.timestep_features(t_cond[:, None]))  # (B, embed)
# patched:   t_cond[..., None] is (B,1) for scalar-t  OR  (B,T,1) for per-frame t.
#            FourierFeatures does `input @ weight.T` (blocks.py:46) → batched matmul →
#            (B,T,1)@(1,F) = (B,T,F); the Linear stack maps last dim → (B,T,embed).
```
Then, for `timestep_cond_type == "global"` with per-frame t, instead of
`global_embed = global_embed + timestep_embed` (`dit.py:245`, both `(B,embed)`), produce a
**per-token** global cond:
```python
gtok = (global_embed[:, None, :] if global_embed is not None else 0) + timestep_embed  # (B,T,embed)
```
and hand it to the transformer as a per-token global cond (see 3b). The scalar path is
unchanged.

`_t_to_logsnr_cond` (`dit.py:143`) is elementwise → works on `(B,T)` unchanged.

### 3b. `ContinuousTransformer.forward` (`models/transformer.py:1180`)
The transformer concatenates `num_memory_tokens` (+ any prepend_embeds) to the **front**
of x (`transformer.py:1205-1214`), so the working sequence length is
`seq = num_memory_tokens + prepend_len + T`. The per-token global cond covers only the
`T` latent tokens, so before the per-block embedder (`transformer.py:1230`) we **left-pad**
it to `seq`:
```python
# gtok: (B, T, embed). prepend rows get a representative cond (we use the σ→σ_max row,
# i.e. the noisiest/back frame's modulation, or equivalently the mean; memory/prepend
# tokens are global aggregators and have no intrinsic noise level).
pad = gtok[:, -1:, :].expand(B, prepend_total, embed)        # or mean over T
gtok_full = cat([pad, gtok], dim=1)                          # (B, seq, embed)
global_cond = self.global_cond_embedder(gtok_full)           # (B, seq, 6*dim)
```
Plumbed via a private attribute set by `_forward` (e.g. `self._fifo_pertoken_gcond`) rather
than changing the public signature, so the patch stays local. The scalar path (2-D
`global_cond`) is untouched.

### 3c. `TransformerBlock.forward` (`models/transformer.py:1039`)
```python
# original:  (self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)
#            global_cond (B,6dim) → +param (6dim) → (B,6dim) → unsqueeze(1) → (B,1,6dim) → broadcast over tokens
# patched:   if global_cond.dim() == 3 (B,seq,6dim): DO NOT unsqueeze; chunk → six (B,seq,dim);
#            x*(1+scale)+shift broadcasts elementwise over the token axis. Per-token modulation.
```

### Out of scope / known incompatibilities of the patch
- **CFG must be done manually** (see §4). The DiT's internal CFG block assumes scalar σ:
  `sigma[0]` scalar comparisons (`dit.py:466,473,479`) and `sigma[:, None, None]`
  broadcasts (`dit.py:577,581`) both break for `(B,T)`. We call the wrapper at
  `cfg_scale=1.0` so control flow takes the clean `else` branch (`dit.py:627`) straight
  to `_forward`, never touching that code.
- **LoRA is incompatible** while a 2-D t is active (`dit.py:466,473` use `sigma[0]`).
  The prototype loads no LoRA; documented, not handled.

---

## 4. FIFO sampler

### Slot schedule (fixed σ per slot, data flows through slots)
`n` = window length in latent frames. Slots indexed `0..n-1`, slot 0 = most denoised.
```
σ_slots[k] monotonically increasing, σ_slots[n-1] = 1.0 (pure noise enters here),
σ_slots[0] = σ_min.
optional front slots σ_slots[0 .. n_buffer-1] pinned at σ_min  (see buffer-zone note).
```
The ramp slots follow **SA3's native schedule** (`build_schedule(...)` warped by the
model's `sampling_dist_shift`, reversed to ascending σ) so each migrating frame's
per-step Δσ corresponds to a real native sampling step — not an ad-hoc ramp.

> **Buffer-zone caveat (default `n_buffer=0`).** Pinning the front slots at σ_min gives
> them Δσ≈0, so they are *frozen held model outputs*, **not** clean ground-truth context.
> This does **not** replicate InfiniteAudio's buffer zone (clean known-good frames that
> close the train/inference gap), and adds no anchoring. The SA3-native anchor is the
> **inpaint clamp** (set `inpaint_mask=1` + `inpaint_masked_input` = most-recently-emitted
> latents on the front slots, shifting them forward each step) — that path is wired but
> currently fed zeros; turning it into a real anchor needs per-step bookkeeping and is
> **deferred to v2**. The prototype ships with the buffer disabled.

### Step (one model forward; manual CFG)
SA3 RF convention (from `sample_discrete_euler`, `sampling.py:147`): `t == σ`,
`v = model(x, t)`, clean `z0 = x − t·v`, Euler `x ← x + dt·v` with `dt = t_next − t_curr < 0`
(descending σ). Per frame:
```
τ           = σ_slots                                  # (n,) → broadcast to (B,n) timestep
v_cond      = wrapper(x, τ, cfg_scale=1.0, **cond_pos) # patched per-frame forward
v_uncond    = wrapper(x, τ, cfg_scale=1.0, **cond_neg) # empty/neg prompt
v           = v_uncond + cfg_scale * (v_cond − v_uncond)
dt[k]       = σ_slots[k-1] − σ_slots[k]   (k≥1);  dt[0] = 0 − σ_slots[0]
x           = x + dt[None,None,:] * v                  # per-frame Euler
out_frame   = x[..., 0]                                # slot-0 now at σ=0 → emit
x[..., :-1] = x[..., 1:].clone()                       # frames advance toward slot 0
x[..., -1]  = randn_like(x[..., -1])                   # enqueue pure noise at σ=1
```
After the shift, slot `k` again holds a frame at level `σ_slots[k]` (the old slot `k+1`
frame, just advanced from `σ_slots[k+1]` to `σ_slots[k]`) — the invariant that makes the
window stationary. To emit `N` frames, run `N` iterations. The **sampling loop** is
constant-memory (fixed window); each iteration is **1 forward at cfg=1.0, else 2**
(manual cond+uncond). Collecting + decoding the emitted latents is linear in output
length — true streaming would pop frames to disk / decode in overlapping chunks.

CFG note: the combine `v = v_uncond + cfg·(v_cond − v_uncond)` is **velocity-space
vanilla CFG**, which is algebraically identical to `dit.py`'s `apg_scale=0` branch for
rectified flow (z0 = x − σ·v is affine), but **differs from the DiT default
`apg_scale=1.0`** (orthogonal-projection APG, `dit.py:599-602`) by ~30% relative at
cfg≈6. FIFO bypasses the DiT CFG path entirely (always calls at `cfg_scale=1.0`), so the
wrapper's `apg_scale` is inert. The unconditional pass uses an **empty-prompt** embedding
(not `generate()`'s zeroed null-embed); identical global timing cond, differing only in
the cross-attn text embedding.

### Warmup
Before the first valid pop, the window must fill: run `n` (or `n − n_buffer`) priming
iterations whose pops are discarded (they come from not-yet-coherent slots). The paper's
buffer zone reduces but does not remove this warmup.

### Conditioning
Built exactly as `generate()` does: `_build_conditioning_dicts` →
`model.conditioner(...)` → `get_conditioning_inputs(...)` (`model.py:161,254,302`).
- `seconds_total` is set to the **window** length (`n / latent_fps`), not the (unbounded)
  output length — the model's "how long" signal should describe what it actually sees.
- `inpaint_mask` (zeros) and `inpaint_masked_input` (zeros) are rebuilt at length `n` and
  injected before `get_conditioning_inputs`. For this model config they ride in as
  **`local_add_cond`** (verified live: `local_add_cond_ids=['inpaint_mask',
  'inpaint_masked_input']`, `input_concat_ids=[]`) — additive, projected per-frame via
  `to_local_embed`, then left-padded to the working sequence; **not** `input_concat_cond`.
- Two dicts are built: positive (prompt) and uncond (empty prompt) for manual CFG.

### Decoding
Popped latent frames accumulate in a list; decode via `pretransform.decode` either in
chunks (streaming) or once at the end (smoke). Note 1 latent frame ≈ 93 ms of audio, so
emit/decode granularity is coarse — fine.

---

## 5. Risks (reviewed 2026-06-18; ✓ = adversarially verified correct)

1. ✓ **Sign of `dt`** vs SA3's descending-σ `x += dt·v` — verified correct.
2. ✓ **Per-token cond padding** / 64 memory-token alignment — verified: latent tokens are
   the last T, modulation is left-padded so row `i` ↔ token `i`.
3. ⚠ **Rotary positions** recomputed per forward over the seq len (`transformer.py:1217`).
   FIFO content migrates through absolute positions as it denoises — the canonical
   FIFO-on-a-non-FIFO-model positional-consistency problem, and SA3 encodes the beat grid
   positionally. **This is the most likely failure mode** (tempo/pitch drift, window-period
   phasing). No cheap in-block fix; real path is *detect → finetune with shifted content*.
   The smoke prints a per-segment drift report (`_positional_drift_report`).
4. ✓ **`global_embed is None`** handled (gtok = timestep only).
5. ✓ **dtype** fp32 t / model-dtype embed preserved.
6. ✓ **CFG combine** — velocity-space vanilla CFG, algebraically identical to the
   `apg_scale=0` branch; differs from the DiT default APG. Documented in §4.
7. **Warmup discard count** — default `warmup = window`; confirm no off-by-one on the
   first emitted frame.
8. ✓ **Window stationarity invariant** after shift+push — verified.
9. ⚠ **Buffer zone** as shipped is frozen-context, not a clean anchor (default disabled).
   See §4 caveat; real anchor = inpaint-clamp (v2).

---

## 6. Validation plan (run when GPU is free)

1. **Unit: per-frame forward parity** (`--parity`). With a *constant* per-frame t (all
   frames = same σ), the patched `(B,T)` forward must match the original `(B,)` forward.
   Gate on **relative** error (<5e-3): the ROCm/MIOpen GEMM noise floor on this
   shape-divergent path is ~1e-3..1e-2 absolute (rel ~1e-3), **not** ~1e-4. A live run
   measured abs 4.9e-3 / rel 1.1e-3 on a *correct* patch.
2. **Smoke: short FIFO gen** on `small-music-base`. Defaults are deliberately small
   (window 64, emit 32 ≈ 3 s; ~192 forwards) for a fast first run — scale up once coherent.
   Assert finite, sane RMS, save WAV, listen; check the drift report. (window 256 / emit
   512 ≈ 1536 forwards ≈ 12 min on the RX 9070 XT — not a CPU smoke.)
3. **Seam check.** Compare FIFO output against the existing inpaint-continuation loop for
   discontinuities (spectrogram + onset envelope).
4. **A/B vs sliding-window** (the non-surgical alternative) on the same prompt/seed.
5. Only if (1)–(3) look coherent: port to `medium-base` on the AMD card.

If per-frame conditioning proves too OOD (likely, untrained), the documented fallback is
**sliding-window continuation** via SA3's native inpainting — same product (unbounded,
constant memory, seamless) without model surgery. A short LoRA/full finetune *with*
per-frame heterogeneous-noise batches would be the path to make true FIFO non-OOD.

---

## 7. Files

- `stable_audio_3/inference/fifo_infinite.py` — patch installer + FIFO sampler + config.
- `scripts/fifo_infinite_smoke.py` — load model, build cond, install patch, run, save WAV.
- this doc.

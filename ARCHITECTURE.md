# stable-audio-3 — architecture (thin fork)

**Read `/home/kim/Projects/SAO/MASTER.md` first** — cross-repo facts (data paths,
venv-per-task, ROCm/RDNA4 gotchas) live there, not here. `SAO/ARCHITECTURE.md` is the
full pipeline map. This file describes only what is *local* to this fork, so other
instances don't recreate tooling that already lives in the master repo.

## Role
Thin fork of Stability-AI **stable-audio-3** (SA3 medium model). Changed **only where
the model interface needs components upstream doesn't provide** — training / inference
glue, LatCH guidance, ONNX/TensorRT export, FIFO streaming. General tooling does **not**
live here.

## What lives HERE (genuine package deltas — preserve on upstream rebase)
- `stable_audio_3/models/latch.py` — SA3 LatCH head + canonical loader
  `load_latch_from_checkpoint` (medium heads: adaln_zero / standardized / depth-4).
- `stable_audio_3/inference/latch_guided.py` + `latch_targets.py` — LatCH-guided
  sampling. DiT forward is under `torch.no_grad()` → only the fp32 head needs grad, so
  the default runs **fp16 + CK flash-attn at ≈ base speed** (MASTER §5; not fp32).
- `stable_audio_3/models/dit.py` / `transformer.py` — attention + APG patches.
- `stable_audio_3/data/dataset.py` — `PreEncodedDataset(beat_aware_crop=…)` +
  `_get_downbeat_starts` (downbeat-aligned crops from per-latent `.TIMESERIES.npz`).
- `stable_audio_3/training/diffusion.py` — full-state checkpointing (resumable).
- `stable_audio_3/rocm_env.py` — ROCm env (mirrors SAT's canonical profile).
- Local ROCm wheels + venv plumbing (`pyproject.toml` / `uv.lock`, ROCm-7.15 + CK FA).
- Upstream scripts we still drive from here: `scripts/train_lora.py`,
  `pre_encode_dataset.py`, `precache_dit_cond.py`.

## What does NOT live here — canonical home is the SAO master repo
The ONNX suite, `sa3_control` adapter training, LatCH-head training, and eval / render
/ soup tooling live in `SAO/{onnx,control,latch,eval}/`. **Look there before writing an
export or eval.** Some legacy copies still sit under `scripts/` here (pre-thinning) —
prefer the SAO copy. `eval_dora_*` and `soup_*` are canonical in `SAO/eval/`.

## Findings & work log
Cross-repo findings → `SAO/WORKLOG.md`. Local analyses (e.g.
`checkpoint_steering_analysis.md`) stay here. Append when you finish, so the next
instance doesn't redo it.

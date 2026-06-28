#!/usr/bin/env python
"""make_text_cond.py — regenerate the TEXT cond/uncond npzs the control-DiT ONNX
runner (dit_control_onnx_infer.py) consumes as --cond / --uncond.

The stock precache_dit_cond.py is broken on this fork (cuda/cpu dtype mismatch +
inpaint_mask KeyError), so this is the cdm.conditioner()-direct workaround: run the
T5-Gemma conditioner on CPU and dump exactly the three arrays the runner's load_cond
expects — cross_attn_cond[SEQ,768], cross_attn_mask[SEQ], global_embed[768] — pad/
truncate the cross seq to SEQ. CPU-only (text encode is cheap), leaves the GPU free.

These are NOT the FiLM .cond.npz (the scalar control conditioner, saved beside the
.onnx at export). Distinct files, distinct runner flags.

    SA3 venv:  /home/kim/Projects/SAO/stable-audio-3/.venv/bin/python
    python scripts/make_text_cond.py            # writes the two npzs beside the onnx
"""
import os
os.environ["SA3_DISABLE_FLASH_ATTN"] = "1"
from pathlib import Path

import numpy as np
import torch
from stable_audio_3 import StableAudioModel

# Must match the fp16 control-DiT export: dit_medium-base_L256_ctrl_onset_density_fp16.onnx
PROMPT = "goa trance, psychedelic, driving bassline, 145 bpm"
SECONDS = 23.8          # seconds_total handed to the conditioner (≈ T=256 latent frames)
SEQ = 128              # cross-attn token budget (pad/truncate); matches the export
T = 256                # latent frames — L256 rung; only needed to satisfy get_conditioning_inputs
HERE = Path(__file__).resolve().parent.parent     # stable-audio-3/
COND_OUT = HERE / "dit_medium-base_L256_ctrl.cond.npz"
UNCOND_OUT = HERE / "dit_medium-base_L256_ctrl.uncond.npz"


def load_conditioner(model_name: str = "medium-base"):
    """Load the SA3 DiT+conditioner module on CPU/fp32 (WORKAROUND 1: flash attn off,
    cuda/cpu dtype mismatch avoided). Returns cdm = model.model.

    device="cpu" + model_half=False so the 1.4B weights load STRAIGHT to CPU and never
    touch cuda — otherwise from_pretrained defaults to cuda (loading gigabytes before any
    .to("cpu")) and OOMs when the GPU is busy with a training run. The GPU stays *visible*
    (do NOT hide it via HIP_VISIBLE_DEVICES — that breaks the flash_attn/aiter import,
    which probes a driver at import time); we simply never allocate on it."""
    os.environ["SA3_DISABLE_FLASH_ATTN"] = "1"
    model = StableAudioModel.from_pretrained(model_name, device="cpu", model_half=False)
    return model.model.to("cpu").float()


def build_text_cond(cdm, prompt: str, seconds: float, seq: int, T: int):
    """Run the T5-Gemma conditioner and return (cross, mask, glob) pad/truncated to seq."""
    with torch.no_grad():
        ct = cdm.conditioner(
            [{"prompt": prompt, "seconds_start": 0.0, "seconds_total": seconds}], "cpu")
        # stock get_conditioning_inputs KeyErrors on inpaint_mask while building local_add_cond;
        # feed it zero inpaint tensors (we don't use its local_add_cond — the runner zeros that).
        ct["inpaint_mask"] = [torch.zeros(1, 1, T)]
        ct["inpaint_masked_input"] = [torch.zeros(1, 256, T)]
        ci = cdm.get_conditioning_inputs(ct)
    cross = ci["cross_attn_cond"][0].float().numpy().astype("float32")
    mask = ci["cross_attn_mask"][0].numpy().astype(bool)
    glob = ci["global_cond"][0].float().numpy().astype("float32")
    S = cross.shape[0]
    if S < seq:
        pc = np.zeros((seq, cross.shape[1]), "float32"); pc[:S] = cross
        pm = np.zeros((seq,), bool); pm[:S] = mask
        cross, mask = pc, pm
    else:
        cross, mask = cross[:seq], mask[:seq]
    return cross, mask, glob


def build(cdm, prompt: str):
    return build_text_cond(cdm, prompt, SECONDS, SEQ, T)


def main():
    print("[load] medium-base on CPU (incl T5-Gemma conditioner) ...", flush=True)
    cdm = load_conditioner("medium-base")

    c = build(cdm, PROMPT)
    u = build(cdm, "")
    np.savez(COND_OUT, cross_attn_cond=c[0], cross_attn_mask=c[1], global_embed=c[2])
    np.savez(UNCOND_OUT, cross_attn_cond=u[0], cross_attn_mask=u[1], global_embed=u[2])
    print(f"[cond]   {PROMPT!r}")
    print(f"         cross{c[0].shape} mask(sum={int(c[1].sum())}/{SEQ}) glob{c[2].shape}")
    print(f"[uncond] '' cross{u[0].shape} mask(sum={int(u[1].sum())}/{SEQ}) glob{u[2].shape}")
    print(f"[out] {COND_OUT}")
    print(f"[out] {UNCOND_OUT}")


if __name__ == "__main__":
    main()

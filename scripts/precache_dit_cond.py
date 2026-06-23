#!/usr/bin/env python
"""precache_dit_cond.py — precompute the DiT conditioning (T5-Gemma cross-attn +
NumberConditioner global) for a prompt+duration, offline in the SA3 venv, so the
ONNX runtime (dit_onnx_infer.py, mir venv) needs no torch / T5-Gemma.

Emits a `cond.npz` (the prompt) and an `uncond.npz` (empty/negative prompt) each:
  cross_attn_cond  float32 [seq,768]   — T5-Gemma embeddings, padded/truncated to --text-seq
  cross_attn_mask  bool    [seq]
  global_embed     float32 [768]       — NumberConditioner over seconds_start/total

The DiT's `_forward` projects these internally, so we save the raw conditioner output
(matches export_dit_onnx.py's input contract).

Usage
-----
    python scripts/precache_dit_cond.py --prompt "earth chakra, 1996, 123" \\
        --seconds 24 --text-seq 128 --out-prefix /tmp/earthchakra
    # -> /tmp/earthchakra.cond.npz  /tmp/earthchakra.uncond.npz
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("SA3_DISABLE_FLASH_ATTN", "1")

import numpy as np
import torch


def pad_seq(emb: np.ndarray, mask: np.ndarray, seq: int):
    """emb [S,768], mask [S] -> padded/truncated to [seq,768],[seq]."""
    S = emb.shape[0]
    if S >= seq:
        return emb[:seq], mask[:seq]
    pe = np.zeros((seq, emb.shape[1]), emb.dtype); pe[:S] = emb
    pm = np.zeros((seq,), mask.dtype); pm[:S] = mask
    return pe, pm


def build(model, prompt, seconds, seq, device):
    cdm = model.model
    cond = [{"prompt": prompt, "seconds_start": 0.0, "seconds_total": float(seconds)}]
    with torch.no_grad():
        ct = cdm.conditioner(cond, device)
        ci = cdm.get_conditioning_inputs(ct)
    cross = ci["cross_attn_cond"][0].float().cpu().numpy()        # [S,768]
    mask = ci["cross_attn_mask"][0].cpu().numpy().astype(bool)    # [S]
    glob = ci["global_cond"][0].float().cpu().numpy()             # [768]
    cross, mask = pad_seq(cross.astype(np.float32), mask, seq)
    return cross, mask, glob.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--neg-prompt", default="", help="negative/uncond prompt (default empty)")
    ap.add_argument("--seconds", type=float, required=True, help="seconds_total (duration)")
    ap.add_argument("--text-seq", type=int, default=128, help="fixed seq (match the DiT export)")
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>.cond.npz / .uncond.npz")
    args = ap.parse_args()

    from stable_audio_3 import StableAudioModel
    print(f"[load] {args.model} (CPU; includes T5-Gemma) ...")
    model = StableAudioModel.from_pretrained(args.model)
    dev = "cpu"

    for tag, prompt in [("cond", args.prompt), ("uncond", args.neg_prompt)]:
        cross, mask, glob = build(model, prompt, args.seconds, args.text_seq, dev)
        out = Path(f"{args.out_prefix}.{tag}.npz")
        np.savez(out, cross_attn_cond=cross, cross_attn_mask=mask, global_embed=glob)
        print(f"[{tag}] {prompt!r:40s} -> {out}  cross{cross.shape} mask(sum={int(mask.sum())}) glob{glob.shape}")


if __name__ == "__main__":
    main()

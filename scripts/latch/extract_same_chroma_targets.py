#!/usr/bin/env python3
"""
extract_same_chroma_targets.py -- per-crop 384-d SAME-chroma targets for the
384d guidance head (task #48; CHROMA_CONTROL_HEAD_PLAN.md).

SAME's latent is chroma-regularized by its training objective, so chroma is
linearly decodable: target[b] = W[b] @ z + bias[b] per frame, with the refit
readout from mir-same-chroma/sc_run/chroma_heads_real.npz (W (3,128,256),
b (3,128); bands = octaves 1/5/9). Applying the readout to the CLEAN latent
gives the target; the guidance head then learns to predict it from NOISED
latents across the RF schedule (the readout itself only works at t=0 -- that
gap is the whole reason the head exists).

CPU-only, numpy. Writes <stem>.npz {full_mix: (3,128,T) float16} next to
nothing -- into --out-dir, the trainer's --chroma-dir.

Run:  python3 extract_same_chroma_targets.py [--latent-dir ...] [--out-dir ...]
"""
import argparse
from pathlib import Path

import numpy as np

READOUT = "/home/kim/Projects/mir-same-chroma/sc_run/chroma_heads_real.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-dir", default="/home/kim/Projects/latents_sa3")
    ap.add_argument("--out-dir", default="/home/kim/Projects/latents_sa3_chroma")
    ap.add_argument("--readout", default=READOUT)
    args = ap.parse_args()

    z = np.load(args.readout)
    W, b = z["W"].astype(np.float32), z["b"].astype(np.float32)   # (3,128,256), (3,128)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    items = sorted(Path(args.latent_dir).glob("*.npy"))
    n_new = 0
    for p in items:
        dst = out / (p.stem + ".npz")
        if dst.exists():
            continue
        lat = np.load(p).astype(np.float32)          # (256, T) or (1, 256, T)
        if lat.ndim == 3:
            lat = lat[0]
        # (3,128,256) x (256,T) -> (3,128,T)
        tgt = np.einsum("bkc,ct->bkt", W, lat) + b[:, :, None]
        np.savez_compressed(dst, full_mix=tgt.astype(np.float16))
        n_new += 1
        if n_new % 500 == 0:
            print(f"[chroma-targets] {n_new} done", flush=True)
    print(f"[chroma-targets] DONE: {n_new} new, {len(items)} total -> {out}", flush=True)


if __name__ == "__main__":
    main()

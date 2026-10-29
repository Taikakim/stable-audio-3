#!/usr/bin/env python3
"""
extract_avp_same_chroma.py -- parallel per-crop 384-d SAME-chroma targets for
the AVP augmented dataset (latents_avp, 2393 crops).

Applies the canonical linear SAME readout W[b] @ z + bias[b] from
/home/kim/Projects/mir-same-chroma/sc_run/chroma_heads_real.npz
to clean latents (256, T) -> (3, 128, T).
Saves .npz with keys 'other' and 'full_mix' into --out-dir.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

READOUT = "/home/kim/Projects/mir-same-chroma/sc_run/chroma_heads_real.npz"

# Globals for workers
_W = None
_b = None
_out_dir = None


def init_worker(readout_path, out_dir):
    global _W, _b, _out_dir
    z = np.load(readout_path)
    _W = z["W"].astype(np.float32)  # (3, 128, 256)
    _b = z["b"].astype(np.float32)  # (3, 128)
    _out_dir = Path(out_dir)


def process_item(p_str):
    p = Path(p_str)
    dst = _out_dir / (p.stem + ".npz")
    if dst.exists():
        return 0  # skipped
    try:
        lat = np.load(p).astype(np.float32)
        if lat.ndim == 3:
            lat = lat[0]
        # (3,128,256) x (256,T) -> (3,128,T)
        tgt = np.einsum("bkc,ct->bkt", _W, lat) + _b[:, :, None]
        tgt_f16 = tgt.astype(np.float16)
        np.savez_compressed(dst, other=tgt_f16, full_mix=tgt_f16)
        return 1  # processed
    except Exception as e:
        print(f"Error on {p}: {e}", flush=True)
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-dir", default="/run/media/kim/Kosmos/latents/latents_avp")
    ap.add_argument("--out-dir", default="/run/media/kim/Lehto/latents_avp_chroma")
    ap.add_argument("--readout", default=READOUT)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    items = sorted(Path(args.latent_dir).glob("*.npy"))
    print(f"[chroma-targets] Found {len(items)} latent files in {args.latent_dir}, using {args.workers} workers")
    t0 = time.time()

    item_strs = [str(p) for p in items]
    n_new = 0
    n_skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args.readout, args.out_dir)) as pool:
        for i, res in enumerate(pool.map(process_item, item_strs, chunksize=16)):
            if res == 1:
                n_new += 1
            elif res == 0:
                n_skipped += 1
            if (i + 1) % 500 == 0 or (i + 1) == len(items):
                dt = time.time() - t0
                rate = (i + 1) / max(1, dt)
                print(f"[chroma-targets] {i + 1}/{len(items)} ({n_new} new, {n_skipped} skipped) [{rate:.1f} crops/s]", flush=True)

    dt = time.time() - t0
    print(f"[chroma-targets] DONE: {n_new} written, {n_skipped} existing in {dt:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()

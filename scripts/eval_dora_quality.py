#!/home/kim/Projects/mir/mir/bin/python
"""eval_dora_quality.py — POST-RUN quality scorer for the SA3 DoRA finetune.

Fires AFTER training finishes and the GPU frees. Given a render dir produced by
``scripts/eval_dora_cpu.py`` (3 prompts x seed 1234 per checkpoint, named
``<tag>__p<i>_seed<seed>.wav``), it scores every checkpoint's clips on two axes:

  (a) **Audiobox Aesthetics** (CE / PQ / PC / CU, 1-10) via mir's
      ``timbral.audiobox_aesthetics.analyze_audiobox_aesthetics`` in **single-file
      mode** (batch mode OOMs WavLM on 16 GB). One forward per clip.

  (b) **MERT distance-to-Goa** — how close each checkpoint sits to the real Goa
      corpus in MERT embedding space. Two complementary numbers per checkpoint:
        * **frechet**  — FAD-style Fréchet distance (mean+cov) between the
          checkpoint's render embeddings and the Goa reference embeddings.
          ``d^2 = ||mu_r - mu_g||^2 + Tr(Cr + Cg - 2*sqrt(Cr*Cg))``.
        * **cos_dist** — ``1 - mean_i cos(emb_i, goa_centroid)`` (robust with the
          3-clip render sets; the Fréchet covariance term is degenerate there).
      Lower = closer to Goa. The best checkpoint = lowest distance at good CE.

MERT layer space (reuses the latch-eval embedder ``sa3_control.mert_selector.MERTEmbedder``):
  primary distance is computed on the **mid** layer group **(3,4,5,6)** — groove /
  rhythm / timbre, the genre-discriminative band for goa. The **upper** layer (23,
  melody/harmony) cosine is reported alongside as a secondary signal. Each
  per-clip embedding is mean-pooled over time and L2-normalized (the embedder's
  own output).

The **Goa reference embeddings are cached to disk** (``--cache-dir``) keyed by
(corpus, n_ref, ref_seconds, ref_seed, layers) so they are computed ONCE and
reused across every checkpoint and every future run.

VENV: run with the **mir venv** ``/home/kim/Projects/mir/mir/bin/python`` (has
Audiobox + transformers/torchaudio for MERT + scipy for the Fréchet sqrtm).

Usage
-----
    MV=/home/kim/Projects/mir/mir/bin/python

    # dry-validate WITHOUT touching the GPU/models (paths, globbing, loaders):
    $MV scripts/eval_dora_quality.py --render-dir renders_dora --dry-run

    # the real GPU eval (after training frees the card):
    $MV scripts/eval_dora_quality.py --render-dir renders_dora

Output: a per-checkpoint table to stdout + a JSON sidecar (``--out``, default
``<render-dir>/quality_eval.json``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- repo seams ---------------------------------------------------------------
AVP_SA3 = "/home/kim/Projects/SAO/stable-audio-tools/avp_sa3"
MIR_SRC = "/home/kim/Projects/mir/src"
DEFAULT_GOA_ROOT = "/run/media/kim/Mantu/ai-music/Goa_Separated"
DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "sa3_dora_eval")

for _p in (AVP_SA3, MIR_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
# Clip discovery / grouping
# --------------------------------------------------------------------------- #
def group_render_clips(render_dir: Path) -> dict[str, list[Path]]:
    """Group ``<tag>__p<i>_seed<seed>.wav`` clips by checkpoint tag (prefix before
    ``__``). Falls back to a single 'all' group for clips with no ``__``."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for wav in sorted(render_dir.rglob("*.wav")):
        tag = wav.name.split("__", 1)[0] if "__" in wav.name else "all"
        groups[tag].append(wav)
    return dict(groups)


# --------------------------------------------------------------------------- #
# Goa reference corpus
# --------------------------------------------------------------------------- #
def find_goa_tracks(goa_root: Path) -> list[Path]:
    """All ``full_mix.flac`` under Goa_Separated, or ``*_0.flac``-style crops under
    a goa_crops-shaped tree. Returns a sorted, deterministic list."""
    tracks = sorted(goa_root.rglob("full_mix.flac"))
    if tracks:
        return tracks
    # goa_crops fallback: top-level audio crops (skip stem files: *_bass/_drums/...)
    stem_suffixes = ("_bass", "_drums", "_other", "_vocals")
    crops = [
        p for p in sorted(goa_root.rglob("*.flac"))
        if not any(p.stem.endswith(s) for s in stem_suffixes)
    ]
    return crops


def sample_goa_refs(tracks: list[Path], n_ref: int, seed: int) -> list[Path]:
    """Deterministic random sample of n_ref reference tracks."""
    rng = np.random.default_rng(seed)
    if len(tracks) <= n_ref:
        return list(tracks)
    idx = rng.choice(len(tracks), size=n_ref, replace=False)
    return [tracks[int(i)] for i in sorted(idx)]


def cache_path(cache_dir: Path, goa_root: Path, n_ref: int, ref_seconds: float,
               ref_seed: int) -> Path:
    key = f"{goa_root.name}_n{n_ref}_s{ref_seconds:g}_seed{ref_seed}_mid3456_up23"
    return cache_dir / f"goa_mert_ref__{key}.npz"


# --------------------------------------------------------------------------- #
# Audio loading (mir-free; soundfile + center/window crop)
# --------------------------------------------------------------------------- #
def load_mono_window(path: Path, seconds: float | None) -> tuple[np.ndarray, int]:
    """Load a mono float32 waveform. If ``seconds`` is set, take a centered window
    of that length (keeps MERT comparable to the ~47 s render clips and bounds
    compute on multi-minute full mixes)."""
    import soundfile as sf

    info = sf.info(str(path))
    sr = info.samplerate
    if seconds is not None and info.frames > int(seconds * sr):
        win = int(seconds * sr)
        start = max(0, (info.frames - win) // 2)
        x, sr = sf.read(str(path), start=start, frames=win, dtype="float32",
                        always_2d=True)
    else:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(axis=1), int(sr)


# --------------------------------------------------------------------------- #
# Fréchet distance (FAD-style, regularized for high-dim / low-sample)
# --------------------------------------------------------------------------- #
def frechet_distance(mu1, cov1, mu2, cov2, eps: float = 1e-6) -> float:
    """Fréchet distance between two Gaussians (the FID/FAD formula). Covariances
    are diagonally regularized by ``eps`` because MERT is 1024-d while the
    reference (~150) and especially the render (3) sample counts give singular
    covariances. Port of the canonical pytorch-fid implementation."""
    from scipy import linalg

    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    cov1, cov2 = np.atleast_2d(cov1), np.atleast_2d(cov2)
    diff = mu1 - mu2

    offset = np.eye(cov1.shape[0]) * eps
    covmean, _ = linalg.sqrtm((cov1 + offset) @ (cov2 + offset), disp=False)
    if not np.isfinite(covmean).all():
        covmean, _ = linalg.sqrtm((cov1 + offset * 100) @ (cov2 + offset * 100),
                                  disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1) + np.trace(cov2)
                 - 2.0 * np.trace(covmean))


def gaussian_stats(embs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean + covariance of a (N, D) embedding stack (rowvar=False)."""
    mu = embs.mean(axis=0)
    cov = np.cov(embs, rowvar=False) if embs.shape[0] > 1 else np.zeros(
        (embs.shape[1], embs.shape[1]))
    return mu, cov


# --------------------------------------------------------------------------- #
# Embedding (real model path — MERT via the latch-eval embedder)
# --------------------------------------------------------------------------- #
def build_embedder(device: str):
    from sa3_control.mert_selector import MERTEmbedder
    return MERTEmbedder(device=device)


def embed_paths(embedder, paths: list[Path], seconds: float | None,
                tag: str = "") -> dict[str, np.ndarray]:
    """Embed each path -> stacked (N, D) arrays for 'mid' and 'upper'."""
    mids, ups = [], []
    for i, p in enumerate(paths):
        wav, sr = load_mono_window(p, seconds)
        emb = embedder.embed(wav, sr)
        mids.append(np.asarray(emb["mid"], dtype=np.float64))
        ups.append(np.asarray(emb["upper"], dtype=np.float64))
        if tag:
            print(f"  [{tag}] {i + 1}/{len(paths)} {p.name}", flush=True)
    return {"mid": np.stack(mids), "upper": np.stack(ups)}


def load_or_build_goa_ref(args, embedder) -> dict:
    cdir = Path(args.cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    cpath = cache_path(cdir, Path(args.goa_root), args.n_ref, args.ref_seconds,
                       args.ref_seed)
    if cpath.exists() and not args.refresh_cache:
        print(f"[goa-ref] loading cached embeddings: {cpath}", flush=True)
        z = np.load(cpath, allow_pickle=True)
        return {"mid": z["mid"], "upper": z["upper"],
                "files": list(z["files"]), "centroid_mid": z["centroid_mid"],
                "centroid_upper": z["centroid_upper"]}

    tracks = find_goa_tracks(Path(args.goa_root))
    refs = sample_goa_refs(tracks, args.n_ref, args.ref_seed)
    print(f"[goa-ref] embedding {len(refs)} Goa clips (cache miss) ...", flush=True)
    t0 = time.time()
    embs = embed_paths(embedder, refs, args.ref_seconds, tag="goa")
    centroid_mid = embs["mid"].mean(axis=0)
    centroid_mid /= max(np.linalg.norm(centroid_mid), 1e-8)
    centroid_upper = embs["upper"].mean(axis=0)
    centroid_upper /= max(np.linalg.norm(centroid_upper), 1e-8)
    np.savez(cpath, mid=embs["mid"], upper=embs["upper"],
             files=np.array([str(p) for p in refs]),
             centroid_mid=centroid_mid, centroid_upper=centroid_upper)
    print(f"[goa-ref] cached {len(refs)} embeddings to {cpath} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return {"mid": embs["mid"], "upper": embs["upper"],
            "files": [str(p) for p in refs], "centroid_mid": centroid_mid,
            "centroid_upper": centroid_upper}


def mean_cosine_dist(embs: np.ndarray, centroid: np.ndarray) -> float:
    """1 - mean cosine(emb_i, centroid). Embeddings + centroid are L2-normalized."""
    c = centroid / max(np.linalg.norm(centroid), 1e-8)
    cos = embs @ c / (np.linalg.norm(embs, axis=1) + 1e-8)
    return float(1.0 - cos.mean())


# --------------------------------------------------------------------------- #
# Dry validation (NO models, NO GPU)
# --------------------------------------------------------------------------- #
def dry_run(args) -> int:
    render_dir = Path(args.render_dir)
    print("=== DRY VALIDATION (no GPU, no models loaded) ===")
    ok = True

    # 1. render dir + clip globbing
    if not render_dir.is_dir():
        print(f"[FAIL] render-dir not found: {render_dir}")
        ok = False
        groups = {}
    else:
        groups = group_render_clips(render_dir)
        n_clips = sum(len(v) for v in groups.values())
        print(f"[ok] render-dir: {render_dir}")
        print(f"[ok] {len(groups)} checkpoint group(s), {n_clips} clip(s):")
        for tag, clips in list(groups.items())[:10]:
            print(f"       {tag}: {len(clips)} clips  e.g. {clips[0].name}")
        if not groups:
            print("[warn] no .wav clips yet — run eval_dora_cpu.py first "
                  "(this is expected pre-training-finish).")

    # 2. Goa corpus path + reference sample
    goa_root = Path(args.goa_root)
    if not goa_root.is_dir():
        print(f"[FAIL] goa-root not found (drive unmounted?): {goa_root}")
        ok = False
    else:
        tracks = find_goa_tracks(goa_root)
        if not tracks:
            print(f"[FAIL] no audio found under {goa_root}")
            ok = False
        else:
            refs = sample_goa_refs(tracks, args.n_ref, args.ref_seed)
            print(f"[ok] goa-root: {goa_root}")
            print(f"[ok] {len(tracks)} candidate tracks -> sampling "
                  f"{len(refs)} refs (seed={args.ref_seed}, "
                  f"{args.ref_seconds:g}s window each)")
            print(f"       e.g. {refs[0]}")
            print(f"       e.g. {refs[len(refs) // 2]}")

    # 3. cache location
    cpath = cache_path(Path(args.cache_dir), goa_root, args.n_ref,
                       args.ref_seconds, args.ref_seed)
    print(f"[ok] goa-ref cache path: {cpath} "
          f"({'EXISTS — will reuse' if cpath.exists() else 'will be built once'})")

    # 4. loaders importable (import only — do NOT instantiate / load weights)
    try:
        from sa3_control.mert_selector import MERTEmbedder, LAYERS_MID, LAYERS_UPPER  # noqa: F401
        print(f"[ok] MERT embedder import: sa3_control.mert_selector.MERTEmbedder "
              f"(mid={LAYERS_MID}, upper={LAYERS_UPPER}, MERT-v1-330M)")
    except Exception as e:
        print(f"[FAIL] MERT embedder import: {e}")
        ok = False
    try:
        from timbral.audiobox_aesthetics import (  # noqa: F401
            analyze_audiobox_aesthetics, AUDIOBOX_AVAILABLE)
        print(f"[ok] Audiobox import: timbral.audiobox_aesthetics."
              f"analyze_audiobox_aesthetics (available={AUDIOBOX_AVAILABLE})")
    except Exception as e:
        print(f"[FAIL] Audiobox import: {e}")
        ok = False
    try:
        from scipy import linalg  # noqa: F401
        print("[ok] scipy.linalg.sqrtm available (Fréchet)")
    except Exception as e:
        print(f"[FAIL] scipy import: {e}")
        ok = False

    print(f"\n=== DRY VALIDATION: {'PASS' if ok else 'FAIL'} ===")
    print("Distance-to-Goa = MERT mid-layer (3,4,5,6) Fréchet (FAD-style mean+cov)"
          " + (1 - mean cosine to Goa centroid); lower = closer to Goa.")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Real run
# --------------------------------------------------------------------------- #
def run(args) -> int:
    render_dir = Path(args.render_dir)
    groups = group_render_clips(render_dir)
    if not groups:
        print(f"[error] no .wav clips under {render_dir}")
        return 1

    embedder = build_embedder(args.mert_device)
    goa = load_or_build_goa_ref(args, embedder)
    mu_g, cov_g = gaussian_stats(goa["mid"])

    audiobox = None
    if not args.no_audiobox:
        from timbral.audiobox_aesthetics import analyze_audiobox_aesthetics
        audiobox = analyze_audiobox_aesthetics

    rows = []
    for tag, clips in groups.items():
        print(f"\n=== {tag} ({len(clips)} clips) ===", flush=True)
        embs = embed_paths(embedder, clips, args.ref_seconds, tag=tag)
        mu_r, cov_r = gaussian_stats(embs["mid"])
        frechet = frechet_distance(mu_r, cov_r, mu_g, cov_g)
        cos_dist_mid = mean_cosine_dist(embs["mid"], goa["centroid_mid"])
        cos_dist_upper = mean_cosine_dist(embs["upper"], goa["centroid_upper"])

        ab = {"CE": None, "PQ": None, "PC": None, "CU": None}
        if audiobox is not None:
            ces, pqs, pcs, cus = [], [], [], []
            for c in clips:
                r = audiobox(c)  # single-file mode (batch OOMs WavLM @16GB)
                ces.append(r["content_enjoyment"]); pqs.append(r["production_quality"])
                pcs.append(r["production_complexity"]); cus.append(r["content_usefulness"])
            ab = {"CE": float(np.mean(ces)), "PQ": float(np.mean(pqs)),
                  "PC": float(np.mean(pcs)), "CU": float(np.mean(cus))}

        rows.append({"tag": tag, "n_clips": len(clips),
                     "frechet_mid": frechet, "cos_dist_mid": cos_dist_mid,
                     "cos_dist_upper": cos_dist_upper, **ab})

    rows.sort(key=lambda r: r["cos_dist_mid"])

    out = Path(args.out) if args.out else render_dir / "quality_eval.json"
    meta = {"goa_root": args.goa_root, "n_ref": len(goa["files"]),
            "ref_seconds": args.ref_seconds, "ref_seed": args.ref_seed,
            "mert_layers_mid": [3, 4, 5, 6], "mert_layers_upper": [23],
            "distance_def": "frechet(mid) FAD-style + (1-mean cos to centroid); "
                            "lower=closer to Goa"}
    out.write_text(json.dumps({"meta": meta, "checkpoints": rows}, indent=2))

    # table
    print("\n" + "=" * 96)
    hdr = (f"{'checkpoint':<34}{'CE':>6}{'PQ':>6}{'PC':>6}{'CU':>6}"
           f"{'frechet':>11}{'cosD_mid':>10}{'cosD_up':>9}")
    print(hdr); print("-" * 96)
    for r in rows:
        def f(x, p=2): return f"{x:.{p}f}" if x is not None else "  -  "
        print(f"{r['tag'][:33]:<34}{f(r['CE']):>6}{f(r['PQ']):>6}{f(r['PC']):>6}"
              f"{f(r['CU']):>6}{f(r['frechet_mid'],3):>11}"
              f"{f(r['cos_dist_mid'],4):>10}{f(r['cos_dist_upper'],4):>9}")
    print("=" * 96)
    print("sorted by cos_dist_mid (closest to Goa first). best = low distance @ good CE.")
    print(f"[written] {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render-dir", type=Path, required=True,
                    help="dir of <tag>__p<i>_seed<seed>.wav clips (eval_dora_cpu.py output)")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON output (default <render-dir>/quality_eval.json)")
    ap.add_argument("--goa-root", default=DEFAULT_GOA_ROOT,
                    help="Goa corpus root (full_mix.flac tree, or goa_crops)")
    ap.add_argument("--n-ref", type=int, default=150, help="# Goa reference clips")
    ap.add_argument("--ref-seconds", type=float, default=47.0,
                    help="window length embedded per clip (match render duration)")
    ap.add_argument("--ref-seed", type=int, default=0, help="Goa sample seed")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                    help="where Goa reference embeddings are cached")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="recompute the Goa reference embeddings")
    ap.add_argument("--mert-device", default="cuda", help="cuda|cpu for MERT")
    ap.add_argument("--no-audiobox", action="store_true",
                    help="skip Audiobox (MERT distance only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate paths/globbing/loaders WITHOUT loading models/GPU")
    args = ap.parse_args()

    return dry_run(args) if args.dry_run else run(args)


if __name__ == "__main__":
    raise SystemExit(main())

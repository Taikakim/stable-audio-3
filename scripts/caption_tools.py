"""Tiered caption tools (docs/prompting-conditioning-plan.md steps 4-5).

T1 = deterministic template from features, ERA-FRONTED (Kim: within goa, year
steers strongest): "mid 90s goa trance, psy-trance, space energetic mood, 148 bpm".
T2 = Granite-compressed Flamingo (filled by the caption pass), T3 = raw Flamingo.

Captions live in SIDECAR files (never in the latent jsons — pristine-layout
rule), keyed by latent index stem. `make_caption_sampler` builds the
custom_metadata_fn that PreEncodedDataset already supports: it samples a tier
per __getitem__ call and overrides `prompt`, so one track narrates itself
differently across epochs (cheap caption augmentation, hallucination dilution).

Tested in tests/test_caption_tools.py.
"""
from __future__ import annotations

import json
import os
import random


def era_bucket(year):
    """Human musical-era bucket, or None when unknown."""
    if not year:
        return None
    y = int(year)
    if y <= 1994:
        return "early 90s"
    if y <= 1997:
        return "mid 90s"
    if y <= 2000:
        return "late 90s"
    if y <= 2004:
        return "early 2000s"
    if y <= 2007:
        return "mid 2000s"
    if y <= 2010:
        return "late 2000s"
    if y <= 2019:
        return "2010s"
    return "2020s"


def build_t1(genres, moods, bpm, year):
    """Era-fronted comma-separated descriptor template (SA3 prompt style)."""
    parts = []
    era = era_bucket(year)
    glist = [g.lower() for g in genres]
    head = " ".join(x for x in [era, glist[0] if glist else None] if x)
    if head:
        parts.append(head)
    parts.extend(glist[1:])
    if moods:
        parts.append(" ".join(m.lower() for m in moods) + " mood")
    if bpm:
        parts.append(f"{round(bpm)} bpm")
    return ", ".join(parts)


TRACK_TYPE_MUSIC = "TrackType: Music, VocalType: Instrumental, "


def make_caption_sampler(sidecar_path, probs=(0.6, 0.3, 0.1), seed=None,
                         track_type_prob=0.0):
    """custom_metadata_fn for PreEncodedDataset: sample a caption tier per call.

    Tiers with missing text fall back to t1. Unknown index -> {} (dataset keeps
    its stored prompt). Deterministic when `seed` is given (own RNG stream, so
    worker seeding elsewhere is untouched). Closure state survives dill.

    track_type_prob: probability of prepending the SA3-paper TrackType prefix
    ("TrackType: Music, VocalType: Instrumental, ") to the sampled caption.
    The base model trained with AudioSparx metadata prefixes present ~50% of
    the time and Stability recommends them at inference (SA3 paper §5.1) —
    0.5 mirrors base training; default 0.0 preserves existing behaviour.
    """
    with open(sidecar_path) as f:
        table = json.load(f)
    rng = random.Random(seed)
    p1, p2, p3 = probs

    def sampler(info, latents):
        stem = os.path.splitext(os.path.basename(info["latent_filename"]))[0]
        entry = table.get(stem)
        if entry is None:
            return {}
        r = rng.random() * (p1 + p2 + p3)
        if r < p1:
            tier = "t1"
        elif r < p1 + p2:
            tier = "t2"
        else:
            tier = "t3"
        prompt = entry.get(tier) or entry.get("t1")
        if not prompt:
            return {}
        if track_type_prob > 0 and rng.random() < track_type_prob:
            prompt = TRACK_TYPE_MUSIC + prompt
        return {"prompt": prompt}

    return sampler


def generate_sidecar(rows, out_path):
    """Write a T1-only sidecar (t2/t3 = None, filled by the Flamingo pass).

    rows: iterable of dicts with index_keys (list of latent stems), genres,
    moods, bpm, year. Returns number of index entries written.
    """
    table = {}
    for row in rows:
        t1 = build_t1(row["genres"], row["moods"], row.get("bpm"), row.get("year"))
        for key in row["index_keys"]:
            table[key] = {"t1": t1, "t2": None, "t3": None}
    with open(out_path, "w") as f:
        json.dump(table, f, indent=1)
    return len(table)

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
                         track_type_prob=0.0, caption_dropout_prob=0.0,
                         case_aug_prob=0.0, tag_shuffle_prob=0.0, key_fn=None):
    """custom_metadata_fn for PreEncodedDataset AND LocalDataset (live-encode).

    Samples a caption tier per call; tiers with missing text fall back to t1;
    unknown index -> {} (dataset keeps its stored prompt). Deterministic when
    `seed` is given (own RNG stream). Closure state survives dill.

    track_type_prob: prob of prepending the SA3-paper TrackType prefix
      ("TrackType: Music, VocalType: Instrumental, "). Base model saw AudioSparx
      prefixes ~50% (SA3 paper §5.1); 0.4-0.5 stays in-distribution. Default 0.0.
    caption_dropout_prob: prob of emitting the EMPTY prompt for CFG-uncond training.
      Leave 0.0 if the training wrapper already applies conditioning dropout (avoid
      double-dropping); set here (~0.1) for the --data_dir path if it does not.
    case_aug_prob: prob of lowercasing the whole caption (case-robustness).
    tag_shuffle_prob: prob of shuffling the comma-separated segments (order-robustness);
      applied to the BASE caption BEFORE the TrackType prefix, so the prefix stays front.
    key_fn: info->sidecar-key. Default resolves the basename-stem of the first present
      of latent_filename / path / relpath / filename (so PreEncoded=latent stem,
      Local=audio stem). Pass a custom one when sidecar keys are e.g. relpaths/hashes.
    """
    with open(sidecar_path) as f:
        table = json.load(f)
    rng = random.Random(seed)
    p1, p2, p3 = probs

    def _default_key(info):
        for f in ("latent_filename", "path", "relpath", "filename", "audio_filename"):
            v = info.get(f) if isinstance(info, dict) else None
            if v:
                return os.path.splitext(os.path.basename(v))[0]
        return None
    keyf = key_fn or _default_key

    def sampler(info, audio_or_latents):
        if caption_dropout_prob > 0 and rng.random() < caption_dropout_prob:
            return {"prompt": ""}
        key = keyf(info)
        entry = table.get(key) if key is not None else None
        if entry is None:
            return {}
        r = rng.random() * (p1 + p2 + p3)
        tier = "t1" if r < p1 else ("t2" if r < p1 + p2 else "t3")
        prompt = entry.get(tier) or entry.get("t1")
        if not prompt:
            return {}
        if tag_shuffle_prob > 0 and ", " in prompt and rng.random() < tag_shuffle_prob:
            parts = [s for s in prompt.split(", ") if s]
            rng.shuffle(parts)
            prompt = ", ".join(parts)
        if case_aug_prob > 0 and rng.random() < case_aug_prob:
            prompt = prompt.lower()
        if track_type_prob > 0 and rng.random() < track_type_prob:
            prompt = TRACK_TYPE_MUSIC + prompt
        return {"prompt": prompt}

    return sampler


# ---------------------------------------------------------------------------
# Pre-encoded multi-source caption routing (C, 2026-09-23). train_lora.py solved per-source tiers
# for --encoded_dir but its sidecar KEY is the latent stem, which matches nothing in the goa bigset
# (sidecar keyed by source path relative to the archive root; every stored prompt is the string
# "None"), so that corpus trained on the word "None" wherever it was used pre-encoded. These helpers
# are shared by train_lora_modular.py.
# ---------------------------------------------------------------------------

_EMPTY_PROMPTS = (None, "", "None", "none", "null")


def parse_caption_probs(spec, n_sources):
    """One (t1,t2,t3) tuple for every source, or semicolon-separated tuples in --encoded_dir order.
    Raises on a count mismatch rather than recycling (same contract as train_lora.py)."""
    parts = [p.strip() for p in str(spec).split(";") if p.strip()]
    out = []
    for p in parts:
        t = tuple(float(x) for x in p.split(","))
        if len(t) != 3 or any(v < 0 for v in t) or sum(t) <= 0:
            raise ValueError(f"--caption_probs tuple {p!r} must be 3 non-negative values, not all 0")
        out.append(t)
    if len(out) == 1:
        return out * n_sources
    if len(out) != n_sources:
        raise ValueError(f"--caption_probs has {len(out)} tuples for {n_sources} encoded_dirs")
    return out


def make_suffix_key_fn(table_keys):
    """Find an item's sidecar key whatever the sidecar was keyed by: latent stem, relpath, or ANY
    trailing sub-path of the source path (which covers keys relative to an archive root that only
    existed on another machine, e.g. /scratch/.../goa_archive/<key>)."""
    keys = frozenset(table_keys)

    def key_fn(info):
        cands = []
        for f in ("latent_filename", "path", "relpath", "filename", "audio_filename"):
            v = info.get(f) if isinstance(info, dict) else None
            if v:
                cands.append(os.path.splitext(os.path.basename(str(v)))[0])
        for f in ("relpath", "path", "source_path"):
            v = info.get(f) if isinstance(info, dict) else None
            if v:
                parts = str(v).strip("/").split("/")
                cands.extend("/".join(parts[i:]) for i in range(len(parts)))
        for c in cands:
            if c in keys:
                return c
        return None
    return key_fn


_SOURCE_TABLES = {}   # per-process cache: sidecar path -> (table, key_fn)


class SourceCaptionFn:
    """custom_metadata_fn for one pre-encoded source. With a sidecar: tier-sample it via the
    suffix-key lookup. Either way: REJECT an item whose final prompt is empty or the literal 'None',
    so a caption miss resamples instead of training on a placeholder.

    A top-level class holding only the PATH, not a closure over the table: PreEncodedDataset
    dill-loads the metadata fn on EVERY item (dataset.py:591), and a closure would carry the whole
    sidecar (54 MB for the goa bigset) through that on each sample. The table loads once per
    worker process into _SOURCE_TABLES."""

    def __init__(self, sidecar_path, probs=(0.0, 0.9, 0.1), track_type_prob=0.0):
        self.sidecar_path = sidecar_path
        self.probs = tuple(probs)
        self.track_type_prob = float(track_type_prob)

    def _lookup(self):
        hit = _SOURCE_TABLES.get(self.sidecar_path)
        if hit is None:
            with open(self.sidecar_path) as f:
                table = json.load(f)
            hit = (table, make_suffix_key_fn(table.keys()))
            _SOURCE_TABLES[self.sidecar_path] = hit
        return hit

    def __call__(self, info, latents):
        out = {}
        if self.sidecar_path:
            table, key_fn = self._lookup()
            entry = table.get(key_fn(info))
            if entry:
                p1, p2, p3 = self.probs
                r = random.random() * (p1 + p2 + p3)
                tier = "t1" if r < p1 else ("t2" if r < p1 + p2 else "t3")
                prompt = entry.get(tier) or entry.get("t1")
                if prompt:
                    if self.track_type_prob > 0 and random.random() < self.track_type_prob:
                        prompt = TRACK_TYPE_MUSIC + prompt
                    out = {"prompt": prompt}
        prompt = out.get("prompt", info.get("prompt") if isinstance(info, dict) else None)
        if prompt in _EMPTY_PROMPTS:
            return {"__reject__": True}
        return out


def make_source_caption_fn(sidecar_path, probs, track_type_prob=0.0):
    return SourceCaptionFn(sidecar_path, probs, track_type_prob)

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

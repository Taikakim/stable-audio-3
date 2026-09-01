"""
sa3_encode_from_manifest.py -- SAME-L encode the crops listed in a
sa3_beat_manifest.py CSV, appending to the latents_sa3 corpus.

Reconstructed 2026-07-04 (see sa3_beat_manifest.py docstring for context: the
originals were /tmp-only and got lost).

Per crop, writes (matching the existing /home/kim/Projects/latents_sa3 schema
exactly -- verified against 000000.{npy,json,TIMESERIES.npz}):
  <id>.npy            SAME-L latent, (256, 4096) fp16
  <id>.json           source .INFO features merged with crop-specific fields
                       (prompt, seconds_start/total, padding_mask, offsets...)
  <id>.TIMESERIES.npz whole-track npz sliced to [start,end] + resampled to 4096
                       frames, plus relative_position_ts

Known gap vs. the original: `style_genre` (a genre-classifier softmax dict) is
omitted -- these new source datasets never ran essentia's genre/mood/instrument
classifier (mir config gap, not this script's), and it isn't required by
train_lora.py's PreEncodedDataset (only padding_mask/seconds_total are). Not
worth re-running a 400-class Discogs classifier just to backfill one optional
field.

2026-07-06 (WINTERMUTE spec, avp corpus): two additions, both opt-in so the
existing corpora (prog_psytechno etc.) are untouched.
  --trigger-caption: prompt = "aavepyora"/"aavepyörä" only, chosen
    deterministically per PARENT track name (sha1 hash % 2 -- verbatim copy of
    mir/src/tools/inject_trigger_caption.py's word_for(), so this produces the
    identical word that script would write, without depending on it having run
    first or on cross-venv import). Every crop AND every augmentation variant
    of the same track shares one spelling (hashed on the parent name, never
    the variant folder name) -- matches that script's own stated intent.
  Co-located timeseries: if the manifest's parent_track_name/variant_name
    columns are present (sa3_beat_manifest.py --include-augmentations), the
    whole-track npz is looked up NEXT TO the crop's own track_folder
    (<track_folder>/<track_folder.name>.TIMESERIES.npz) instead of a flat
    --timeseries-root, since avp's mir output colocates it per-track/variant.
    Falls back to --timeseries-root if no colocated file exists (keeps the
    older flat-layout corpora working unchanged).
  Augmentation-variant crops get a MINIMAL info dict (crop-mechanical fields
    only: prompt/timestamps/padding_mask/provenance) -- the parent .INFO's
    audio-domain scalar features (bpm_madmom, onset_density, harmonic_*, etc.)
    describe the UNSHIFTED original and would be actively wrong if copied
    onto a pitch/tempo-shifted variant, so they're deliberately left out
    rather than guessed (that derivation is Phase D's open question, see
    sa3_beat_manifest.py's docstring -- not resolved here).

Non-44.1kHz sources (real bug, hit mid-run on the avp corpus: 233/1177 source
files are 48kHz personal-collection files): the crop window is always
computed in SECONDS from the manifest, but reading by a FIXED 44100-based
sample offset/count against a 48kHz file reads the wrong time window entirely
-- not just a dtype crash. Fixed by reading at the file's OWN native rate
(offset/frame-count computed from its real samplerate) and resampling to
44100 in fp32 ourselves before any fp16 cast, so stable_audio_3's internal
resampler (which crashes on fp16 input -- HalfTensor vs the resample kernel's
FloatTensor, a latent bug in preprocess_audio_list_for_encoder) never triggers.

Requires: SA3 venv (torch + stable_audio_3), run from the stable-audio-3 repo
root so `import stable_audio_3` resolves.
Usage:
  .venv/bin/python scripts/sa3_encode_from_manifest.py \
      --manifest /tmp/sa3_crop_manifest.csv \
      --timeseries-root /run/media/kim/Lehto/timeseries \
      --out /home/kim/Projects/latents_sa3 \
      --model_half

  # avp corpus (trigger caption, co-located timeseries, augmentation variants):
  .venv/bin/python scripts/sa3_encode_from_manifest.py \
      --manifest /tmp/avp_crop_manifest.csv \
      --out /home/kim/Projects/latents_avp \
      --trigger-caption --model_half
"""
import os
# ROCm/RDNA4: must be set before torch is imported (MASTER.md §5) --
# PYTORCH_TUNABLEOP_ENABLED=1 (the default) freezes on RDNA4/torch-2.12 when
# the ROCBLAS_VERSION validator fails (observed here: reading a tunableop
# cache built for a different ROCBLAS build than the one on this box).
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_audio_3 import AutoencoderModel

CROP_SAMPLES = 16777216
SAMPLE_RATE = 44100
LATENT_FRAMES = 4096


def _load_w_resampler():
    """W's per-field crop resampler lives in mir BY DESIGN -- how each field may legally be
    downsampled is a property of the measurement, not of this consumer (see its module docstring).
    Import it; never copy it (drift). Local tool -> the mir checkout is always present here."""
    import importlib.util
    cands = [Path("/home/kim/Projects/mir/src/tools/crop_timeseries_resample.py"),
             Path(__file__).resolve().parent.parent.parent / "lumi" / "vendor" / "crop_timeseries_resample.py"]
    for cand in cands:
        if cand.exists():
            spec = importlib.util.spec_from_file_location("crop_timeseries_resample", cand)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m.build_crop_timeseries
    raise ImportError(
        "crop_timeseries_resample not found (looked in mir/src/tools/ and lumi/vendor/). It owns the "
        "per-field pooling rules (native rate, masked-mean f0, mode chords) -- point at the mir checkout.")


_w_build_crop = _load_w_resampler()

TRIGGER_WORDS = ["aavepyora", "aavepyörä"]


def trigger_word_for(track_name: str) -> str:
    """Verbatim copy of mir/src/tools/inject_trigger_caption.py's word_for() --
    sha1(track_name) % 2, so every crop/variant of a track gets the identical
    deterministic spelling that script would assign, without depending on it
    having run first (avp's .INFO caption field is still None as of writing)."""
    import hashlib
    h = int(hashlib.sha1(track_name.encode("utf-8")).hexdigest(), 16)
    return TRIGGER_WORDS[h % 2]


def find_next_index(out_dir: Path) -> int:
    existing = [int(p.stem) for p in out_dir.glob("[0-9]" * 6 + ".npy") if p.stem.isdigit()]
    return (max(existing) + 1) if existing else 0


def parse_track_and_title(folder_name: str) -> tuple[str, str]:
    """'32. Sofia Kourtesis - La Perla' -> ('Sofia Kourtesis', 'La Perla').

    mir's in-place organize (output=null) does NOT strip the leading track
    number the way its output-dir path does -- strip it here instead of
    bulk-renaming hundreds of already-organized folders on Mantu.
    """
    import re
    name = re.sub(r"^\d{1,3}[.\s\-_]+", "", folder_name).strip()
    if " - " in name:
        artist, title = name.split(" - ", 1)
        return artist.strip(), title.strip()
    return "", name


def build_prompt(artist: str, title: str, bpm: float | None) -> str:
    """SA3 paper §3.5-style: random subset of available fields, comma-joined,
    50% lowercased. Field pool here is smaller than the original Goa corpus
    (no artist/year/genre from metadata lookup -- disabled for these runs)."""
    fields = []
    if title:
        fields.append(title)
    if artist:
        fields.append(artist)
    if bpm and bpm > 0:
        fields.append(str(round(bpm)))
    if not fields:
        return ""
    k = random.randint(1, len(fields))
    chosen = random.sample(fields, k)
    random.shuffle(chosen)
    prompt = ", ".join(chosen)
    if random.random() < 0.5:
        prompt = prompt.lower()
    return prompt


def load_info(track_folder: Path) -> dict:
    info_files = list(track_folder.glob("*.INFO"))
    if not info_files:
        return {}
    try:
        return json.loads(info_files[0].read_text())
    except Exception:
        return {}


def find_whole_track_npz(timeseries_root: Path | None, track_folder: Path, track_name: str) -> Path | None:
    """Co-located first (avp: <track_folder>/<track_folder.name>.TIMESERIES.npz,
    same convention for both primary tracks and augmentation-variant folders),
    else the flat --timeseries-root/<track_name>.TIMESERIES.npz layout the
    older corpora use."""
    colocated = track_folder / f"{track_folder.name}.TIMESERIES.npz"
    if colocated.exists():
        return colocated
    if timeseries_root is not None:
        try:
            flat = timeseries_root / f"{track_name}.TIMESERIES.npz"
            if flat.exists():
                return flat
        except OSError:
            pass  # unmounted/unreachable fallback root -- co-located already covers avp
    return None


def load_whole_track_npz(path: Path):
    if not path.exists():
        return None, None
    with np.load(str(path)) as z:
        meta = json.loads(str(z["__meta__"])) if "__meta__" in z.files else {}
        data = {k: z[k] for k in z.files if k != "__meta__"}
    return data, meta


def build_crop_timeseries(timeseries_root: Path | None, track_folder: Path, track_name: str,
                           start: float, end: float, fields: set | None = None) -> dict | None:
    npz_path = find_whole_track_npz(timeseries_root, track_folder, track_name)
    if npz_path is None:
        return None
    data, meta = load_whole_track_npz(npz_path)
    if data is None:
        return None
    if fields is not None:
        # Restrict to a whitelist BEFORE pooling -- avoids computing the fields we won't keep (e.g.
        # the 768-d maest embedding). W's resampler derives each sentinel field's voicing mask from
        # the values (>0) if the mask array isn't present, so a value-only whitelist still pools
        # correctly; we pass the masks too for exact validity weights.
        data = {k: v for k, v in data.items() if k in fields}
        if not data:
            return None
    # Per-field slice + pool via W's resampler (mir): each field at its OWN native rate (the store
    # now spans 0.2-100 Hz), masked-mean for f0 sentinel fields (0.0 = unvoiced, NOT silence),
    # mode-pool for categorical chords, and a LOUD ValueError if a field cannot cover the crop
    # window -- the old single-100Hz-rate mean loop silently mis-sliced/dropped all three (bugs
    # #1/#2/#3, measured 2026-08-12). strict=True + catch = a distinct, non-fatal per-crop skip.
    try:
        return _w_build_crop(data, meta, start, end, LATENT_FRAMES, strict=True)
    except ValueError as e:
        print(f"  SKIP (crop coverage): {track_name} [{start:.1f},{end:.1f}]s: {e}", file=sys.stderr)
        return None


def rebuild_companions(out_dir: Path, timeseries_root: Path | None,
                       fields: set | None = None, additive: bool = False):
    """--companion-only: rebuild the <idx>.TIMESERIES.npz beside existing <idx>.npy latents, from
    each crop's <idx>.json (source_track + timestamps). CPU only -- no model, no audio, no GPU.
    Latents are read-only; only .TIMESERIES.npz changes.

    fields   -- if set, compute ONLY these whitelisted store fields (e.g. the 4 f0 fields; excludes
                the 768-d maest bloat).
    additive -- if True, MERGE the (whitelisted) fields INTO the existing companion, preserving every
                field already there (incl relative_position_ts). This is the safe way to ADD f0 to a
                corpus's companions without dropping the legacy fields training already uses. If False,
                the companion is fully rebuilt from the store (+ relative_position_ts re-added)."""
    jsons = sorted(p for p in out_dir.glob("*.json") if p.stem.isdigit())
    if not jsons:
        print(f"companion-only: no <idx>.json sidecars in {out_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"companion-only: {len(jsons)} crops in {out_dir} | fields={sorted(fields) if fields else 'ALL'} "
          f"| mode={'ADDITIVE-merge' if additive else 'full-rebuild'}", flush=True)
    n_ok = n_skip = n_nolatent = 0
    for jp in jsons:
        idx_str = jp.stem
        try:
            info = json.loads(jp.read_text())
        except Exception as e:
            print(f"  SKIP (bad json): {idx_str}: {e}", file=sys.stderr); n_skip += 1; continue
        track_name = info.get("source_track")
        stamps = info.get("timestamps")
        if not track_name or not stamps or len(stamps) != 2:
            print(f"  SKIP (json missing source_track/timestamps): {idx_str}", file=sys.stderr)
            n_skip += 1; continue
        start, end = float(stamps[0]), float(stamps[1])
        # track_folder = the source mix's parent, for find_whole_track_npz's colocated branch. For
        # the flat goa store the field is resolved by track_name via --timeseries-root regardless,
        # so this only matters where a colocated .npz sits beside the track.
        src = info.get("source_path") or info.get("path") or ""
        track_folder = Path(src).parent if src else out_dir
        ts = build_crop_timeseries(timeseries_root, track_folder, track_name, start, end, fields=fields)
        if ts is None:
            print(f"  SKIP (no coverage/timeseries): {idx_str} {track_name}", file=sys.stderr)
            n_skip += 1; continue
        npz_path = out_dir / f"{idx_str}.TIMESERIES.npz"
        if not (out_dir / f"{idx_str}.npy").exists():
            n_nolatent += 1
        if additive:
            if not npz_path.exists():
                print(f"  SKIP (additive needs an existing companion): {idx_str}", file=sys.stderr)
                n_skip += 1; continue
            with np.load(npz_path) as z:
                merged = {k: z[k] for k in z.files}      # preserve ALL existing fields
            merged.update(ts)                            # add/overwrite only the whitelisted new ones
            np.savez(npz_path, **merged)
        else:
            rp0, rp1 = info.get("relative_position_start"), info.get("relative_position_end")
            if rp0 is not None and rp1 is not None:      # re-add exactly as the full path does
                ts["relative_position_ts"] = np.linspace(float(rp0), float(rp1), LATENT_FRAMES, dtype=np.float32)
            np.savez(npz_path, **ts)
        n_ok += 1
        if n_ok % 200 == 0:
            print(f"  {n_ok} companions written", flush=True)
    print(f"companion-only DONE: {n_ok} written, {n_skip} skipped, {n_nolatent} had no .npy "
          f"(latents untouched).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None,
                     help="required unless --companion-only")
    ap.add_argument("--companion-only", action="store_true",
                     help="rebuild ONLY the .TIMESERIES.npz companions beside existing --out latents "
                          "(CPU, no re-encode), from each crop's <idx>.json. Use after a whole-track "
                          "field is corrected/added (e.g. W's f0) so it reaches the crops without "
                          "re-encoding pristine latents. Ignores --manifest.")
    ap.add_argument("--fields", type=str, default=None,
                     help="companion-only: comma-separated whitelist of store fields to emit (e.g. "
                          "f0_other_ts,f0_other_voiced_ts,f0_bass_ts,f0_bass_voiced_ts). Excludes "
                          "everything else (e.g. the 768-d maest bloat). Default = all store fields.")
    ap.add_argument("--additive", action="store_true",
                     help="companion-only: MERGE the (whitelisted) fields INTO the existing companion, "
                          "preserving every field already there. The safe way to ADD f0 to a corpus "
                          "without dropping the legacy fields training uses. Default = full rebuild.")
    ap.add_argument("--timeseries-root", type=Path, default=Path("/run/media/kim/Lehto/timeseries"))
    ap.add_argument("--out", type=Path, required=True,
                     help="Per-source destination dir, e.g. /home/kim/Projects/latents_<source>. "
                          "NEVER latents_sa3 -- that corpus is the pristine Goa originals and must "
                          "not be mixed with other sources (Kim's directive, 2026-07-04).")
    ap.add_argument("--model_half", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", type=int, default=None,
                     help="skip the first N manifest rows -- resume after a crash without "
                          "re-encoding rows that already succeeded (find_next_index only "
                          "continues the OUTPUT numbering, it doesn't dedupe against the manifest)")
    ap.add_argument("--trigger-caption", action="store_true",
                     help="prompt = deterministic aavepyora/aavepyörä trigger word per parent "
                          "track (avp personal-style LoRA), instead of the §3.5 artist/title/bpm prompt")
    args = ap.parse_args()
    is_pristine = args.out.resolve() == Path("/home/kim/Projects/latents_sa3").resolve()

    if args.companion_only:
        # CPU-only companion rebuild -- no model, no manifest. Returns before any GPU work.
        # The pristine-corpus guard is about writing LATENTS. companion-only never writes a .npy;
        # in --additive mode it only MERGES fields into the existing companion (legacy fields +
        # latents both preserved, verified byte-identical), so it is safe on latents_sa3. A
        # non-additive full companion rebuild on the pristine corpus is refused -- use --additive.
        if is_pristine and not args.additive:
            print("REFUSING: full companion rebuild on latents_sa3 (the pristine Goa corpus). "
                  "Use --additive so existing companion fields are preserved (latent-safe).", file=sys.stderr)
            sys.exit(1)
        wl = {f.strip() for f in args.fields.split(",") if f.strip()} if args.fields else None
        rebuild_companions(args.out, args.timeseries_root, fields=wl, additive=args.additive)
        return

    # Full-encode path writes NEW latents -> the pristine-corpus guard applies here.
    if is_pristine:
        print("REFUSING: --out points at latents_sa3, the pristine Goa corpus. "
              "Use a per-source sibling dir instead.", file=sys.stderr)
        sys.exit(1)

    if args.manifest is None:
        print("REFUSING: --manifest is required (unless --companion-only).", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = AutoencoderModel.from_pretrained("same-l", device=str(device))
    if args.model_half:
        ae.autoencoder = ae.autoencoder.half()

    args.out.mkdir(parents=True, exist_ok=True)
    next_idx = find_next_index(args.out)

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    if args.skip:
        rows = rows[args.skip:]
    if args.limit:
        rows = rows[: args.limit]

    n_ok, n_skip = 0, 0
    info_cache: dict[str, dict] = {}
    sr_cache: dict[str, int] = {}
    crop_duration_s = CROP_SAMPLES / SAMPLE_RATE

    for row in rows:
        track_folder = Path(row["track_folder"])
        track_name = row["track_name"]
        parent_track_name = row.get("parent_track_name") or track_name
        variant_name = row.get("variant_name") or ""
        full_mix = Path(row["full_mix"])
        start = float(row["start_time"])
        end = float(row["end_time"])
        duration = float(row["duration"])
        crop_idx = int(row["crop_index"])

        ts = build_crop_timeseries(args.timeseries_root, track_folder, track_name, start, end)
        if ts is None:
            print(f"  SKIP (no whole-track timeseries): {track_name} crop {crop_idx}", file=sys.stderr)
            n_skip += 1
            continue

        fm_key = str(full_mix)
        if fm_key not in sr_cache:
            try:
                sr_cache[fm_key] = sf.info(fm_key).samplerate
            except Exception as e:
                print(f"  SKIP (unreadable): {track_name} crop {crop_idx}: {e}", file=sys.stderr)
                n_skip += 1
                continue
        native_sr = sr_cache[fm_key]

        # Read at the file's OWN rate (crop window is in seconds; a fixed
        # 44100-based sample offset/count would read the wrong time window
        # entirely on a non-44.1kHz source, not just need a resample).
        # start_sample stays in the 44100-normalized space -- it's an output
        # metadata field describing position in the encoded (always-44100)
        # representation, not the source file's own sample offset.
        start_sample = int(round(start * SAMPLE_RATE))
        start_sample_native = int(round(start * native_sr))
        frames_native = int(round(crop_duration_s * native_sr))
        try:
            audio, sr = sf.read(str(full_mix), start=start_sample_native, frames=frames_native,
                                 dtype="float32", always_2d=True)
        except Exception as e:
            print(f"  SKIP (read failed): {track_name} crop {crop_idx}: {e}", file=sys.stderr)
            n_skip += 1
            continue
        if audio.shape[0] < frames_native:
            print(f"  SKIP (short read, {audio.shape[0]}/{frames_native}): {track_name} crop {crop_idx}", file=sys.stderr)
            n_skip += 1
            continue

        if sr != SAMPLE_RATE:
            # Resample in fp32 BEFORE any fp16 cast -- stable_audio_3's own
            # internal resampler (T.Resample, fp32 kernel) crashes on a fp16
            # waveform, so sr must already equal SAMPLE_RATE by the time we
            # hand off to ae.encode().
            wav_t = torch.from_numpy(audio.T).contiguous()  # (C, T) fp32
            wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
            n = wav_t.shape[-1]
            if n < CROP_SAMPLES:
                wav_t = torch.nn.functional.pad(wav_t, (0, CROP_SAMPLES - n))
            elif n > CROP_SAMPLES:
                wav_t = wav_t[:, :CROP_SAMPLES]
            audio = wav_t.numpy().T
            sr = SAMPLE_RATE
        elif audio.shape[0] != CROP_SAMPLES:
            # native_sr == SAMPLE_RATE but rounding still left it off by a
            # sample or two -- pad/trim to the exact latent-frame-aligned count.
            pad = CROP_SAMPLES - audio.shape[0]
            audio = (np.pad(audio, ((0, pad), (0, 0))) if pad > 0 else audio[:CROP_SAMPLES])

        if audio.shape[0] < CROP_SAMPLES:
            print(f"  SKIP (short after resample, {audio.shape[0]}/{CROP_SAMPLES}): {track_name} crop {crop_idx}", file=sys.stderr)
            n_skip += 1
            continue

        audio_t = torch.from_numpy(audio.T).contiguous()  # (C, T)
        if args.model_half:
            audio_t = audio_t.half()

        with torch.no_grad():
            latent = ae.encode(audio_t.unsqueeze(0).to(device), sr)  # (1, 256, 4096)
        latent_np = latent.squeeze(0).to(torch.float16).cpu().numpy()

        is_variant = bool(variant_name)
        if is_variant:
            # Augmentation-variant crop: no .INFO of its own, and the PARENT's
            # audio-domain scalar features (bpm/onset_density/harmonic_*) describe
            # the unshifted original -- copying them here would be wrong, not
            # merely incomplete, so start from an empty dict rather than the
            # parent's info (see module docstring).
            info = {}
        else:
            if str(track_folder) not in info_cache:
                info_cache[str(track_folder)] = load_info(track_folder)
            info = dict(info_cache[str(track_folder)])

        if args.trigger_caption:
            prompt = trigger_word_for(parent_track_name)
        elif is_variant:
            prompt = trigger_word_for(parent_track_name)  # variants never have .INFO to build the §3.5 prompt from
        else:
            artist, title = parse_track_and_title(track_name)
            bpm = info.get("bpm_madmom") or info.get("bpm_essentia")
            prompt = build_prompt(artist, title, bpm)

        onset_density = info.get("onset_density")
        bpm_for_opb = info.get("bpm_madmom") or info.get("bpm_essentia")
        onset_per_beat = None
        if onset_density and bpm_for_opb:
            onset_per_beat = onset_density / (bpm_for_opb / 60.0)

        idx_str = f"{next_idx:06d}"
        info.update({
            "prompt": prompt,
            "seconds_start": start,
            "seconds_total": end - start,
            "padding_mask": [1] * LATENT_FRAMES,
            "source_track": track_name,
            "source_path": str(full_mix),
            "path": str(full_mix),
            "relpath": f"{track_name}/{idx_str}",
            "crop_idx": crop_idx,
            "start_sample": start_sample,
            "end_sample": start_sample + CROP_SAMPLES,
            "sample_rate": SAMPLE_RATE,
            "timestamps": [start, end],
            "load_time": 0.0,
            "relative_position_start": start / duration,
            "relative_position_end": end / duration,
            "source_total_samples": int(round(duration * SAMPLE_RATE)),
            "track_metadata_year": None,  # no metadata lookup ran for this batch (WINTERMUTE field map, 2026-07-04)
        })
        if is_variant:
            info.update({
                "is_augmentation": True,
                "variant_name": variant_name,
                "parent_track": parent_track_name,
            })
        if onset_per_beat is not None:
            info["onset_per_beat"] = onset_per_beat

        rel_pos = np.linspace(start / duration, end / duration, LATENT_FRAMES, dtype=np.float32)
        ts["relative_position_ts"] = rel_pos

        np.save(args.out / f"{idx_str}.npy", latent_np)
        (args.out / f"{idx_str}.json").write_text(json.dumps(info))
        np.savez(args.out / f"{idx_str}.TIMESERIES.npz", **ts)

        n_ok += 1
        next_idx += 1
        if n_ok % 20 == 0:
            print(f"  {n_ok} encoded ({track_name} crop {crop_idx})", flush=True)

    print(f"Done: {n_ok} encoded, {n_skip} skipped. Corpus now ends at {next_idx - 1:06d}.")


if __name__ == "__main__":
    main()

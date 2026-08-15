import numpy as np
import json
import os
import dill
import random
import time
import torch
import torchaudio

from os import path
from torchaudio import transforms as T
from typing import Optional, Callable, List

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T, VolumeNorm, strip_trailing_silence

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")

# --- ffmpeg audio-decode fallback (torchcodec-free path) -------------------------------
# torchaudio 2.x delegates decode to torchcodec, which is ABSENT on the LUMI multitorch image
# (torch 2.10+rocm7; a matching torchcodec wheel is fragile per our history — MASTER §5). ffmpeg
# IS present there, so decode via ffmpeg when torchcodec is missing. ffmpeg only DECODES at native
# rate; resampling stays with torchaudio in load_file(). NEVER use sox for anything (CLAUDE.md).
import subprocess as _subprocess
import shutil as _shutil
import importlib.util as _importlib_util
_HAVE_TORCHCODEC = _importlib_util.find_spec("torchcodec") is not None
_FFMPEG = _shutil.which("ffmpeg")
_FFPROBE = _shutil.which("ffprobe")


_MAX_DECODE_SECONDS = 1800.0  # 30 min


def _ffmpeg_load(filename, max_duration_s=_MAX_DECODE_SECONDS):
    """Decode any container to a float32 (C, N) tensor via ffmpeg at native rate.
    Returns (audio, sr). Used only when torchcodec is unavailable (multitorch image).

    max_duration_s guards against DJ-mix/compilation-length files (2026-08-15,
    preencode_bigset job 21148858 OOM investigation): subprocess.run(capture_output=True)
    buffers the ENTIRE decoded PCM stream in memory before returning, so an 80-min
    stereo f32 track is ~1.7GB just for the raw buffer -- with several parallel
    GCD-pinned shards each landing on one of these (goa_archive's "VA - ..." compilation
    folders plausibly contain some), that's a real node-RAM exhaustion mechanism. A
    >30min file is also not representative single-track training data regardless --
    it would just get truncated to whatever --sample_size crops to, wasting the decode.
    Skipping fast via ffprobe's duration (no extra subprocess) turns a potential OOM
    into a clean, logged reject."""
    if not _FFMPEG or not _FFPROBE:
        raise RuntimeError("ffmpeg/ffprobe not on PATH — cannot decode audio without torchcodec")
    probe = _subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels:format=duration", "-of", "default=nw=1", filename],
        capture_output=True, text=True)
    meta = dict(l.split("=", 1) for l in probe.stdout.split() if "=" in l)
    if "sample_rate" not in meta or "channels" not in meta:
        raise RuntimeError(f"ffprobe could not read {filename}: {probe.stderr[:200]}")
    try:
        duration = float(meta.get("duration", 0.0) or 0.0)
    except ValueError:
        duration = 0.0
    if duration > max_duration_s:
        raise RuntimeError(
            f"{filename}: duration {duration:.0f}s exceeds the {max_duration_s:.0f}s cap "
            "(likely a DJ-mix/compilation track, not representative training data)")
    sr = int(meta["sample_rate"]); ch = max(1, int(meta["channels"]))
    proc = _subprocess.run(
        [_FFMPEG, "-nostdin", "-v", "error", "-i", filename,
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(ch), "-ar", str(sr), "pipe:1"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg decode failed ({filename}): {proc.stderr[:200].decode('utf-8', 'replace')}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, ch).T.copy()
    return torch.from_numpy(audio), sr

# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions,
    #max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any(
                        [keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words])
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus'],
    filelist_path=None
):
    "recursively get a list of audio filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames

        if filelist_path is None:
            # Check for filelist.txt at the root of the directory
            filelist_path = os.path.join(path, "filelist.txt")
            
        if os.path.exists(filelist_path):
            with open(filelist_path, "r") as f:
                files = f.readlines()
                files = [os.path.join(path, file.strip()) for file in files]
                filenames.extend(files)
            continue

        if keywords is not None:
            subfolders, files = keyword_scandir(path, exts, keywords)
        else:
            subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames

def get_latent_filenames(
    paths,  # directories in which to search
    extension='npy',
    filelist_path=None
):
    "recursively get a list of pre-encoded filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames

        if filelist_path is None:
            # Check for filelist.txt at the root of the directory
            filelist_path = os.path.join(path, "filelist.txt")
        
        if os.path.exists(filelist_path):
            with open(filelist_path, "r") as f:
                files = f.readlines()
                files = [os.path.join(path, file.strip()) for file in files]
                filenames.extend(files)
            continue

        _, files = fast_scandir(path, [extension])
        filenames.extend(files)

    # Filter out silence.npy (used for silence latent padding, not a data sample)
    filenames = [f for f in filenames if os.path.basename(f) != "silence.npy"]

    # Add metadata paths
    filenames = [(filename, filename.replace(f".{extension}", ".json")) for filename in filenames]

    return filenames

class LocalDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        keywords: Optional[List[str]]=None,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        filelist_path = None,
        weight: float = 1.0,
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.keywords = keywords
        self.filelist_path = filelist_path
        self.weight = weight

class LatentDatasetConfig(LocalDatasetConfig):
    def __init__(
        self,
        latent_extension: str = "npy",
        filelist_path = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.latent_extension = latent_extension
        self.filelist_path = filelist_path
        # weight is inherited from LocalDatasetConfig via **kwargs

class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs,
        sample_size=65536,
        sample_rate=48000,
        random_crop=True,
        force_channels="stereo",
        volume_norm=False,
        volume_norm_param=(-16, 2),
        strip_silence=False,
        pad=True,
    ):
        super().__init__()
        self.filenames = []
        self.sample_weights = []

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
            #nn.Identity()
        )


        self.root_paths = []

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop, pad=pad)
        self.strip_silence = strip_silence

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity()
        )

        self.sr = sample_rate

        self.volume_norm = VolumeNorm(volume_norm_param, self.sr) if volume_norm else torch.nn.Identity()

        self.custom_metadata_fns = {}

        for config in configs:
            self.root_paths.append(config.path)
            new_files = get_audio_filenames(config.path, config.keywords, filelist_path=config.filelist_path)
            self.filenames.extend(new_files)
            self.sample_weights.extend([config.weight] * len(new_files))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = dill.dumps(config.custom_metadata_fn)

        print(f'Found {len(self.filenames)} files')

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        if _HAVE_TORCHCODEC:
            audio, in_sr = torchaudio.load(filename, format=ext)
        else:
            # torchaudio 2.x needs torchcodec (absent on the multitorch image) — decode via ffmpeg.
            audio, in_sr = _ffmpeg_load(filename)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx, _retries=0):
        if _retries > 100:
            raise RuntimeError(
                f"LocalDataset: >100 consecutive load/skip retries (idx {idx}) — dataset likely "
                f"broken (missing audio decoder, or all-silent/all-rejected). Aborting, not recursing.")
        audio_filename = self.filenames[idx]
        try:
            start_time = time.time()
            audio = self.load_file(audio_filename)

            audio = self.volume_norm(audio)

            if self.strip_silence:
                audio = strip_trailing_silence(audio, self.sr)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            # Check for silence
            if is_silence(audio):
                return self.__getitem__(random.randrange(len(self)), _retries + 1)

            # Run augmentations on this sample (including random crop)
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)

            info = {}

            info["path"] = audio_filename

            for root_path in self.root_paths:
                if root_path in audio_filename:
                    info["relpath"] = path.relpath(audio_filename, root_path)

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = [padding_mask]
            info["sample_rate"] = self.sr

            end_time = time.time()

            info["load_time"] = end_time - start_time

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in audio_filename:
                    custom_metadata_fn = dill.loads(self.custom_metadata_fns[custom_md_path])
                    custom_metadata = custom_metadata_fn(info, audio)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self.__getitem__(random.randrange(len(self)), _retries + 1)

                # Provide audio inputs as their own dictionary to be merged into info, each audio element will be normalized in the same way as the main audio
                if "__audio__" in info:
                    for audio_key, audio_value in info["__audio__"].items():
                        # Process the audio_value tensor, which should be a torch tensor
                        audio_value, _, _, _, _, _ = self.pad_crop(audio_value)
                        audio_value = audio_value.clamp(-1, 1)
                        if self.encoding is not None:
                            audio_value = self.encoding(audio_value)
                        info[audio_key] = audio_value
                
                    del info["__audio__"]

            return (audio, info)
        except Exception as e:
            print(f'Couldn\'t load file {audio_filename}: {e}')
            return self.__getitem__(random.randrange(len(self)), _retries + 1)


class PreEncodedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs: List[LatentDatasetConfig],
        latent_crop_length=None,
        min_length_sec=None,
        max_length_sec=None,
        random_crop=False,
        beat_aware_crop=False,
        tokenizers: Optional[dict] = None,
    ):
        super().__init__()
        self.filenames = []
        self.sample_weights = []

        self.custom_metadata_fns = {}

        self.silence_latents = {}

        for config in configs:
            new_files = get_latent_filenames(config.path, config.latent_extension, config.filelist_path)
            self.filenames.extend(new_files)
            self.sample_weights.extend([config.weight] * len(new_files))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = dill.dumps(config.custom_metadata_fn)

            # Load silence latent if available (for variable-length padding)
            paths = config.path if isinstance(config.path, list) else [config.path]
            for path in paths:
                silence_path = os.path.join(path, "silence.npy")
                if os.path.exists(silence_path):
                    self.silence_latents[path] = np.load(silence_path).squeeze(0)  # [C, N]
                    print(f'Loaded silence latent from {silence_path}')

        self.latent_crop_length = latent_crop_length
        self.random_crop = random_crop
        self.beat_aware_crop = beat_aware_crop

        self.min_length_sec = min_length_sec
        self.max_length_sec = max_length_sec

        # tokenizers: dict mapping metadata key -> (tokenizer, max_length)
        # If provided, text fields will be pre-tokenized in DataLoader workers
        self.tokenizers = tokenizers

        print(f'Found {len(self.filenames)} files')

    def __len__(self):
        return len(self.filenames)

    def _get_downbeat_starts(self, latent_filename, stored_length, max_start):
        """Return a list of valid crop-start frame indices that land on a downbeat
        (fallback: beat), scaled to latent-frame units. Empty list => caller falls
        back to uniform random_crop. max_start = last_ix - latent_crop_length (inclusive)."""
        # Resolve sibling <stem>.TIMESERIES.npz (latent_filename ends in .npy)
        base = latent_filename[:-4] if latent_filename.endswith(".npy") else os.path.splitext(latent_filename)[0]
        npz_path = base + ".TIMESERIES.npz"
        if not os.path.exists(npz_path) or max_start < 0:
            return []
        try:
            with np.load(npz_path) as npz:
                if "downbeat_activation_ts" in npz:
                    acts = npz["downbeat_activation_ts"]
                elif "beat_activation_ts" in npz:
                    acts = npz["beat_activation_ts"]
                else:
                    return []
                acts = np.asarray(acts, dtype=np.float32).reshape(-1)
        except Exception:
            return []

        if acts.size == 0 or not np.isfinite(acts).any():
            return []

        # Peak-pick downbeat activations. NOTE: the whole-track timeseries producer
        # resamples madmom activations to the ~10.767 Hz latent grid, which smears the
        # original sharp peaks down to ~0.1-0.2 — a fixed ">0.5" threshold never fires.
        # Use scipy peak-picking with an adaptive (per-track) height threshold instead,
        # plus a minimum inter-peak distance so close duplicates collapse.
        amax = float(acts.max())
        if amax <= 0.0:
            return []
        thr = max(acts.mean() + acts.std(), 0.4 * amax)
        try:
            from scipy.signal import find_peaks
            cand, _ = find_peaks(acts, height=thr, distance=4)
        except Exception:
            # numpy-only local-maxima fallback
            ge_prev = np.r_[True, acts[1:] >= acts[:-1]]
            ge_next = np.r_[acts[:-1] >= acts[1:], True]
            cand = np.flatnonzero(ge_prev & ge_next & (acts >= thr))
        if cand.size == 0:
            return []

        # Scale npz indices -> latent-frame units (handles length mismatch)
        if acts.size != stored_length:
            cand = np.floor(cand.astype(np.float64) * (stored_length / acts.size)).astype(np.int64)

        # Keep only candidates that yield a fully in-range crop
        cand = cand[(cand >= 0) & (cand <= max_start)]
        return cand.tolist()

    def _get_silence_for_file(self, latent_filename):
        """Return the silence latent for the dataset that contains this file, or None."""
        for path, silence in self.silence_latents.items():
            if path in latent_filename:
                return silence
        return None

    def __getitem__(self, idx):
        latent_filename, md_filename = self.filenames[idx]
        try:
            latents = torch.from_numpy(np.load(latent_filename)) # [C, N]

            with open(md_filename, "r") as f:
                try:
                    info = json.load(f)
                except:
                    raise Exception(f"Couldn't load metadata file {md_filename}")

            info["latent_filename"] = latent_filename

            if self.latent_crop_length is not None:
                stored_length = latents.shape[1]

                if stored_length > self.latent_crop_length:
                    # Crop to latent_crop_length (existing logic)
                    # Get the last index from the padding mask, the index of the last 1 in the sequence
                    last_ix = len(info["padding_mask"]) - 1 - info["padding_mask"][::-1].index(1)

                    if self.random_crop and last_ix > self.latent_crop_length:
                        max_start = last_ix - self.latent_crop_length
                        start = None
                        if self.beat_aware_crop:
                            db_starts = self._get_downbeat_starts(
                                latent_filename, stored_length, max_start
                            )
                            if db_starts:
                                start = random.choice(db_starts)
                        if start is None:
                            # Fallback: uniform random offset
                            start = random.randint(0, max_start)
                    else:
                        start = 0

                    latents = latents[:, start:start+self.latent_crop_length]
                    info["padding_mask"] = info["padding_mask"][start:start+self.latent_crop_length]
                    info["latent_crop_start"] = start

                elif stored_length < self.latent_crop_length:
                    # Pad with silence latent to reach latent_crop_length
                    pad_needed = self.latent_crop_length - stored_length
                    silence = self._get_silence_for_file(latent_filename)

                    if silence is not None:
                        # Slice or tile silence latent to cover pad_needed frames
                        if silence.shape[1] >= pad_needed:
                            silence_pad = silence[:, :pad_needed]
                        else:
                            silence_pad = np.tile(silence, (1, (pad_needed // silence.shape[1]) + 1))[:, :pad_needed]
                        latents = torch.cat([latents, torch.from_numpy(silence_pad)], dim=1)
                    else:
                        # No silence latent available — zero-pad as fallback
                        latents = torch.nn.functional.pad(latents, (0, pad_needed))

                    # Build padding_mask: valid frames from stored mask, zeros for padding
                    info["padding_mask"] = info["padding_mask"][:stored_length] + [0] * pad_needed
                    info["latent_crop_start"] = 0

                else:
                    # Exact match
                    info["latent_crop_start"] = 0

                info["latent_crop_length"] = self.latent_crop_length

            info["padding_mask"] = [torch.tensor(info["padding_mask"])]

            seconds_total = info["seconds_total"]

            if self.min_length_sec is not None and seconds_total < self.min_length_sec:
                return self[random.randrange(len(self))]

            if self.max_length_sec is not None and seconds_total > self.max_length_sec:
                return self[random.randrange(len(self))]

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in latent_filename:
                    custom_metadata_fn = dill.loads(self.custom_metadata_fns[custom_md_path])
                    custom_metadata = custom_metadata_fn(info, latents)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

                if "__replace__" in info and info["__replace__"] is not None:
                    # Replace the latents with the new latents if the custom metadata function returns a new set of latents
                    latents = info["__replace__"]

            info["audio"] = latents

            # Pre-tokenize text fields in DataLoader workers to avoid
            # CPU contention with the main training thread
            if self.tokenizers is not None:
                for key, (tokenizer, max_length) in self.tokenizers.items():
                    if key in info and isinstance(info[key], str):
                        # Save raw text before replacing with tokens (needed by CLAP and other text-based losses)
                        info[f"{key}_text"] = info[key]
                        encoded = tokenizer(
                            info[key],
                            truncation=True,
                            max_length=max_length,
                            padding="max_length",
                            return_tensors="pt",
                        )
                        info[key] = {
                            "input_ids": encoded["input_ids"].squeeze(0),
                            "attention_mask": encoded["attention_mask"].squeeze(0),
                        }

            return (latents, info)
        except Exception as e:
            print(f'Couldn\'t load file {latent_filename}: {e}')
            return self[random.randrange(len(self))]

class ArcRolloutDataset(torch.utils.data.Dataset):
    """ARC-Forcing rollout dataset (shared data contract, SAO task #46).

    Reads a directory of .npz files, one per training sample, with keys:
      context_latent (fp16, [C, Tctx]) -- the model's own drifted rollout context;
                                          REPLACES the clamped (mask==1) region
      target_latent  (fp16, [C, T])    -- the TRUE full window (context + continuation);
                                          loss frames are mask==0
      mask           (uint8, [T])      -- 1 = frame is CLAMPED context visible to
                                          the model, 0 = to-generate. mask[:Tctx] = 1,
                                          so the context lives INSIDE the target window
      prompt         (str)
      meta           (json str)        -- source stem, crop offsets, rollout nl, ckpt id
    T is a multiple of 256 (MASTER §5); Tctx = mask.sum(). A manifest.json at the
    dir root lists files + generation params (manifest-v2).

    Returns (latents [C, T], info) where latents = target_latent (the true window,
    used only for the mask==0 loss region) and info carries the inpaint conditioning:
      inpaint_mask         [1, T] -- the contract mask
      inpaint_masked_input [C, T] -- the DRIFTED context_latent written into the
                                     mask==1 frames, zeros elsewhere. Never
                                     latents * mask: that would leak the clean
                                     teacher context and defeat the ARC
                                     exposure-bias objective.
    The training wrapper routes these through the same conditioning keys inference
    uses ('inpaint_mask'/'inpaint_masked_input' -> local_add_cond) and restricts
    the diffusion loss to the free (mask=0) region.
    """

    def __init__(
        self,
        path,
        sample_rate=44100,
        downsampling_ratio=4096,
        tokenizers: Optional[dict] = None,
    ):
        super().__init__()
        paths = path if isinstance(path, list) else [path]

        self.filenames = []
        self.manifest = None
        for p in paths:
            _, files = fast_scandir(p, ["npz"])
            # Exclude timeseries siblings that may live next to latents in mixed dirs
            self.filenames.extend(f for f in files if not f.endswith(".TIMESERIES.npz"))

            manifest_path = os.path.join(p, "manifest.json")
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r") as f:
                        self.manifest = json.load(f)
                except Exception as e:
                    print(f"Couldn't load manifest file {manifest_path}: {e}")
        self.filenames.sort()

        self.sample_rate = sample_rate
        self.downsampling_ratio = downsampling_ratio
        self.tokenizers = tokenizers

        print(f'Found {len(self.filenames)} ARC rollout samples')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        npz_filename = self.filenames[idx]
        try:
            with np.load(npz_filename) as npz:
                context = npz["context_latent"].astype(np.float32)  # [C, Tctx]
                target = npz["target_latent"].astype(np.float32)    # [C, T]
                mask = npz["mask"].astype(np.float32)                # [T]
                prompt = str(npz["prompt"])
                meta = json.loads(str(npz["meta"])) if "meta" in npz.files else {}

            if mask.shape[0] != target.shape[1]:
                raise Exception(
                    f"contract violation: mask length {mask.shape[0]} != target T {target.shape[1]}"
                )
            n_clamped = int(mask.sum())
            if context.shape[1] != n_clamped:
                raise Exception(
                    f"contract violation: context Tctx {context.shape[1]} != "
                    f"mask.sum() {n_clamped}"
                )

            # The context is EMBEDDED in the target window (mask[:Tctx]=1), so the
            # true window IS the full training sequence — no concatenation.
            latents = torch.from_numpy(target)  # [C, T]
            t_total = latents.shape[1]

            inpaint_mask = torch.from_numpy(mask).unsqueeze(0)  # [1, T]
            # Model-visible input: the DRIFTED rollout context in the clamped
            # frames, zeros in the to-generate frames. NOT latents * mask — that
            # would clamp the clean teacher context and defeat the ARC objective.
            inpaint_masked_input = torch.zeros_like(latents)
            clamped = torch.from_numpy(mask).bool()
            inpaint_masked_input[:, clamped] = torch.from_numpy(context)

            info = {
                "path": npz_filename,
                "latent_filename": npz_filename,
                "prompt": prompt,
                "seconds_total": t_total * self.downsampling_ratio / self.sample_rate,
                "padding_mask": [torch.ones(t_total, dtype=torch.bool)],
                "inpaint_mask": [inpaint_mask],
                "inpaint_masked_input": [inpaint_masked_input],
                "arc_meta": meta,
            }

            # Pre-tokenize text fields in DataLoader workers (same as PreEncodedDataset)
            if self.tokenizers is not None:
                for key, (tokenizer, max_length) in self.tokenizers.items():
                    if key in info and isinstance(info[key], str):
                        info[f"{key}_text"] = info[key]
                        encoded = tokenizer(
                            info[key],
                            truncation=True,
                            max_length=max_length,
                            padding="max_length",
                            return_tensors="pt",
                        )
                        info[key] = {
                            "input_ids": encoded["input_ids"].squeeze(0),
                            "attention_mask": encoded["attention_mask"].squeeze(0),
                        }

            return (latents, info)
        except Exception as e:
            print(f'Couldn\'t load file {npz_filename}: {e}')
            return self[random.randrange(len(self))]


# get_dbmax and is_silence copied from https://github.com/drscotthawley/aeiou/blob/main/aeiou/core.py under Apache 2.0 License
# License can be found in LICENSES/LICENSE_AEIOU.txt
def get_dbmax(
    audio,       # torch tensor of (multichannel) audio
    ):
    "finds the loudest value in the entire clip and puts that into dB (full scale)"
    return 20*torch.log10(torch.flatten(audio.abs()).max()).cpu().numpy()

def is_silence(
    audio,       # torch tensor of (multichannel) audio
    thresh=-60,  # threshold in dB below which we declare to be silence
    ):
    "checks if entire clip is 'silence' below some dB threshold"
    dBmax = get_dbmax(audio)
    return dBmax < thresh


def collation_fn(samples):
        batched = list(zip(*samples))
        result = []
        for b in batched:
            if isinstance(b[0], (int, float)):
                b = np.array(b)
            elif isinstance(b[0], torch.Tensor):
                b = torch.stack(b)
            elif isinstance(b[0], np.ndarray):
                b = np.array(b)
            else:
                b = b
            result.append(b)
        return result



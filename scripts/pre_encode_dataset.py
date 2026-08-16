"""
Pre-encode a dataset of audio clips into latents using Stable Audio 3, saving the latents and metadata to disk.

Dataset layout:
  data_dir/
    clip1.wav   (or .flac, .mp3, .ogg)
    clip1.txt   ← text prompt for clip1
    clip2.wav
    clip2.txt
    ...

Saves .npy files for latents and .json files for metadata, compatible with train_lora.py --encoded_dir.

Usage:
  uv run python scripts/pre_encode_dataset.py --model same-s --data_dir ./my_data --output_path ./latents_out
  uv run python scripts/pre_encode_dataset.py --model same-l --data_dir ./my_data --output_path ./latents_out --batch_size 4
"""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
# 8 GCD-pinned shards on one LUMI node, each spawning its own DataLoader workers, decoding
# long goa_archive tracks through the default shared-memory IPC -> exhausts /dev/shm / node
# RAM -> kernel oom_kill (job 21073662, 2026-08-15: died in <8min, 0 real latents written).
# Same bug + same fix as train_lora.py's live-encode DDP path (job 20687866, 2026-08-06):
# 'file_system' shares tensors via /tmp files (bound in the container) instead of /dev/shm.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass
from torch.nn import functional as F

from stable_audio_3 import AutoencoderModel
from stable_audio_3.model_configs import ae_models
from stable_audio_3.data.dataset import (
    LocalDatasetConfig,
    SampleDataset,
    collation_fn,
)


def _log_rss(tag):
    """Main-process RSS (model + tensors received from DataLoader workers via the
    file_system sharing strategy). Sibling of dataset.py's per-worker _log_rss --
    added together 2026-08-15 to catch the still-unresolved preencode OOM in the act."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    print(f"[rss pid={os.getpid()} main] {tag} {line.strip()}", flush=True)
                    return
    except Exception:
        pass


def _log_leak_scan(tag, min_bytes=1_000_000):
    """Scan the live object graph for large numpy/torch buffers still reachable -- the
    decisive test between 'a Python reference leak' (this climbs in lockstep with RSS;
    gc.get_referrers() on a hit would show WHO holds it) and 'a lower-level allocator
    effect below Python's visibility' (this stays flat while RSS still grows). Added
    2026-08-15 after tracemalloc was ruled out (it only sees allocations routed through
    CPython's own allocator; numpy/torch buffers this size go through raw malloc,
    invisible to it) and static code reading found no obvious accumulating list/cache."""
    gc.collect()
    total = 0
    count = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, np.ndarray) and obj.nbytes >= min_bytes:
                total += obj.nbytes
                count += 1
            elif isinstance(obj, torch.Tensor) and obj.numel() * obj.element_size() >= min_bytes:
                total += obj.numel() * obj.element_size()
                count += 1
        except Exception:
            pass
    print(f"[leak-scan] {tag} live_large_objs={count} total_mb={total/1e6:.1f}", flush=True)


def caption_metadata_fn(info, _audio):
    txt = Path(info["path"]).with_suffix(".txt")
    if not txt.exists():
        return {"__reject__": True}
    return {"prompt": txt.read_text().strip()}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = AutoencoderModel.from_pretrained(args.model, device=str(device))
    if args.model_half:
        ae.autoencoder = ae.autoencoder.half()

    dataset = SampleDataset(
        [
            LocalDatasetConfig(
                id="train", path=args.data_dir,
                custom_metadata_fn=None if args.no_caption_check else caption_metadata_fn,
            )
        ],
        sample_size=args.sample_size,
        sample_rate=ae.sample_rate,
        force_channels="stereo",
    )
    shard_i = 0
    if getattr(args, "shard", None):
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        dataset.filenames = dataset.filenames[shard_i::shard_n]   # deterministic 1/N slice
        print(f"[shard {shard_i}/{shard_n}] encoding {len(dataset.filenames)} of the corpus", flush=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else min(4, os.cpu_count() or 1),
        drop_last=False,
        collate_fn=collation_fn,
    )

    os.makedirs(args.output_path, exist_ok=True)

    silence_path = os.path.join(args.output_path, "silence.npy")
    if not os.path.exists(silence_path):
        print("Saving silence latent")
        silence_audio = torch.zeros(
            1, ae.autoencoder.io_channels, args.sample_size, device=device
        )
        if args.model_half:
            silence_audio = silence_audio.half()
        with torch.no_grad():
            silence_latent = ae.encode(silence_audio, ae.sample_rate)
        np.save(silence_path, silence_latent.cpu().numpy())

    for nb, (audio, metadata) in enumerate(loader):
        print(f"Processing batch {nb}")
        if nb % 20 == 0:
            _log_rss(f"batch {nb}")
            _log_leak_scan(f"batch {nb}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        audio = audio.to(device)
        if args.model_half:
            audio = audio.half()

        latents = ae.encode(audio, ae.sample_rate)

        for i, latent in enumerate(latents):
            latent_np = latent.cpu().numpy()
            latent_id = f"{shard_i:02d}{nb:06d}{i:04d}"   # shard-prefixed -> collision-free merge

            md = dict(metadata[i])
            padding_mask = (
                F.interpolate(
                    md["padding_mask"][0].unsqueeze(0).unsqueeze(1).float(),
                    size=latent_np.shape[-1],
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .int()
            )
            if not args.pad:
                padding_np = padding_mask.cpu().numpy()
                valid_indices = np.where(padding_np == 1)[0]
                if len(valid_indices) > 0:
                    valid_length = valid_indices[-1] + 1
                    latent_np = latent_np[:, :valid_length]
                    padding_mask = padding_mask[:valid_length]

            np.save(os.path.join(args.output_path, f"{latent_id}.npy"), latent_np)

            md["padding_mask"] = padding_mask.cpu().numpy().tolist()
            for k, v in md.items():
                if isinstance(v, torch.Tensor):
                    md[k] = v.cpu().numpy().tolist()

            with open(os.path.join(args.output_path, f"{latent_id}.json"), "w") as f:
                json.dump(md, f)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-encode audio dataset to latents")
    parser.add_argument("--model", choices=list(ae_models), default="same-l")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Folder with audio files and matching .txt captions",
    )
    parser.add_argument(
        "--output_path", required=True, help="Folder to write .npy/.json latent pairs"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=12582912,  # 380s at 44.1kHz, 2 channels
        help="Audio samples to pad/crop to (default ~380s at 44.1kHz)",
    )
    parser.add_argument(
        "--model_half", action="store_true", help="Run autoencoder in fp16"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="DataLoader workers. Default None = min(4, cpu_count()), fine for a solo run "
             "but OOMs the node when several GCD-pinned shards run in parallel on one node "
             "(8 shards x 4 workers = 32 processes decoding long tracks at once -> kernel "
             "oom_kill, job 21073662). Pass a small explicit value (e.g. 1-2) for parallel "
             "multi-shard preencode jobs.",
    )
    parser.add_argument(
        "--pad", action="store_true", help="Pad audio samples to --sample_size"
    )
    parser.add_argument(
        "--shard", default=None,
        help="'I/N' -- encode only the deterministic 1/N slice filenames[I::N] of the corpus, "
             "with I-prefixed latent ids so 8 GCD-pinned shards write one collision-free dir.",
    )
    parser.add_argument(
        "--no_caption_check", action="store_true",
        help="Skip caption_metadata_fn's per-file .txt sidecar requirement (default: every file "
             "missing a matching .txt is __reject__-ed). Use when captions are merged in a LATER "
             "local step, not expected at encode time (e.g. goa_archive, whose captions come from "
             "a sidecar JSON, not one .txt per file -- 2026-08-17: this requirement silently "
             "rejected 100% of goa_archive, root-causing both the '>100 consecutive retries' abort "
             "AND the preencode 'memory leak' misdiagnosis, which was actually RSS high-water-mark "
             "growth from repeatedly decoding full tracks that then got discarded on rejection).",
    )
    args = parser.parse_args()

    if not args.pad and args.batch_size > 1:
        parser.error(
            "padding is required for batch_size > 1; pass --pad or use --batch_size 1"
        )

    main(args)

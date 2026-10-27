#!/usr/bin/env python3
"""Stage and package SA3 Drum Separation boards into self-contained web-standard sites.

Packages evaluation runs (from /home/kim/sao_runs/...) into /home/kim/evals_aac/...:
1. Transcodes uncompressed .wav files to lightweight web-streaming AAC .m4a (192 kbps).
2. Rewrites .wav references in HTML and JSON files to .m4a.
3. Sanitizes all URLs to relative paths (no localhost, no file://, no absolute paths).
4. Adds seamless master navigation links between consoles and dashboards.
5. Generates landing HTML entry points in /home/kim/evals_aac/.
"""

import os
import sys
import shutil
import re
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def transcode_one_file(args):
    src_wav, dst_m4a = args
    if dst_m4a.exists() and dst_m4a.stat().st_size > 1000:
        return dst_m4a, True, "skipped"
    
    dst_m4a.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(src_wav),
        "-c:a", "aac", "-b:a", "192k",
        str(dst_m4a)
    ]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        return dst_m4a, True, "transcoded"
    else:
        return dst_m4a, False, "error"

def package_board(src_dir: Path, dest_dir: Path, board_title: str):
    print(f"\n================================================================================")
    print(f"Packaging Board: {board_title}")
    print(f"Source:      {src_dir}")
    print(f"Destination: {dest_dir}")
    print(f"================================================================================")

    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find all files in src_dir
    wav_files = []
    other_files = []
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            src_file = Path(root) / f
            rel = src_file.relative_to(src_dir)
            if f.endswith(".wav"):
                dst_m4a = dest_dir / rel.with_suffix(".m4a")
                wav_files.append((src_file, dst_m4a))
            else:
                other_files.append((src_file, dest_dir / rel))

    print(f"Found {len(wav_files)} WAV audio files and {len(other_files)} other files (HTML/JSON/etc).")

    # 2. Transcode audio in parallel
    print("Transcoding audio to web-optimized AAC (.m4a @ 192k)...")
    transcoded_count = 0
    skipped_count = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(transcode_one_file, pair) for pair in wav_files]
        for fut in as_completed(futures):
            dst, ok, status = fut.result()
            if status == "transcoded":
                transcoded_count += 1
            elif status == "skipped":
                skipped_count += 1
    print(f"Audio processing complete: {transcoded_count} transcoded, {skipped_count} up-to-date.")

    # 3. Copy and sanitize non-wav files (HTML, JSON, CSS)
    print("Copying and sanitizing HTML, JSON and assets...")
    for src_file, dst_file in other_files:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if src_file.suffix in [".html", ".json", ".js", ".css"]:
            content = src_file.read_text(encoding="utf-8")
            # Replace .wav references with .m4a
            content = re.sub(r'([a-zA-Z0-9_\-\./]+)\.wav', r'\1.m4a', content)
            
            # Sanitize local server ports and paths
            content = content.replace("http://localhost:8088/drum_sep_scratch_ranks_eval/", "./")
            content = content.replace("http://localhost:8088/drum_sep_lr_brackets_eval/", "./")
            content = re.sub(r'file:///home/kim/[a-zA-Z0-9_\-\./]+index\.html', 'index.html', content)

            # Ensure back-link in sub-consoles if in a subdirectory
            if src_file.name == "index.html" and src_file.parent != src_dir:
                if "Back to Master" not in content:
                    back_nav = '''
    <div style="margin-bottom: 16px;">
      <a href="../index.html" style="display: inline-flex; align-items: center; gap: 8px; color: #38bdf8; text-decoration: none; font-size: 13px; font-weight: 600; padding: 6px 12px; background: #0f172a; border: 1px solid #1e293b; border-radius: 6px;">
        &larr; Back to Master Comparison Dashboard
      </a>
    </div>
'''
                    content = content.replace("<header>", "<header>" + back_nav)

            dst_file.write_text(content, encoding="utf-8")
        else:
            shutil.copy2(src_file, dst_file)

    # 4. Create top-level landing redirect in evals_aac root
    landing_file = dest_dir.parent / f"{dest_dir.name}.html"
    rel_index = f"{dest_dir.name}/index.html"
    landing_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{board_title}</title>
  <meta http-equiv="refresh" content="0; url={rel_index}">
  <script>window.location.href = "{rel_index}";</script>
</head>
<body style="background:#070b14;color:#f8fafc;font-family:sans-serif;padding:40px;text-align:center;">
  <h2>Redirecting to {board_title}...</h2>
  <p><a href="{rel_index}" style="color:#38bdf8;">Click here if not redirected automatically.</a></p>
</body>
</html>
"""
    landing_file.write_text(landing_html, encoding="utf-8")
    print(f"Top-level landing redirect written: {landing_file}")

    print(f"Board successfully packaged at: file://{dest_dir}/index.html")
    print(f"Public target URL on site:     https://aavepyora.online/files/evals/{dest_dir.name}/index.html\n")

def main():
    evals_aac = Path("/home/kim/evals_aac")
    evals_aac.mkdir(parents=True, exist_ok=True)

    # Board 1: LR Brackets
    lr_src = Path("/home/kim/sao_runs/drum_sep_lr_brackets_eval")
    if lr_src.exists():
        package_board(lr_src, evals_aac / "drum_sep_lr_brackets", "SA3 Drum Separation — LR Brackets Evaluation")

    # Board 2: Scratch Ranks & Optimizer Shootout
    scratch_src = Path("/home/kim/sao_runs/drum_sep_scratch_ranks_eval")
    if scratch_src.exists():
        package_board(scratch_src, evals_aac / "drum_sep_scratch_ranks", "SA3 Drum Separation — From-Scratch Ranks & Optimizer Shootout")

if __name__ == "__main__":
    main()

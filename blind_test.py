#!/usr/bin/env python3
# CALLING AHJIN — Brought to you by Ahmed Hussein and Julius Norén from JTH.
"""
blind_test.py
-------------
Copies every audio from Cross_Examination into a new Test/ folder
with randomised numerical names (1.wav, 2.wav, ...) so you can
listen without knowing which model produced which clip.

A key file is saved so you can reveal the answers afterwards:
    Test/key.txt   — maps each number back to the original filename

Usage:
    python blind_test.py
    python blind_test.py --folder Cross_Examination --out Test
"""

from __future__ import annotations
import argparse, random, shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

AUDIO_EXTS = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


def main() -> None:
    """
    Copy audio files from Cross_Examination into a numbered Test/ folder.

    Workflow:
      1. Collect all audio files from the source folder.
      2. Shuffle them with an optional fixed seed for reproducibility.
      3. Copy each file to the output folder with a numeric name (1.wav, 2.wav, …).
      4. Write Test/key.txt mapping each number back to the original filename.

    Listeners hear the numbered clips without seeing the source filenames,
    then reveal the key.txt answers to see which model produced each clip.
    """
    parser = argparse.ArgumentParser(description="Blind test: randomise audio filenames")
    parser.add_argument("--folder", default=None,
        help="Source folder (default: config.cross_examination_folder)")
    parser.add_argument("--out", default="Test",
        help="Output folder name (default: Test)")
    parser.add_argument("--seed", type=int, default=None,
        help="Random seed for reproducibility (default: random each run)")
    args = parser.parse_args()

    # Resolve source folder — use config default if not specified
    if args.folder:
        src_folder = Path(args.folder).resolve()
    else:
        src_folder = Path(config.cross_examination_folder)
        if not src_folder.is_absolute():
            src_folder = (Path(config.__file__).resolve().parent / src_folder).resolve()

    if not src_folder.is_dir():
        sys.exit(f"Source folder not found: {src_folder}")

    # Collect supported audio files (sorted for determinism before shuffling)
    files = sorted(
        p for p in src_folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )

    if not files:
        sys.exit(f"No audio files found in {src_folder}")

    # Shuffle — fixed seed gives the same ordering for AB-test re-runs
    if args.seed is not None:
        random.seed(args.seed)
    shuffled = files[:]
    random.shuffle(shuffled)

    # Create output folder (adjacent to config.py, i.e. project root)
    out_folder = Path(config.__file__).resolve().parent / args.out
    out_folder.mkdir(parents=True, exist_ok=True)

    # Copy each file with a numeric name, preserving the original extension
    mapping = []
    for i, src in enumerate(shuffled, start=1):
        dst_name = f"{i}{src.suffix.lower()}"
        dst = out_folder / dst_name
        shutil.copy2(str(src), str(dst))
        mapping.append((dst_name, src.name))

    # Write the answer key so the blind test can be revealed afterwards
    key_path = out_folder / "key.txt"
    with open(key_path, "w", encoding="utf-8") as f:
        f.write("BLIND TEST KEY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Source : {src_folder}\n\n")
        f.write(f"{'Number':<12}  Original filename\n")
        f.write(f"{'-'*12}  {'-'*30}\n")
        for dst_name, orig_name in mapping:
            f.write(f"{dst_name:<12}  {orig_name}\n")

    # Print summary
    print()
    print(f"  {len(files)} clips shuffled into: {out_folder}")
    print()
    print(f"  {'Number':<12}  Original filename")
    print(f"  {'-'*12}  {'-'*30}")
    for dst_name, orig_name in mapping:
        print(f"  {dst_name:<12}  {orig_name}")
    print()
    print(f"  Key saved to: {key_path}")
    print()


if __name__ == "__main__":
    main()

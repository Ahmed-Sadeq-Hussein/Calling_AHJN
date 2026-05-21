#!/usr/bin/env python3
"""
convert_manifest.py
-------------------
Converts the LibriTTS/MLS manifest produced by download_mls.py into the
CSV format expected by F5-TTS training/fine-tuning.

Input  : data/processed/train_filelist.txt  (wav_path|speaker_id|text)
Output : data/processed/metadata.csv        (audio_path|text)  ← full train set
         data/processed/metadata_val.csv    (audio_path|text)  ← held-out 5%

Audio is NOT re-encoded by default (F5-TTS resamples internally).
Use --resample to convert everything to 24 kHz WAV upfront — only worth
doing if you have ample disk space and want faster training I/O.

Usage:
    python convert_manifest.py
    python convert_manifest.py --input data/processed/my_voice_filelist.txt
    python convert_manifest.py --val-split 0.05 --min-duration 1.5 --max-duration 30
    python convert_manifest.py --resample   # writes 24 kHz copies alongside originals
"""

from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys, argparse, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def read_manifest(path: Path) -> list[tuple[str, str, str]]:
    """
    Reads a manifest file with lines of the form:
        wav_path|speaker_id|text
    Returns list of (wav_path, speaker_id, text) tuples.
    Lines that don't split into at least 2 fields are skipped.
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 3:
                wav_path, speaker_id, text = parts
            elif len(parts) == 2:
                # Fallback: wav_path|text  (prepare_voice.py format without speaker col)
                wav_path, text = parts
                speaker_id = "unknown"
            else:
                print(f"  [skip] line {lineno}: unexpected format")
                continue
            rows.append((wav_path.strip(), speaker_id.strip(), text.strip()))
    return rows


def filter_rows(rows: list[tuple[str, str, str]],
                min_dur: float, max_dur: float,
                max_rows: int | None,
                skip_duration_check: bool = False) -> list[tuple[str, str, str]]:
    """
    Keeps rows where the WAV file exists on disk.
    Duration check is skipped by default because download_mls.py already
    filtered clips to the configured duration range — re-reading every file
    header is unnecessary and takes ~30 minutes for 100k+ files.
    Pass skip_duration_check=False to re-enable it.
    """
    from tqdm import tqdm

    kept: list[tuple[str, str, str]] = []
    skipped_missing  = 0
    skipped_duration = 0

    if skip_duration_check:
        # Fast path — just check existence (os.path.exists is very fast)
        pbar = tqdm(rows, desc="  Checking files", unit="clip")
        for wav_path, speaker_id, text in pbar:
            if not Path(wav_path).exists():
                skipped_missing += 1
                continue
            kept.append((wav_path, speaker_id, text))
            if max_rows and len(kept) >= max_rows:
                break
        pbar.close()
    else:
        import soundfile as sf
        pbar = tqdm(rows, desc="  Checking files+duration", unit="clip")
        for wav_path, speaker_id, text in pbar:
            if not Path(wav_path).exists():
                skipped_missing += 1
                continue
            try:
                info = sf.info(wav_path)
                dur  = info.frames / info.samplerate
            except Exception:
                skipped_missing += 1
                continue
            if not (min_dur <= dur <= max_dur):
                skipped_duration += 1
                continue
            kept.append((wav_path, speaker_id, text))
            if max_rows and len(kept) >= max_rows:
                break
        pbar.close()

    if skipped_missing:
        print(f"  Skipped {skipped_missing} rows (file missing or unreadable)")
    if skipped_duration:
        print(f"  Skipped {skipped_duration} rows (duration outside {min_dur}-{max_dur}s)")

    return kept


def resample_rows(rows: list[tuple[str, str, str]],
                  target_sr: int) -> list[tuple[str, str, str]]:
    """
    Resamples each WAV to target_sr, writes a new file with _24k suffix,
    and returns an updated rows list pointing to the resampled files.
    Skips files already at the target rate.
    """
    import numpy as np
    import soundfile as sf
    import librosa
    from tqdm import tqdm

    new_rows: list[tuple[str, str, str]] = []
    pbar = tqdm(rows, desc=f"resampling to {target_sr} Hz", unit="clip")

    for wav_path, speaker_id, text in pbar:
        p = Path(wav_path)
        info = sf.info(str(p))

        if info.samplerate == target_sr:
            # Already correct — no rewrite needed
            new_rows.append((wav_path, speaker_id, text))
            continue

        new_p = p.with_stem(p.stem + "_24k") if hasattr(p, "with_stem") \
            else p.with_name(p.stem + "_24k" + p.suffix)

        if not new_p.exists():
            wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sf.write(str(new_p), wav, target_sr, subtype="PCM_16")

        new_rows.append((str(new_p), speaker_id, text))

    return new_rows


def write_csv(rows: list[tuple[str, str, str]], path: Path) -> None:
    """Writes F5-TTS metadata CSV: audio_path|text"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for wav_path, _speaker, text in rows:
            # Escape any | inside the text field (rare but possible)
            safe_text = text.replace("|", " ")
            fh.write(f"{wav_path}|{safe_text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert download_mls.py manifest to F5-TTS metadata CSV")
    parser.add_argument(
        "--input", default=str(config.PROCESSED_DATA_DIR / "train_filelist.txt"),
        help="Input manifest (wav|speaker|text). Default: data/processed/train_filelist.txt")
    parser.add_argument(
        "--out-dir", default=str(config.PROCESSED_DATA_DIR),
        help="Output directory for metadata.csv / metadata_val.csv")
    parser.add_argument(
        "--val-split", type=float, default=0.05,
        help="Fraction of clips to put in the validation CSV (default 0.05)")
    parser.add_argument(
        "--min-duration", type=float, default=1.0,
        help="Minimum clip duration in seconds (default 1.0)")
    parser.add_argument(
        "--max-duration", type=float, default=30.0,
        help="Maximum clip duration in seconds (default 30.0)")
    parser.add_argument(
        "--max-clips", type=int, default=None,
        help="Cap number of training clips (default: all)")
    parser.add_argument(
        "--check-duration", action="store_true",
        help="Re-verify each file's duration (slow — ~30 min for 100k files). "
             "Not needed if the manifest came from download_mls.py which already filters duration.")
    parser.add_argument(
        "--resample", action="store_true",
        help="Resample audio to 24 kHz (F5-TTS native rate). Slow for large sets.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split (default 42)")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_dir  = Path(args.out_dir)
    train_csv = out_dir / "metadata.csv"
    val_csv   = out_dir / "metadata_val.csv"

    if not in_path.exists():
        sys.exit(f"Input manifest not found: {in_path}")

    print(f"Input  : {in_path}")
    print(f"Output : {train_csv}")
    print(f"Val    : {val_csv}")
    print()

    # 1. Read
    print("Reading manifest …")
    rows = read_manifest(in_path)
    print(f"  {len(rows)} total rows")

    # 2. Filter
    skip_dur = not args.check_duration
    if skip_dur:
        print("Filtering (existence check only — use --check-duration to re-verify durations) …")
    else:
        print("Filtering (checking files + durations) …")
    rows = filter_rows(rows, args.min_duration, args.max_duration,
                       args.max_clips, skip_duration_check=skip_dur)
    print(f"  {len(rows)} rows after filtering")

    if not rows:
        sys.exit("No valid clips found. Is the download still running? "
                 "You can re-run this script once more data has been downloaded.")

    # 3. Optionally resample
    if args.resample:
        print(f"\nResampling to {config.F5TTS_SAMPLE_RATE} Hz …")
        rows = resample_rows(rows, config.F5TTS_SAMPLE_RATE)

    # 4. Shuffle + split
    random.seed(args.seed)
    random.shuffle(rows)
    n_val   = max(1, int(len(rows) * args.val_split))
    val_rows   = rows[:n_val]
    train_rows = rows[n_val:]
    print(f"\nSplit  : {len(train_rows)} train  /  {n_val} val")

    # 5. Write
    write_csv(train_rows, train_csv)
    write_csv(val_rows,   val_csv)
    print(f"\nWritten:")
    print(f"  {train_csv}  ({len(train_rows)} lines)")
    print(f"  {val_csv}    ({n_val} lines)")
    print()
    print("Next:")
    print("  python clone_voice.py --reference ref.wav --text 'Hello world'")
    print("  python finetune_voice.py --speaker ahmed  (to fine-tune on your voice)")


if __name__ == "__main__":
    main()

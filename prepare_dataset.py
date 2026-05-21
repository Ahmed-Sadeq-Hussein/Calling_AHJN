#!/usr/bin/env python3
"""
prepare_dataset.py
------------------
Converts metadata.csv into the Arrow format expected by F5-TTS's finetune CLI.

F5-TTS (pip install) resolves dataset paths to:
    <anaconda>/Lib/data/{dataset_name}_{tokenizer}/raw.arrow
    <anaconda>/Lib/data/{dataset_name}_{tokenizer}/duration.json

This script builds both files from our existing metadata.csv.
It mirrors the official f5_tts/train/datasets/prepare_csv_wavs.py but:
  - Reads audio_path|text format (no header required)
  - Skips pinyin conversion (we use --tokenizer custom)
  - Uses multithreading for fast header reads

Expected runtime: ~2-4 minutes for 119 k clips (soundfile header reads only).

Usage:
    python prepare_dataset.py              # uses config defaults
    python prepare_dataset.py --dry-run    # preview target path, don't write
    python prepare_dataset.py --workers 8  # explicit thread count
"""

from __future__ import annotations
import os, sys, json, argparse, concurrent.futures, multiprocessing
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from importlib.resources import files

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ── Defaults ───────────────────────────────────────────────────────────────────

DATASET_NAME = "librispeech"
TOKENIZER    = "custom"          # matches --tokenizer custom in train_model.py
MIN_DUR      = 0.3               # seconds
MAX_DUR      = 30.0
MAX_WORKERS  = max(1, multiprocessing.cpu_count() - 1)


# ── Path helpers ───────────────────────────────────────────────────────────────

def get_target_dir(dataset_name: str, tokenizer: str) -> Path:
    """
    Compute where F5-TTS expects the pre-built Arrow dataset.
    Mirrors the logic inside f5_tts/model/dataset.py:
        files("f5_tts").joinpath("../../data/{name}_{tok}")
    """
    raw = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
    return Path(raw).resolve()


# ── Audio helpers ───────────────────────────────────────────────────────────────

def get_duration(audio_path: str) -> float | None:
    """Return duration in seconds from file header only (no full decode)."""
    try:
        import soundfile as sf
        return sf.info(audio_path).duration
    except Exception:
        pass
    try:
        import torchaudio
        info = torchaudio.info(audio_path)
        if info.sample_rate > 0:
            return info.num_frames / info.sample_rate
    except Exception:
        pass
    return None


def process_row(row: tuple[str, str]) -> dict | None:
    audio_path, text = row
    if not Path(audio_path).exists():
        return None
    dur = get_duration(audio_path)
    if dur is None or not (MIN_DUR <= dur <= MAX_DUR):
        return None
    return {"audio_path": audio_path, "text": text, "duration": float(dur)}


# ── Reading metadata.csv ───────────────────────────────────────────────────────

def read_metadata(csv_path: Path) -> list[tuple[str, str]]:
    """Read audio_path|text CSV (no header)."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                print(f"  [skip] line {lineno}: unexpected format")
                continue
            audio_path, text = parts
            rows.append((audio_path.strip(), text.strip()))
    return rows


# ── Arrow writing ──────────────────────────────────────────────────────────────

def write_arrow(records: list[dict], target_dir: Path) -> None:
    from datasets.arrow_writer import ArrowWriter
    from tqdm import tqdm

    target_dir.mkdir(parents=True, exist_ok=True)

    # raw.arrow  — same format as prepare_csv_wavs.py
    arrow_path = target_dir / "raw.arrow"
    print(f"  Writing {arrow_path} ...")
    with ArrowWriter(path=str(arrow_path)) as writer:
        for rec in tqdm(records, desc="  arrow", unit="clip"):
            writer.write(rec)
        writer.finalize()
    print(f"  {arrow_path.stat().st_size / 1e6:.1f} MB")

    # duration.json
    durations = [r["duration"] for r in records]
    dur_path = target_dir / "duration.json"
    with open(dur_path, "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f, ensure_ascii=False)
    print(f"  Written: {dur_path}")

    total_h = sum(durations) / 3600
    print(f"\n  {len(records):,} clips  /  {total_h:.2f} h  →  {target_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare F5-TTS Arrow dataset from metadata.csv")
    parser.add_argument(
        "--input", default=str(config.PROCESSED_DATA_DIR / "metadata.csv"),
        help="CSV with audio_path|text rows. Default: data/processed/metadata.csv")
    parser.add_argument(
        "--dataset-name", default=DATASET_NAME,
        help=f"Dataset name used in F5-TTS (default: {DATASET_NAME})")
    parser.add_argument(
        "--tokenizer", default=TOKENIZER,
        help=f"Tokenizer suffix appended to dir name (default: {TOKENIZER})")
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Thread pool size for header reads (default: {MAX_WORKERS})")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show target path and clip count, but don't write anything")
    args = parser.parse_args()

    csv_path   = Path(args.input)
    target_dir = get_target_dir(args.dataset_name, args.tokenizer)

    print()
    print(f"Input  CSV : {csv_path}")
    print(f"Target dir : {target_dir}")
    print()

    if not csv_path.exists():
        sys.exit(f"Metadata CSV not found: {csv_path}\n"
                 "Run: python convert_manifest.py")

    # Check (or attempt to create) target dir early
    if not args.dry_run:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            sys.exit(
                f"Cannot write to {target_dir}\n"
                "You may need to run this in an elevated prompt or change permissions."
            )

    # Read CSV
    print("Reading metadata.csv …")
    rows = read_metadata(csv_path)
    print(f"  {len(rows):,} rows")

    # Read durations in parallel
    print(f"Reading durations ({args.workers} threads) …")
    from tqdm import tqdm
    records: list[dict] = []
    skipped = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_row, r): r for r in rows}
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(futures), unit="clip"):
            result = fut.result()
            if result is None:
                skipped += 1
            else:
                records.append(result)

    print(f"  Kept   : {len(records):,}")
    print(f"  Skipped: {skipped:,}")

    if not records:
        sys.exit("No valid clips found.")

    if args.dry_run:
        print()
        print("[dry-run] Would write to:", target_dir)
        print("Run without --dry-run to proceed.")
        return

    # Write Arrow + duration.json
    print()
    print("Writing Arrow dataset …")
    write_arrow(records, target_dir)

    print()
    print("=" * 60)
    print("Dataset ready!")
    print()
    print("Now run:")
    print("  python train_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

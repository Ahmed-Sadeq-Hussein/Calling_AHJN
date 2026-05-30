#!/usr/bin/env python3
# CALLING AHJIN — Brought to you by Ahmed Hussein and Julius Norén from JTH.
"""
evaluate_ahmed_test.py
----------------------
Full evaluation pipeline for the Ahmed_test dataset.

For every clip in Ahmed_test:
  1. Creates a results folder:  test_results/<clip_stem>/
  2. Copies the original recording as human_baseline.wav (ground truth)
  3. Uses the NEXT clip in the dataset as the voice reference for generation
  4. Generates the target text with all 5 models:
       Baseline_F5TTS, Ahmed, Ahmed_2, Ahmed_on_Ahmed_2, Ahmed_2_on_Ahmed
  5. Runs Resemblyzer on the folder  (human_baseline as primary reference)
  6. Runs WER on the folder against the target text

Output structure:
    test_results/
        Ahmed_test_00001/
            human_baseline.wav
            Baseline_F5TTS.wav
            Ahmed.wav
            Ahmed_2.wav
            Ahmed_on_Ahmed_2.wav
            Ahmed_2_on_Ahmed.wav
            speaker_similarity_heatmap.png
            speaker_similarity_bar.png
            wer_results.png
            wer_results.csv
        Ahmed_test_00002/
            ...

Usage:
    python evaluate_ahmed_test.py
    python evaluate_ahmed_test.py --clips 3        # only first 3 clips
    python evaluate_ahmed_test.py --skip-existing  # skip folders already done
"""

from __future__ import annotations
import os, sys, shutil, subprocess, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

PROJECT = Path(config.__file__).resolve().parent

# ── Model definitions ──────────────────────────────────────────────────────────
# None checkpoint = pretrained F5-TTS baseline (no fine-tuning)
MODELS: dict[str, str | None] = {
    "Baseline_F5TTS":   None,
    "Ahmed":            str(PROJECT / "models/f5tts_finetune/ahmed/model_last.pt"),
    "Ahmed_2":          str(PROJECT / "models/f5tts_finetune/Ahmed_2/model_last.pt"),
    "Ahmed_on_Ahmed_2": str(PROJECT / "models/f5tts_finetune/ahmed_on_Ahmed_2/model_last.pt"),
    "Ahmed_2_on_Ahmed": str(PROJECT / "models/f5tts_finetune/Ahmed_2_on_ahmed/model_last.pt"),
}

RESULTS_DIR   = PROJECT / "test_results"
CLONE_SCRIPT  = PROJECT / "clone_voice.py"
RESEM_SCRIPT  = PROJECT / "resembler metric.py"
WER_SCRIPT    = PROJECT / "wer_graph.py"
FILELIST      = PROJECT / "data/processed/Ahmed_test_filelist.txt"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_filelist(path: Path) -> list[tuple[Path, str]]:
    """Parse filelist: wav_path|speaker|text → [(Path, text), ...]"""
    clips = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) >= 3:
            clips.append((Path(parts[0]), parts[2].strip()))
    if not clips:
        sys.exit(f"No entries found in {path}")
    return clips


def run(cmd: list[str], label: str) -> bool:
    """Run a subprocess, print result, return success."""
    print(f"    Running {label} …")
    result = subprocess.run(cmd, env={**os.environ})
    if result.returncode != 0:
        print(f"    WARNING: {label} exited with code {result.returncode}")
        return False
    return True


# ── Per-clip pipeline ──────────────────────────────────────────────────────────

def process_clip(clip_path: Path, target_text: str,
                 ref_path: Path, ref_text: str,
                 skip_existing: bool) -> None:
    """Run the full pipeline for one Ahmed_test clip."""

    folder = RESULTS_DIR / clip_path.stem
    folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Clip   : {clip_path.name}")
    print(f"  Text   : {target_text[:70]}")
    print(f"  Ref    : {ref_path.name}  |  \"{ref_text[:50]}\"")
    print(f"  Folder : {folder}")
    print(f"{'='*64}")

    # ── 1. Human baseline ──────────────────────────────────────────────────────
    baseline = folder / "human_baseline.wav"
    if not baseline.exists():
        shutil.copy2(str(clip_path), str(baseline))
        print(f"  Copied human baseline → {baseline.name}")
    else:
        print(f"  Human baseline already present.")

    # ── 2. Generate from each model ────────────────────────────────────────────
    for model_name, checkpoint in MODELS.items():
        out_path = folder / f"{model_name}.wav"

        if skip_existing and out_path.exists():
            print(f"  Skipping {model_name} (already exists)")
            continue

        print(f"\n  [{model_name}]")
        if checkpoint:
            print(f"    Checkpoint : {checkpoint}")
        else:
            print(f"    Checkpoint : (pretrained F5-TTS baseline — no fine-tuning)")
        cmd = [
            sys.executable, str(CLONE_SCRIPT),
            "--reference", str(ref_path),
            "--ref-text",  ref_text,
            "--text",      target_text,
            "--out",       str(out_path),
            "--seed",      "42",
            "--nfe-steps", "32",
        ]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]

        run(cmd, model_name)

    # ── 3. Resemblyzer ─────────────────────────────────────────────────────────
    print(f"\n  Running Resemblyzer …")
    run([
        sys.executable, str(RESEM_SCRIPT),
        "--folder",    str(folder),
        "--reference", "human_baseline.wav",
    ], "Resemblyzer")

    # ── 4. WER ────────────────────────────────────────────────────────────────
    print(f"\n  Running WER …")
    run([
        sys.executable, str(WER_SCRIPT),
        "--text",   target_text,
        "--folder", str(folder),
        "--model",  "base",
    ], "WER")

    print(f"\n  Done: {folder}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full evaluation pipeline for Ahmed_test dataset")
    parser.add_argument(
        "--clips", type=int, default=None,
        help="Only process the first N clips (default: all)")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip model generation if the output .wav already exists")
    args = parser.parse_args()

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    if not FILELIST.exists():
        sys.exit(
            f"Filelist not found: {FILELIST}\n"
            f"Record the Ahmed_test dataset first:\n"
            f"  python record_dataset.py --speaker Ahmed_test"
        )

    for model_name, ckpt in MODELS.items():
        if ckpt and not Path(ckpt).exists():
            print(f"  WARNING: checkpoint not found for {model_name}: {ckpt}")

    clips = load_filelist(FILELIST)
    if args.clips:
        clips = clips[:args.clips]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nAhmed_test evaluation pipeline")
    print(f"  Clips   : {len(clips)}")
    print(f"  Models  : {list(MODELS.keys())}")
    print(f"  Output  : {RESULTS_DIR}")

    # ── Process each clip ──────────────────────────────────────────────────────
    for i, (clip_path, target_text) in enumerate(clips):
        # Reference = next clip (wraps around at end)
        ref_idx   = (i + 1) % len(clips)
        ref_path, ref_text = clips[ref_idx]

        process_clip(
            clip_path    = clip_path,
            target_text  = target_text,
            ref_path     = ref_path,
            ref_text     = ref_text,
            skip_existing = args.skip_existing,
        )

    print(f"\n{'='*64}")
    print(f"  ALL CLIPS COMPLETE")
    print(f"  Results in: {RESULTS_DIR}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()

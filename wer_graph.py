#!/usr/bin/env python3
"""
wer_graph.py
------------
Measures Word Error Rate (WER) for every audio file in the
Cross_Examination folder and saves a bar graph there.

For each clip:
  1. Transcribe with faster-whisper (no FFmpeg needed for WAV)
  2. Compare to the target text you provide
  3. WER = (substitutions + deletions + insertions) / reference_word_count

Output:
    Cross_Examination/wer_results.png   — bar chart, one bar per clip
    Cross_Examination/wer_results.csv   — raw scores for further analysis

Usage:
    python wer_graph.py --text "She sells seashells by the seashore"
    python wer_graph.py --text "She sells seashells..." --model medium
    python wer_graph.py --text "..." --folder path/to/other/folder
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import sys
import argparse
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

AUDIO_EXTS = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


# ── WER calculation ────────────────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> tuple[float, int, int, int, int]:
    """
    Compute WER using dynamic programming (edit distance on word sequences).

    Returns (wer, substitutions, deletions, insertions, ref_len).
    WER = (S + D + I) / ref_len
    """
    ref   = reference.lower().split()
    hyp   = hypothesis.lower().split()
    n     = len(ref)
    m     = len(hyp)

    # dp[i][j] = min edit ops to align ref[:i] with hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],       # deletion
                    dp[i][j - 1],       # insertion
                    dp[i - 1][j - 1],   # substitution
                )

    # Back-track to count S / D / I
    i, j = n, m
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1; i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ins += 1; j -= 1
        else:
            dels += 1; i -= 1

    wer = (subs + dels + ins) / n if n > 0 else 0.0
    return wer, subs, dels, ins, n


# ── Transcription ──────────────────────────────────────────────────────────────

def transcribe(audio_path: Path, whisper_model_size: str) -> str:
    """Transcribe an audio file with faster-whisper. No FFmpeg needed for WAV."""
    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    from faster_whisper import WhisperModel

    print(f"  Transcribing {audio_path.name} …")

    # Try soundfile first (fast, no dependencies)
    wav, sr = None, None
    try:
        wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
    except Exception:
        pass

    # Fall back to librosa (handles mp3, mp4, etc.)
    if wav is None:
        try:
            wav, sr = librosa.load(str(audio_path), sr=None, mono=True)
        except Exception as e:
            raise RuntimeError(f"Could not load {audio_path.name}: {e}")

    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"

    model = WhisperModel(whisper_model_size, device=device, compute_type=compute)
    segments, _ = model.transcribe(wav, beam_size=5, language="en", vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    del model  # free VRAM for next clip

    print(f"    -> \"{text}\"")
    return text


# ── Plotting ───────────────────────────────────────────────────────────────────

def save_bar_chart(labels: list[str], wers: list[float],
                   out_path: Path, target_text: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(labels)
    x = np.arange(n)

    # Colour: green if WER <= 15%, amber if <= 30%, red above
    colours = []
    for w in wers:
        if w <= 0.15:
            colours.append("#2ecc71")   # green
        elif w <= 0.30:
            colours.append("#f39c12")   # amber
        else:
            colours.append("#e74c3c")   # red

    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 5))
    bars = ax.bar(x, [w * 100 for w in wers], color=colours,
                  edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels on each bar
    for bar, w in zip(bars, wers):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{w:.1%}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Word Error Rate (%)", fontsize=12)
    ax.set_title("Word Error Rate per Clip\n"
                 f"Target: \"{target_text[:80]}{'…' if len(target_text) > 80 else ''}\"",
                 fontsize=12, pad=12)
    ax.set_ylim(0, max(max(w * 100 for w in wers) * 1.25, 20))
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="WER <= 15%  (good)"),
        Patch(facecolor="#f39c12", label="WER 15–30%  (fair)"),
        Patch(facecolor="#e74c3c", label="WER > 30%   (poor)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nBar chart saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WER bar chart for every audio in the Cross_Examination folder")
    parser.add_argument(
        "--text", required=True,
        help="The target text all clips are supposed to say")
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large-v2"],
        help="Whisper model size for transcription (default: base)")
    parser.add_argument(
        "--folder", default=None,
        help="Override the folder path (default: config.cross_examination_folder)")
    args = parser.parse_args()

    # Resolve folder
    if args.folder:
        folder = Path(args.folder).resolve()
    else:
        folder = Path(config.cross_examination_folder)
        if not folder.is_absolute():
            folder = (Path(config.__file__).resolve().parent / folder).resolve()

    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")

    # Collect audio files (sorted, no sub-folders)
    audio_files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )

    if len(audio_files) < 1:
        sys.exit(f"No audio files found in {folder}")

    print()
    print("=" * 60)
    print("  WORD ERROR RATE EVALUATION")
    print("=" * 60)
    print(f"  Folder   : {folder}")
    print(f"  Clips    : {len(audio_files)}")
    print(f"  Target   : \"{args.text[:70]}\"")
    print(f"  Whisper  : {args.model}")
    print()

    # ── Transcribe + score ─────────────────────────────────────────────────────
    results = []   # (label, hypothesis, wer, subs, dels, ins, ref_len)

    for path in audio_files:
        try:
            hypothesis = transcribe(path, args.model)
            wer, subs, dels, ins, ref_len = compute_wer(args.text, hypothesis)
            results.append((path.name, hypothesis, wer, subs, dels, ins, ref_len))
            print(f"    WER: {wer:.1%}  "
                  f"(S={subs} D={dels} I={ins} / {ref_len} words)")
            print()
        except Exception as e:
            print(f"  ERROR on {path.name}: {e}")
            print()

    if not results:
        sys.exit("No clips could be processed.")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Clip':<35}  {'WER':>6}  {'S':>4}  {'D':>4}  {'I':>4}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*4}")
    for label, _, wer, s, d, i, _ in results:
        short = label[:35]
        print(f"  {short:<35}  {wer:>5.1%}  {s:>4}  {d:>4}  {i:>4}")
    print()

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = folder / "wer_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "transcription", "wer",
                         "substitutions", "deletions", "insertions", "ref_words"])
        for label, hyp, wer, s, d, i, ref_len in results:
            writer.writerow([label, hyp, f"{wer:.4f}", s, d, i, ref_len])
    print(f"  CSV saved : {csv_path}")

    # ── Save bar chart ─────────────────────────────────────────────────────────
    labels = [r[0] for r in results]
    wers   = [r[2] for r in results]
    png_path = folder / "wer_results.png"
    save_bar_chart(labels, wers, png_path, args.text)

    print()
    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

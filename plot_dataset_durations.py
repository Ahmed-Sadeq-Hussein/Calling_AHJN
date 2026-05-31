#!/usr/bin/env python3
#CALLING AHJIN Brought to you by Ahmed Hussein and Julius Norén from JTH.

"""
plot_dataset_durations.py
-------------------------
Plots overlapping duration distributions for the ahmed and Ahmed_2 datasets,
showing the difference in average clip length and variation between the two.
Used to show he diffrences in the two datasets.
Output:
    Cross_Examination/dataset_durations.png

Usage:
    python plot_dataset_durations.py
"""

import sys
from pathlib import Path

import soundfile as sf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

DATASETS = {
    "Ahmed\n(short sentences)":   config.PROCESSED_DATA_DIR / "ahmed",
    "Ahmed_2\n(long sentences)":  config.PROCESSED_DATA_DIR / "Ahmed_2",
}
COLORS = ["#3498db", "#e74c3c"]   # blue, red
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def get_durations(folder: Path) -> list[float]:
    """Return a list of clip durations (seconds) for every audio file in folder."""
    durations = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in AUDIO_EXTS:
            try:
                durations.append(sf.info(str(p)).duration)
            except Exception:
                pass
    return durations


def main() -> None:
    # ── Collect durations ──────────────────────────────────────────────────────
    all_durations = {}
    for label, folder in DATASETS.items():
        if not folder.is_dir():
            sys.exit(f"Dataset folder not found: {folder}")
        durs = get_durations(folder)
        if not durs:
            sys.exit(f"No audio files found in {folder}")
        all_durations[label] = durs
        total = sum(durs)
        print(f"{label.replace(chr(10), ' '):<30}  "
              f"n={len(durs):>3}  "
              f"total={total:.1f}s ({total/60:.1f} min)  "
              f"mean={np.mean(durs):.2f}s  "
              f"min={min(durs):.2f}s  "
              f"max={max(durs):.2f}s")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    all_vals = [d for durs in all_durations.values() for d in durs]
    bin_min, bin_max = min(all_vals), max(all_vals)
    bins = np.linspace(bin_min, bin_max, 25)

    for (label, durs), color in zip(all_durations.items(), COLORS):
        mean_val  = np.mean(durs)
        total_val = sum(durs)

        # Histogram (semi-transparent so both are visible)
        ax.hist(durs, bins=bins, alpha=0.45, color=color,
                edgecolor="white", linewidth=0.6,
                label=f"{label}  (n={len(durs)}, total={total_val/60:.1f} min, mean={mean_val:.2f}s)")

        # Mean line
        ax.axvline(mean_val, color=color, linewidth=2.2, linestyle="--")

        # Mean label
        ax.text(mean_val + 0.05, ax.get_ylim()[1] * 0.95,
                f"μ={mean_val:.2f}s",
                color=color, fontsize=10, fontweight="bold", va="top")

    ax.set_xlabel("Clip duration (seconds)", fontsize=12)
    ax.set_ylabel("Number of clips", fontsize=12)
    ax.set_title("Recording Duration Distribution per Dataset\n"
                 "Dashed lines = mean",
                 fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()

    out_path = Path(config.cross_examination_folder)
    if not out_path.is_absolute():
        out_path = Path(config.__file__).resolve().parent / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "dataset_durations.png"

    plt.savefig(str(out_file), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()

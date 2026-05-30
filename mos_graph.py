#!/usr/bin/env python3
# CALLING AHJIN — Brought to you by Ahmed Hussein and Julius Norén from JTH.
"""
mos_graph.py
------------
Reads all MOS_<name>.txt files from the Cross_Examination folder,
averages scores across all raters, normalises to 0–1, and produces
a grouped bar chart ordered from best to worst overall score.

Expected file format (one line per clip):
    1. 4, 5, 5
    2. 3, 4, 3
    ...
Three comma-separated scores per line: MOS, SMOS, PSMOS

Normalisation: (raw_score - 1) / 4
    1 → 0.00  |  2 → 0.25  |  3 → 0.50  |  4 → 0.75  |  5 → 1.00

If a key.txt file exists in the folder it maps clip numbers to
original filenames so the chart shows meaningful labels.

Output:
    Cross_Examination/mos_results.png
    Cross_Examination/mos_results.csv

Usage:
    python mos_graph.py
"""

from __future__ import annotations
import sys, csv, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ── Helpers ────────────────────────────────────────────────────────────────────

def resolve_folder() -> Path:
    folder = Path(config.cross_examination_folder)
    if not folder.is_absolute():
        folder = (Path(config.__file__).resolve().parent / folder).resolve()
    if not folder.is_dir():
        sys.exit(f"Cross_Examination folder not found: {folder}")
    return folder


def load_key(folder: Path) -> dict[int, str]:
    """Parse key.txt into {clip_number: original_filename}.
    Checks Cross_Examination/ and Test/ folders."""
    project_root = Path(config.__file__).resolve().parent
    candidates = [
        folder / "key.txt",
        project_root / "Test" / "key.txt",
    ]
    key_path = next((p for p in candidates if p.exists()), None)
    if key_path is None:
        return {}
    print(f"Using key: {key_path}")
    mapping = {}
    for line in key_path.read_text(encoding="utf-8").splitlines():
        # e.g.  "1.mp3         Ash see shells.mp3"
        m = re.match(r"(\d+)\.\S*\s+(.+)", line.strip())
        if m:
            mapping[int(m.group(1))] = m.group(2).strip()
    return mapping


def parse_mos_file(path: Path) -> dict[int, tuple[float, float, float]]:
    """
    Parse one MOS file. Each line:  '<n>. <mos>, <smos>, <psmos>'
    Returns {clip_number: (mos, smos, psmos)}.
    """
    scores = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\d+)\.\s*(.+)", line)
        if not m:
            continue
        clip_num = int(m.group(1))
        vals = [float(v.strip()) for v in m.group(2).split(",") if v.strip()]
        if len(vals) == 3:
            scores[clip_num] = tuple(vals)
    return scores


def normalise(score: float) -> float:
    """Map 1–5 scale to 0–1: (score - 1) / 4."""
    return (score - 1.0) / 4.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    folder = resolve_folder()

    # ── Load all MOS files ─────────────────────────────────────────────────────
    mos_files = sorted(folder.glob("MOS_*.txt"))
    if not mos_files:
        sys.exit(f"No MOS_*.txt files found in {folder}")

    print(f"Found {len(mos_files)} rater file(s): {[f.name for f in mos_files]}")

    # Accumulate scores per clip: {clip_num: [[mos,smos,psmos], ...]}
    all_scores: dict[int, list[list[float]]] = defaultdict(list)
    for path in mos_files:
        parsed = parse_mos_file(path)
        for clip_num, (mos, smos, psmos) in parsed.items():
            all_scores[clip_num].append([mos, smos, psmos])

    # ── Load key.txt for label names ───────────────────────────────────────────
    key = load_key(folder)

    # ── Average + normalise ────────────────────────────────────────────────────
    results = []   # (label, avg_mos_norm, avg_smos_norm, avg_psmos_norm, overall_norm)
    for clip_num in sorted(all_scores.keys()):
        rater_scores = np.array(all_scores[clip_num])  # (n_raters, 3)
        avg = rater_scores.mean(axis=0)                # (3,) raw averages

        mos_n   = normalise(avg[0])
        smos_n  = normalise(avg[1])
        psmos_n = normalise(avg[2])
        overall = (mos_n + smos_n + psmos_n) / 3.0

        label = key.get(clip_num, f"Clip {clip_num}")
        # Shorten long filenames for readability
        label = Path(label).stem[:28]

        results.append((label, mos_n, smos_n, psmos_n, overall, clip_num))

    # Sort best → worst by overall score
    results.sort(key=lambda r: r[4], reverse=True)

    # ── Print summary ──────────────────────────────────────────────────────────
    print()
    print(f"  {'Clip':<30}  {'MOS':>6}  {'SMOS':>6}  {'PSMOS':>6}  {'Overall':>8}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")
    for label, mos_n, smos_n, psmos_n, overall, _ in results:
        print(f"  {label:<30}  {mos_n:>6.3f}  {smos_n:>6.3f}  {psmos_n:>6.3f}  {overall:>8.3f}")
    print()

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = folder / "mos_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "clip_number", "MOS_norm", "SMOS_norm",
                         "PSMOS_norm", "overall_norm"])
        for label, mos_n, smos_n, psmos_n, overall, clip_num in results:
            writer.writerow([label, clip_num,
                             f"{mos_n:.4f}", f"{smos_n:.4f}",
                             f"{psmos_n:.4f}", f"{overall:.4f}"])
    print(f"  CSV saved: {csv_path}")

    # ── Plot: one bar chart per dimension, sorted independently ───────────────
    dimensions = [
        ("Naturalness MOS",          "Does it sound human?",              [r[1] for r in results], "#3498db", "mos_naturalness.png"),
        ("Speaker Similarity SMOS",  "Does it sound like Ash?",           [r[2] for r in results], "#2ecc71", "mos_speaker_similarity.png"),
        ("Paralinguistic PSMOS",     "Does the speaker speak like Ash?",  [r[3] for r in results], "#e74c3c", "mos_paralinguistic.png"),
    ]

    all_labels = [r[0] for r in results]

    for dim_title, dim_subtitle, dim_scores, color, filename in dimensions:
        # Sort this dimension independently best → worst
        pairs = sorted(zip(all_labels, dim_scores), key=lambda x: x[1], reverse=True)
        s_labels, s_scores = zip(*pairs)
        n = len(s_labels)
        x = np.arange(n)

        fig, ax = plt.subplots(figsize=(max(12, n * 1.5), 5))
        bars = ax.bar(x, s_scores, color=color, edgecolor="white",
                      linewidth=0.8, zorder=3)

        # Value labels
        for bar, score in zip(bars, s_scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{score:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(s_labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Normalised Score (0 – 1)", fontsize=12)
        ax.set_ylim(0, 1.18)
        ax.set_title(
            f"{dim_title}  —  \"{dim_subtitle}\"\n"
            f"{len(mos_files)} rater(s)  |  Ordered best → worst  |  Score = (raw − 1) / 4",
            fontsize=11, pad=12)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        plt.tight_layout()
        out = folder / filename
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  PNG saved: {out}")

    # ── Plot: combined overview chart sorted by overall ────────────────────────
    results_by_overall = sorted(results, key=lambda r: r[4], reverse=True)
    n         = len(results_by_overall)
    labels    = [r[0] for r in results_by_overall]
    mos_v     = [r[1] for r in results_by_overall]
    smos_v    = [r[2] for r in results_by_overall]
    psmos_v   = [r[3] for r in results_by_overall]
    overall_v = [r[4] for r in results_by_overall]

    x     = np.arange(n)
    width = 0.22

    fig, ax = plt.subplots(figsize=(max(12, n * 1.6), 6))
    bars1 = ax.bar(x - width, mos_v,   width, label="Naturalness (MOS)",
                   color="#3498db", edgecolor="white", linewidth=0.6)
    bars2 = ax.bar(x,         smos_v,  width, label="Speaker Similarity (SMOS)",
                   color="#2ecc71", edgecolor="white", linewidth=0.6)
    bars3 = ax.bar(x + width, psmos_v, width, label="Paralinguistic (PSMOS)",
                   color="#e74c3c", edgecolor="white", linewidth=0.6)

    ax.plot(x, overall_v, "D", color="#f39c12", markersize=7,
            zorder=5, label="Overall average")
    ax.plot(x, overall_v, color="#f39c12", linewidth=1.2, alpha=0.5, zorder=4)

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Normalised Score (0 – 1)", fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_title(
        f"MOS Evaluation — All Dimensions  ({len(mos_files)} rater(s))\n"
        "Ordered best → worst overall  |  Score = (raw − 1) / 4",
        fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    png_path = folder / "mos_results.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PNG saved: {png_path}")


if __name__ == "__main__":
    main()

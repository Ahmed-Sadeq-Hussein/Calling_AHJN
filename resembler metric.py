#!/usr/bin/env python3
# CALLING AHJIN Brought to you by Ahmed Hussein and Julius Norén from JTH.
"""
cross_examine_speakers.py
-------------------------
Reads settings from config.py, collects every audio file directly inside
`cross_examination_folder`, embeds each one with Resemblyzer, computes the
full pairwise cosine-similarity matrix (0..1), and writes into that same
folder:

    speaker_similarity_heatmap.png   annotated heatmap
    speaker_similarity_matrix.csv    raw scores for further analysis

Cells at/above `Resemble_threshold` are flagged with a star and a box.
The diagonal (a clip vs itself = 1.00) is shown but not flagged, since
it's trivially a match.

Works with any format ffmpeg/librosa can decode (wav, mp3, mp4, m4a,
flac, ogg, ...) and mixes formats freely.

Install:
    pip install resemblyzer librosa soundfile matplotlib numpy

Run it from anywhere; it locates config.py and resolves the folder
relative to config.py's location if the path is not absolute.
Note : The heatmap is generated in the folder specifiedby the config file. 
"""

from __future__ import annotations

# Windows/Anaconda: PyTorch (Intel MKL) and numpy/numba each link their own
# OpenMP runtime, causing "OMP: Error #15 ... libiomp5md.dll already
# initialized" and an abort. Tolerating the duplicate is the standard
# pragmatic fix for inference workloads. MUST be set before numpy / torch /
# librosa are imported, so it lives at the very top.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import argparse
from pathlib import Path

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from resemblyzer import VoiceEncoder, preprocess_wav

# --- locate and load config.py -------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
try:
    import config  # noqa: E402
except ImportError:
    sys.exit("Could not import config.py — keep it next to this script.")

CONFIG_DIR = Path(config.__file__).resolve().parent
AUDIO_EXTS = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


def resolve_folder() -> Path:
    folder = Path(config.cross_examination_folder)
    if not folder.is_absolute():
        folder = (CONFIG_DIR / folder).resolve()
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")
    return folder


def collect_audio(folder: Path) -> list[Path]:
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if len(files) < 2:
        sys.exit(f"Need at least 2 audio files in {folder}, found {len(files)}.")
    return files


def load_waveform(path: Path) -> np.ndarray:
    """Decode any supported file, then resample/normalize/VAD-trim."""
    raw, source_sr = librosa.load(str(path), sr=None, mono=True)
    if raw.size == 0:
        raise RuntimeError("no audio samples")
    wav = preprocess_wav(raw, source_sr=source_sr)
    if wav.size == 0:
        raise RuntimeError("no speech left after silence trimming")
    return wav


def build_similarity_matrix(encoder: VoiceEncoder, files: list[Path], folder: Path):
    """Embed each file once, then S = E @ E.T (embeddings are L2-normalized)."""
    labels, embeddings, kept = [], [], []
    for i, path in enumerate(files, 1):
        rel = path.name
        print(f"[{i}/{len(files)}] embedding {rel}")
        try:
            embeddings.append(encoder.embed_utterance(load_waveform(path)))
            labels.append(rel)
            kept.append(path)
        except Exception as exc:  # noqa: BLE001
            print(f"    skipped ({exc})")
    if len(embeddings) < 2:
        sys.exit("Fewer than 2 files could be embedded — nothing to compare.")
    E = np.vstack(embeddings)            # (N, 256), already unit vectors
    S = E @ E.T                          # (N, N) cosine similarities
    return labels, S, E


def draw_reference_bar(ref_label: str, labels: list[str], similarities: list[float],
                       threshold: float, out_path: Path) -> None:
    """Bar chart of every clip's similarity to the primary reference, sorted high to low."""
    # Sort descending by similarity
    pairs = sorted(zip(labels, similarities), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_sims = zip(*pairs)

    n = len(sorted_labels)
    colors = ["#2ecc71" if s >= threshold else "#e74c3c" for s in sorted_sims]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 5))
    x = np.arange(n)
    bars = ax.bar(x, sorted_sims, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels on each bar
    for bar, s in zip(bars, sorted_sims):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{s:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Threshold line
    ax.axhline(threshold, color="#f39c12", linewidth=1.8, linestyle="--",
               label=f"Threshold ({threshold:.2f})", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Similarity to Primary Reference: {ref_label}\n"
                 f"Ordered most → least similar",
                 fontsize=11, pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label=f"≥ threshold (same speaker)"),
        Patch(facecolor="#e74c3c", label=f"< threshold (different speaker)"),
    ]
    legend_elements.append(plt.Line2D([0], [0], color="#f39c12", linewidth=1.8,
                                       linestyle="--", label=f"Threshold ({threshold:.2f})"))
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def save_csv(labels, S, out_path: Path) -> None:
    header = "," + ",".join(labels)
    rows = [header]
    for name, row in zip(labels, S):
        rows.append(name + "," + ",".join(f"{v:.4f}" for v in row))
    out_path.write_text("\n".join(rows), encoding="utf-8")


def draw_heatmap(labels, S, threshold: float, out_path: Path) -> None:
    n = len(labels)
    cell = 0.6                                   # inches per cell
    size = max(6.0, cell * n + 4.0)
    fig, ax = plt.subplots(figsize=(size, size))

    disp = np.clip(S, 0.0, 1.0)                   # display only; CSV keeps raw
    im = ax.imshow(disp, cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(
        f"Speaker similarity  (★ = ≥ threshold {threshold:.2f})",
        fontsize=12, pad=12,
    )

    for i in range(n):
        for j in range(n):
            val = S[i, j]
            txt_color = "white" if disp[i, j] < 0.5 else "black"
            star = ""
            if i != j and val >= threshold:        # flag off-diagonal matches
                star = "\n★"
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="red", linewidth=2.0,
                ))
            ax.text(j, i, f"{val:.2f}{star}",
                    ha="center", va="center",
                    color=txt_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speaker similarity heatmap + optional ranked bar chart")
    parser.add_argument(
        "--reference", default=None, metavar="FILENAME",
        help="Primary reference audio (filename inside the folder, or full path). "
             "When set, generates a bar chart ranking all other clips by similarity to it.")
    parser.add_argument(
        "--folder", default=None, metavar="PATH",
        help="Override the folder to analyse (default: config.cross_examination_folder).")
    args = parser.parse_args()

    folder = Path(args.folder).resolve() if args.folder else resolve_folder()
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")
    threshold = float(config.Resemble_threshold)
    files = collect_audio(folder)
    print(f"Folder    : {folder}")
    print(f"Threshold : {threshold:.3f}")
    print(f"Files     : {len(files)}\n")

    encoder = VoiceEncoder()
    labels, S, E = build_similarity_matrix(encoder, files, folder)

    csv_path = folder / "speaker_similarity_matrix.csv"
    png_path = folder / "speaker_similarity_heatmap.png"
    save_csv(labels, S, csv_path)
    draw_heatmap(labels, S, threshold, png_path)

    print(f"\nWrote {png_path}")
    print(f"Wrote {csv_path}")

    # ── Optional ranked bar chart against a primary reference ─────────────────
    if args.reference:
        ref_path = Path(args.reference)
        if not ref_path.is_absolute():
            ref_path = (folder / args.reference).resolve()

        if ref_path.name in labels:
            # Reference is already embedded in the matrix — use its row
            ref_idx = labels.index(ref_path.name)
            other_labels = [l for i, l in enumerate(labels) if i != ref_idx]
            other_sims   = [float(S[ref_idx, i]) for i in range(len(labels)) if i != ref_idx]
            ref_label    = ref_path.name
        else:
            # Reference is an external file — embed it separately
            if not ref_path.exists():
                print(f"WARNING: reference file not found: {ref_path} — skipping bar chart.")
            else:
                print(f"\nEmbedding external reference: {ref_path.name} …")
                ref_emb    = encoder.embed_utterance(load_waveform(ref_path))
                other_sims = (E @ ref_emb).tolist()
                other_labels = labels
                ref_label    = ref_path.name

        bar_path = folder / "speaker_similarity_bar.png"
        draw_reference_bar(ref_label, other_labels, other_sims, threshold, bar_path)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
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
    return labels, S


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
    folder = resolve_folder()
    threshold = float(config.Resemble_threshold)
    files = collect_audio(folder)
    print(f"Folder    : {folder}")
    print(f"Threshold : {threshold:.3f}")
    print(f"Files     : {len(files)}\n")

    encoder = VoiceEncoder()
    labels, S = build_similarity_matrix(encoder, files, folder)

    csv_path = folder / "speaker_similarity_matrix.csv"
    png_path = folder / "speaker_similarity_heatmap.png"
    save_csv(labels, S, csv_path)
    draw_heatmap(labels, S, threshold, png_path)

    print(f"\nWrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
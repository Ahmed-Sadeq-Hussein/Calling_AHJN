#!/usr/bin/env python3
# CALLING AHJIN Brought to you by Ahmed Hussein and Julius Norén from JTH.
"""
compare_finetune.py
-------------------
Sequentially fine-tunes F5-TTS on two voice datasets so you can
compare the effect of short clips (ahmed) vs longer clips (Ahmed_2).

Training order:
    1. ahmed    — 98 clips, ~2–5s each  (short phonetic sentences)
    2. Ahmed_2  — 71 clips, ~7–9s each  (longer natural sentences)

Each produces its own checkpoint in:
    models/f5tts_finetune/ahmed/
    models/f5tts_finetune/Ahmed_2/

After both finish, use clone_voice.py to compare them side-by-side.

Usage:
    python compare_finetune.py
    python compare_finetune.py --epochs 100 --batch-size 6
    python compare_finetune.py --only ahmed       # run just one
    python compare_finetune.py --only Ahmed_2
    python compare_finetune.py --resume           # resume interrupted runs
"""

from __future__ import annotations
import os, sys, subprocess, argparse, time
# KMP_DUPLICATE_LIB_OK: avoids fatal OMP #15 abort from duplicate OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Suppress noisy HuggingFace symlinks warning on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ── Dataset definitions ────────────────────────────────────────────────────────
# Two contrasting datasets recorded from the same speaker (Ahmed Hussein):
#   ahmed   — phonetically diverse short sentences, useful for articulation coverage
#   Ahmed_2 — natural longer sentences, closer to conversational speech patterns
# Training on both lets us compare how clip length and prosody affect clone quality.

DATASETS = {
    "ahmed": {
        "speaker":  "ahmed",
        "manifest": config.PROCESSED_DATA_DIR / "ahmed_filelist.txt",
        "clips":    98,
        "avg_dur":  4.6,   # seconds per clip (used for time estimation)
        "desc":     "Short phonetic sentences (2–5s each)",
    },
    "Ahmed_2": {
        "speaker":  "Ahmed_2",
        "manifest": config.PROCESSED_DATA_DIR / "Ahmed_2_filelist.txt",
        "clips":    71,
        "avg_dur":  7.1,   # seconds per clip
        "desc":     "Longer natural sentences (7–9s each)",
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def estimate_time(clips: int, avg_dur: float, batch_size: int, epochs: int) -> str:
    """
    Rough wall-clock estimate for a training run on an RTX 3080.

    Formula:  updates/epoch × seconds/update × epochs
    seconds/update = 3.5 s base overhead + 0.4 s per second of audio in the batch.
    Returns a human-readable string like '~2h 30min' or '~45min'.
    """
    import math
    updates_per_epoch = math.ceil(clips / batch_size)
    s_per_update      = 3.5 + avg_dur * 0.4   # empirically measured on RTX 3080
    total_minutes     = updates_per_epoch * s_per_update * epochs / 60
    h = int(total_minutes // 60)
    m = int(total_minutes % 60)
    return f"~{h}h {m:02d}min" if h else f"~{m}min"


def run_finetune(speaker: str, manifest: Path, epochs: int, batch_size: int,
                 save_every: int, warmup: int, resume: bool) -> tuple[bool, float]:
    """
    Launch finetune_voice.py as a subprocess for one dataset.

    Delegates all training logic to finetune_voice.py rather than duplicating it.
    Returns (success: bool, elapsed_seconds: float).
    """
    script = Path(__file__).resolve().parent / "finetune_voice.py"

    cmd = [
        sys.executable, str(script),
        "--speaker",    speaker,
        "--manifest",   str(manifest),
        "--epochs",     str(epochs),
        "--batch-size", str(batch_size),
        "--save-every", str(save_every),
        "--warmup",     str(warmup),
    ]
    if resume:
        cmd.append("--resume")

    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0

    return result.returncode == 0, elapsed


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential fine-tune on ahmed then Ahmed_2 for comparison")
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Epochs per dataset (default 100)")
    parser.add_argument(
        "--batch-size", type=int, default=6,
        help="Clips per step (default 6 — safe for both short and long clips)")
    parser.add_argument(
        "--save-every", type=int, default=50,
        help="Save checkpoint every N updates (default 50 ≈ every 3–4 epochs)")
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="LR warmup steps (default 10)")
    parser.add_argument(
        "--only", choices=["ahmed", "Ahmed_2"], default=None,
        help="Run only one dataset instead of both")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume each training from its latest checkpoint")
    args = parser.parse_args()

    # Decide which datasets to run
    if args.only:
        run_order = [args.only]
    else:
        run_order = ["ahmed", "Ahmed_2"]

    # ── Print plan ─────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  SEQUENTIAL VOICE FINE-TUNE COMPARISON")
    print("=" * 62)
    print(f"  Epochs per run  : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Save every      : {args.save_every} updates")
    print()

    total_est = 0
    for key in run_order:
        ds  = DATASETS[key]
        est = estimate_time(ds["clips"], ds["avg_dur"], args.batch_size, args.epochs)
        print(f"  {ds['speaker']:12s}  {ds['clips']} clips  {est}  —  {ds['desc']}")

    print()

    # Check manifests exist
    for key in run_order:
        ds = DATASETS[key]
        if not ds["manifest"].exists():
            sys.exit(f"Manifest not found: {ds['manifest']}\n"
                     f"Run record_dataset.py --speaker {ds['speaker']} first.")

    # ── Run sequentially ───────────────────────────────────────────────────────
    results = {}
    t_total_start = time.perf_counter()

    for i, key in enumerate(run_order, 1):
        ds = DATASETS[key]
        print(f"{'─' * 62}")
        print(f"  [{i}/{len(run_order)}]  Training on {ds['speaker']}  —  {ds['desc']}")
        print(f"{'─' * 62}")
        print()

        success, elapsed = run_finetune(
            speaker    = ds["speaker"],
            manifest   = ds["manifest"],
            epochs     = args.epochs,
            batch_size = args.batch_size,
            save_every = args.save_every,
            warmup     = args.warmup,
            resume     = args.resume,
        )

        results[key] = {"success": success, "elapsed": elapsed}

        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        status = "✓ complete" if success else "✗ stopped early"
        print()
        print(f"  {ds['speaker']} — {status}  ({h}h {m:02d}m {s:02d}s)")
        print()

    # ── Final summary ──────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total_start
    th = int(total_elapsed // 3600)
    tm = int((total_elapsed % 3600) // 60)

    print("=" * 62)
    print("  ALL DONE — COMPARISON SUMMARY")
    print("=" * 62)
    print()

    for key in run_order:
        ds  = DATASETS[key]
        res = results.get(key, {})
        el  = res.get("elapsed", 0)
        h, m = int(el // 3600), int((el % 3600) // 60)
        ckpt = config.MODELS_DIR / "f5tts_finetune" / ds["speaker"] / "model_last.pt"
        print(f"  {ds['speaker']:12s}  {h}h {m:02d}m  "
              f"{'checkpoint saved' if ckpt.exists() else 'no checkpoint'}")

    print()
    print(f"  Total wall-clock : {th}h {tm:02d}m")
    print()

    # Print ready-to-use comparison commands
    ref = config.DATA_DIR / "references" / "ahmed_reference.wav"
    test_text = "Hi, I am Ashy Washy your personal AI assistant. I miss you already."

    print("  Test each model with the same reference and text:")
    print()
    for key in run_order:
        ds   = DATASETS[key]
        ckpt = config.MODELS_DIR / "f5tts_finetune" / ds["speaker"] / "model_last.pt"
        name = f"compare_{ds['speaker'].lower()}"
        print(f"  # {ds['speaker']} ({ds['desc']})")
        print(f"  python clone_voice.py --reference \"{ref}\" \\")
        print(f"      --checkpoint \"{ckpt}\" \\")
        print(f"      --text \"{test_text}\" \\")
        print(f"      --out-name {name}")
        print()

    print("=" * 62)


if __name__ == "__main__":
    main()

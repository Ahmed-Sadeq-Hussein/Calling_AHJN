#!/usr/bin/env python3
"""
finetune_voice.py
-----------------
Fine-tunes F5-TTS on YOUR recorded voice clips, then saves the checkpoint
to models/f5tts_finetune/<speaker>/ for permanent reuse.

Prerequisites:
    1. python record_dataset.py --speaker ahmed   (builds data/processed/ahmed_filelist.txt)
    2. python finetune_voice.py --speaker ahmed   (this script)

After training, use your voice model with clone_voice.py:
    python clone_voice.py --text "Hello, it's me." \\
        --checkpoint models/f5tts_finetune/ahmed/model_last.pt

Usage:
    python finetune_voice.py --speaker ahmed
    python finetune_voice.py --speaker ahmed --epochs 50 --lr 1e-5
    python finetune_voice.py --speaker ahmed --resume
"""

from __future__ import annotations
import os, sys, shutil, subprocess, argparse, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path
from importlib.resources import files as pkg_files

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

MODEL_NAME = "F5TTS_v1_Base"   # must match one of F5TTS_v1_Base / F5TTS_Base / E2TTS_Base


# ── Paths ──────────────────────────────────────────────────────────────────────

def arrow_dir(speaker: str) -> Path:
    """Where F5-TTS looks for the pre-built Arrow dataset."""
    raw = str(pkg_files("f5_tts").joinpath(f"../../data/{speaker}_custom"))
    return Path(raw).resolve()

def ckpt_dir_f5(speaker: str) -> Path:
    """Where F5-TTS saves checkpoints internally (inside Anaconda)."""
    raw = str(pkg_files("f5_tts").joinpath(f"../../ckpts/{speaker}"))
    return Path(raw).resolve()

def out_dir(speaker: str) -> Path:
    """Our project directory — final home for the saved checkpoint."""
    return config.MODELS_DIR / "f5tts_finetune" / speaker


# ── Vocab ──────────────────────────────────────────────────────────────────────

def find_vocab_path() -> str:
    import site
    dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    try:
        dirs = list(dirs) + [site.getusersitepackages()]
    except Exception:
        pass
    for sp in dirs:
        v = Path(sp) / "f5_tts" / "infer" / "examples" / "vocab.txt"
        if v.exists():
            return str(v)
    for entry in sys.path:
        v = Path(entry) / "f5_tts" / "infer" / "examples" / "vocab.txt"
        if v.exists():
            return str(v)
    sys.exit("Could not find F5-TTS vocab.txt.")


# ── Arrow dataset prep ─────────────────────────────────────────────────────────

def build_arrow_dataset(manifest: Path, speaker: str) -> None:
    """
    Converts <speaker>_filelist.txt  →  Arrow dataset at arrow_dir(speaker).
    Skips if already built.
    """
    target = arrow_dir(speaker)
    arrow  = target / "raw.arrow"
    dur_js = target / "duration.json"

    if arrow.exists() and dur_js.exists():
        print(f"Arrow dataset already exists: {target}")
        return

    print(f"Building Arrow dataset for {speaker} …")

    import soundfile as sf
    from datasets.arrow_writer import ArrowWriter
    from tqdm import tqdm

    rows: list[tuple[str, str]] = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 3:
                wav_path, _spk, text = parts
            elif len(parts) == 2:
                wav_path, text = parts
            else:
                continue
            rows.append((wav_path.strip(), text.strip()))

    records, durations = [], []
    for wav_path, text in tqdm(rows, desc="  reading durations", unit="clip"):
        p = Path(wav_path)
        if not p.exists():
            continue
        try:
            dur = sf.info(str(p)).duration
        except Exception:
            continue
        if not (0.3 <= dur <= 30.0):
            continue
        records.append({"audio_path": str(p), "text": text, "duration": float(dur)})
        durations.append(float(dur))

    if not records:
        sys.exit(f"No valid clips found in {manifest}")

    target.mkdir(parents=True, exist_ok=True)
    with ArrowWriter(path=str(arrow)) as writer:
        for rec in tqdm(records, desc="  writing arrow", unit="clip"):
            writer.write(rec)
        writer.finalize()

    with open(dur_js, "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f)

    total_h = sum(durations) / 60
    print(f"  {len(records)} clips  /  {total_h:.1f} min  →  {target}")


# ── Find latest checkpoint ──────────────────────────────────────────────────────

def find_latest_ckpt(speaker: str) -> str | None:
    """Look in both the F5-TTS internal dir and our project dir."""
    search_dirs = [ckpt_dir_f5(speaker), out_dir(speaker)]
    ckpts = []
    for d in search_dirs:
        ckpts += list(d.rglob("*.pt")) + list(d.rglob("*.safetensors"))
    if not ckpts:
        return None
    return str(max(ckpts, key=lambda p: p.stat().st_mtime))


# ── Copy checkpoint to project dir ─────────────────────────────────────────────

def copy_checkpoints_to_project(speaker: str) -> None:
    """
    F5-TTS saves checkpoints inside Anaconda's Lib directory.
    This copies them to models/f5tts_finetune/<speaker>/ in your project.
    """
    src = ckpt_dir_f5(speaker)
    dst = out_dir(speaker)

    if not src.exists():
        print(f"  No checkpoints found at {src}")
        return

    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in list(src.rglob("*.pt")) + list(src.rglob("*.safetensors")):
        dest_file = dst / f.name
        if not dest_file.exists() or f.stat().st_mtime > dest_file.stat().st_mtime:
            shutil.copy2(str(f), str(dest_file))
            copied += 1

    if copied:
        print(f"  Copied {copied} checkpoint(s) → {dst}")
    else:
        print(f"  Checkpoints already up to date in {dst}")


# ── Training ───────────────────────────────────────────────────────────────────

def run_finetune(speaker: str, resume: bool, epochs: int, lr: float,
                 batch_size: int, save_every: int, warmup: int) -> None:

    vocab_path = find_vocab_path()
    print(f"Vocab      : {vocab_path}")

    cli = shutil.which("f5-tts_finetune-cli")
    cli_args = [cli] if cli else [sys.executable, "-m", "f5_tts.train.finetune_cli"]

    cmd = cli_args + [
        "--exp_name",             MODEL_NAME,
        "--dataset_name",         speaker,         # Arrow dir: data/<speaker>_custom/
        "--learning_rate",        str(lr),
        "--batch_size_per_gpu",   str(batch_size),
        "--batch_size_type",      "sample",
        "--epochs",               str(epochs),
        "--num_warmup_updates",   str(warmup),
        "--save_per_updates",     str(save_every),
        "--last_per_updates",     str(save_every), # same cadence — avoids divide-by-zero
        "--keep_last_n_checkpoints", "2",          # keep only 2 checkpoints — prevents disk fill
        "--finetune",
        "--tokenizer",            "custom",
        "--tokenizer_path",       vocab_path,
    ]

    if resume:
        ckpt = find_latest_ckpt(speaker)
        if ckpt:
            print(f"Resuming   : {ckpt}")
            cmd += ["--pretrain", ckpt]
        else:
            print("No checkpoint found — starting from F5-TTS pretrained model.")

    print()
    print("=" * 60)
    print(f"Fine-tuning F5-TTS on speaker: {speaker}")
    print("=" * 60)
    print(f"  Epochs      : {epochs}")
    print(f"  LR          : {lr}")
    print(f"  Batch size  : {batch_size} clips/step")
    print(f"  Save every  : {save_every} updates")
    print(f"  Arrow data  : {arrow_dir(speaker)}")
    print(f"  Checkpoints : {out_dir(speaker)}")
    print()
    print("  Press Ctrl+C to stop — resume with:  python finetune_voice.py --resume")
    print()

    out_dir(speaker).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd)

    # Always try to copy whatever was saved, even on non-zero exit
    print()
    print("Copying checkpoints to project directory …")
    copy_checkpoints_to_project(speaker)

    if result.returncode == 0:
        print()
        print("=" * 60)
        print("Fine-tuning complete!")
        dst = out_dir(speaker)
        pts = list(dst.glob("*.pt")) + list(dst.glob("*.safetensors"))
        if pts:
            latest = max(pts, key=lambda p: p.stat().st_mtime)
            print(f"  Checkpoint : {latest}")
            print()
            print("Use your voice model:")
            print(f"  python clone_voice.py --text \"Hello.\" \\")
            print(f"      --checkpoint \"{latest}\"")
        print("=" * 60)
    else:
        print()
        print("Training stopped (non-zero exit or Ctrl+C).")
        print(f"Resume: python finetune_voice.py --speaker {speaker} --resume")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune F5-TTS on your recorded voice")
    parser.add_argument("--speaker", required=True,
        help="Speaker name, e.g. ahmed")
    parser.add_argument("--manifest", default=None,
        help="Path to filelist (default: data/processed/<speaker>_filelist.txt)")
    parser.add_argument("--resume", action="store_true",
        help="Resume from the latest checkpoint")
    parser.add_argument("--epochs", type=int, default=100,
        help="Training epochs (default 100 — fast with ~100 clips)")
    parser.add_argument("--lr", type=float, default=1e-5,
        help="Learning rate (default 1e-5)")
    parser.add_argument("--batch-size", type=int, default=4,
        help="Clips per step (default 4 — safe for 10 GB VRAM)")
    parser.add_argument("--save-every", type=int, default=10,
        help="Save checkpoint every N updates (default 10 — ~every half epoch for 99 clips)")
    parser.add_argument("--warmup", type=int, default=10,
        help="LR warmup steps (default 10)")
    parser.add_argument("--rebuild-dataset", action="store_true",
        help="Force rebuild the Arrow dataset even if it already exists")
    args = parser.parse_args()

    manifest = (Path(args.manifest) if args.manifest
                else config.PROCESSED_DATA_DIR / f"{args.speaker}_filelist.txt")

    if not manifest.exists():
        sys.exit(
            f"Manifest not found: {manifest}\n"
            f"Run first: python record_dataset.py --speaker {args.speaker}")

    print()
    print(f"=== F5-TTS Voice Fine-tuning: {args.speaker} ===")
    print()
    print(f"Manifest   : {manifest}")

    # Force rebuild if requested
    if args.rebuild_dataset:
        target = arrow_dir(args.speaker)
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed old Arrow dataset: {target}")

    build_arrow_dataset(manifest, args.speaker)
    print()

    run_finetune(
        speaker    = args.speaker,
        resume     = args.resume,
        epochs     = args.epochs,
        lr         = args.lr,
        batch_size = args.batch_size,
        save_every = args.save_every,
        warmup     = args.warmup,
    )


if __name__ == "__main__":
    main()

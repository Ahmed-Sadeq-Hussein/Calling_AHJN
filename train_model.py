#!/usr/bin/env python3
"""
train_model.py
--------------
Continues training the F5-TTS pretrained model on your downloaded LibriTTS data.
This is "continued pre-training" — starts from the pretrained checkpoint so you
get quality from day 1, not weeks from now.

Prerequisites:
    1. python download_mls.py   (done — 220h / 119k clips)
    2. python convert_manifest.py  (creates data/processed/metadata.csv)
    3. python train_model.py       (this script)

Hardware: RTX 3080 (10 GB VRAM)
Expected time: ~12–24 h for 3 epochs on 220h of data

Checkpoints are saved to:
    models/f5tts_librispeech/

Resume after interruption:
    python train_model.py           (auto-detects latest checkpoint)
    python train_model.py --resume  (same thing, explicit)

Use the trained model for cloning:
    python clone_voice.py --text "Hello." \\
        --checkpoint models/f5tts_librispeech/f5tts_librispeech/model_last.pt
"""

from __future__ import annotations
import os, sys, shutil, subprocess, argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduces fragmentation OOM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


MODEL_NAME   = "F5TTS_v1_Base"   # must be one of F5TTS_v1_Base / F5TTS_Base / E2TTS_Base
DATASET_NAME = "librispeech"     # free-form name; F5-TTS looks for data/librispeech/
TRAIN_CSV    = config.PROCESSED_DATA_DIR / "metadata.csv"
VAL_CSV      = config.PROCESSED_DATA_DIR / "metadata_val.csv"
# F5-TTS saves checkpoints to ckpts/<MODEL_NAME>/ by default
OUT_DIR      = config.MODELS_DIR / "f5tts_librispeech"

# ── Manual vocab override ──────────────────────────────────────────────────────
# If find_vocab_path() fails, set this to the absolute path of vocab.txt, e.g.:
#   VOCAB_PATH_OVERRIDE = r"C:\Users\music\anaconda3\Lib\site-packages\f5_tts\infer\examples\vocab.txt"
VOCAB_PATH_OVERRIDE: str | None = None


def setup_dataset_dir() -> Path:
    """
    F5-TTS finetune CLI looks for data in  data/<dataset_name>/metadata.csv
    relative to the current working directory.
    We create that folder and copy our metadata CSVs there.
    """
    import shutil
    dataset_dir = config.DATA_DIR / DATASET_NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # F5-TTS expects metadata.csv (train) and optionally metadata_eval.csv (val)
    dst_train = dataset_dir / "metadata.csv"
    dst_val   = dataset_dir / "metadata_eval.csv"

    shutil.copy2(TRAIN_CSV, dst_train)
    shutil.copy2(VAL_CSV,   dst_val)

    print(f"Dataset dir : {dataset_dir}")
    print(f"  metadata.csv      → {sum(1 for _ in open(dst_train)):,} lines")
    print(f"  metadata_eval.csv → {sum(1 for _ in open(dst_val)):,} lines")
    return dataset_dir


def get_arrow_dir() -> Path:
    """Compute the path where F5-TTS expects the pre-built Arrow dataset."""
    from importlib.resources import files as pkg_files
    raw = str(pkg_files("f5_tts").joinpath(f"../../data/{DATASET_NAME}_custom"))
    return Path(raw).resolve()


def check_prerequisites() -> None:
    missing = []
    if not TRAIN_CSV.exists():
        missing.append(f"  {TRAIN_CSV}  (run: python convert_manifest.py)")
    if not VAL_CSV.exists():
        missing.append(f"  {VAL_CSV}    (run: python convert_manifest.py)")
    if missing:
        print("Missing files:")
        for m in missing:
            print(m)
        sys.exit(1)

    n_train = sum(1 for _ in open(TRAIN_CSV, encoding="utf-8"))
    n_val   = sum(1 for _ in open(VAL_CSV,   encoding="utf-8"))
    print(f"Train clips : {n_train:,}")
    print(f"Val   clips : {n_val:,}")

    # Check Arrow dataset
    arrow_dir = get_arrow_dir()
    arrow_ok  = (arrow_dir / "raw.arrow").exists() or (arrow_dir / "raw").is_dir()
    dur_ok    = (arrow_dir / "duration.json").exists()

    if not (arrow_ok and dur_ok):
        print()
        print("=" * 60)
        print("Arrow dataset not found.")
        print(f"  Expected: {arrow_dir}")
        print()
        print("Run this first (takes ~2-4 min for 119 k clips):")
        print("  python prepare_dataset.py")
        print("=" * 60)
        sys.exit(1)

    print(f"Arrow dir   : {arrow_dir}  ✓")


def find_latest_checkpoint() -> str | None:
    """Look for the most recently modified .pt / .safetensors in OUT_DIR."""
    ckpts = list(OUT_DIR.rglob("*.pt")) + list(OUT_DIR.rglob("*.safetensors"))
    if not ckpts:
        return None
    latest = max(ckpts, key=lambda p: p.stat().st_mtime)
    return str(latest)


def find_vocab_path() -> str:
    """
    Locate the F5-TTS vocab.txt from the installed package.
    The finetune CLI tries to build a path relative to the repo root,
    which breaks for pip installs — we point it to the file directly.

    Tries four strategies in order:
      1. f5_tts.__file__ (standard import)
      2. importlib.util.find_spec (more reliable for namespace packages)
      3. site.getsitepackages() scan
      4. site.getusersitepackages() for --user installs
    """
    # Strategy 1: standard import
    try:
        import f5_tts
        vocab = Path(f5_tts.__file__).parent / "infer" / "examples" / "vocab.txt"
        if vocab.exists():
            return str(vocab)
    except Exception:
        pass

    # Strategy 2: importlib.util.find_spec
    try:
        import importlib.util
        spec = importlib.util.find_spec("f5_tts")
        if spec and spec.origin:
            vocab = Path(spec.origin).parent / "infer" / "examples" / "vocab.txt"
            if vocab.exists():
                return str(vocab)
    except Exception:
        pass

    # Strategy 3: scan all site-packages directories
    try:
        import site
        dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
        # also add user site
        try:
            dirs = list(dirs) + [site.getusersitepackages()]
        except Exception:
            pass
        for sp in dirs:
            vocab = Path(sp) / "f5_tts" / "infer" / "examples" / "vocab.txt"
            if vocab.exists():
                return str(vocab)
    except Exception:
        pass

    # Strategy 4: walk sys.path entries
    for entry in sys.path:
        vocab = Path(entry) / "f5_tts" / "infer" / "examples" / "vocab.txt"
        if vocab.exists():
            return str(vocab)

    sys.exit(
        "Could not find F5-TTS vocab.txt.\n"
        "Expected: <site-packages>/f5_tts/infer/examples/vocab.txt\n"
        "Confirm it exists with:\n"
        "  python -c \"import f5_tts, pathlib; "
        "p = pathlib.Path(f5_tts.__file__).parent / 'infer' / 'examples' / 'vocab.txt'; "
        "print(p, p.exists())\"\n"
        "Or pass the path directly by editing VOCAB_PATH at the top of train_model.py."
    )


def run_training(resume: bool, epochs: int, lr: float,
                 batch_size: int, save_every: int) -> None:

    setup_dataset_dir()

    cli = shutil.which("f5-tts_finetune-cli")
    if cli is None:
        cli_args = [sys.executable, "-m", "f5_tts.train.finetune_cli"]
    else:
        cli_args = [cli]

    if VOCAB_PATH_OVERRIDE:
        vocab_path = VOCAB_PATH_OVERRIDE
        print(f"Vocab file  : {vocab_path}  (manual override)")
    else:
        vocab_path = find_vocab_path()
        print(f"Vocab file  : {vocab_path}")

    cmd = cli_args + [
        "--exp_name",             MODEL_NAME,
        "--dataset_name",         DATASET_NAME,
        "--learning_rate",        str(lr),
        "--batch_size_per_gpu",   str(batch_size),
        "--batch_size_type",      "sample",   # 'sample' = N clips/step; 'frame' = N mel frames/step
        "--epochs",               str(epochs),
        "--num_warmup_updates",   "500",
        "--save_per_updates",          str(save_every),
        "--last_per_updates",          str(save_every),  # same cadence — 0 causes divide-by-zero
        "--keep_last_n_checkpoints",   "2",              # only keep 2 — prevents disk filling
        "--finetune",                                     # start from pretrained checkpoint
        "--tokenizer",            "custom",    # use vocab.txt directly
        "--tokenizer_path",       vocab_path,
        # --logger omitted → defaults to None (no logging)
    ]

    # Resume from a specific checkpoint
    if resume:
        ckpt = find_latest_checkpoint()
        if ckpt:
            print(f"Resuming from: {ckpt}")
            cmd += ["--pretrain", ckpt]
        else:
            print("No checkpoint found — starting from F5-TTS pretrained model.")

    print()
    print("=" * 60)
    print("Starting F5-TTS continued pre-training on LibriTTS")
    print("=" * 60)
    print(f"  Epochs      : {epochs}")
    print(f"  LR          : {lr}")
    print(f"  Batch size  : {batch_size} clips/step  (sample mode)")
    print(f"  Save every  : {save_every} updates")
    print(f"  Output dir  : {OUT_DIR}")
    print()
    print("  Press Ctrl+C to pause — resume anytime with:")
    print("    python train_model.py --resume")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print()
        print("=" * 60)
        print("Training CLI returned a non-zero exit code.")
        print("This may be a flag-name mismatch in your F5-TTS version.")
        print()
        print("Check available flags:")
        print("  f5-tts_finetune-cli --help")
        print()
        print("Or run the command manually:")
        print("  " + " ".join(str(a) for a in cmd))
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("Training complete!")
        ckpt = find_latest_checkpoint()
        if ckpt:
            print(f"Latest checkpoint: {ckpt}")
            print()
            print("Test your trained model:")
            print(f"  python clone_voice.py --text \"Hello world.\" \\")
            print(f"      --checkpoint \"{ckpt}\"")
        print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune F5-TTS on LibriTTS data")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest checkpoint in models/f5tts_librispeech/")
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Training epochs (default 3 — ~12-24h on RTX 3080). "
             "Each epoch sees all 119k clips once.")
    parser.add_argument(
        "--lr", type=float, default=5e-6,
        help="Learning rate (default 5e-6, lower than scratch to avoid forgetting)")
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Clips per GPU per step (default 2 — safe for 10GB VRAM with long LibriTTS clips)")
    parser.add_argument(
        "--save-every", type=int, default=1000,
        help="Save a checkpoint every N gradient updates (default 1000)")
    args = parser.parse_args()

    print()
    print("=== F5-TTS LibriTTS Training ===")
    print()
    check_prerequisites()
    print()

    run_training(
        resume     = args.resume,
        epochs     = args.epochs,
        lr         = args.lr,
        batch_size = args.batch_size,
        save_every = args.save_every,
    )


if __name__ == "__main__":
    main()

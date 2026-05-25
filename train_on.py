#!/usr/bin/env python3
# CALLING AHJIN — Brought to you by Ahmed Hussein and Julius Norén from JTH.
"""
train_on.py
-----------
General-purpose training script: pick any model and any dataset.

The chosen model's weights are used as the starting point.
The optimizer state and step counter are always reset so training
runs the full requested number of epochs from scratch.

Usage:
    python train_on.py --model ahmed --dataset Ahmed_2
    python train_on.py --model Ahmed_2 --dataset ahmed
    python train_on.py --model ahmed --dataset Ahmed_2 --epochs 50
    python train_on.py --list          # show all available models and datasets

Available model names  : any folder under  models/f5tts_finetune/
Available dataset names: any filelist under data/processed/<name>_filelist.txt

Output is saved to:
    models/f5tts_finetune/<model>_on_<dataset>/model_last.pt

Examples for cross-training:
    python train_on.py --model ahmed   --dataset Ahmed_2   # Ahmed_1_to_2 equivalent
    python train_on.py --model Ahmed_2 --dataset ahmed     # Ahmed_2_to_1 equivalent
"""

from __future__ import annotations
import os, sys, shutil, subprocess, argparse, time, tempfile
# KMP_DUPLICATE_LIB_OK: prevents fatal OMP #15 abort on Windows when both
# PyTorch (Intel MKL) and numpy load their own OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Suppress noisy HuggingFace symlinks warning on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path
from importlib.resources import files as pkg_files

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# F5-TTS CLI accepts only three hardcoded exp_name values; always use this one.
MODEL_NAME = "F5TTS_v1_Base"
PROJECT_ROOT = Path(__file__).resolve().parent


# ── Discovery ──────────────────────────────────────────────────────────────────

def available_models() -> dict[str, Path]:
    """
    Scan models/f5tts_finetune/ and return {folder_name: model_last.pt path}
    for every sub-folder that contains a model_last.pt checkpoint.
    """
    base = config.MODELS_DIR / "f5tts_finetune"
    models = {}
    if base.is_dir():
        for folder in sorted(base.iterdir()):
            ckpt = folder / "model_last.pt"
            if ckpt.exists():
                models[folder.name] = ckpt
    return models

def available_datasets() -> dict[str, Path]:
    """
    Scan data/processed/ and return {speaker_name: filelist_path} for every
    <name>_filelist.txt found there.
    """
    base = config.PROCESSED_DATA_DIR
    datasets = {}
    if base.is_dir():
        for f in sorted(base.glob("*_filelist.txt")):
            name = f.stem.replace("_filelist", "")
            datasets[name] = f
    return datasets

def list_available() -> None:
    """Print all discovered models and datasets to stdout for the --list flag."""
    print()
    print("Available models (from models/f5tts_finetune/):")
    for name, path in available_models().items():
        print(f"  {name:<20}  {path}")
    print()
    print("Available datasets (from data/processed/):")
    for name, path in available_datasets().items():
        print(f"  {name:<20}  {path}")
    print()


# ── Paths ──────────────────────────────────────────────────────────────────────

def arrow_dir(dataset: str) -> Path:
    """Return the path where F5-TTS expects the Arrow dataset for 'dataset'."""
    raw = str(pkg_files("f5_tts").joinpath(f"../../data/{dataset}_custom"))
    return Path(raw).resolve()

def find_ckpt_save_dir(dataset_name: str) -> Path:
    """
    Return the path where F5-TTS's finetune CLI saves checkpoints automatically.

    The CLI saves to <anaconda>/Lib/ckpts/<dataset_name>/ — two levels above
    the f5_tts package directory — regardless of which output name we choose.
    copy_checkpoints() moves them into our project folder after each run.
    """
    raw = str(pkg_files("f5_tts").joinpath(f"../../ckpts/{dataset_name}"))
    return Path(raw).resolve()

def out_dir(output_name: str) -> Path:
    """Return the project-local output directory for a training run."""
    return config.MODELS_DIR / "f5tts_finetune" / output_name

def find_vocab_path() -> str:
    """Locate the F5-TTS character vocabulary file from the pip-installed package."""
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


# ── Checkpoint stripping ───────────────────────────────────────────────────────

def strip_checkpoint(source: Path, dest: Path) -> None:
    """
    Write a clean checkpoint containing only model weights to 'dest'.

    A checkpoint saved by F5-TTS contains several keys beyond the weights:
        step / update counter  — causes the CLI to think training is already done
        optimizer state        — stale momentum for a different dataset
        scheduler state        — stale LR schedule position

    Stripping these forces the CLI to start from step 0 with fresh optimizer
    state, which is required for cross-training runs (train_on.py, cross_train.py).
    The EMA weights (ema_model_state_dict) are kept if present — they typically
    produce better inference quality than the raw model_state_dict.
    """
    import torch

    print(f"  Loading weights from  : {source.name}")
    state = torch.load(str(source), map_location="cpu", weights_only=False)

    if not isinstance(state, dict):
        # Already a raw state dict — pass through as-is
        torch.save(state, str(dest))
        print(f"  Stripped checkpoint   : {dest.name}  (raw state dict, passed through)")
        return

    clean = {}

    # Copy only the weight tensors, drop everything else
    if "model_state_dict" in state:
        clean["model_state_dict"] = state["model_state_dict"]
    if "ema_model_state_dict" in state:
        clean["ema_model_state_dict"] = state["ema_model_state_dict"]

    # If neither key found, just copy the whole dict (unknown format)
    if not clean:
        clean = state

    dropped = [k for k in state if k not in clean]
    torch.save(clean, str(dest))
    print(f"  Stripped checkpoint   : {dest.name}")
    if dropped:
        print(f"  Dropped keys          : {', '.join(dropped)}")


# ── Copy checkpoints from save dir to output dir ───────────────────────────────

def copy_checkpoints(output_name: str) -> None:
    """
    Copy new/updated checkpoints from F5-TTS's internal ckpts/ dir to our project folder.

    We build the Arrow dataset under output_name so the CLI's checkpoint directory is
    ckpts/output_name/ — this function copies from there to models/f5tts_finetune/output_name/.
    """
    # dataset_name == output_name (we build a fresh Arrow dataset under output_name)
    src = find_ckpt_save_dir(output_name)
    dst = out_dir(output_name)

    if not src.exists():
        print(f"  No checkpoint save dir found at: {src}")
        return

    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in list(src.rglob("*.pt")) + list(src.rglob("*.safetensors")):
        dest_file = dst / f.name
        if not dest_file.exists() or f.stat().st_mtime > dest_file.stat().st_mtime:
            shutil.copy2(str(f), str(dest_file))
            copied += 1

    if copied:
        print(f"  Copied {copied} checkpoint(s) to {dst}")
    else:
        print(f"  No new checkpoints to copy (already up to date)")


# ── Time estimate ──────────────────────────────────────────────────────────────

def count_clips(manifest: Path) -> int:
    """Count non-empty lines in the filelist manifest (= number of training clips)."""
    with open(manifest, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def estimate_time(clips: int, batch_size: int, epochs: int) -> str:
    """
    Rough wall-clock estimate (RTX 3080): ~5.5 s per gradient update.
    Returns a human-readable string like '~1h 30min' or '~45min'.
    """
    import math
    updates = math.ceil(clips / batch_size)
    total_s = updates * 5.5 * epochs   # 5.5 s/update measured on RTX 3080
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    return f"~{h}h {m:02d}min" if h else f"~{m}min"


# ── Main training function ─────────────────────────────────────────────────────

def run_training(model_name: str, model_ckpt: Path,
                 dataset_name: str, manifest: Path,
                 output_name: str,
                 epochs: int, batch_size: int,
                 save_every: int, warmup: int) -> tuple[bool, float]:
    """
    Execute a full training run: build Arrow dataset, strip checkpoint, launch CLI.

    Steps:
      1. Build the Arrow dataset under output_name (not dataset_name) so that
         F5-TTS finds no existing checkpoint for this run and does not exit early.
      2. Strip the step counter and optimizer state from model_ckpt into a temp file.
      3. Invoke the finetune CLI with the stripped checkpoint as --pretrain.
      4. Copy checkpoints from Anaconda's ckpts/ into models/f5tts_finetune/output_name/.
      5. Delete the temp stripped checkpoint.

    Returns (success: bool, elapsed_seconds: float).
    """
    # 1 — Build Arrow dataset under output_name (NOT dataset_name).
    #     Using a unique name means the CLI sees no pre-existing checkpoint folder
    #     and does not false-exit claiming training is already complete.
    arrow_output_name = output_name
    a_dir = arrow_dir(arrow_output_name)
    if not (a_dir / "raw.arrow").exists() and not (a_dir / "raw").is_dir():
        print(f"  Building Arrow dataset '{arrow_output_name}' from {dataset_name} clips ...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "finetune_voice", PROJECT_ROOT / "finetune_voice.py")
        fv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fv)
        fv.build_arrow_dataset(manifest, arrow_output_name)
    else:
        print(f"  Arrow dataset ready   : {a_dir}")

    # 2 — Strip step counter / optimizer / scheduler from the source checkpoint
    #     into a temporary file so training always starts from step 0.
    print()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        strip_checkpoint(model_ckpt, tmp_path)

        # 3 — Build CLI command using the stripped temp checkpoint as the starting point
        vocab_path = find_vocab_path()
        cli = shutil.which("f5-tts_finetune-cli")
        cli_args = [cli] if cli else [sys.executable, "-m", "f5_tts.train.finetune_cli"]

        cmd = cli_args + [
            "--exp_name",                MODEL_NAME,
            "--dataset_name",            arrow_output_name,  # unique per run → no checkpoint conflict
            "--learning_rate",           "1e-5",
            "--batch_size_per_gpu",      str(batch_size),
            "--batch_size_type",         "sample",
            "--epochs",                  str(epochs),
            "--num_warmup_updates",      str(warmup),
            "--save_per_updates",        str(save_every),
            "--last_per_updates",        str(save_every),
            "--keep_last_n_checkpoints", "2",
            "--finetune",
            "--pretrain",                str(tmp_path),
            "--tokenizer",               "custom",
            "--tokenizer_path",          vocab_path,
        ]

        out_dir(output_name).mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"  Starting training")
        print("=" * 60)
        print(f"  Model weights  : {model_name}  ({model_ckpt.name})")
        print(f"  Dataset        : {dataset_name}  ({count_clips(manifest)} clips)")
        print(f"  Output folder  : {out_dir(output_name)}")
        print(f"  Epochs         : {epochs}")
        print(f"  Batch size     : {batch_size}")
        print(f"  Save every     : {save_every} updates")
        print()

        t0 = time.perf_counter()
        result = subprocess.run(cmd)
        elapsed = time.perf_counter() - t0

    finally:
        # Always clean up temp file
        try:
            tmp_path.unlink()
        except Exception:
            pass

    # 4 — Copy checkpoints to output folder
    print()
    print("Copying checkpoints ...")
    copy_checkpoints(output_name)

    return result.returncode == 0, elapsed


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train any model on any dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_on.py --list
  python train_on.py --model ahmed   --dataset Ahmed_2
  python train_on.py --model Ahmed_2 --dataset ahmed
  python train_on.py --model ahmed   --dataset Ahmed_2 --epochs 50
  python train_on.py --model ahmed   --dataset Ahmed_2 --output my_experiment
""")

    parser.add_argument("--list", action="store_true",
        help="List all available models and datasets then exit")
    parser.add_argument("--model", default=None,
        help="Model name (folder under models/f5tts_finetune/) or path to a .pt file")
    parser.add_argument("--dataset", default=None,
        help="Dataset name (from data/processed/<name>_filelist.txt) or path to filelist")
    parser.add_argument("--output", default=None,
        help="Output folder name under models/f5tts_finetune/ "
             "(default: <model>_on_<dataset>)")
    parser.add_argument("--epochs", type=int, default=100,
        help="Training epochs (default 100)")
    parser.add_argument("--batch-size", type=int, default=6,
        help="Clips per gradient step (default 6)")
    parser.add_argument("--save-every", type=int, default=50,
        help="Save checkpoint every N updates (default 50)")
    parser.add_argument("--warmup", type=int, default=10,
        help="LR warmup steps (default 10)")
    args = parser.parse_args()

    if args.list:
        list_available()
        return

    # ── Resolve model ──────────────────────────────────────────────────────────
    if not args.model:
        list_available()
        sys.exit("Specify a model with --model <name>")

    models = available_models()
    model_ckpt = Path(args.model)

    if model_ckpt.exists() and model_ckpt.suffix == ".pt":
        model_name = model_ckpt.stem
    elif args.model in models:
        model_name = args.model
        model_ckpt = models[args.model]
    else:
        print(f"Model '{args.model}' not found.")
        list_available()
        sys.exit(1)

    # ── Resolve dataset ────────────────────────────────────────────────────────
    if not args.dataset:
        list_available()
        sys.exit("Specify a dataset with --dataset <name>")

    datasets = available_datasets()
    manifest = Path(args.dataset)

    if manifest.exists():
        dataset_name = manifest.stem.replace("_filelist", "")
    elif args.dataset in datasets:
        dataset_name = args.dataset
        manifest = datasets[args.dataset]
    else:
        print(f"Dataset '{args.dataset}' not found.")
        list_available()
        sys.exit(1)

    # ── Output name ────────────────────────────────────────────────────────────
    output_name = args.output or f"{model_name}_on_{dataset_name}"

    # ── Estimate and confirm ───────────────────────────────────────────────────
    clips = count_clips(manifest)
    est   = estimate_time(clips, args.batch_size, args.epochs)

    print()
    print("=" * 60)
    print("  TRAINING PLAN")
    print("=" * 60)
    print(f"  Model      : {model_name}")
    print(f"  Dataset    : {dataset_name}  ({clips} clips)")
    print(f"  Output     : {output_name}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Estimated  : {est}")
    print()

    # ── Run ────────────────────────────────────────────────────────────────────
    success, elapsed = run_training(
        model_name  = model_name,
        model_ckpt  = model_ckpt,
        dataset_name = dataset_name,
        manifest    = manifest,
        output_name = output_name,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        save_every  = args.save_every,
        warmup      = args.warmup,
    )

    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)

    print()
    print("=" * 60)
    ckpt = out_dir(output_name) / "model_last.pt"
    if ckpt.exists():
        status = "complete"
        print(f"  Training {status}  ({h}h {m:02d}m {s:02d}s)")
        print(f"  Checkpoint : {ckpt}")
        print()
        print("  Test with:")
        print(f"  python clone_voice.py --text \"Hello.\" --checkpoint \"{ckpt}\"")
    else:
        print(f"  Training stopped early  ({h}h {m:02d}m {s:02d}s)")
        print(f"  No checkpoint found at {ckpt}")
    print("=" * 60)


if __name__ == "__main__":
    main()

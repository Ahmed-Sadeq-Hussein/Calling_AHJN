#!/usr/bin/env python3
# CALLING AHJIN — Brought to you by Ahmed Hussein and Julius Norén from JTH.
"""
cross_train.py
--------------
Cross-trains the two fine-tuned voice models on each other's datasets:

    Ahmed_1_to_2 :  ahmed model  (trained on short clips)
                    continued on Ahmed_2 data  (longer natural sentences)

    Ahmed_2_to_1 :  Ahmed_2 model (trained on longer clips)
                    continued on ahmed data  (short phonetic sentences)

Starting checkpoints:
    models/f5tts_finetune/ahmed/model_last.pt
    models/f5tts_finetune/Ahmed_2/model_last.pt

Output checkpoints:
    models/f5tts_finetune/Ahmed_1_to_2/model_last.pt
    models/f5tts_finetune/Ahmed_2_to_1/model_last.pt

Usage:
    python cross_train.py
    python cross_train.py --epochs 100 --batch-size 6
    python cross_train.py --only Ahmed_1_to_2
    python cross_train.py --only Ahmed_2_to_1
    python cross_train.py --resume
"""

from __future__ import annotations
import os, sys, shutil, subprocess, argparse, time
# KMP_DUPLICATE_LIB_OK: avoids fatal OMP #15 abort from duplicate OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Suppress noisy HuggingFace symlinks warning on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path
from importlib.resources import files as pkg_files

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# F5-TTS CLI accepts only three hardcoded exp_name choices; always use this one.
MODEL_NAME = "F5TTS_v1_Base"


# ── Cross-training run definitions ────────────────────────────────────────────
# Cross-training tests whether a model trained on one data distribution generalises
# better after being continued on a different distribution from the same speaker.
#
# Fields:
#   output_name  : new sub-folder created under models/f5tts_finetune/
#   dataset      : Arrow dataset name (must already exist from finetune_voice.py)
#   start_ckpt   : F5-TTS checkpoint to use as weight initialisation
#   clips        : total clips in the dataset (used only for time estimation)
#   avg_dur      : average clip duration in seconds (used only for time estimation)
#   desc         : human-readable label for console output

RUNS = {
    "Ahmed_1_to_2": {
        "output_name": "Ahmed_1_to_2",
        "dataset":     "Ahmed_2",
        "start_ckpt":  config.MODELS_DIR / "f5tts_finetune" / "ahmed"   / "model_last.pt",
        "clips":       71,
        "avg_dur":     7.1,
        "desc":        "ahmed model  →  continued on Ahmed_2 data (longer sentences)",
    },
    "Ahmed_2_to_1": {
        "output_name": "Ahmed_2_to_1",
        "dataset":     "ahmed",
        "start_ckpt":  config.MODELS_DIR / "f5tts_finetune" / "Ahmed_2" / "model_last.pt",
        "clips":       98,
        "avg_dur":     4.6,
        "desc":        "Ahmed_2 model  →  continued on ahmed data (short sentences)",
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def arrow_dir(dataset: str) -> Path:
    """Return the path where F5-TTS expects the Arrow dataset for 'dataset'."""
    raw = str(pkg_files("f5_tts").joinpath(f"../../data/{dataset}_custom"))
    return Path(raw).resolve()

def ckpt_dir_f5(exp_name: str) -> Path:
    """Return the F5-TTS internal checkpoint directory for an experiment name."""
    raw = str(pkg_files("f5_tts").joinpath(f"../../ckpts/{exp_name}"))
    return Path(raw).resolve()

def out_dir(output_name: str) -> Path:
    """Return the project-local output directory for a cross-training run."""
    return config.MODELS_DIR / "f5tts_finetune" / output_name

def find_vocab_path() -> str:
    """Locate the F5-TTS vocab.txt from the pip-installed package."""
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

def estimate_time(clips: int, avg_dur: float, batch_size: int, epochs: int) -> str:
    """Rough wall-clock estimate (RTX 3080): updates/epoch × ~(3.5 + avg_dur×0.4) s/update."""
    import math
    updates_per_epoch = math.ceil(clips / batch_size)
    s_per_update      = 3.5 + avg_dur * 0.4
    total_minutes     = updates_per_epoch * s_per_update * epochs / 60
    h = int(total_minutes // 60)
    m = int(total_minutes % 60)
    return f"~{h}h {m:02d}min" if h else f"~{m}min"

def copy_checkpoints(exp_name: str, output_name: str) -> None:
    """
    Copy new/updated checkpoints from the F5-TTS internal ckpts/ directory
    to our project's models/f5tts_finetune/<output_name>/ folder.

    F5-TTS always saves to ckpts/F5TTS_v1_Base/ regardless of dataset name, so
    consecutive cross-training runs would overwrite each other's checkpoints if
    we didn't copy them out after each run.
    """
    src = ckpt_dir_f5(exp_name)
    dst = out_dir(output_name)
    if not src.exists():
        print(f"  No checkpoints at {src}")
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
        print(f"  Checkpoints already up to date in {dst}")

def find_latest_ckpt(output_name: str) -> str | None:
    """
    Return the path to the most recently modified checkpoint in the project
    output folder for output_name, or None if the folder is empty.
    Used by --resume to pick up training where it left off.
    """
    d = out_dir(output_name)
    if not d.exists():
        return None
    ckpts = list(d.rglob("*.pt")) + list(d.rglob("*.safetensors"))
    if not ckpts:
        return None
    return str(max(ckpts, key=lambda p: p.stat().st_mtime))


# ── Run one cross-training job ─────────────────────────────────────────────────

def run_cross_train(run: dict, epochs: int, batch_size: int,
                    save_every: int, warmup: int,
                    resume: bool) -> tuple[bool, float]:
    """
    Execute one cross-training job using the F5-TTS finetune CLI.

    Cross-training means: load weights from run['start_ckpt'], continue training
    on run['dataset']'s Arrow data, save results to out_dir(run['output_name']).

    After training, copy_checkpoints() moves the files from Anaconda's internal
    ckpts/ directory into our project folder so the next run does not overwrite them.

    Returns (success: bool, elapsed_seconds: float).
    """
    output_name = run["output_name"]
    dataset     = run["dataset"]
    start_ckpt  = run["start_ckpt"]

    vocab_path  = find_vocab_path()

    # Verify the Arrow dataset exists — it must have been built by finetune_voice.py first
    a_dir = arrow_dir(dataset)
    if not (a_dir / "raw.arrow").exists() and not (a_dir / "raw").is_dir():
        sys.exit(
            f"Arrow dataset not found for '{dataset}': {a_dir}\n"
            f"Run first:  python finetune_voice.py --speaker {dataset}\n"
            f"(This builds the Arrow dataset without needing to retrain.)"
        )

    cli = shutil.which("f5-tts_finetune-cli")
    cli_args = [cli] if cli else [sys.executable, "-m", "f5_tts.train.finetune_cli"]

    # exp_name MUST be one of three hardcoded choices — always use F5TTS_v1_Base.
    # Checkpoints always land in ckpts/F5TTS_v1_Base/; copy_checkpoints() moves them
    # to our named output folder immediately after each run so the next run
    # doesn't overwrite them.
    cmd = cli_args + [
        "--exp_name",                "F5TTS_v1_Base",
        "--dataset_name",            dataset,
        "--learning_rate",           "1e-5",
        "--batch_size_per_gpu",      str(batch_size),
        "--batch_size_type",         "sample",
        "--epochs",                  str(epochs),
        "--num_warmup_updates",      str(warmup),
        "--save_per_updates",        str(save_every),
        "--last_per_updates",        str(save_every),
        "--keep_last_n_checkpoints", "2",
        "--finetune",
        "--tokenizer",               "custom",
        "--tokenizer_path",          vocab_path,
    ]

    # Decide starting checkpoint: resume from output folder OR use original model
    if resume:
        resume_ckpt = find_latest_ckpt(output_name)
        if resume_ckpt:
            print(f"  Resuming from: {resume_ckpt}")
            cmd += ["--pretrain", resume_ckpt]
        else:
            print(f"  No resume checkpoint found — starting from: {start_ckpt}")
            cmd += ["--pretrain", str(start_ckpt)]
    else:
        print(f"  Starting from: {start_ckpt}")
        cmd += ["--pretrain", str(start_ckpt)]

    out_dir(output_name).mkdir(parents=True, exist_ok=True)

    t0     = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0

    print()
    print("Copying checkpoints to project directory …")
    # F5-TTS always saves to ckpts/F5TTS_v1_Base/ — copy from there to our folder
    copy_checkpoints("F5TTS_v1_Base", output_name)

    return result.returncode == 0, elapsed


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-train ahmed <-> Ahmed_2 models on each other's datasets")
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs per run (default 100)")
    parser.add_argument(
        "--batch-size", type=int, default=6,
        help="Clips per step (default 6 — safe for both clip lengths)")
    parser.add_argument(
        "--save-every", type=int, default=50,
        help="Save checkpoint every N updates (default 50)")
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="LR warmup steps (default 10)")
    parser.add_argument(
        "--only", choices=["Ahmed_1_to_2", "Ahmed_2_to_1"], default=None,
        help="Run only one of the two cross-training jobs")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume each run from its latest checkpoint in models/f5tts_finetune/")
    args = parser.parse_args()

    run_order = [args.only] if args.only else ["Ahmed_1_to_2", "Ahmed_2_to_1"]

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    for key in run_order:
        run = RUNS[key]
        if not run["start_ckpt"].exists():
            sys.exit(
                f"Starting checkpoint not found: {run['start_ckpt']}\n"
                f"Run compare_finetune.py first to train the base models."
            )

    # ── Print plan ─────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  CROSS-TRAINING PLAN")
    print("=" * 64)
    print(f"  Epochs per run  : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Save every      : {args.save_every} updates")
    print()

    total_mins = 0
    for key in run_order:
        run = RUNS[key]
        est = estimate_time(run["clips"], run["avg_dur"], args.batch_size, args.epochs)
        print(f"  {key:<16}  {est:<12}  {run['desc']}")

    print()
    print("  Starting checkpoints:")
    for key in run_order:
        run = RUNS[key]
        print(f"  {key:<16}  {run['start_ckpt']}")
    print()
    print("  Output directories:")
    for key in run_order:
        run = RUNS[key]
        print(f"  {key:<16}  {out_dir(run['output_name'])}")
    print()

    # ── Run sequentially ───────────────────────────────────────────────────────
    results = {}
    t_total_start = time.perf_counter()

    for i, key in enumerate(run_order, 1):
        run = RUNS[key]
        print(f"{'─' * 64}")
        print(f"  [{i}/{len(run_order)}]  {key}")
        print(f"  {run['desc']}")
        print(f"{'─' * 64}")
        print()

        success, elapsed = run_cross_train(
            run        = run,
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
        status = "complete" if success else "stopped early"
        print()
        print(f"  {key} — {status}  ({h}h {m:02d}m {s:02d}s)")
        print()

    # ── Final summary ──────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total_start
    th = int(total_elapsed // 3600)
    tm = int((total_elapsed % 3600) // 60)

    print("=" * 64)
    print("  CROSS-TRAINING COMPLETE")
    print("=" * 64)
    print()

    for key in run_order:
        run = RUNS[key]
        res = results.get(key, {})
        el  = res.get("elapsed", 0)
        h, m = int(el // 3600), int((el % 3600) // 60)
        ckpt = out_dir(run["output_name"]) / "model_last.pt"
        ok   = "checkpoint saved" if ckpt.exists() else "no checkpoint"
        print(f"  {key:<16}  {h}h {m:02d}m  {ok}")

    print()
    print(f"  Total wall-clock : {th}h {tm:02d}m")
    print()

    # ── Ready-to-use test commands ─────────────────────────────────────────────
    ref    = config.DATA_DIR / "references" / "ahmed_reference.wav"
    text   = "She sells seashells by the seashore, shells she sells are great"

    print("  Test with the same reference and text used for compare_finetune:")
    print()
    for key in run_order:
        run  = RUNS[key]
        ckpt = out_dir(run["output_name"]) / "model_last.pt"
        name = key.lower()
        print(f"  # {key}  ({run['desc']})")
        print(f"  python clone_voice.py --reference \"{ref}\" \\")
        print(f"      --checkpoint \"{ckpt}\" \\")
        print(f"      --text \"{text}\" \\")
        print(f"      --out-name {name}")
        print()

    print("=" * 64)


if __name__ == "__main__":
    main()

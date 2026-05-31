#!/usr/bin/env python3
# CALLING AHJIN Brought to you by Ahmed Hussein and Julius Norén from JTH.
"""
record_dataset.py
-----------------
Guided voice recording session for building a personal TTS dataset.

- Shows one script at a time
- Press Enter to START recording
- Press Enter again to STOP recording
- Saves the clip, updates progress, shows next script
- Ctrl+C saves progress and exits cleanly — resume anytime
- Skips already-recorded scripts so you never duplicate work
- Writes a manifest compatible with finetune_voice.py

Install dependency (one time):
    pip install sounddevice

Usage:
    python record_dataset.py --speaker ahmed
    python record_dataset.py --speaker ahmed --scripts scripts/voice_scripts.txt
    python record_dataset.py --speaker ahmed --list-devices
    python record_dataset.py --speaker ahmed --device 1
"""

from __future__ import annotations
import os, sys, json, argparse, threading, signal, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError:
    sys.exit("Run:  pip install sounddevice soundfile numpy")

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_SCRIPTS = Path(__file__).resolve().parent / "scripts" / "voice_scripts.txt"
SAMPLE_RATE     = config.F5TTS_SAMPLE_RATE   # 24 kHz — F5-TTS native rate, avoids resampling


# ── Progress file ──────────────────────────────────────────────────────────────

def progress_path(speaker: str) -> Path:
    return config.PROCESSED_DATA_DIR / f"{speaker}_session.json"

def manifest_path(speaker: str) -> Path:
    return config.PROCESSED_DATA_DIR / f"{speaker}_filelist.txt"

def clip_dir(speaker: str) -> Path:
    return config.PROCESSED_DATA_DIR / speaker


def load_progress(speaker: str, scripts: list[str]) -> dict:
    """Load existing session or create a fresh one."""
    p = progress_path(speaker)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        # Merge in any new scripts not seen before
        recorded_texts = {r["text"] for r in data["recordings"].values()}
        for script in scripts:
            if script not in recorded_texts and script not in data["remaining"]:
                data["remaining"].append(script)
        print(f"Resuming session — {data['completed']} / {data['total']} clips done.")
        return data
    else:
        data = {
            "speaker":           speaker,
            "total":             len(scripts),
            "completed":         0,
            "manifest_complete": False,
            "remaining":         list(scripts),
            "recordings":        {},   # clip_id → {text, wav_path, duration, timestamp}
        }
        print(f"New session — {len(scripts)} scripts to record.")
        return data


def save_progress(speaker: str, data: dict) -> None:
    p = progress_path(speaker)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_manifest(speaker: str, wav_path: Path, text: str) -> None:
    """Append one line to the manifest — incremental, never rewrites from scratch."""
    mp = manifest_path(speaker)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "a", encoding="utf-8") as f:
        f.write(f"{wav_path}|{speaker}|{text}\n")


def rebuild_manifest(speaker: str, data: dict) -> None:
    """Rewrite the manifest from scratch using only clips that exist on disk."""
    mp = manifest_path(speaker)
    mp.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(mp, "w", encoding="utf-8") as f:
        for rec in data["recordings"].values():
            if Path(rec["wav_path"]).exists():
                f.write(f"{rec['wav_path']}|{speaker}|{rec['text']}\n")
                written += 1
    print(f"  Manifest rewritten: {written} clips → {mp}")


def repair_session(speaker: str) -> None:
    """
    Scan all recordings in the session JSON.
    Any whose WAV file is missing gets moved back to 'remaining'
    so it will be re-recorded next session.
    Also rebuilds the manifest to remove dead lines.
    """
    p = progress_path(speaker)
    if not p.exists():
        print(f"No session file found for speaker '{speaker}'.")
        return

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    missing_ids   = []
    missing_texts = []

    for clip_id, rec in list(data["recordings"].items()):
        if not Path(rec["wav_path"]).exists():
            missing_ids.append(clip_id)
            missing_texts.append(rec["text"])

    if not missing_ids:
        print(f"No missing files found for speaker '{speaker}'. Everything looks clean.")
        return

    print(f"Found {len(missing_ids)} missing clip(s) — moving back to queue:")
    for clip_id, text in zip(missing_ids, missing_texts):
        print(f"  {clip_id}: \"{text[:60]}\"")
        del data["recordings"][clip_id]
        if text not in data["remaining"]:
            data["remaining"].append(text)

    data["completed"]         = len(data["recordings"])
    data["manifest_complete"] = False   # needs re-recording

    save_progress(speaker, data)
    rebuild_manifest(speaker, data)

    print()
    print(f"Session repaired: {data['completed']} clips kept, "
          f"{len(missing_ids)} returned to queue.")
    print(f"Resume recording: python record_dataset.py --speaker {speaker}")


def mark_manifest_complete(speaker: str, data: dict) -> None:
    data["manifest_complete"] = True
    save_progress(speaker, data)
    mp = manifest_path(speaker)
    print()
    print("=" * 60)
    print("All scripts recorded!")
    print(f"  Manifest : {mp}  ({data['completed']} clips)")
    print()
    print("Next — fine-tune F5-TTS on your voice:")
    print(f"  python finetune_voice.py --speaker {data['speaker']}")
    print("=" * 60)


# ── Recording ──────────────────────────────────────────────────────────────────

def record_clip(sample_rate: int, device: int | None) -> np.ndarray | None:
    """
    Records until the user presses Enter.
    Returns float32 mono array, or None if nothing was captured.
    """
    frames: list[np.ndarray] = []
    recording = threading.Event()
    recording.set()

    def callback(indata, frame_count, time_info, status):
        if recording.is_set():
            frames.append(indata.copy())

    stream = sd.InputStream(samplerate=sample_rate, channels=1,
                             dtype="float32", device=device,
                             callback=callback, blocksize=1024)

    elapsed_ref = [0.0]
    stop_display = threading.Event()

    def show_elapsed():
        start = time.time()
        while not stop_display.is_set():
            elapsed_ref[0] = time.time() - start
            print(f"\r  {elapsed_ref[0]:.1f}s  [recording]", end="", flush=True)
            time.sleep(0.1)

    display_thread = threading.Thread(target=show_elapsed, daemon=True)

    try:
        stream.start()
        display_thread.start()
        input()   # blocks until Enter
    finally:
        recording.clear()
        stop_display.set()
        stream.stop()
        stream.close()
        display_thread.join(timeout=0.5)

    print()  # newline after elapsed counter

    if not frames:
        return None
    return np.concatenate(frames, axis=0).squeeze()


# ── Main session loop ──────────────────────────────────────────────────────────

def run_session(speaker: str, scripts: list[str],
                device: int | None, sample_rate: int) -> None:

    data = load_progress(speaker, scripts)

    if data["manifest_complete"]:
        print("Session already complete! All scripts have been recorded.")
        print(f"Manifest: {manifest_path(speaker)}")
        print("Delete the session file to start over:")
        print(f"  del \"{progress_path(speaker)}\"")
        return

    if not data["remaining"]:
        mark_manifest_complete(speaker, data)
        return

    out_dir = clip_dir(speaker)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Graceful Ctrl+C handler
    interrupted = threading.Event()
    original_sigint = signal.getsignal(signal.SIGINT)

    def handle_interrupt(sig, frame):
        interrupted.set()
        print("\n\nCtrl+C — saving progress...")
        save_progress(speaker, data)
        print(f"  Progress saved: {progress_path(speaker)}")
        print(f"  Clips recorded this session: {data['completed']}")
        print(f"  Remaining: {len(data['remaining'])}")
        print(f"\nResume anytime: python record_dataset.py --speaker {speaker}")
        signal.signal(signal.SIGINT, original_sigint)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    total    = data["total"]
    remaining = list(data["remaining"])   # snapshot so we can iterate safely

    print()
    print(f"Speaker : {speaker}")
    print(f"Device  : {sd.query_devices(device or sd.default.device[0])['name']}")
    print(f"Sample  : {sample_rate} Hz")
    print()
    print("Controls:")
    print("  Enter    start recording")
    print("  Enter    stop recording  →  choose:")
    print("    1      save & next script")
    print("    2      redo the take")
    print("    3      save progress & quit")
    print("  Ctrl+C   save progress & quit  (anytime)")
    print()

    for script in remaining:
        if interrupted.is_set():
            break

        done_count = data["completed"]
        print(f"─── Script {done_count + 1} / {total} " + "─" * 40)
        print()
        print(f"  \"{script}\"")
        print()
        print("  Press Enter to start recording...")

        try:
            input()
        except EOFError:
            break

        if interrupted.is_set():
            break

        # ── Record loop (redo until satisfied) ────────────────────────────
        while True:
            print("  ● RECORDING — read the script above — press Enter to stop")
            print()
            print(f"  \"{script}\"")
            print()

            try:
                audio = record_clip(sample_rate, device)
            except KeyboardInterrupt:
                handle_interrupt(None, None)

            if audio is None or len(audio) < sample_rate * 0.5:
                print("  (too short — try again)")
                print()
                continue

            duration = len(audio) / sample_rate
            print(f"  Recorded: {duration:.1f}s")
            print()
            print("  1 → save & next   2 → redo   3 → save & quit")

            while True:
                try:
                    choice = input("  > ").strip()
                except EOFError:
                    choice = "1"

                if choice == "1" or choice == "":
                    break        # accept
                elif choice == "2":
                    print()
                    print("  Re-recording...")
                    print()
                    break        # redo — inner while breaks, outer while loops again
                elif choice == "3":
                    handle_interrupt(None, None)
                else:
                    print("  Please enter 1, 2, or 3.")
                    continue

            if choice == "2":
                continue         # outer while: record again
            else:
                break            # outer while: done, save

        # Save WAV
        clip_id  = f"{speaker}_{done_count + 1:05d}"
        wav_path = out_dir / f"{clip_id}.wav"
        sf.write(str(wav_path), audio, sample_rate, subtype="PCM_16")

        # Update progress
        data["recordings"][clip_id] = {
            "text":      script,
            "wav_path":  str(wav_path),
            "duration":  round(duration, 2),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        data["remaining"].remove(script)
        data["completed"] += 1
        save_progress(speaker, data)
        append_manifest(speaker, wav_path, script)

        print(f"  ✓ Saved ({duration:.1f}s)  [{data['completed']}/{total} done]")
        print()

        # Check if all done
        if not data["remaining"]:
            mark_manifest_complete(speaker, data)
            break

    if data["remaining"] and not interrupted.is_set():
        print(f"\nSession paused. {data['completed']}/{total} clips recorded.")
        print(f"Resume: python record_dataset.py --speaker {speaker}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guided voice recording session for TTS dataset creation")
    parser.add_argument(
        "--speaker", required=True,
        help="Your speaker name/ID, e.g. ahmed")
    parser.add_argument(
        "--scripts", default=str(DEFAULT_SCRIPTS),
        help=f"Text file with one sentence per line (default: {DEFAULT_SCRIPTS})")
    parser.add_argument(
        "--device", type=int, default=None,
        help="Microphone device index (run --list-devices to see options)")
    parser.add_argument(
        "--sample-rate", type=int, default=SAMPLE_RATE,
        help=f"Recording sample rate in Hz (default {SAMPLE_RATE})")
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available microphone devices and exit")
    parser.add_argument(
        "--repair", action="store_true",
        help="Scan for deleted WAV files and return those scripts to the recording queue")
    args = parser.parse_args()

    if args.repair:
        repair_session(args.speaker)
        sys.exit(0)

    if args.list_devices:
        print("\nAvailable input devices:")
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                marker = " <-- default" if i == sd.default.device[0] else ""
                print(f"  [{i}] {dev['name']}{marker}")
        print()
        sys.exit(0)

    # Load scripts
    scripts_file = Path(args.scripts)
    if not scripts_file.exists():
        sys.exit(f"Scripts file not found: {scripts_file}\n"
                 f"Default scripts are at: {DEFAULT_SCRIPTS}")

    scripts = [ln.strip() for ln in scripts_file.read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.startswith("#")]

    if not scripts:
        sys.exit("Scripts file is empty.")

    # Mic sanity check
    try:
        sd.check_input_settings(device=args.device, channels=1,
                                dtype="float32", samplerate=args.sample_rate)
    except Exception as e:
        print(f"Microphone error: {e}")
        print("Run --list-devices to pick a different one.")
        sys.exit(1)

    run_session(args.speaker, scripts, args.device, args.sample_rate)


if __name__ == "__main__":
    main()

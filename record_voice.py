#!/usr/bin/env python3
"""
record_voice.py
---------------
Records from your microphone and saves a WAV file.
Optionally clones your voice immediately after recording.

Install dependency (one time):
    pip install sounddevice

Usage:
    # Record until you press Enter, then clone:
    python record_voice.py --text "Hello, this is my cloned voice."

    # Set a max duration safety cap (default 120s):
    python record_voice.py --max-duration 30 --text "Hello."

    # List microphones and pick one:
    python record_voice.py --list-devices
    python record_voice.py --device 1 --text "Hello."

    # Just record (no cloning):
    python record_voice.py

    # Clone from an existing file:
    python record_voice.py --reference existing.wav --text "Hello."
"""

from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys, argparse, subprocess, threading, time
from pathlib import Path


def next_free_path(path: Path) -> Path:
    """Return path unchanged if it doesn't exist, otherwise add _001, _002, …"""
    if not path.exists():
        return path
    n = 1
    while True:
        candidate = path.with_stem(f"{path.stem}_{n:03d}")
        if not candidate.exists():
            return candidate
        n += 1

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError:
    sys.exit("Run:  pip install sounddevice soundfile numpy")


# ── Recording ──────────────────────────────────────────────────────────────────

def list_devices() -> None:
    print("\nAvailable input devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = " <-- default" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{marker}")
    print()


def record(max_duration: int, sample_rate: int, device: int | None) -> np.ndarray:
    """
    Record until the user presses Enter OR max_duration seconds elapse.
    Ctrl+C also stops cleanly and saves whatever was captured.
    Returns float32 mono array.
    """
    frames: list[np.ndarray] = []
    stop_event = threading.Event()

    def audio_callback(indata, frame_count, time_info, status):
        if not stop_event.is_set():
            frames.append(indata.copy())

    print(f"\nRecording... (max {max_duration}s)")
    print("Speak now. Press Enter to stop.\n")

    stream = sd.InputStream(
        samplerate=sample_rate, channels=1,
        dtype="float32", device=device,
        callback=audio_callback,
        blocksize=1024,
    )

    # Thread that waits for Enter key
    enter_pressed = threading.Event()
    def wait_for_enter():
        try:
            input()   # blocks until Enter
        except Exception:
            pass
        enter_pressed.set()

    enter_thread = threading.Thread(target=wait_for_enter, daemon=True)

    try:
        stream.start()
        enter_thread.start()
        start = time.time()

        while not enter_pressed.is_set() and not stop_event.is_set():
            elapsed = time.time() - start
            if elapsed >= max_duration:
                print(f"\r  {elapsed:5.1f}s  (max duration reached)         ", flush=True)
                break
            print(f"\r  {elapsed:5.1f}s  |  Press Enter to stop", end="", flush=True)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n  (Ctrl+C — saving what was recorded...)")
    finally:
        stop_event.set()
        stream.stop()
        stream.close()

    elapsed = time.time() - start
    print(f"\nStopped at {elapsed:.1f}s")

    if not frames:
        sys.exit("No audio captured. Check your microphone.")

    audio = np.concatenate(frames, axis=0).squeeze()
    return audio


def save_wav(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    duration = len(audio) / sample_rate
    print(f"Saved  : {path}  ({duration:.1f}s)")


# ── Clone ──────────────────────────────────────────────────────────────────────

def clone(reference_path: Path, text: str) -> None:
    script = Path(__file__).resolve().parent / "clone_voice.py"
    cmd = [sys.executable, str(script),
           "--reference", str(reference_path),
           "--text", text]
    print(f"\nRunning clone_voice.py...\n")
    subprocess.run(cmd)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record your voice and optionally clone it with F5-TTS")

    parser.add_argument(
        "--max-duration", type=int, default=120,
        help="Maximum recording length in seconds (default 120). "
             "Recording stops sooner when you press Enter.")
    parser.add_argument(
        "--out", default=None,
        help="Output WAV path (default: output/recording.wav)")
    parser.add_argument(
        "--text", default=None,
        help="Text to synthesise in your voice right after recording. "
             "Omit to just save the WAV.")
    parser.add_argument(
        "--reference", default=None,
        help="Skip recording — use this existing file as the reference instead.")
    parser.add_argument(
        "--device", type=int, default=None,
        help="Microphone device index (run --list-devices to see options)")
    parser.add_argument(
        "--sample-rate", type=int, default=22050,
        help="Sample rate in Hz (default 22050)")
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available microphone devices and exit")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    out_path = Path(args.out) if args.out else next_free_path(config.OUTPUT_DIR / "recording.wav")

    # ── Use existing file or record fresh ─────────────────────────────────────
    if args.reference:
        reference_path = Path(args.reference)
        if not reference_path.exists():
            sys.exit(f"Reference file not found: {reference_path}")
        print(f"Using existing reference: {reference_path}")
    else:
        # Quick mic sanity check
        try:
            sd.check_input_settings(device=args.device, channels=1,
                                    dtype="float32", samplerate=args.sample_rate)
        except Exception as e:
            print(f"Microphone error: {e}")
            list_devices()
            print("Use --device <index> to pick a specific microphone.")
            sys.exit(1)

        audio = record(args.max_duration, args.sample_rate, args.device)
        save_wav(audio, out_path, args.sample_rate)
        reference_path = out_path

    # ── Clone if text was given ────────────────────────────────────────────────
    if args.text:
        clone(reference_path, args.text)
        print(f"\nDone.")
        print(f"  Cloned audio   : {config.OUTPUT_DIR / 'cloned_voice.wav'}")
        print(f"  Your recording : {reference_path}")
    else:
        print(f"\nRecording saved. To clone your voice:")
        print(f'  python clone_voice.py --reference "{reference_path}" '
              f'--text "Your text here."')


if __name__ == "__main__":
    main()

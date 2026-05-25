#!/usr/bin/env python3
# CALLING AHJIN — Brought to you by Ahmed Hussein and Julius Norén from JTH.
"""
create_single_audio.py
----------------------
Record a single continuous reference audio clip and save it to
data/references/<name>.wav for use as a voice cloning reference.

At startup you choose your script interactively:
    1. Use the built-in phonetically rich default script
    2. Type / paste your own script right in the terminal
    3. Load from a .txt file

Usage:
    python create_single_audio.py --name ahmed_reference
    python create_single_audio.py --name ahmed_reference --list-devices
    python create_single_audio.py --name ahmed_reference --device 1
"""

from __future__ import annotations
import os, sys, argparse, threading, time
# KMP_DUPLICATE_LIB_OK: prevents fatal OMP #15 abort on Windows when both
# PyTorch (Intel MKL) and numpy each load their own OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError:
    sys.exit("Run:  pip install sounddevice soundfile numpy")

# Record at 24 kHz — F5-TTS's native sample rate. Recording at this rate avoids
# any resampling artefacts when the clip is used as a reference audio.
SAMPLE_RATE    = config.F5TTS_SAMPLE_RATE
REFERENCES_DIR = config.DATA_DIR / "references"

# ── Built-in default script ────────────────────────────────────────────────────
# Phonetically rich, ~25 seconds at natural pace.
# Deliberately covers all English vowels, common consonant clusters, varied
# sentence rhythm, and pitch contours — giving F5-TTS a good spread of
# phoneme representations to clone from.
DEFAULT_SCRIPT = """\
The thing about mornings is they never wait for you.
Years of rushing out half-awake taught me that.
But lately I've tried something different —
waking before the noise starts,
stepping outside into the cool, quiet air.
There's something almost sacred about that early hour.
Nobody needs anything from you yet.
The birds sing simply because that's what birds do.
And just for a moment, everything feels exactly right.\
"""


# ── Script chooser ─────────────────────────────────────────────────────────────

def choose_script() -> str:
    """
    Present an interactive menu so the user can choose what to read.

    Options:
      1. Built-in phonetically rich default script (recommended for best cloning).
      2. Type / paste a custom script in the terminal.
      3. Load text from a .txt file (lines starting with # are treated as comments).

    Returns the final script text as a single string.
    """
    print()
    print("=" * 60)
    print("  CHOOSE YOUR SCRIPT")
    print("=" * 60)
    print()
    print("  1  Use the built-in phonetically rich script (recommended)")
    print("  2  Type / paste your own script now")
    print("  3  Load from a .txt file")
    print()

    while True:
        try:
            choice = input("  > ").strip()
        except EOFError:
            choice = "1"

        if choice in ("1", ""):
            print()
            return DEFAULT_SCRIPT

        elif choice == "2":
            print()
            print("  Type your script below.")
            print("  Press Enter after each line.")
            print("  When finished, type  DONE  on its own line and press Enter.")
            print()
            lines = []
            while True:
                try:
                    line = input("  | ")
                except EOFError:
                    break
                if line.strip().upper() == "DONE":
                    break
                lines.append(line)
            text = "\n".join(lines).strip()
            if not text:
                print("  (nothing entered — using default script)")
                return DEFAULT_SCRIPT
            print()
            return text

        elif choice == "3":
            print()
            try:
                path_str = input("  File path: ").strip().strip('"').strip("'")
            except EOFError:
                return DEFAULT_SCRIPT
            p = Path(path_str)
            if not p.exists():
                print(f"  File not found: {p}  — using default script.")
                print()
                return DEFAULT_SCRIPT
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.startswith("#")]
            text = "\n".join(lines)
            if not text:
                print("  File is empty — using default script.")
                return DEFAULT_SCRIPT
            print()
            return text

        else:
            print("  Please enter 1, 2, or 3.")


# ── Recording ──────────────────────────────────────────────────────────────────

def record_clip(sample_rate: int, device: int | None) -> np.ndarray | None:
    """
    Open the microphone and record until the user presses Enter.

    Uses a sounddevice InputStream with a callback that appends each audio
    block to a list. A background thread prints an elapsed-time counter so
    the user knows recording is active. Recording stops cleanly whether the
    user presses Enter or Ctrl+C interrupts the input() call.

    Returns a float32 mono numpy array, or None if no audio was captured.
    """
    frames: list[np.ndarray] = []
    recording = threading.Event()
    recording.set()

    def callback(indata, frame_count, time_info, status):
        # sounddevice calls this on a high-priority audio thread — only append, no I/O
        if recording.is_set():
            frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32",
        device=device, callback=callback, blocksize=1024,
    )

    stop_display = threading.Event()

    def show_elapsed():
        """Background thread: prints elapsed recording time every 100 ms."""
        start = time.time()
        while not stop_display.is_set():
            elapsed = time.time() - start
            print(f"\r  {elapsed:.1f}s  [recording]", end="", flush=True)
            time.sleep(0.1)

    display_thread = threading.Thread(target=show_elapsed, daemon=True)

    try:
        stream.start()
        display_thread.start()
        input()   # block until user presses Enter
    finally:
        # Stop recording and the display thread regardless of how we exited
        recording.clear()
        stop_display.set()
        stream.stop()
        stream.close()
        display_thread.join(timeout=0.5)

    print()
    if not frames:
        return None
    # Concatenate all captured blocks into a single 1-D mono array
    return np.concatenate(frames, axis=0).squeeze()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point: parse CLI args, validate the microphone, and run the record loop.

    The record loop allows multiple takes; the user can redo a take if they
    are not happy with the result. Saves to data/references/<name>.wav at
    F5TTS_SAMPLE_RATE (24 kHz) in 16-bit PCM format.
    """
    parser = argparse.ArgumentParser(
        description="Record a single reference audio clip for voice cloning")
    parser.add_argument(
        "--name", required=True,
        help="Output filename stem, e.g. ahmed_reference  →  data/references/ahmed_reference.wav")
    parser.add_argument(
        "--device", type=int, default=None,
        help="Microphone device index (use --list-devices to see options)")
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print all available input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable input devices:")
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                marker = " <-- default" if i == sd.default.device[0] else ""
                print(f"  [{i}] {dev['name']}{marker}")
        print()
        sys.exit(0)

    # Verify the chosen (or default) mic supports our target settings before recording
    try:
        sd.check_input_settings(device=args.device, channels=1,
                                dtype="float32", samplerate=SAMPLE_RATE)
    except Exception as e:
        print(f"Microphone error: {e}")
        print("Run --list-devices to pick a different one.")
        sys.exit(1)

    out_dir  = REFERENCES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.wav"

    if out_path.exists():
        print(f"\nNote: {out_path} already exists — a successful recording will overwrite it.")

    # ── Choose script ──────────────────────────────────────────────────────────
    script_text = choose_script()

    # ── Display header ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  REFERENCE AUDIO RECORDER")
    print("=" * 60)
    print(f"  Output  : {out_path}")
    print(f"  Device  : {sd.query_devices(args.device or sd.default.device[0])['name']}")
    print(f"  Rate    : {SAMPLE_RATE} Hz")
    print()
    print("  YOUR SCRIPT (read at a natural, unhurried pace):")
    print()
    for line in script_text.splitlines():
        print(f"    {line}")
    print()
    print("  Target: 20–35 seconds")
    print("=" * 60)
    print()
    print("  Controls:")
    print("    Enter   start recording")
    print("    Enter   stop recording  →  1 save  /  2 redo")
    print("    Ctrl+C  quit anytime")
    print()

    # ── Record loop ────────────────────────────────────────────────────────────
    take = 1
    while True:
        if take > 1:
            print("─" * 60)
            print()
            print("  YOUR SCRIPT:")
            print()
            for line in script_text.splitlines():
                print(f"    {line}")
            print()

        print(f"  Take {take} — Press Enter to START recording...")
        try:
            input()
        except EOFError:
            sys.exit(0)

        print("  ● RECORDING — read your script — press Enter to STOP")
        print()

        try:
            audio = record_clip(SAMPLE_RATE, args.device)
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

        if audio is None or len(audio) < SAMPLE_RATE * 2:
            print("  (too short — minimum 2 seconds, try again)")
            print()
            take += 1
            continue

        duration = len(audio) / SAMPLE_RATE
        print(f"  Recorded: {duration:.1f}s")
        print()

        if duration < 10:
            # Clips under 10 s often lack enough phonetic variety for good cloning
            print("  ⚠  Under 10 seconds — a longer clip gives better cloning quality.")
            print()
        elif duration > 35:
            # F5-TTS truncates references to 12 s internally; extra audio is ignored
            print("  ⚠  Over 35 seconds — F5-TTS clips references to 12s internally.")
            print("      Consider reading only the first half of the script.")
            print()

        print("  1 → save    2 → redo    (Enter = save)")

        while True:
            try:
                choice = input("  > ").strip()
            except EOFError:
                choice = "1"

            if choice in ("1", ""):
                break
            elif choice == "2":
                print()
                print("  Re-recording...")
                print()
                break
            else:
                print("  Please enter 1 or 2.")

        if choice == "2":
            take += 1
            continue

        # ── Save ──────────────────────────────────────────────────────────────
        sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
        print()
        print("=" * 60)
        print(f"  Saved  : {out_path}")
        print(f"  Length : {duration:.1f}s  @  {SAMPLE_RATE} Hz")
        if take > 1:
            print(f"  Takes  : {take}")
        print()
        print("  Use as reference for voice cloning:")
        print(f'    python clone_voice.py --reference "{out_path}" ^')
        print(f'        --text "Your text here."')
        print("=" * 60)
        print()
        break


if __name__ == "__main__":
    main()

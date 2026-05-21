#!/usr/bin/env python3
"""
prepare_voice.py
----------------
Takes one long recording of your voice, transcribes + aligns with
faster-whisper (CUDA-accelerated, no FFmpeg needed for WAV input),
splits into training clips, and writes a manifest ready for fine-tuning.

Install:
    pip install faster-whisper

Usage:
    python prepare_voice.py --audio your_recording.wav
    python prepare_voice.py --audio your_recording.wav --speaker ahmed --model medium

Output:
    data/processed/your_voice/   individual wav clips
    data/processed/your_voice_filelist.txt   manifest for fine-tuning

Tips for recording:
  - Quiet room, no echo
  - Same mic distance throughout
  - Read varied sentences (news, Wikipedia, short stories)
  - Aim for 30-45 minutes total
  - Save as WAV (most recording apps support this)
"""

from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys, argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

try:
    from faster_whisper import WhisperModel
except ImportError:
    sys.exit("Run:  pip install faster-whisper")

# ── Clip duration targets ──────────────────────────────────────────────────────
MIN_CLIP_SEC = 3.0
MAX_CLIP_SEC = 15.0   # shorter than download limit — cleaner for fine-tuning
PAD_SEC      = 0.05   # silence padding added to each side of a clip

WHISPER_SR   = 16_000  # Whisper always expects 16 kHz


def load_audio(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (audio_22k, audio_16k) — full quality for saving, 16 kHz for Whisper.
    Uses soundfile so no FFmpeg is needed for WAV/FLAC input.
    """
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)   # stereo → mono

    audio_22k = librosa.resample(wav, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    audio_16k  = librosa.resample(wav, orig_sr=sr, target_sr=WHISPER_SR)
    return audio_22k, audio_16k


def transcribe(audio_16k: np.ndarray, model_size: str) -> list[dict]:
    """
    Run faster-whisper on the 16 kHz array.
    Returns a flat list of segment dicts: {start, end, text}.
    Uses CUDA if available, falls back to CPU.
    """
    import torch
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    compute     = "float16" if device == "cuda" else "int8"

    print(f"Loading Whisper '{model_size}' on {device.upper()} …")
    model = WhisperModel(model_size, device=device, compute_type=compute)

    print("Transcribing … (this takes 1–5 minutes for a 30-min recording)")
    segments_iter, info = model.transcribe(
        audio_16k,
        beam_size=5,
        word_timestamps=False,   # segment-level is enough for splitting
        vad_filter=True,         # skip long silences automatically
        language="en",
    )

    segments = []
    for seg in tqdm(segments_iter, desc="transcribing", unit="seg"):
        text = seg.text.strip()
        if text:
            segments.append({"start": seg.start, "end": seg.end, "text": text})

    print(f"  {len(segments)} segments found  "
          f"({info.duration / 60:.1f} min audio)")
    return segments


def group_segments(segments: list[dict]) -> list[list[dict]]:
    """
    Greedily group consecutive segments into clips of MIN_CLIP_SEC–MAX_CLIP_SEC.
    Cuts at segment boundaries (natural sentence breaks) only.
    """
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_dur = 0.0

    for seg in segments:
        seg_dur = seg["end"] - seg["start"]

        # If adding this segment would exceed the max, save current group first
        if current and (current_dur + seg_dur > MAX_CLIP_SEC):
            if current_dur >= MIN_CLIP_SEC:
                groups.append(current)
            current = []
            current_dur = 0.0

        current.append(seg)
        current_dur += seg_dur

        # If we've accumulated a comfortable clip, save and start fresh
        if current_dur >= MIN_CLIP_SEC * 2:
            groups.append(current)
            current = []
            current_dur = 0.0

    # Tail
    if current and current_dur >= MIN_CLIP_SEC:
        groups.append(current)

    return groups


def save_clips(groups: list[list[dict]], audio_22k: np.ndarray,
               out_dir: Path, speaker_id: str) -> list[str]:
    """
    Extract each clip group from the audio, save as WAV, return manifest lines.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    sr = config.SAMPLE_RATE

    for i, group in enumerate(tqdm(groups, desc="saving clips", unit="clip")):
        start_s = max(0.0, group[0]["start"] - PAD_SEC)
        end_s   = min(len(audio_22k) / sr, group[-1]["end"] + PAD_SEC)
        text    = " ".join(seg["text"] for seg in group).strip()

        start_i = int(start_s * sr)
        end_i   = int(end_s   * sr)
        clip    = audio_22k[start_i:end_i]

        clip_id  = f"{speaker_id}_{i:05d}"
        wav_path = out_dir / f"{clip_id}.wav"
        sf.write(str(wav_path), clip, sr, subtype="PCM_16")
        lines.append(f"{wav_path}|{speaker_id}|{text}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare voice recording for TTS fine-tuning")
    parser.add_argument("--audio",   required=True,
                        help="Path to your recording (WAV recommended)")
    parser.add_argument("--speaker", default="my_voice",
                        help="Speaker ID used in the manifest (default: my_voice)")
    parser.add_argument("--model",   default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v2"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--out",     default=None,
                        help="Output folder (default: data/processed/<speaker>)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        sys.exit(f"Audio file not found: {audio_path}")

    out_dir  = Path(args.out) if args.out else config.PROCESSED_DATA_DIR / args.speaker
    manifest = config.PROCESSED_DATA_DIR / f"{args.speaker}_filelist.txt"

    print(f"\nInput  : {audio_path}")
    print(f"Speaker: {args.speaker}")
    print(f"Output : {out_dir}\n")

    # 1. Load
    print("Loading audio …")
    audio_22k, audio_16k = load_audio(audio_path)
    duration_min = len(audio_22k) / config.SAMPLE_RATE / 60
    print(f"  Duration: {duration_min:.1f} minutes\n")

    # 2. Transcribe
    segments = transcribe(audio_16k, args.model)
    if not segments:
        sys.exit("No speech detected. Check your recording.")

    # 3. Group into clips
    groups = group_segments(segments)
    total_dur = sum(g[-1]["end"] - g[0]["start"] for g in groups)
    print(f"\n{len(groups)} clips  ({total_dur/60:.1f} min usable speech)\n")

    # 4. Save
    lines = save_clips(groups, audio_22k, out_dir, args.speaker)

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDone.")
    print(f"  Clips   : {out_dir}")
    print(f"  Manifest: {manifest}")
    print(f"\nNext — fine-tune on your voice:")
    print(f"  python finetune.py --manifest {manifest}")


if __name__ == "__main__":
    main()

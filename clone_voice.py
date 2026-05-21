#!/usr/bin/env python3
"""
clone_voice.py
--------------
Zero-shot voice cloning using F5-TTS.

No training required — F5-TTS uses its pretrained model to clone any voice
from a short reference clip (5–30 seconds is ideal).

The pretrained model (~1.2 GB) is downloaded automatically from HuggingFace
the first time you run this script.

Usage:
    python clone_voice.py --reference path/to/ref.wav --text "Hello, this is a cloned voice."

    # With an explicit transcript of the reference (faster, no auto-transcription):
    python clone_voice.py --reference ref.wav --ref-text "This is what the reference says." \
                          --text "Hello world."

    # Use a fine-tuned checkpoint (after running finetune_voice.py):
    python clone_voice.py --reference ref.wav --text "Hello." \
                          --checkpoint models/f5tts_finetune/my_model.pt

    # Control speed / output path:
    python clone_voice.py --reference ref.wav --text "Hello." \
                          --speed 0.9 --out output/my_clone.wav
"""

from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import sys, argparse
from pathlib import Path


def next_free_path(path: Path) -> Path:
    """
    If path doesn't exist, return it as-is.
    Otherwise return path with an incrementing number, e.g.:
        cloned_voice.wav → cloned_voice_001.wav → cloned_voice_002.wav
    """
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


def ensure_wav(audio_path: Path) -> Path:
    """
    F5-TTS uses pydub internally which needs FFmpeg for non-WAV formats.
    We bypass that by converting to WAV ourselves using installed backends.
    Returns the original path unchanged if it is already a WAV.
    Converted files are cached in output/ so the conversion only runs once.
    """
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    out_wav = config.OUTPUT_DIR / f"{audio_path.stem}_ref.wav"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if out_wav.exists():
        print(f"Using cached WAV: {out_wav}")
        return out_wav

    print(f"Converting {audio_path.name} → WAV (FFmpeg-free)...")

    wav, sr = None, None

    # 1. soundfile — handles WAV/FLAC/OGG; newer libsndfile also reads MP3
    if wav is None:
        try:
            import soundfile as sf
            wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        except Exception:
            pass

    # 2. librosa — uses audioread which on Windows tries Media Foundation
    if wav is None:
        try:
            import librosa
            wav, sr = librosa.load(str(audio_path), sr=None, mono=True)
        except Exception:
            pass

    # 3. torchaudio — bundled codec support for common formats
    if wav is None:
        try:
            import torch, torchaudio
            waveform, sr = torchaudio.load(str(audio_path))
            import numpy as np
            wav = waveform.mean(0).numpy()   # stereo → mono
        except Exception:
            pass

    # 4. PyAV (av package, ships its own FFmpeg DLLs — installed via faster-whisper)
    if wav is None:
        try:
            import av, numpy as np
            container = av.open(str(audio_path))
            stream = next(s for s in container.streams if s.type == "audio")
            frames = [f.to_ndarray() for f in container.decode(stream)]
            wav = np.concatenate(frames, axis=-1).mean(axis=0).astype("float32")
            wav /= max(abs(wav).max(), 1e-6)   # normalise to [-1, 1]
            sr  = stream.codec_context.sample_rate
        except Exception:
            pass

    if wav is None:
        sys.exit(
            f"Could not read {audio_path}.\n"
            "Convert it to WAV manually:  ffmpeg -i input.mp3 output.wav\n"
            "Or install FFmpeg:  conda install -c conda-forge ffmpeg"
        )

    import soundfile as sf
    import numpy as np
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    sf.write(str(out_wav), wav.astype("float32"), sr, subtype="PCM_16")
    print(f"  Converted: {out_wav}")
    return out_wav


def auto_transcribe(audio_path: Path) -> str:
    """
    Transcribe reference audio using faster-whisper.
    No FFmpeg needed — reads WAV directly with soundfile.
    """
    import torch
    import numpy as np
    import soundfile as sf
    import librosa

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Run:  pip install faster-whisper")

    print("Auto-transcribing reference audio...")
    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model   = WhisperModel("base", device=device, compute_type=compute)

    segments, _ = model.transcribe(wav, beam_size=5, language="en",
                                   vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"  Transcript : \"{text}\"")
    return text


def trim_and_fade(wav: "np.ndarray", sample_rate: int,
                  silence_db: float = -42.0,
                  fade_ms: float = 60.0) -> "np.ndarray":
    """
    Remove trailing silence then apply a short fade-out.
    Prevents the vocoder's hard boundary cut from sounding like a word being clipped.
    """
    import numpy as np
    threshold = 10 ** (silence_db / 20)
    abs_wav = np.abs(wav)
    # Walk backwards to find last frame above the silence threshold
    end_idx = len(wav)
    for i in range(len(wav) - 1, -1, -1):
        if abs_wav[i] > threshold:
            end_idx = i + 1
            break
    # Keep a small natural tail after last voiced sample, then fade
    tail_samples = int(0.08 * sample_rate)   # 80 ms natural decay
    end_idx = min(end_idx + tail_samples, len(wav))
    wav = wav[:end_idx].copy()
    # Fade-out over the last fade_ms milliseconds
    fade_samples = min(int(fade_ms / 1000 * sample_rate), len(wav))
    wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    return wav


def split_sentences(text: str) -> list[str]:
    """
    Split text into individual sentences at . ! ? boundaries.
    Each sentence stays within the model's comfortable generation range (~3–6 s).
    """
    import re
    # Split after . ! ? followed by whitespace or end-of-string
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Drop empty parts, keep punctuation attached to each sentence
    return [p.strip() for p in parts if p.strip()]


def infer(reference_path: Path, ref_text: str, gen_text: str,
          out_path: Path, checkpoint: str | None,
          speed: float, seed: int, nfe_steps: int) -> None:
    """
    Run F5-TTS zero-shot cloning.

    reference_path : WAV/FLAC recording of the voice to clone (5–30 s)
    ref_text       : transcript of the reference clip (empty string → auto)
    gen_text       : text to synthesise in the cloned voice
    out_path       : where to write the synthesised WAV
    checkpoint     : path to a fine-tuned .pt checkpoint (None = pretrained)
    speed          : playback speed multiplier (1.0 = normal)
    seed           : RNG seed for reproducibility (-1 = random)
    nfe_steps      : diffusion steps — more = slower but slightly better (16–32)
    """
    try:
        import numpy as np
        import soundfile as sf
        from f5_tts.api import F5TTS
    except ImportError:
        sys.exit("F5-TTS not installed. Run:  .\\setup_f5tts.ps1")

    # Convert MP3/M4A/etc. to WAV — F5-TTS's pydub needs FFmpeg for non-WAV
    reference_path = ensure_wav(reference_path)

    # Transcribe reference ourselves (no FFmpeg needed) if text not supplied
    if not ref_text:
        ref_text = auto_transcribe(reference_path)

    ckpt_file = checkpoint if checkpoint else ""

    print(f"Loading F5-TTS model {'(fine-tuned)' if ckpt_file else '(pretrained)'}…")
    tts = F5TTS(ckpt_file=ckpt_file)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reference  : {reference_path}")
    print(f"Ref text   : {ref_text[:80]}{'…' if len(ref_text) > 80 else ''}")
    print(f'Generating : "{gen_text[:80]}{"…" if len(gen_text) > 80 else ""}"')
    print()

    # ── Sentence-by-sentence generation ──────────────────────────────────────
    # Fine-tuned models trained on short clips (2–5 s) lose coherence on long
    # sequences. Splitting into individual sentences keeps each inference call
    # within the model's comfortable range, then we stitch the pieces together.
    sentences = split_sentences(gen_text)

    if len(sentences) <= 1:
        # Single sentence — generate directly
        wav, sr, _ = tts.infer(
            ref_file=str(reference_path),
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=str(out_path),
            speed=speed,
            seed=seed if seed != -1 else None,
            nfe_step=nfe_steps,
        )
    else:
        print(f"  Splitting into {len(sentences)} sentence(s) for clean generation…")
        waves = []
        sample_rate = None
        silence_gap = 0.18   # seconds of silence between sentences

        for i, sentence in enumerate(sentences):
            print(f"  [{i+1}/{len(sentences)}] \"{sentence}\"")
            wav_i, sr_i, _ = tts.infer(
                ref_file=str(reference_path),
                ref_text=ref_text,
                gen_text=sentence,
                file_wave=None,          # don't write intermediate files
                speed=speed,
                seed=seed if seed != -1 else None,
                nfe_step=nfe_steps,
            )
            wav_i = trim_and_fade(wav_i, sr_i)
            waves.append(wav_i)
            sample_rate = sr_i

        # Add short silence gap between sentences, then concatenate
        gap = np.zeros(int(silence_gap * sample_rate), dtype=np.float32)
        stitched = []
        for i, w in enumerate(waves):
            stitched.append(w)
            if i < len(waves) - 1:
                stitched.append(gap)

        wav = np.concatenate(stitched)
        sr  = sample_rate
        sf.write(str(out_path), wav, sr)

    duration_s = len(wav) / sr
    print(f"Saved  : {out_path}  ({duration_s:.1f}s  @ {sr} Hz)")


def resolve_output(out_arg: str | None, out_name_arg: str | None,
                   reference_path: Path, default_ref: str) -> Path:
    """
    Decide the output path with this priority:
      1. --out PATH      explicit full path, used as-is (still auto-numbered)
      2. --out-name NAME saves to output/<NAME>.wav   (auto-numbered)
      3. custom --reference provided → output/<reference_stem>_clone.wav
      4. default reference → output/cloned_voice.wav
    All options are auto-numbered (_001, _002, …) if the file already exists.
    """
    out_dir = config.OUTPUT_DIR

    if out_arg:
        base = Path(out_arg)
    elif out_name_arg:
        base = out_dir / f"{out_name_arg}.wav"
    elif str(reference_path) != default_ref:
        # Custom reference supplied — name after it so outputs don't collide
        base = out_dir / f"{reference_path.stem}_clone.wav"
    else:
        base = out_dir / "cloned_voice.wav"

    return next_free_path(base)


def main() -> None:
    default_ref = str(config.CLONE_REFERENCE_AUDIO) if config.CLONE_REFERENCE_AUDIO else ""

    parser = argparse.ArgumentParser(
        description="Zero-shot voice cloning with F5-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use your default reference (set in config.py):
  python clone_voice.py --text "Hello world."

  # Clone a specific person's voice, auto-named output:
  python clone_voice.py --reference voices/john.wav --ref-text "What John said." --text "Hello."
  # → output/john_clone.wav

  # Clone and give the output a custom name:
  python clone_voice.py --reference voices/john.wav --ref-text "What John said." \\
                        --text "Hello." --out-name john_greeting
  # → output/john_greeting.wav
""")

    parser.add_argument(
        "--reference", default=default_ref, required=not bool(default_ref),
        help="Reference audio clip of the voice to clone (5-30 s, WAV/FLAC/MP3). "
             "Default: CLONE_REFERENCE_AUDIO from config.py")
    parser.add_argument(
        "--ref-text", default="",
        help="Transcript of the reference clip. If omitted, Whisper auto-transcribes.")
    parser.add_argument(
        "--text", required=True,
        help="Text to synthesise in the cloned voice")
    parser.add_argument(
        "--out-name", default=None, metavar="NAME",
        help="Output filename stem saved inside output/. "
             "e.g. --out-name ahmed_greeting  →  output/ahmed_greeting.wav")
    parser.add_argument(
        "--out", default=None, metavar="PATH",
        help="Full output path override (overrides --out-name). Auto-numbered if exists.")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a fine-tuned F5-TTS checkpoint .pt file (optional)")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speech speed multiplier, e.g. 0.9 for slightly slower (default 1.0)")
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="RNG seed for reproducibility (-1 = random, default -1)")
    parser.add_argument(
        "--nfe-steps", type=int, default=32,
        help="Diffusion NFE steps: 16 (fast) to 32 (quality). Default 32.")
    args = parser.parse_args()

    reference_path = Path(args.reference)
    if not reference_path.exists():
        sys.exit(f"Reference audio not found: {reference_path}\n"
                 "Pass --reference or set CLONE_REFERENCE_AUDIO in config.py")

    if args.checkpoint and not Path(args.checkpoint).exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    out_path = resolve_output(args.out, args.out_name, reference_path, default_ref)

    infer(
        reference_path = reference_path,
        ref_text       = args.ref_text.strip(),
        gen_text       = args.text.strip(),
        out_path       = out_path,
        checkpoint     = args.checkpoint,
        speed          = args.speed,
        seed           = args.seed,
        nfe_steps      = args.nfe_steps,
    )


if __name__ == "__main__":
    main()

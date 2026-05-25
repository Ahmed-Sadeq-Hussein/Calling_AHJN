#!/usr/bin/env python3
# CALLING AHJIN — Brought to you by Ahmed Hussein and Julius Norén from JTH.
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
# KMP_DUPLICATE_LIB_OK: PyTorch (Intel MKL) and numpy/numba each ship their own
# OpenMP runtime. On Windows they collide on first import, causing a fatal OMP #15
# abort. Allowing the duplicate is the standard inference-time workaround.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Suppress noisy HuggingFace symlinks warning on Windows (symlinks need admin rights).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# TorchDynamo's LLVM/SVML vectoriser crashes on some Windows + Anaconda combos.
# Disabling it falls back to eager mode — no quality loss for inference.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

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
    Convert any audio file to 16-bit PCM WAV so F5-TTS can read it without FFmpeg.

    F5-TTS calls pydub internally, which requires a system FFmpeg for MP3/M4A etc.
    We avoid that dependency by trying four pure-Python audio backends in order.
    Converted files are cached in output/ so repeated calls are instant.

    Returns the original path unchanged if it is already a WAV file.
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

    # Backend 1: soundfile — fast, zero overhead; handles WAV/FLAC/OGG natively.
    # Newer libsndfile (≥1.1.0) also reads MP3.
    if wav is None:
        try:
            import soundfile as sf
            wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        except Exception:
            pass

    # Backend 2: librosa — uses audioread which falls back to Windows Media Foundation
    # for MP3/M4A when FFmpeg is absent.
    if wav is None:
        try:
            import librosa
            wav, sr = librosa.load(str(audio_path), sr=None, mono=True)
        except Exception:
            pass

    # Backend 3: torchaudio — ships bundled codec support via ffmpeg/sox backends.
    if wav is None:
        try:
            import torch, torchaudio
            waveform, sr = torchaudio.load(str(audio_path))
            import numpy as np
            wav = waveform.mean(0).numpy()   # mix stereo channels to mono
        except Exception:
            pass

    # Backend 4: PyAV — the av package (installed with faster-whisper) bundles its
    # own FFmpeg DLLs, so this works even without a system FFmpeg installation.
    if wav is None:
        try:
            import av, numpy as np
            container = av.open(str(audio_path))
            stream = next(s for s in container.streams if s.type == "audio")
            frames = [f.to_ndarray() for f in container.decode(stream)]
            wav = np.concatenate(frames, axis=-1).mean(axis=0).astype("float32")
            wav /= max(abs(wav).max(), 1e-6)   # normalise amplitude to [-1, 1]
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
    Transcribe a WAV reference clip with faster-whisper (Whisper base model).

    Reads the audio with soundfile (no FFmpeg needed for WAV), resamples to
    16 kHz (Whisper's required input rate), and runs beam-search with VAD
    filtering to skip silence segments.

    Returns the raw transcript string for use as ref_text in F5-TTS inference.
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
        wav = wav.mean(axis=1)   # mix to mono before resampling
    if sr != 16000:
        # Whisper was trained at 16 kHz; resample if the WAV was recorded at a different rate
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
    Remove trailing silence and apply a short cosine fade-out.

    F5-TTS sometimes leaves an abrupt hard boundary at the end of a generated
    clip, which sounds like a word being cut mid-syllable. This function:
      1. Scans backward to find the last voiced sample (above silence_db dBFS).
      2. Keeps an 80 ms natural decay tail after that sample.
      3. Applies a linear fade-out over the final fade_ms milliseconds.

    silence_db: amplitude threshold in dBFS; default -42 dB ≈ near-silence.
    fade_ms   : length of the fade-out window in milliseconds.
    """
    import numpy as np
    threshold = 10 ** (silence_db / 20)
    abs_wav = np.abs(wav)
    # Walk backwards from the end to find the last sample above silence threshold
    end_idx = len(wav)
    for i in range(len(wav) - 1, -1, -1):
        if abs_wav[i] > threshold:
            end_idx = i + 1
            break
    # Retain 80 ms of natural ring-off after the last voiced sample
    tail_samples = int(0.08 * sample_rate)
    end_idx = min(end_idx + tail_samples, len(wav))
    wav = wav[:end_idx].copy()
    # Linear fade-out over the last fade_ms milliseconds
    fade_samples = min(int(fade_ms / 1000 * sample_rate), len(wav))
    wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    return wav


def split_sentences(text: str) -> list[str]:
    """
    Split text into individual sentences at . ! ? boundaries.

    F5-TTS degrades on very long inputs (>100 tokens).  Splitting at sentence
    boundaries keeps each inference call within the model's comfortable range
    (~3–6 seconds of speech), then infer() stitches the pieces back together
    with a short silence gap between sentences.
    """
    import re
    # Lookbehind: split after a sentence-ending punctuation mark followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Drop empty segments; punctuation stays attached to its sentence
    return [p.strip() for p in parts if p.strip()]


def infer(reference_path: Path, ref_text: str, gen_text: str,
          out_path: Path, checkpoint: str | None,
          speed: float, seed: int, nfe_steps: int,
          t_script_start: float) -> None:
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
    t_script_start : time.perf_counter() value from the top of main()
    """
    import time
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
    t_load_start = time.perf_counter()
    tts = F5TTS(ckpt_file=ckpt_file)
    t_load_end = time.perf_counter()
    print(f"  Model loaded in {t_load_end - t_load_start:.1f}s")
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reference  : {reference_path}")
    print(f"Ref text   : {ref_text[:80]}{'…' if len(ref_text) > 80 else ''}")
    print(f'Generating : "{gen_text[:80]}{"…" if len(gen_text) > 80 else ""}"')
    print()

    # ── Sentence-by-sentence generation ──────────────────────────────────────
    # Fine-tuned models trained on short clips (2–5 s) can produce muffled or
    # repetitive audio on long inputs. Splitting at sentence boundaries keeps
    # each call within the model's training distribution, then we concatenate
    # the individual WAVs with a short silence gap.
    sentences = split_sentences(gen_text)

    t_infer_start = time.perf_counter()

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
        sentence_times = []
        sample_rate = None
        silence_gap = 0.18   # seconds of silence between sentences

        for i, sentence in enumerate(sentences):
            t_sent_start = time.perf_counter()
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
            t_sent_end = time.perf_counter()
            wav_i = trim_and_fade(wav_i, sr_i)
            sent_duration = len(wav_i) / sr_i
            sent_infer    = t_sent_end - t_sent_start
            sentence_times.append((sentence, sent_infer, sent_duration))
            print(f"      {sent_infer:.1f}s to generate  "
                  f"→  {sent_duration:.1f}s audio  "
                  f"(RTF {sent_infer / sent_duration:.2f})")
            waves.append(wav_i)
            sample_rate = sr_i

        # Insert a 180 ms silence gap between sentences to mimic natural speech
        # pausing, then concatenate all pieces into a single waveform.
        gap = np.zeros(int(silence_gap * sample_rate), dtype=np.float32)
        stitched = []
        for i, w in enumerate(waves):
            stitched.append(w)
            if i < len(waves) - 1:
                stitched.append(gap)

        wav = np.concatenate(stitched)
        sr  = sample_rate
        sf.write(str(out_path), wav, sr)

    t_done = time.perf_counter()

    # ── Latency report ────────────────────────────────────────────────────────
    audio_duration  = len(wav) / sr
    infer_time      = t_done - t_infer_start
    total_time      = t_done - t_script_start
    model_load_time = t_load_end - t_load_start
    rtf             = infer_time / audio_duration

    print()
    print("─" * 50)
    print("  LATENCY REPORT")
    print("─" * 50)
    print(f"  Model load time   : {model_load_time:.2f}s")
    print(f"  Inference time    : {infer_time:.2f}s")
    print(f"  Total wall-clock  : {total_time:.2f}s")
    print(f"  Output duration   : {audio_duration:.2f}s")
    print(f"  Real-time factor  : {rtf:.3f}x  "
          f"({'faster' if rtf < 1 else 'slower'} than real-time)")
    print(f"  Sentences         : {len(sentences)}")
    print(f"  NFE steps         : {nfe_steps}")
    print(f"  Checkpoint        : {'fine-tuned' if checkpoint else 'pretrained base'}")
    print("─" * 50)
    print(f"  Saved : {out_path}")
    print("─" * 50)


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


def load_config_json(name: str) -> dict:
    """
    Load a JSON config file from the Voice_configuration/ folder.

    'name' can be:
      - a bare stem:        "ahmed_test"
      - with extension:     "ahmed_test.json"
      - a full/relative path: "some/other/folder/ahmed_test.json"

    Supported JSON keys (all optional except 'text'):
        reference   – path to reference WAV/MP3
        ref_text    – transcript of the reference clip
        text        – text to synthesise  (required)
        checkpoint  – path to fine-tuned .pt checkpoint
        out_name    – output filename stem  (saves to output/<out_name>.wav)
        out         – full output path override
        speed       – float, default 1.0
        seed        – int,   default -1
        nfe_steps   – int,   default 32
    """
    import json

    p = Path(name)

    # If it looks like an existing path already, use it directly
    if p.suffix.lower() == ".json" and p.exists():
        cfg_path = p
    else:
        # Strip .json if user accidentally typed it, then look in Voice_configuration/
        stem = p.stem if p.suffix.lower() == ".json" else p.name
        cfg_dir = Path(config.__file__).resolve().parent / "Voice_configuration"
        cfg_path = cfg_dir / f"{stem}.json"

    if not cfg_path.exists():
        # List available configs to help the user
        cfg_dir = Path(config.__file__).resolve().parent / "Voice_configuration"
        available = sorted(cfg_dir.glob("*.json")) if cfg_dir.is_dir() else []
        names = "\n  ".join(p.stem for p in available) if available else "(none found)"
        sys.exit(
            f"Config file not found: {cfg_path}\n\n"
            f"Available configs in Voice_configuration/:\n  {names}\n\n"
            f"Usage:  python clone_voice.py --config <name>"
        )

    with open(cfg_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Config loaded: {cfg_path.name}")
    return data


def main() -> None:
    import time
    t_script_start = time.perf_counter()   # capture as early as possible

    default_ref = str(config.CLONE_REFERENCE_AUDIO) if config.CLONE_REFERENCE_AUDIO else ""

    parser = argparse.ArgumentParser(
        description="Zero-shot voice cloning with F5-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load everything from a JSON config file:
  python clone_voice.py --config ahmed_crossexamination
  # → reads Voice_configuration/ahmed_crossexamination.json

  # Use your default reference (set in config.py):
  python clone_voice.py --text "Hello world."

  # Clone a specific person's voice, auto-named output:
  python clone_voice.py --reference voices/john.wav --ref-text "What John said." --text "Hello."
  # → output/john_clone.wav

  # Clone and give the output a custom name:
  python clone_voice.py --reference voices/john.wav --ref-text "What John said." \\
                        --text "Hello." --out-name john_greeting
  # → output/john_greeting.wav

  # CLI flags always override values from --config:
  python clone_voice.py --config ahmed_crossexamination --nfe-steps 16
""")

    parser.add_argument(
        "--config", default=None, metavar="NAME",
        help="Name of a JSON file in Voice_configuration/ (no .json needed). "
             "All other flags are optional when using --config; CLI flags override JSON values.")
    parser.add_argument(
        "--reference", default=None,
        help="Reference audio clip of the voice to clone (5-30 s, WAV/FLAC/MP3).")
    parser.add_argument(
        "--ref-text", default=None,
        help="Transcript of the reference clip. If omitted, Whisper auto-transcribes.")
    parser.add_argument(
        "--text", default=None,
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
        "--speed", type=float, default=None,
        help="Speech speed multiplier, e.g. 0.9 for slightly slower (default 1.0)")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducibility (-1 = random, default -1)")
    parser.add_argument(
        "--nfe-steps", type=int, default=None,
        help="Diffusion NFE steps: 16 (fast) to 32 (quality). Default 32.")
    args = parser.parse_args()

    # ── Merge JSON config + CLI (CLI always wins) ──────────────────────────────
    cfg: dict = {}
    if args.config:
        cfg = load_config_json(args.config)

    # Resolve each value: CLI arg → JSON value → built-in default
    reference  = args.reference  or cfg.get("reference",  default_ref)
    ref_text   = args.ref_text   if args.ref_text is not None else cfg.get("ref_text",  "")
    gen_text   = args.text       or cfg.get("text",        None)
    out_name   = args.out_name   or cfg.get("out_name",    None)
    out        = args.out        or cfg.get("out",         None)
    checkpoint = args.checkpoint or cfg.get("checkpoint",  None)
    speed      = args.speed      if args.speed      is not None else float(cfg.get("speed",     1.0))
    seed       = args.seed       if args.seed       is not None else int(cfg.get("seed",       -1))
    nfe_steps  = args.nfe_steps  if args.nfe_steps  is not None else int(cfg.get("nfe_steps",  32))

    # ── Validate ───────────────────────────────────────────────────────────────
    if not reference:
        sys.exit("No reference audio specified.\n"
                 "Use --reference, set CLONE_REFERENCE_AUDIO in config.py, "
                 "or add 'reference' to your JSON config.")

    if not gen_text:
        sys.exit("No text to synthesise.\n"
                 "Use --text or add 'text' to your JSON config.")

    reference_path = Path(reference)
    if not reference_path.is_absolute():
        reference_path = Path(config.__file__).resolve().parent / reference_path
    if not reference_path.exists():
        sys.exit(f"Reference audio not found: {reference_path}")

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(config.__file__).resolve().parent / ckpt_path
        if not ckpt_path.exists():
            sys.exit(f"Checkpoint not found: {ckpt_path}")
        checkpoint = str(ckpt_path)

    out_path = resolve_output(out, out_name, reference_path, default_ref)

    infer(
        reference_path = reference_path,
        ref_text       = ref_text.strip(),
        gen_text       = gen_text.strip(),
        out_path       = out_path,
        checkpoint     = checkpoint,
        speed          = speed,
        seed           = seed,
        nfe_steps      = nfe_steps,
        t_script_start = t_script_start,
    )


if __name__ == "__main__":
    main()

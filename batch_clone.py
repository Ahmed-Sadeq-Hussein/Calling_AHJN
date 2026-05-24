#!/usr/bin/env python3
"""
batch_clone.py
--------------
Batch voice cloning — generates every TARGET_TEXT in every REFERENCE_AUDIO's voice.

Configure everything in config.py:

    REFERENCE_AUDIOS = {
        "ahmed":  ("data/references/ahmed_reference.wav", ""),
        "church": ("Cross_Examination/Church world tour.wav", "South Africa ..."),
    }

    TARGET_TEXTS = {
        "greeting":   "Hi, I am your personal AI assistant.",
        "bible_open": "In the beginning God created the heaven and the earth.",
    }

    BATCH_CHECKPOINT = ""   # "" = pretrained base model
                            # or "models/f5tts_finetune/ahmed/model_last.pt"

Output structure:
    Output_results/
        ahmed/
            ahmed_greeting.wav
            ahmed_bible_open.wav
        church/
            church_greeting.wav
            church_bible_open.wav

The model is loaded ONCE and reused for all generations (faster).

Usage:
    python batch_clone.py
    python batch_clone.py --nfe-steps 16      # faster, slightly lower quality
    python batch_clone.py --dry-run           # list what would be generated
"""

from __future__ import annotations
import os, sys, argparse, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    sys.exit("Run:  pip install soundfile numpy")


# ── Helpers (shared with clone_voice.py) ──────────────────────────────────────

def ensure_wav(audio_path: Path) -> Path:
    """Convert any audio format to WAV if needed. Returns WAV path."""
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    out_wav = config.OUTPUT_DIR / f"{audio_path.stem}_ref.wav"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if out_wav.exists():
        return out_wav

    print(f"  Converting {audio_path.name} → WAV …")

    wav, sr = None, None

    try:
        import soundfile as sf_
        wav, sr = sf_.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception:
        pass

    if wav is None:
        try:
            import torch, torchaudio
            waveform, sr = torchaudio.load(str(audio_path))
            wav = waveform.mean(0).numpy()
        except Exception:
            pass

    if wav is None:
        try:
            import av
            container = av.open(str(audio_path))
            stream = next(s for s in container.streams if s.type == "audio")
            frames = [f.to_ndarray() for f in container.decode(stream)]
            wav = np.concatenate(frames, axis=-1).mean(axis=0).astype("float32")
            wav /= max(abs(wav).max(), 1e-6)
            sr = stream.codec_context.sample_rate
        except Exception:
            pass

    if wav is None:
        sys.exit(f"Could not convert {audio_path} — install ffmpeg or av package.")

    import soundfile as sf_
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    sf_.write(str(out_wav), wav.astype("float32"), sr, subtype="PCM_16")
    return out_wav


def split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries for clean generation."""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def trim_and_fade(wav: np.ndarray, sample_rate: int,
                  silence_db: float = -42.0, fade_ms: float = 60.0) -> np.ndarray:
    """Remove trailing silence and apply a short fade-out."""
    threshold = 10 ** (silence_db / 20)
    abs_wav = np.abs(wav)
    end_idx = len(wav)
    for i in range(len(wav) - 1, -1, -1):
        if abs_wav[i] > threshold:
            end_idx = i + 1
            break
    tail = int(0.08 * sample_rate)
    end_idx = min(end_idx + tail, len(wav))
    wav = wav[:end_idx].copy()
    fade_samples = min(int(fade_ms / 1000 * sample_rate), len(wav))
    wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    return wav


def generate_one(tts, ref_wav: str, ref_text: str, gen_text: str,
                 out_path: Path, nfe_steps: int, seed: int | None) -> tuple[float, float]:
    """
    Generate one clip. Returns (inference_seconds, audio_duration_seconds).
    Handles sentence splitting and stitching internally.
    """
    sentences   = split_sentences(gen_text)
    silence_gap = 0.18

    t0 = time.perf_counter()

    if len(sentences) <= 1:
        wav, sr, _ = tts.infer(
            ref_file=ref_wav,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=None,
            nfe_step=nfe_steps,
            seed=seed,
        )
    else:
        waves = []
        sample_rate = None
        for sentence in sentences:
            wav_i, sr_i, _ = tts.infer(
                ref_file=ref_wav,
                ref_text=ref_text,
                gen_text=sentence,
                file_wave=None,
                nfe_step=nfe_steps,
                seed=seed,
            )
            wav_i = trim_and_fade(wav_i, sr_i)
            waves.append(wav_i)
            sample_rate = sr_i

        gap = np.zeros(int(silence_gap * sample_rate), dtype=np.float32)
        stitched = []
        for i, w in enumerate(waves):
            stitched.append(w)
            if i < len(waves) - 1:
                stitched.append(gap)
        wav = np.concatenate(stitched)
        sr  = sample_rate

    infer_time = time.perf_counter() - t0
    audio_dur  = len(wav) / sr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, sr)

    return infer_time, audio_dur


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Batch voice cloning — all references × all target texts")
    parser.add_argument(
        "--nfe-steps", type=int, default=32,
        help="Diffusion steps: 16 (fast) or 32 (quality, default)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default 42)")
    parser.add_argument(
        "--checkpoint", default=config.BATCH_CHECKPOINT or None,
        help="Fine-tuned checkpoint .pt (default: pretrained base model)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be generated without running inference")
    args = parser.parse_args()

    # ── Validate config ────────────────────────────────────────────────────────
    ref_audios  = config.REFERENCE_AUDIOS
    target_texts = config.TARGET_TEXTS

    if not ref_audios:
        sys.exit(
            "REFERENCE_AUDIOS is empty in config.py.\n"
            "Add at least one entry, e.g.:\n"
            '  "ahmed": ("data/references/ahmed_reference.wav", "")')

    if not target_texts:
        sys.exit(
            "TARGET_TEXTS is empty in config.py.\n"
            "Add at least one entry, e.g.:\n"
            '  "greeting": "Hi, I am your personal AI assistant."')

    out_root = config.BATCH_OUTPUT_DIR
    total    = len(ref_audios) * len(target_texts)

    # ── Print plan ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  BATCH VOICE CLONING")
    print("=" * 60)
    print(f"  References   : {len(ref_audios)}")
    print(f"  Target texts : {len(target_texts)}")
    print(f"  Total clips  : {total}")
    print(f"  Output dir   : {out_root}")
    print(f"  Checkpoint   : {args.checkpoint or 'pretrained base model'}")
    print(f"  NFE steps    : {args.nfe_steps}")
    print()

    print("  Plan:")
    for ref_label, (ref_path, _) in ref_audios.items():
        for tgt_label in target_texts:
            out = out_root / ref_label / f"{ref_label}_{tgt_label}.wav"
            print(f"    {ref_label:15s}  ×  {tgt_label:20s}  →  {out.name}")
    print()

    if args.dry_run:
        print("  Dry run — exiting.")
        return

    # ── Load model once ────────────────────────────────────────────────────────
    try:
        from f5_tts.api import F5TTS
    except ImportError:
        sys.exit("F5-TTS not installed. Run:  .\\setup_f5tts.ps1")

    print("Loading F5-TTS model …")
    t_load = time.perf_counter()
    ckpt   = args.checkpoint or ""
    tts    = F5TTS(ckpt_file=ckpt)
    print(f"  Model ready in {time.perf_counter() - t_load:.1f}s")
    print()

    # ── Generate ───────────────────────────────────────────────────────────────
    results = []   # (ref_label, tgt_label, infer_s, audio_s, out_path)
    done    = 0

    for ref_label, (ref_path_raw, ref_text) in ref_audios.items():
        ref_path = Path(ref_path_raw)
        if not ref_path.is_absolute():
            ref_path = config.PROJECT_ROOT / ref_path

        if not ref_path.exists():
            print(f"  ✗ Reference not found, skipping: {ref_path}")
            continue

        # Convert to WAV once per reference
        ref_wav = str(ensure_wav(ref_path))

        # Auto-transcribe once if transcript not provided
        if not ref_text.strip():
            print(f"  Auto-transcribing {ref_label} …")
            try:
                import torch, soundfile as sf_
                import numpy as np, librosa
                from faster_whisper import WhisperModel
                wav_data, wav_sr = sf_.read(ref_wav, dtype="float32", always_2d=False)
                if wav_data.ndim > 1:
                    wav_data = wav_data.mean(axis=1)
                if wav_sr != 16000:
                    wav_data = librosa.resample(wav_data, orig_sr=wav_sr, target_sr=16000)
                device  = "cuda" if torch.cuda.is_available() else "cpu"
                compute = "float16" if device == "cuda" else "int8"
                whisper = WhisperModel("base", device=device, compute_type=compute)
                segs, _ = whisper.transcribe(wav_data, beam_size=5, language="en",
                                              vad_filter=True)
                ref_text = " ".join(s.text.strip() for s in segs).strip()
                del whisper
                print(f"    Transcript: \"{ref_text[:80]}\"")
            except Exception as e:
                print(f"  ✗ Transcription failed ({e}) — skipping {ref_label}")
                continue

        print(f"  ── {ref_label} ──────────────────────────────────")

        for tgt_label, tgt_text in target_texts.items():
            done += 1
            out_path = out_root / ref_label / f"{ref_label}_{tgt_label}.wav"

            print(f"  [{done}/{total}]  {tgt_label}")
            print(f"         \"{tgt_text[:70]}{'…' if len(tgt_text) > 70 else ''}\"")

            try:
                infer_s, audio_s = generate_one(
                    tts      = tts,
                    ref_wav  = ref_wav,
                    ref_text = ref_text,
                    gen_text = tgt_text,
                    out_path = out_path,
                    nfe_steps = args.nfe_steps,
                    seed     = args.seed,
                )
                rtf = infer_s / audio_s if audio_s > 0 else 0
                print(f"         ✓  {audio_s:.1f}s audio  |  "
                      f"{infer_s:.1f}s inference  |  RTF {rtf:.2f}  →  {out_path.name}")
                results.append((ref_label, tgt_label, infer_s, audio_s, out_path))

            except Exception as e:
                print(f"         ✗  Failed: {e}")

        print()

    # ── Summary ────────────────────────────────────────────────────────────────
    total_wall   = time.perf_counter() - t_start
    total_infer  = sum(r[2] for r in results)
    total_audio  = sum(r[3] for r in results)
    overall_rtf  = total_infer / total_audio if total_audio > 0 else 0

    print("=" * 60)
    print("  BATCH COMPLETE")
    print("=" * 60)
    print(f"  Clips generated  : {len(results)} / {total}")
    print(f"  Total audio out  : {total_audio:.1f}s  ({total_audio/60:.1f} min)")
    print(f"  Total inference  : {total_infer:.1f}s")
    print(f"  Overall RTF      : {overall_rtf:.3f}x")
    print(f"  Wall-clock time  : {total_wall:.1f}s")
    print(f"  Output folder    : {out_root}")
    print()
    print("  Files saved:")
    for ref_label, tgt_label, infer_s, audio_s, out_path in results:
        print(f"    {out_path.relative_to(config.PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

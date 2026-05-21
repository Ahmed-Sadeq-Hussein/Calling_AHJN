#!/usr/bin/env python3
"""
download_mls.py
---------------
Downloads multi-speaker TTS data. No account or token required.

  English  → LibriTTS  streamed directly from openslr.org
             ~100 h clean (train-clean-100) + ~360 h clean (train-clean-360)
  DE FR NL ES IT PT PL → MLS (facebook/multilingual_librispeech on HuggingFace)

Audio decoded with soundfile only — no torchcodec or FFmpeg needed.

Output:
    data/processed/en/   wav files (22 050 Hz)
    data/processed/<xx>/ wav files per extra language
    data/processed/train_filelist.txt
"""

from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Patch datasets Audio.encode_example before import to avoid torchcodec
try:
    import datasets.features.audio as _hf_audio

    def _patched_encode(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        return {"bytes": None, "path": str(value)}

    _hf_audio.Audio.encode_example = _patched_encode
except Exception:
    pass

import sys, io, tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# LibriTTS splits in quality order — stops early once hour cap is hit
LIBRITTS_SPLITS = [
    ("train-clean-100", "https://www.openslr.org/resources/60/train-clean-100.tar.gz"),
    ("train-clean-360", "https://www.openslr.org/resources/60/train-clean-360.tar.gz"),
]

MLS_LANG_MAP = {
    "DE": "german", "FR": "french", "NL": "dutch",
    "ES": "spanish", "IT": "italian", "PT": "portuguese", "PL": "polish",
}


def resample(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == config.SAMPLE_RATE:
        return wav.astype(np.float32)
    return librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=config.SAMPLE_RATE)


# ── LibriTTS (English) ─────────────────────────────────────────────────────────

def iter_libritts_tar(url: str, pbar: tqdm):
    """
    Stream a LibriTTS tar.gz from url, yielding (speaker_id, clip_id, wav, sr, text).
    Matches each .wav with its .normalized.txt by stem — buffers unmatched halves.
    """
    wav_buf: dict[str, bytes] = {}
    txt_buf: dict[str, str]  = {}

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        # r|gz = pipe/streaming mode; does not seek — perfect for HTTP streams
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                p   = Path(member.name)
                ext = p.suffix.lower()
                f   = tar.extractfile(member)
                if f is None:
                    continue
                raw = f.read()

                if ext == ".wav":
                    stem = p.stem                              # e.g. 1234_5678_000001
                    wav_buf[stem] = raw
                elif ext == ".txt" and ".normalized" in p.name:
                    stem = p.stem.replace(".normalized", "")  # strip .normalized
                    txt_buf[stem] = raw.decode("utf-8").strip()

                # Yield as soon as both halves of a clip are in hand
                for stem in list(wav_buf.keys() & txt_buf.keys()):
                    wav_bytes = wav_buf.pop(stem)
                    text      = txt_buf.pop(stem)
                    speaker   = stem.split("_")[0]
                    try:
                        wav, sr = sf.read(io.BytesIO(wav_bytes),
                                          dtype="float32", always_2d=False)
                        yield speaker, stem, wav, sr, text
                    except Exception as e:
                        pbar.write(f"[skip] {stem}: {e}")


def download_english(out_root: Path, max_sec: float,
                     manifest_fh) -> tuple[int, float]:
    print("[EN] Source: LibriTTS via openslr.org (no auth needed)")
    lang_dir = out_root / "en"
    lang_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    total_sec = 0.0

    for split_name, url in LIBRITTS_SPLITS:
        if total_sec >= max_sec:
            break
        print(f"  Streaming {split_name} …")
        pbar = tqdm(desc=f"EN/{split_name}", unit="clip")

        for speaker_id, clip_id, wav, sr, text in iter_libritts_tar(url, pbar):
            if total_sec >= max_sec:
                break

            duration = len(wav) / sr
            if not (config.MLS_MIN_DURATION_SEC <= duration <= config.MLS_MAX_DURATION_SEC):
                continue

            wav = resample(wav, sr)
            wav_path = lang_dir / f"{clip_id}.wav"
            sf.write(str(wav_path), wav, config.SAMPLE_RATE, subtype="PCM_16")

            # Write + flush immediately so the line is visible to training now
            manifest_fh.write(f"{wav_path}|{speaker_id}|{text}\n")
            manifest_fh.flush()

            count += 1
            total_sec += duration
            pbar.set_postfix(hours=f"{total_sec/3600:.2f}")
            pbar.update()

        pbar.close()
        if total_sec >= max_sec:
            break

    print(f"  [EN] {total_sec/3600:.2f} h  ({count} clips)")
    return count, total_sec


# ── MLS (non-English) ──────────────────────────────────────────────────────────

def download_mls_lang(iso: str, out_root: Path, max_sec: float,
                      manifest_fh) -> tuple[int, float]:
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        sys.exit("Run:  pip install datasets")

    mls_name = MLS_LANG_MAP[iso]
    print(f"[{iso}] Source: MLS/{mls_name} via HuggingFace")
    lang_dir = out_root / iso.lower()
    lang_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "facebook/multilingual_librispeech",
        mls_name,
        split="train",
        streaming=True,
        cache_dir=str(config.MLS_CACHE_DIR),
        trust_remote_code=False,
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    count = 0
    total_sec = 0.0
    pbar = tqdm(desc=f"[{iso}]", unit="clip")

    for sample in ds:
        if total_sec >= max_sec:
            break

        audio_field = sample.get("audio", {})
        raw = audio_field.get("bytes")
        if not raw:
            continue
        try:
            wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception as e:
            pbar.write(f"[skip] {e}")
            continue

        duration = len(wav) / sr
        if not (config.MLS_MIN_DURATION_SEC <= duration <= config.MLS_MAX_DURATION_SEC):
            continue

        wav = resample(wav, sr)
        clip_id    = str(sample.get("id", pbar.n)).replace("/", "_")
        speaker_id = str(sample.get("speaker_id", "unknown"))
        text       = str(sample.get("text", "")).strip()

        wav_path = lang_dir / f"{clip_id}.wav"
        sf.write(str(wav_path), wav, config.SAMPLE_RATE, subtype="PCM_16")

        # Write + flush immediately so training can start on partial data
        manifest_fh.write(f"{wav_path}|{speaker_id}|{text}\n")
        manifest_fh.flush()

        count += 1
        total_sec += duration
        pbar.set_postfix(hours=f"{total_sec/3600:.2f}")
        pbar.update()

    pbar.close()
    print(f"  [{iso}] {total_sec/3600:.2f} h  ({count} clips)")
    return count, total_sec


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    out_root = config.PROCESSED_DATA_DIR
    manifest = out_root / "train_filelist.txt"
    max_sec  = (config.MLS_MAX_HOURS * 3600) if config.MLS_MAX_HOURS else float("inf")

    print("Multi-speaker TTS downloader")
    print(f"Languages : {config.MLS_LANGUAGES}")
    print(f"Max hours : {config.MLS_MAX_HOURS} per language")
    print(f"Duration  : {config.MLS_MIN_DURATION_SEC}s – {config.MLS_MAX_DURATION_SEC}s\n")
    print(f"Manifest written live to: {manifest}")
    print("You can start  python partial_train.py  in another terminal at any time.\n")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    total_clips = 0

    # Open manifest once upfront — each function flushes after every clip
    with open(manifest, "w", encoding="utf-8") as manifest_fh:
        for iso in config.MLS_LANGUAGES:
            iso = iso.upper()
            if iso == "EN":
                count, _ = download_english(out_root, max_sec, manifest_fh)
            elif iso in MLS_LANG_MAP:
                count, _ = download_mls_lang(iso, out_root, max_sec, manifest_fh)
            else:
                valid = f"EN {' '.join(MLS_LANG_MAP)}"
                print(f"[skip] '{iso}' not supported. Valid codes: {valid}")
                count = 0
            total_clips += count

    print(f"\nTotal: {total_clips} clips written to manifest")
    print(f"Manifest: {manifest}")
    print("\nNext: python preprocess_emilia.py")


if __name__ == "__main__":
    main()

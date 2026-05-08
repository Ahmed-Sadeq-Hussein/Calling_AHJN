from pathlib import Path
import random

from nemo_text_processing.text_normalization.normalize import Normalizer


LJSPEECH_DIR = Path("data/LJSpeech-1.1")
METADATA_PATH = LJSPEECH_DIR / "metadata.csv"
WAVS_DIR = LJSPEECH_DIR / "wavs"

OUTPUT_DIR = Path("filelists")
TRAIN_FILE = OUTPUT_DIR / "train.txt"
VAL_FILE = OUTPUT_DIR / "val.txt"

VAL_RATIO = 0.1


normalizer = Normalizer(input_case='cased', lang='en')


def normalize_text(text):
    text = text.strip()

    normalized = normalizer.normalize(text, verbose=False, punct_post_process=True)

    return normalized.strip()


def read_ljspeech_metadata(metadata_path):
    entries = []

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue
            
            parts = line.split("|")

            if len(parts) < 2:
                continue

            audio_id = parts[0]
            lj_normalized_text = parts[2]

            wav_path = WAVS_DIR / f"{audio_id}.wav"

            if not wav_path.exists():
                print(f"Warning: missing wav file: {wav_path}")
                continue
        
            normalized_text = normalize_text(lj_normalized_text)

            entries.append((Path("wavs") / f"{audio_id}.wav", normalized_text))

    return entries


def write_filelist(entries, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for wav_path, text in entries:
            f.write(f"{wav_path}|{text}\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading LJ Speech metadata...")
    entries = read_ljspeech_metadata(METADATA_PATH)

    print(f"Loaded {len(entries)} entries.")

    random.shuffle(entries)

    val_size = int(len(entries) * VAL_RATIO)

    val_entries = entries[:val_size]
    train_entries = entries[val_size:]

    write_filelist(train_entries, TRAIN_FILE)
    write_filelist(val_entries, VAL_FILE)

    print(f"Saved training filelist to: {TRAIN_FILE}")
    print(f"Saved validation filelist to: {VAL_FILE}")
    print(f"Training samples: {len(train_entries)}")
    print(f"Validation samples: {len(val_entries)}")


if __name__ == "__main__":
    main()
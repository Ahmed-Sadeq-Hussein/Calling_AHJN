from pathlib import Path
import json
import wave

TRAIN_FILE = Path("filelists/train.txt")
VAL_FILE = Path("filelists/val.txt")

TRAIN_MANIFEST = Path("filelists/train_manifest.json")
VAL_MANIFEST = Path("filelists/val_manifest.json")

def get_wav_duration(wav_path):
    with wave.open(str(wav_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        return frames / float(sample_rate)


def convert_filelist_to_manifest(filelist_path, manifest_path):
    with open(filelist_path, "r", encoding="utf-8") as infile, \
         open(manifest_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            line = line.strip()

            if not line:
                continue

            wav_path_str, text = line.split("|", maxsplit=1)
            wav_path = Path(wav_path_str)

            duration = get_wav_duration(wav_path)

            item = {
                "audio_filepath": str(wav_path),
                "text": text,
                "duration": duration
            }

            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    convert_filelist_to_manifest(TRAIN_FILE, TRAIN_MANIFEST)
    convert_filelist_to_manifest(VAL_FILE, VAL_MANIFEST)

    print(f"Saved: {TRAIN_MANIFEST}")
    print(f"Saved: {VAL_MANIFEST}")


if __name__ == "__main__":
    main()
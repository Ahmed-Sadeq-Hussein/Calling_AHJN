from pathlib import Path
import json

manifest = Path("filelists/train_small_manifest.json")

with open(manifest, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        item = json.loads(line)

        if not Path(item["audio_filepath"]).exists():
            print("Missing:", item["audio_filepath"])
        else:
            print(item)

        if i == 10:
            break

print("Checked first 10 samples")
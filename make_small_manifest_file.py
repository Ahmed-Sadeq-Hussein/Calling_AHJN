from pathlib import Path

def make_subset(input_path, output_path, n):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for i, line in enumerate(infile):
            if i >= n:
                break
            outfile.write(line)


make_subset("filelists/train_manifest.json", "filelists/train_small_manifest.json", 300)
make_subset("filelists/val_manifest.json", "filelists/val_small_manifest.json", 30)
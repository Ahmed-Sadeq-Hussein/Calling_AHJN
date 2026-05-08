from pathlib import Path

def make_subset(input_file, output_file, n):
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile):
            if i >= n:
                break
            outfile.write(line)


make_subset("filelists/train.txt", "filelists/train_small.txt", 300)
make_subset("filelists/val.txt", "filelists/val_small.txt", 30)
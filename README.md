# CALLING AHJIN
Brought to you by Ahmed Hussein and Julius Norén from JTH.

---

## Project Overview

CALLING AHJIN is an F5-TTS voice cloning and fine-tuning research project developed at Jönköping University (JTH). It investigates how training data characteristics — clip length, phonetic diversity, and dataset size — affect the quality of zero-shot and fine-tuned voice cloning.

**Core technology:** [F5-TTS](https://github.com/SWivid/F5-TTS) — a flow-matching text-to-speech model that can clone any voice from a short reference clip without any training (zero-shot), and can be further adapted to a specific speaker through fine-tuning.

**Research questions addressed:**
- How does fine-tuning on short phonetic clips compare to fine-tuning on longer natural sentences?
- What happens when a model trained on one distribution is continued on a different one (cross-training)?
- Can objective metrics (WER, speaker similarity) reliably predict perceived clone quality?

---

## Setup

Install all required Python packages:

```
pip install f5-tts jiwer faster-whisper resemblyzer librosa soundfile matplotlib
```

For microphone recording (`create_single_audio.py`):

```
pip install sounddevice
```

Optional but recommended for MP3/M4A reference audio support:

```
pip install av          # bundles its own FFmpeg DLLs
pip install torchaudio  # alternative codec backend
```

Python 3.10+ is required. An NVIDIA GPU is strongly recommended for training and inference.

---

## Scripts

### config.py

Central configuration file. All other scripts import this module. Edit it to set your paths and hyperparameters before running anything else.

**No CLI arguments** — edit the constants directly.

**Key settings:**

| Variable | Purpose |
|---|---|
| `training_voice` | Path to the folder containing the training speaker's WAV files |
| `target_voices` | List of paths to target-speaker folders for comparison |
| `experiment_name` | Sub-folder name for organising experiment outputs |
| `cross_examination_folder` | Folder scanned by `wer_graph.py`, `blind_test.py`, and `resembler metric.py` |
| `Resemble_threshold` | Cosine similarity threshold for "same speaker" judgment (default 0.75) |
| `F5TTS_FINETUNE_EPOCHS` | Default epoch count for fine-tuning |
| `F5TTS_FINETUNE_LR` | Learning rate for fine-tuning (default 1e-5) |
| `F5TTS_FINETUNE_BATCH_SIZE` | Clips per gradient step (default 8; reduce to 4 if OOM) |
| `CLONE_REFERENCE_AUDIO` | Default reference WAV used by `clone_voice.py` |
| `REFERENCE_AUDIOS` | Dict of `"label": (path, transcript)` for `batch_clone.py` |
| `TARGET_TEXTS` | Dict of `"label": "text"` for `batch_clone.py` |
| `BATCH_CHECKPOINT` | Optional fine-tuned checkpoint path for batch cloning |

---

### clone_voice.py

Zero-shot voice cloning using F5-TTS. No training required — provide a 5–30 second reference clip and the text to synthesise. The pretrained model (~1.2 GB) is downloaded automatically from HuggingFace on first use.

Multi-sentence texts are split at sentence boundaries and each sentence is generated independently, then stitched together with a short silence gap. This keeps each inference call within the model's comfortable generation range.

**CLI arguments:**

| Argument | Description | Default |
|---|---|---|
| `--reference PATH` | Reference audio clip (WAV/MP3/FLAC, 5–30 s) | `config.CLONE_REFERENCE_AUDIO` |
| `--ref-text TEXT` | Transcript of the reference clip | Auto-transcribed by Whisper |
| `--text TEXT` | Text to synthesise in the cloned voice | (required) |
| `--out-name NAME` | Output stem saved as `output/<NAME>.wav` | Based on reference name |
| `--out PATH` | Full output path override | — |
| `--checkpoint PATH` | Fine-tuned `.pt` checkpoint (optional) | Pretrained base model |
| `--speed FLOAT` | Speech speed multiplier, e.g. 0.9 for slower | 1.0 |
| `--seed INT` | RNG seed for reproducibility (-1 = random) | -1 |
| `--nfe-steps INT` | Diffusion steps: 16 (fast) to 32 (quality) | 32 |
| `--config NAME` | Load all settings from `Voice_configuration/<NAME>.json` | — |

**Usage examples:**

```bash
# Basic zero-shot clone using your default reference
python clone_voice.py --text "Hello, this is a cloned voice."

# Provide reference and transcript explicitly (fastest — no Whisper needed)
python clone_voice.py --reference ref.wav \
    --ref-text "This is what the reference says." \
    --text "Hello world."

# Use a fine-tuned checkpoint and name the output
python clone_voice.py --reference ref.wav \
    --text "Hello." \
    --checkpoint models/f5tts_finetune/ahmed/model_last.pt \
    --out-name ahmed_hello

# Load everything from a JSON config file
python clone_voice.py --config ahmed_greeting
```

---

### batch_clone.py

Batch voice cloning that generates every `TARGET_TEXT` in every `REFERENCE_AUDIO`'s voice. Loads the F5-TTS model once and reuses it for all combinations, which is much faster than calling `clone_voice.py` repeatedly.

Configure `REFERENCE_AUDIOS` and `TARGET_TEXTS` in `config.py` before running.

**CLI arguments:**

| Argument | Description | Default |
|---|---|---|
| `--nfe-steps INT` | Diffusion steps: 16 (fast) or 32 (quality) | 32 |
| `--seed INT` | RNG seed for reproducibility | 42 |
| `--checkpoint PATH` | Fine-tuned checkpoint (all voices use the same one) | `config.BATCH_CHECKPOINT` |
| `--dry-run` | Print what would be generated without running inference | — |

**Output structure:**

```
Output_results/
    ahmed/
        ahmed_greeting.wav
        ahmed_bible_open.wav
    church/
        church_greeting.wav
        church_bible_open.wav
```

**Usage examples:**

```bash
# Run with all defaults from config.py
python batch_clone.py

# Faster generation, slightly lower quality
python batch_clone.py --nfe-steps 16

# Preview the plan without running
python batch_clone.py --dry-run
```

---

### finetune_voice.py

Fine-tunes F5-TTS on your recorded voice clips. Reads a filelist manifest produced by `record_dataset.py`, builds an Apache Arrow dataset, then invokes the F5-TTS finetune CLI. Checkpoints are saved both inside Anaconda's internal directory and copied to `models/f5tts_finetune/<speaker>/`.

**Prerequisites:**

```bash
python record_dataset.py --speaker ahmed    # produces data/processed/ahmed_filelist.txt
python finetune_voice.py --speaker ahmed    # fine-tunes on those clips
```

**CLI arguments:**

| Argument | Description | Default |
|---|---|---|
| `--speaker NAME` | Speaker name (matches the filelist stem) | (required) |
| `--manifest PATH` | Path to filelist (default: `data/processed/<speaker>_filelist.txt`) | Auto |
| `--resume` | Resume from the latest checkpoint | — |
| `--epochs INT` | Training epochs | 100 |
| `--lr FLOAT` | Learning rate | 1e-5 |
| `--batch-size INT` | Clips per gradient step (reduce to 4 on OOM) | 4 |
| `--save-every INT` | Save checkpoint every N updates | 10 |
| `--warmup INT` | LR warmup steps | 10 |
| `--rebuild-dataset` | Force rebuild the Arrow dataset even if it exists | — |

**Usage examples:**

```bash
python finetune_voice.py --speaker ahmed
python finetune_voice.py --speaker ahmed --epochs 50 --lr 1e-5
python finetune_voice.py --speaker ahmed --resume
```

---

### compare_finetune.py

Sequentially fine-tunes F5-TTS on two contrasting datasets from the same speaker:
- `ahmed` — 98 short phonetic clips (~2–5 s each)
- `Ahmed_2` — 71 longer natural sentences (~7–9 s each)

Produces two independent checkpoints so you can compare the effect of clip length and prosodic variety on clone quality.

**CLI arguments:**

| Argument | Description | Default |
|---|---|---|
| `--epochs INT` | Epochs per dataset | 100 |
| `--batch-size INT` | Clips per gradient step | 6 |
| `--save-every INT` | Save checkpoint every N updates | 50 |
| `--warmup INT` | LR warmup steps | 10 |
| `--only CHOICE` | Run only `ahmed` or `Ahmed_2` | Both |
| `--resume` | Resume each run from its latest checkpoint | — |

**Usage examples:**

```bash
python compare_finetune.py
python compare_finetune.py --epochs 100 --batch-size 6
python compare_finetune.py --only ahmed
python compare_finetune.py --resume
```

---

### train_on.py

General-purpose training utility: pick any existing fine-tuned model as the weight initialisation and any recorded dataset as the training data. The step counter and optimizer state are stripped from the source checkpoint so training always runs the full requested number of epochs from a fresh start.

**CLI arguments:**

| Argument | Description | Default |
|---|---|---|
| `--list` | Show all available models and datasets then exit | — |
| `--model NAME` | Model name (folder under `models/f5tts_finetune/`) or path to `.pt` | (required) |
| `--dataset NAME` | Dataset name (from `data/processed/<name>_filelist.txt`) or path | (required) |
| `--output NAME` | Output folder name under `models/f5tts_finetune/` | `<model>_on_<dataset>` |
| `--epochs INT` | Training epochs | 100 |
| `--batch-size INT` | Clips per gradient step | 6 |
| `--save-every INT` | Save checkpoint every N updates | 50 |
| `--warmup INT` | LR warmup steps | 10 |

**Usage examples:**

```bash
python train_on.py --list
python train_on.py --model ahmed --dataset Ahmed_2
python train_on.py --model Ahmed_2 --dataset ahmed --epochs 50
python train_on.py --model ahmed --dataset Ahmed_2 --output my_experiment
```

---

### wer_graph.py

Measures Word Error Rate (WER) for every audio file in the `Cross_Examination` folder (or any folder you specify) and produces a colour-coded bar chart plus a CSV of raw scores.

For each clip:
1. Transcribe with faster-whisper.
2. Compare to the target text using jiwer (lowercase + strip punctuation normalisation).
3. WER = (substitutions + deletions + insertions) / reference word count.

**Output files** (written into the evaluated folder):
- `wer_results.png` — bar chart; green <= 15%, amber <= 30%, red > 30%
- `wer_results.csv` — per-clip scores with substitutions, deletions, insertions

**CLI arguments:**

| Argument | Description | Default |
|---|---|---|
| `--text TEXT` | The target text all clips are supposed to say | (required) |
| `--model CHOICE` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v2` | `base` |
| `--folder PATH` | Override the folder path | `config.cross_examination_folder` |

**Usage examples:**

```bash
python wer_graph.py --text "She sells seashells by the seashore"
python wer_graph.py --text "She sells seashells..." --model medium
python wer_graph.py --text "..." --folder path/to/other/folder
```

---

### blind_test.py

Copies every audio from `Cross_Examination` (or a specified folder) into a new `Test/` folder with randomised numerical filenames (1.wav, 2.wav, ...) so listeners can evaluate clips without knowing which model produced each one. A `key.txt` file is written so results can be revealed afterwards.

**CLI arguments:**

| Argument | Description | Default |
|---|---|---|
| `--folder PATH` | Source folder containing audio files | `config.cross_examination_folder` |
| `--out NAME` | Output folder name (created in project root) | `Test` |
| `--seed INT` | Random seed for reproducible shuffling | Random |

**Usage examples:**

```bash
python blind_test.py
python blind_test.py --folder Cross_Examination --out Test
python blind_test.py --seed 42
```

---

### create_single_audio.py

Interactive tool for recording a single continuous reference audio clip. Presents a script chooser (built-in phonetically rich default, custom text input, or load from file), then runs a record-and-review loop with retake support.

Saves to `data/references/<name>.wav` at 24 kHz (F5-TTS native rate) in 16-bit PCM format.

**CLI arguments:**

| Argument | Description | Default |
|---|---|---|
| `--name NAME` | Output filename stem, e.g. `ahmed_reference` | (required) |
| `--device INT` | Microphone device index | System default |
| `--list-devices` | Print all available input devices and exit | — |

**Usage examples:**

```bash
python create_single_audio.py --name ahmed_reference
python create_single_audio.py --name ahmed_reference --list-devices
python create_single_audio.py --name ahmed_reference --device 1
```

---

### resembler metric.py

Computes pairwise speaker similarity between every audio file in `Cross_Examination` using Resemblyzer d-vector embeddings (256-dimensional cosine similarity). Pairs scoring at or above `config.Resemble_threshold` are flagged with a red box and a star in the heatmap.

**No CLI arguments** — reads `config.cross_examination_folder` and `config.Resemble_threshold` directly.

**Output files** (written into `Cross_Examination/`):
- `speaker_similarity_heatmap.png` — annotated NxN heatmap
- `speaker_similarity_matrix.csv` — full NxN raw cosine similarity scores

**Run:**

```bash
python "resembler metric.py"
```

---

### train_model.py

Continues pre-training the F5-TTS base model on downloaded LibriTTS data. This is large-scale "continued pre-training" (not speaker fine-tuning) — the goal is to improve the model's general English TTS quality before speaker adaptation.

**Prerequisites:**
1. `python download_mls.py` — download ~220 h of LibriTTS audio
2. `python convert_manifest.py` — produce `data/processed/metadata.csv`
3. `python prepare_dataset.py` — build the Arrow dataset (~2–4 min for 119 k clips)

**CLI arguments:**

| Argument | Description | Default |
|---|---|---|
| `--resume` | Auto-detect and resume from latest checkpoint | — |
| `--epochs INT` | Training epochs (~12–24 h per epoch on RTX 3080) | 3 |
| `--lr FLOAT` | Learning rate (lower than scratch to avoid forgetting) | 5e-6 |
| `--batch-size INT` | Clips per GPU step (reduce on OOM) | 2 |
| `--save-every INT` | Save a checkpoint every N gradient updates | 1000 |

**Usage examples:**

```bash
python train_model.py
python train_model.py --resume
python train_model.py --epochs 5 --lr 5e-6 --batch-size 2
```

---

### cross_train.py

Cross-trains the two fine-tuned voice models on each other's datasets to investigate distribution transfer:
- `Ahmed_1_to_2`: ahmed model weights (short clips) continued on Ahmed_2 data (longer sentences)
- `Ahmed_2_to_1`: Ahmed_2 model weights (longer sentences) continued on ahmed data (short clips)

Requires both base models to exist in `models/f5tts_finetune/` (run `compare_finetune.py` first).

**CLI arguments:**

| Argument | Description | Default |
|---|---|---|
| `--epochs INT` | Training epochs per run | 100 |
| `--batch-size INT` | Clips per gradient step | 6 |
| `--save-every INT` | Save checkpoint every N updates | 50 |
| `--warmup INT` | LR warmup steps | 10 |
| `--only CHOICE` | Run only `Ahmed_1_to_2` or `Ahmed_2_to_1` | Both |
| `--resume` | Resume each run from its latest checkpoint | — |

**Usage examples:**

```bash
python cross_train.py
python cross_train.py --epochs 100 --batch-size 6
python cross_train.py --only Ahmed_1_to_2
python cross_train.py --resume
```

---

## Workflow

Recommended order of operations from raw recording to final evaluation.

### 1. Record reference audio

```bash
python create_single_audio.py --name ahmed_reference
# Output: data/references/ahmed_reference.wav
```

### 2. Record dataset for fine-tuning

```bash
python record_dataset.py --speaker ahmed
# Output: data/processed/ahmed_filelist.txt  +  per-clip WAVs
```

### 3. Fine-tune the model

Single dataset:

```bash
python finetune_voice.py --speaker ahmed
# Output: models/f5tts_finetune/ahmed/model_last.pt
```

Comparison experiment (both datasets in sequence):

```bash
python compare_finetune.py
# Output: models/f5tts_finetune/ahmed/  and  models/f5tts_finetune/Ahmed_2/
```

Cross-training (requires compare_finetune.py to have completed):

```bash
python cross_train.py
# Output: models/f5tts_finetune/Ahmed_1_to_2/  and  models/f5tts_finetune/Ahmed_2_to_1/
```

### 4. Clone voice

Zero-shot (no fine-tuning needed):

```bash
python clone_voice.py \
    --reference data/references/ahmed_reference.wav \
    --text "She sells seashells by the seashore."
```

With fine-tuned checkpoint:

```bash
python clone_voice.py \
    --reference data/references/ahmed_reference.wav \
    --text "She sells seashells by the seashore." \
    --checkpoint models/f5tts_finetune/ahmed/model_last.pt \
    --out-name ahmed_seashells
```

Batch all voices x all texts:

```bash
python batch_clone.py
```

### 5. Evaluate

Place all clips to compare into `Cross_Examination/`, then run:

```bash
# Speaker similarity heatmap (Resemblyzer)
python "resembler metric.py"

# Word Error Rate bar chart
python wer_graph.py --text "She sells seashells by the seashore"

# Blind listening test (randomised filenames)
python blind_test.py
```

---

## Voice_configuration JSON format

JSON files placed in `Voice_configuration/` store all settings for a specific cloning job. Pass the file stem to `clone_voice.py --config <name>` to load it. CLI flags always override JSON values.

**All keys are optional except `text`.**

```json
{
    "reference":   "data/references/ahmed_reference.wav",
    "ref_text":    "The exact words spoken in the reference clip.",
    "text":        "The text to synthesise in the cloned voice.",
    "checkpoint":  "models/f5tts_finetune/ahmed/model_last.pt",
    "out_name":    "ahmed_greeting",
    "out":         "output/ahmed_greeting_v2.wav",
    "speed":       1.0,
    "seed":        42,
    "nfe_steps":   32
}
```

**Key descriptions:**

| Key | Type | Description |
|---|---|---|
| `reference` | string | Path to reference WAV/MP3 (5–30 s). Relative paths are resolved from the project root. |
| `ref_text` | string | Transcript of the reference clip. Omit to auto-transcribe with Whisper (slower). |
| `text` | string | Text to synthesise. Multi-sentence texts are split at `.!?` boundaries and stitched. |
| `checkpoint` | string | Path to a fine-tuned `.pt` checkpoint. Omit to use the pretrained base model. |
| `out_name` | string | Output filename stem. Saves to `output/<out_name>.wav`. Auto-numbered if the file exists. |
| `out` | string | Full output path override. Takes priority over `out_name`. |
| `speed` | float | Speech speed multiplier. 0.9 = slower, 1.1 = faster. Default 1.0. |
| `seed` | int | RNG seed for reproducible generation. -1 = random. Default -1. |
| `nfe_steps` | int | Number of diffusion steps. 16 = fast, 32 = best quality. Default 32. |

# Tacotron 2
For the tacotron2 model, we used [Nvidia's implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2#quick-start-guide). It is found in DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2.

## Training
For training, we ran `python train.py -m Tacotron2 -o <output_file> -lr 1e-4 --epochs 476 -bs 4 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json --epochs-per-checkpoint 25 --freeze --checkpoint-path <path\to\checkpoint> --training-files=path/to/training-files --validation-files=path/to/validation-files --dataset-path path\to\dataset`. 

`--checkpoint-path` is used to select a starting point to begin training from. We used `nvidia_tacotron2pyt_fp32_20190427.pt` as starting point, which was downloaded from https://catalog.ngc.nvidia.com/orgs/nvidia/models/tacotron2pyt_fp32?version=2.

We added the flag `--freeze`. It can be included to freeze training on the embedding and encoder layers.

## Inference
For generating an audio using a tacotron 2 model, we ran `python inference.py --tacotron2 .\path\to\taacotron2model --waveglow .\path\to\waveglowmodel --wn-channels 256 -o <output_file> -i .\path\to\input-text`.

`--waveglow` specifies which waveglow model to use. We used `waveglow_1076430_14000_amp.pt`, which was downloaded from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/adlr/models/waveglow?version=WaveGlow-LJS_256_Channels.

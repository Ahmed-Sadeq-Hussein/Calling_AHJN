# CALLING AHJIN — Brought to you by Ahmed Hussein and Julius Norén from JTH.
# NOTE : a lot of what was written in config is not relevant due to us chifting to a Json config file system. 
from pathlib import Path

# ── Project root (everything is relative to this file) ────────────────────────
# __file__ resolves to config.py; using .resolve().parent gives the absolute
# directory regardless of where Python is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Resemblyzer settings (do not modify) ──────────────────────────────────────
# Resemblyzer produces 256-d d-vector embeddings from raw audio.
# These are compared with cosine similarity (0 = unrelated, 1 = same speaker).

# Path to the folder containing the training-speaker's WAV files.
training_voice = ""
# List of paths to folders containing target speakers' WAV files for comparison.
target_voices = []
# Name of the experiment sub-folder — used to organise outputs per run.
experiment_name = "Experiment_1"
# Metadata tag embedded in the training-voice recording to track provenance.
training_voice_metadata = "Training_Voice" + experiment_name

# Folder scanned by resembler metric.py, wer_graph.py and blind_test.py.
cross_examination_folder = "Cross_Examination"
# Cosine similarity threshold: pairs scoring at or above this are flagged
# as "same speaker" in the heatmap output of resembler metric.py.
Resemble_threshold = 0.75

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR           = PROJECT_ROOT / "models"
DATA_DIR             = PROJECT_ROOT / "data"
MLS_CACHE_DIR        = DATA_DIR / "mls_cache"      # HuggingFace streaming cache for MLS downloads
PROCESSED_DATA_DIR   = DATA_DIR / "processed"      # filelist .txt files produced by record_dataset.py

# ── MLS (Multilingual LibriSpeech) / LibriTTS dataset settings ────────────────
# ISO codes: EN=LibriTTS, DE=german, FR=french, NL=dutch,
#            ES=spanish, IT=italian, PT=portuguese, PL=polish
MLS_LANGUAGES = ["EN"]

# Max hours per language (None = all available). 500h English ≈ ~25 GB WAV.
MLS_MAX_HOURS = 500

# Duration filter (seconds). LibriTTS/MLS clips vary widely.
MLS_MIN_DURATION_SEC = 2.0
MLS_MAX_DURATION_SEC = 20.0

# ── Audio settings ─────────────────────────────────────────────────────────────
# Used by download_mls.py and prepare_voice.py to write WAV files.
SAMPLE_RATE = 22050   # storage sample rate for downloaded clips

# F5-TTS natively operates at 24 kHz.  The model resamples internally during
# inference, but convert_manifest.py can write 24 kHz copies via --resample.
F5TTS_SAMPLE_RATE = 24000

# ── Speaker embedding settings ─────────────────────────────────────────────────
# Resemblyzer d-vectors (256-d) — used by resembler metric.py
SPEAKER_EMBED_DIM = 256

# ── F5-TTS model settings ──────────────────────────────────────────────────────
# Model variant: "F5-TTS" (default, recommended) or "E2-TTS"
F5TTS_MODEL_TYPE = "F5-TTS"

# Where fine-tuned checkpoints are saved  (one sub-folder per speaker)
F5TTS_FINETUNE_DIR = MODELS_DIR / "f5tts_finetune"

# Path to a specific fine-tuned checkpoint to use at inference.
# Leave empty to use the pretrained model (auto-downloaded from HuggingFace).
F5TTS_CKPT_PATH = ""

# ── F5-TTS fine-tuning hyperparameters ────────────────────────────────────────
# These defaults are used by finetune_voice.py when no CLI flags are given.
# With ~30 min of voice recording on an RTX 3080:
#   - 10 epochs ≈ 10-20 min training  → good quality
#   - 20 epochs ≈ 20-40 min training  → very good quality (risk of overfitting)
F5TTS_FINETUNE_EPOCHS      = 10
F5TTS_FINETUNE_LR          = 1e-5       # lower than pretraining to avoid catastrophic forgetting
F5TTS_FINETUNE_BATCH_SIZE  = 8          # clips per GPU step; reduce to 4 if OOM
F5TTS_FINETUNE_SAVE_EVERY  = 300        # save checkpoint every N gradient updates
F5TTS_FINETUNE_WARMUP_STEPS = 10        # short warmup appropriate for fine-tuning (not scratch)

# ── Inference / voice cloning ──────────────────────────────────────────────────
# Path to a reference audio clip for zero-shot cloning (5–30 seconds ideal).
# Set this once and clone_voice.py will use it as the default --reference.
# ── Output (synthesised audio — gitignored via /output/) ──────────────────────
OUTPUT_DIR = PROJECT_ROOT / "output"

CLONE_REFERENCE_AUDIO = str(OUTPUT_DIR / "recording.wav")  # set once, reuse forever

# ── Batch voice cloning (used by batch_clone.py) ──────────────────────────────
#
# REFERENCE_AUDIOS  — voices to clone from
#   Format:  "label": ("path/to/audio", "exact transcript of that audio")
#   Tips:
#     - Path can be relative to project root or absolute
#     - Leave transcript as "" to auto-transcribe with Whisper (slower)
#     - Label becomes part of the output filename
#
REFERENCE_AUDIOS: dict[str, tuple[str, str]] = {
    # "ahmed":   ("data/references/ahmed_reference.wav", ""),
    # "church":  ("Cross_Examination/Church world tour.wav",
    #             "South Africa, Australia, New Zealand, Hawaii and Canada, "
    #             "all in seven weeks following Agatha Christie on her world tour to promote the"),
    # "ash":     ("Cross_Examination/Ash see shells.mp3",
    #             "She sells sea shells by the sea shore."),
}

# TARGET_TEXTS — what each reference voice should say
#   Format:  "label": "text to synthesise"
#   Tips:
#     - Label becomes part of the output filename
#     - Sentences are auto-split for clean generation
#
TARGET_TEXTS: dict[str, str] = {
    # "greeting":     "Hi, I am your personal AI assistant.",
    # "bible_open":   "In the beginning God created the heaven and the earth. "
    #                 "And the earth was without form, and void.",
    # "intro":        "Welcome. My name is Ashy Washy and I am here to help you.",
}

# Optional: path to a fine-tuned checkpoint to use for all generations.
# Leave as "" to use the pretrained base model.
BATCH_CHECKPOINT: str = ""

# Output folder — one sub-folder per reference label
BATCH_OUTPUT_DIR = PROJECT_ROOT / "Output_results"

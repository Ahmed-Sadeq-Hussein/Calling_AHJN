from pathlib import Path

# ── Project root (everything is relative to this file) ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Resemblyzer settings (do not modify) ──────────────────────────────────────
# Here you insert your Path to the folder of the training voice
training_voice = ""
# Here you insert your Path to the folder of the target voices.
target_voices = []
# Here we write the name of the folder where the run experiment is meant to be made inside.
experiment_name = "Experiment_1"
# Here we insert metadata into the recording in order to identify the voice.
training_voice_metadata = "Training_Voice" + experiment_name

# Resemblyzer global variables
cross_examination_folder = "Cross_Examination"
Resemble_threshold = 0.75  # Cosine score at/above which speakers are judged the same.

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR           = PROJECT_ROOT / "models"
DATA_DIR             = PROJECT_ROOT / "data"
MLS_CACHE_DIR        = DATA_DIR / "mls_cache"      # HuggingFace streaming cache
PROCESSED_DATA_DIR   = DATA_DIR / "processed"      # downloaded + preprocessed wav files

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
SAMPLE_RATE = 22050   # download/clip storage rate

# F5-TTS native sample rate — the model resamples internally, but convert_manifest.py
# can optionally write 24 kHz copies when --resample is passed.
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
# These are used by finetune_voice.py.
# With ~30 min of voice recording on an RTX 3080:
#   - 10 epochs ≈ 10-20 min training  → good quality
#   - 20 epochs ≈ 20-40 min training  → very good quality (risk of overfitting)
F5TTS_FINETUNE_EPOCHS      = 10
F5TTS_FINETUNE_LR          = 1e-5       # learning rate (lower than pretraining)
F5TTS_FINETUNE_BATCH_SIZE  = 8          # batch size per GPU; reduce to 4 on OOM
F5TTS_FINETUNE_SAVE_EVERY  = 300        # save checkpoint every N gradient updates
F5TTS_FINETUNE_WARMUP_STEPS = 10        # LR warmup steps (short for fine-tuning)

# ── Inference / voice cloning ──────────────────────────────────────────────────
# Path to a reference audio clip for zero-shot cloning (5–30 seconds ideal).
# Set this once and clone_voice.py will use it as the default --reference.
# ── Output (synthesised audio — gitignored via /output/) ──────────────────────
OUTPUT_DIR = PROJECT_ROOT / "output"

CLONE_REFERENCE_AUDIO = str(OUTPUT_DIR / "recording.wav")  # set once, reuse forever

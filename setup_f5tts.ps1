# setup_f5tts.ps1
# ----------------
# Installs F5-TTS and its dependencies into the current conda/venv environment.
# Requires PyTorch with CUDA already installed (run after verifying GPU works).
#
# Usage:
#   .\setup_f5tts.ps1

$OriginalDir = $PSScriptRoot

Write-Host ""
Write-Host "=== F5-TTS Setup ===" -ForegroundColor Cyan
Write-Host ""

# 1. Verify PyTorch + CUDA -------------------------------------------------------
Write-Host "Checking PyTorch + CUDA..." -ForegroundColor Yellow
python -c @"
import torch
ok = torch.cuda.is_available()
print('CUDA OK' if ok else 'NO CUDA - CPU only')
if ok:
    print('  Device:', torch.cuda.get_device_name(0))
    print('  VRAM  :', torch.cuda.get_device_properties(0).total_memory // 1024**2, 'MB')
"@
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyTorch import failed. Install CUDA PyTorch first:" -ForegroundColor Red
    Write-Host "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    Set-Location $OriginalDir; exit 1
}

# 2. Install F5-TTS --------------------------------------------------------------
Write-Host ""
Write-Host "Installing F5-TTS..." -ForegroundColor Yellow
pip install f5-tts
if ($LASTEXITCODE -ne 0) {
    Write-Host "f5-tts install failed." -ForegroundColor Red
    Set-Location $OriginalDir; exit 1
}

# 3. Install extra dependencies used by our scripts ------------------------------
Write-Host ""
Write-Host "Installing supporting packages..." -ForegroundColor Yellow
pip install soundfile librosa tqdm faster-whisper resemblyzer

# 4. Verify F5-TTS import --------------------------------------------------------
Write-Host ""
Write-Host "Verifying F5-TTS import..." -ForegroundColor Yellow
python -c @"
from f5_tts.api import F5TTS
print('  f5_tts.api.F5TTS  ... OK')
import soundfile, librosa, tqdm
print('  soundfile / librosa / tqdm  ... OK')
try:
    from resemblyzer import VoiceEncoder
    print('  resemblyzer  ... OK')
except Exception as e:
    print('  resemblyzer  ... WARNING:', e)
"@
if ($LASTEXITCODE -ne 0) {
    Write-Host "F5-TTS import failed - check error above." -ForegroundColor Red
    Set-Location $OriginalDir; exit 1
}

# 5. Create required directories -------------------------------------------------
$dirs = @(
    "models\f5tts_finetune",
    "data\processed",
    "output"
)
foreach ($d in $dirs) {
    $full = Join-Path $OriginalDir $d
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Force $full | Out-Null
        Write-Host "  Created $d" -ForegroundColor DarkGray
    }
}

# 6. Done ------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Wait for download_mls.py to finish  (or kill it once you have enough data)"
Write-Host "  2. python convert_manifest.py           (converts LibriTTS manifest to F5-TTS format)"
Write-Host "  3. python clone_voice.py --reference ref.wav --text 'Hello world'"
Write-Host "     (zero-shot cloning - no training needed, uses F5-TTS pretrained model)"
Write-Host ""
Write-Host "  To fine-tune on YOUR voice:"
Write-Host "  4. python prepare_voice.py --audio your_recording.wav --speaker ahmed"
Write-Host "  5. python finetune_voice.py --speaker ahmed"
Write-Host ""

Set-Location $OriginalDir

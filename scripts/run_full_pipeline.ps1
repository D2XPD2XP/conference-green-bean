<#
.SYNOPSIS
    Pipeline lengkap: crop -> augment -> latih 3 CNN -> evaluasi -> analisis.

.DESCRIPTION
    Versi PowerShell dari scripts/run_full_pipeline.sh, untuk dipakai langsung
    di terminal Windows tanpa Git Bash.

.PARAMETER Clean
    Kosongkan data/crops sebelum augmentasi. WAJIB dipakai setelah ganti versi
    dataset, agar crop dari foto lama tidak tertinggal dan mencemari split.

.PARAMETER Epochs
    Override jumlah epoch untuk ketiga model (default: dari configs/cnn_config.py).

.PARAMETER Strategy
    Strategi penanganan imbalance: none, class_weights, weighted_sampler, focal_loss.

.PARAMETER Models
    Daftar model yang dilatih (default: ketiganya).

.PARAMETER SkipCrop
    Lewati tahap crop (anotasi tidak berubah).

.PARAMETER SkipTrain
    Lewati pelatihan; hanya evaluasi + analisis ulang dari checkpoint yang ada.

.EXAMPLE
    .\scripts\run_full_pipeline.ps1

.EXAMPLE
    .\scripts\run_full_pipeline.ps1 -Clean -Strategy focal_loss

.EXAMPLE
    .\scripts\run_full_pipeline.ps1 -Epochs 2 -Models mobilenetv3
#>

[CmdletBinding()]
param(
    [switch]$Clean,
    [int]$Epochs = 0,
    [ValidateSet("none", "class_weights", "weighted_sampler", "focal_loss")]
    [string]$Strategy = "class_weights",
    [string[]]$Models = @("resnet50", "mobilenetv3", "inceptionv3"),
    [switch]$SkipCrop,
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Pilih interpreter: venv lebih dulu, baru python dari PATH.
$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
    Write-Warning "venv tidak ditemukan, memakai 'python' dari PATH."
}

function Invoke-Step {
    param([string]$Label, [string[]]$Arguments)

    Write-Host ""
    Write-Host "--- $Label ---" -ForegroundColor Cyan
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label gagal (exit code $LASTEXITCODE)."
    }
}

Write-Host "==============================================================" -ForegroundColor Green
Write-Host " Pipeline green bean - 5 kelas"
Write-Host " python   : $py"
Write-Host " model    : $($Models -join ', ')"
Write-Host " strategi : $Strategy"
if ($Epochs -gt 0) { Write-Host " epochs   : $Epochs" } else { Write-Host " epochs   : default dari config" }
Write-Host "==============================================================" -ForegroundColor Green

Invoke-Step "[0/5] Environment: torch" @("-c", "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())")
Invoke-Step "[0/5] Environment: konfigurasi kelas" @("-c", "from configs.base_config import CLASS_NAMES, NUM_CLASSES; print('kelas:', CLASS_NAMES, '| n =', NUM_CLASSES)")
Invoke-Step "[1/5] Validasi dataset anotasi" @(".claude/skills/check-dataset/validate.py")

if ($SkipCrop) {
    Write-Host "`n[2/5] Crop - DILEWATI (-SkipCrop)" -ForegroundColor Yellow
}
else {
    Invoke-Step "[2/5] Crop biji dari anotasi -> data/crops_raw" @("-m", "scripts.crop_dataset")
}

$augmentArgs = @("-m", "scripts.augment_dataset")
if ($Clean) { $augmentArgs += "--clean" }
Invoke-Step "[3/5] Augmentasi offline -> data/crops" $augmentArgs

if ($SkipTrain) {
    Write-Host "`n[4/5] Pelatihan - DILEWATI (-SkipTrain)" -ForegroundColor Yellow
}
else {
    foreach ($model in $Models) {
        $trainArgs = @("-m", "scripts.train", "--model", $model, "--imbalance-strategy", $Strategy)
        if ($Epochs -gt 0) { $trainArgs += @("--epochs", "$Epochs") }
        Invoke-Step "[4/5] Latih $model" $trainArgs
    }
}

foreach ($model in $Models) {
    $folder = "results\models\$model"
    if (-not (Test-Path $folder)) {
        Write-Warning "Folder $folder tidak ada; evaluasi $model dilewati."
        continue
    }
    $checkpoint = Get-ChildItem -Path $folder -Filter *.pt -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $checkpoint) {
        Write-Warning "Checkpoint $model tidak ditemukan; evaluasi dilewati."
        continue
    }
    Invoke-Step "[5/5] Evaluasi $model ($($checkpoint.Name))" @(
        "-m", "scripts.evaluate", "--model", $model, "--checkpoint", $checkpoint.FullName
    )
}

Invoke-Step "[5/5] Analisis lengkap" @("-m", "scripts.analyze")

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host " Pipeline selesai."
Write-Host "   metrik  : results/metrics/"
Write-Host "   figur   : results/figures/"
Write-Host "   tabel   : results/analysis/"
Write-Host "   log     : results/logs/"
Write-Host "==============================================================" -ForegroundColor Green

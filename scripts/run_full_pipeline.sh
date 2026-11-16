#!/usr/bin/env bash
# Pipeline lengkap: crop -> augment -> latih 3 CNN -> evaluasi -> analisis.
#
# Pakai dari root proyek:
#   bash scripts/run_full_pipeline.sh
#   CLEAN=1 bash scripts/run_full_pipeline.sh          # kosongkan data/crops dulu
#   EPOCHS=2 bash scripts/run_full_pipeline.sh         # uji cepat
#   STRATEGY=focal_loss bash scripts/run_full_pipeline.sh
#
# Variabel lingkungan:
#   CLEAN=1       kosongkan data/crops sebelum augmentasi (WAJIB saat ganti dataset)
#   EPOCHS=<n>    override jumlah epoch ketiga model (default: config, 30)
#   STRATEGY=<s>  strategi imbalance: none|class_weights|weighted_sampler|focal_loss
#   MODELS="a b"  daftar model yang dilatih (default: ketiganya)
#   SKIP_CROP=1   lewati tahap crop (anotasi tidak berubah)
#   SKIP_TRAIN=1  lewati pelatihan (hanya evaluasi + analisis ulang)

set -euo pipefail

cd "$(dirname "$0")/.."

# Pilih interpreter: venv lebih dulu, baru python sistem.
if [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
  echo "PERINGATAN: venv tidak ditemukan, memakai '$PY' dari PATH."
fi

MODELS="${MODELS:-resnet50 mobilenetv3 inceptionv3}"
STRATEGY="${STRATEGY:-class_weights}"
EPOCH_ARG=""
if [[ -n "${EPOCHS:-}" ]]; then
  EPOCH_ARG="--epochs ${EPOCHS}"
fi

echo "=============================================================="
echo " Pipeline green bean - 5 kelas"
echo " python   : $PY"
echo " model    : $MODELS"
echo " strategi : $STRATEGY"
echo " epochs   : ${EPOCHS:-default dari config}"
echo "=============================================================="

echo ""
echo "[0/5] Environment"
$PY -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
$PY -c "from configs.base_config import CLASS_NAMES, NUM_CLASSES; print('kelas:', CLASS_NAMES, '| n =', NUM_CLASSES)"

echo ""
echo "[1/5] Validasi dataset anotasi"
$PY .claude/skills/check-dataset/validate.py

if [[ "${SKIP_CROP:-0}" == "1" ]]; then
  echo ""
  echo "[2/5] Crop  - DILEWATI (SKIP_CROP=1)"
else
  echo ""
  echo "[2/5] Crop biji dari anotasi -> data/crops_raw"
  $PY -m scripts.crop_dataset
fi

echo ""
echo "[3/5] Augmentasi offline -> data/crops"
if [[ "${CLEAN:-0}" == "1" ]]; then
  $PY -m scripts.augment_dataset --clean
else
  $PY -m scripts.augment_dataset
fi

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo ""
  echo "[4/5] Pelatihan - DILEWATI (SKIP_TRAIN=1)"
else
  echo ""
  echo "[4/5] Pelatihan ${MODELS}"
  for model in $MODELS; do
    echo ""
    echo "--- latih $model ---"
    # shellcheck disable=SC2086
    $PY -m scripts.train --model "$model" --imbalance-strategy "$STRATEGY" $EPOCH_ARG
  done
fi

echo ""
echo "[5/5] Evaluasi + analisis"
for model in $MODELS; do
  checkpoint=$(ls -t "results/models/$model"/*.pt 2>/dev/null | head -n 1 || true)
  if [[ -z "$checkpoint" ]]; then
    echo "PERINGATAN: checkpoint $model tidak ditemukan; evaluasi dilewati."
    continue
  fi
  echo ""
  echo "--- evaluasi $model ($(basename "$checkpoint")) ---"
  $PY -m scripts.evaluate --model "$model" --checkpoint "$checkpoint"
done

echo ""
$PY -m scripts.analyze

echo ""
echo "=============================================================="
echo " Pipeline selesai."
echo "   metrik  : results/metrics/"
echo "   figur   : results/figures/"
echo "   tabel   : results/analysis/"
echo "   log     : results/logs/"
echo "=============================================================="

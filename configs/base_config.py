"""Konstanta global proyek klasifikasi kualitas green bean kopi.

Pipeline::

    foto multi-bean -> anotasi Roboflow (5 kelas) -> crop offline (crops_raw)
    -> augmentasi offline (crops) -> perbandingan 3 CNN + penanganan imbalance
    -> analisis kaya (Grad-CAM, confusion, confidence, error, t-SNE, efisiensi)

Modul ini menjadi sumber tunggal (single source of truth) untuk lokasi
direktori dan pengaturan yang dipakai lintas komponen. Gunakan konstanta
dari sini alih-alih menuliskan path atau nama kelas secara literal di modul
lain.
"""

from __future__ import annotations

from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Path
# --------------------------------------------------------------------------- #
#: Root proyek (satu level di atas folder configs/).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Direktori dataset (raw, annotated, crops_raw, crops).
DATA_DIR: Path = PROJECT_ROOT / "data"

#: Foto multi-bean mentah, belum dianotasi.
RAW_DATA_DIR: Path = DATA_DIR / "raw"

#: Ekspor Roboflow format YOLO (train/valid/test berisi images/ dan labels/).
ANNOTATED_DATA_DIR: Path = DATA_DIR / "annotated"

#: Crop biji individual hasil ``scripts.crop_dataset`` — **tanpa augmentasi**.
#: Ini artefak mentah yang tidak pernah dipakai langsung untuk pelatihan.
CROPS_RAW_DIR: Path = DATA_DIR / "crops_raw"

#: Dataset siap latih hasil ``scripts.augment_dataset`` (train diaugmentasi
#: offline; valid & test hanya disalin apa adanya).
CROPS_DIR: Path = DATA_DIR / "crops"

#: Alias eksplisit agar pembacaan kode jelas: ``crops`` = pasca-augmentasi.
CROPS_AUGMENTED_DIR: Path = CROPS_DIR

#: Direktori keluaran eksperimen (models, figures, metrics, logs, analysis).
RESULTS_DIR: Path = PROJECT_ROOT / "results"

#: Direktori bobot model terlatih.
MODELS_DIR: Path = RESULTS_DIR / "models"

#: Direktori figur (kurva loss, confusion matrix, Grad-CAM, t-SNE).
FIGURES_DIR: Path = RESULTS_DIR / "figures"

#: Direktori metrik (CSV/JSON).
METRICS_DIR: Path = RESULTS_DIR / "metrics"

#: Direktori log pelatihan.
LOGS_DIR: Path = RESULTS_DIR / "logs"

#: Direktori artefak analisis mentah (prediksi CSV, feature vector .npz).
ANALYSIS_DIR: Path = RESULTS_DIR / "analysis"

# --------------------------------------------------------------------------- #
# Reproducibility & device
# --------------------------------------------------------------------------- #
#: Seed global untuk seluruh operasi acak (torch, numpy, random, albumentations).
RANDOM_SEED: int = 42

#: Device komputasi; gunakan GPU bila tersedia.
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
# Kelas klasifikasi (5 kelas granular)
# --------------------------------------------------------------------------- #
#: Nama kelas dalam kode & nama folder. **Urutan menentukan indeks label**,
#: yaitu ``class_id`` pada label YOLO hasil ekspor Roboflow. Jangan diubah
#: urutannya tanpa mengekspor ulang dataset.
CLASS_NAMES: list[str] = ["good", "black", "chipped", "dry_pods", "insect_damage"]

#: Jumlah kelas klasifikasi (diturunkan, tidak pernah di-hardcode).
NUM_CLASSES: int = len(CLASS_NAMES)

#: Kelas non-cacat; sisanya dianggap cacat. Dipakai untuk metrik agregat
#: "layak vs cacat" pada pelaporan quality control.
GOOD_CLASS: str = "good"

#: Kelas cacat (semua kelas selain ``GOOD_CLASS``).
DEFECT_CLASSES: list[str] = [name for name in CLASS_NAMES if name != GOOD_CLASS]

#: Padanan bahasa Indonesia untuk pelaporan/figur (kode tetap bahasa Inggris).
CLASS_NAMES_ID: dict[str, str] = {
    "good": "Baik",
    "black": "Hitam",
    "chipped": "Serpihan",
    "dry_pods": "Polong Kering",
    "insect_damage": "Rusak Serangga",
}

#: Definisi tiap kelas menurut kebijakan quality control.
CLASS_DESCRIPTIONS: dict[str, str] = {
    "good": "Biji utuh tanpa cacat; diterima quality control",
    "black": "Biji menghitam akibat fermentasi berlebih atau busuk",
    "chipped": "Biji pecah/terkelupas sehingga bentuknya tidak utuh",
    "dry_pods": "Polong/kulit kering yang terbawa saat pengolahan",
    "insect_damage": "Biji berlubang atau tergerek serangga",
}

#: Warna per kelas untuk seluruh visualisasi (konsisten antar-figur).
CLASS_COLORS: dict[str, str] = {
    "good": "#2ecc71",  # hijau
    "black": "#2c3e50",  # hitam
    "chipped": "#f39c12",  # oranye
    "dry_pods": "#e74c3c",  # merah
    "insect_damage": "#9b59b6",  # ungu
}

# --------------------------------------------------------------------------- #
# Penanganan ketimpangan kelas (imbalance)
# --------------------------------------------------------------------------- #
#: Strategi penanganan imbalance yang dipakai saat pelatihan. Pilihan:
#:
#: - ``"none"``             : CrossEntropyLoss biasa.
#: - ``"class_weights"``    : CrossEntropyLoss dengan bobot per kelas.
#: - ``"weighted_sampler"`` : ``WeightedRandomSampler`` di DataLoader.
#: - ``"focal_loss"``       : Focal Loss (Lin et al., ICCV 2017).
#:
#: Dapat di-override dari CLI: ``--imbalance-strategy``.
IMBALANCE_STRATEGY: str = "class_weights"

#: Daftar strategi yang valid (dipakai untuk validasi argumen CLI).
IMBALANCE_STRATEGIES: tuple[str, ...] = (
    "none",
    "class_weights",
    "weighted_sampler",
    "focal_loss",
)

#: Kompatibilitas mundur: ``True`` ekuivalen dengan ``IMBALANCE_STRATEGY``
#: bernilai ``"class_weights"``.
USE_CLASS_WEIGHTS: bool = IMBALANCE_STRATEGY == "class_weights"

#: Skema perhitungan bobot kelas (lihat ``src.training.imbalance``).
CLASS_WEIGHTS_STRATEGY: str = "balanced"

#: Parameter gamma default untuk Focal Loss.
FOCAL_LOSS_GAMMA: float = 2.0

# --------------------------------------------------------------------------- #
# Pemilihan checkpoint & BatchNorm (revisi reviewer, 2026-11)
# --------------------------------------------------------------------------- #
#: Metrik VALID untuk memilih checkpoint terbaik. ``"val_f1"`` = F1 makro
#: (default sejak revisi R2.2, karena val didominasi ``good`` sehingga akurasi
#: bias ke kelas termudah). ``"val_acc"`` = kriteria lama (run 2026-10-15).
#: Seri -> epoch lebih awal yang dipertahankan.
CHECKPOINT_METRIC: str = "val_f1"

#: Pilihan metrik checkpoint yang valid.
CHECKPOINT_METRICS: tuple[str, ...] = ("val_f1", "val_acc")

#: Perlakuan BatchNorm backbone saat training. ``"adapt"`` = perilaku bawaan
#: ``model.train()``: bobot beku, tetapi ``running_mean/var`` ikut diperbarui
#: (konfigurasi utama). ``"frozen"`` = modul BN tetap mode eval, sehingga
#: statistik ImageNet tidak berubah (ablasi R1.7).
BN_MODE: str = "adapt"

#: Pilihan mode BatchNorm yang valid.
BN_MODES: tuple[str, ...] = ("adapt", "frozen")

# --------------------------------------------------------------------------- #
# Test-time augmentation (evaluasi)
# --------------------------------------------------------------------------- #

#: Mode TTA saat evaluasi (lihat ``src.evaluation.tta``). ``"d4"`` = rata-rata
#: logit dari 8 orientasi (flip x rotasi 90 derajat). Dipilih pada split VALID
#: (2026-10-15): terbaik untuk ketiga arsitektur. ``"none"`` = prediksi tunggal.
TTA_MODE: str = "d4"

# --------------------------------------------------------------------------- #
# Augmentasi offline
# --------------------------------------------------------------------------- #
#: Jumlah total sampel keluaran per citra asli, **per kelas**, saat augmentasi
#: offline. Nilai ``1`` berarti citra asli disalin tanpa varian tambahan;
#: nilai ``3`` berarti 1 citra asli + 2 varian teraugmentasi.
#:
#: Kelas mayoritas (``good``) sengaja tidak di-oversample agar distribusi
#: train mendekati seimbang.
AUGMENTATION_MULTIPLIERS: dict[str, int] = {
    "good": 1,
    "black": 3,
    "chipped": 3,
    "dry_pods": 3,
    "insect_damage": 3,
}

#: Split yang diaugmentasi. Valid & test **tidak boleh** diaugmentasi: metrik
#: harus diukur pada citra asli, bukan pada varian buatan.
AUGMENT_SPLITS: tuple[str, ...] = ("train",)

#: Split yang hanya disalin dari ``crops_raw`` ke ``crops`` tanpa augmentasi.
COPY_SPLITS: tuple[str, ...] = ("valid", "test")

#: Seluruh split dataset, dalam urutan pipeline.
SPLITS: tuple[str, ...] = ("train", "valid", "test")

# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #
#: Margin di sekitar bounding box saat crop (0.0 = tight crop, 0.05 = 5%).
CROP_PADDING_RATIO: float = 0.0

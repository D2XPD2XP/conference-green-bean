# Coffee Bean Quality Classification

Klasifikasi kualitas green bean kopi ke dalam **5 kelas** menggunakan tiga
arsitektur CNN dengan transfer learning.

## Overview

Implementasi Tugas Akhir **"Analisis Perbandingan Arsitektur Convolutional
Neural Network untuk Klasifikasi Kualitas pada Green Bean Kopi"**.

Penelitian ini membandingkan tiga arsitektur CNN — **ResNet-50**,
**MobileNetV3**, dan **InceptionV3** — untuk mengklasifikasi biji kopi
individual ke dalam lima kategori: satu kelas layak dan empat jenis cacat.

Ketiganya dilatih dengan **transfer learning gaya *feature extraction***
(backbone ImageNet dibekukan, hanya head klasifikasi yang dilatih) dan
**hyperparameter yang seragam**, sehingga perbedaan skor dapat diatribusikan ke
arsitekturnya, bukan ke pengaturan pelatihan.

Tiga hal yang menjadi fokus di luar perbandingan arsitektur:

1. **Penanganan ketimpangan kelas** sebagai eksperimen tersendiri — empat
   strategi yang dapat dipertukarkan lewat satu argumen CLI.
2. **Augmentasi offline** yang dibekukan ke disk, bukan on-the-fly, demi
   reproducibility dan kontrol penuh atas distribusi dataset.
3. **Analisis yang kaya** — Grad-CAM, confusion matrix, kurva pembelajaran,
   distribusi confidence, galeri error, t-SNE/UMAP, dan trade-off efisiensi.

## Pipeline

```
Foto multi-bean → Anotasi Roboflow → Crop → Augmentasi offline → 3 CNN → Analisis
  (data/raw)      (data/annotated)  (crops_raw)   (crops)    (results/models) (results/figures)
```

1. **Pengumpulan data** — foto multi-bean pada kondisi terkontrol.
2. **Anotasi** — bounding box + label 5 kelas tiap biji di Roboflow, diekspor
   dalam format YOLO.
3. **Crop** — tiap bounding box dipotong menjadi citra biji individual ke
   `data/crops_raw/`.
4. **Augmentasi offline** — split train di-oversample per kelas dan disimpan ke
   `data/crops/`; valid & test disalin apa adanya.
5. **Pelatihan CNN** — tiga arsitektur dilatih terpisah dengan konfigurasi
   identik dan strategi imbalance yang sama.
6. **Evaluasi & analisis** — metrik tingkat biji (makro + per kelas + agregat
   layak/cacat), efisiensi, dan seluruh figur interpretasi.

Crop diambil dari **koordinat anotasi ground truth**, bukan dari prediksi model
deteksi mana pun, agar perbandingan arsitektur terisolasi dari error tahap lain.

## Class Definitions

| `class_id` | Code | Bahasa Indonesia | Deskripsi |
| ---------- | ---- | ---------------- | --------- |
| `0` | `good` | Baik | Biji utuh tanpa cacat |
| `1` | `black` | Hitam | Biji menghitam (fermentasi berlebih / busuk) |
| `2` | `chipped` | Serpihan | Biji pecah atau terkelupas |
| `3` | `dry_pods` | Polong Kering | Polong/kulit kering yang terbawa |
| `4` | `insect_damage` | Rusak Serangga | Biji berlubang / tergerek serangga |

Kode dan nama folder memakai bahasa Inggris; padanan bahasa Indonesia tersedia
di `CLASS_NAMES_ID` pada [`configs/base_config.py`](configs/base_config.py)
untuk keperluan pelaporan.

Selain metrik 5 kelas, pipeline juga melaporkan metrik agregat **layak vs
cacat** (keempat kelas cacat digabung), karena untuk quality control pertanyaan
praktisnya adalah "biji ini lolos atau tidak".

## Project Structure

```
ta-green-bean/
├── configs/       # Konfigurasi global (base_config) & hyperparameter CNN
├── src/           # Logika inti
│   ├── data/      #   crop, augmentasi offline, Dataset, transform
│   ├── models/    #   builder 3 arsitektur + model_factory
│   ├── training/  #   training loop, callbacks, penanganan imbalance
│   ├── evaluation/#   metrik, confusion matrix, efisiensi, prediksi
│   ├── analysis/  #   Grad-CAM, confusion, confidence, error, t-SNE, efisiensi
│   └── utils/     #   seed, logger, I/O checkpoint & anotasi
├── notebooks/     # Orkestrasi & visualisasi (01–12); logika inti tetap di src/
├── scripts/       # Entry point CLI (crop, augment, train, evaluate, analyze, run_revision, ...)
├── archive/       # Modul era pipeline YOLOv12 (lihat archive/README.md)
├── data/          # Dataset (di-.gitignore; lihat data/README.md)
├── results/       # Model, figur, metrik, analisis, log (di-.gitignore)
├── runs/          # Run multi-seed revisi: {model}_{config}_seed{seed}/ (di-.gitignore)
└── tests/         # Unit test (pytest)
```

Notebook di [`notebooks/`](notebooks/) hanya memanggil modul dari
[`src/`](src/) — tidak ada logika inti yang ditulis di notebook.

> **Konteks lengkap proyek** (arsitektur, keputusan desain, konvensi, dan
> status terkini) terangkum di [CLAUDE.md](CLAUDE.md).

## Prerequisites

- Python **3.10+**
- GPU **opsional** (NVIDIA + CUDA untuk akselerasi; CPU tetap didukung). Kode
  otomatis memakai GPU bila terdeteksi (`DEVICE` di `configs/base_config.py`).

## Setup

1. **Clone repo**

   ```bash
   git clone <url-repo>
   cd ta-green-bean
   ```

2. **Buat & aktifkan virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux / macOS
   .venv\Scripts\activate           # Windows (PowerShell / cmd)
   ```

3. **(GPU NVIDIA) Install torch build CUDA lebih dulu — WAJIB bila ingin pakai GPU**

   `pip install torch` dari PyPI di Windows memberi build **CPU-only**. Untuk
   GPU, install torch dari index CUDA PyTorch **sebelum** langkah 4:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

   > **RTX seri 50 (Blackwell, mis. RTX 5050)** wajib CUDA **12.8+** → gunakan
   > `cu128`. Wheel lama `cu118`/`cu121` tidak mendukung GPU ini. Lewati langkah
   > ini bila hanya memakai CPU.

4. **Install sisa dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Verifikasi**

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   python -c "from configs.base_config import CLASS_NAMES, NUM_CLASSES; print(CLASS_NAMES, NUM_CLASSES)"
   pytest tests/
   ```

6. **Konfigurasi environment (opsional)**

   ```bash
   cp .env.example .env             # lalu isi kredensial Roboflow
   ```

## Usage

### Step 1 — Anotasi di Roboflow

Anotasi dengan 5 kelas berurutan: `good`, `black`, `chipped`, `dry_pods`,
`insect_damage`. Export sebagai **YOLOv8** ke `data/annotated/`.

Setting versi yang wajib: Auto-Orient **ON**, Resize **None**, Augmentation
**kosong**, split **per foto**.

> Augmentasi dimatikan di Roboflow karena dilakukan offline oleh pipeline ini,
> dengan kontrol multiplier per kelas. Resize dimatikan karena biji sudah kecil
> pada foto beresolusi tinggi.

### Step 2 — Crop biji individual

```bash
python -m scripts.crop_dataset
```

Output: `data/crops_raw/{train,valid,test}/{kelas}/`

| Argumen | Default | Keterangan |
| ------- | ------- | ---------- |
| `--dataset-root` | `data/annotated` | root ekspor Roboflow |
| `--output-root` | `data/crops_raw` | root keluaran crop mentah |
| `--padding-ratio` | `0.0` | margin di sekitar bbox (mis. `0.05` = 5%) |
| `--class-names` | 5 kelas | urutan nama kelas = `class_id` |
| `--splits` | `train valid test` | split yang diproses |

### Step 3 — Augmentasi offline

```bash
python -m scripts.augment_dataset           # menambah
python -m scripts.augment_dataset --clean   # setelah ganti versi dataset
```

Output: `data/crops/{train,valid,test}/{kelas}/`

- **Train**: citra asli (`_orig`) + varian teraugmentasi (`_aug01`, `_aug02`, …)
- **Valid & test**: salinan dari `crops_raw` **tanpa** augmentasi

| Argumen | Default | Keterangan |
| ------- | ------- | ---------- |
| `--multiplier KELAS=N` | lihat config | override multiplier satu kelas, dapat diulang |
| `--augment-splits` | `train` | split yang diaugmentasi |
| `--copy-splits` | `valid test` | split yang hanya disalin |
| `--clean` | mati | kosongkan folder tujuan lebih dulu |
| `--random-seed` | `42` | seed augmentasi (deterministik per citra) |

### Step 4 — Latih ketiga CNN

```bash
python -m scripts.train --model resnet50    --imbalance-strategy class_weights
python -m scripts.train --model mobilenetv3 --imbalance-strategy class_weights
python -m scripts.train --model inceptionv3 --imbalance-strategy class_weights
```

Strategi imbalance yang tersedia:

| Nilai | Cara kerja |
| ----- | ---------- |
| `none` | `CrossEntropyLoss` biasa (baseline) |
| `class_weights` | bobot per kelas pada loss (**default**) |
| `weighted_sampler` | `WeightedRandomSampler` menyeimbangkan tiap batch |
| `focal_loss` | Focal Loss (Lin et al., ICCV 2017) |

Argumen lain: `--epochs`, `--batch-size`, `--lr`, `--num-workers`,
`--output-dir`, `--class-weights-strategy`, `--focal-gamma`,
`--save-predictions`, `--save-features`, serta:

| Argumen | Default | Fungsi |
| ------- | ------- | ------ |
| `--seed` | `42` | seed run (inisialisasi head, shuffle, sampler) |
| `--checkpoint-metric` | `val_f1` | `val_f1` (F1 makro, seri → epoch awal) atau `val_acc` (kriteria lama) |
| `--bn-mode` | `adapt` | `adapt` = statistik BN beradaptasi (default lama); `frozen` = BN mode eval selama training |

Checkpoint diberi nama `<model>_best_epoch{N}_{f1|acc}{X.XX}.pt` sesuai kriteria.

> Untuk eksperimen cepat, pakai `--output-dir results/tmp/<model>` agar history
> run utama tidak tertimpa.

### Step 5 — Evaluasi

```bash
python -m scripts.evaluate --model resnet50 --checkpoint <path.pt>
```

Menyimpan metrik, confusion matrix, prediksi CSV, dan feature vector `.npz`.

### Step 6 — Analisis lengkap

```bash
python -m scripts.analyze
```

Menghasilkan seluruh figur & tabel untuk laporan dalam satu perintah. Opsi:
`--models`, `--skip-gradcam`, `--skip-tsne`, `--skip-umap`,
`--gradcam-per-class`, `--error-samples`.

### Jalan pintas: pipeline end-to-end

```bash
bash scripts/run_full_pipeline.sh                 # Git Bash / Linux / macOS
CLEAN=1 EPOCHS=30 bash scripts/run_full_pipeline.sh
```

```powershell
.\scripts\run_full_pipeline.ps1                   # Windows PowerShell
.\scripts\run_full_pipeline.ps1 -Clean -Strategy focal_loss
```

### Alur notebook

| Notebook | Isi |
| -------- | --- |
| `01_explore_dataset.ipynb` | statistik dataset, distribusi 5 kelas, anotasi full vs partial |
| `02_crop_and_augment.ipynb` | crop + augmentasi offline + verifikasi visual |
| `03_train_resnet50.ipynb` | latih & evaluasi ResNet-50 |
| `04_train_mobilenetv3.ipynb` | latih & evaluasi MobileNetV3 |
| `05_train_inceptionv3.ipynb` | latih & evaluasi InceptionV3 |
| `06_compare_results.ipynb` | tabel perbandingan + pemilihan model terbaik |
| `07_gradcam_analysis.ipynb` | Grad-CAM ketiga arsitektur |
| `08_error_analysis.ipynb` | confusion, pola error, galeri, confidence |
| `09_feature_visualization.ipynb` | t-SNE & UMAP ruang fitur |
| `10_efficiency_tradeoff.ipynb` | efisiensi & rekomendasi deployment |
| `11_paper_figures.ipynb` | figur draft paper (run seed tunggal, `results/`) |
| `12_revision_results.ipynb` | hasil revisi multi-seed (hanya membaca `runs/`) |

### Eksperimen revisi (multi-seed)

Revisi reviewer menjalankan 63 run: konfigurasi utama 5 seed (42, 0, 1, 2, 3)
dan 6 ablasi × 3 seed (42, 0, 1). Hasil dan pembahasan ada di
[`REVISION_RESULTS.md`](REVISION_RESULTS.md); angka mentah di
`revision_results.json`.

```bash
python -m scripts.run_revision                 # ≈ 7 jam; run yang punya metrics.json dilewati
python -m scripts.aggregate_revision           # runs/summary/tables.md, per_run.csv
python -m scripts.revision_extras photo-stats  # statistik per foto
python -m scripts.revision_extras timing       # waktu per sampel 50 g (GPU harus menganggur)
python -m scripts.revision_extras support      # bootstrap CI, silhouette, confidence, efisiensi
python -m scripts.revision_figures             # results/figures/paper_revision/
```

Dataset ablasi augmentasi (`aug_all_x3`, `aug_all_balanced`) dibuat ke folder
terpisah dengan `scripts.augment_dataset --output-dir data/crops_aug_<nama>`
(plus `--replace-original good=<fraksi>` untuk varian balanced);
`data/crops/` tidak disentuh.

## Reproducibility

Seluruh operasi acak menghormati `RANDOM_SEED` (default `42`) yang diset melalui
[`src/utils/seed.py`](src/utils/seed.py). Augmentasi offline memakai seed
**per citra** yang diturunkan dari hash `(seed, kelas, nama file, indeks
varian)`, sehingga dua kali run menghasilkan berkas yang identik dan tidak
bergantung pada urutan pemrosesan.

Konfigurasi global ada di [`configs/base_config.py`](configs/base_config.py);
hyperparameter ketiga CNN di [`configs/cnn_config.py`](configs/cnn_config.py).

## Verifikasi Setup

```bash
# Unit test
pytest tests/

# Validasi dataset (read-only)
python .claude/skills/check-dataset/validate.py --crops

# Cek seluruh modul analisis dapat diimpor
python -c "from src.data.offline_augmentation import generate_augmented_dataset"
python -c "from src.training.imbalance import compute_class_weights, FocalLoss"
python -c "from src.analysis.gradcam import GradCAMAnalyzer"
python -c "from src.analysis.feature_visualization import extract_features"
python -c "from src.analysis.efficiency_analysis import measure_model_efficiency"
```

## Referensi

- Proposal Tugas Akhir — *Analisis Perbandingan Arsitektur Convolutional Neural
  Network untuk Klasifikasi Kualitas pada Green Bean Kopi* (Tabel 3.2–3.4 untuk
  hyperparameter CNN).
- Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
- Cui et al., "Class-Balanced Loss Based on Effective Number of Samples",
  CVPR 2019.
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization", ICCV 2017.

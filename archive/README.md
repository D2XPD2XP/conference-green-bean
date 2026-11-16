# Archive — modul era pipeline YOLOv12 + CNN

**Tanggal arsip: 13 September 2026.**

Folder ini menyimpan kode dari desain sebelum proyek beralih ke **pipeline
CNN-only**. Tidak ada file yang dihapus permanen — semuanya diarsipkan sebagai
*insurance policy* bila suatu saat tahap segmentasi otomatis perlu dihidupkan
kembali.

## Alasan arsip

Pipeline lama menjalankan dua tahap: YOLOv12 melokalisasi tiap biji pada foto
multi-bean, lalu tiga CNN mengklasifikasi kualitas hasil deteksinya. Setelah
mempertimbangkan kompleksitas, timeline, dan risiko kegagalan, tahap YOLOv12
**dihapus sepenuhnya** dari pipeline.

Pipeline baru jauh lebih sederhana dan langsung menjawab pertanyaan penelitian
(perbandingan arsitektur CNN):

```
Foto multi-bean → Anotasi Roboflow (bbox + kelas)
                → Crop batch dari anotasi ground truth
                → Dataset biji individual per kelas
                → Latih 3 CNN → Bandingkan
```

Judul Tugas Akhir tidak berubah. Yang hilang hanyalah tahap segmentasi
otomatis; crop kini diambil langsung dari koordinat anotasi Roboflow lewat
[`src/data/crop_from_annotation.py`](../src/data/crop_from_annotation.py).

## Isi

| File | Peran lama |
| ---- | ---------- |
| `configs_yolo_config.py` | hyperparameter YOLOv12 (`yolo12n.pt`, 640 px, 100 epoch) |
| `src_training_train_yolo.py` | pelatihan YOLOv12 via `ultralytics` |
| `src_evaluation_yolo_metrics.py` | metrik deteksi (mAP@0.5, precision, recall) |
| `src_evaluation_matching.py` | IoU + greedy matching prediksi YOLOv12 ↔ ground truth |
| `src_data_crop_from_yolo.py` | generate dataset CNN dari prediksi YOLOv12 yang *matched* |
| `src_inference/end_to_end.py` | `EndToEndPipeline` — deteksi → crop → klasifikasi dalam satu objek |
| `src_inference/__init__.py` | penanda package `src.inference` |
| `scripts_generate_cnn_dataset.py` | CLI `python -m scripts.generate_cnn_dataset` |
| `scripts_evaluate_end_to_end.py` | CLI `python -m scripts.evaluate_end_to_end` |
| `notebooks_02_generate_cnn_dataset.ipynb` | notebook generate dataset dari YOLOv12 |
| `notebooks_03_train_yolov12.ipynb` | notebook pelatihan YOLOv12 |
| `notebooks_08_end_to_end_evaluation.ipynb` | notebook evaluasi terintegrasi + overlay bbox |
| `tests_test_matching.py` | unit test IoU & greedy matching |

Nama file diratakan menjadi `<area>_<nama asli>` agar isi folder ini tidak
tertukar dengan paket `src/` yang aktif dan tidak ikut ter-*import* tanpa
sengaja.

## Cara mengembalikan bila diperlukan

1. **Kembalikan file ke lokasi aslinya**, mengikuti awalan namanya:

   ```bash
   git mv archive/configs_yolo_config.py        configs/yolo_config.py
   git mv archive/src_training_train_yolo.py    src/training/train_yolo.py
   git mv archive/src_evaluation_matching.py    src/evaluation/matching.py
   git mv archive/src_evaluation_yolo_metrics.py src/evaluation/yolo_metrics.py
   git mv archive/src_data_crop_from_yolo.py    src/data/crop_from_yolo.py
   git mv archive/src_inference                 src/inference
   git mv archive/scripts_generate_cnn_dataset.py scripts/generate_cnn_dataset.py
   git mv archive/tests_test_matching.py        tests/test_matching.py
   ```

2. **Kembalikan konstanta yang sudah dihapus** dari
   [`configs/base_config.py`](../configs/base_config.py):
   `IOU_MATCH_THRESHOLD`, `MATCHING_STATS_PATH`, dan `CROPS_GT_DIR`.

3. **Kembalikan helper yang sudah dihapus**:
   `parse_yolo_label_file` di [`src/utils/io.py`](../src/utils/io.py), serta
   `compute_end_to_end_metrics` dan `compute_e2e_reporting_metrics` di
   [`src/evaluation/metrics.py`](../src/evaluation/metrics.py). Ketiganya masih
   dapat dilihat pada riwayat git sebelum commit overhaul.

4. **Pasang kembali dependensi**: tambahkan `ultralytics>=8.1.0` ke
   `requirements.txt` (sudah dilepas dari pipeline CNN-only).

5. **Sesuaikan nama kelas.** Kode di sini ditulis saat kelas masih bernama
   `layak` / `tidak_layak`; pipeline sekarang memakai `accepted` / `rejected`.

6. **Daftarkan ulang export** di `__init__.py` masing-masing paket dan
   kembalikan cabang `--model yolo` di `scripts/train.py` &
   `scripts/evaluate.py`.

> Singkatnya: kode di sini **tidak akan langsung berjalan** setelah dipindahkan
> kembali. Anggap sebagai referensi implementasi, bukan modul siap pakai.

## Kenapa diarsipkan, bukan dihapus

Logika IoU matching, inferensi batch, overlay visualisasi bounding box, dan
pelaporan statistik deteksi di dalamnya tetap berguna sebagai referensi bila
ruang lingkup penelitian diperluas, atau sebagai bahan pembanding di bagian
pembahasan laporan.

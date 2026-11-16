# Results

Keluaran eksperimen. **Isinya di-`.gitignore`** kecuali `.gitkeep` dan file ini.

```
results/
├── models/     # Bobot terlatih per arsitektur — <model>/<model>_best_epoch12_acc0.93.pt
│               #   plus <model>_history.json (loss/akurasi tiap epoch + strategi imbalance)
├── figures/    # Seluruh figur .png (confusion matrix, Grad-CAM, t-SNE, kurva, galeri error)
├── metrics/    # Metrik test per model (.json) — <model>_test_metrics.json
├── analysis/   # Artefak & tabel analisis (.csv, .npz, .json)
└── logs/       # Log pelatihan, evaluasi, crop, augmentasi, analisis (.log)
```

Path direktori ini didefinisikan di `configs/base_config.py`
(`RESULTS_DIR`, `MODELS_DIR`, `FIGURES_DIR`, `METRICS_DIR`, `ANALYSIS_DIR`,
`LOGS_DIR`).

## Isi `metrics/<model>_test_metrics.json`

Dihasilkan `python -m scripts.evaluate` maupun notebook pelatihan 03–05, dengan
format yang sama sehingga notebook 06–10 dan `scripts.analyze` dapat membaca
keduanya:

| Kunci | Keterangan |
| ----- | ---------- |
| `accuracy`, `precision`, `recall`, `f1` | metrik macro-average tingkat biji (5 kelas) |
| `per_class` | presisi/recall/F1/support tiap kelas |
| `defect_accuracy`, `defect_precision`, `defect_recall`, `defect_f1` | metrik agregat layak vs cacat |
| `defect_miss_rate` | proporsi biji cacat yang lolos sebagai `good` |
| `efficiency` | params, ukuran, waktu inferensi, throughput, FLOPs |
| `total_params`, `trainable_params`, `inference_ms` | duplikat datar untuk kompatibilitas |
| `top_error_patterns` | pasangan kelas yang paling sering tertukar |
| `n_samples`, `class_distribution`, `mean_confidence` | ringkasan split test |

## Isi `analysis/`

| Berkas | Dihasilkan oleh | Dipakai untuk |
| ------ | --------------- | ------------- |
| `<model>_test_predictions.csv` | `scripts.evaluate` | confidence & error analysis (notebook 08) |
| `<model>_test_features.npz` | `scripts.evaluate` | t-SNE / UMAP (notebook 09) |
| `model_comparison.csv` | `scripts.analyze` | tabel perbandingan utama |
| `per_class_metrics.csv` | `scripts.analyze` | metrik per kelas semua model |
| `convergence_summary.csv` | `scripts.analyze` | konvergensi & sinyal overfitting |
| `confidence_summary.csv` | `scripts.analyze` | selisih confidence benar vs salah |
| `feature_separation.csv` | `scripts.analyze` | silhouette score ruang fitur |
| `efficiency_comparison.csv` | `scripts.analyze` | trade-off efisiensi |
| `<model>_error_patterns.csv` | `scripts.analyze` | pola kesalahan terbanyak |
| `<model>_error_summary.csv` | `scripts.analyze` | tingkat error per kelas |
| `<model>_accuracy_vs_threshold.csv` | `scripts.analyze` | simulasi human-in-the-loop |
| `analysis_summary.json` | `scripts.analyze` | seluruh ringkasan dalam satu berkas |

## Mengarsipkan sebelum run baru

Hasil dari versi dataset atau skema kelas berbeda **tidak sebanding**. Sebelum
menjalankan ulang pipeline dengan dataset baru, pindahkan isi folder ini ke
arsip, misalnya `results_archive/<tanggal>/`, agar hasil lama tidak tercampur
dengan yang baru.

Checkpoint dari skema kelas lama juga tidak bisa dimuat ke model dengan jumlah
kelas berbeda — `scripts.analyze` akan melewatinya dengan peringatan.

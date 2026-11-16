"""Konfigurasi pelatihan YOLOv12 untuk segmentasi biji.

Tahap segmentasi menggunakan satu kelas (`biji`) untuk melokalisasi tiap green
bean pada citra multi-bean. Bobot awal memakai varian nano (`yolo12n.pt`) sebagai
baseline; dapat ditingkatkan ke s/m/l/x sesuai kebutuhan.
"""

from __future__ import annotations

#: Konfigurasi YOLOv12 (dikonsumsi oleh src/training/train_yolo.py).
YOLO_CONFIG: dict[str, object] = {
    "model": "yolo12n.pt",  # nano untuk baseline; bisa ke s/m/l/x
    "input_size": 640,
    "epochs": 100,
    "batch_size": 16,
    "patience": 20,  # early stopping
    "num_classes": 1,  # satu kelas: "biji"
    "class_names": ["biji"],
}

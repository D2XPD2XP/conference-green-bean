"""Pelatihan YOLOv12 untuk segmentasi biji via package ``ultralytics``.

Data harus dideskripsikan oleh file ``data.yaml`` bergaya YOLO yang menunjuk ke
split train/valid/test dan mendefinisikan kelas (satu kelas: ``biji``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from configs.yolo_config import YOLO_CONFIG

logger = logging.getLogger(__name__)


def train_yolo(
    data_yaml: str | Path,
    config: dict | None = None,
    project: str | Path = "runs",
    name: str = "yolov12_biji",
):
    """Latih model YOLOv12 pada dataset segmentasi biji.

    Args:
        data_yaml: Path ke file ``data.yaml`` (format ultralytics) yang
            mendefinisikan path split dan kelas.
        config: Override ``YOLO_CONFIG``. Bila ``None``, pakai default.
        project: Direktori induk keluaran ultralytics.
        name: Nama subfolder run.

    Returns:
        Objek hasil ``model.train`` dari ultralytics.

    Raises:
        ImportError: Bila package ``ultralytics`` belum terinstal.
        FileNotFoundError: Bila ``data_yaml`` tidak ada.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - bergantung environment
        raise ImportError(
            "Package 'ultralytics' belum terinstal. Jalankan: pip install ultralytics"
        ) from exc

    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan: {data_yaml}")

    cfg = {**YOLO_CONFIG, **(config or {})}
    logger.info("Memuat bobot awal YOLO: %s", cfg["model"])
    model = YOLO(cfg["model"])

    results = model.train(
        data=str(data_yaml),
        imgsz=cfg["input_size"],
        epochs=cfg["epochs"],
        batch=cfg["batch_size"],
        patience=cfg["patience"],
        project=str(project),
        name=name,
    )
    logger.info("Pelatihan YOLO selesai. Hasil di %s/%s", project, name)
    return results

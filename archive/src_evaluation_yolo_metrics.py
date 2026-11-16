"""Evaluasi tahap segmentasi: metrik deteksi YOLOv12 (mAP@0.5, precision, recall).

Metrik ini adalah keluaran tahap [2] pipeline dan menjadi konteks penting bagi
metrik CNN: makin rendah recall YOLOv12, makin banyak biji yang tidak pernah
sampai ke tahap klasifikasi.
"""

from __future__ import annotations

import logging
from pathlib import Path

from configs.yolo_config import YOLO_CONFIG

logger = logging.getLogger(__name__)


def evaluate_yolo(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    imgsz: int | None = None,
) -> dict[str, float]:
    """Hitung metrik deteksi YOLOv12 pada sebuah split via ``ultralytics``.

    Args:
        weights: Path bobot YOLOv12 terlatih (``.pt``).
        data_yaml: Path ``data.yaml`` (format ultralytics) yang mendefinisikan
            split dan kelas.
        split: Split yang dievaluasi (``"test"``, ``"val"``, atau ``"train"``).
        imgsz: Ukuran input. Bila ``None``, memakai ``YOLO_CONFIG['input_size']``.

    Returns:
        Dict berisi ``map50`` (mAP@0.5), ``map50_95``, ``precision``, ``recall``,
        serta ``split`` dan ``weights`` sebagai metadata.

    Raises:
        ImportError: Bila package ``ultralytics`` belum terinstal.
        FileNotFoundError: Bila bobot atau ``data.yaml`` tidak ditemukan.

    Example:
        >>> evaluate_yolo("results/models/yolov12_best.pt",
        ...               "data/annotated/data.yaml")["map50"]  # doctest: +SKIP
        0.962
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - bergantung environment
        raise ImportError(
            "Package 'ultralytics' belum terinstal. Jalankan: pip install ultralytics"
        ) from exc

    weights = Path(weights)
    data_yaml = Path(data_yaml)
    if not weights.exists():
        raise FileNotFoundError(f"Bobot YOLOv12 tidak ditemukan: {weights}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan: {data_yaml}")

    size = int(imgsz if imgsz is not None else YOLO_CONFIG["input_size"])
    logger.info("Evaluasi YOLOv12 (%s) pada split '%s' ...", weights.name, split)

    model = YOLO(str(weights))
    results = model.val(data=str(data_yaml), split=split, imgsz=size, verbose=False)

    box = results.box
    metrics = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "split": split,
        "weights": str(weights),
    }
    logger.info(
        "YOLOv12 | mAP@0.5=%.4f | mAP@0.5:0.95=%.4f | precision=%.4f | recall=%.4f",
        metrics["map50"],
        metrics["map50_95"],
        metrics["precision"],
        metrics["recall"],
    )
    return metrics

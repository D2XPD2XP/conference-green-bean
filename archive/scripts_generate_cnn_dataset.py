"""CLI: generate dataset crop CNN dari hasil segmentasi YOLOv12.

Tahap ini **wajib dijalankan setelah YOLOv12 dilatih** dan sebelum pelatihan CNN,
karena seluruh crop di ``data/crops/`` berasal dari prediksi YOLOv12 yang
dicocokkan dengan anotasi ground truth (IoU greedy matching).

Contoh::

    python -m scripts.generate_cnn_dataset
    python -m scripts.generate_cnn_dataset --iou-threshold 0.6 --splits train valid
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from configs.base_config import (
    ANNOTATED_DATA_DIR,
    CLASS_NAMES,
    CROPS_DIR,
    IOU_MATCH_THRESHOLD,
    MODELS_DIR,
)
from src.data.crop_from_yolo import generate_all_splits
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse argumen CLI."""
    parser = argparse.ArgumentParser(
        description="Generate crop CNN dari prediksi YOLOv12 yang matched dengan ground truth."
    )
    parser.add_argument(
        "--yolo-weights",
        default=str(MODELS_DIR / "yolov12_best.pt"),
        help="Path bobot YOLOv12 terlatih (.pt).",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(ANNOTATED_DATA_DIR),
        help="Root ekspor Roboflow (berisi <split>/images dan <split>/labels).",
    )
    parser.add_argument(
        "--output-root",
        default=str(CROPS_DIR),
        help="Root keluaran crop CNN.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=IOU_MATCH_THRESHOLD,
        help="Ambang IoU untuk mencocokkan prediksi ke ground truth.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Ambang confidence deteksi YOLOv12 (default: bawaan ultralytics).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Daftar split yang diproses.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=CLASS_NAMES,
        help="Nama kelas; indeks = class_id anotasi.",
    )
    return parser.parse_args()


def _print_summary(results: dict, splits: list[str]) -> None:
    """Cetak ringkasan statistik matching per split dan totalnya."""
    header = f"{'split':<8}{'foto':>7}{'GT':>8}{'matched':>9}{'FN':>7}{'FP':>7}{'recall':>9}{'prec':>9}"
    print("\n=== Statistik Matching YOLOv12 → Ground Truth ===")
    print(header)
    print("-" * len(header))

    for split in splits:
        stats = results["splits"].get(split)
        if stats is None:
            print(f"{split:<8}{'(dilewati)':>7}")
            continue
        s = stats["summary"]
        print(
            f"{split:<8}{s['num_images']:>7}{s['num_ground_truths']:>8}{s['num_matched']:>9}"
            f"{s['num_false_negatives']:>7}{s['num_false_positives']:>7}"
            f"{s['yolo_recall']:>9.4f}{s['yolo_precision']:>9.4f}"
        )

    total = results["total"]
    print("-" * len(header))
    print(
        f"{'TOTAL':<8}{total['num_images']:>7}{total['num_ground_truths']:>8}{total['num_matched']:>9}"
        f"{total['num_false_negatives']:>7}{total['num_false_positives']:>7}"
        f"{total['yolo_recall']:>9.4f}{total['yolo_precision']:>9.4f}"
    )
    print(f"\nCrop tersimpan : {total['num_crops_saved']}")
    print(f"Statistik      : {results['stats_path']}")
    print(
        f"\nCatatan: {total['num_false_negatives']} biji ground truth tidak terdeteksi YOLOv12 "
        "sehingga tidak pernah dievaluasi CNN. Laporkan angka ini bersama metrik CNN."
    )


def main() -> int:
    """Jalankan generate dataset CNN. Mengembalikan exit code (0 = sukses)."""
    args = parse_args()

    set_seed()
    logger = setup_logger(
        "generate_cnn_dataset", log_file="generate_cnn_dataset.log", level=logging.INFO
    )
    logger.info(
        "Mulai generate dataset CNN | yolo=%s | dataset=%s | output=%s | iou_thr=%.2f",
        args.yolo_weights,
        args.dataset_root,
        args.output_root,
        args.iou_threshold,
    )

    try:
        results = generate_all_splits(
            yolo_model_path=args.yolo_weights,
            dataset_root=Path(args.dataset_root),
            output_root=Path(args.output_root),
            class_names=args.class_names,
            iou_threshold=args.iou_threshold,
            splits=tuple(args.splits),
            conf_threshold=args.conf_threshold,
        )
    except (FileNotFoundError, ImportError) as exc:
        logger.error("Gagal generate dataset: %s", exc)
        return 1
    except Exception:  # pragma: no cover - jaring pengaman agar exit code benar
        logger.exception("Error tak terduga saat generate dataset.")
        return 1

    if not results["splits"]:
        logger.error("Tidak ada split yang berhasil diproses; periksa --dataset-root.")
        return 1

    _print_summary(results, list(args.splits))
    return 0


if __name__ == "__main__":
    sys.exit(main())

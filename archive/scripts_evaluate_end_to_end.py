"""CLI: evaluasi sistem terintegrasi YOLOv12 + CNN pada split test.

Berbeda dengan ``scripts.evaluate`` (crop dari anotasi ground truth), skrip ini
memakai crop dari **prediksi YOLOv12** sehingga error segmentasi ikut terukur.

Contoh::

    python -m scripts.evaluate_end_to_end \
        --cnn-weights results/models/resnet50_best_epoch12_acc0.93.pt \
        --cnn-arch resnet50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from configs.base_config import (
    DATA_DIR,
    END_TO_END_CROPS_DIR,
    END_TO_END_RESULTS_DIR,
    IOU_MATCH_THRESHOLD,
    MODELS_DIR,
)
from src.inference.end_to_end import EndToEndPipeline
from src.models.model_factory import AVAILABLE_MODELS
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse argumen CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluasi end-to-end YOLOv12 + CNN pada citra multi-bean."
    )
    parser.add_argument(
        "--yolo-weights",
        default=str(MODELS_DIR / "yolov12_best.pt"),
        help="Path bobot YOLOv12 (.pt).",
    )
    parser.add_argument("--cnn-weights", required=True, help="Path checkpoint CNN (.pt).")
    parser.add_argument(
        "--cnn-arch", required=True, choices=AVAILABLE_MODELS, help="Arsitektur CNN."
    )
    parser.add_argument(
        "--test-images",
        default=str(DATA_DIR / "annotated" / "test" / "images"),
        help="Folder gambar test.",
    )
    parser.add_argument(
        "--test-labels",
        default=str(DATA_DIR / "annotated" / "test" / "labels"),
        help="Folder label test (.txt format YOLO).",
    )
    parser.add_argument(
        "--output-dir", default=str(END_TO_END_RESULTS_DIR), help="Folder keluaran hasil."
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=IOU_MATCH_THRESHOLD,
        help="Ambang IoU untuk matching prediksi ke ground truth.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Ambang confidence deteksi YOLOv12 (default: bawaan ultralytics).",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help=f"Simpan crop hasil deteksi ke {END_TO_END_CROPS_DIR}.",
    )
    return parser.parse_args()


def main() -> None:
    """Jalankan evaluasi end-to-end dan cetak ringkasan hasil."""
    args = parse_args()

    set_seed()
    logger = setup_logger(
        f"eval_e2e_{args.cnn_arch}",
        log_file=f"eval_end_to_end_{args.cnn_arch}.log",
        level=logging.INFO,
    )

    logger.info(
        "Mulai evaluasi end-to-end | yolo=%s | cnn=%s (%s) | iou_thr=%.2f",
        args.yolo_weights,
        args.cnn_weights,
        args.cnn_arch,
        args.iou_threshold,
    )

    pipeline = EndToEndPipeline(
        yolo_weights=args.yolo_weights,
        cnn_weights=args.cnn_weights,
        cnn_arch=args.cnn_arch,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )

    results = pipeline.evaluate_on_test_set(
        test_images_dir=Path(args.test_images),
        test_labels_dir=Path(args.test_labels),
        output_dir=Path(args.output_dir),
        crops_dir=END_TO_END_CROPS_DIR if args.save_crops else None,
    )

    metrics = results["metrics"]
    logger.info("Metrik end-to-end: %s", metrics)

    # Ringkasan untuk pengguna CLI.
    print("\n=== Ringkasan Evaluasi End-to-End ===")
    print(f"Arsitektur CNN      : {args.cnn_arch}")
    print(f"Jumlah foto test    : {results['config']['num_images']}")
    print(f"Biji ter-match (TP) : {metrics['num_matched']}")
    print(f"False positive YOLO : {metrics['num_false_positives']}")
    print(f"False negative YOLO : {metrics['num_false_negatives']}")
    print(f"YOLO precision      : {metrics['yolo_precision']:.4f}")
    print(f"YOLO recall         : {metrics['yolo_recall']:.4f}")
    print(f"CNN accuracy        : {metrics['cnn_accuracy']:.4f}")
    print(f"CNN precision       : {metrics['cnn_precision']:.4f}")
    print(f"CNN recall          : {metrics['cnn_recall']:.4f}")
    print(f"CNN F1              : {metrics['cnn_f1']:.4f}")
    print(f"\nHasil lengkap: {Path(args.output_dir) / 'end_to_end_results.json'}")


if __name__ == "__main__":
    main()

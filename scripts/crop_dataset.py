"""CLI: crop biji individual dari anotasi Roboflow (tahap 1 dari 2).

Setiap foto multi-bean di ``data/annotated/<split>/`` dipotong menjadi citra
biji individual di ``data/crops_raw/<split>/<kelas>/``.

Keluarannya **belum siap latih**: augmentasi offline dijalankan terpisah oleh
``python -m scripts.augment_dataset``, yang mengisi ``data/crops/``. Pemisahan
ini disengaja — ``crops_raw`` adalah artefak mentah yang bisa diperiksa dan
di-crop ulang tanpa menyentuh dataset pelatihan.

Contoh::

    python -m scripts.crop_dataset
    python -m scripts.crop_dataset --padding-ratio 0.05
    python -m scripts.crop_dataset --splits train valid
    python -m scripts.crop_dataset --class-names good black chipped dry_pods insect_damage
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from configs.base_config import (
    ANNOTATED_DATA_DIR,
    CLASS_NAMES,
    CROP_PADDING_RATIO,
    CROPS_RAW_DIR,
    SPLITS,
)
from src.data.crop_from_annotation import crop_beans_from_annotations
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse argumen CLI."""
    parser = argparse.ArgumentParser(
        description="Crop biji individual dari anotasi Roboflow menjadi dataset CNN."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(ANNOTATED_DATA_DIR),
        help="Root ekspor Roboflow berisi <split>/images dan <split>/labels.",
    )
    parser.add_argument(
        "--output-root",
        default=str(CROPS_RAW_DIR),
        help="Root keluaran crop mentah (default: data/crops_raw).",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=CROP_PADDING_RATIO,
        help="Margin di sekitar bbox sebagai fraksi sisi box (0.0 = tight crop).",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=CLASS_NAMES,
        help="Nama kelas; indeks = class_id pada label YOLO.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        help="Daftar split yang diproses.",
    )
    return parser.parse_args()


def _print_split_report(split: str, stats: dict, class_names: list[str]) -> None:
    """Cetak ringkasan satu split ke stdout."""
    total = stats["num_crops"]
    print(f"Processing split: {split}")
    print(f"  Total images: {stats['num_images']}")
    if stats["num_images_skipped"]:
        print(f"  Skipped images (tanpa label / gagal dibaca): {stats['num_images_skipped']}")
    print(f"  Total crops generated: {total}")

    distribution = stats["class_distribution"]
    parts = [
        f"{name}={distribution.get(name, 0)} "
        f"({(distribution.get(name, 0) / total * 100) if total else 0:.0f}%)"
        for name in class_names
    ]
    print(f"  Distribution: {', '.join(parts)}")
    print()


def main() -> None:
    """Jalankan crop untuk tiap split yang diminta dan cetak ringkasannya."""
    args = parse_args()
    logger = setup_logger("crop_dataset", log_file="crop_dataset.log", level=logging.INFO)

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    class_names: list[str] = list(args.class_names)

    if args.padding_ratio < 0:
        raise ValueError(f"--padding-ratio tidak boleh negatif: {args.padding_ratio}")

    logger.info(
        "Mulai crop dari %s ke %s (padding_ratio=%.3f, kelas=%s)",
        dataset_root,
        output_root,
        args.padding_ratio,
        class_names,
    )

    start = time.perf_counter()
    grand_total = 0
    grand_distribution = {name: 0 for name in class_names}

    for split in args.splits:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"

        if not images_dir.is_dir() or not labels_dir.is_dir():
            logger.warning("Split '%s' tidak lengkap (%s); dilewati.", split, images_dir)
            print(f"Processing split: {split}\n  DILEWATI - {images_dir} tidak ditemukan.\n")
            continue

        logger.info("Memproses split '%s'...", split)
        stats = crop_beans_from_annotations(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=output_root / split,
            class_names=class_names,
            padding_ratio=args.padding_ratio,
        )

        _print_split_report(split, stats, class_names)
        grand_total += stats["num_crops"]
        for name in class_names:
            grand_distribution[name] += stats["class_distribution"].get(name, 0)

    elapsed = time.perf_counter() - start
    print(f"Total crops (semua split): {grand_total}")
    print(
        "Total distribution: "
        + ", ".join(f"{name}={grand_distribution[name]}" for name in class_names)
    )
    print(f"Total processing time: {elapsed:.1f}s")

    logger.info(
        "Selesai. Total %d crop dalam %.1f detik | distribusi=%s",
        grand_total,
        elapsed,
        grand_distribution,
    )

    print()
    print("Langkah berikutnya - augmentasi offline supaya datanya siap dilatih:")
    print("  python -m scripts.augment_dataset")


if __name__ == "__main__":
    main()

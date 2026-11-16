"""CLI: augmentasi offline dataset crop (tahap 2 dari 2).

Membaca ``data/crops_raw/`` dan menulis dataset siap latih ke ``data/crops/``:

- split ``train`` diaugmentasi (oversampling kelas minoritas lewat multiplier),
- split ``valid`` dan ``test`` hanya disalin apa adanya.

Augmentasi dilakukan **sekali di muka** lalu dibekukan ke disk, bukan
on-the-fly tiap epoch. Seed-nya deterministik per citra, jadi dua kali run
menghasilkan berkas yang identik.

Contoh::

    python -m scripts.augment_dataset
    python -m scripts.augment_dataset --clean
    python -m scripts.augment_dataset --multiplier black=4 --multiplier good=2
    python -m scripts.augment_dataset --augment-splits train --copy-splits valid test
    python -m scripts.augment_dataset --input-dir data/crops_raw --output-dir data/crops
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from configs.base_config import (
    AUGMENT_SPLITS,
    AUGMENTATION_MULTIPLIERS,
    CLASS_NAMES,
    COPY_SPLITS,
    CROPS_DIR,
    CROPS_RAW_DIR,
    RANDOM_SEED,
)
from src.data.offline_augmentation import build_dataset, format_augmentation_report
from src.utils.io import save_json
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_multipliers(entries: list[str] | None, class_names: list[str]) -> dict[str, int]:
    """Gabungkan multiplier default dengan override dari CLI.

    Args:
        entries: Daftar string ``"<kelas>=<angka>"`` dari ``--multiplier``.
        class_names: Nama kelas yang valid.

    Returns:
        Dict ``nama_kelas -> multiplier``.

    Raises:
        ValueError: Bila format salah, kelasnya tidak dikenal, atau nilainya
            bukan integer >= 1.
    """
    multipliers = {name: int(AUGMENTATION_MULTIPLIERS.get(name, 1)) for name in class_names}

    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"--multiplier harus berformat kelas=angka, bukan '{entry}'")
        name, _, raw = entry.partition("=")
        name = name.strip()
        if name not in multipliers:
            raise ValueError(f"Kelas '{name}' tidak dikenal. Pilihan: {class_names}")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"Multiplier untuk '{name}' bukan integer: '{raw}'") from exc
        if value < 1:
            raise ValueError(f"Multiplier untuk '{name}' harus >= 1, bukan {value}")
        multipliers[name] = value

    return multipliers


def parse_args() -> argparse.Namespace:
    """Parse argumen CLI."""
    parser = argparse.ArgumentParser(
        description="Augmentasi offline crops_raw -> crops (dataset siap latih)."
    )
    parser.add_argument(
        "--input-dir",
        default=str(CROPS_RAW_DIR),
        help="Root crop mentah (default: data/crops_raw).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CROPS_DIR),
        help="Root dataset siap latih (default: data/crops).",
    )
    parser.add_argument(
        "--augment-splits",
        nargs="*",
        default=list(AUGMENT_SPLITS),
        help="Split yang diaugmentasi (default: train).",
    )
    parser.add_argument(
        "--copy-splits",
        nargs="*",
        default=list(COPY_SPLITS),
        help="Split yang hanya disalin tanpa augmentasi (default: valid test).",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=list(CLASS_NAMES),
        help="Nama kelas (urutan = class_id).",
    )
    parser.add_argument(
        "--multiplier",
        action="append",
        metavar="KELAS=N",
        help="Override multiplier satu kelas; dapat diulang.",
    )
    parser.add_argument(
        "--replace-original",
        action="append",
        metavar="KELAS=FRAKSI",
        help=(
            "Ganti fraksi citra asli kelas (multiplier 1) dengan varian _aug01, "
            "jumlah tetap. Mis. good=0.6667 untuk ablasi aug_all_balanced."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Seed augmentasi (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Kosongkan folder kelas tujuan lebih dulu. WAJIB dipakai saat ganti "
            "versi dataset, agar crop dari foto lama tidak tertinggal."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Jalankan augmentasi offline lalu cetak laporan distribusinya."""
    args = parse_args()

    set_seed(args.random_seed)
    logger = setup_logger(
        "augment_dataset",
        log_file="augment_dataset.log",
        level=logging.INFO,
        attach_to=["src"],
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    class_names: list[str] = list(args.class_names)

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Crop mentah belum ada di {input_dir}. Jalankan lebih dulu: "
            "python -m scripts.crop_dataset"
        )

    overlap = set(args.augment_splits) & set(args.copy_splits)
    if overlap:
        raise ValueError(
            f"Split {sorted(overlap)} ada di --augment-splits sekaligus "
            "--copy-splits; pilih salah satu."
        )

    multipliers = parse_multipliers(args.multiplier, class_names)
    replace_fractions: dict[str, float] = {}
    for entry in args.replace_original or []:
        name, _, raw = entry.partition("=")
        if name.strip() not in class_names:
            raise ValueError(f"Kelas '{name}' tidak dikenal. Pilihan: {class_names}")
        replace_fractions[name.strip()] = float(raw)

    logger.info(
        "Mulai augmentasi offline %s -> %s | augment=%s copy=%s | seed=%d | clean=%s",
        input_dir,
        output_dir,
        args.augment_splits,
        args.copy_splits,
        args.random_seed,
        args.clean,
    )
    logger.info("Multiplier per kelas: %s", multipliers)
    if replace_fractions:
        logger.info("Fraksi asli yang diganti varian: %s", replace_fractions)

    if not args.clean:
        existing = [
            split
            for split in (*args.augment_splits, *args.copy_splits)
            if any((output_dir / split).glob("*/*.jpg"))
        ]
        if existing:
            print(
                "PERINGATAN: sudah ada crop di "
                f"{', '.join(f'{output_dir}/{s}' for s in existing)}.\n"
                "            Berkas dengan nama sama akan ditimpa, tetapi sisa "
                "dari versi dataset sebelumnya TIDAK terhapus\n"
                "            dan akan mencemari dataset. Jalankan ulang dengan "
                "--clean bila datanya baru diganti.\n"
            )
            logger.warning("Folder keluaran sudah berisi crop; dijalankan tanpa --clean.")

    start = time.perf_counter()
    results = build_dataset(
        input_root=input_dir,
        output_root=output_dir,
        augment_splits=tuple(args.augment_splits),
        copy_splits=tuple(args.copy_splits),
        multipliers=multipliers,
        random_seed=args.random_seed,
        class_names=class_names,
        clean=args.clean,
        replace_fractions=replace_fractions or None,
    )
    elapsed = time.perf_counter() - start

    if not results:
        print(f"Tidak ada split yang diproses di {input_dir}.")
        logger.warning("Tidak ada split yang diproses.")
        return

    report = format_augmentation_report(results, class_names=class_names)
    print(report)
    print(f"\nWaktu proses: {elapsed:.1f}s")

    total_in = sum(stats["total_original"] for stats in results.values())
    total_out = sum(stats["total_output"] for stats in results.values())
    print(f"Total sampel: {total_in:,} -> {total_out:,}")

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "random_seed": args.random_seed,
        "multipliers": multipliers,
        "augment_splits": list(args.augment_splits),
        "copy_splits": list(args.copy_splits),
        "clean": bool(args.clean),
        "elapsed_s": round(elapsed, 2),
        "splits": results,
    }
    manifest = save_json(metadata, output_dir / "augmentation_manifest.json")
    print(f"Manifest tersimpan: {manifest}")

    logger.info("Selesai dalam %.1f detik | %d -> %d sampel", elapsed, total_in, total_out)

    print("\nLangkah berikutnya - latih ketiga arsitektur:")
    for name in ("resnet50", "mobilenetv3", "inceptionv3"):
        print(f"  python -m scripts.train --model {name}")


if __name__ == "__main__":
    main()

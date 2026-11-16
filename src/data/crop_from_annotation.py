"""Modul utama untuk crop biji individual dari anotasi Roboflow.

Modul ini membaca file label YOLO format (dari ekspor Roboflow) dan memotong
tiap bounding box menjadi citra biji individual, dikelompokkan ke folder per
kelas.

Ini adalah **sumber utama dataset** untuk pelatihan CNN: setiap citra
multi-bean dari Roboflow menghasilkan puluhan sampai ratusan citra biji
individual.

Kelas yang dipakai (urutan = ``class_id`` pada label YOLO Roboflow):

    0. ``good``          - biji utuh tanpa cacat
    1. ``black``         - biji menghitam
    2. ``chipped``       - biji pecah/terkelupas
    3. ``dry_pods``      - polong/kulit kering
    4. ``insect_damage`` - biji tergerek serangga

Keluaran modul ini adalah artefak **mentah** (``data/crops_raw/``): belum
diaugmentasi dan belum dipakai langsung untuk pelatihan. Dataset siap latih
dihasilkan tahap berikutnya oleh :mod:`src.data.offline_augmentation`.

Catatan strategi anotasi: sebagian foto hanya dianotasi pada biji cacatnya
(*partial annotation*). Biji ``good`` pada foto itu tidak berlabel, jadi tidak
ikut di-crop - itu memang perilaku yang diinginkan, karena crop tanpa label
tidak punya kelas yang sahih.

Format label YOLO (satu baris per objek)::

    <class_id> <x_center> <y_center> <width> <height>

Seluruh koordinat ternormalisasi terhadap dimensi gambar (rentang 0..1).

Struktur keluaran mengikuti konvensi PyTorch ``ImageFolder``::

    output_dir/<class_name>/<image_stem>_bean_<idx:03d>.jpg
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

from configs.base_config import CLASS_NAMES
from src.utils.io import load_yolo_annotations

logger = logging.getLogger(__name__)

#: Ekstensi gambar yang didukung saat mencari pasangan gambar-label.
_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def yolo_to_pixel_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
    padding_ratio: float = 0.0,
) -> tuple[int, int, int, int]:
    """Konversi bounding box YOLO ternormalisasi ke koordinat piksel.

    Args:
        x_center: Titik tengah horizontal (ternormalisasi, 0..1).
        y_center: Titik tengah vertikal (ternormalisasi, 0..1).
        width: Lebar box (ternormalisasi, 0..1).
        height: Tinggi box (ternormalisasi, 0..1).
        img_width: Lebar gambar dalam piksel.
        img_height: Tinggi gambar dalam piksel.
        padding_ratio: Margin tambahan di tiap sisi, sebagai fraksi dari sisi
            box (``0.05`` → box melebar 5% ke kiri, kanan, atas, dan bawah).
            Nilai negatif diabaikan.

    Returns:
        Tuple ``(x1, y1, x2, y2)`` dalam piksel, sudah di-clip ke batas gambar
        dengan ``x1 < x2`` dan ``y1 < y2``.

    Example:
        >>> yolo_to_pixel_bbox(0.5, 0.5, 1.0, 1.0, 100, 100)
        (0, 0, 100, 100)
        >>> yolo_to_pixel_bbox(0.5, 0.5, 0.5, 0.5, 100, 100, padding_ratio=0.1)
        (20, 20, 80, 80)
    """
    bw = width * img_width
    bh = height * img_height

    if padding_ratio > 0:
        bw *= 1.0 + 2.0 * padding_ratio
        bh *= 1.0 + 2.0 * padding_ratio

    cx = x_center * img_width
    cy = y_center * img_height

    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))

    # Clip ke batas gambar, sekaligus jamin box tetap punya luas > 0.
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(x1 + 1, min(x2, img_width))
    y2 = max(y1 + 1, min(y2, img_height))
    return x1, y1, x2, y2


def _find_label_for_image(image_path: Path, labels_dir: Path) -> Path | None:
    """Cari file label ``.txt`` yang berpasangan dengan sebuah gambar.

    Args:
        image_path: Path file gambar.
        labels_dir: Direktori berisi label YOLO.

    Returns:
        Path label yang cocok, atau ``None`` bila tidak ditemukan.
    """
    candidate = labels_dir / f"{image_path.stem}.txt"
    return candidate if candidate.exists() else None


def crop_beans_from_annotations(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    class_names: list[str] | None = None,
    padding_ratio: float = 0.0,
) -> dict:
    """Crop tiap biji dari gambar berdasarkan anotasi YOLO dan simpan per kelas.

    Untuk tiap pasangan gambar-label, setiap bounding box dikonversi ke piksel
    (di-clip ke dimensi gambar), di-crop, lalu disimpan ke
    ``output_dir/<class_name>/<image_stem>_bean_<idx:03d>.jpg``. Gambar tanpa
    file label dilewati dengan peringatan; baris label yang rusak atau
    ``class_id`` di luar rentang juga dilewati dengan peringatan.

    Args:
        images_dir: Direktori gambar sumber.
        labels_dir: Direktori label YOLO (``.txt``) berpasangan.
        output_dir: Direktori keluaran; subfolder per kelas dibuat otomatis.
        class_names: Daftar nama kelas; indeks daftar = ``class_id`` YOLO.
            Default ``configs.base_config.CLASS_NAMES`` (5 kelas).
        padding_ratio: Margin di sekitar bbox (fraksi sisi box). ``0.0`` berarti
            tight crop.

    Returns:
        Dict statistik berisi ``num_images`` (gambar yang diproses),
        ``num_images_skipped`` (gambar tanpa label atau gagal dibaca),
        ``num_crops`` (total crop tersimpan), dan ``class_distribution``
        (dict ``nama_kelas -> jumlah crop``).

    Raises:
        FileNotFoundError: Bila ``images_dir`` atau ``labels_dir`` tidak ada.

    Example:
        >>> stats = crop_beans_from_annotations(
        ...     "data/annotated/train/images",
        ...     "data/annotated/train/labels",
        ...     "data/crops_raw/train",
        ... )  # doctest: +SKIP
        >>> stats["num_crops"]  # doctest: +SKIP
        2693
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    class_names = list(class_names) if class_names is not None else list(CLASS_NAMES)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir tidak ditemukan: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels_dir tidak ditemukan: {labels_dir}")

    # Siapkan subfolder keluaran per kelas agar strukturnya konsisten walau
    # sebuah kelas tidak menghasilkan crop sama sekali.
    for name in class_names:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    if not image_paths:
        logger.warning("Tidak ada gambar di %s", images_dir)

    distribution: Counter[str] = Counter()
    num_processed = 0
    num_skipped = 0

    # ascii=True: konsol Windows (cp1252) tidak bisa menulis karakter blok default
    # tqdm dan akan melempar UnicodeEncodeError di tengah proses.
    for image_path in tqdm(image_paths, desc=f"crop {output_dir.name}", unit="foto", ascii=True):
        label_path = _find_label_for_image(image_path, labels_dir)
        if label_path is None:
            logger.warning("Label untuk gambar %s tidak ditemukan; dilewati.", image_path.name)
            num_skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Gagal membaca gambar %s; dilewati.", image_path.name)
            num_skipped += 1
            continue

        img_h, img_w = image.shape[:2]
        _crop_single_image(
            image=image,
            label_path=label_path,
            image_stem=image_path.stem,
            img_w=img_w,
            img_h=img_h,
            output_dir=output_dir,
            class_names=class_names,
            padding_ratio=padding_ratio,
            distribution=distribution,
        )
        num_processed += 1

    total_crops = sum(distribution.values())
    logger.info(
        "Selesai: %d gambar diproses (%d dilewati) -> %d crop di %s",
        num_processed,
        num_skipped,
        total_crops,
        output_dir,
    )
    for name in class_names:
        count = distribution.get(name, 0)
        share = (count / total_crops * 100) if total_crops else 0.0
        logger.info("  %-10s : %6d crop (%.1f%%)", name, count, share)

    return {
        "num_images": num_processed,
        "num_images_skipped": num_skipped,
        "num_crops": total_crops,
        "class_distribution": {name: distribution.get(name, 0) for name in class_names},
    }


def _crop_single_image(
    image,
    label_path: Path,
    image_stem: str,
    img_w: int,
    img_h: int,
    output_dir: Path,
    class_names: list[str],
    padding_ratio: float,
    distribution: Counter,
) -> int:
    """Crop seluruh box dalam satu file label.

    Args:
        image: Citra BGR hasil ``cv2.imread``.
        label_path: Path file label YOLO pasangannya.
        image_stem: Nama file gambar tanpa ekstensi (untuk penamaan crop).
        img_w: Lebar gambar dalam piksel.
        img_h: Tinggi gambar dalam piksel.
        output_dir: Direktori keluaran split ini.
        class_names: Nama kelas; indeks daftar = ``class_id``.
        padding_ratio: Margin di sekitar bbox.
        distribution: Counter yang diperbarui in-place per nama kelas.

    Returns:
        Jumlah crop yang berhasil disimpan dari gambar ini.
    """
    saved = 0

    for idx, (class_id, x_c, y_c, w, h) in enumerate(load_yolo_annotations(label_path)):
        if not 0 <= class_id < len(class_names):
            logger.warning(
                "class_id %d di %s di luar rentang %s; dilewati.",
                class_id,
                label_path.name,
                list(range(len(class_names))),
            )
            continue

        x1, y1, x2, y2 = yolo_to_pixel_bbox(
            x_c, y_c, w, h, img_w, img_h, padding_ratio=padding_ratio
        )
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning("Crop kosong pada %s box %d; dilewati.", label_path.name, idx)
            continue

        class_name = class_names[class_id]
        out_path = output_dir / class_name / f"{image_stem}_bean_{idx:03d}.jpg"
        if not cv2.imwrite(str(out_path), crop):
            logger.warning("Gagal menulis crop ke %s; dilewati.", out_path)
            continue

        distribution[class_name] += 1
        saved += 1

    return saved

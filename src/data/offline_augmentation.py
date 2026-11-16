"""Augmentasi offline: hasilkan varian citra crop lalu simpan ke disk.

Alur kerja::

    crops_raw/<split>/<kelas>/   (hasil crop dari anotasi, apa adanya)
        -> augmentasi offline  ->  crops/<split>/<kelas>/   (siap latih)

Keputusan desain yang disengaja: augmentasi **tidak** dilakukan on-the-fly saat
pelatihan, melainkan sekali di muka lalu dibekukan ke disk. Konsekuensinya:

- **Reproducible.** Dataset yang dilatih adalah file yang bisa diperiksa,
  dihitung, dan dilampirkan ke laporan — bukan hasil undian tiap epoch.
- **Distribusi kelas terkendali.** Multiplier per kelas memakai augmentasi
  sebagai mekanisme oversampling untuk kelas minoritas.
- **Harga yang dibayar.** Varian per citra terbatas pada jumlah yang dibuat
  (tidak ada varian baru tiap epoch), dan dataset memakan ruang disk.

Aturan yang tidak boleh dilanggar:

1. **Hanya split ``train`` yang diaugmentasi.** Valid dan test disalin apa
   adanya; metrik harus diukur pada citra asli, bukan varian buatan.
2. **Citra asli selalu ikut disimpan** (sufiks ``_orig``), supaya model tetap
   melihat distribusi aslinya.
3. **Seed deterministik per citra.** Seed tiap varian diturunkan dari hash
   ``(random_seed, kelas, nama file, indeks varian)``, jadi hasilnya identik
   antar-run dan tidak bergantung pada urutan pemrosesan.

Konvensi penamaan keluaran::

    <nama_asli>_orig.jpg        # salinan citra asli
    <nama_asli>_aug01.jpg       # varian teraugmentasi ke-1
    <nama_asli>_aug02.jpg       # varian teraugmentasi ke-2
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

from configs.base_config import AUGMENTATION_MULTIPLIERS, CLASS_NAMES, RANDOM_SEED

logger = logging.getLogger(__name__)

#: Ekstensi citra yang dipindai sebagai sumber augmentasi.
_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

#: Batas atas seed yang diterima numpy/albumentations (32-bit unsigned).
_SEED_MODULO: int = 2**31 - 1


def build_augmentation_pipeline(seed: int | None = None) -> A.Compose:
    """Bangun pipeline augmentasi albumentations untuk citra biji kopi.

    Pilihan transformasi disesuaikan dengan karakter objeknya: biji kopi
    difoto dari atas, jadi flip horizontal maupun vertikal tetap menghasilkan
    citra yang masuk akal secara fisik. Rotasi dibatasi +/-15 derajat dengan
    ``BORDER_REFLECT`` agar sudut citra tidak terisi hitam. Perubahan
    brightness/contrast dan hue/saturation dibuat kecil karena **warna adalah
    fitur diskriminatif** untuk kelas ``black`` dan ``dry_pods`` — menggeser
    warna terlalu jauh akan menghapus informasi yang justru ingin dipelajari
    model.

    Args:
        seed: Seed untuk generator acak internal albumentations. Bila
            diberikan, keluaran untuk citra yang sama selalu identik.

    Returns:
        ``albumentations.Compose`` yang menerima ``image`` RGB ``uint8``.
    """
    return A.Compose(
        [
            A.OneOf(
                [
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                ],
                p=0.7,
            ),
            A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_REFLECT),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15, contrast_limit=0.1, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=1.0
                    ),
                ],
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ],
        seed=seed,
    )


def variant_seed(base_seed: int, class_name: str, stem: str, variant_idx: int) -> int:
    """Turunkan seed deterministik untuk satu varian augmentasi.

    Seed dihitung dari hash isi argumen, bukan dari urutan iterasi, sehingga
    varian ke-``k`` dari sebuah citra selalu sama walaupun jumlah file berubah,
    urutan folder berbeda, atau proses dijalankan ulang sebagian.

    Args:
        base_seed: Seed global (``RANDOM_SEED``).
        class_name: Nama kelas citra.
        stem: Nama file citra tanpa ekstensi.
        variant_idx: Indeks varian (1-based).

    Returns:
        Integer seed dalam rentang ``0..2**31-2``.

    Example:
        >>> variant_seed(42, "black", "img001_bean_000", 1) == variant_seed(
        ...     42, "black", "img001_bean_000", 1
        ... )
        True
    """
    key = f"{base_seed}|{class_name}|{stem}|{variant_idx}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULO


def augment_image(image_bgr: np.ndarray, seed: int) -> np.ndarray:
    """Terapkan pipeline augmentasi pada satu citra BGR.

    Konversi ke RGB dilakukan lebih dulu karena transformasi warna
    (``HueSaturationValue``) mengasumsikan urutan kanal RGB; hasilnya
    dikonversi kembali ke BGR agar bisa ditulis langsung dengan
    ``cv2.imwrite``.

    Args:
        image_bgr: Citra BGR ``uint8`` hasil ``cv2.imread``.
        seed: Seed deterministik untuk varian ini.

    Returns:
        Citra BGR ``uint8`` teraugmentasi, berukuran sama dengan masukan.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pipeline = build_augmentation_pipeline(seed=seed)
    augmented = pipeline(image=rgb)["image"]
    return cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)


def _list_images(directory: Path) -> list[Path]:
    """Daftar citra dalam folder, mengabaikan ``.gitkeep`` dan file non-citra."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)


def _prepare_output_dir(output_dir: Path, clean: bool) -> int:
    """Siapkan folder keluaran; opsional kosongkan isinya lebih dulu.

    Args:
        output_dir: Folder kelas tujuan.
        clean: Bila ``True``, hapus seluruh citra lama di folder tersebut.

    Returns:
        Jumlah file yang dihapus.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not clean:
        return 0

    removed = 0
    for path in _list_images(output_dir):
        path.unlink()
        removed += 1
    return removed


def generate_augmented_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    multipliers: dict[str, int] | None = None,
    random_seed: int = RANDOM_SEED,
    class_names: list[str] | None = None,
    apply_augmentation: bool = True,
    clean: bool = False,
    jpeg_quality: int = 95,
    replace_fractions: dict[str, float] | None = None,
) -> dict:
    """Hasilkan dataset teraugmentasi untuk satu split.

    Untuk tiap kelas, setiap citra di ``input_dir/<kelas>/`` menghasilkan
    ``multipliers[kelas]`` berkas keluaran: satu salinan asli (``_orig``) dan
    ``multiplier - 1`` varian teraugmentasi (``_aug01``, ``_aug02``, ...).
    Multiplier ``1`` berarti citra hanya disalin tanpa varian.

    Bila ``apply_augmentation=False``, seluruh multiplier dianggap ``1``:
    fungsi ini menjadi operasi salin murni. Itulah mode yang dipakai untuk
    split ``valid`` dan ``test``.

    Args:
        input_dir: Folder split sumber (berisi subfolder per kelas).
        output_dir: Folder split tujuan; subfolder kelas dibuat otomatis.
        multipliers: Jumlah keluaran per citra asli per kelas. Default
            ``AUGMENTATION_MULTIPLIERS``. Kelas yang tidak tercantum dianggap
            bermultiplier ``1``.
        random_seed: Seed global; menentukan seluruh varian secara deterministik.
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.
        apply_augmentation: Bila ``False``, hanya menyalin citra asli.
        clean: Bila ``True``, kosongkan folder kelas tujuan lebih dulu. Mencegah
            sisa file dari versi dataset sebelumnya ikut terbawa.
        jpeg_quality: Kualitas JPEG keluaran (1-100).
        replace_fractions: Fraksi citra asli per kelas yang **diganti** oleh
            varian ``_aug01``-nya (bukan ditambah), mis. ``{"good": 2/3}``.
            Jumlah keluaran kelas itu tetap sama dengan jumlah aslinya, tetapi
            sebagian besar berupa varian teraugmentasi: memisahkan efek
            *keragaman* dari efek *jumlah* (ablasi ``aug_all_balanced``).
            Hanya boleh dipakai pada kelas bermultiplier ``1``. Pemilihan citra
            yang diganti deterministik (urutan hash ``variant_seed`` indeks 0),
            dan varian ``_aug01`` identik dengan ``_aug01`` di konfigurasi lain.

    Returns:
        Dict statistik::

            {
                "split": <nama folder split>,
                "augmented": bool,
                "per_class": {
                    "<kelas>": {"original": int, "output": int, "multiplier": int,
                                "replaced": int}
                },
                "total_original": int,
                "total_output": int,
                "num_removed": int,     # file lama yang dihapus (clean=True)
            }

    Raises:
        FileNotFoundError: Bila ``input_dir`` tidak ada.

    Example:
        >>> stats = generate_augmented_dataset(
        ...     "data/crops_raw/train", "data/crops/train"
        ... )  # doctest: +SKIP
        >>> stats["total_output"] >= stats["total_original"]  # doctest: +SKIP
        True
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    mult = dict(multipliers) if multipliers is not None else dict(AUGMENTATION_MULTIPLIERS)
    replace = dict(replace_fractions or {}) if apply_augmentation else {}

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir tidak ditemukan: {input_dir}")

    per_class: dict[str, dict[str, int]] = {}
    total_original = 0
    total_output = 0
    num_removed = 0
    write_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    for class_name in names:
        class_in = input_dir / class_name
        class_out = output_dir / class_name
        num_removed += _prepare_output_dir(class_out, clean=clean)

        images = _list_images(class_in)
        multiplier = max(1, int(mult.get(class_name, 1))) if apply_augmentation else 1
        n_written = 0

        fraction = float(replace.get(class_name, 0.0))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"replace_fractions[{class_name}] harus di [0, 1], dapat {fraction}")
        if fraction > 0 and multiplier != 1:
            raise ValueError(
                f"replace_fractions hanya untuk kelas bermultiplier 1; {class_name} = {multiplier}"
            )
        n_replace = int(round(fraction * len(images)))
        ranked = sorted(
            images, key=lambda p: variant_seed(random_seed, class_name, p.stem, 0)
        )
        replaced_stems = {p.stem for p in ranked[:n_replace]}

        # ascii=True: konsol Windows (cp1252) tidak bisa menulis karakter blok
        # default tqdm dan akan melempar UnicodeEncodeError di tengah proses.
        desc = f"{'aug' if apply_augmentation else 'copy'} {input_dir.name}/{class_name}"
        for image_path in tqdm(images, desc=desc, unit="crop", ascii=True, leave=False):
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Gagal membaca %s; dilewati.", image_path)
                continue

            if image_path.stem in replaced_stems:
                # Citra asli diganti varian _aug01 (seed sama dengan konfigurasi lain).
                seed = variant_seed(random_seed, class_name, image_path.stem, 1)
                variant = augment_image(image, seed=seed)
                out_path = class_out / f"{image_path.stem}_aug01.jpg"
                if cv2.imwrite(str(out_path), variant, write_params):
                    n_written += 1
                else:
                    logger.warning("Gagal menulis %s; dilewati.", out_path)
                continue

            # Varian 0 = citra asli, selalu ikut disimpan.
            orig_path = class_out / f"{image_path.stem}_orig.jpg"
            if cv2.imwrite(str(orig_path), image, write_params):
                n_written += 1
            else:
                logger.warning("Gagal menulis %s; dilewati.", orig_path)
                continue

            for variant_idx in range(1, multiplier):
                seed = variant_seed(random_seed, class_name, image_path.stem, variant_idx)
                try:
                    variant = augment_image(image, seed=seed)
                except Exception as exc:  # pragma: no cover - bergantung isi citra
                    logger.warning(
                        "Augmentasi gagal untuk %s varian %d (%s); dilewati.",
                        image_path.name,
                        variant_idx,
                        exc,
                    )
                    continue

                out_path = class_out / f"{image_path.stem}_aug{variant_idx:02d}.jpg"
                if cv2.imwrite(str(out_path), variant, write_params):
                    n_written += 1
                else:
                    logger.warning("Gagal menulis %s; dilewati.", out_path)

        per_class[class_name] = {
            "original": len(images),
            "output": n_written,
            "multiplier": multiplier,
            "replaced": len(replaced_stems),
        }
        total_original += len(images)
        total_output += n_written
        logger.info(
            "%-14s original=%5d -> output=%5d (multiplier %dx)",
            class_name,
            len(images),
            n_written,
            multiplier,
        )

    return {
        "split": input_dir.name,
        "augmented": bool(apply_augmentation),
        "per_class": per_class,
        "total_original": total_original,
        "total_output": total_output,
        "num_removed": num_removed,
    }


def copy_split(
    input_dir: str | Path,
    output_dir: str | Path,
    class_names: list[str] | None = None,
    clean: bool = False,
) -> dict:
    """Salin satu split tanpa augmentasi (untuk ``valid`` dan ``test``).

    Berkas disalin dengan nama asli (tanpa sufiks ``_orig``) agar jejaknya ke
    crop sumber tetap mudah dilacak saat error analysis.

    Args:
        input_dir: Folder split sumber.
        output_dir: Folder split tujuan.
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.
        clean: Bila ``True``, kosongkan folder kelas tujuan lebih dulu.

    Returns:
        Dict statistik dengan bentuk sama seperti
        :func:`generate_augmented_dataset`.

    Raises:
        FileNotFoundError: Bila ``input_dir`` tidak ada.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir tidak ditemukan: {input_dir}")

    per_class: dict[str, dict[str, int]] = {}
    total = 0
    num_removed = 0

    for class_name in names:
        class_in = input_dir / class_name
        class_out = output_dir / class_name
        num_removed += _prepare_output_dir(class_out, clean=clean)

        images = _list_images(class_in)
        for image_path in tqdm(
            images,
            desc=f"copy {input_dir.name}/{class_name}",
            unit="crop",
            ascii=True,
            leave=False,
        ):
            shutil.copy2(image_path, class_out / image_path.name)

        per_class[class_name] = {
            "original": len(images),
            "output": len(images),
            "multiplier": 1,
        }
        total += len(images)
        logger.info("%-14s disalin %5d crop (tanpa augmentasi)", class_name, len(images))

    return {
        "split": input_dir.name,
        "augmented": False,
        "per_class": per_class,
        "total_original": total,
        "total_output": total,
        "num_removed": num_removed,
    }


def build_dataset(
    input_root: str | Path,
    output_root: str | Path,
    augment_splits: tuple[str, ...] | list[str] = ("train",),
    copy_splits: tuple[str, ...] | list[str] = ("valid", "test"),
    multipliers: dict[str, int] | None = None,
    random_seed: int = RANDOM_SEED,
    class_names: list[str] | None = None,
    clean: bool = False,
    replace_fractions: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Bangun seluruh dataset siap latih dari ``crops_raw``.

    Split di ``augment_splits`` diaugmentasi; split di ``copy_splits`` hanya
    disalin. Split yang foldernya tidak ada dilewati dengan peringatan.

    Args:
        input_root: Root ``crops_raw``.
        output_root: Root ``crops`` (keluaran siap latih).
        augment_splits: Split yang diaugmentasi (normalnya hanya ``train``).
        copy_splits: Split yang hanya disalin (``valid``, ``test``).
        multipliers: Multiplier per kelas untuk split yang diaugmentasi.
        random_seed: Seed global.
        class_names: Urutan nama kelas.
        clean: Bila ``True``, kosongkan folder kelas tujuan lebih dulu.
        replace_fractions: Lihat :func:`generate_augmented_dataset`.

    Returns:
        Dict ``nama_split -> statistik`` untuk tiap split yang diproses.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    results: dict[str, dict] = {}

    for split in augment_splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            logger.warning("Split '%s' tidak ada di %s; dilewati.", split, input_root)
            continue
        results[split] = generate_augmented_dataset(
            input_dir=split_dir,
            output_dir=output_root / split,
            multipliers=multipliers,
            random_seed=random_seed,
            class_names=class_names,
            apply_augmentation=True,
            clean=clean,
            replace_fractions=replace_fractions,
        )

    for split in copy_splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            logger.warning("Split '%s' tidak ada di %s; dilewati.", split, input_root)
            continue
        results[split] = copy_split(
            input_dir=split_dir,
            output_dir=output_root / split,
            class_names=class_names,
            clean=clean,
        )

    return results


def format_augmentation_report(
    results: dict[str, dict],
    class_names: list[str] | None = None,
) -> str:
    """Susun laporan teks statistik augmentasi per split.

    Args:
        results: Keluaran :func:`build_dataset`.
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.

    Returns:
        Laporan multi-baris siap dicetak, memuat jumlah sebelum/sesudah per
        kelas dan rasio keseimbangan kelas.
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    lines: list[str] = ["Augmentation Report", "=" * 66]

    for split, stats in results.items():
        mode = "augmented" if stats["augmented"] else "copied"
        lines.append(f"\n[{split}] ({mode})")
        for name in names:
            info = stats["per_class"].get(name, {"original": 0, "output": 0, "multiplier": 1})
            lines.append(
                f"  {name:<15} {info['original']:>6,} -> {info['output']:>6,} "
                f"(multiplier: {info['multiplier']}x)"
            )
        lines.append(
            f"  {'TOTAL':<15} {stats['total_original']:>6,} -> {stats['total_output']:>6,}"
        )
        if stats["num_removed"]:
            lines.append(f"  (menghapus {stats['num_removed']:,} file lama sebelum menulis)")

        before = Counter({n: stats["per_class"].get(n, {}).get("original", 0) for n in names})
        after = Counter({n: stats["per_class"].get(n, {}).get("output", 0) for n in names})
        lines.append(f"  balance before : {_ratio_string(before, names)}")
        lines.append(f"  balance after  : {_ratio_string(after, names)}")

    return "\n".join(lines)


def _ratio_string(counts: Counter, names: list[str]) -> str:
    """Format rasio jumlah antar-kelas relatif terhadap kelas terbanyak."""
    largest = max((counts.get(n, 0) for n in names), default=0)
    if largest == 0:
        return "n/a (tidak ada sampel)"
    return " : ".join(f"{counts.get(n, 0) / largest:.1f}" for n in names)

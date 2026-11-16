"""Generate dataset CNN dari hasil segmentasi YOLOv12 (sumber crop utama).

Alur end-to-end penuh: YOLOv12 mendeteksi biji pada tiap foto, prediksi
dicocokkan ke anotasi ground truth via IoU (:mod:`src.evaluation.matching`),
lalu wilayah prediksi yang *matched* di-crop dan diberi label dari ground truth.
Hasilnya menjadi isi ``data/crops/{split}/{kelas}/`` yang dipakai melatih dan
menguji ketiga arsitektur CNN.

Konsekuensi metodologis yang harus selalu dilaporkan: biji yang tidak terdeteksi
YOLOv12 (*false negative*) tidak pernah sampai ke CNN, dan deteksi tanpa
pasangan ground truth (*false positive*) dibuang karena labelnya tidak diketahui.
Statistik ini disimpan ke ``matching_stats.json`` agar metrik CNN dapat dibaca
dalam konteks yang jujur.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from configs.base_config import IOU_MATCH_THRESHOLD
from src.evaluation.matching import crop_matched_regions, match_predictions_to_ground_truth
from src.utils.io import parse_yolo_label_file, save_json

logger = logging.getLogger(__name__)

#: Ekstensi gambar yang dipindai pada direktori split.
_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

#: Nama file statistik matching (relatif terhadap root keluaran crop).
_STATS_FILENAME: str = "matching_stats.json"


def load_yolo_model(yolo_model_path: str | Path):
    """Muat model YOLOv12 terlatih dari file bobot.

    Args:
        yolo_model_path: Path bobot ``.pt`` hasil pelatihan YOLOv12.

    Returns:
        Instance ``ultralytics.YOLO`` yang siap dipakai inferensi.

    Raises:
        ImportError: Bila package ``ultralytics`` belum terinstal.
        FileNotFoundError: Bila file bobot tidak ada.

    Example:
        >>> model = load_yolo_model("results/models/yolov12_best.pt")  # doctest: +SKIP
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - bergantung environment
        raise ImportError(
            "Package 'ultralytics' belum terinstal. Jalankan: pip install ultralytics"
        ) from exc

    yolo_model_path = Path(yolo_model_path)
    if not yolo_model_path.exists():
        raise FileNotFoundError(
            f"Bobot YOLOv12 tidak ditemukan: {yolo_model_path}. "
            "Latih YOLOv12 terlebih dahulu (notebooks/03_train_yolov12.ipynb)."
        )

    logger.info("Memuat model YOLOv12 dari %s", yolo_model_path)
    return YOLO(str(yolo_model_path))


def predict_boxes(yolo_model, image_path: str | Path, conf_threshold: float | None = None) -> list[dict]:
    """Jalankan YOLOv12 pada satu foto dan kembalikan bounding box biji.

    Args:
        yolo_model: Model ultralytics hasil :func:`load_yolo_model`.
        image_path: Path foto multi-bean.
        conf_threshold: Ambang confidence minimal. ``None`` memakai default
            ultralytics.

    Returns:
        Daftar dict ``{'bbox': (xmin, ymin, xmax, ymax), 'confidence': float}``
        dalam piksel, terurut dari confidence tertinggi.

    Example:
        >>> predict_boxes(model, "foto.jpg")[0]["confidence"]  # doctest: +SKIP
        0.93
    """
    kwargs: dict = {"verbose": False}
    if conf_threshold is not None:
        kwargs["conf"] = conf_threshold

    results = yolo_model.predict(str(image_path), **kwargs)
    detections: list[dict] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            detections.append(
                {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "confidence": float(conf),
                }
            )

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def _summarize(per_image: list[dict]) -> dict:
    """Agregasi statistik per foto menjadi ringkasan satu split."""
    n_gt = sum(item["num_ground_truths"] for item in per_image)
    n_matched = sum(item["true_positives"] for item in per_image)
    n_fp = sum(item["false_positives"] for item in per_image)
    n_fn = sum(item["false_negatives"] for item in per_image)
    n_crops = sum(item["crops_saved"] for item in per_image)

    return {
        "num_images": len(per_image),
        "num_ground_truths": n_gt,
        "num_matched": n_matched,
        "num_false_positives": n_fp,
        "num_false_negatives": n_fn,
        "num_crops_saved": n_crops,
        "yolo_recall": float(n_matched / (n_matched + n_fn)) if (n_matched + n_fn) > 0 else 0.0,
        "yolo_precision": float(n_matched / (n_matched + n_fp)) if (n_matched + n_fp) > 0 else 0.0,
    }


def _merge_stats_file(
    stats_path: Path,
    split_name: str,
    split_stats: dict,
    iou_threshold: float,
    yolo_model_path: str | Path,
) -> Path:
    """Tulis statistik satu split ke file bersama tanpa menghapus split lain."""
    data: dict = {}
    if stats_path.exists():
        try:
            loaded = json.loads(stats_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except json.JSONDecodeError:
            logger.warning("%s rusak; file statistik ditulis ulang dari awal.", stats_path)

    splits = data.get("splits") if isinstance(data.get("splits"), dict) else {}
    splits[split_name] = split_stats

    all_per_image = [item for split in splits.values() for item in split.get("per_image", [])]
    data.update(
        {
            "iou_threshold": float(iou_threshold),
            "yolo_weights": str(yolo_model_path),
            "splits": splits,
            "total": _summarize(all_per_image),
        }
    )
    return save_json(data, stats_path)


def generate_crops_from_yolo(
    yolo_model_path: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    class_names: list[str],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
    conf_threshold: float | None = None,
    yolo_model=None,
    stats_path: str | Path | None = None,
) -> dict:
    """Generate crop CNN satu split dari prediksi YOLOv12 yang matched dengan GT.

    Untuk tiap foto: YOLOv12 mendeteksi biji → prediksi dicocokkan ke anotasi
    ground truth (IoU greedy) → wilayah ``pred_bbox`` tiap pasangan di-crop dan
    disimpan ke ``output_dir/<class_names[gt_class_id]>/<stem>_bean_<idx>.jpg``.
    Foto tanpa file label diproses dengan ground truth kosong (semua deteksi
    tercatat sebagai false positive) dan hanya menghasilkan peringatan.

    Args:
        yolo_model_path: Path bobot YOLOv12 (dipakai bila ``yolo_model`` None).
        images_dir: Direktori foto split ini.
        labels_dir: Direktori label YOLO (``.txt``) berpasangan.
        output_dir: Direktori keluaran crop split ini (mis. ``data/crops/train``).
        class_names: Nama kelas; indeks daftar = ``class_id`` anotasi.
        iou_threshold: Ambang IoU minimal agar deteksi dianggap cocok dengan GT.
        conf_threshold: Ambang confidence deteksi YOLOv12 (opsional).
        yolo_model: Model ultralytics yang sudah dimuat; menghindari load ulang
            saat memproses banyak split.
        stats_path: Lokasi file statistik. Default ``output_dir/../matching_stats.json``.

    Returns:
        Dict ``{'summary': {...}, 'per_image': [...]}``; ``summary`` memuat
        jumlah GT, matched, FP, FN, serta recall & precision YOLOv12.

    Raises:
        FileNotFoundError: Bila ``images_dir``/``labels_dir`` tidak ada atau
            bobot YOLOv12 tidak ditemukan.

    Example:
        >>> hasil = generate_crops_from_yolo(
        ...     "results/models/yolov12_best.pt",
        ...     "data/annotated/train/images",
        ...     "data/annotated/train/labels",
        ...     "data/crops/train",
        ...     ["layak", "tidak_layak"],
        ... )  # doctest: +SKIP
        >>> hasil["summary"]["num_matched"]  # doctest: +SKIP
        842
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir tidak ditemukan: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels_dir tidak ditemukan: {labels_dir}")

    model = yolo_model if yolo_model is not None else load_yolo_model(yolo_model_path)

    # Siapkan subfolder keluaran per kelas agar split tetap konsisten
    # walau sebuah kelas tidak menghasilkan crop sama sekali.
    for name in class_names:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    if not image_paths:
        logger.warning("Tidak ada gambar di %s", images_dir)

    per_image: list[dict] = []
    split_name = output_dir.name

    for image_path in tqdm(image_paths, desc=f"crop {split_name}", unit="foto"):
        with Image.open(image_path) as img:
            img_w, img_h = img.size

        detections = predict_boxes(model, image_path, conf_threshold=conf_threshold)

        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            ground_truths = parse_yolo_label_file(label_path, img_w, img_h)
        else:
            logger.warning(
                "Label untuk %s tidak ditemukan; foto diproses tanpa ground truth.",
                image_path.name,
            )
            ground_truths = []

        matching = match_predictions_to_ground_truth(
            detections, ground_truths, iou_threshold=iou_threshold
        )
        matched_pairs = matching["matched_pairs"]

        saved = crop_matched_regions(
            image_path=image_path,
            matched_pairs=matched_pairs,
            output_dir=output_dir,
            class_names=class_names,
            image_stem=image_path.stem,
        )

        per_image.append(
            {
                "image": image_path.name,
                "num_detections": len(detections),
                "num_ground_truths": len(ground_truths),
                "true_positives": len(matched_pairs),
                "false_positives": len(matching["false_positives"]),
                "false_negatives": len(matching["false_negatives"]),
                "crops_saved": len(saved),
            }
        )

    summary = _summarize(per_image)
    split_stats = {"summary": summary, "per_image": per_image}

    stats_file = Path(stats_path) if stats_path is not None else output_dir.parent / _STATS_FILENAME
    _merge_stats_file(stats_file, split_name, split_stats, iou_threshold, yolo_model_path)

    logger.info(
        "Split '%s': %d foto | GT=%d | matched=%d | FN=%d | FP=%d | "
        "YOLO recall=%.4f precision=%.4f | crop tersimpan=%d",
        split_name,
        summary["num_images"],
        summary["num_ground_truths"],
        summary["num_matched"],
        summary["num_false_negatives"],
        summary["num_false_positives"],
        summary["yolo_recall"],
        summary["yolo_precision"],
        summary["num_crops_saved"],
    )
    if summary["num_false_negatives"] > 0:
        logger.warning(
            "%d biji ground truth pada split '%s' tidak terdeteksi YOLOv12 dan "
            "tidak masuk dataset CNN; laporkan angka ini bersama metrik CNN.",
            summary["num_false_negatives"],
            split_name,
        )

    return split_stats


def generate_all_splits(
    yolo_model_path: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    class_names: list[str],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    conf_threshold: float | None = None,
) -> dict:
    """Generate crop CNN untuk seluruh split sekaligus (train, valid, test).

    Model YOLOv12 dimuat sekali lalu dipakai ulang antar-split. File statistik
    lama dihapus lebih dulu agar hasilnya selalu mencerminkan run terakhir
    (deterministik, tanpa sisa entri usang).

    Args:
        yolo_model_path: Path bobot YOLOv12 terlatih.
        dataset_root: Root ekspor Roboflow (berisi ``<split>/images`` dan
            ``<split>/labels``).
        output_root: Root keluaran crop (mis. ``data/crops``).
        class_names: Nama kelas; indeks daftar = ``class_id`` anotasi.
        iou_threshold: Ambang IoU untuk matching.
        splits: Nama split yang diproses.
        conf_threshold: Ambang confidence deteksi YOLOv12 (opsional).

    Returns:
        Dict ``{'splits': {...}, 'total': {...}, 'stats_path': str}``; ``total``
        adalah agregasi seluruh split yang berhasil diproses.

    Raises:
        FileNotFoundError: Bila bobot YOLOv12 tidak ditemukan.

    Example:
        >>> hasil = generate_all_splits(
        ...     "results/models/yolov12_best.pt", "data/annotated", "data/crops",
        ...     ["layak", "tidak_layak"],
        ... )  # doctest: +SKIP
        >>> hasil["total"]["yolo_recall"]  # doctest: +SKIP
        0.96
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    stats_path = output_root / _STATS_FILENAME

    model = load_yolo_model(yolo_model_path)

    if stats_path.exists():
        stats_path.unlink()
        logger.info("Statistik matching lama dihapus: %s", stats_path)

    results: dict[str, dict] = {}
    for split in splits:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"

        if not images_dir.is_dir() or not labels_dir.is_dir():
            logger.warning("Split '%s' tidak lengkap (%s); dilewati.", split, images_dir)
            continue

        results[split] = generate_crops_from_yolo(
            yolo_model_path=yolo_model_path,
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=output_root / split,
            class_names=class_names,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
            yolo_model=model,
            stats_path=stats_path,
        )

    all_per_image = [item for split in results.values() for item in split["per_image"]]
    total = _summarize(all_per_image)

    logger.info(
        "Seluruh split selesai | GT=%d | matched=%d | FN=%d | FP=%d | "
        "YOLO recall=%.4f precision=%.4f",
        total["num_ground_truths"],
        total["num_matched"],
        total["num_false_negatives"],
        total["num_false_positives"],
        total["yolo_recall"],
        total["yolo_precision"],
    )
    logger.info("Statistik matching tersimpan di %s", stats_path)

    return {"splits": results, "total": total, "stats_path": str(stats_path)}

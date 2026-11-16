"""Pencocokan prediksi YOLOv12 dengan bounding box ground truth via IoU.

Modul ini dipakai pada **evaluasi tambahan** (sistem end-to-end): deteksi biji
oleh YOLOv12 harus dipasangkan dulu dengan anotasi ground truth agar label
kualitas (``layak`` / ``tidak_layak``) diketahui, sebelum crop hasil deteksi
diklasifikasi oleh CNN.

Konvensi bounding box di seluruh modul: tuple ``(x_min, y_min, x_max, y_max)``
dalam satuan piksel, dengan ``x_min < x_max`` dan ``y_min < y_max``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def compute_iou(box1: tuple, box2: tuple) -> float:
    """Hitung Intersection over Union (IoU) dua bounding box.

    Box yang tidak valid (bukan 4 nilai, nilai non-numerik, atau luas nol
    karena ``x_max <= x_min``) tidak menimbulkan error: fungsi mencatat
    peringatan dan mengembalikan ``0.0``.

    Args:
        box1: Bounding box pertama ``(x_min, y_min, x_max, y_max)`` dalam piksel.
        box2: Bounding box kedua dengan format sama.

    Returns:
        Nilai IoU dalam rentang ``0.0``–``1.0``.

    Example:
        >>> round(compute_iou((0, 0, 10, 10), (5, 5, 15, 15)), 3)
        0.143
    """
    try:
        x1_min, y1_min, x1_max, y1_max = (float(v) for v in box1)
        x2_min, y2_min, x2_max, y2_max = (float(v) for v in box2)
    except (TypeError, ValueError):
        logger.warning("Bounding box tidak valid: %r vs %r; IoU dianggap 0.", box1, box2)
        return 0.0

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    if area1 <= 0.0 or area2 <= 0.0:
        logger.warning("Bounding box berluas nol: %r vs %r; IoU dianggap 0.", box1, box2)
        return 0.0

    inter_w = min(x1_max, x2_max) - max(x1_min, x2_min)
    inter_h = min(y1_max, y2_max) - max(y1_min, y2_min)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0

    intersection = inter_w * inter_h
    union = area1 + area2 - intersection
    if union <= 0.0:  # pragma: no cover - hanya mungkin bila luas nol (sudah ditangani).
        return 0.0
    return float(intersection / union)


def _is_valid_box(box: object) -> bool:
    """Periksa apakah ``box`` berupa 4 nilai numerik yang membentuk luas positif."""
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        return False
    try:
        x_min, y_min, x_max, y_max = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    return x_max > x_min and y_max > y_min


def match_predictions_to_ground_truth(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Cocokkan prediksi ke ground truth dengan algoritma greedy berbasis IoU.

    Prediksi diproses berurutan dari confidence tertinggi. Untuk tiap prediksi,
    dicari ground truth dengan IoU terbesar yang belum terpakai; bila IoU
    ``>= iou_threshold`` pasangan dicatat sebagai *matched pair* (IoU tepat sama
    dengan threshold tetap dianggap cocok). Prediksi tanpa pasangan menjadi
    *false positive*, ground truth yang tersisa menjadi *false negative*.

    Entri dengan bbox tidak valid dilewati beserta peringatan (tidak crash).

    Args:
        predictions: Daftar dict ``{'bbox': (xmin, ymin, xmax, ymax),
            'confidence': float}`` hasil deteksi YOLOv12.
        ground_truths: Daftar dict ``{'bbox': (xmin, ymin, xmax, ymax),
            'class_id': int}`` dari anotasi Roboflow.
        iou_threshold: Ambang IoU minimal agar sebuah pasangan dianggap cocok.

    Returns:
        Dict dengan kunci:
            - ``matched_pairs``: daftar ``{'pred_bbox', 'gt_class_id', 'iou',
              'confidence'}``.
            - ``false_positives``: daftar ``{'bbox', 'confidence'}``.
            - ``false_negatives``: daftar ``{'bbox', 'class_id'}``.

    Example:
        >>> preds = [{'bbox': (0, 0, 10, 10), 'confidence': 0.9}]
        >>> gts = [{'bbox': (0, 0, 10, 10), 'class_id': 1}]
        >>> hasil = match_predictions_to_ground_truth(preds, gts)
        >>> len(hasil['matched_pairs']), hasil['matched_pairs'][0]['gt_class_id']
        (1, 1)
    """
    valid_preds: list[dict] = []
    for pred in predictions or []:
        bbox = pred.get("bbox") if isinstance(pred, dict) else None
        if not _is_valid_box(bbox):
            logger.warning("Prediksi dilewati karena bbox tidak valid: %r", pred)
            continue
        valid_preds.append(
            {"bbox": tuple(bbox), "confidence": float(pred.get("confidence", 0.0))}
        )

    valid_gts: list[dict] = []
    for gt in ground_truths or []:
        bbox = gt.get("bbox") if isinstance(gt, dict) else None
        if not _is_valid_box(bbox):
            logger.warning("Ground truth dilewati karena bbox tidak valid: %r", gt)
            continue
        valid_gts.append({"bbox": tuple(bbox), "class_id": int(gt.get("class_id", 0))})

    # Greedy: confidence tertinggi mendapat prioritas memilih ground truth.
    order = sorted(range(len(valid_preds)), key=lambda i: valid_preds[i]["confidence"], reverse=True)
    gt_used: list[bool] = [False] * len(valid_gts)

    matched_pairs: list[dict] = []
    false_positives: list[dict] = []

    for pred_idx in order:
        pred = valid_preds[pred_idx]
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(valid_gts):
            if gt_used[gt_idx]:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            gt_used[best_gt_idx] = True
            matched_pairs.append(
                {
                    "pred_bbox": pred["bbox"],
                    "gt_class_id": valid_gts[best_gt_idx]["class_id"],
                    "iou": float(best_iou),
                    "confidence": pred["confidence"],
                }
            )
        else:
            false_positives.append({"bbox": pred["bbox"], "confidence": pred["confidence"]})

    false_negatives = [
        {"bbox": gt["bbox"], "class_id": gt["class_id"]}
        for gt, used in zip(valid_gts, gt_used)
        if not used
    ]

    return {
        "matched_pairs": matched_pairs,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def crop_matched_regions(
    image_path: Path,
    matched_pairs: list[dict],
    output_dir: Path,
    class_names: list[str],
    image_stem: str,
) -> list[Path]:
    """Simpan crop wilayah prediksi yang cocok, dikelompokkan per kelas ground truth.

    Tiap ``pred_bbox`` di-crop dari citra asli lalu disimpan ke
    ``output_dir/<class_names[gt_class_id]>/<image_stem>_bean_<idx>.jpg``.
    Pasangan yang gagal (bbox di luar citra, ``class_id`` di luar rentang, crop
    kosong) dilewati dengan peringatan.

    Args:
        image_path: Path citra multi-bean sumber.
        matched_pairs: Keluaran ``matched_pairs`` dari
            :func:`match_predictions_to_ground_truth`.
        output_dir: Direktori keluaran; subfolder per kelas dibuat otomatis.
        class_names: Daftar nama kelas; indeks daftar = ``gt_class_id``.
        image_stem: Prefiks nama berkas crop (biasanya nama citra tanpa ekstensi).

    Returns:
        Daftar ``Path`` crop yang berhasil disimpan.

    Example:
        >>> pairs = [{'pred_bbox': (0, 0, 10, 10), 'gt_class_id': 0,
        ...           'iou': 0.9, 'confidence': 0.8}]
        >>> crop_matched_regions(Path('img.jpg'), pairs, Path('out'),
        ...                      ['layak', 'tidak_layak'], 'img')  # doctest: +SKIP
        [PosixPath('out/layak/img_bean_000.jpg')]
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("Gagal membaca gambar %s; tidak ada crop disimpan.", image_path)
        return []

    img_h, img_w = image.shape[:2]
    saved_paths: list[Path] = []

    for idx, pair in enumerate(matched_pairs or []):
        bbox = pair.get("pred_bbox")
        if not _is_valid_box(bbox):
            logger.warning("Matched pair %d pada %s punya bbox tidak valid; dilewati.", idx, image_stem)
            continue

        class_id = int(pair.get("gt_class_id", -1))
        if not 0 <= class_id < len(class_names):
            logger.warning(
                "gt_class_id %d pada %s di luar rentang kelas; dilewati.", class_id, image_stem
            )
            continue

        # Clamp ke batas citra agar slicing selalu aman.
        x1 = max(0, min(int(round(float(bbox[0]))), img_w - 1))
        y1 = max(0, min(int(round(float(bbox[1]))), img_h - 1))
        x2 = max(x1 + 1, min(int(round(float(bbox[2]))), img_w))
        y2 = max(y1 + 1, min(int(round(float(bbox[3]))), img_h))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning("Crop kosong pada %s indeks %d; dilewati.", image_stem, idx)
            continue

        class_dir = output_dir / class_names[class_id]
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"{image_stem}_bean_{idx:03d}.jpg"
        if not cv2.imwrite(str(out_path), crop):
            logger.warning("Gagal menulis crop ke %s; dilewati.", out_path)
            continue
        saved_paths.append(out_path)

    return saved_paths

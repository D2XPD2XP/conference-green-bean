"""Pipeline end-to-end: deteksi biji dengan YOLOv12 lalu klasifikasi tiap crop.

Modul ini melayani **evaluasi tambahan** (Tabel 4.2): berbeda dengan evaluasi
utama yang memakai crop dari koordinat ground truth, di sini crop berasal dari
prediksi YOLOv12 sehingga error segmentasi ikut terhitung.

Modul hanya *memakai* komponen pelatihan/evaluasi yang sudah ada (model factory,
transforms, metrics, matching) seperti memanggil library eksternal; tidak ada
kode pelatihan yang dimodifikasi.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from configs.base_config import CLASS_NAMES, DEVICE, IOU_MATCH_THRESHOLD, NUM_CLASSES
from configs.cnn_config import CNN_CONFIG
from src.data.transforms import get_transforms
from src.evaluation.matching import crop_matched_regions, match_predictions_to_ground_truth
from src.evaluation.metrics import compute_end_to_end_metrics
from src.models.model_factory import get_model
from src.utils.io import load_checkpoint, parse_yolo_label_file, save_json

logger = logging.getLogger(__name__)

#: Ekstensi gambar yang dipindai pada direktori test.
_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class EndToEndPipeline:
    """Pipeline terintegrasi YOLOv12 → CNN untuk citra multi-bean.

    Args:
        yolo_weights: Path bobot YOLOv12 terlatih (``.pt``).
        cnn_weights: Path checkpoint CNN terlatih (``.pt``, hasil
            ``src.training.train_cnn``).
        cnn_arch: Nama arsitektur CNN (``"resnet50"``, ``"mobilenetv3"``,
            ``"inceptionv3"``); menentukan transforms dan ukuran input.
        device: Device komputasi. Default mengikuti ``configs.base_config.DEVICE``.
        class_names: Nama kelas kualitas. Default ``CLASS_NAMES``.
        conf_threshold: Ambang confidence minimal deteksi YOLOv12. ``None``
            berarti memakai default ultralytics.
        iou_threshold: Ambang IoU untuk mencocokkan deteksi ke ground truth.

    Attributes:
        yolo_model: Model YOLOv12 terlatih (Ultralytics).
        cnn_model: Model CNN terlatih (PyTorch, mode ``eval``).
        cnn_transforms: Transform torchvision untuk crop (split ``test``).
        class_names: Daftar nama kelas.
        device: ``torch.device`` yang dipakai.

    Raises:
        ImportError: Bila package ``ultralytics`` belum terinstal.
        FileNotFoundError: Bila bobot YOLOv12 atau checkpoint CNN tidak ada.
        KeyError: Bila ``cnn_arch`` tidak dikenal.

    Example:
        >>> pipeline = EndToEndPipeline(
        ...     "results/models/yolov12_best.pt",
        ...     "results/models/resnet50_best_epoch12_acc0.93.pt",
        ...     "resnet50",
        ... )  # doctest: +SKIP
        >>> pipeline.predict_single_image(Path("foto.jpg"))["num_detections"]  # doctest: +SKIP
        24
    """

    def __init__(
        self,
        yolo_weights: str | Path,
        cnn_weights: str | Path,
        cnn_arch: str,
        device: str | torch.device = DEVICE,
        class_names: list[str] | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float = IOU_MATCH_THRESHOLD,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - bergantung environment
            raise ImportError(
                "Package 'ultralytics' belum terinstal. Jalankan: pip install ultralytics"
            ) from exc

        cnn_arch = cnn_arch.lower()
        if cnn_arch not in CNN_CONFIG:
            raise KeyError(
                f"Arsitektur '{cnn_arch}' tidak dikenal. Pilihan: {list(CNN_CONFIG)}"
            )

        yolo_weights = Path(yolo_weights)
        cnn_weights = Path(cnn_weights)
        if not yolo_weights.exists():
            raise FileNotFoundError(f"Bobot YOLOv12 tidak ditemukan: {yolo_weights}")
        if not cnn_weights.exists():
            raise FileNotFoundError(f"Checkpoint CNN tidak ditemukan: {cnn_weights}")

        self.device = torch.device(device)
        self.class_names = list(class_names) if class_names is not None else list(CLASS_NAMES)
        self.cnn_arch = cnn_arch
        self.conf_threshold = conf_threshold
        self.iou_threshold = float(iou_threshold)
        self.batch_size = int(CNN_CONFIG[cnn_arch]["batch_size"])

        logger.info("Memuat YOLOv12 dari %s", yolo_weights)
        self.yolo_model = YOLO(str(yolo_weights))

        logger.info("Memuat CNN %s dari %s", cnn_arch, cnn_weights)
        model = get_model(cnn_arch, num_classes=NUM_CLASSES, pretrained=False)
        load_checkpoint(model, cnn_weights, map_location=str(self.device))
        self.cnn_model = model.to(self.device).eval()

        self.cnn_transforms = get_transforms(cnn_arch, "test")

    # ------------------------------------------------------------------ #
    # Inferensi
    # ------------------------------------------------------------------ #
    def detect_beans(self, image_path: str | Path) -> list[dict]:
        """Jalankan YOLOv12 pada satu citra dan kembalikan deteksi biji.

        Args:
            image_path: Path citra multi-bean.

        Returns:
            Daftar dict ``{'bbox': (xmin, ymin, xmax, ymax), 'confidence': float}``
            dalam koordinat piksel, terurut dari confidence tertinggi.

        Example:
            >>> pipeline.detect_beans("foto.jpg")[0]["confidence"]  # doctest: +SKIP
            0.91
        """
        kwargs: dict = {"verbose": False}
        if self.conf_threshold is not None:
            kwargs["conf"] = self.conf_threshold

        results = self.yolo_model.predict(str(image_path), **kwargs)
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

    @torch.no_grad()
    def classify_crops(self, image: Image.Image, bboxes: list[tuple]) -> tuple[list[int], list[list[float]]]:
        """Klasifikasi sekumpulan wilayah citra dengan CNN.

        Args:
            image: Citra sumber (PIL, mode RGB).
            bboxes: Daftar bbox ``(xmin, ymin, xmax, ymax)`` dalam piksel.

        Returns:
            Tuple ``(predictions, probabilities)``; ``predictions`` berisi indeks
            kelas per bbox, ``probabilities`` berisi vektor probabilitas softmax.

        Example:
            >>> preds, probs = pipeline.classify_crops(img, [(0, 0, 50, 50)])  # doctest: +SKIP
            >>> len(preds), len(probs[0])
            (1, 2)
        """
        if not bboxes:
            return [], []

        img_w, img_h = image.size
        tensors: list[torch.Tensor] = []
        for bbox in bboxes:
            x1 = max(0, min(int(round(float(bbox[0]))), img_w - 1))
            y1 = max(0, min(int(round(float(bbox[1]))), img_h - 1))
            x2 = max(x1 + 1, min(int(round(float(bbox[2]))), img_w))
            y2 = max(y1 + 1, min(int(round(float(bbox[3]))), img_h))
            crop = image.crop((x1, y1, x2, y2))
            tensors.append(self.cnn_transforms(crop))

        predictions: list[int] = []
        probabilities: list[list[float]] = []

        # Inferensi per-batch agar memori GPU tetap aman pada foto padat biji.
        for start in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[start : start + self.batch_size]).to(self.device)
            outputs = self.cnn_model(batch)
            if not isinstance(outputs, torch.Tensor):  # InceptionOutputs, dsb.
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            predictions.extend(probs.argmax(dim=1).cpu().tolist())
            probabilities.extend(probs.cpu().tolist())

        return predictions, probabilities

    def predict_single_image(self, image_path: Path) -> dict:
        """Jalankan pipeline utuh (deteksi + klasifikasi) pada satu foto.

        Args:
            image_path: Path citra multi-bean.

        Returns:
            Dict berisi ``bboxes`` (list tuple piksel), ``confidences``
            (confidence YOLOv12), ``cnn_predictions`` (indeks kelas per biji),
            ``cnn_probabilities`` (probabilitas softmax), dan ``num_detections``.

        Raises:
            FileNotFoundError: Bila citra tidak ada.

        Example:
            >>> hasil = pipeline.predict_single_image(Path("foto.jpg"))  # doctest: +SKIP
            >>> hasil["num_detections"] == len(hasil["cnn_predictions"])
            True
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

        detections = self.detect_beans(image_path)
        bboxes = [d["bbox"] for d in detections]
        confidences = [d["confidence"] for d in detections]

        image = Image.open(image_path).convert("RGB")
        predictions, probabilities = self.classify_crops(image, bboxes)

        return {
            "bboxes": bboxes,
            "confidences": confidences,
            "cnn_predictions": predictions,
            "cnn_probabilities": probabilities,
            "num_detections": len(bboxes),
        }

    # ------------------------------------------------------------------ #
    # Evaluasi
    # ------------------------------------------------------------------ #
    def evaluate_on_test_set(
        self,
        test_images_dir: Path,
        test_labels_dir: Path,
        output_dir: Path,
        crops_dir: Path | None = None,
    ) -> dict:
        """Evaluasi sistem end-to-end pada seluruh split test.

        Untuk tiap foto: deteksi YOLOv12 → cocokkan ke ground truth (IoU) →
        crop wilayah yang cocok → klasifikasi CNN. Hasil seluruh foto
        diagregasi menjadi metrik CNN pada matched pairs plus statistik
        matching, lalu disimpan ke ``output_dir/end_to_end_results.json``.

        Args:
            test_images_dir: Direktori citra test.
            test_labels_dir: Direktori label YOLO (``.txt``) berpasangan.
            output_dir: Direktori keluaran JSON hasil.
            crops_dir: Bila diberikan, crop hasil deteksi ikut disimpan ke
                direktori ini (subfolder per kelas ground truth).

        Returns:
            Dict berisi ``metrics`` (keluaran
            :func:`~src.evaluation.metrics.compute_end_to_end_metrics`),
            ``per_image`` (statistik TP/FP/FN tiap foto), ``y_true``, ``y_pred``,
            dan ``config`` evaluasi.

        Raises:
            FileNotFoundError: Bila direktori citra atau label tidak ada.

        Example:
            >>> hasil = pipeline.evaluate_on_test_set(
            ...     Path("data/annotated/test/images"),
            ...     Path("data/annotated/test/labels"),
            ...     Path("results/end_to_end"),
            ... )  # doctest: +SKIP
            >>> hasil["metrics"]["cnn_accuracy"]  # doctest: +SKIP
            0.91
        """
        test_images_dir = Path(test_images_dir)
        test_labels_dir = Path(test_labels_dir)
        output_dir = Path(output_dir)

        if not test_images_dir.is_dir():
            raise FileNotFoundError(f"Direktori gambar test tidak ditemukan: {test_images_dir}")
        if not test_labels_dir.is_dir():
            raise FileNotFoundError(f"Direktori label test tidak ditemukan: {test_labels_dir}")

        image_paths = sorted(
            p for p in test_images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not image_paths:
            logger.warning("Tidak ada gambar di %s", test_images_dir)

        all_pairs: list[dict] = []
        all_preds: list[int] = []
        per_image: list[dict] = []
        total_fp = 0
        total_fn = 0

        for image_path in tqdm(image_paths, desc="Evaluasi end-to-end", unit="foto"):
            label_path = test_labels_dir / f"{image_path.stem}.txt"
            image = Image.open(image_path).convert("RGB")
            img_w, img_h = image.size

            detections = self.detect_beans(image_path)

            if label_path.exists():
                ground_truths = parse_yolo_label_file(label_path, img_w, img_h)
            else:
                logger.warning("Label untuk %s tidak ditemukan; dianggap tanpa GT.", image_path.name)
                ground_truths = []

            matching = match_predictions_to_ground_truth(
                detections, ground_truths, iou_threshold=self.iou_threshold
            )
            pairs = matching["matched_pairs"]
            n_fp = len(matching["false_positives"])
            n_fn = len(matching["false_negatives"])

            if crops_dir is not None:
                crop_matched_regions(
                    image_path=image_path,
                    matched_pairs=pairs,
                    output_dir=Path(crops_dir),
                    class_names=self.class_names,
                    image_stem=image_path.stem,
                )

            preds, _ = self.classify_crops(image, [pair["pred_bbox"] for pair in pairs])

            all_pairs.extend(pairs)
            all_preds.extend(preds)
            total_fp += n_fp
            total_fn += n_fn

            per_image.append(
                {
                    "image": image_path.name,
                    "num_detections": len(detections),
                    "num_ground_truths": len(ground_truths),
                    "true_positives": len(pairs),
                    "false_positives": n_fp,
                    "false_negatives": n_fn,
                }
            )

        metrics = compute_end_to_end_metrics(
            matched_pairs=all_pairs,
            cnn_predictions=all_preds,
            num_false_negatives=total_fn,
            num_false_positives=total_fp,
        )

        results = {
            "config": {
                "cnn_arch": self.cnn_arch,
                "iou_threshold": self.iou_threshold,
                "conf_threshold": self.conf_threshold,
                "class_names": self.class_names,
                "device": str(self.device),
                "num_images": len(image_paths),
            },
            "metrics": metrics,
            "per_image": per_image,
            "y_true": [int(pair["gt_class_id"]) for pair in all_pairs],
            "y_pred": [int(p) for p in all_preds],
        }

        output_path = save_json(results, output_dir / "end_to_end_results.json")
        logger.info("Hasil evaluasi end-to-end tersimpan di %s", output_path)
        return results

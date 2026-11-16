"""CLI: evaluasi satu arsitektur CNN pada split test.

Menghitung metrik klasifikasi tingkat biji pada ``data/crops/test``:

- metrik makro 5 kelas (accuracy, precision, recall, F1),
- metrik per kelas,
- metrik agregat "layak vs cacat" (semua kelas cacat digabung),
- confusion matrix ternormalisasi,
- efisiensi (parameter, ukuran, waktu inferensi, FLOPs).

Prediksi dan fitur penultimate ikut disimpan secara default, karena seluruh
notebook analisis (07-10) membacanya dari ``results/analysis/``.

Contoh::

    python -m scripts.evaluate --model resnet50 \\
        --checkpoint results/models/resnet50/resnet50_best_epoch12_acc0.93.pt
    python -m scripts.evaluate --model mobilenetv3 --checkpoint <path.pt> --no-save-features
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # CLI: figur hanya disimpan ke file, tanpa jendela.

from torch.utils.data import DataLoader  # noqa: E402

from configs.base_config import (  # noqa: E402
    ANALYSIS_DIR,
    CLASS_NAMES,
    CROPS_DIR,
    DEVICE,
    FIGURES_DIR,
    METRICS_DIR,
    NUM_CLASSES,
    TTA_MODE,
)
from configs.cnn_config import CNN_CONFIG
from src.analysis.confusion_analysis import (
    error_pattern_analysis,
    plot_confusion_matrix,
)
from src.analysis.efficiency_analysis import measure_model_efficiency
from src.analysis.feature_visualization import extract_features, save_features
from src.data.dataset import GreenBeanDataset
from src.data.transforms import get_transforms
from src.evaluation.metrics import (
    compute_defect_metrics,
    compute_metrics,
    compute_per_class_metrics,
)
from src.evaluation.efficiency import measure_inference_time
from src.evaluation.predict import collect_predictions, save_predictions
from src.evaluation.tta import TTA_CHOICES, TTA_VIEWS, wrap_tta
from src.models.model_factory import AVAILABLE_MODELS, get_model
from src.utils.io import load_checkpoint, save_json
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse argumen CLI."""
    parser = argparse.ArgumentParser(description="Evaluasi satu CNN pada data test.")
    parser.add_argument(
        "--model",
        required=True,
        choices=AVAILABLE_MODELS,
        help="Arsitektur yang dievaluasi.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path checkpoint (.pt).")
    parser.add_argument(
        "--crops-dir",
        default=None,
        help="Root dataset siap latih (default: data/crops).",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Jangan simpan CSV prediksi.",
    )
    parser.add_argument(
        "--no-save-features",
        action="store_true",
        help="Jangan simpan fitur penultimate (.npz).",
    )
    parser.add_argument(
        "--tta",
        choices=TTA_CHOICES,
        default=TTA_MODE,
        help=f"Test-time augmentation saat prediksi (default: {TTA_MODE}).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Root keluaran alternatif; metrik/figur/analisis ditulis ke "
            "<dir>/{metrics,figures,analysis}. Untuk eksperimen (mis. strategi "
            "imbalance lain) agar hasil utama di results/ tidak tertimpa."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Jalankan evaluasi satu arsitektur CNN pada split test."""
    args = parse_args()

    set_seed()
    logger = setup_logger(
        f"eval_{args.model}",
        log_file=f"eval_{args.model}.log",
        level=logging.INFO,
        attach_to=["src"],
    )

    config = CNN_CONFIG[args.model]
    batch_size = args.batch_size or int(config["batch_size"])
    crops_dir = Path(args.crops_dir) if args.crops_dir else CROPS_DIR
    input_size = int(config["input_size"])

    results_dir = Path(args.results_dir) if args.results_dir else None
    metrics_dir = results_dir / "metrics" if results_dir else METRICS_DIR
    figures_dir = results_dir / "figures" if results_dir else FIGURES_DIR
    analysis_dir = results_dir / "analysis" if results_dir else ANALYSIS_DIR
    for folder in (metrics_dir, figures_dir, analysis_dir):
        folder.mkdir(parents=True, exist_ok=True)

    # Dataloader test: shuffle=False wajib agar kolom path pada tabel prediksi
    # tetap sinkron dengan urutan dataset.
    test_ds = GreenBeanDataset(crops_dir / "test", transform=get_transforms(args.model, "test"))
    if len(test_ds) == 0:
        raise FileNotFoundError(
            f"Tidak ada crop di {crops_dir / 'test'}. Jalankan lebih dulu:\n"
            "  python -m scripts.crop_dataset\n"
            "  python -m scripts.augment_dataset"
        )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    logger.info("Distribusi kelas (test): %s", test_ds.class_counts())

    aug_counts = test_ds.augmentation_counts()
    if aug_counts["augmented"]:
        logger.warning(
            "Split test memuat %d citra teraugmentasi. Metrik test harus diukur "
            "pada citra asli; periksa --augment-splits pada scripts.augment_dataset.",
            aug_counts["augmented"],
        )

    # Model + checkpoint.
    model = get_model(args.model, num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    load_checkpoint(model, args.checkpoint, map_location=DEVICE)
    logger.info("Checkpoint dimuat: %s", args.checkpoint)

    # Prediksi (sekali jalan, dipakai untuk seluruh metrik). TTA hanya mengubah
    # cara prediksi; efisiensi & fitur tetap diukur pada model tunggal.
    tta_model = wrap_tta(model, args.tta)
    predictions = collect_predictions(
        tta_model, test_loader, device=DEVICE, class_names=CLASS_NAMES
    )
    y_true = predictions["y_true"].tolist()
    y_pred = predictions["y_pred"].tolist()

    metrics = compute_metrics(y_true, y_pred)
    per_class = compute_per_class_metrics(y_true, y_pred, CLASS_NAMES)
    defect = compute_defect_metrics(y_true, y_pred, CLASS_NAMES)
    efficiency = measure_model_efficiency(
        model,
        input_shape=(1, 3, input_size, input_size),
        device=DEVICE,
        checkpoint_path=args.checkpoint,
    )
    efficiency["tta"] = args.tta
    efficiency["tta_views"] = TTA_VIEWS[args.tta]
    if args.tta != "none":
        efficiency["inference_time_ms_tta"] = measure_inference_time(
            tta_model, input_shape=(1, 3, input_size, input_size), device=DEVICE
        )
    errors = error_pattern_analysis(y_true, y_pred, CLASS_NAMES, top_n=10)

    report = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "tta": args.tta,
        **metrics,
        **defect,
        "per_class": per_class,
        "efficiency": efficiency,
        # Kunci datar di bawah ini dipertahankan agar notebook & skrip lama
        # yang membaca metrics JSON tetap berfungsi.
        "total_params": efficiency["params_total"],
        "trainable_params": efficiency["params_trainable"],
        "inference_ms": efficiency["inference_time_ms"],
        "n_samples": len(y_true),
        "class_distribution": test_ds.class_counts(),
        "top_error_patterns": errors.to_dict(orient="records") if not errors.empty else [],
        "mean_confidence": round(float(predictions["confidence"].mean()), 4),
    }
    save_json(report, metrics_dir / f"{args.model}_test_metrics.json")

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=CLASS_NAMES,
        normalize="true",
        title=f"Confusion Matrix (recall) - {args.model}",
        output_path=figures_dir / f"{args.model}_confusion_matrix.png",
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=CLASS_NAMES,
        normalize=None,
        title=f"Confusion Matrix (jumlah) - {args.model}",
        output_path=figures_dir / f"{args.model}_confusion_matrix_counts.png",
    )

    if not args.no_save_predictions:
        path = save_predictions(predictions, analysis_dir / f"{args.model}_test_predictions.csv")
        logger.info("Prediksi test tersimpan: %s", path)

    if not args.no_save_features:
        features, labels = extract_features(
            model, test_loader, model_name=args.model, device=DEVICE
        )
        path = save_features(
            features,
            labels,
            analysis_dir / f"{args.model}_test_features.npz",
            paths=[str(p) for p in test_ds.paths],
        )
        logger.info("Fitur test tersimpan: %s (shape=%s)", path, features.shape)

    logger.info("Metrik makro: %s | defect: %s", metrics, defect)

    # --- Ringkasan ke stdout ---
    print(f"\n=== {args.model} (split test, n={len(y_true)}, tta={args.tta}) ===")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"{key:<12}: {metrics[key]:.4f}")
    print(f"{'params':<12}: {efficiency['params_total']:,} "
          f"(trainable {efficiency['params_trainable']:,})")
    print(f"{'size':<12}: {efficiency['model_size_mb']:.2f} MB")
    print(f"{'inference':<12}: {efficiency['inference_time_ms']:.2f} ms/citra")
    if "inference_time_ms_tta" in efficiency:
        print(f"{'inference+TTA':<12}: {efficiency['inference_time_ms_tta']:.2f} ms/citra "
              f"({efficiency['tta_views']} orientasi)")
    print(f"{'flops':<12}: {efficiency['flops'] / 1e9:.2f} GMAC")

    print("\n--- Per kelas ---")
    print(f"{'kelas':<16}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for name, stats in per_class.items():
        print(
            f"{name:<16}{stats['precision']:>10.4f}{stats['recall']:>9.4f}"
            f"{stats['f1']:>8.4f}{stats['support']:>9d}"
        )

    print("\n--- Layak vs cacat (semua kelas cacat digabung) ---")
    print(f"{'accuracy':<16}{defect['defect_accuracy']:.4f}")
    print(f"{'recall cacat':<16}{defect['defect_recall']:.4f}")
    print(
        f"{'miss rate':<16}{defect['defect_miss_rate']:.4f}  "
        f"<- proporsi biji cacat yang lolos sebagai layak"
    )

    if not errors.empty:
        print("\n--- Pola kesalahan terbanyak ---")
        for row in errors.head(5).itertuples():
            print(
                f"  {row.true_class:<15} -> {row.pred_class:<15} "
                f"{row.count:>4} ({row.share_of_true * 100:.1f}% dari kelasnya)"
            )

    print(f"\nMetrik & figur tersimpan di {metrics_dir} dan {figures_dir}.")


if __name__ == "__main__":
    main()

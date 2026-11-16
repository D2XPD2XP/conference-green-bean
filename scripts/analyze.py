"""CLI: hasilkan seluruh figur & tabel analisis untuk laporan/paper.

Satu perintah untuk semua analisis yang di notebook 06-10 dijalankan
terpisah. Artefak yang dibaca:

- ``results/models/<model>/<model>_history.json``      (kurva pembelajaran)
- ``results/metrics/<model>_test_metrics.json``        (metrik & efisiensi)
- ``results/analysis/<model>_test_predictions.csv``    (confidence & error)
- ``results/analysis/<model>_test_features.npz``       (t-SNE & UMAP)
- ``results/models/<model>/*.pt``                      (Grad-CAM)

Artefak yang belum ada dilewati dengan peringatan, jadi skrip ini aman
dijalankan di tengah pipeline.

Contoh::

    python -m scripts.analyze
    python -m scripts.analyze --models resnet50 mobilenetv3
    python -m scripts.analyze --skip-gradcam --skip-umap
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # CLI: tidak butuh backend interaktif.

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from configs.base_config import (  # noqa: E402
    ANALYSIS_DIR,
    CLASS_NAMES,
    CROPS_DIR,
    DEVICE,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
)
from src.analysis.confidence_analysis import (
    accuracy_vs_threshold,
    confidence_summary,
    plot_confidence_distribution,
    plot_confidence_per_class,
)
from src.analysis.confusion_analysis import (
    error_pattern_analysis,
    generate_classification_report_table,
    plot_confusion_matrices_comparison,
    plot_error_pattern_heatmap,
)
from src.analysis.efficiency_analysis import (
    efficiency_table,
    plot_efficiency_accuracy_tradeoff,
    plot_inference_time_comparison,
    plot_size_vs_metric,
)
from src.analysis.error_analysis import error_summary_by_class, plot_error_gallery
from src.analysis.feature_visualization import (
    cluster_separation_score,
    compute_tsne,
    compute_umap,
    load_features,
    plot_projection_comparison,
)
from src.analysis.learning_curves import (
    convergence_summary,
    load_all_histories,
    plot_learning_curves,
    plot_learning_curves_grid,
)
from src.evaluation.predict import load_predictions
from src.models.model_factory import AVAILABLE_MODELS, get_model
from src.utils.io import load_checkpoint, save_json
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = logging.getLogger("analyze")


def parse_args() -> argparse.Namespace:
    """Parse argumen CLI."""
    parser = argparse.ArgumentParser(
        description="Hasilkan seluruh figur & tabel analisis dari artefak di results/."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(AVAILABLE_MODELS),
        choices=AVAILABLE_MODELS,
        help="Model yang dianalisis (default: ketiganya).",
    )
    parser.add_argument(
        "--crops-dir",
        default=None,
        help="Root dataset siap latih, untuk sampel Grad-CAM (default: data/crops).",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Folder keluaran figur (default: results/figures).",
    )
    parser.add_argument(
        "--tables-dir",
        default=None,
        help="Folder keluaran tabel CSV (default: results/analysis).",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Folder checkpoint & history (default: results/models).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Root hasil alternatif (mis. results/experiments/focal_loss): membaca "
            "dan menulis <dir>/{models,metrics,analysis,figures}. Flag *-dir "
            "yang lebih spesifik tetap diutamakan."
        ),
    )
    parser.add_argument("--skip-gradcam", action="store_true", help="Lewati Grad-CAM.")
    parser.add_argument("--skip-umap", action="store_true", help="Lewati proyeksi UMAP.")
    parser.add_argument("--skip-tsne", action="store_true", help="Lewati proyeksi t-SNE.")
    parser.add_argument(
        "--gradcam-per-class",
        type=int,
        default=2,
        help="Jumlah sampel Grad-CAM per kelas (default: 2).",
    )
    parser.add_argument(
        "--max-feature-samples",
        type=int,
        default=2000,
        help="Batas sampel untuk t-SNE/UMAP agar tidak lama (default: 2000).",
    )
    parser.add_argument(
        "--error-samples",
        type=int,
        default=20,
        help="Jumlah biji di galeri error (default: 20).",
    )
    return parser.parse_args()


def _load_metrics(models: list[str]) -> dict[str, dict]:
    """Muat metrik test tiap model yang sudah dievaluasi."""
    reports: dict[str, dict] = {}
    for name in models:
        path = METRICS_DIR / f"{name}_test_metrics.json"
        if not path.exists():
            logger.warning("Metrik %s belum ada (%s); dilewati.", name, path)
            continue
        reports[name] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def _load_prediction_tables(models: list[str]) -> dict[str, pd.DataFrame]:
    """Muat tabel prediksi tiap model yang sudah dievaluasi."""
    tables: dict[str, pd.DataFrame] = {}
    for name in models:
        path = ANALYSIS_DIR / f"{name}_test_predictions.csv"
        if not path.exists():
            logger.warning(
                "Prediksi %s belum ada (%s); jalankan scripts.evaluate dulu.", name, path
            )
            continue
        tables[name] = load_predictions(path)
    return tables


def _find_checkpoint(model_name: str, models_dir: Path = MODELS_DIR) -> Path | None:
    """Temukan satu-satunya checkpoint model (ModelCheckpoint menyisakan satu)."""
    folder = models_dir / model_name
    if not folder.is_dir():
        return None
    checkpoints = sorted(folder.glob("*.pt"))
    if not checkpoints:
        return None
    if len(checkpoints) > 1:
        logger.warning(
            "Ada %d checkpoint di %s; memakai yang terbaru.", len(checkpoints), folder
        )
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
    return checkpoints[-1]


def _analyze_learning_curves(
    models: list[str], figures: Path, tables: Path, models_dir: Path = MODELS_DIR
) -> pd.DataFrame:
    """Kurva pembelajaran + tabel ringkasan konvergensi."""
    histories = load_all_histories(models, models_dir=models_dir)
    if not histories:
        logger.warning("Tidak ada history pelatihan; kurva pembelajaran dilewati.")
        return pd.DataFrame()

    plot_learning_curves(histories, output_path=figures / "learning_curves.png")
    plot_learning_curves_grid(histories, output_path=figures / "learning_curves_grid.png")

    summary = convergence_summary(histories)
    summary.to_csv(tables / "convergence_summary.csv", encoding="utf-8")
    print("\n--- Konvergensi ---")
    print(summary.to_string())
    return summary


def _analyze_confusion(
    predictions: dict[str, pd.DataFrame], figures: Path, tables: Path
) -> None:
    """Confusion matrix berdampingan, tabel per kelas, dan pola kesalahan."""
    if not predictions:
        return

    pairs = {
        name: (frame["y_true"].tolist(), frame["y_pred"].tolist())
        for name, frame in predictions.items()
    }
    plot_confusion_matrices_comparison(
        pairs,
        class_names=CLASS_NAMES,
        normalize="true",
        output_path=figures / "confusion_matrices_comparison.png",
    )

    per_class_frames = []
    for name, (y_true, y_pred) in pairs.items():
        table = generate_classification_report_table(y_true, y_pred, CLASS_NAMES)
        table.insert(0, "model", name)
        per_class_frames.append(table)

        plot_error_pattern_heatmap(
            y_true,
            y_pred,
            class_names=CLASS_NAMES,
            output_path=figures / f"{name}_error_pattern.png",
            title=f"Pola kesalahan - {name}",
        )
        patterns = error_pattern_analysis(y_true, y_pred, CLASS_NAMES, top_n=15)
        if not patterns.empty:
            patterns.insert(0, "model", name)
            patterns.to_csv(tables / f"{name}_error_patterns.csv", index=False, encoding="utf-8")

    combined = pd.concat(per_class_frames)
    combined.to_csv(tables / "per_class_metrics.csv", encoding="utf-8")
    print("\n--- Metrik per kelas ---")
    print(combined.to_string())


def _analyze_confidence(
    predictions: dict[str, pd.DataFrame], figures: Path, tables: Path
) -> pd.DataFrame:
    """Distribusi confidence, confidence per kelas, dan simulasi ambang."""
    rows = []
    for name, frame in predictions.items():
        plot_confidence_distribution(
            frame,
            output_path=figures / f"{name}_confidence_distribution.png",
            title=f"Distribusi confidence - {name}",
        )
        plot_confidence_per_class(
            frame,
            class_names=CLASS_NAMES,
            output_path=figures / f"{name}_confidence_per_class.png",
            title=f"Confidence per kelas - {name}",
        )

        summary = confidence_summary(frame)
        summary["model"] = name
        rows.append(summary)

        thresholds = accuracy_vs_threshold(frame)
        thresholds.insert(0, "model", name)
        thresholds.to_csv(
            tables / f"{name}_accuracy_vs_threshold.csv", index=False, encoding="utf-8"
        )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).set_index("model")
    frame.to_csv(tables / "confidence_summary.csv", encoding="utf-8")
    print("\n--- Confidence ---")
    print(frame.to_string())
    return frame


def _analyze_errors(
    predictions: dict[str, pd.DataFrame],
    figures: Path,
    tables: Path,
    n_samples: int,
) -> None:
    """Galeri biji yang salah diklasifikasi + ringkasan error per kelas."""
    for name, frame in predictions.items():
        summary = error_summary_by_class(frame, class_names=CLASS_NAMES)
        summary.insert(0, "model", name)
        summary.to_csv(tables / f"{name}_error_summary.csv", encoding="utf-8")

        if "path" not in frame.columns:
            logger.warning("Prediksi %s tanpa kolom path; galeri error dilewati.", name)
            continue

        wrong = frame.loc[~frame["correct"]].sort_values("confidence", ascending=False)
        plot_error_gallery(
            wrong.head(n_samples).reset_index(drop=True),
            n_cols=5,
            output_path=figures / f"{name}_error_gallery.png",
            title=f"Kesalahan paling percaya diri - {name}",
        )


def _analyze_features(
    models: list[str],
    figures: Path,
    tables: Path,
    max_samples: int,
    skip_tsne: bool,
    skip_umap: bool,
) -> pd.DataFrame:
    """Proyeksi t-SNE/UMAP dan silhouette score per arsitektur."""
    tsne_projections: dict[str, tuple] = {}
    umap_projections: dict[str, tuple] = {}
    rows = []

    for name in models:
        path = ANALYSIS_DIR / f"{name}_test_features.npz"
        if not path.exists():
            logger.warning("Fitur %s belum ada (%s); dilewati.", name, path)
            continue

        features, labels = load_features(path)
        if features.shape[0] > max_samples:
            features, labels = features[:max_samples], labels[:max_samples]

        rows.append(
            {
                "model": name,
                "n_samples": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "silhouette_score": round(cluster_separation_score(features, labels), 4),
            }
        )

        if not skip_tsne:
            tsne_projections[name] = (compute_tsne(features, random_seed=RANDOM_SEED), labels)
        if not skip_umap:
            try:
                umap_projections[name] = (
                    compute_umap(features, random_seed=RANDOM_SEED),
                    labels,
                )
            except ImportError as exc:
                logger.warning("UMAP dilewati: %s", exc)
                skip_umap = True

    if tsne_projections:
        plot_projection_comparison(
            tsne_projections,
            class_names=CLASS_NAMES,
            output_path=figures / "tsne_comparison.png",
            suptitle="t-SNE ruang fitur (split test)",
        )
    if umap_projections:
        plot_projection_comparison(
            umap_projections,
            class_names=CLASS_NAMES,
            output_path=figures / "umap_comparison.png",
            suptitle="UMAP ruang fitur (split test)",
        )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).set_index("model")
    frame.to_csv(tables / "feature_separation.csv", encoding="utf-8")
    print("\n--- Pemisahan ruang fitur ---")
    print(frame.to_string())
    return frame


def _analyze_efficiency(
    reports: dict[str, dict], figures: Path, tables: Path
) -> pd.DataFrame:
    """Tabel & plot trade-off efisiensi vs akurasi."""
    if not reports:
        return pd.DataFrame()

    merged: dict[str, dict] = {}
    for name, report in reports.items():
        efficiency = dict(report.get("efficiency", {}))
        efficiency.pop("device", None)
        merged[name] = {
            **efficiency,
            "accuracy": report.get("accuracy", 0.0),
            "f1": report.get("f1", 0.0),
            "defect_recall": report.get("defect_recall", 0.0),
        }

    plot_efficiency_accuracy_tradeoff(
        merged, metric="f1", output_path=figures / "efficiency_accuracy_tradeoff.png"
    )
    plot_inference_time_comparison(
        merged, output_path=figures / "inference_time_comparison.png"
    )
    plot_size_vs_metric(merged, metric="f1", output_path=figures / "size_vs_f1.png")

    table = efficiency_table(merged)
    table.to_csv(tables / "efficiency_comparison.csv", encoding="utf-8")
    print("\n--- Efisiensi ---")
    print(table.to_string())
    return table


def _analyze_gradcam(
    models: list[str],
    crops_dir: Path,
    figures: Path,
    n_per_class: int,
    models_dir: Path = MODELS_DIR,
) -> None:
    """Grad-CAM berdampingan untuk ketiga arsitektur pada sampel yang sama."""
    # Impor di dalam fungsi: pytorch-grad-cam hanya dibutuhkan di tahap ini.
    from src.analysis.gradcam import GradCAMAnalyzer, compare_architectures, select_sample_images

    test_dir = crops_dir / "test"
    if not test_dir.is_dir():
        logger.warning("Split test tidak ada di %s; Grad-CAM dilewati.", test_dir)
        return

    loaded: dict[str, object] = {}
    for name in models:
        checkpoint = _find_checkpoint(name, models_dir=models_dir)
        if checkpoint is None:
            logger.warning("Checkpoint %s tidak ditemukan; Grad-CAM dilewati.", name)
            continue

        model = get_model(name, num_classes=NUM_CLASSES, pretrained=False)
        try:
            load_checkpoint(model, checkpoint, map_location=DEVICE)
        except RuntimeError as exc:
            # Penyebab paling umum: checkpoint dari pipeline versi lama dengan
            # jumlah kelas berbeda (2 kelas accepted/rejected vs 5 kelas).
            logger.warning(
                "Checkpoint %s tidak cocok dengan model %d kelas (%s). "
                "Latih ulang model ini sebelum menganalisisnya.",
                checkpoint.name,
                NUM_CLASSES,
                str(exc).splitlines()[0],
            )
            print(
                f"PERINGATAN: {checkpoint.name} tampaknya dari versi pipeline lama "
                f"(jumlah kelas berbeda). Grad-CAM {name} dilewati."
            )
            continue

        loaded[name] = model
        logger.info("Grad-CAM memuat %s dari %s", name, checkpoint.name)

    if not loaded:
        return

    paths, labels = select_sample_images(
        test_dir, class_names=CLASS_NAMES, n_per_class=n_per_class, seed=RANDOM_SEED
    )
    if not paths:
        logger.warning("Tidak ada sampel citra untuk Grad-CAM.")
        return

    compare_architectures(
        loaded,  # type: ignore[arg-type]
        paths,
        true_labels=labels,
        class_names=CLASS_NAMES,
        device=DEVICE,
        output_path=figures / "gradcam_comparison.png",
    )

    for name, model in loaded.items():
        analyzer = GradCAMAnalyzer(model, name, device=DEVICE)  # type: ignore[arg-type]
        analyzer.visualize_batch(
            paths[: min(len(paths), 6)],
            true_labels=labels[: min(len(labels), 6)],
            class_names=CLASS_NAMES,
            output_path=figures / f"{name}_gradcam.png",
        )


def main() -> None:
    """Jalankan seluruh analisis yang artefaknya tersedia."""
    args = parse_args()

    # Masukan (metrik, prediksi, fitur) dibaca helper lewat konstanta modul,
    # jadi --results-dir mengalihkan keduanya di sini.
    global METRICS_DIR, ANALYSIS_DIR
    root = Path(args.results_dir) if args.results_dir else None
    if root is not None:
        METRICS_DIR = root / "metrics"
        ANALYSIS_DIR = root / "analysis"

    set_seed()
    setup_logger("analyze", log_file="analyze.log", level=logging.INFO, attach_to=["src"])

    default_figures = root / "figures" if root else FIGURES_DIR
    default_models = root / "models" if root else MODELS_DIR
    figures = Path(args.figures_dir) if args.figures_dir else default_figures
    tables = Path(args.tables_dir) if args.tables_dir else ANALYSIS_DIR
    crops_dir = Path(args.crops_dir) if args.crops_dir else CROPS_DIR
    models_dir = Path(args.models_dir) if args.models_dir else default_models
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    models: list[str] = list(args.models)
    reports = _load_metrics(models)
    predictions = _load_prediction_tables(models)

    print(f"Model dianalisis   : {models}")
    print(f"Metrik tersedia    : {sorted(reports) or 'belum ada'}")
    print(f"Prediksi tersedia  : {sorted(predictions) or 'belum ada'}")

    # plt.close('all') setelah tiap bagian: figur matplotlib tidak dilepas
    # otomatis, dan analisis penuh membuat puluhan figur sekaligus.
    convergence = _analyze_learning_curves(models, figures, tables, models_dir=models_dir)
    plt.close("all")

    _analyze_confusion(predictions, figures, tables)
    plt.close("all")

    confidence = _analyze_confidence(predictions, figures, tables)
    plt.close("all")

    _analyze_errors(predictions, figures, tables, n_samples=args.error_samples)
    plt.close("all")
    separation = _analyze_features(
        models,
        figures,
        tables,
        max_samples=args.max_feature_samples,
        skip_tsne=args.skip_tsne,
        skip_umap=args.skip_umap,
    )
    plt.close("all")

    efficiency = _analyze_efficiency(reports, figures, tables)
    plt.close("all")

    if not args.skip_gradcam:
        _analyze_gradcam(
            models,
            crops_dir,
            figures,
            n_per_class=args.gradcam_per_class,
            models_dir=models_dir,
        )
        plt.close("all")

    # Ringkasan perbandingan utama.
    if reports:
        comparison = pd.DataFrame(
            [
                {
                    "model": name,
                    "accuracy": report.get("accuracy"),
                    "precision": report.get("precision"),
                    "recall": report.get("recall"),
                    "f1": report.get("f1"),
                    "defect_recall": report.get("defect_recall"),
                    "defect_miss_rate": report.get("defect_miss_rate"),
                    "params_total": report.get("total_params"),
                    "inference_ms": report.get("inference_ms"),
                }
                for name, report in reports.items()
            ]
        ).set_index("model")
        if "f1" in comparison.columns:
            comparison = comparison.sort_values("f1", ascending=False)
        comparison.to_csv(tables / "model_comparison.csv", encoding="utf-8")
        print("\n--- Perbandingan utama (urut F1 macro) ---")
        print(comparison.to_string())

        save_json(
            {
                "models": models,
                "evaluated": sorted(reports),
                "comparison": comparison.reset_index().to_dict(orient="records"),
                "convergence": convergence.reset_index().to_dict(orient="records")
                if not convergence.empty
                else [],
                "confidence": confidence.reset_index().to_dict(orient="records")
                if not confidence.empty
                else [],
                "feature_separation": separation.reset_index().to_dict(orient="records")
                if not separation.empty
                else [],
                "efficiency": efficiency.reset_index().to_dict(orient="records")
                if not efficiency.empty
                else [],
            },
            tables / "analysis_summary.json",
        )

    print(f"\nFigur tersimpan di : {figures}")
    print(f"Tabel tersimpan di : {tables}")


if __name__ == "__main__":
    main()

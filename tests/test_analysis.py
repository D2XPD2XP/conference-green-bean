"""Smoke test modul analisis.

Fokusnya bukan menguji estetika gambar, melainkan memastikan tiap fungsi
analisis berjalan sampai selesai pada data 5 kelas dan benar-benar menulis
berkas keluaran — sehingga kegagalan API (mis. perubahan versi matplotlib atau
albumentations) tertangkap di `pytest`, bukan saat pipeline panjang berjalan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from configs.base_config import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES
from configs.cnn_config import CLASSIFIER_ATTR, TARGET_LAYERS
from src.analysis.confidence_analysis import (
    accuracy_vs_threshold,
    confidence_summary,
    identify_uncertain_predictions,
    plot_confidence_distribution,
    plot_confidence_per_class,
)
from src.analysis.confusion_analysis import (
    error_pattern_analysis,
    generate_classification_report_table,
    plot_confusion_matrices_comparison,
    plot_confusion_matrix,
    plot_error_pattern_heatmap,
)
from src.analysis.efficiency_analysis import (
    count_flops,
    efficiency_table,
    measure_model_efficiency,
    plot_efficiency_accuracy_tradeoff,
    plot_inference_time_comparison,
)
from src.analysis.error_analysis import error_summary_by_class, plot_error_gallery
from src.analysis.feature_visualization import (
    cluster_separation_score,
    compute_tsne,
    extract_features,
    load_features,
    plot_feature_projection,
    save_features,
)
from src.analysis.gradcam import resolve_target_layer
from src.analysis.learning_curves import (
    convergence_summary,
    plot_learning_curves,
    plot_learning_curves_grid,
)
from src.analysis.style import class_palette, tensor_to_image
from src.models.model_factory import AVAILABLE_MODELS, get_model


@pytest.fixture()
def predictions() -> pd.DataFrame:
    """Tabel prediksi sintetis 5 kelas dengan sebagian prediksi salah."""
    rng = np.random.default_rng(3)
    n = 60
    y_true = rng.integers(0, NUM_CLASSES, size=n)
    y_pred = y_true.copy()
    # Rusak 20 prediksi agar ada kesalahan untuk dianalisis.
    y_pred[:20] = (y_pred[:20] + 1) % NUM_CLASSES

    probs = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
    frame = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "true_label": [CLASS_NAMES[i] for i in y_true],
            "pred_label": [CLASS_NAMES[i] for i in y_pred],
            "confidence": probs.max(axis=1),
        }
    )
    frame["correct"] = frame["y_true"] == frame["y_pred"]
    for index, name in enumerate(CLASS_NAMES):
        frame[f"prob_{name}"] = probs[:, index]
    return frame


@pytest.fixture()
def histories() -> dict[str, dict]:
    """History pelatihan sintetis untuk dua model."""
    return {
        "resnet50": {
            "train_loss": [1.4, 1.0, 0.8],
            "train_acc": [0.4, 0.6, 0.7],
            "val_loss": [1.5, 1.1, 0.9],
            "val_acc": [0.35, 0.55, 0.65],
            "best_val_acc": 0.65,
            "best_epoch": 3,
            "epochs_ran": 3,
            "imbalance_strategy": "class_weights",
        },
        "mobilenetv3": {
            "train_loss": [1.5, 1.1, 0.7],
            "train_acc": [0.3, 0.6, 0.8],
            # val_loss berbalik naik: sinyal overfitting yang harus terdeteksi.
            "val_loss": [1.6, 1.0, 1.2],
            "val_acc": [0.3, 0.6, 0.58],
            "best_val_acc": 0.6,
            "best_epoch": 2,
            "epochs_ran": 3,
            "imbalance_strategy": "focal_loss",
        },
    }


# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
def test_palette_covers_every_class() -> None:
    """Tiap kelas punya warna sendiri yang unik."""
    colors = class_palette(CLASS_NAMES)
    assert len(colors) == NUM_CLASSES
    assert len(set(colors)) == NUM_CLASSES
    assert all(name in CLASS_COLORS for name in CLASS_NAMES)


def test_tensor_to_image_denormalizes_to_unit_range() -> None:
    """Hasil denormalisasi berada di rentang 0..1 dengan bentuk HWC."""
    image = tensor_to_image(torch.randn(3, 16, 16))
    assert image.shape == (16, 16, 3)
    assert image.min() >= 0.0 and image.max() <= 1.0


# --------------------------------------------------------------------------- #
# confusion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("normalize", ["true", "pred", None])
def test_confusion_matrix_saves_file(predictions, tmp_path: Path, normalize) -> None:
    """Confusion matrix tersimpan untuk setiap mode normalisasi."""
    out = tmp_path / f"cm_{normalize}.png"
    plot_confusion_matrix(
        predictions["y_true"],
        predictions["y_pred"],
        class_names=CLASS_NAMES,
        normalize=normalize,
        output_path=out,
    )
    assert out.exists() and out.stat().st_size > 0


def test_confusion_matrix_rejects_bad_normalize(predictions) -> None:
    """Mode normalisasi tak dikenal memicu ValueError."""
    with pytest.raises(ValueError):
        plot_confusion_matrix(
            predictions["y_true"], predictions["y_pred"], normalize="kolom"
        )


def test_confusion_comparison_saves_file(predictions, tmp_path: Path) -> None:
    """Confusion matrix berdampingan beberapa model tersimpan."""
    out = tmp_path / "cm_comparison.png"
    pairs = {
        "resnet50": (predictions["y_true"], predictions["y_pred"]),
        "mobilenetv3": (predictions["y_true"], predictions["y_true"]),
    }
    plot_confusion_matrices_comparison(pairs, class_names=CLASS_NAMES, output_path=out)
    assert out.exists()


def test_classification_report_table_structure(predictions) -> None:
    """Tabel per kelas memuat seluruh kelas plus rata-rata."""
    table = generate_classification_report_table(
        predictions["y_true"], predictions["y_pred"], CLASS_NAMES
    )
    for name in CLASS_NAMES:
        assert name in table.index
    assert "macro avg" in table.index
    assert list(table.columns) == ["precision", "recall", "f1", "support"]


def test_error_pattern_excludes_diagonal(predictions) -> None:
    """Pola kesalahan tidak memuat prediksi benar (true == pred)."""
    patterns = error_pattern_analysis(
        predictions["y_true"], predictions["y_pred"], CLASS_NAMES
    )
    assert not patterns.empty
    assert (patterns["true_class"] != patterns["pred_class"]).all()
    assert (patterns["share_of_true"] <= 1.0).all()


def test_error_pattern_empty_when_perfect() -> None:
    """Prediksi sempurna menghasilkan tabel pola kesalahan kosong."""
    labels = list(range(NUM_CLASSES))
    assert error_pattern_analysis(labels, labels, CLASS_NAMES).empty


def test_error_pattern_heatmap_saves_file(predictions, tmp_path: Path) -> None:
    """Heatmap pola kesalahan tersimpan."""
    out = tmp_path / "pattern.png"
    plot_error_pattern_heatmap(
        predictions["y_true"], predictions["y_pred"], CLASS_NAMES, output_path=out
    )
    assert out.exists()


# --------------------------------------------------------------------------- #
# confidence
# --------------------------------------------------------------------------- #
def test_confidence_plots_save_files(predictions, tmp_path: Path) -> None:
    """Histogram dan box plot confidence tersimpan."""
    hist = tmp_path / "conf_hist.png"
    box = tmp_path / "conf_box.png"
    plot_confidence_distribution(predictions, output_path=hist)
    plot_confidence_per_class(predictions, class_names=CLASS_NAMES, output_path=box)
    assert hist.exists() and box.exists()


def test_confidence_summary_keys(predictions) -> None:
    """Ringkasan confidence memuat selisih benar-salah."""
    summary = confidence_summary(predictions)
    assert set(summary) == {
        "mean_confidence",
        "mean_confidence_correct",
        "mean_confidence_wrong",
        "confidence_gap",
        "share_below_0.5",
    }


def test_confidence_requires_columns() -> None:
    """Tabel tanpa kolom wajib memunculkan KeyError yang menjelaskan sebabnya."""
    with pytest.raises(KeyError):
        confidence_summary(pd.DataFrame({"foo": [1, 2]}))


def test_uncertain_predictions_filtered_and_sorted(predictions) -> None:
    """Hasil filter hanya memuat confidence di bawah ambang, terurut naik."""
    subset = identify_uncertain_predictions(predictions, threshold=0.5)
    assert (subset["confidence"] < 0.5).all()
    assert subset["confidence"].is_monotonic_increasing


def test_accuracy_vs_threshold_monotonic_coverage(predictions) -> None:
    """Cakupan menurun saat ambang dinaikkan."""
    table = accuracy_vs_threshold(predictions)
    assert list(table.columns) == [
        "threshold",
        "coverage",
        "accuracy_on_covered",
        "n_referred",
    ]
    assert table["coverage"].is_monotonic_decreasing


# --------------------------------------------------------------------------- #
# error analysis
# --------------------------------------------------------------------------- #
def test_error_summary_by_class(predictions) -> None:
    """Ringkasan error memuat semua kelas dan tingkat error 0..1."""
    summary = error_summary_by_class(predictions, class_names=CLASS_NAMES)
    assert len(summary) == NUM_CLASSES
    assert ((summary["error_rate"] >= 0) & (summary["error_rate"] <= 1)).all()


def test_error_gallery_requires_path_column(predictions) -> None:
    """Tanpa kolom path, galeri menolak dengan pesan yang jelas."""
    with pytest.raises(KeyError):
        plot_error_gallery(predictions)


def test_error_gallery_handles_missing_images(predictions, tmp_path: Path) -> None:
    """Citra yang hilang tidak menggagalkan galeri, hanya diberi tanda."""
    wrong = predictions.loc[~predictions["correct"]].head(4).copy()
    wrong["path"] = str(tmp_path / "tidak_ada.jpg")
    out = tmp_path / "gallery.png"
    plot_error_gallery(wrong.reset_index(drop=True), n_cols=2, output_path=out)
    assert out.exists()


def test_error_gallery_handles_empty_frame(tmp_path: Path) -> None:
    """Tanpa kesalahan sama sekali, galeri tetap menghasilkan figur."""
    empty = pd.DataFrame(columns=["path", "true_label", "pred_label", "confidence"])
    out = tmp_path / "gallery_empty.png"
    plot_error_gallery(empty, output_path=out)
    assert out.exists()


# --------------------------------------------------------------------------- #
# feature space
# --------------------------------------------------------------------------- #
def test_feature_roundtrip_and_projection(tmp_path: Path) -> None:
    """Fitur dapat disimpan, dimuat, diproyeksikan, dan diplot."""
    rng = np.random.default_rng(1)
    # Dua klaster yang jelas terpisah agar silhouette-nya positif.
    features = np.vstack(
        [rng.normal(0.0, 0.3, size=(30, 8)), rng.normal(5.0, 0.3, size=(30, 8))]
    ).astype(np.float32)
    labels = np.array([0] * 30 + [1] * 30)

    path = save_features(features, labels, tmp_path / "feat.npz")
    loaded_features, loaded_labels = load_features(path)
    assert loaded_features.shape == features.shape
    assert np.array_equal(loaded_labels, labels)

    assert cluster_separation_score(features, labels) > 0.5

    projection = compute_tsne(features, perplexity=10.0)
    assert projection.shape == (60, 2)

    out = tmp_path / "proj.png"
    plot_feature_projection(projection, labels, class_names=CLASS_NAMES, output_path=out)
    assert out.exists()


def test_cluster_separation_single_class_is_nan() -> None:
    """Satu kelas saja -> silhouette tidak terdefinisi (NaN), bukan exception."""
    score = cluster_separation_score(np.random.rand(10, 4), np.zeros(10, dtype=int))
    assert np.isnan(score)


def test_tsne_rejects_tiny_input() -> None:
    """Sampel < 3 memunculkan ValueError."""
    with pytest.raises(ValueError):
        compute_tsne(np.random.rand(2, 4))


def test_load_features_missing_file(tmp_path: Path) -> None:
    """Berkas fitur yang tidak ada memunculkan FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_features(tmp_path / "tidak_ada.npz")


# --------------------------------------------------------------------------- #
# model-dependent: hook fitur, target layer, efisiensi
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", AVAILABLE_MODELS)
def test_target_layer_resolvable(name: str) -> None:
    """Target layer Grad-CAM tiap arsitektur dapat diselesaikan."""
    model = get_model(name, num_classes=NUM_CLASSES, pretrained=False)
    layer = resolve_target_layer(model, TARGET_LAYERS[name])
    assert isinstance(layer, torch.nn.Module)


def test_resolve_target_layer_unknown_attribute() -> None:
    """Path atribut yang salah memunculkan AttributeError."""
    model = get_model("resnet50", num_classes=NUM_CLASSES, pretrained=False)
    with pytest.raises(AttributeError):
        resolve_target_layer(model, "layer_tidak_ada")


@pytest.mark.parametrize("name", AVAILABLE_MODELS)
def test_classifier_attr_exists(name: str) -> None:
    """Atribut head yang didaftarkan di config benar-benar ada di model."""
    model = get_model(name, num_classes=NUM_CLASSES, pretrained=False)
    assert hasattr(model, CLASSIFIER_ATTR[name])


def test_extract_features_shape() -> None:
    """Fitur penultimate ResNet-50 berdimensi 2048 per sampel."""
    model = get_model("resnet50", num_classes=NUM_CLASSES, pretrained=False)
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, NUM_CLASSES, (4,))
    loader = [(images, labels)]

    features, extracted_labels = extract_features(
        model, loader, model_name="resnet50", device="cpu"
    )

    assert features.shape == (4, 2048)
    assert extracted_labels.tolist() == labels.tolist()


def test_count_flops_scales_with_input_size() -> None:
    """Input lebih besar menghasilkan estimasi FLOPs lebih besar."""
    model = get_model("resnet50", num_classes=NUM_CLASSES, pretrained=False)
    small = count_flops(model, input_shape=(1, 3, 112, 112), device="cpu")
    large = count_flops(model, input_shape=(1, 3, 224, 224), device="cpu")
    assert 0 < small < large


def test_measure_efficiency_keys() -> None:
    """Pengukuran efisiensi memuat seluruh metrik yang dipakai laporan."""
    model = get_model("mobilenetv3", num_classes=NUM_CLASSES, pretrained=False)
    result = measure_model_efficiency(
        model, input_shape=(1, 3, 224, 224), device="cpu", n_runs=2, n_warmup=1
    )

    for key in (
        "params_total",
        "params_trainable",
        "model_size_mb",
        "inference_time_ms",
        "throughput_img_per_s",
        "flops",
    ):
        assert key in result
    assert result["params_trainable"] < result["params_total"]
    assert result["inference_time_ms"] > 0


def test_efficiency_plots_and_table(tmp_path: Path) -> None:
    """Tabel dan plot trade-off efisiensi tersimpan."""
    results = {
        "resnet50": {"params_total": 23_500_000, "inference_time_ms": 5.4, "f1": 0.81},
        "mobilenetv3": {"params_total": 4_200_000, "inference_time_ms": 5.2, "f1": 0.82},
    }
    table = efficiency_table(results)
    assert list(table.index) == ["mobilenetv3", "resnet50"]  # urut F1 menurun

    tradeoff = tmp_path / "tradeoff.png"
    timing = tmp_path / "timing.png"
    plot_efficiency_accuracy_tradeoff(results, output_path=tradeoff)
    plot_inference_time_comparison(results, output_path=timing)
    assert tradeoff.exists() and timing.exists()


def test_efficiency_plot_rejects_empty() -> None:
    """Dict kosong memicu ValueError, bukan figur kosong."""
    with pytest.raises(ValueError):
        plot_efficiency_accuracy_tradeoff({})


# --------------------------------------------------------------------------- #
# learning curves
# --------------------------------------------------------------------------- #
def test_learning_curve_plots_save_files(histories, tmp_path: Path) -> None:
    """Kurva pembelajaran gabungan dan grid tersimpan."""
    combined = tmp_path / "curves.png"
    grid = tmp_path / "curves_grid.png"
    plot_learning_curves(histories, output_path=combined)
    plot_learning_curves_grid(histories, output_path=grid)
    assert combined.exists() and grid.exists()


def test_convergence_summary_detects_overfitting(histories) -> None:
    """val_loss yang berbalik naik terbaca sebagai sinyal overfitting."""
    summary = convergence_summary(histories)

    assert summary.loc["resnet50", "overfitting_signal"] == 0.0
    assert summary.loc["mobilenetv3", "overfitting_signal"] > 0.0
    assert summary.loc["mobilenetv3", "imbalance_strategy"] == "focal_loss"


def test_learning_curves_rejects_empty() -> None:
    """History kosong memicu ValueError."""
    with pytest.raises(ValueError):
        plot_learning_curves({})


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


def test_select_correct_samples_keeps_only_agreed_correct(tmp_path: Path) -> None:
    """Hanya sampel yang diprediksi benar oleh SEMUA model yang dipilih."""
    from PIL import Image

    from src.analysis.gradcam import select_correct_samples

    for name in CLASS_NAMES[:2]:
        (tmp_path / name).mkdir()
        for i in range(3):
            Image.new("RGB", (40, 40), (i * 40, 80, 120)).save(tmp_path / name / f"{i}.jpg")

    class _AlwaysClass(torch.nn.Module):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(x.shape[0], NUM_CLASSES)
            logits[:, self.index] = 1.0
            return logits

    # Kedua "model" selalu menebak kelas 1 -> hanya sampel kelas 1 yang lolos.
    models = {"resnet50": _AlwaysClass(1), "mobilenetv3": _AlwaysClass(1)}
    paths, labels = select_correct_samples(
        models, tmp_path, class_names=CLASS_NAMES[:2], n_per_class=2, device="cpu"
    )
    assert labels == [1, 1]
    assert all(p.parent.name == CLASS_NAMES[1] for p in paths)

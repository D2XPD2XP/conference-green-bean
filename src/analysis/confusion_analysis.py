"""Analisis mendalam confusion matrix untuk klasifikasi 5 kelas.

Pada 5 kelas, confusion matrix jauh lebih informatif daripada metrik agregat:
bukan hanya "berapa yang salah", tetapi **tertukar dengan apa**. Pola seperti
``dry_pods`` sering diprediksi ``black`` menunjukkan kemiripan visual nyata
antar-kelas, dan itu temuan yang layak dibahas di laporan.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from configs.base_config import CLASS_NAMES
from src.analysis.style import apply_plot_style, save_figure

import matplotlib.pyplot as plt  # noqa: E402  (style.py sudah menetapkan backend Agg)
import seaborn as sns  # noqa: E402


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str] | None = None,
    normalize: str | None = "true",
    title: str = "Confusion Matrix",
    output_path: str | Path | None = None,
    ax=None,
    cbar: bool | None = None,
    tight: bool = True,
):
    """Buat heatmap confusion matrix, opsional ternormalisasi.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.
        class_names: Nama kelas untuk sumbu. Default ``CLASS_NAMES``.
        normalize: ``"true"`` (per baris = recall per kelas), ``"pred"``
            (per kolom = precision per kelas), atau ``None`` (jumlah mentah).
        title: Judul plot.
        output_path: Bila diberikan, simpan figur ke path ini (PNG).
        ax: Axes tujuan; bila ``None``, figur baru dibuat.
        cbar: Tampilkan colorbar. ``None`` berarti otomatis (tampil bila
            ternormalisasi). Dipakai figur perbandingan agar cukup satu
            colorbar untuk semua panel.
        tight: Panggil ``fig.tight_layout()``. Diset ``False`` saat menggambar
            ke figur yang memakai ``layout="constrained"``, karena kedua
            mekanisme penataan itu tidak boleh dipakai bersamaan.

    Returns:
        Objek ``matplotlib.figure.Figure``.

    Raises:
        ValueError: Bila ``normalize`` tidak dikenal.
    """
    if normalize not in {"true", "pred", None}:
        raise ValueError("normalize harus 'true', 'pred', atau None")

    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))

    fmt = "d"
    data = matrix.astype(float)
    if normalize == "true":
        data = data / data.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
    elif normalize == "pred":
        data = data / data.sum(axis=0, keepdims=True).clip(min=1)
        fmt = ".2f"

    apply_plot_style()
    if ax is None:
        size = max(5.5, 1.15 * len(names))
        fig, ax = plt.subplots(figsize=(size + 1.0, size))
    else:
        fig = ax.get_figure()

    sns.heatmap(
        data if normalize else matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        vmin=0,
        vmax=1 if normalize else None,
        xticklabels=names,
        yticklabels=names,
        cbar=(normalize is not None) if cbar is None else cbar,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(title)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    if tight:
        fig.tight_layout()

    save_figure(fig, output_path)
    return fig


def plot_confusion_matrices_comparison(
    results: dict[str, tuple[Sequence[int], Sequence[int]]],
    class_names: list[str] | None = None,
    normalize: str | None = "true",
    output_path: str | Path | None = None,
):
    """Tampilkan confusion matrix beberapa model berdampingan (grid 1xN).

    Args:
        results: Dict ``nama_model -> (y_true, y_pred)``.
        class_names: Nama kelas. Default ``CLASS_NAMES``.
        normalize: Skema normalisasi (lihat :func:`plot_confusion_matrix`).
        output_path: Bila diberikan, simpan figur ke path ini.

    Returns:
        Objek ``matplotlib.figure.Figure``.

    Raises:
        ValueError: Bila ``results`` kosong.
    """
    if not results:
        raise ValueError("results kosong.")

    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    apply_plot_style()
    width = max(4.8, 1.0 * len(names))
    # layout="constrained": colorbar bersama di bawah ini ikut diperhitungkan
    # saat menata panel, sehingga lebar ketiga matriks tetap sama.
    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(width * len(results) + 1.2, width + 0.6),
        squeeze=False,
        layout="constrained",
    )

    for index, (ax, (model_name, (y_true, y_pred))) in enumerate(
        zip(axes[0], results.items())
    ):
        # Colorbar per panel dimatikan; skalanya identik (0..1) untuk semua
        # model, jadi satu colorbar bersama lebih jelas sekaligus menyisakan
        # lebar yang lebih berguna untuk matriksnya.
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names=names,
            normalize=normalize,
            title=model_name,
            ax=ax,
            cbar=False,
            tight=False,
        )
        if index > 0:
            ax.set_ylabel("")

    if normalize is not None:
        mappable = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=1))
        fig.colorbar(mappable, ax=axes[0].tolist(), fraction=0.03, pad=0.02)

    suffix = {"true": " (ternormalisasi per baris = recall)", "pred": " (per kolom = precision)"}
    fig.suptitle("Confusion Matrix" + suffix.get(normalize or "", ""))
    save_figure(fig, output_path)
    return fig


def generate_classification_report_table(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """Susun tabel precision/recall/F1 per kelas plus rata-rata.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.
        class_names: Nama kelas. Default ``CLASS_NAMES``.

    Returns:
        DataFrame berindeks nama kelas (plus ``macro avg`` dan
        ``weighted avg``) dengan kolom ``precision``, ``recall``, ``f1``,
        ``support``.

    Example:
        >>> table = generate_classification_report_table([0, 1], [0, 1], ["good", "black"])
        >>> "macro avg" in table.index
        True
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(names))),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for key in [*names, "macro avg", "weighted avg"]:
        if key not in report:
            continue
        stats = report[key]
        rows.append(
            {
                "kelas": key,
                "precision": round(float(stats["precision"]), 4),
                "recall": round(float(stats["recall"]), 4),
                "f1": round(float(stats["f1-score"]), 4),
                "support": int(stats["support"]),
            }
        )

    return pd.DataFrame(rows).set_index("kelas")


def error_pattern_analysis(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str] | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Daftar pasangan kesalahan paling sering (kelas X diprediksi Y).

    Diagonal confusion matrix (prediksi benar) dikeluarkan, lalu sisanya
    diurutkan menurun. Kolom ``share_of_true`` menjawab pertanyaan yang lebih
    berguna daripada jumlah absolut: *berapa persen* biji kelas X yang bocor ke
    kelas Y.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.
        class_names: Nama kelas. Default ``CLASS_NAMES``.
        top_n: Jumlah pola teratas yang dikembalikan.

    Returns:
        DataFrame kolom ``true_class``, ``pred_class``, ``count``,
        ``share_of_true``, terurut menurun menurut ``count``.
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    per_true = matrix.sum(axis=1)

    rows = []
    for i, true_name in enumerate(names):
        for j, pred_name in enumerate(names):
            if i == j or matrix[i, j] == 0:
                continue
            rows.append(
                {
                    "true_class": true_name,
                    "pred_class": pred_name,
                    "count": int(matrix[i, j]),
                    "share_of_true": round(
                        float(matrix[i, j]) / float(max(per_true[i], 1)), 4
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)


def plot_error_pattern_heatmap(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str] | None = None,
    output_path: str | Path | None = None,
    title: str = "Pola kesalahan (diagonal dihapus)",
):
    """Heatmap confusion matrix dengan diagonal dinolkan.

    Menghapus diagonal membuat skala warna ditentukan oleh kesalahan, bukan
    oleh prediksi benar yang biasanya mendominasi — pola tertukar antar-kelas
    jadi langsung terlihat.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.
        class_names: Nama kelas. Default ``CLASS_NAMES``.
        output_path: Bila diberikan, simpan figur ke path ini.
        title: Judul plot.

    Returns:
        Objek ``matplotlib.figure.Figure``.
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names)))).astype(float)
    np.fill_diagonal(matrix, 0.0)

    apply_plot_style()
    size = max(5.5, 1.15 * len(names))
    fig, ax = plt.subplots(figsize=(size + 1.0, size))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        xticklabels=names,
        yticklabels=names,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(title)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig

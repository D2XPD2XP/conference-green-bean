"""Error analysis: kumpulkan dan tampilkan biji yang salah diklasifikasi.

Tabel metrik memberi tahu *berapa banyak* yang salah; galeri citra memberi
tahu **kenapa**. Melihat langsung biji yang tertukar sering mengungkap hal
yang tidak muncul di angka: crop yang terlalu gelap, dua biji dalam satu
kotak, atau label anotasi yang memang keliru.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from configs.base_config import CLASS_NAMES, DEVICE
from src.analysis.style import apply_plot_style, save_figure, tensor_to_image
from src.evaluation.predict import collect_predictions

import matplotlib.pyplot as plt  # noqa: E402  (style.py sudah menetapkan backend Agg)

logger = logging.getLogger(__name__)


def collect_misclassified_samples(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = DEVICE,
    n_samples: int = 20,
    class_names: list[str] | None = None,
    worst_first: bool = True,
) -> pd.DataFrame:
    """Kumpulkan prediksi yang salah, lengkap dengan path citranya.

    Args:
        model: Model terlatih.
        test_loader: DataLoader split test (**harus** ``shuffle=False``).
        device: Device komputasi.
        n_samples: Jumlah sampel maksimum yang dikembalikan.
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.
        worst_first: Bila ``True``, urutkan dari confidence tertinggi —
            yaitu kesalahan yang paling "percaya diri", biasanya paling
            informatif. Bila ``False``, urutkan dari confidence terendah.

    Returns:
        DataFrame berisi kolom seperti :func:`collect_predictions`, hanya baris
        salah, dipotong ``n_samples`` baris pertama.
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    frame = collect_predictions(model, test_loader, device=device, class_names=names)
    if frame.empty:
        return frame

    wrong = frame.loc[~frame["correct"]].sort_values(
        "confidence", ascending=not worst_first
    )
    return wrong.head(int(n_samples)).reset_index(drop=True)


def plot_error_gallery(
    misclassified: pd.DataFrame,
    n_cols: int = 5,
    output_path: str | Path | None = None,
    title: str = "Biji yang salah diklasifikasi",
):
    """Tampilkan galeri citra yang salah diklasifikasi dengan label true/pred.

    Args:
        misclassified: DataFrame hasil :func:`collect_misclassified_samples`;
            butuh kolom ``path``, ``true_label``, ``pred_label``,
            ``confidence``.
        n_cols: Jumlah kolom grid.
        output_path: Bila diberikan, simpan figur ke path ini.
        title: Judul figur.

    Returns:
        Objek ``matplotlib.figure.Figure``.

    Raises:
        KeyError: Bila kolom ``path`` tidak ada (tabel prediksi dibuat dari
            dataset tanpa informasi path).
    """
    if "path" not in misclassified.columns:
        raise KeyError(
            "Kolom 'path' tidak ada; galeri tidak bisa dibuat. Pastikan "
            "DataLoader memakai GreenBeanDataset (menyediakan atribut 'paths')."
        )

    apply_plot_style()
    if misclassified.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Tidak ada prediksi salah.", ha="center", va="center")
        ax.axis("off")
        save_figure(fig, output_path)
        return fig

    n_items = len(misclassified)
    n_cols = max(1, min(int(n_cols), n_items))
    n_rows = int(np.ceil(n_items / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.7 * n_rows), squeeze=False)

    for position, (_, row) in enumerate(misclassified.iterrows()):
        ax = axes[position // n_cols][position % n_cols]
        path = Path(str(row["path"]))
        if path.exists():
            ax.imshow(Image.open(path).convert("RGB"))
        else:
            ax.text(0.5, 0.5, "citra\nhilang", ha="center", va="center", fontsize=8)
            logger.warning("Citra tidak ditemukan: %s", path)
        ax.set_title(
            f"true: {row['true_label']}\npred: {row['pred_label']} ({row['confidence']:.2f})",
            fontsize=8,
            color="#922b21",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    # Sembunyikan sel sisa pada baris terakhir.
    for position in range(n_items, n_rows * n_cols):
        axes[position // n_cols][position % n_cols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_class_gallery(
    predictions: pd.DataFrame,
    class_names: list[str] | None = None,
    n_per_class: int = 5,
    only_correct: bool = True,
    output_path: str | Path | None = None,
    title: str = "Contoh prediksi per kelas",
):
    """Galeri contoh citra per kelas (default: yang diprediksi benar).

    Berguna sebagai pembanding visual galeri error: seperti apa biji yang
    "mudah" bagi model dibanding yang membuatnya tertukar.

    Args:
        predictions: Tabel prediksi lengkap.
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.
        n_per_class: Jumlah contoh per kelas.
        only_correct: Bila ``True``, hanya ambil prediksi benar.
        output_path: Bila diberikan, simpan figur ke path ini.
        title: Judul figur.

    Returns:
        Objek ``matplotlib.figure.Figure``.
    """
    if "path" not in predictions.columns:
        raise KeyError("Kolom 'path' tidak ada; galeri tidak bisa dibuat.")

    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    source = predictions.loc[predictions["correct"]] if only_correct else predictions

    apply_plot_style()
    fig, axes = plt.subplots(
        len(names), n_per_class, figsize=(2.1 * n_per_class, 2.3 * len(names)), squeeze=False
    )

    for row, name in enumerate(names):
        subset = source.loc[source["true_label"] == name].head(n_per_class)
        for col in range(n_per_class):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if col < len(subset):
                item = subset.iloc[col]
                path = Path(str(item["path"]))
                if path.exists():
                    ax.imshow(Image.open(path).convert("RGB"))
                ax.set_xlabel(f"{item['confidence']:.2f}", fontsize=7)
            else:
                ax.axis("off")
        axes[row][0].set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center", labelpad=42)

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def error_summary_by_class(
    predictions: pd.DataFrame,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """Ringkas jumlah dan tingkat kesalahan per kelas sebenarnya.

    Args:
        predictions: Tabel prediksi.
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.

    Returns:
        DataFrame berindeks kelas, kolom ``n_samples``, ``n_wrong``,
        ``error_rate``, ``mean_confidence_wrong``, dan ``most_confused_with``.
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)

    rows = []
    for name in names:
        subset = predictions.loc[predictions["true_label"] == name]
        wrong = subset.loc[~subset["correct"]] if len(subset) else subset
        confused = (
            wrong["pred_label"].value_counts().idxmax() if len(wrong) else ""
        )
        rows.append(
            {
                "kelas": name,
                "n_samples": int(len(subset)),
                "n_wrong": int(len(wrong)),
                "error_rate": round(len(wrong) / len(subset), 4) if len(subset) else 0.0,
                "mean_confidence_wrong": round(float(wrong["confidence"].mean()), 4)
                if len(wrong)
                else 0.0,
                "most_confused_with": confused,
            }
        )

    return pd.DataFrame(rows).set_index("kelas")


def plot_misclassified_tensors(
    images: torch.Tensor,
    true_labels: list[int],
    pred_labels: list[int],
    confidences: list[float],
    class_names: list[str] | None = None,
    n_cols: int = 5,
    output_path: str | Path | None = None,
):
    """Varian galeri error yang menerima tensor, bukan path citra.

    Dipakai bila DataLoader tidak menyediakan path (mis. dataset custom di
    notebook eksperimen).

    Args:
        images: Tensor ``(N, 3, H, W)`` yang sudah dinormalisasi ImageNet.
        true_labels: Label sebenarnya tiap citra.
        pred_labels: Label prediksi tiap citra.
        confidences: Confidence tiap prediksi.
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.
        n_cols: Jumlah kolom grid.
        output_path: Bila diberikan, simpan figur ke path ini.

    Returns:
        Objek ``matplotlib.figure.Figure``.
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    n_items = int(images.shape[0])
    if n_items == 0:
        raise ValueError("Tidak ada citra untuk ditampilkan.")

    apply_plot_style()
    n_cols = max(1, min(int(n_cols), n_items))
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.7 * n_rows), squeeze=False)

    for position in range(n_rows * n_cols):
        ax = axes[position // n_cols][position % n_cols]
        if position >= n_items:
            ax.axis("off")
            continue
        ax.imshow(tensor_to_image(images[position]))
        true_name = names[true_labels[position]] if true_labels[position] < len(names) else "?"
        pred_name = names[pred_labels[position]] if pred_labels[position] < len(names) else "?"
        ax.set_title(
            f"true: {true_name}\npred: {pred_name} ({confidences[position]:.2f})", fontsize=8
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig

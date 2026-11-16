"""Analisis distribusi confidence (probabilitas softmax) prediksi model.

Pertanyaan yang dijawab: **apakah model tahu kapan dirinya ragu?** Model yang
berguna di produksi bukan hanya akurat, tetapi juga memberi confidence rendah
pada kasus yang salah — sehingga ambang batas dapat dipakai untuk melimpahkan
kasus sulit ke pemeriksa manusia.

Semua fungsi di sini bekerja di atas DataFrame hasil
:func:`src.evaluation.predict.collect_predictions`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from configs.base_config import CLASS_NAMES
from src.analysis.style import apply_plot_style, class_palette, save_figure

import matplotlib.pyplot as plt  # noqa: E402  (style.py sudah menetapkan backend Agg)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Pastikan DataFrame memuat kolom yang dibutuhkan."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(
            f"Kolom {missing} tidak ada di tabel prediksi. "
            "Pakai src.evaluation.predict.collect_predictions untuk membuatnya."
        )


def plot_confidence_distribution(
    predictions: pd.DataFrame,
    output_path: str | Path | None = None,
    bins: int = 20,
    title: str = "Distribusi confidence: prediksi benar vs salah",
):
    """Histogram confidence untuk prediksi benar dan salah.

    Args:
        predictions: Tabel prediksi (butuh kolom ``confidence`` dan ``correct``).
        output_path: Bila diberikan, simpan figur ke path ini.
        bins: Jumlah bin histogram.
        title: Judul plot.

    Returns:
        Objek ``matplotlib.figure.Figure``.

    Raises:
        KeyError: Bila kolom yang dibutuhkan tidak ada.
    """
    _require_columns(predictions, ["confidence", "correct"])

    correct = predictions.loc[predictions["correct"], "confidence"]
    wrong = predictions.loc[~predictions["correct"], "confidence"]

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    edges = np.linspace(0.0, 1.0, bins + 1)

    ax.hist(correct, bins=edges, alpha=0.75, label=f"benar (n={len(correct)})", color="#2ecc71")
    ax.hist(wrong, bins=edges, alpha=0.75, label=f"salah (n={len(wrong)})", color="#e74c3c")

    if len(correct):
        ax.axvline(correct.mean(), color="#1e8449", linestyle="--", linewidth=1.2)
    if len(wrong):
        ax.axvline(wrong.mean(), color="#922b21", linestyle="--", linewidth=1.2)

    ax.set_xlabel("Confidence (probabilitas kelas prediksi)")
    ax.set_ylabel("Jumlah biji")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_confidence_per_class(
    predictions: pd.DataFrame,
    class_names: list[str] | None = None,
    output_path: str | Path | None = None,
    title: str = "Confidence per kelas sebenarnya",
):
    """Box plot confidence dikelompokkan menurut kelas sebenarnya.

    Args:
        predictions: Tabel prediksi (butuh ``confidence`` dan ``true_label``).
        class_names: Urutan kelas pada sumbu x. Default ``CLASS_NAMES``.
        output_path: Bila diberikan, simpan figur ke path ini.
        title: Judul plot.

    Returns:
        Objek ``matplotlib.figure.Figure``.
    """
    _require_columns(predictions, ["confidence", "true_label"])
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)

    data = [predictions.loc[predictions["true_label"] == name, "confidence"].to_numpy() for name in names]
    colors = class_palette(names)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2.0, 4.4))
    box = ax.boxplot(
        [d if len(d) else np.array([np.nan]) for d in data],
        tick_labels=names,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Confidence")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def identify_uncertain_predictions(
    predictions: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Ambil prediksi dengan confidence di bawah ambang.

    Biji di tabel ini adalah kandidat terbaik untuk pemeriksaan manual: model
    sendiri tidak yakin, jadi biayanya paling rendah dan manfaatnya paling
    tinggi.

    Args:
        predictions: Tabel prediksi.
        threshold: Ambang confidence (0..1).

    Returns:
        DataFrame baris ber-confidence ``< threshold``, terurut naik.
    """
    _require_columns(predictions, ["confidence"])
    subset = predictions.loc[predictions["confidence"] < float(threshold)]
    return subset.sort_values("confidence").reset_index(drop=True)


def confidence_summary(predictions: pd.DataFrame) -> dict[str, float]:
    """Ringkas statistik confidence untuk dilaporkan di tabel.

    Args:
        predictions: Tabel prediksi (butuh ``confidence`` dan ``correct``).

    Returns:
        Dict berisi ``mean_confidence_correct``, ``mean_confidence_wrong``,
        ``confidence_gap`` (selisih keduanya; makin besar makin baik),
        ``mean_confidence``, dan ``share_below_0.5``.
    """
    _require_columns(predictions, ["confidence", "correct"])

    correct = predictions.loc[predictions["correct"], "confidence"]
    wrong = predictions.loc[~predictions["correct"], "confidence"]
    mean_correct = float(correct.mean()) if len(correct) else 0.0
    mean_wrong = float(wrong.mean()) if len(wrong) else 0.0
    total = len(predictions)

    return {
        "mean_confidence": round(float(predictions["confidence"].mean()), 4) if total else 0.0,
        "mean_confidence_correct": round(mean_correct, 4),
        "mean_confidence_wrong": round(mean_wrong, 4),
        "confidence_gap": round(mean_correct - mean_wrong, 4),
        "share_below_0.5": round(
            float((predictions["confidence"] < 0.5).sum()) / total, 4
        )
        if total
        else 0.0,
    }


def accuracy_vs_threshold(
    predictions: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Hitung akurasi dan cakupan bila prediksi ber-confidence rendah ditolak.

    Mensimulasikan skenario *human-in-the-loop*: biji dengan confidence di
    bawah ambang dilimpahkan ke pemeriksa manual, sisanya dipercayakan ke
    model. Trade-off-nya terbaca langsung — makin tinggi ambang, makin akurat
    yang otomatis, tetapi makin banyak yang perlu diperiksa manusia.

    Args:
        predictions: Tabel prediksi.
        thresholds: Daftar ambang. Default ``0.0, 0.1, ..., 0.9``.

    Returns:
        DataFrame kolom ``threshold``, ``coverage`` (proporsi otomatis),
        ``accuracy_on_covered``, ``n_referred`` (jumlah dilimpahkan manual).
    """
    _require_columns(predictions, ["confidence", "correct"])
    grid = thresholds if thresholds is not None else [round(0.1 * i, 2) for i in range(10)]

    total = len(predictions)
    rows = []
    for threshold in grid:
        covered = predictions.loc[predictions["confidence"] >= threshold]
        rows.append(
            {
                "threshold": threshold,
                "coverage": round(len(covered) / total, 4) if total else 0.0,
                "accuracy_on_covered": round(float(covered["correct"].mean()), 4)
                if len(covered)
                else float("nan"),
                "n_referred": int(total - len(covered)),
            }
        )
    return pd.DataFrame(rows)

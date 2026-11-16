"""Plot dan penyimpanan confusion matrix."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# Backend tidak diset di sini agar inline Jupyter tidak tertimpa
# (lihat catatan di src/analysis/style.py).
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from configs.base_config import CLASS_NAMES


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    output_path: str | Path | None = None,
    normalize: bool = False,
):
    """Buat heatmap confusion matrix dan (opsional) simpan ke file.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.
        class_names: Nama kelas untuk sumbu. Default ``CLASS_NAMES``.
        title: Judul plot.
        output_path: Bila diberikan, simpan figur ke path ini (PNG).
        normalize: Bila ``True``, tampilkan proporsi per baris (recall).

    Returns:
        Objek ``matplotlib.figure.Figure`` yang dihasilkan.
    """
    labels = class_names if class_names is not None else list(CLASS_NAMES)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    fmt = "d"
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(title)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig

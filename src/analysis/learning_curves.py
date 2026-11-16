"""Kurva pembelajaran: konvergensi dan gejala overfitting antar-arsitektur.

Membaca berkas ``<model>_history.json`` yang ditulis
:func:`src.training.train_cnn.train_cnn`. Yang dicari di sini bukan sekadar
"nilai akhirnya berapa", melainkan bentuk kurvanya: kapan konvergen, apakah
``val_loss`` mulai naik sementara ``train_loss`` masih turun (overfitting), dan
seberapa jauh jarak train-val (generalization gap).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from configs.base_config import MODELS_DIR
from src.analysis.style import apply_plot_style, save_figure

import matplotlib.pyplot as plt  # noqa: E402  (style.py sudah menetapkan backend Agg)

logger = logging.getLogger(__name__)

#: Warna per model agar konsisten di semua kurva.
_MODEL_COLORS: dict[str, str] = {
    "resnet50": "#3498db",
    "mobilenetv3": "#e67e22",
    "inceptionv3": "#16a085",
}

#: Warna cadangan bila nama model tidak dikenal.
_FALLBACK_COLOR: str = "#7f8c8d"


def load_history(path: str | Path) -> dict:
    """Muat satu berkas history pelatihan.

    Args:
        path: Path ``<model>_history.json``.

    Returns:
        Dict history.

    Raises:
        FileNotFoundError: Bila file tidak ada.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"History tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_histories(
    model_names: list[str],
    models_dir: str | Path = MODELS_DIR,
) -> dict[str, dict]:
    """Muat history seluruh model yang tersedia.

    Model yang belum dilatih dilewati dengan peringatan, bukan error — supaya
    analisis tetap bisa dijalankan di tengah pipeline.

    Args:
        model_names: Daftar nama model.
        models_dir: Root folder ``results/models``.

    Returns:
        Dict ``nama_model -> history`` hanya untuk model yang history-nya ada.
    """
    root = Path(models_dir)
    histories: dict[str, dict] = {}
    for name in model_names:
        path = root / name / f"{name}_history.json"
        if not path.exists():
            logger.warning("History %s belum ada (%s); dilewati.", name, path)
            continue
        histories[name] = load_history(path)
    return histories


def plot_learning_curves(
    histories: dict[str, dict],
    output_path: str | Path | None = None,
    title: str = "Kurva pembelajaran",
):
    """Plot loss dan akurasi per epoch untuk beberapa model dalam satu figur.

    Garis solid = validasi, garis putus-putus = train. Menumpuk keduanya di
    satu axes membuat *generalization gap* terbaca langsung sebagai jarak
    vertikal antar-pasangan garis.

    Args:
        histories: Dict ``nama_model -> history``.
        output_path: Bila diberikan, simpan figur ke path ini.
        title: Judul figur.

    Returns:
        Objek ``matplotlib.figure.Figure``.

    Raises:
        ValueError: Bila ``histories`` kosong.
    """
    if not histories:
        raise ValueError("histories kosong; tidak ada yang bisa diplot.")

    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    for name, history in histories.items():
        color = _MODEL_COLORS.get(name, _FALLBACK_COLOR)
        epochs = range(1, len(history.get("train_loss", [])) + 1)

        axes[0].plot(epochs, history.get("train_loss", []), "--", color=color, alpha=0.65,
                     label=f"{name} train")
        axes[0].plot(epochs, history.get("val_loss", []), "-", color=color, linewidth=1.8,
                     label=f"{name} valid")

        if history.get("train_acc"):
            axes[1].plot(epochs, history["train_acc"], "--", color=color, alpha=0.65,
                         label=f"{name} train")
        axes[1].plot(epochs, history.get("val_acc", []), "-", color=color, linewidth=1.8,
                     label=f"{name} valid")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss per epoch")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy per epoch")
    axes[1].legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_learning_curves_grid(
    histories: dict[str, dict],
    output_path: str | Path | None = None,
):
    """Satu baris per model: loss dan akurasi dipisah agar skalanya jelas.

    Berguna bila kurva antar-model berbeda skala sehingga saling menutupi
    kalau ditumpuk.

    Args:
        histories: Dict ``nama_model -> history``.
        output_path: Bila diberikan, simpan figur ke path ini.

    Returns:
        Objek ``matplotlib.figure.Figure``.
    """
    if not histories:
        raise ValueError("histories kosong; tidak ada yang bisa diplot.")

    apply_plot_style()
    fig, axes = plt.subplots(
        len(histories), 2, figsize=(11.0, 3.4 * len(histories)), squeeze=False
    )

    for row, (name, history) in enumerate(histories.items()):
        color = _MODEL_COLORS.get(name, _FALLBACK_COLOR)
        epochs = range(1, len(history.get("train_loss", [])) + 1)

        axes[row][0].plot(epochs, history.get("train_loss", []), "--", color=color, label="train")
        axes[row][0].plot(epochs, history.get("val_loss", []), "-", color=color, label="valid")
        axes[row][0].set_ylabel("Loss")
        axes[row][0].set_title(f"{name} - loss")
        axes[row][0].legend(fontsize=8)

        if history.get("train_acc"):
            axes[row][1].plot(epochs, history["train_acc"], "--", color=color, label="train")
        axes[row][1].plot(epochs, history.get("val_acc", []), "-", color=color, label="valid")
        axes[row][1].set_ylabel("Accuracy")
        axes[row][1].set_title(f"{name} - accuracy")
        axes[row][1].legend(fontsize=8)

        for ax in axes[row]:
            ax.set_xlabel("Epoch")

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def convergence_summary(histories: dict[str, dict]) -> pd.DataFrame:
    """Ringkas perilaku konvergensi tiap model menjadi satu tabel.

    Kolom ``overfitting_signal`` membandingkan ``val_loss`` terakhir dengan
    ``val_loss`` terendah: nilai positif yang besar berarti loss validasi
    sudah berbalik naik — model mulai menghafal data latih.

    Args:
        histories: Dict ``nama_model -> history``.

    Returns:
        DataFrame berindeks nama model dengan kolom ``epochs_ran``,
        ``best_epoch``, ``best_val_acc``, ``final_train_loss``,
        ``final_val_loss``, ``min_val_loss``, ``overfitting_signal``,
        ``generalization_gap``, dan ``imbalance_strategy``.
    """
    rows = []
    for name, history in histories.items():
        train_loss = history.get("train_loss", []) or [float("nan")]
        val_loss = history.get("val_loss", []) or [float("nan")]
        train_acc = history.get("train_acc", []) or [float("nan")]
        val_acc = history.get("val_acc", []) or [float("nan")]

        min_val_loss = min(val_loss)
        rows.append(
            {
                "model": name,
                "epochs_ran": history.get("epochs_ran", len(history.get("train_loss", []))),
                "best_epoch": history.get("best_epoch", 0),
                "best_val_acc": round(float(history.get("best_val_acc", max(val_acc))), 4),
                "final_train_loss": round(float(train_loss[-1]), 4),
                "final_val_loss": round(float(val_loss[-1]), 4),
                "min_val_loss": round(float(min_val_loss), 4),
                "overfitting_signal": round(float(val_loss[-1] - min_val_loss), 4),
                "generalization_gap": round(float(train_acc[-1] - val_acc[-1]), 4),
                "imbalance_strategy": history.get("imbalance_strategy", ""),
            }
        )

    return pd.DataFrame(rows).set_index("model")

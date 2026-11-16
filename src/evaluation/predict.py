"""Pengumpulan prediksi model menjadi tabel siap analisis.

Seluruh modul analisis (confidence, error analysis, confusion) bekerja di atas
satu struktur yang sama: ``pandas.DataFrame`` satu baris per biji, memuat path
citra, label sebenarnya, label prediksi, confidence, dan probabilitas tiap
kelas. Menyimpannya ke CSV membuat analisis bisa diulang tanpa memuat model
lagi.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.base_config import CLASS_NAMES

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """Jalankan inferensi pada seluruh loader dan kumpulkan hasilnya.

    ``loader`` **harus** dibuat dengan ``shuffle=False`` agar urutan baris
    cocok dengan ``loader.dataset.samples``; kalau tidak, kolom ``path`` akan
    menunjuk citra yang salah. Fungsi ini memeriksanya dan memberi peringatan.

    Args:
        model: Model terlatih.
        loader: DataLoader split yang dievaluasi (``shuffle=False``).
        device: Device komputasi.
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.

    Returns:
        DataFrame dengan kolom:

        - ``path``        : path citra (bila dataset menyediakannya)
        - ``y_true``      : indeks kelas sebenarnya
        - ``y_pred``      : indeks kelas prediksi
        - ``true_label``  : nama kelas sebenarnya
        - ``pred_label``  : nama kelas prediksi
        - ``correct``     : bool
        - ``confidence``  : probabilitas kelas prediksi (softmax maksimum)
        - ``prob_<kelas>``: probabilitas tiap kelas
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    model = model.to(device).eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[np.ndarray] = []

    for images, labels in loader:
        logits = model(images.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
        probabilities.append(probs)
        y_pred.extend(probs.argmax(axis=1).tolist())
        y_true.extend(int(v) for v in labels.tolist())

    if not probabilities:
        logger.warning("Loader kosong; tidak ada prediksi yang dikumpulkan.")
        columns = ["path", "y_true", "y_pred", "true_label", "pred_label", "correct", "confidence"]
        return pd.DataFrame(columns=columns + [f"prob_{n}" for n in names])

    probs_all = np.concatenate(probabilities, axis=0)

    frame = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "true_label": [names[i] if 0 <= i < len(names) else str(i) for i in y_true],
            "pred_label": [names[i] if 0 <= i < len(names) else str(i) for i in y_pred],
            "confidence": probs_all.max(axis=1),
        }
    )
    frame["correct"] = frame["y_true"] == frame["y_pred"]
    for idx, name in enumerate(names):
        frame[f"prob_{name}"] = probs_all[:, idx]

    paths = getattr(loader.dataset, "paths", None)
    if paths is not None and len(paths) == len(frame):
        frame.insert(0, "path", [str(p) for p in paths])
        # Label dataset harus cocok dengan label yang dikembalikan loader;
        # kalau tidak, loader memakai shuffle=True dan path jadi tidak sinkron.
        dataset_labels = getattr(loader.dataset, "labels", None)
        if dataset_labels is not None and list(dataset_labels) != y_true:
            logger.warning(
                "Urutan label loader tidak sama dengan urutan dataset — "
                "loader kemungkinan memakai shuffle=True, sehingga kolom 'path' "
                "tidak sinkron dengan prediksi."
            )
    elif paths is not None:
        logger.warning(
            "Jumlah path dataset (%d) != jumlah prediksi (%d); kolom 'path' dilewati.",
            len(paths),
            len(frame),
        )

    return frame


def save_predictions(frame: pd.DataFrame, path: str | Path) -> Path:
    """Simpan tabel prediksi ke CSV (UTF-8).

    Args:
        frame: DataFrame hasil :func:`collect_predictions`.
        path: Path tujuan ``.csv``; direktori induk dibuat bila perlu.

    Returns:
        Path file yang tersimpan.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Muat tabel prediksi dari CSV.

    Args:
        path: Path file ``.csv`` hasil :func:`save_predictions`.

    Returns:
        DataFrame prediksi.

    Raises:
        FileNotFoundError: Bila file tidak ada.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File prediksi tidak ditemukan: {path}")
    return pd.read_csv(path)

"""Ambang keputusan untuk kelas ``good`` (pasca-pelatihan, tanpa melatih ulang).

Untuk quality control, biji cacat yang lolos sebagai ``good`` jauh lebih mahal
daripada biji baik yang ditolak. Saat ``insect_damage`` salah ditebak sebagai
``good``, model biasanya tidak terlalu yakin. Aturan di sini: sebuah biji baru
dinyatakan ``good`` bila ``P(good) >= threshold``; bila tidak, ia dialihkan ke
kelas cacat dengan probabilitas tertinggi.

``threshold = 0`` identik dengan argmax biasa. Ambang **dipilih pada split
valid** lalu diterapkan apa adanya ke test — memilihnya pada test membuat angka
yang dilaporkan bias optimis.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from configs.base_config import CLASS_NAMES
from src.evaluation.metrics import (
    compute_defect_metrics,
    compute_metrics,
    compute_per_class_metrics,
)

#: Grid ambang yang dicoba (0 = argmax biasa). Sejak revisi 2026-11: 0,30-0,90
#: langkah 0,05 (sebelumnya 0,40-0,95).
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0, *np.round(np.arange(0.30, 0.901, 0.05), 2))


def probability_matrix(frame: pd.DataFrame, class_names: Sequence[str] = CLASS_NAMES) -> np.ndarray:
    """Ambil matriks probabilitas ``(N, C)`` dari tabel prediksi."""
    return frame[[f"prob_{name}" for name in class_names]].to_numpy()


def apply_good_threshold(
    probs: np.ndarray,
    threshold: float,
    good_index: int = 0,
) -> np.ndarray:
    """Prediksi dengan ambang minimum untuk kelas ``good``.

    Args:
        probs: Probabilitas softmax ``(N, C)``.
        threshold: ``P(good)`` minimum agar biji dinyatakan ``good``.
        good_index: Indeks kelas ``good``.

    Returns:
        Array prediksi ``(N,)``.
    """
    preds = probs.argmax(axis=1)
    doubtful = (preds == good_index) & (probs[:, good_index] < threshold)
    if doubtful.any():
        defect_probs = probs[doubtful].copy()
        defect_probs[:, good_index] = -np.inf
        preds[doubtful] = defect_probs.argmax(axis=1)
    return preds


def sweep_thresholds(
    y_true: Sequence[int],
    probs: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    class_names: Sequence[str] = CLASS_NAMES,
) -> pd.DataFrame:
    """Hitung metrik untuk tiap ambang.

    Returns:
        DataFrame indeks ``threshold`` dengan kolom ``acc``, ``f1``,
        ``miss_rate``, ``good_recall``, ``f1_insect_damage``, dan
        ``n_rejected_good`` (biji ``good`` asli yang ikut tertolak).
    """
    names = list(class_names)
    good = names.index("good")
    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        y_pred = apply_good_threshold(probs, t, good_index=good)
        macro = compute_metrics(y_true.tolist(), y_pred.tolist())
        defect = compute_defect_metrics(y_true.tolist(), y_pred.tolist(), names)
        per_class = compute_per_class_metrics(y_true.tolist(), y_pred.tolist(), names)
        rows.append(
            {
                "threshold": float(t),
                "acc": macro["accuracy"],
                "f1": macro["f1"],
                "miss_rate": defect["defect_miss_rate"],
                "good_recall": per_class["good"]["recall"],
                "f1_insect_damage": per_class["insect_damage"]["f1"],
                "n_rejected_good": int(((y_true == good) & (y_pred != good)).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("threshold")


def select_threshold(
    sweep: pd.DataFrame,
    max_f1_drop: float = 0.01,
    reference: str = "argmax",
) -> float:
    """Pilih ambang dengan miss rate terendah yang F1-nya tidak turun berarti.

    Kandidat = ambang yang F1 makronya paling banyak ``max_f1_drop`` di bawah
    F1 acuan. Dari kandidat itu dipilih miss rate terendah (seri: F1
    tertinggi, lalu ambang terkecil). Dijalankan pada sweep **valid**.

    Args:
        sweep: Keluaran :func:`sweep_thresholds` pada split valid (wajib
            memuat baris ambang ``0.0`` bila ``reference="argmax"``).
        max_f1_drop: Penurunan F1 makro yang masih ditoleransi.
        reference: ``"argmax"`` = F1 pada ambang 0 / argmax biasa (kriteria
            revisi 2026-11); ``"best"`` = F1 tertinggi di sweep (kriteria
            lama, dipertahankan untuk reproduksi).

    Returns:
        Ambang terpilih.

    Raises:
        ValueError: Bila ``reference`` tidak dikenal.
    """
    if reference == "argmax":
        ref_f1 = float(sweep.loc[0.0, "f1"])
    elif reference == "best":
        ref_f1 = float(sweep["f1"].max())
    else:
        raise ValueError(f"reference '{reference}' tidak dikenal (argmax/best)")
    candidates = sweep[sweep["f1"] >= ref_f1 - max_f1_drop - 1e-12]
    best = candidates.reset_index().sort_values(
        ["miss_rate", "f1", "threshold"], ascending=[True, False, True]
    )
    return float(best["threshold"].iloc[0])

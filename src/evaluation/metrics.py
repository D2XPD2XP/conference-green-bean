"""Metrik klasifikasi tingkat biji individual.

Seluruh metrik dihitung dengan ``sklearn.metrics``. Presisi/recall/F1 memakai
rata-rata *macro* agar tiap kelas berkontribusi setara meski distribusinya
timpang.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from configs.base_config import CLASS_NAMES, GOOD_CLASS

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = "macro",
) -> dict[str, float]:
    """Hitung akurasi, presisi, recall, dan F1.

    Args:
        y_true: Label sebenarnya (per biji).
        y_pred: Label prediksi (per biji).
        average: Skema rata-rata untuk presisi/recall/F1 (default ``"macro"``).

    Returns:
        Dict berisi kunci ``accuracy``, ``precision``, ``recall``, ``f1``
        (semua ``float``).

    Example:
        >>> compute_metrics([0, 1, 1, 0], [0, 1, 0, 0])["accuracy"]
        0.75
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def compute_per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    """Hitung presisi, recall, F1, dan dukungan (support) untuk tiap kelas.

    Berguna saat distribusi kelas timpang: metrik macro dapat menyembunyikan
    kelas minoritas yang performanya buruk.

    Args:
        y_true: Label sebenarnya (per biji).
        y_pred: Label prediksi (per biji).
        class_names: Nama kelas terurut sesuai indeks label.

    Returns:
        Dict ``nama_kelas -> {'precision', 'recall', 'f1', 'support'}``.

    Example:
        >>> hasil = compute_per_class_metrics([0, 1], [0, 1], ["good", "black"])
        >>> hasil["good"]["recall"]
        1.0
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in class_names
    }


def compute_defect_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str] | None = None,
    good_class: str = GOOD_CLASS,
) -> dict[str, float]:
    """Hitung metrik agregat "layak vs cacat" dari prediksi 5 kelas.

    Untuk keperluan quality control, pertanyaan praktisnya bukan "jenis cacat
    apa", melainkan "biji ini lolos atau tidak". Metrik ini meruntuhkan 5 kelas
    menjadi 2: ``good`` versus gabungan seluruh kelas cacat. Salah menebak
    *jenis* cacat (mis. ``black`` diprediksi ``insect_damage``) tidak dihitung
    sebagai kesalahan di sini, karena keputusan akhirnya tetap benar: ditolak.

    Kelas positif adalah **cacat**, sehingga ``recall`` di sini berarti proporsi
    biji cacat yang berhasil ditangkap — angka yang paling mahal bila rendah.

    Args:
        y_true: Label sebenarnya (indeks kelas 5 kelas).
        y_pred: Label prediksi (indeks kelas 5 kelas).
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.
        good_class: Nama kelas non-cacat.

    Returns:
        Dict berisi ``defect_accuracy``, ``defect_precision``,
        ``defect_recall``, ``defect_f1``, dan ``defect_miss_rate``
        (= 1 - recall, proporsi biji cacat yang lolos sebagai layak).

    Example:
        >>> compute_defect_metrics([0, 1, 2], [0, 2, 1], ["good", "black", "chipped"])["defect_recall"]
        1.0
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    good_idx = names.index(good_class) if good_class in names else -1

    # 1 = cacat, 0 = layak.
    bin_true = [0 if int(v) == good_idx else 1 for v in y_true]
    bin_pred = [0 if int(v) == good_idx else 1 for v in y_pred]

    recall = float(recall_score(bin_true, bin_pred, pos_label=1, zero_division=0))
    return {
        "defect_accuracy": float(accuracy_score(bin_true, bin_pred)),
        "defect_precision": float(
            precision_score(bin_true, bin_pred, pos_label=1, zero_division=0)
        ),
        "defect_recall": recall,
        "defect_f1": float(f1_score(bin_true, bin_pred, pos_label=1, zero_division=0)),
        "defect_miss_rate": round(1.0 - recall, 6),
    }

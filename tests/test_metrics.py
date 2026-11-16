"""Unit test metrik: bandingkan dengan perhitungan sklearn langsung."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.evaluation.metrics import (
    compute_defect_metrics,
    compute_metrics,
    compute_per_class_metrics,
)

#: Nama kelas 5 kelas (urutan = indeks label).
CLASS_NAMES: list[str] = ["good", "black", "chipped", "dry_pods", "insect_damage"]


def test_perfect_prediction() -> None:
    """Prediksi sempurna -> semua metrik = 1.0."""
    y_true = [0, 1, 2, 3, 4, 0]
    metrics = compute_metrics(y_true, y_true)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_matches_sklearn() -> None:
    """Nilai metrik cocok dengan sklearn (macro average)."""
    y_true = [0, 1, 2, 3, 4, 0, 1, 2]
    y_pred = [0, 1, 1, 3, 0, 0, 1, 4]
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == accuracy_score(y_true, y_pred)
    assert metrics["precision"] == precision_score(y_true, y_pred, average="macro", zero_division=0)
    assert metrics["recall"] == recall_score(y_true, y_pred, average="macro", zero_division=0)
    assert metrics["f1"] == f1_score(y_true, y_pred, average="macro", zero_division=0)


def test_returns_expected_keys() -> None:
    """Dict hasil memuat seluruh kunci metrik yang diharapkan."""
    metrics = compute_metrics([0, 1], [0, 1])
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}


def test_per_class_metrics_keys() -> None:
    """Metrik per kelas memuat tiap nama kelas dan kunci yang diharapkan."""
    result = compute_per_class_metrics([0, 1, 2, 3, 4], [0, 1, 1, 3, 4], CLASS_NAMES)

    assert set(result) == set(CLASS_NAMES)
    for stats in result.values():
        assert set(stats) == {"precision", "recall", "f1", "support"}


def test_per_class_metrics_values() -> None:
    """Nilai per kelas cocok dengan sklearn tanpa rata-rata."""
    y_true = [0, 1, 1, 0, 2, 3, 4]
    y_pred = [0, 1, 0, 0, 2, 4, 4]
    result = compute_per_class_metrics(y_true, y_pred, CLASS_NAMES)

    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    for index, name in enumerate(CLASS_NAMES):
        assert result[name]["recall"] == per_class_recall[index]

    # Support = jumlah sampel sebenarnya per kelas.
    assert result["good"]["support"] == 2
    assert result["black"]["support"] == 2
    assert result["dry_pods"]["support"] == 1


def test_per_class_metrics_handles_empty_class() -> None:
    """Kelas tanpa sampel tidak memunculkan exception (zero_division=0)."""
    result = compute_per_class_metrics([0, 0, 0], [0, 0, 0], CLASS_NAMES)

    assert result["good"]["recall"] == 1.0
    assert result["insect_damage"]["support"] == 0
    assert result["insect_damage"]["f1"] == 0.0


# --------------------------------------------------------------------------- #
# Metrik agregat layak vs cacat
# --------------------------------------------------------------------------- #
def test_defect_metrics_ignores_defect_type_confusion() -> None:
    """Salah menebak jenis cacat tidak dihitung salah pada metrik layak/cacat."""
    # black (1) diprediksi insect_damage (4), dan sebaliknya: keputusan akhir
    # (ditolak) tetap benar, jadi metrik defect harus sempurna.
    y_true = [0, 1, 4, 2]
    y_pred = [0, 4, 1, 3]
    result = compute_defect_metrics(y_true, y_pred, CLASS_NAMES)

    assert result["defect_accuracy"] == 1.0
    assert result["defect_recall"] == 1.0
    assert result["defect_miss_rate"] == 0.0


def test_defect_metrics_counts_missed_defects() -> None:
    """Biji cacat yang diprediksi good menurunkan recall cacat."""
    # 4 biji cacat, 2 di antaranya lolos sebagai good -> recall 0.5.
    y_true = [1, 2, 3, 4, 0]
    y_pred = [0, 0, 3, 4, 0]
    result = compute_defect_metrics(y_true, y_pred, CLASS_NAMES)

    assert result["defect_recall"] == 0.5
    assert result["defect_miss_rate"] == 0.5
    assert result["defect_precision"] == 1.0  # tidak ada good yang dituduh cacat


def test_defect_metrics_keys() -> None:
    """Dict hasil memuat seluruh kunci yang diharapkan."""
    result = compute_defect_metrics([0, 1], [0, 1], CLASS_NAMES)
    assert set(result) == {
        "defect_accuracy",
        "defect_precision",
        "defect_recall",
        "defect_f1",
        "defect_miss_rate",
    }


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__, "-v"])

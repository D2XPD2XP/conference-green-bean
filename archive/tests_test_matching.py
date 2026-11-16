"""Unit test pencocokan prediksi YOLOv12 dengan ground truth (IoU + greedy match)."""

from __future__ import annotations

import pytest

from src.evaluation.matching import compute_iou, match_predictions_to_ground_truth


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def gt_boxes() -> list[tuple[int, int, int, int]]:
    """Tiga bounding box ground truth yang saling terpisah."""
    return [(0, 0, 10, 10), (20, 20, 30, 30), (40, 40, 50, 50)]


@pytest.fixture
def ground_truths(gt_boxes) -> list[dict]:
    """Ground truth berlabel bergantian: layak (0), tidak_layak (1), layak (0)."""
    return [
        {"bbox": gt_boxes[0], "class_id": 0},
        {"bbox": gt_boxes[1], "class_id": 1},
        {"bbox": gt_boxes[2], "class_id": 0},
    ]


@pytest.fixture
def perfect_predictions(gt_boxes) -> list[dict]:
    """Prediksi yang persis menimpa ketiga ground truth."""
    return [
        {"bbox": gt_boxes[0], "confidence": 0.95},
        {"bbox": gt_boxes[1], "confidence": 0.90},
        {"bbox": gt_boxes[2], "confidence": 0.85},
    ]


# --------------------------------------------------------------------------- #
# compute_iou
# --------------------------------------------------------------------------- #
def test_compute_iou_perfect_overlap() -> None:
    """Dua box identik → IoU = 1.0."""
    box = (0, 0, 10, 10)
    assert compute_iou(box, box) == pytest.approx(1.0)


def test_compute_iou_no_overlap() -> None:
    """Dua box terpisah → IoU = 0.0."""
    assert compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)


def test_compute_iou_partial_overlap() -> None:
    """Tumpang tindih sebagian → IoU = 25/175 ≈ 0.143."""
    iou = compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert iou == pytest.approx(25 / 175, abs=1e-6)


def test_compute_iou_invalid_box_returns_zero() -> None:
    """Box tidak valid (luas nol / jumlah nilai salah) → 0.0, bukan exception."""
    assert compute_iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0
    assert compute_iou((0, 0, 10), (0, 0, 10, 10)) == 0.0


# --------------------------------------------------------------------------- #
# match_predictions_to_ground_truth
# --------------------------------------------------------------------------- #
def test_match_all_true_positives(perfect_predictions, ground_truths) -> None:
    """Tiga prediksi cocok sempurna → 3 matched pair, tanpa FP/FN."""
    result = match_predictions_to_ground_truth(perfect_predictions, ground_truths)

    assert len(result["matched_pairs"]) == 3
    assert result["false_positives"] == []
    assert result["false_negatives"] == []
    assert all(pair["iou"] == pytest.approx(1.0) for pair in result["matched_pairs"])
    # Label ground truth ikut terbawa ke pasangan yang cocok.
    assert sorted(pair["gt_class_id"] for pair in result["matched_pairs"]) == [0, 0, 1]


def test_match_with_false_positive(gt_boxes, ground_truths) -> None:
    """Tiga prediksi, dua ground truth → satu prediksi menjadi false positive."""
    predictions = [
        {"bbox": gt_boxes[0], "confidence": 0.95},
        {"bbox": gt_boxes[1], "confidence": 0.90},
        {"bbox": (100, 100, 110, 110), "confidence": 0.70},  # tanpa pasangan GT
    ]
    result = match_predictions_to_ground_truth(predictions, ground_truths[:2])

    assert len(result["matched_pairs"]) == 2
    assert len(result["false_positives"]) == 1
    assert result["false_positives"][0]["bbox"] == (100, 100, 110, 110)
    assert result["false_negatives"] == []


def test_match_with_false_negative(gt_boxes, ground_truths) -> None:
    """Dua prediksi, tiga ground truth → satu ground truth menjadi false negative."""
    predictions = [
        {"bbox": gt_boxes[0], "confidence": 0.95},
        {"bbox": gt_boxes[1], "confidence": 0.90},
    ]
    result = match_predictions_to_ground_truth(predictions, ground_truths)

    assert len(result["matched_pairs"]) == 2
    assert result["false_positives"] == []
    assert len(result["false_negatives"]) == 1
    assert result["false_negatives"][0]["bbox"] == gt_boxes[2]
    assert result["false_negatives"][0]["class_id"] == 0


def test_match_below_threshold() -> None:
    """Overlap ada tetapi IoU < threshold → tidak dianggap cocok."""
    predictions = [{"bbox": (5, 5, 15, 15), "confidence": 0.9}]  # IoU ≈ 0.143
    ground_truths = [{"bbox": (0, 0, 10, 10), "class_id": 1}]

    result = match_predictions_to_ground_truth(predictions, ground_truths, iou_threshold=0.5)

    assert result["matched_pairs"] == []
    assert len(result["false_positives"]) == 1
    assert len(result["false_negatives"]) == 1


def test_match_iou_exactly_at_threshold() -> None:
    """IoU tepat sama dengan threshold tetap dihitung cocok (perbandingan >=)."""
    predictions = [{"bbox": (5, 5, 15, 15), "confidence": 0.9}]
    ground_truths = [{"bbox": (0, 0, 10, 10), "class_id": 0}]
    threshold = 25 / 175  # IoU persis pasangan di atas.

    result = match_predictions_to_ground_truth(predictions, ground_truths, iou_threshold=threshold)

    assert len(result["matched_pairs"]) == 1


def test_match_greedy_prioritizes_highest_confidence() -> None:
    """Saat dua prediksi berebut satu GT, confidence tertinggi yang menang."""
    gt = [{"bbox": (0, 0, 10, 10), "class_id": 1}]
    predictions = [
        {"bbox": (1, 1, 11, 11), "confidence": 0.60},  # IoU lebih rendah
        {"bbox": (0, 0, 10, 10), "confidence": 0.99},  # IoU sempurna
    ]
    result = match_predictions_to_ground_truth(predictions, gt)

    assert len(result["matched_pairs"]) == 1
    assert result["matched_pairs"][0]["confidence"] == pytest.approx(0.99)
    assert len(result["false_positives"]) == 1


def test_match_empty_predictions(ground_truths) -> None:
    """Tanpa prediksi (YOLOv12 gagal mendeteksi) → seluruh GT menjadi false negative."""
    result = match_predictions_to_ground_truth([], ground_truths)

    assert result["matched_pairs"] == []
    assert result["false_positives"] == []
    assert len(result["false_negatives"]) == len(ground_truths)
    assert [fn["class_id"] for fn in result["false_negatives"]] == [0, 1, 0]


def test_match_empty_ground_truths(perfect_predictions) -> None:
    """Tanpa ground truth (foto tanpa anotasi) → seluruh prediksi menjadi false positive."""
    result = match_predictions_to_ground_truth(perfect_predictions, [])

    assert result["matched_pairs"] == []
    assert result["false_negatives"] == []
    assert len(result["false_positives"]) == len(perfect_predictions)


def test_match_both_empty() -> None:
    """Kedua sisi kosong tidak menimbulkan error."""
    result = match_predictions_to_ground_truth([], [])
    assert result == {"matched_pairs": [], "false_positives": [], "false_negatives": []}


def test_match_skips_invalid_boxes() -> None:
    """Entri dengan bbox rusak dilewati tanpa crash."""
    predictions = [
        {"bbox": (0, 0, 10, 10), "confidence": 0.9},
        {"bbox": (10, 10, 5, 5), "confidence": 0.8},  # x_max < x_min
        {"confidence": 0.7},  # tanpa bbox
    ]
    ground_truths = [{"bbox": (0, 0, 10, 10), "class_id": 0}]

    result = match_predictions_to_ground_truth(predictions, ground_truths)

    assert len(result["matched_pairs"]) == 1
    assert result["false_positives"] == []


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

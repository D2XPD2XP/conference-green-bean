"""Test ambang keputusan kelas good."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.threshold import apply_good_threshold, select_threshold, sweep_thresholds


def test_threshold_zero_is_argmax() -> None:
    probs = np.array([[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.6]])
    assert apply_good_threshold(probs, 0.0).tolist() == [0, 4]


def test_doubtful_good_goes_to_best_defect() -> None:
    """P(good)=0.55 < 0.7 -> dialihkan ke cacat dengan prob tertinggi (insect)."""
    probs = np.array([[0.55, 0.05, 0.05, 0.05, 0.30], [0.90, 0.02, 0.02, 0.02, 0.04]])
    assert apply_good_threshold(probs, 0.7).tolist() == [4, 0]


def test_sweep_and_select_reduce_miss_rate() -> None:
    # Biji 0-1 good yakin; biji 2 insect yang 'hampir lolos' sebagai good.
    y_true = [0, 0, 4, 1]
    probs = np.array(
        [
            [0.95, 0.01, 0.01, 0.01, 0.02],
            [0.90, 0.02, 0.02, 0.02, 0.04],
            [0.55, 0.05, 0.05, 0.05, 0.30],
            [0.05, 0.85, 0.05, 0.03, 0.02],
        ]
    )
    sweep = sweep_thresholds(y_true, probs, thresholds=(0.0, 0.7))
    assert sweep.loc[0.0, "miss_rate"] == pytest.approx(0.5)
    assert sweep.loc[0.7, "miss_rate"] == pytest.approx(0.0)
    assert select_threshold(sweep) == pytest.approx(0.7)

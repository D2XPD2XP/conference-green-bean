"""Test test-time augmentation (ttach)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.evaluation.tta import TTA_VIEWS, wrap_tta


class _CornerProbe(nn.Module):
    """Model mainan yang hanya melihat piksel pojok kiri atas (peka orientasi)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        corner = x[:, 0, 0, 0]
        return torch.stack([corner, -corner], dim=1)


def test_none_returns_model_unchanged() -> None:
    model = nn.Linear(2, 2)
    assert wrap_tta(model, "none") is model


@pytest.mark.parametrize("mode", ["hflip", "flip_hv", "d4"])
def test_tta_keeps_output_shape(mode: str) -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5))
    assert wrap_tta(model, mode)(torch.rand(2, 3, 16, 16)).shape == (2, 5)


def test_d4_averages_over_eight_orientations() -> None:
    """Hanya 1 dari 4 pojok bernilai 1; D4 merata-rata -> nilai pojok efektif 1/4."""
    x = torch.zeros(1, 3, 8, 8)
    x[0, 0, 0, 0] = 1.0
    out = wrap_tta(_CornerProbe(), "d4")(x)
    assert out[0, 0].item() == pytest.approx(0.25)
    assert TTA_VIEWS["d4"] == 8


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        wrap_tta(nn.Linear(2, 2), "rot45")

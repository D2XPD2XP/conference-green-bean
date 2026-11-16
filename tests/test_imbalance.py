"""Unit test strategi penanganan ketimpangan kelas."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

from src.training.imbalance import (
    FocalLoss,
    build_criterion,
    class_counts_from_labels,
    compute_class_weights,
    create_weighted_sampler,
    describe_imbalance,
)

NUM_CLASSES: int = 5

#: Distribusi mirip dataset nyata: good mayoritas, 4 kelas cacat minoritas.
LABELS: list[int] = [0] * 100 + [1] * 25 + [2] * 25 + [3] * 25 + [4] * 25


# --------------------------------------------------------------------------- #
# Hitung distribusi & bobot
# --------------------------------------------------------------------------- #
def test_class_counts_includes_empty_classes() -> None:
    """Kelas tanpa sampel tetap muncul dengan jumlah 0."""
    counts = class_counts_from_labels([0, 0, 4], num_classes=NUM_CLASSES)
    assert counts.tolist() == [2, 0, 0, 0, 1]


def test_balanced_weights_match_sklearn() -> None:
    """Skema 'balanced' identik dengan sklearn compute_class_weight."""
    weights = compute_class_weights(LABELS, num_classes=NUM_CLASSES, strategy="balanced")
    expected = compute_class_weight(
        "balanced", classes=np.arange(NUM_CLASSES), y=np.asarray(LABELS)
    )
    assert np.allclose(weights.numpy(), expected)


def test_weights_favour_minority_classes() -> None:
    """Kelas minoritas mendapat bobot lebih besar daripada kelas mayoritas."""
    for strategy in ("balanced", "inverse_frequency", "effective_num_samples"):
        weights = compute_class_weights(LABELS, num_classes=NUM_CLASSES, strategy=strategy)
        assert weights[0] < weights[1], f"strategi {strategy} tidak menaikkan bobot minoritas"


def test_balanced_dataset_gives_neutral_weights() -> None:
    """Distribusi seimbang (mis. pasca-oversampling) -> bobot semua 1.0.

    Ini jaminan bahwa augmentasi offline + class weighting tidak mengoreksi
    ketimpangan dua kali.
    """
    balanced = [index for index in range(NUM_CLASSES) for _ in range(40)]
    weights = compute_class_weights(balanced, num_classes=NUM_CLASSES)
    assert np.allclose(weights.numpy(), np.ones(NUM_CLASSES))


def test_missing_class_gets_zero_weight() -> None:
    """Kelas tanpa sampel diberi bobot 0, bukan membagi nol."""
    weights = compute_class_weights([0, 0, 1, 1], num_classes=NUM_CLASSES)
    assert weights[2].item() == 0.0
    assert weights[3].item() == 0.0
    assert torch.isfinite(weights).all()


def test_unknown_strategy_raises() -> None:
    """Skema bobot tak dikenal memicu ValueError."""
    with pytest.raises(ValueError):
        compute_class_weights(LABELS, num_classes=NUM_CLASSES, strategy="tidak_ada")


def test_empty_labels_raises() -> None:
    """Daftar label kosong memicu ValueError, bukan NaN yang diam-diam lolos."""
    with pytest.raises(ValueError):
        compute_class_weights([], num_classes=NUM_CLASSES)


# --------------------------------------------------------------------------- #
# WeightedRandomSampler
# --------------------------------------------------------------------------- #
def test_weighted_sampler_balances_classes() -> None:
    """Sampling berbobot membuat kelas minoritas jauh lebih sering terambil."""
    sampler = create_weighted_sampler(LABELS, num_classes=NUM_CLASSES, seed=42)
    drawn = [LABELS[index] for index in sampler]

    counts = np.bincount(drawn, minlength=NUM_CLASSES)
    shares = counts / counts.sum()
    # Tanpa sampler, kelas 0 menguasai 50%; dengan sampler harus ~20%.
    assert abs(shares[0] - 0.2) < 0.08, f"distribusi tidak seimbang: {shares}"


def test_weighted_sampler_is_reproducible() -> None:
    """Seed yang sama menghasilkan urutan sampling yang sama."""
    first = list(create_weighted_sampler(LABELS, num_classes=NUM_CLASSES, seed=42))
    second = list(create_weighted_sampler(LABELS, num_classes=NUM_CLASSES, seed=42))
    third = list(create_weighted_sampler(LABELS, num_classes=NUM_CLASSES, seed=7))

    assert first == second
    assert first != third


def test_weighted_sampler_length() -> None:
    """num_samples default = jumlah sampel dataset."""
    assert len(create_weighted_sampler(LABELS, num_classes=NUM_CLASSES)) == len(LABELS)
    assert len(create_weighted_sampler(LABELS, num_samples=50, num_classes=NUM_CLASSES)) == 50


def test_weighted_sampler_empty_raises() -> None:
    """Label kosong memicu ValueError."""
    with pytest.raises(ValueError):
        create_weighted_sampler([], num_classes=NUM_CLASSES)


# --------------------------------------------------------------------------- #
# Focal Loss
# --------------------------------------------------------------------------- #
def test_focal_loss_equals_cross_entropy_when_gamma_zero() -> None:
    """gamma = 0 membuat Focal Loss setara CrossEntropyLoss."""
    torch.manual_seed(0)
    logits = torch.randn(16, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (16,))

    focal = FocalLoss(gamma=0.0)(logits, targets)
    cross_entropy = nn.CrossEntropyLoss()(logits, targets)

    assert torch.allclose(focal, cross_entropy, atol=1e-6)


def test_focal_loss_downweights_easy_examples() -> None:
    """Sampel yang sudah diprediksi yakin menyumbang loss lebih kecil."""
    easy = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    hard = torch.tensor([[0.2, 0.1, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])

    focal = FocalLoss(gamma=2.0, reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")

    easy_ratio = (focal(easy, target) / ce(easy, target)).item()
    hard_ratio = (focal(hard, target) / ce(hard, target)).item()

    assert easy_ratio < hard_ratio, "sampel mudah seharusnya ditekan lebih kuat"


def test_focal_loss_accepts_class_weights() -> None:
    """alpha (bobot kelas) diterima dan menghasilkan skalar berhingga."""
    weights = compute_class_weights(LABELS, num_classes=NUM_CLASSES)
    torch.manual_seed(0)
    loss = FocalLoss(alpha=weights, gamma=2.0)(
        torch.randn(8, NUM_CLASSES), torch.randint(0, NUM_CLASSES, (8,))
    )
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_focal_loss_rejects_bad_arguments() -> None:
    """Argumen tidak valid ditolak di konstruktor."""
    with pytest.raises(ValueError):
        FocalLoss(gamma=-1.0)
    with pytest.raises(ValueError):
        FocalLoss(reduction="rata-rata")


def test_focal_loss_alpha_follows_device() -> None:
    """alpha terdaftar sebagai buffer sehingga ikut berpindah device."""
    loss_fn = FocalLoss(alpha=torch.ones(NUM_CLASSES))
    assert "alpha" in dict(loss_fn.named_buffers())


# --------------------------------------------------------------------------- #
# build_criterion
# --------------------------------------------------------------------------- #
def test_build_criterion_per_strategy() -> None:
    """Tiap strategi memetakan ke fungsi loss yang tepat."""
    weights = compute_class_weights(LABELS, num_classes=NUM_CLASSES)

    assert isinstance(build_criterion("none"), nn.CrossEntropyLoss)
    assert build_criterion("none").weight is None

    weighted = build_criterion("class_weights", class_weights=weights)
    assert isinstance(weighted, nn.CrossEntropyLoss)
    assert weighted.weight is not None

    # weighted_sampler menyeimbangkan di sisi sampling, jadi loss tanpa bobot.
    sampler_loss = build_criterion("weighted_sampler", class_weights=weights)
    assert isinstance(sampler_loss, nn.CrossEntropyLoss)
    assert sampler_loss.weight is None

    assert isinstance(build_criterion("focal_loss", class_weights=weights), FocalLoss)


def test_build_criterion_unknown_strategy_raises() -> None:
    """Strategi tak dikenal memicu ValueError."""
    with pytest.raises(ValueError):
        build_criterion("smote_ajaib")


# --------------------------------------------------------------------------- #
# describe_imbalance
# --------------------------------------------------------------------------- #
def test_describe_imbalance_reports_ratio_and_baseline() -> None:
    """Rasio ketimpangan dan garis dasar tebakan mayoritas dihitung benar."""
    info = describe_imbalance({"good": 100, "black": 25, "chipped": 25})
    assert info["majority_class"] == "good"
    assert info["imbalance_ratio"] == 4.0
    assert info["majority_share"] == 0.6667


def test_describe_imbalance_handles_empty() -> None:
    """Distribusi kosong tidak memunculkan exception."""
    info = describe_imbalance({"good": 0, "black": 0})
    assert info["total"] == 0
    assert info["imbalance_ratio"] == 0.0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

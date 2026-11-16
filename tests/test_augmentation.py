"""Unit test augmentasi offline.

Citra sintetis dibuat di ``tmp_path`` (bukan di dalam repo), sesuai konvensi
proyek: tidak ada data dummy yang ikut ter-commit.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.offline_augmentation import (
    augment_image,
    build_dataset,
    copy_split,
    format_augmentation_report,
    generate_augmented_dataset,
    variant_seed,
)

CLASS_NAMES: list[str] = ["good", "black", "chipped", "dry_pods", "insect_damage"]


def _write_crops(split_dir: Path, counts: dict[str, int], size: int = 72) -> None:
    """Buat citra crop sintetis per kelas di ``split_dir``."""
    rng = np.random.default_rng(7)
    for class_name, count in counts.items():
        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            array = rng.integers(0, 255, size=(size, size + 3, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"img_bean_{index:03d}.jpg"), array)


@pytest.fixture()
def raw_split(tmp_path: Path) -> Path:
    """Sediakan satu split crops_raw dengan distribusi timpang."""
    split_dir = tmp_path / "crops_raw" / "train"
    _write_crops(split_dir, {"good": 4, "black": 2, "chipped": 1, "dry_pods": 0, "insect_damage": 1})
    return split_dir


# --------------------------------------------------------------------------- #
# Seed & determinisme
# --------------------------------------------------------------------------- #
def test_variant_seed_is_deterministic() -> None:
    """Seed varian hanya bergantung pada argumennya, bukan urutan pemanggilan."""
    first = variant_seed(42, "black", "img_bean_000", 1)
    second = variant_seed(42, "black", "img_bean_000", 1)
    assert first == second


def test_variant_seed_differs_per_variant_and_class() -> None:
    """Varian, kelas, dan nama file yang berbeda menghasilkan seed berbeda."""
    base = variant_seed(42, "black", "img_bean_000", 1)
    assert base != variant_seed(42, "black", "img_bean_000", 2)
    assert base != variant_seed(42, "chipped", "img_bean_000", 1)
    assert base != variant_seed(42, "black", "img_bean_001", 1)
    assert base != variant_seed(43, "black", "img_bean_000", 1)


def test_augment_image_is_reproducible() -> None:
    """Seed yang sama menghasilkan citra teraugmentasi yang identik."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(72, 75, 3), dtype=np.uint8)

    first = augment_image(image, seed=123)
    second = augment_image(image, seed=123)
    different = augment_image(image, seed=124)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert first.shape == image.shape
    assert first.dtype == np.uint8


# --------------------------------------------------------------------------- #
# generate_augmented_dataset
# --------------------------------------------------------------------------- #
def test_multiplier_controls_output_count(raw_split: Path, tmp_path: Path) -> None:
    """Jumlah keluaran per kelas = multiplier x jumlah citra asli."""
    out_dir = tmp_path / "crops" / "train"
    multipliers = {"good": 1, "black": 3, "chipped": 3, "dry_pods": 3, "insect_damage": 3}

    stats = generate_augmented_dataset(
        raw_split, out_dir, multipliers=multipliers, class_names=CLASS_NAMES
    )

    assert stats["per_class"]["good"]["output"] == 4  # 4 asli x 1
    assert stats["per_class"]["black"]["output"] == 6  # 2 asli x 3
    assert stats["per_class"]["chipped"]["output"] == 3  # 1 asli x 3
    assert stats["per_class"]["dry_pods"]["output"] == 0  # tidak ada sumbernya
    assert stats["total_output"] == 4 + 6 + 3 + 0 + 3
    assert len(list((out_dir / "black").glob("*.jpg"))) == 6


def test_original_image_always_saved(raw_split: Path, tmp_path: Path) -> None:
    """Tiap citra asli disimpan dengan sufiks _orig, varian dengan _augNN."""
    out_dir = tmp_path / "crops" / "train"
    generate_augmented_dataset(
        raw_split,
        out_dir,
        multipliers={name: 3 for name in CLASS_NAMES},
        class_names=CLASS_NAMES,
    )

    names = sorted(p.name for p in (out_dir / "black").glob("*.jpg"))
    assert "img_bean_000_orig.jpg" in names
    assert "img_bean_000_aug01.jpg" in names
    assert "img_bean_000_aug02.jpg" in names
    assert len([n for n in names if n.endswith("_orig.jpg")]) == 2


def test_apply_augmentation_false_only_copies(raw_split: Path, tmp_path: Path) -> None:
    """apply_augmentation=False mengabaikan multiplier dan hanya menyalin."""
    out_dir = tmp_path / "crops" / "valid"
    stats = generate_augmented_dataset(
        raw_split,
        out_dir,
        multipliers={name: 5 for name in CLASS_NAMES},
        class_names=CLASS_NAMES,
        apply_augmentation=False,
    )

    assert stats["total_output"] == stats["total_original"] == 8
    assert not list(out_dir.glob("*/*_aug*.jpg"))


def test_generated_dataset_is_byte_identical_across_runs(
    raw_split: Path, tmp_path: Path
) -> None:
    """Dua kali run dengan seed sama menghasilkan citra yang identik."""
    first_dir = tmp_path / "run1" / "train"
    second_dir = tmp_path / "run2" / "train"
    multipliers = {name: 3 for name in CLASS_NAMES}

    for out_dir in (first_dir, second_dir):
        generate_augmented_dataset(
            raw_split, out_dir, multipliers=multipliers, class_names=CLASS_NAMES, random_seed=42
        )

    produced = sorted(p.relative_to(first_dir) for p in first_dir.glob("*/*.jpg"))
    assert produced, "tidak ada citra yang dihasilkan"
    for relative in produced:
        left = cv2.imread(str(first_dir / relative))
        right = cv2.imread(str(second_dir / relative))
        assert np.array_equal(left, right), f"{relative} berbeda antar-run"


def test_clean_removes_stale_files(raw_split: Path, tmp_path: Path) -> None:
    """clean=True menghapus sisa crop versi sebelumnya di folder tujuan."""
    out_dir = tmp_path / "crops" / "train"
    stale = out_dir / "black"
    stale.mkdir(parents=True)
    cv2.imwrite(str(stale / "versi_lama_bean_999_orig.jpg"), np.zeros((8, 8, 3), np.uint8))

    stats = generate_augmented_dataset(
        raw_split,
        out_dir,
        multipliers={name: 1 for name in CLASS_NAMES},
        class_names=CLASS_NAMES,
        clean=True,
    )

    assert stats["num_removed"] == 1
    assert not (stale / "versi_lama_bean_999_orig.jpg").exists()


def test_missing_input_raises(tmp_path: Path) -> None:
    """Folder masukan yang tidak ada memunculkan FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        generate_augmented_dataset(tmp_path / "tidak_ada", tmp_path / "out")


# --------------------------------------------------------------------------- #
# copy_split & build_dataset
# --------------------------------------------------------------------------- #
def test_copy_split_preserves_filenames(raw_split: Path, tmp_path: Path) -> None:
    """copy_split menyalin apa adanya, tanpa sufiks dan tanpa augmentasi."""
    out_dir = tmp_path / "crops" / "test"
    stats = copy_split(raw_split, out_dir, class_names=CLASS_NAMES)

    assert stats["total_output"] == 8
    assert (out_dir / "black" / "img_bean_000.jpg").exists()
    assert not list(out_dir.glob("*/*_aug*.jpg"))


def test_build_dataset_augments_train_only(tmp_path: Path) -> None:
    """Hanya split di augment_splits yang diaugmentasi; sisanya disalin."""
    raw_root = tmp_path / "crops_raw"
    for split in ("train", "valid", "test"):
        _write_crops(raw_root / split, {"good": 2, "black": 1})

    out_root = tmp_path / "crops"
    results = build_dataset(
        raw_root,
        out_root,
        augment_splits=("train",),
        copy_splits=("valid", "test"),
        multipliers={"good": 1, "black": 3},
        class_names=CLASS_NAMES,
    )

    assert results["train"]["augmented"] is True
    assert results["valid"]["augmented"] is False
    assert results["train"]["per_class"]["black"]["output"] == 3
    assert results["valid"]["per_class"]["black"]["output"] == 1
    # Valid & test wajib bersih dari citra buatan.
    assert not list((out_root / "valid").glob("*/*_aug*.jpg"))
    assert not list((out_root / "test").glob("*/*_aug*.jpg"))


def test_build_dataset_skips_missing_split(tmp_path: Path, caplog) -> None:
    """Split yang foldernya tidak ada dilewati dengan peringatan."""
    raw_root = tmp_path / "crops_raw"
    _write_crops(raw_root / "train", {"good": 1})

    with caplog.at_level("WARNING"):
        results = build_dataset(
            raw_root,
            tmp_path / "crops",
            augment_splits=("train",),
            copy_splits=("valid", "test"),
            class_names=CLASS_NAMES,
        )

    assert set(results) == {"train"}
    assert "valid" in caplog.text


def test_report_mentions_every_class(raw_split: Path, tmp_path: Path) -> None:
    """Laporan augmentasi memuat seluruh kelas dan rasio keseimbangan."""
    results = build_dataset(
        raw_split.parent,
        tmp_path / "crops",
        augment_splits=("train",),
        copy_splits=(),
        multipliers={name: 2 for name in CLASS_NAMES},
        class_names=CLASS_NAMES,
    )
    report = format_augmentation_report(results, class_names=CLASS_NAMES)

    for name in CLASS_NAMES:
        assert name in report
    assert "balance before" in report
    assert "balance after" in report


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

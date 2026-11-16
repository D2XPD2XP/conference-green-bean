"""Unit test crop biji dari anotasi Roboflow.

Citra dan label dibuat sementara di ``tmp_path`` (bukan di dalam repo) agar
pengujian tidak bergantung pada dataset nyata.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.crop_from_annotation import crop_beans_from_annotations, yolo_to_pixel_bbox

#: Nama kelas yang dipakai seluruh test (urutan = class_id).
CLASS_NAMES: list[str] = ["good", "black", "chipped", "dry_pods", "insect_damage"]


# --------------------------------------------------------------------------- #
# Fixture pembantu
# --------------------------------------------------------------------------- #
@pytest.fixture()
def dataset_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Sediakan tuple ``(images_dir, labels_dir, output_dir)`` yang kosong."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "crops"
    images_dir.mkdir()
    labels_dir.mkdir()
    return images_dir, labels_dir, output_dir


def _write_image(images_dir: Path, stem: str, width: int = 100, height: int = 100) -> Path:
    """Tulis citra sintetis berwarna acak deterministik ke ``images_dir``."""
    rng = np.random.default_rng(42)
    array = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    path = images_dir / f"{stem}.jpg"
    cv2.imwrite(str(path), array)
    return path


def _write_label(labels_dir: Path, stem: str, lines: list[str]) -> Path:
    """Tulis file label YOLO ke ``labels_dir``."""
    path = labels_dir / f"{stem}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Konversi koordinat YOLO -> piksel
# --------------------------------------------------------------------------- #
def test_crop_coordinate_conversion() -> None:
    """Format YOLO ternormalisasi dikonversi benar ke koordinat piksel."""
    assert yolo_to_pixel_bbox(0.5, 0.5, 1.0, 1.0, 100, 100) == (0, 0, 100, 100)
    assert yolo_to_pixel_bbox(0.5, 0.5, 0.5, 0.5, 200, 100) == (50, 25, 150, 75)
    assert yolo_to_pixel_bbox(0.25, 0.25, 0.5, 0.5, 100, 100) == (0, 0, 50, 50)


def test_crop_out_of_bounds_clipping() -> None:
    """Box yang melewati tepi gambar di-clip ke dimensi gambar."""
    x1, y1, x2, y2 = yolo_to_pixel_bbox(0.95, 0.95, 0.5, 0.5, 100, 100)
    assert (x1, y1) >= (0, 0)
    assert x2 <= 100 and y2 <= 100
    assert x1 < x2 and y1 < y2


def test_result_is_valid_box() -> None:
    """Box sangat kecil tetap menghasilkan area > 0 (x1 < x2, y1 < y2)."""
    x1, y1, x2, y2 = yolo_to_pixel_bbox(0.1, 0.1, 0.01, 0.01, 64, 64)
    assert x1 < x2
    assert y1 < y2


def test_padding_ratio_expands_box() -> None:
    """``padding_ratio`` memperlebar box secara simetris lalu tetap di-clip."""
    tight = yolo_to_pixel_bbox(0.5, 0.5, 0.5, 0.5, 100, 100)
    padded = yolo_to_pixel_bbox(0.5, 0.5, 0.5, 0.5, 100, 100, padding_ratio=0.1)

    assert tight == (25, 25, 75, 75)
    assert padded == (20, 20, 80, 80)

    # Padding besar tidak boleh membuat box keluar dari gambar.
    clipped = yolo_to_pixel_bbox(0.5, 0.5, 0.9, 0.9, 100, 100, padding_ratio=0.5)
    assert clipped == (0, 0, 100, 100)


# --------------------------------------------------------------------------- #
# Perilaku crop_beans_from_annotations
# --------------------------------------------------------------------------- #
def test_crop_produces_correct_folders(dataset_dirs: tuple[Path, Path, Path]) -> None:
    """Setelah crop, subfolder tiap kelas tersedia — termasuk kelas tanpa crop."""
    images_dir, labels_dir, output_dir = dataset_dirs
    _write_image(images_dir, "img001")
    _write_label(labels_dir, "img001", ["0 0.25 0.25 0.4 0.4"])

    crop_beans_from_annotations(images_dir, labels_dir, output_dir, CLASS_NAMES)

    for name in CLASS_NAMES:
        assert (output_dir / name).is_dir(), f"subfolder {name} tidak dibuat"


def test_crop_correct_class_assignment(dataset_dirs: tuple[Path, Path, Path]) -> None:
    """Tiap class_id masuk ke folder kelas sesuai indeksnya di CLASS_NAMES."""
    images_dir, labels_dir, output_dir = dataset_dirs
    _write_image(images_dir, "img001")
    _write_label(
        labels_dir,
        "img001",
        [
            "0 0.25 0.25 0.3 0.3",  # good
            "1 0.75 0.75 0.3 0.3",  # black
            "1 0.75 0.25 0.3 0.3",  # black
            "4 0.25 0.75 0.3 0.3",  # insect_damage
        ],
    )

    stats = crop_beans_from_annotations(images_dir, labels_dir, output_dir, CLASS_NAMES)

    assert stats["num_crops"] == 4
    assert stats["class_distribution"] == {
        "good": 1,
        "black": 2,
        "chipped": 0,
        "dry_pods": 0,
        "insect_damage": 1,
    }
    assert len(list((output_dir / "good").glob("*.jpg"))) == 1
    assert len(list((output_dir / "black").glob("*.jpg"))) == 2
    assert len(list((output_dir / "insect_damage").glob("*.jpg"))) == 1


def test_crop_defaults_to_config_class_names(dataset_dirs: tuple[Path, Path, Path]) -> None:
    """Tanpa argumen class_names, dipakai CLASS_NAMES dari configs."""
    from configs.base_config import CLASS_NAMES as CONFIG_CLASS_NAMES

    images_dir, labels_dir, output_dir = dataset_dirs
    _write_image(images_dir, "img001")
    _write_label(labels_dir, "img001", ["3 0.5 0.5 0.4 0.4"])

    stats = crop_beans_from_annotations(images_dir, labels_dir, output_dir)

    assert set(stats["class_distribution"]) == set(CONFIG_CLASS_NAMES)
    assert stats["class_distribution"][CONFIG_CLASS_NAMES[3]] == 1


def test_crop_handles_missing_labels(
    dataset_dirs: tuple[Path, Path, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Gambar tanpa file label dilewati dengan peringatan, bukan exception."""
    images_dir, labels_dir, output_dir = dataset_dirs
    _write_image(images_dir, "berlabel")
    _write_image(images_dir, "tanpa_label")
    _write_label(labels_dir, "berlabel", ["0 0.5 0.5 0.4 0.4"])

    with caplog.at_level("WARNING"):
        stats = crop_beans_from_annotations(images_dir, labels_dir, output_dir, CLASS_NAMES)

    assert stats["num_images"] == 1
    assert stats["num_images_skipped"] == 1
    assert stats["num_crops"] == 1
    assert "tanpa_label" in caplog.text


def test_crop_skips_invalid_class_id(
    dataset_dirs: tuple[Path, Path, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """class_id di luar rentang class_names dilewati dengan peringatan."""
    images_dir, labels_dir, output_dir = dataset_dirs
    _write_image(images_dir, "img001")
    _write_label(labels_dir, "img001", ["0 0.5 0.5 0.4 0.4", "7 0.25 0.25 0.2 0.2"])

    with caplog.at_level("WARNING"):
        stats = crop_beans_from_annotations(images_dir, labels_dir, output_dir, CLASS_NAMES)

    assert stats["num_crops"] == 1
    assert "class_id 7" in caplog.text


def test_crop_saves_within_image_bounds(dataset_dirs: tuple[Path, Path, Path]) -> None:
    """Box yang melewati tepi tetap menghasilkan crop valid (sudah di-clip)."""
    images_dir, labels_dir, output_dir = dataset_dirs
    _write_image(images_dir, "img001", width=100, height=100)
    # Box berpusat di pojok kanan-bawah, separuhnya di luar gambar.
    _write_label(labels_dir, "img001", ["0 0.98 0.98 0.4 0.4"])

    stats = crop_beans_from_annotations(images_dir, labels_dir, output_dir, CLASS_NAMES)

    assert stats["num_crops"] == 1
    crop_path = next((output_dir / "good").glob("*.jpg"))
    crop = cv2.imread(str(crop_path))
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.shape[0] <= 100 and crop.shape[1] <= 100


def test_crop_raises_on_missing_directory(tmp_path: Path) -> None:
    """Direktori sumber yang tidak ada memunculkan FileNotFoundError yang jelas."""
    with pytest.raises(FileNotFoundError):
        crop_beans_from_annotations(
            tmp_path / "tidak_ada", tmp_path, tmp_path / "out", CLASS_NAMES
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

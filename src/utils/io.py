"""I/O: penyimpanan/pemuatan checkpoint model, JSON, dan anotasi YOLO."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(model: nn.Module, path: str | Path, **extra: object) -> Path:
    """Simpan state_dict model (plus metadata opsional) ke disk.

    Args:
        model: Model yang akan disimpan.
        path: Path tujuan (``.pt``). Direktori induk dibuat bila perlu.
        **extra: Pasangan kunci-nilai tambahan (mis. ``epoch``, ``val_acc``)
            yang ikut disimpan dalam checkpoint.

    Returns:
        Path checkpoint yang tersimpan.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model_state_dict": model.state_dict(), **extra}
    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    map_location: str | None = None,
) -> dict:
    """Muat state_dict ke model dan kembalikan seluruh isi checkpoint.

    Args:
        model: Model target (diisi in-place via ``load_state_dict``).
        path: Path file checkpoint.
        map_location: Argumen ``map_location`` untuk ``torch.load`` (mis.
            ``"cpu"``).

    Returns:
        Dict checkpoint lengkap (berisi ``model_state_dict`` dan metadata).

    Raises:
        FileNotFoundError: Bila file tidak ada.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_json(data: dict, path: str | Path, indent: int = 2) -> Path:
    """Simpan dict ke file JSON (UTF-8).

    Args:
        data: Data serializable.
        path: Path tujuan ``.json``.
        indent: Indentasi JSON.

    Returns:
        Path file yang tersimpan.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    return path


def load_yolo_annotations(label_path: str | Path) -> list[tuple[int, float, float, float, float]]:
    """Baca satu file label YOLO menjadi daftar bounding box ternormalisasi.

    Args:
        label_path: Path file ``.txt`` berformat
            ``<class_id> <x_center> <y_center> <width> <height>`` per baris.

    Returns:
        Daftar tuple ``(class_id, x_center, y_center, width, height)``. Baris
        yang tidak valid dilewati dengan peringatan di log.

    Raises:
        FileNotFoundError: Bila file tidak ada.

    Example:
        >>> # File berisi satu baris "0 0.5 0.5 1.0 1.0"
        >>> load_yolo_annotations("img001.txt")  # doctest: +SKIP
        [(0, 0.5, 0.5, 1.0, 1.0)]
    """
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"File label tidak ditemukan: {label_path}")

    boxes: list[tuple[int, float, float, float, float]] = []
    for lineno, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            logger.warning(
                "Baris %d di %s tidak valid (butuh 5 nilai, ada %d); dilewati.",
                lineno,
                label_path.name,
                len(parts),
            )
            continue

        try:
            class_id = int(float(parts[0]))
            x_c, y_c, w, h = (float(v) for v in parts[1:])
        except ValueError:
            logger.warning("Baris %d di %s gagal di-parse; dilewati.", lineno, label_path.name)
            continue

        boxes.append((class_id, x_c, y_c, w, h))
    return boxes

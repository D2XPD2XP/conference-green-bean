"""Test-time augmentation (TTA) untuk evaluasi, memakai library ``ttach``.

Setiap biji diprediksi dari beberapa orientasi lalu logit-nya dirata-ratakan.
Orientasi biji kopi tidak bermakna (biji yang dibalik/diputar tetap kelas yang
sama), sehingga rata-rata beberapa pandangan meredam kesalahan yang hanya
muncul pada satu orientasi — termasuk lubang kecil ``insect_damage`` yang
tertangkap pada sebagian orientasi saja.

TTA hanya mengubah cara **prediksi**; bobot model dan pelatihan tidak berubah,
sehingga skema feature extraction tetap utuh. Biayanya: waktu inferensi kali
jumlah orientasi.

Hasil uji saring pada VALID (2026-10-15), F1 makro tanpa TTA -> D4:
ResNet-50 0,753 -> 0,786 · MobileNetV3 0,721 -> 0,756 · InceptionV3 0,668 -> 0,717.
"""

from __future__ import annotations

import torch.nn as nn
import ttach as tta

from configs.base_config import TTA_MODE

#: Mode TTA yang didukung -> jumlah orientasi.
TTA_VIEWS: dict[str, int] = {"none": 1, "hflip": 2, "flip_hv": 4, "d4": 8}

#: Pilihan mode TTA untuk CLI.
TTA_CHOICES: tuple[str, ...] = tuple(TTA_VIEWS)


def _transforms(mode: str) -> tta.Compose:
    if mode == "hflip":
        return tta.Compose([tta.HorizontalFlip()])
    if mode == "flip_hv":
        return tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
    if mode == "d4":
        return tta.aliases.d4_transform()
    raise ValueError(f"Mode TTA '{mode}' tidak dikenal. Pilihan: {list(TTA_CHOICES)}")


def wrap_tta(model: nn.Module, mode: str = TTA_MODE) -> nn.Module:
    """Bungkus model agar prediksinya rata-rata dari beberapa orientasi.

    Args:
        model: Model terlatih (keluaran logit ``(N, C)``).
        mode: ``"none"``, ``"hflip"``, ``"flip_hv"``, atau ``"d4"``. Default
            ``TTA_MODE`` dari config.

    Returns:
        ``model`` apa adanya bila ``mode == "none"``, selain itu wrapper
        ``ttach.ClassificationTTAWrapper`` (rata-rata logit).

    Raises:
        ValueError: Bila ``mode`` tidak dikenal.
    """
    if mode == "none":
        return model
    return tta.ClassificationTTAWrapper(model, _transforms(mode), merge_mode="mean")

"""Factory pemilih model CNN berdasarkan nama.

Menyediakan satu titik masuk ``get_model(name)`` yang mendelegasikan ke builder
masing-masing arsitektur. Seluruh model dikembalikan dalam mode feature
extraction (backbone beku, head baru dapat dilatih).
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from configs.base_config import NUM_CLASSES
from src.models.inceptionv3 import build_inceptionv3
from src.models.mobilenetv3 import build_mobilenetv3
from src.models.resnet50 import build_resnet50

#: Registri builder per nama model.
_BUILDERS: dict[str, Callable[[int, bool], nn.Module]] = {
    "resnet50": build_resnet50,
    "mobilenetv3": build_mobilenetv3,
    "inceptionv3": build_inceptionv3,
}

#: Nama model yang tersedia melalui factory.
AVAILABLE_MODELS: tuple[str, ...] = tuple(_BUILDERS)


def get_model(
    name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True
) -> nn.Module:
    """Factory untuk mendapatkan model CNN dengan feature extraction.

    Semua model:
        - Memuat bobot pra-latih ImageNet.
        - Membekukan backbone (``requires_grad = False``).
        - Mengganti lapisan klasifikasi akhir dengan
          ``Linear(in_features, num_classes)``.

    Args:
        name: Nama arsitektur (``"resnet50"``, ``"mobilenetv3"``,
            ``"inceptionv3"``).
        num_classes: Jumlah kelas keluaran. Default ``NUM_CLASSES`` dari
            ``configs.base_config`` (5), jadi memanggil tanpa argumen ini tidak
            pernah diam-diam membangun model dengan jumlah kelas yang salah.
        pretrained: Bila ``True``, muat bobot ImageNet.

    Returns:
        Instance ``nn.Module`` yang diminta.

    Raises:
        ValueError: Bila ``name`` tidak dikenal.

    Example:
        >>> model = get_model("resnet50")
        >>> trainable = [p for p in model.parameters() if p.requires_grad]
        >>> all(p.requires_grad for p in trainable)
        True
    """
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(
            f"Model '{name}' tidak dikenal. Pilihan: {list(AVAILABLE_MODELS)}"
        )
    return _BUILDERS[key](num_classes, pretrained)

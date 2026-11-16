"""ResNet-50 dengan transfer learning (feature extraction).

Backbone dibekukan; hanya lapisan fully-connected akhir yang dilatih.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from configs.base_config import NUM_CLASSES


def build_resnet50(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Bangun ResNet-50 untuk feature extraction.

    Memuat bobot pra-latih ImageNet, membekukan seluruh backbone
    (``requires_grad = False``), lalu mengganti head ``fc`` dengan
    ``Linear(in_features, num_classes)`` yang tetap dapat dilatih.

    Args:
        num_classes: Jumlah kelas keluaran. Default ``NUM_CLASSES`` (5).
        pretrained: Bila ``True``, muat bobot ImageNet (``IMAGENET1K_V2``).

    Returns:
        Model ``nn.Module`` dengan hanya head yang ``requires_grad = True``.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Bekukan backbone.
    for param in model.parameters():
        param.requires_grad = False

    # Ganti head; parameter baru default requires_grad = True.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

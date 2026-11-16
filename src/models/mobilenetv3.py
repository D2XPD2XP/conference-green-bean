"""MobileNetV3-Large dengan transfer learning (feature extraction).

Backbone dibekukan; hanya lapisan Linear akhir pada classifier yang dilatih.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights

from configs.base_config import NUM_CLASSES


def build_mobilenetv3(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Bangun MobileNetV3-Large untuk feature extraction.

    Memuat bobot pra-latih ImageNet, membekukan seluruh backbone, lalu mengganti
    lapisan Linear terakhir pada ``classifier`` dengan
    ``Linear(in_features, num_classes)``.

    Args:
        num_classes: Jumlah kelas keluaran. Default ``NUM_CLASSES`` (5).
        pretrained: Bila ``True``, muat bobot ImageNet.

    Returns:
        Model ``nn.Module`` dengan hanya head yang ``requires_grad = True``.
    """
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    # Bekukan backbone.
    for param in model.parameters():
        param.requires_grad = False

    # Lapisan Linear terakhir pada classifier MobileNetV3 berada di indeks -1.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

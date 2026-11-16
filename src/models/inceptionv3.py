"""InceptionV3 dengan transfer learning (feature extraction).

Input harus berukuran 299x299. Backbone dibekukan; head utama (``fc``) diganti.
Auxiliary classifier dinonaktifkan agar keluaran tunggal saat train/eval.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import Inception_V3_Weights

from configs.base_config import NUM_CLASSES


def build_inceptionv3(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Bangun InceptionV3 untuk feature extraction.

    Memuat bobot pra-latih ImageNet, membekukan seluruh backbone, menonaktifkan
    auxiliary logits, lalu mengganti head ``fc`` dengan
    ``Linear(in_features, num_classes)``.

    Args:
        num_classes: Jumlah kelas keluaran. Default ``NUM_CLASSES`` (5).
        pretrained: Bila ``True``, muat bobot ImageNet.

    Returns:
        Model ``nn.Module`` dengan hanya head yang ``requires_grad = True``.

    Note:
        Model ini mengharapkan citra input 299x299 (lihat ``CNN_CONFIG``).
    """
    weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
    # transform_input WAJIB eksplisit True. torchvision hanya menyalakannya
    # otomatis bila weights diberikan; flag ini tidak tersimpan di state_dict.
    # Tanpa ini, model yang dibangun ulang dengan pretrained=False untuk memuat
    # checkpoint (scripts.evaluate, analisis) menerima input berskala salah dan
    # prediksinya kolaps ke satu kelas, padahal bobotnya identik.
    # aux_logits mengikuti bobot; nonaktifkan cabang aux agar forward tunggal.
    model = models.inception_v3(
        weights=weights,
        aux_logits=True,
        transform_input=True,
        init_weights=not pretrained,
    )
    model.aux_logits = False
    model.AuxLogits = None

    # Bekukan backbone.
    for param in model.parameters():
        param.requires_grad = False

    # Ganti head utama.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

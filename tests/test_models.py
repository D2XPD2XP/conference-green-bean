"""Unit test model factory: pastikan hanya head yang dapat dilatih.

Test ini memuat bobot pra-latih dari torchvision; lewati bila unduhan gagal
(mis. tanpa koneksi) dengan pretrained=False agar arsitektur tetap teruji.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from configs.base_config import NUM_CLASSES
from configs.cnn_config import CNN_CONFIG
from src.models.model_factory import AVAILABLE_MODELS, get_model


@pytest.mark.parametrize("name", AVAILABLE_MODELS)
def test_backbone_frozen_head_trainable(name: str) -> None:
    """Hanya parameter head (baru) yang memiliki requires_grad = True."""
    model = get_model(name, num_classes=NUM_CLASSES, pretrained=False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]

    # Harus ada parameter beku (backbone) dan parameter yang dilatih (head).
    assert len(frozen) > 0, "Backbone seharusnya dibekukan."
    assert len(trainable) > 0, "Head seharusnya dapat dilatih."

    # Jumlah parameter head untuk 5 kelas tetap jauh lebih kecil dari backbone.
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    assert n_trainable < n_total, "Head harus lebih kecil dari total."


@pytest.mark.parametrize("name", AVAILABLE_MODELS)
def test_output_dimension(name: str) -> None:
    """Forward pass menghasilkan logit sebanyak NUM_CLASSES (5), bukan 2."""
    model = get_model(name, num_classes=NUM_CLASSES, pretrained=False).eval()
    assert isinstance(model, nn.Module)

    input_size = int(CNN_CONFIG[name]["input_size"])
    with torch.no_grad():
        output = model(torch.randn(2, 3, input_size, input_size))

    assert output.shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("name", AVAILABLE_MODELS)
def test_num_classes_in_config_matches_base_config(name: str) -> None:
    """CNN_CONFIG tidak boleh meng-hardcode jumlah kelas yang berbeda."""
    assert int(CNN_CONFIG[name]["num_classes"]) == NUM_CLASSES


@pytest.mark.parametrize("name", AVAILABLE_MODELS)
def test_default_num_classes_follows_config(name: str) -> None:
    """Tanpa argumen num_classes, model mengikuti NUM_CLASSES dari config.

    Default yang di-hardcode akan membangun head berukuran salah secara diam-diam:
    tidak ada error, model tetap bisa dilatih, dan baru ketahuan saat metriknya
    aneh. Test ini mengunci defaultnya ke satu sumber kebenaran.
    """
    model = get_model(name, pretrained=False).eval()

    input_size = int(CNN_CONFIG[name]["input_size"])
    with torch.no_grad():
        output = model(torch.randn(1, 3, input_size, input_size))

    assert output.shape[1] == NUM_CLASSES


def test_unknown_model_raises() -> None:
    """Nama model tak dikenal memicu ValueError."""
    with pytest.raises(ValueError):
        get_model("not_a_model")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


def test_inceptionv3_transform_input_independent_of_pretrained() -> None:
    """Model untuk memuat checkpoint harus memproses input sama seperti saat latih.

    ``transform_input`` tidak tersimpan di ``state_dict``. Bila ia bergantung pada
    ``pretrained``, model yang dibangun ulang dengan ``pretrained=False`` untuk
    evaluasi akan menerima input berskala salah.
    """
    model = get_model("inceptionv3", pretrained=False)
    assert model.transform_input is True

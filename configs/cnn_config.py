"""Konfigurasi seragam untuk ketiga arsitektur CNN.

Hyperparameter dibuat identik antar-model (sesuai Tabel 3.2-3.4 proposal) agar
perbandingan arsitektur bersifat adil. Perbedaan utama hanya ``input_size``:
InceptionV3 memerlukan 299x299, sedangkan ResNet-50 dan MobileNetV3 memakai
224x224.

``num_classes`` diturunkan dari ``configs.base_config.NUM_CLASSES`` supaya
jumlah kelas hanya ditulis di satu tempat; tidak ada angka kelas yang
di-hardcode di modul mana pun.
"""

from __future__ import annotations

from configs.base_config import NUM_CLASSES

#: Konfigurasi hyperparameter per arsitektur CNN.
CNN_CONFIG: dict[str, dict[str, object]] = {
    "resnet50": {
        "input_size": 224,
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "cross_entropy",
        "num_classes": NUM_CLASSES,
    },
    "mobilenetv3": {
        "input_size": 224,
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "cross_entropy",
        "num_classes": NUM_CLASSES,
    },
    "inceptionv3": {
        "input_size": 299,
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "cross_entropy",
        "num_classes": NUM_CLASSES,
    },
}

# Normalisasi ImageNet standar (dipakai semua model karena transfer learning).
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

#: Nama atribut layer konvolusi terakhir per arsitektur, dipakai sebagai
#: *target layer* Grad-CAM. Ditulis sebagai path atribut bertitik dan
#: diselesaikan saat runtime oleh ``src.analysis.gradcam.resolve_target_layer``
#: supaya config tetap bebas dari dependensi torch.
TARGET_LAYERS: dict[str, str] = {
    "resnet50": "layer4.-1",  # blok residual terakhir
    "mobilenetv3": "features.-1",  # blok konvolusi terakhir
    "inceptionv3": "Mixed_7c",  # modul inception terakhir
}

#: Nama atribut head klasifikasi per arsitektur. Dipakai untuk mengekstraksi
#: fitur *penultimate* (masukan head) pada analisis t-SNE/UMAP.
CLASSIFIER_ATTR: dict[str, str] = {
    "resnet50": "fc",
    "mobilenetv3": "classifier",
    "inceptionv3": "fc",
}

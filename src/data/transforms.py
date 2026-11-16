"""Transform pra-pemrosesan citra per arsitektur CNN.

**Penting — perubahan dari versi sebelumnya.** Augmentasi sekarang dilakukan
*offline* (lihat :mod:`src.data.offline_augmentation`), jadi pipeline transform
di sini hanya melakukan resize + ToTensor + normalisasi ImageNet, untuk
**semua** split termasuk ``train``. Mengaugmentasi lagi di sini akan menambah
keacakan di atas dataset yang sudah dibekukan ke disk dan merusak
reproducibility yang jadi alasan augmentasi offline dipilih.

Augmentasi on-the-fly masih bisa dihidupkan lewat ``augment=True`` semata-mata
untuk eksperimen pembanding (offline vs on-the-fly); jangan dipakai pada run
yang hasilnya masuk laporan tanpa menyebutkannya eksplisit.
"""

from __future__ import annotations

from torchvision import transforms

from configs.cnn_config import CNN_CONFIG, IMAGENET_MEAN, IMAGENET_STD

#: Split yang dianggap sebagai pelatihan (satu-satunya yang boleh diaugmentasi).
_TRAIN_SPLITS: frozenset[str] = frozenset({"train"})


def get_transforms(
    model_name: str,
    split: str,
    augment: bool = False,
) -> transforms.Compose:
    """Bangun pipeline transform untuk sebuah model dan split.

    Args:
        model_name: Nama model (``"resnet50"``, ``"mobilenetv3"``,
            ``"inceptionv3"``); menentukan ``input_size``.
        split: Salah satu dari ``"train"``, ``"valid"``, ``"test"``.
        augment: Bila ``True`` **dan** ``split`` adalah ``"train"``, tambahkan
            augmentasi on-the-fly (flip + rotasi + jitter warna). Default
            ``False`` karena augmentasi proyek ini dilakukan offline.

    Returns:
        ``torchvision.transforms.Compose`` siap pakai.

    Raises:
        KeyError: Bila ``model_name`` tidak ada di ``CNN_CONFIG``.

    Example:
        >>> tf = get_transforms("resnet50", "train")
        >>> isinstance(tf, transforms.Compose)
        True
        >>> # Default: train dan test memakai pipeline yang sama (deterministik).
        >>> len(tf.transforms) == len(get_transforms("resnet50", "test").transforms)
        True
    """
    if model_name not in CNN_CONFIG:
        raise KeyError(f"Model '{model_name}' tidak dikenal. Pilihan: {list(CNN_CONFIG)}")

    input_size: int = int(CNN_CONFIG[model_name]["input_size"])
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    resize = transforms.Resize((input_size, input_size))

    if augment and split in _TRAIN_SPLITS:
        return transforms.Compose(
            [
                resize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose([resize, transforms.ToTensor(), normalize])


def get_denormalize_transform() -> transforms.Normalize:
    """Kembalikan transform pembalik normalisasi ImageNet.

    Dipakai modul analisis untuk menampilkan kembali tensor sebagai citra yang
    bisa dilihat manusia (mis. overlay Grad-CAM, galeri error).

    Returns:
        ``transforms.Normalize`` yang membatalkan normalisasi ImageNet.
    """
    inv_std = [1.0 / s for s in IMAGENET_STD]
    inv_mean = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Normalize(mean=inv_mean, std=inv_std)

"""Cache crop ter-resize (uint8) di disk agar pelatihan multi-seed cepat.

Transform proyek ini deterministik (resize -> ToTensor -> Normalize, tanpa
augmentasi on-the-fly), sehingga hasil ``Resize`` cukup dihitung **sekali**.
Modul ini menyimpan citra ter-resize sebagai array ``uint8`` ``(N, H, W, 3)``
dalam berkas ``.npy`` yang dibaca lewat memmap: tidak membebani RAM dan tidak
perlu men-decode JPEG ulang di tiap epoch.

Kesetaraan numerik: ``to_tensor`` pada array ``uint8`` HWC dan pada citra PIL
melakukan operasi yang sama (permute lalu ``float / 255``), lalu ``normalize``
yang sama. Tensor keluaran karenanya identik bit-per-bit dengan
``GreenBeanDataset(transform=get_transforms(...))``; diverifikasi di
``tests/test_cache.py``.

Kunci cache = hash (input_size, path relatif, ukuran, mtime) seluruh berkas,
sehingga cache otomatis dibangun ulang bila isi folder berubah.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from configs.base_config import DATA_DIR
from configs.cnn_config import IMAGENET_MEAN, IMAGENET_STD
from src.data.dataset import GreenBeanDataset

logger = logging.getLogger(__name__)

#: Direktori default berkas cache (di-.gitignore).
CACHE_DIR: Path = DATA_DIR / "cache"


def _cache_key(dataset: GreenBeanDataset, input_size: int) -> str:
    """Hitung kunci cache dari daftar berkas + ukuran input."""
    digest = hashlib.sha1(f"v1|{input_size}".encode("utf-8"))
    for path, label in dataset.samples:
        stat = path.stat()
        rel = path.relative_to(dataset.root).as_posix()
        digest.update(f"|{rel}|{label}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()[:16]


def build_resized_cache(
    dataset: GreenBeanDataset,
    input_size: int,
    cache_dir: str | Path = CACHE_DIR,
) -> Path:
    """Bangun (atau pakai ulang) cache citra ter-resize untuk sebuah dataset.

    Args:
        dataset: Dataset sumber (tanpa transform).
        input_size: Sisi citra keluaran (224 atau 299).
        cache_dir: Folder penyimpanan cache.

    Returns:
        Path berkas ``.npy`` berbentuk ``(N, input_size, input_size, 3)`` uint8.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(dataset, input_size)
    stem = f"{dataset.root.parent.name}_{dataset.root.name}_{input_size}_{key}"
    path = cache_dir / f"{stem}.npy"
    meta_path = cache_dir / f"{stem}.json"
    if path.exists() and meta_path.exists():
        return path

    resize = transforms.Resize((input_size, input_size))
    tmp_path = cache_dir / f"{stem}.tmp.npy"
    array = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.uint8, shape=(len(dataset), input_size, input_size, 3)
    )
    desc = f"cache {dataset.root.parent.name}/{dataset.root.name}@{input_size}"
    for idx, (image_path, _) in enumerate(tqdm(dataset.samples, desc=desc, ascii=True, leave=False)):
        with Image.open(image_path) as image:
            array[idx] = np.asarray(resize(image.convert("RGB")), dtype=np.uint8)
    array.flush()
    del array
    tmp_path.replace(path)
    meta = {
        "root": str(dataset.root),
        "input_size": input_size,
        "n": len(dataset),
        "paths": [p.relative_to(dataset.root).as_posix() for p in dataset.paths],
        "labels": dataset.labels,
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    logger.info("Cache dibangun: %s (%d citra)", path.name, len(dataset))
    return path


class CachedCropDataset(GreenBeanDataset):
    """``GreenBeanDataset`` yang membaca citra ter-resize dari cache memmap.

    Atribut ``samples``, ``labels``, ``paths``, ``class_counts()``, dan
    ``augmentation_counts()`` sama persis dengan ``GreenBeanDataset``, sehingga
    dapat dipakai langsung oleh ``train_cnn``, bobot kelas, dan sampler.

    Args:
        root: Folder split (berisi subfolder kelas).
        input_size: Ukuran input model.
        cache_dir: Folder cache.
        class_names: Urutan nama kelas.
    """

    def __init__(
        self,
        root: str | Path,
        input_size: int,
        cache_dir: str | Path = CACHE_DIR,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__(root=root, transform=None, class_names=class_names)
        self.input_size = int(input_size)
        self.cache_path = build_resized_cache(self, self.input_size, cache_dir)
        self._array = np.load(self.cache_path, mmap_mode="r")
        if len(self._array) != len(self.samples):
            raise RuntimeError(f"Ukuran cache {self.cache_path} tidak cocok dengan dataset.")

    def __getitem__(self, index: int):
        """Kembalikan ``(tensor_ternormalisasi, label)``."""
        image = np.array(self._array[index], copy=True)
        tensor = TF.normalize(TF.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return tensor, self.samples[index][1]

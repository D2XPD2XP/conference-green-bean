"""Custom Dataset PyTorch untuk crop biji kopi per kelas.

Struktur folder yang diharapkan (hasil ``scripts.crop_dataset`` lalu
``scripts.augment_dataset``)::

    <root>/
    |-- good/
    |   |-- img001_bean_000_orig.jpg
    |   `-- ...
    |-- black/
    |-- chipped/
    |-- dry_pods/
    `-- insect_damage/

Indeks label mengikuti urutan ``class_names`` (default dari
``configs.base_config.CLASS_NAMES``).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from configs.base_config import CLASS_NAMES

#: Ekstensi gambar yang dipindai sebagai sampel.
_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

#: Penanda nama file varian hasil augmentasi offline (``..._aug01.jpg``).
_AUGMENTED_MARKER: str = "_aug"


class GreenBeanDataset(Dataset):
    """Dataset citra crop biji kopi untuk klasifikasi kualitas 5 kelas.

    Args:
        root: Direktori berisi subfolder per kelas.
        transform: Transform torchvision yang diterapkan pada tiap gambar PIL.
            Jika ``None``, gambar dikembalikan sebagai objek PIL apa adanya.
        class_names: Urutan nama kelas -> indeks label. Default ``CLASS_NAMES``.

    Attributes:
        samples: Daftar pasangan ``(path_gambar, label)``.
        class_to_idx: Pemetaan nama kelas ke indeks integer.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.class_names = list(class_names) if class_names is not None else list(CLASS_NAMES)
        self.class_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }
        self.samples: list[tuple[Path, int]] = self._gather_samples()

    def _gather_samples(self) -> list[tuple[Path, int]]:
        """Kumpulkan seluruh pasangan (path, label) dari subfolder kelas."""
        samples: list[tuple[Path, int]] = []
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in _IMAGE_EXTENSIONS:
                    samples.append((path, label))
        return samples

    @property
    def labels(self) -> list[int]:
        """Daftar label seluruh sampel, urut sesuai ``samples``.

        Dipakai modul penanganan imbalance (bobot kelas & weighted sampler)
        tanpa perlu memuat citranya.
        """
        return [label for _, label in self.samples]

    #: Alias konvensi torchvision (``ImageFolder.targets``).
    @property
    def targets(self) -> list[int]:
        """Alias :attr:`labels`, mengikuti konvensi ``ImageFolder``."""
        return self.labels

    @property
    def paths(self) -> list[Path]:
        """Daftar path citra seluruh sampel, urut sesuai ``samples``."""
        return [path for path, _ in self.samples]

    def class_counts(self) -> dict[str, int]:
        """Hitung jumlah sampel per kelas.

        Dipakai untuk melaporkan ketimpangan kelas dan menurunkan bobot loss
        (lihat :func:`src.training.imbalance.compute_class_weights`).

        Returns:
            Dict ``nama_kelas -> jumlah sampel``, memuat seluruh kelas termasuk
            yang kosong.
        """
        counter = Counter(label for _, label in self.samples)
        return {name: counter.get(idx, 0) for name, idx in self.class_to_idx.items()}

    def augmentation_counts(self) -> dict[str, int]:
        """Hitung sampel asli vs hasil augmentasi offline.

        Berkas varian augmentasi diberi sufiks ``_aug<NN>`` oleh
        :mod:`src.data.offline_augmentation`; sisanya dianggap asli. Berguna
        untuk memverifikasi bahwa ``valid``/``test`` benar-benar bersih dari
        citra buatan.

        Returns:
            Dict ``{"original": int, "augmented": int}``.
        """
        augmented = sum(1 for path, _ in self.samples if _AUGMENTED_MARKER in path.stem)
        return {"original": len(self.samples) - augmented, "augmented": augmented}

    def __len__(self) -> int:
        """Jumlah sampel dalam dataset."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """Ambil sampel ke-``index``.

        Returns:
            Tuple ``(image, label)``. ``image`` berupa tensor bila ``transform``
            diberikan, atau objek PIL bila tidak.
        """
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

"""Strategi penanganan ketimpangan kelas (class imbalance).

Dataset green bean ini timpang secara struktural: kelas ``good`` jauh lebih
banyak daripada tiap jenis cacat, sementara keempat kelas cacat relatif
seimbang antar-sesamanya. Modul ini menyediakan empat strategi yang bisa
dipertukarkan lewat satu argumen CLI, supaya perbandingannya bisa dilaporkan
sebagai eksperimen tersendiri:

============================  ==========================================
``"none"``                    ``CrossEntropyLoss`` biasa (baseline).
``"class_weights"``           ``CrossEntropyLoss`` berbobot per kelas.
``"weighted_sampler"``        ``WeightedRandomSampler`` di DataLoader.
``"focal_loss"``              Focal Loss (Lin et al., ICCV 2017).
============================  ==========================================

**Hati-hati menumpuk dua koreksi.** Augmentasi offline sudah meng-*oversample*
kelas minoritas di disk (lihat :mod:`src.data.offline_augmentation`). Bobot
kelas di sini dihitung dari distribusi dataset yang **benar-benar dimuat**
(yaitu pasca-augmentasi), jadi bila oversampling sudah menyeimbangkan
distribusi, bobotnya otomatis mendekati 1.0 dan tidak terjadi koreksi ganda.
Jangan menghitung bobot dari distribusi pra-augmentasi.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

from configs.base_config import FOCAL_LOSS_GAMMA, NUM_CLASSES, RANDOM_SEED

logger = logging.getLogger(__name__)

#: Skema perhitungan bobot kelas yang didukung.
WEIGHT_STRATEGIES: tuple[str, ...] = (
    "balanced",
    "inverse_frequency",
    "effective_num_samples",
)

#: Beta default untuk skema ``effective_num_samples`` (Cui et al., CVPR 2019).
EFFECTIVE_NUM_BETA: float = 0.9999


def class_counts_from_labels(
    labels: Sequence[int],
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Hitung jumlah sampel per kelas dari daftar label.

    Args:
        labels: Daftar indeks kelas (0..num_classes-1).
        num_classes: Jumlah kelas.

    Returns:
        Array ``int`` berukuran ``(num_classes,)``.

    Example:
        >>> class_counts_from_labels([0, 0, 1, 4], num_classes=5).tolist()
        [2, 1, 0, 0, 1]
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for label in labels:
        idx = int(label)
        if 0 <= idx < num_classes:
            counts[idx] += 1
    return counts


def compute_class_weights(
    labels: Sequence[int],
    num_classes: int = NUM_CLASSES,
    strategy: str = "balanced",
) -> torch.Tensor:
    """Hitung bobot per kelas untuk ``CrossEntropyLoss``.

    Args:
        labels: Daftar indeks kelas dari split **train** (pasca-augmentasi).
        num_classes: Jumlah kelas.
        strategy: Skema perhitungan:

            - ``"balanced"``: ``N / (K * n_c)`` — identik dengan
              ``sklearn.utils.class_weight.compute_class_weight("balanced")``.
            - ``"inverse_frequency"``: ``1 / n_c``, dinormalkan agar rata-rata
              bobot kelas yang hadir = 1.
            - ``"effective_num_samples"``: ``(1 - beta) / (1 - beta**n_c)``
              (Cui et al., CVPR 2019), dinormalkan sama.

    Returns:
        Tensor ``float32`` berukuran ``(num_classes,)``. Kelas tanpa sampel
        diberi bobot ``0.0`` supaya tidak pernah ikut menyumbang loss.

    Raises:
        ValueError: Bila ``strategy`` tidak dikenal atau ``labels`` kosong.

    Example:
        >>> w = compute_class_weights([0, 0, 0, 1], num_classes=2)
        >>> [round(v, 4) for v in w.tolist()]
        [0.6667, 2.0]
    """
    if strategy not in WEIGHT_STRATEGIES:
        raise ValueError(
            f"strategy '{strategy}' tidak dikenal. Pilihan: {list(WEIGHT_STRATEGIES)}"
        )

    counts = class_counts_from_labels(labels, num_classes=num_classes)
    total = int(counts.sum())
    if total == 0:
        raise ValueError("Daftar label kosong; bobot kelas tidak dapat dihitung.")

    present = np.flatnonzero(counts > 0)
    weights = np.zeros(num_classes, dtype=np.float64)

    if strategy == "balanced":
        # sklearn hanya menerima kelas yang benar-benar hadir di y.
        y = np.asarray([int(label) for label in labels], dtype=np.int64)
        computed = compute_class_weight("balanced", classes=present, y=y)
        weights[present] = computed
    elif strategy == "inverse_frequency":
        weights[present] = 1.0 / counts[present]
        weights[present] /= weights[present].mean()
    else:  # effective_num_samples
        effective = (1.0 - EFFECTIVE_NUM_BETA ** counts[present]) / (1.0 - EFFECTIVE_NUM_BETA)
        weights[present] = 1.0 / effective
        weights[present] /= weights[present].mean()

    missing = [int(i) for i in np.flatnonzero(counts == 0)]
    if missing:
        logger.warning(
            "Kelas tanpa sampel di split train: %s. Bobotnya diset 0.0.", missing
        )

    return torch.as_tensor(weights, dtype=torch.float32)


def create_weighted_sampler(
    labels: Sequence[int],
    num_samples: int | None = None,
    num_classes: int = NUM_CLASSES,
    seed: int = RANDOM_SEED,
) -> WeightedRandomSampler:
    """Buat ``WeightedRandomSampler`` agar tiap batch mendekati seimbang.

    Setiap sampel diberi bobot ``1 / n_c`` sesuai kelasnya, sehingga peluang
    terambilnya tiap kelas kira-kira sama. Pengambilan dilakukan **dengan
    pengembalian**: sampel kelas minoritas akan muncul berulang dalam satu
    epoch, sedangkan sebagian sampel kelas mayoritas bisa tidak terpakai.

    Args:
        labels: Daftar indeks kelas untuk tiap sampel, urut sesuai dataset.
        num_samples: Jumlah sampel yang ditarik per epoch. Default
            ``len(labels)``.
        num_classes: Jumlah kelas.
        seed: Seed generator agar urutan sampling reproducible.

    Returns:
        ``WeightedRandomSampler`` siap diberikan ke ``DataLoader(sampler=...)``.

    Raises:
        ValueError: Bila ``labels`` kosong.

    Note:
        ``DataLoader`` tidak boleh memakai ``shuffle=True`` bersamaan dengan
        ``sampler``; PyTorch akan melempar error. Pemilihan sampler sudah
        mengacak urutan.
    """
    if len(labels) == 0:
        raise ValueError("Daftar label kosong; sampler tidak dapat dibuat.")

    counts = class_counts_from_labels(labels, num_classes=num_classes)
    per_class_weight = np.zeros(num_classes, dtype=np.float64)
    present = counts > 0
    per_class_weight[present] = 1.0 / counts[present]

    sample_weights = np.asarray(
        [per_class_weight[int(label)] for label in labels], dtype=np.float64
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(num_samples) if num_samples is not None else len(labels),
        replacement=True,
        generator=generator,
    )


class FocalLoss(nn.Module):
    """Focal Loss untuk menangani ketimpangan kelas dan *hard example*.

    Mengecilkan kontribusi sampel yang sudah diklasifikasikan dengan yakin
    (``pt`` tinggi) lewat faktor ``(1 - pt) ** gamma``, sehingga gradien
    didominasi sampel yang masih sulit. ``gamma = 0`` membuatnya ekuivalen
    dengan ``CrossEntropyLoss``.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha: Bobot per kelas berukuran ``(num_classes,)``, diteruskan sebagai
            argumen ``weight`` ke ``cross_entropy``. ``None`` berarti tanpa
            bobot kelas.
        gamma: Eksponen *focusing*. Default ``FOCAL_LOSS_GAMMA``.
        reduction: ``"mean"``, ``"sum"``, atau ``"none"``.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = FOCAL_LOSS_GAMMA,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction harus 'mean', 'sum', atau 'none'")
        if gamma < 0:
            raise ValueError("gamma tidak boleh negatif")

        self.gamma = float(gamma)
        self.reduction = reduction
        # register_buffer: alpha ikut berpindah device bersama modulnya.
        if alpha is None:
            self.alpha: torch.Tensor | None = None
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Hitung focal loss.

        Args:
            inputs: Logit keluaran model, bentuk ``(N, num_classes)``.
            targets: Label sebenarnya, bentuk ``(N,)``.

        Returns:
            Skalar loss (atau tensor ``(N,)`` bila ``reduction="none"``).
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        # pt = exp(-CE) adalah probabilitas kelas benar (sudah memuat softmax).
        pt = torch.exp(-ce_loss)
        focal = (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def build_criterion(
    strategy: str,
    class_weights: torch.Tensor | None = None,
    gamma: float = FOCAL_LOSS_GAMMA,
) -> nn.Module:
    """Pilih fungsi loss sesuai strategi penanganan imbalance.

    Args:
        strategy: ``"none"``, ``"class_weights"``, ``"weighted_sampler"``, atau
            ``"focal_loss"``.
        class_weights: Bobot per kelas; dipakai oleh ``"class_weights"`` dan
            ``"focal_loss"``. Diabaikan strategi lain.
        gamma: Parameter gamma untuk Focal Loss.

    Returns:
        Modul loss siap pakai.

    Raises:
        ValueError: Bila ``strategy`` tidak dikenal.

    Note:
        ``"weighted_sampler"`` menyeimbangkan di sisi *sampling*, jadi loss-nya
        sengaja dibiarkan tanpa bobot — memakai keduanya sekaligus berarti
        mengoreksi ketimpangan dua kali.
    """
    if strategy == "none":
        return nn.CrossEntropyLoss()
    if strategy == "class_weights":
        return nn.CrossEntropyLoss(weight=class_weights)
    if strategy == "weighted_sampler":
        return nn.CrossEntropyLoss()
    if strategy == "focal_loss":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    raise ValueError(
        f"Strategi imbalance '{strategy}' tidak dikenal. "
        "Pilihan: none, class_weights, weighted_sampler, focal_loss"
    )


def describe_imbalance(
    counts: dict[str, int],
) -> dict[str, float | str]:
    """Ringkas tingkat ketimpangan sebuah distribusi kelas.

    Args:
        counts: Dict ``nama_kelas -> jumlah sampel``.

    Returns:
        Dict berisi ``total``, ``majority_class``, ``minority_class``,
        ``imbalance_ratio`` (mayoritas/minoritas), dan ``majority_share``
        (proporsi kelas terbanyak; ini garis dasar akurasi tebakan mayoritas).

    Example:
        >>> info = describe_imbalance({"good": 80, "black": 20})
        >>> info["imbalance_ratio"], info["majority_class"]
        (4.0, 'good')
    """
    non_empty = {k: v for k, v in counts.items() if v > 0}
    total = sum(counts.values())
    if not non_empty:
        return {
            "total": 0,
            "majority_class": "",
            "minority_class": "",
            "imbalance_ratio": 0.0,
            "majority_share": 0.0,
        }

    majority = max(non_empty, key=lambda k: non_empty[k])
    minority = min(non_empty, key=lambda k: non_empty[k])
    return {
        "total": total,
        "majority_class": majority,
        "minority_class": minority,
        "imbalance_ratio": round(non_empty[majority] / non_empty[minority], 4),
        "majority_share": round(non_empty[majority] / total, 4) if total else 0.0,
    }

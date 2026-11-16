"""Gaya visual bersama untuk seluruh figur analisis.

Satu tempat untuk warna kelas, ukuran font, dan penyimpanan figur, supaya
semua gambar yang masuk laporan/paper terlihat berasal dari satu sistem:
``good`` selalu hijau, ``black`` selalu gelap, dan seterusnya — pembaca tidak
perlu menghafal legenda yang berubah-ubah antar-gambar.
"""

from __future__ import annotations

from pathlib import Path

# Backend matplotlib sengaja TIDAK diset di sini. Modul library yang memanggil
# ``matplotlib.use("Agg")`` saat diimpor akan menimpa backend inline Jupyter,
# sehingga ``plt.show()`` di notebook diam-diam tidak menampilkan apa pun.
# Pemilihan backend non-interaktif dilakukan di entry point CLI (``scripts/``)
# dan ``tests/conftest.py``.
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.base_config import CLASS_COLORS, CLASS_NAMES, CLASS_NAMES_ID
from configs.cnn_config import IMAGENET_MEAN, IMAGENET_STD

#: Warna cadangan untuk kelas yang tidak ada di ``CLASS_COLORS``.
_FALLBACK_COLOR: str = "#7f8c8d"

#: DPI default figur (cukup tajam untuk cetak, tidak berlebihan untuk repo).
FIGURE_DPI: int = 150


def apply_plot_style() -> None:
    """Terapkan rcParams matplotlib yang konsisten untuk semua figur."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.titlesize": 13,
            "figure.titleweight": "bold",
            "savefig.bbox": "tight",
            "savefig.dpi": FIGURE_DPI,
        }
    )


def class_palette(class_names: list[str] | None = None) -> list[str]:
    """Kembalikan daftar warna hex sesuai urutan kelas.

    Args:
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.

    Returns:
        Daftar warna hex, satu per kelas.

    Example:
        >>> class_palette(["good"])
        ['#2ecc71']
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    return [CLASS_COLORS.get(name, _FALLBACK_COLOR) for name in names]


def class_labels_id(class_names: list[str] | None = None) -> list[str]:
    """Kembalikan nama kelas dalam bahasa Indonesia untuk label figur.

    Args:
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.

    Returns:
        Daftar nama Indonesia; kelas tanpa padanan memakai namanya sendiri.
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    return [CLASS_NAMES_ID.get(name, name) for name in names]


def save_figure(fig, output_path: str | Path | None, dpi: int = FIGURE_DPI) -> Path | None:
    """Simpan figur ke berkas bila ``output_path`` diberikan.

    Args:
        fig: Objek ``matplotlib.figure.Figure``.
        output_path: Path tujuan (PNG). ``None`` berarti figur tidak disimpan.
        dpi: Resolusi keluaran.

    Returns:
        Path berkas tersimpan, atau ``None`` bila tidak disimpan.
    """
    if output_path is None:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Ubah tensor citra ternormalisasi ImageNet menjadi array siap tampil.

    Args:
        tensor: Tensor ``(3, H, W)`` hasil transform (sudah dinormalisasi).

    Returns:
        Array ``(H, W, 3)`` ``float32`` dalam rentang 0..1.

    Example:
        >>> img = tensor_to_image(torch.zeros(3, 8, 8))
        >>> img.shape, float(img.min()) >= 0.0
        ((8, 8, 3), True)
    """
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
    return np.clip(array, 0.0, 1.0).astype(np.float32)

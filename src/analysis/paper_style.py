"""Gaya figur paper (IEEE satu kolom), identik dengan ``notebooks/11_paper_figures``.

Dipisah ke modul agar figur revisi (``scripts/revision_figures.py``) memakai
font, palet, dan ukuran yang sama persis dengan draft paper sebelumnya.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

MODELS: list[str] = ["resnet50", "mobilenetv3", "inceptionv3"]
MODEL_LABELS: dict[str, str] = {
    "resnet50": "ResNet-50",
    "mobilenetv3": "MobileNetV3-L",
    "inceptionv3": "InceptionV3",
}
MODEL_COLORS: dict[str, str] = {
    "resnet50": "#2a78d6",
    "mobilenetv3": "#eb6834",
    "inceptionv3": "#1baf7a",
}
CLASS_LABELS: dict[str, str] = {
    "good": "good",
    "black": "black",
    "chipped": "chipped",
    "dry_pods": "dry pods",
    "insect_damage": "insect damage",
}
CLASS_ABBR: dict[str, str] = {
    "good": "G", "black": "B", "chipped": "C", "dry_pods": "D", "insect_damage": "I",
}
CLASS_MARKERS: dict[str, str] = {
    "good": "o", "black": "s", "chipped": "^", "dry_pods": "D", "insect_damage": "v",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
NEUTRAL = "#b5b3ad"
HIGHLIGHT = "#e34948"
TTA_COLORS: dict[str, str] = {"none": NEUTRAL, "d4": "#6250d6"}
TTA_LABELS: dict[str, str] = {"none": "Without TTA", "d4": "$D_4$ TTA"}
CONF_COLORS: dict[str, str] = {"correct": "#2a78d6", "incorrect": "#eb6834"}

#: Lebar satu kolom IEEE (inci).
COL_W: float = 3.5
PNG_DPI: int = 300


def paper_style(base_size: float = 8.0) -> None:
    """rcParams ala IEEE: serif, kecil, grid samar, font tertanam di PDF."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_size,
        "axes.titlesize": base_size,
        "axes.titleweight": "normal",
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_paper(fig, out_dir: str | Path, stem: str, formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """Simpan figur ke ``out_dir`` dalam beberapa format sekaligus."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=PNG_DPI)
        paths.append(path)
    plt.close(fig)
    return paths


def panel_label(ax, text: str, x: float = -0.02, y: float = 1.02) -> None:
    """Label panel (a), (b) di pojok kiri atas sumbu."""
    ax.text(x, y, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=plt.rcParams["font.size"], fontweight="bold")

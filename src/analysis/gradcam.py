"""Grad-CAM: visualisasi wilayah citra yang dipakai model untuk memutuskan.

Dependensi: ``pip install grad-cam`` (paket ``pytorch-grad-cam``).

**Catatan teknis penting.** Seluruh backbone di proyek ini dibekukan
(``requires_grad = False``) karena skemanya *feature extraction*. Grad-CAM
butuh gradien aktivasi di layer konvolusi terakhir, dan pada model yang
seluruh parameternya beku graf autograd tidak terbentuk sehingga
``loss.backward()`` gagal. Modul ini menyelesaikannya dengan
:func:`enable_gradients` — ``requires_grad`` dinyalakan sementara selama
perhitungan CAM lalu dikembalikan ke kondisi semula. Tidak ada optimizer yang
dijalankan, jadi bobot model tidak pernah berubah.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from configs.base_config import CLASS_NAMES, DEVICE, RANDOM_SEED
from configs.cnn_config import TARGET_LAYERS
from src.analysis.style import apply_plot_style, save_figure, tensor_to_image
from src.data.transforms import get_transforms

import matplotlib.pyplot as plt  # noqa: E402  (setelah style.py menetapkan backend Agg)

logger = logging.getLogger(__name__)

#: Ekstensi citra yang dipindai saat memilih sampel.
_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def resolve_target_layer(model: nn.Module, spec: str) -> nn.Module:
    """Selesaikan path atribut bertitik menjadi modul layer yang sebenarnya.

    Mendukung indeks integer untuk modul berurutan, termasuk indeks negatif:
    ``"layer4.-1"`` berarti blok terakhir dari ``model.layer4``.

    Args:
        model: Model yang diperiksa.
        spec: Path atribut, mis. ``"layer4.-1"``, ``"features.-1"``,
            ``"Mixed_7c"``.

    Returns:
        Modul target.

    Raises:
        AttributeError: Bila salah satu komponen path tidak ditemukan.

    Example:
        >>> from src.models import get_model
        >>> layer = resolve_target_layer(get_model("resnet50", pretrained=False), "layer4.-1")
        >>> isinstance(layer, nn.Module)
        True
    """
    current: nn.Module = model
    for part in spec.split("."):
        if part.lstrip("-").isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            if not hasattr(current, part):
                raise AttributeError(
                    f"Layer '{part}' tidak ada pada {type(current).__name__} "
                    f"(spec lengkap: '{spec}')"
                )
            current = getattr(current, part)
    return current


@contextlib.contextmanager
def enable_gradients(model: nn.Module) -> Iterator[nn.Module]:
    """Nyalakan ``requires_grad`` sementara, lalu kembalikan kondisi awal.

    Diperlukan karena backbone beku membuat graf autograd tidak terbentuk,
    sehingga Grad-CAM tidak bisa mengambil gradien aktivasi.

    Args:
        model: Model yang dipakai Grad-CAM.

    Yields:
        Model yang sama, dengan seluruh parameter sementara ``requires_grad=True``.
    """
    original = [param.requires_grad for param in model.parameters()]
    try:
        for param in model.parameters():
            param.requires_grad_(True)
        yield model
    finally:
        for param, flag in zip(model.parameters(), original):
            param.requires_grad_(flag)
        model.zero_grad(set_to_none=True)


class GradCAMAnalyzer:
    """Pembungkus Grad-CAM untuk satu arsitektur CNN.

    Args:
        model: Model terlatih (bobot sudah dimuat dari checkpoint).
        model_name: Nama arsitektur (``"resnet50"``, ``"mobilenetv3"``,
            ``"inceptionv3"``); menentukan target layer dan ukuran input.
        device: Device komputasi.
        target_layer: Modul target custom. Bila ``None``, diambil dari
            ``TARGET_LAYERS[model_name]``.

    Example:
        >>> analyzer = GradCAMAnalyzer(model, "resnet50")        # doctest: +SKIP
        >>> result = analyzer.explain("data/crops/test/good/x.jpg")  # doctest: +SKIP
        >>> result["pred_label"]                                  # doctest: +SKIP
        'good'
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: str = DEVICE,
        target_layer: nn.Module | None = None,
    ) -> None:
        if model_name not in TARGET_LAYERS and target_layer is None:
            raise KeyError(
                f"Target layer untuk '{model_name}' tidak diketahui. "
                f"Tersedia: {list(TARGET_LAYERS)}"
            )

        self.model_name = model_name
        self.device = device
        self.model = model.to(device).eval()
        self.target_layer = target_layer or resolve_target_layer(
            self.model, TARGET_LAYERS[model_name]
        )
        self.transform = get_transforms(model_name, "test")
        self.cam = GradCAM(model=self.model, target_layers=[self.target_layer])

    def explain(
        self,
        image: str | Path | Image.Image,
        target_class: int | None = None,
        class_names: list[str] | None = None,
    ) -> dict:
        """Hasilkan heatmap Grad-CAM untuk satu citra.

        Args:
            image: Path citra atau objek PIL.
            target_class: Kelas yang dijelaskan. ``None`` berarti kelas dengan
                probabilitas tertinggi (yang benar-benar diprediksi model).
            class_names: Urutan nama kelas. Default ``CLASS_NAMES``.

        Returns:
            Dict berisi:

            - ``rgb``        : citra input ``(H, W, 3)`` float 0..1
            - ``cam``        : heatmap ``(H, W)`` float 0..1
            - ``overlay``    : citra + heatmap ``(H, W, 3)`` float 0..1
            - ``pred_index`` : indeks kelas prediksi
            - ``pred_label`` : nama kelas prediksi
            - ``confidence`` : probabilitas kelas prediksi
            - ``target_index``: kelas yang dijelaskan heatmap ini
        """
        names = list(class_names) if class_names is not None else list(CLASS_NAMES)
        pil_image = Image.open(image).convert("RGB") if not isinstance(image, Image.Image) else image

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        pred_index = int(probs.argmax().item())
        confidence = float(probs[pred_index].item())

        target_index = pred_index if target_class is None else int(target_class)
        targets = [ClassifierOutputTarget(target_index)]

        with enable_gradients(self.model):
            # requires_grad pada input menjamin graf terbentuk sampai ke
            # aktivasi target layer, bahkan bila pemanggil tidak menyentuh
            # flag parameter model.
            grayscale = self.cam(
                input_tensor=tensor.clone().requires_grad_(True), targets=targets
            )[0]

        # Citra input dibaca ulang dari tensor supaya ukurannya cocok dengan
        # heatmap (224 atau 299, tergantung arsitektur).
        rgb = tensor_to_image(tensor[0])
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

        return {
            "rgb": rgb,
            "cam": grayscale,
            "overlay": overlay.astype(np.float32) / 255.0,
            "pred_index": pred_index,
            "pred_label": names[pred_index] if pred_index < len(names) else str(pred_index),
            "confidence": confidence,
            "target_index": target_index,
        }

    def visualize_batch(
        self,
        image_paths: Sequence[str | Path],
        true_labels: Sequence[int] | None = None,
        class_names: list[str] | None = None,
        output_path: str | Path | None = None,
        title: str | None = None,
    ):
        """Buat grid ``original | heatmap | overlay`` untuk beberapa citra.

        Args:
            image_paths: Daftar path citra.
            true_labels: Label sebenarnya tiap citra (opsional, untuk judul).
            class_names: Urutan nama kelas. Default ``CLASS_NAMES``.
            output_path: Bila diberikan, simpan figur ke path ini.
            title: Judul figur. Default menyebut nama arsitektur.

        Returns:
            Objek ``matplotlib.figure.Figure``.
        """
        names = list(class_names) if class_names is not None else list(CLASS_NAMES)
        paths = [Path(p) for p in image_paths]
        if not paths:
            raise ValueError("image_paths kosong.")

        apply_plot_style()
        fig, axes = plt.subplots(len(paths), 3, figsize=(7.5, 2.6 * len(paths)))
        axes = np.atleast_2d(axes)

        for row, path in enumerate(paths):
            result = self.explain(path, class_names=names)
            panels = [
                (result["rgb"], "original", None),
                (result["cam"], "Grad-CAM", "jet"),
                (result["overlay"], "overlay", None),
            ]
            for col, (data, label, cmap) in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(data, cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if row == 0:
                    ax.set_title(label)

            true_text = ""
            if true_labels is not None and row < len(true_labels):
                idx = int(true_labels[row])
                true_text = f"true={names[idx] if idx < len(names) else idx}\n"
            axes[row, 0].set_ylabel(
                f"{true_text}pred={result['pred_label']}\n({result['confidence']:.2f})",
                fontsize=8,
                rotation=0,
                ha="right",
                va="center",
                labelpad=38,
            )

        fig.suptitle(title or f"Grad-CAM - {self.model_name}")
        fig.tight_layout()
        save_figure(fig, output_path)
        return fig


def compare_architectures(
    models: dict[str, nn.Module],
    image_paths: Sequence[str | Path],
    true_labels: Sequence[int] | None = None,
    class_names: list[str] | None = None,
    device: str = DEVICE,
    output_path: str | Path | None = None,
):
    """Bandingkan Grad-CAM ketiga arsitektur pada citra yang sama.

    Baris = sampel, kolom = citra asli + satu overlay per arsitektur. Tiap
    arsitektur memakai ukuran input dan target layer-nya sendiri (InceptionV3
    299, dua lainnya 224), jadi resolusi heatmap antar-kolom bisa berbeda —
    itu wajar dan memang bagian dari perbedaan arsitektur.

    Args:
        models: Dict ``nama_arsitektur -> model terlatih``.
        image_paths: Daftar path citra yang dianalisis.
        true_labels: Label sebenarnya tiap citra (opsional, untuk label baris).
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.
        device: Device komputasi.
        output_path: Bila diberikan, simpan figur ke path ini.

    Returns:
        Objek ``matplotlib.figure.Figure``.

    Raises:
        ValueError: Bila ``models`` atau ``image_paths`` kosong.
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    paths = [Path(p) for p in image_paths]
    if not models:
        raise ValueError("models kosong.")
    if not paths:
        raise ValueError("image_paths kosong.")

    analyzers = {
        name: GradCAMAnalyzer(model, name, device=device) for name, model in models.items()
    }

    apply_plot_style()
    n_cols = 1 + len(analyzers)
    fig, axes = plt.subplots(
        len(paths), n_cols, figsize=(2.3 * n_cols, 2.5 * len(paths))
    )
    axes = np.atleast_2d(axes)

    for row, path in enumerate(paths):
        results = {name: an.explain(path, class_names=names) for name, an in analyzers.items()}
        first = next(iter(results.values()))

        ax = axes[row, 0]
        ax.imshow(first["rgb"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if row == 0:
            ax.set_title("original")

        true_text = ""
        if true_labels is not None and row < len(true_labels):
            idx = int(true_labels[row])
            true_text = names[idx] if idx < len(names) else str(idx)
        ax.set_ylabel(true_text, fontsize=9, rotation=0, ha="right", va="center", labelpad=28)

        for col, (name, result) in enumerate(results.items(), start=1):
            ax = axes[row, col]
            ax.imshow(result["overlay"])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if row == 0:
                ax.set_title(name)
            correct = true_labels is not None and row < len(true_labels) and (
                result["pred_index"] == int(true_labels[row])
            )
            mark = "OK" if correct else "X"
            ax.set_xlabel(
                f"{mark} {result['pred_label']} ({result['confidence']:.2f})", fontsize=8
            )

    fig.suptitle("Perbandingan Grad-CAM antar-arsitektur")
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def select_correct_samples(
    models: dict[str, nn.Module],
    crops_root: str | Path,
    class_names: list[str] | None = None,
    n_per_class: int = 3,
    seed: int = RANDOM_SEED,
    device: str = DEVICE,
) -> tuple[list[Path], list[int]]:
    """Pilih sampel per kelas yang diprediksi BENAR oleh semua model.

    Berguna untuk figur perbandingan Grad-CAM: tiap baris menunjukkan bagian
    biji yang dipakai masing-masing arsitektur saat keputusannya benar,
    sehingga perbedaan heatmap mencerminkan arsitektur, bukan kesalahan.
    Kesalahan dianalisis terpisah (lihat galeri error).

    Prediksi yang dicek adalah prediksi **tunggal** (tanpa TTA) — sama dengan
    label yang tercetak di bawah heatmap, karena Grad-CAM menjelaskan satu
    citra.

    Args:
        models: ``nama_arsitektur -> model`` (bobot sudah dimuat).
        crops_root: Folder split (mis. ``data/crops/test``) berisi subfolder kelas.
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.
        n_per_class: Jumlah sampel per kelas.
        seed: Seed pengacakan urutan kandidat (reproducible).
        device: Device komputasi.

    Returns:
        Tuple ``(paths, labels)``. Bila sebuah kelas punya kurang dari
        ``n_per_class`` sampel yang benar di semua model, yang tersedia saja
        yang dikembalikan (dengan peringatan).
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    root = Path(crops_root)
    rng = np.random.default_rng(seed)
    transforms = {name: get_transforms(name, "test") for name in models}
    for model in models.values():
        model.to(device).eval()

    def correct_everywhere(path: Path, label: int) -> bool:
        image = Image.open(path).convert("RGB")
        with torch.no_grad():
            for name, model in models.items():
                tensor = transforms[name](image).unsqueeze(0).to(device)
                if int(model(tensor).argmax(dim=1).item()) != label:
                    return False
        return True

    paths: list[Path] = []
    labels: list[int] = []
    for index, name in enumerate(names):
        class_dir = root / name
        if not class_dir.is_dir():
            logger.warning("Folder kelas %s tidak ada; dilewati.", class_dir)
            continue
        candidates = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        found: list[Path] = []
        for pos in rng.permutation(len(candidates)):
            if correct_everywhere(candidates[int(pos)], index):
                found.append(candidates[int(pos)])
                if len(found) == n_per_class:
                    break
        if len(found) < n_per_class:
            logger.warning(
                "Kelas %s: hanya %d sampel yang benar di semua model.", name, len(found)
            )
        for path in sorted(found):
            paths.append(path)
            labels.append(index)

    return paths, labels


def select_sample_images(
    crops_root: str | Path,
    class_names: list[str] | None = None,
    n_per_class: int = 3,
    seed: int = RANDOM_SEED,
) -> tuple[list[Path], list[int]]:
    """Pilih sampel citra per kelas secara acak namun reproducible.

    Args:
        crops_root: Folder split (mis. ``data/crops/test``) berisi subfolder kelas.
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.
        n_per_class: Jumlah sampel per kelas.
        seed: Seed pemilihan acak.

    Returns:
        Tuple ``(paths, labels)`` dengan ``labels`` berupa indeks kelas.
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    root = Path(crops_root)
    rng = np.random.default_rng(seed)

    paths: list[Path] = []
    labels: list[int] = []

    for index, name in enumerate(names):
        class_dir = root / name
        if not class_dir.is_dir():
            logger.warning("Folder kelas %s tidak ada; dilewati.", class_dir)
            continue
        candidates = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not candidates:
            logger.warning("Tidak ada citra di %s; dilewati.", class_dir)
            continue

        take = min(n_per_class, len(candidates))
        chosen = rng.choice(len(candidates), size=take, replace=False)
        for pos in sorted(int(c) for c in chosen):
            paths.append(candidates[pos])
            labels.append(index)

    return paths, labels

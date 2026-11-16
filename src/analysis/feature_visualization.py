"""Visualisasi ruang fitur (feature space) dengan t-SNE dan UMAP.

Karena ketiga model memakai skema *feature extraction* (backbone beku, hanya
head yang dilatih), kualitas klasifikasi hampir sepenuhnya ditentukan oleh
seberapa terpisah kelas-kelas itu **sudah** berada di ruang fitur ImageNet
masing-masing arsitektur. Proyeksi 2D di sini memperlihatkan pemisahan itu
secara langsung: bila ``black`` dan ``dry_pods`` saling tumpang tindih di
ruang fitur, head linear memang tidak punya cara memisahkannya.

Fitur diambil dari lapisan *penultimate*, yaitu masukan head klasifikasi:
2048-d untuk ResNet-50 dan InceptionV3, 960-d untuk MobileNetV3.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from configs.base_config import CLASS_NAMES, DEVICE, RANDOM_SEED
from configs.cnn_config import CLASSIFIER_ATTR
from src.analysis.style import apply_plot_style, class_palette, save_figure

import matplotlib.pyplot as plt  # noqa: E402  (style.py sudah menetapkan backend Agg)

logger = logging.getLogger(__name__)


def _resolve_classifier(model: nn.Module, model_name: str | None) -> nn.Module:
    """Temukan modul head klasifikasi sebuah model.

    Args:
        model: Model yang diperiksa.
        model_name: Nama arsitektur; bila diketahui, dipakai untuk mencari
            atribut head dari ``CLASSIFIER_ATTR``.

    Returns:
        Modul head klasifikasi.

    Raises:
        AttributeError: Bila head tidak dapat ditemukan.
    """
    if model_name and model_name in CLASSIFIER_ATTR:
        attr = CLASSIFIER_ATTR[model_name]
        if hasattr(model, attr):
            return getattr(model, attr)

    for attr in ("fc", "classifier", "head"):
        if hasattr(model, attr):
            return getattr(model, attr)

    raise AttributeError(
        "Head klasifikasi tidak ditemukan; berikan model_name atau pastikan "
        "model punya atribut 'fc'/'classifier'/'head'."
    )


@torch.no_grad()
def extract_features(
    model: nn.Module,
    data_loader: DataLoader,
    model_name: str | None = None,
    device: str = DEVICE,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ekstrak fitur penultimate (masukan head klasifikasi) untuk satu loader.

    Fitur ditangkap lewat ``register_forward_pre_hook`` pada head, jadi model
    tidak perlu dimodifikasi dan head-nya tetap bisa dipakai memprediksi.

    Args:
        model: Model terlatih.
        data_loader: DataLoader split yang diekstrak (sebaiknya
            ``shuffle=False`` agar urutannya cocok dengan dataset).
        model_name: Nama arsitektur, untuk menentukan atribut head.
        device: Device komputasi.
        max_samples: Batas jumlah sampel (berguna agar t-SNE tidak kelamaan).
            ``None`` berarti seluruh data.

    Returns:
        Tuple ``(features, labels)`` dengan ``features`` berbentuk ``(N, D)``
        dan ``labels`` berbentuk ``(N,)``.
    """
    model = model.to(device).eval()
    classifier = _resolve_classifier(model, model_name)

    captured: list[np.ndarray] = []

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        """Tangkap masukan head klasifikasi (fitur penultimate)."""
        tensor = inputs[0]
        captured.append(torch.flatten(tensor, start_dim=1).detach().cpu().numpy())

    handle = classifier.register_forward_pre_hook(hook)
    labels: list[int] = []

    try:
        for images, batch_labels in data_loader:
            model(images.to(device))
            labels.extend(int(v) for v in batch_labels.tolist())
            if max_samples is not None and len(labels) >= max_samples:
                break
    finally:
        handle.remove()

    if not captured:
        logger.warning("Tidak ada fitur yang terekstrak (loader kosong?).")
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    features = np.concatenate(captured, axis=0)
    label_array = np.asarray(labels, dtype=np.int64)

    if max_samples is not None:
        features = features[:max_samples]
        label_array = label_array[:max_samples]

    logger.info("Fitur terekstrak: %s", features.shape)
    return features, label_array


def save_features(
    features: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    paths: list[str] | None = None,
) -> Path:
    """Simpan fitur dan label ke berkas ``.npz`` terkompresi.

    Args:
        features: Array ``(N, D)``.
        labels: Array ``(N,)``.
        path: Path tujuan ``.npz``.
        paths: Daftar path citra tiap sampel (opsional, untuk pelacakan).

    Returns:
        Path file yang tersimpan.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"features": features, "labels": labels}
    if paths is not None:
        payload["paths"] = np.asarray(paths, dtype=object)
    np.savez_compressed(path, **payload)
    return path


def load_features(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Muat fitur dan label dari berkas ``.npz``.

    Args:
        path: Path file hasil :func:`save_features`.

    Returns:
        Tuple ``(features, labels)``.

    Raises:
        FileNotFoundError: Bila file tidak ada.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File fitur tidak ditemukan: {path}")
    data = np.load(path, allow_pickle=True)
    return data["features"], data["labels"]


def compute_tsne(
    features: np.ndarray,
    perplexity: float = 30.0,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Proyeksikan fitur ke 2D dengan t-SNE.

    Args:
        features: Array ``(N, D)``.
        perplexity: Perplexity t-SNE; otomatis diturunkan bila sampelnya
            sedikit (syarat sklearn: ``perplexity < n_samples``).
        random_seed: Seed agar proyeksi reproducible.

    Returns:
        Array ``(N, 2)``.

    Raises:
        ValueError: Bila jumlah sampel kurang dari 3.
    """
    n_samples = int(features.shape[0])
    if n_samples < 3:
        raise ValueError(f"t-SNE butuh minimal 3 sampel, ada {n_samples}.")

    effective = float(min(perplexity, max(2.0, (n_samples - 1) / 3.0)))
    if effective != perplexity:
        logger.info("Perplexity diturunkan dari %.1f ke %.1f (n=%d)", perplexity, effective, n_samples)

    tsne = TSNE(
        n_components=2,
        perplexity=effective,
        init="pca",
        random_state=random_seed,
        max_iter=1000,
    )
    return tsne.fit_transform(features)


def compute_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Proyeksikan fitur ke 2D dengan UMAP.

    UMAP lebih cepat dari t-SNE pada dataset besar dan cenderung
    mempertahankan struktur global lebih baik.

    Args:
        features: Array ``(N, D)``.
        n_neighbors: Jumlah tetangga UMAP; diturunkan otomatis bila sampelnya
            sedikit.
        min_dist: Jarak minimum antar-titik pada hasil proyeksi.
        random_seed: Seed agar proyeksi reproducible.

    Returns:
        Array ``(N, 2)``.

    Raises:
        ImportError: Bila paket ``umap-learn`` belum terpasang.
        ValueError: Bila jumlah sampel kurang dari 3.
    """
    try:
        import umap  # noqa: PLC0415 - impor opsional, hanya saat dipakai
    except ImportError as exc:  # pragma: no cover - bergantung environment
        raise ImportError(
            "UMAP belum terpasang. Jalankan: pip install umap-learn"
        ) from exc

    n_samples = int(features.shape[0])
    if n_samples < 3:
        raise ValueError(f"UMAP butuh minimal 3 sampel, ada {n_samples}.")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(min(n_neighbors, n_samples - 1)),
        min_dist=min_dist,
        random_state=random_seed,
    )

    # UMAP memperingatkan bahwa random_state memaksa n_jobs=1. Itu memang
    # konsekuensi yang dipilih: proyeksi yang reproducible lebih penting
    # daripada paralelisme pada dataset sebesar ini.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
        return reducer.fit_transform(features)


def plot_feature_projection(
    projection: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Proyeksi ruang fitur",
    output_path: str | Path | None = None,
    ax=None,
):
    """Scatter plot proyeksi 2D, diwarnai menurut kelas.

    Args:
        projection: Array ``(N, 2)`` hasil t-SNE/UMAP.
        labels: Array ``(N,)`` indeks kelas.
        class_names: Urutan nama kelas. Default ``CLASS_NAMES``.
        title: Judul plot.
        output_path: Bila diberikan, simpan figur ke path ini.
        ax: Axes tujuan; bila ``None``, figur baru dibuat.

    Returns:
        Objek ``matplotlib.figure.Figure``.
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    colors = class_palette(names)

    apply_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
    else:
        fig = ax.get_figure()

    for index, name in enumerate(names):
        mask = labels == index
        if not np.any(mask):
            continue
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=12,
            alpha=0.7,
            color=colors[index],
            label=f"{name} (n={int(mask.sum())})",
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(loc="best", markerscale=1.6)
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_tsne_features(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    output_path: str | Path | None = None,
    title: str = "t-SNE ruang fitur",
    perplexity: float = 30.0,
    random_seed: int = RANDOM_SEED,
):
    """Hitung t-SNE lalu langsung plot hasilnya.

    Args:
        features: Array ``(N, D)``.
        labels: Array ``(N,)``.
        class_names: Urutan nama kelas.
        output_path: Bila diberikan, simpan figur ke path ini.
        title: Judul plot.
        perplexity: Perplexity t-SNE.
        random_seed: Seed proyeksi.

    Returns:
        Objek ``matplotlib.figure.Figure``.
    """
    projection = compute_tsne(features, perplexity=perplexity, random_seed=random_seed)
    return plot_feature_projection(
        projection, labels, class_names=class_names, title=title, output_path=output_path
    )


def plot_umap_features(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    output_path: str | Path | None = None,
    title: str = "UMAP ruang fitur",
    random_seed: int = RANDOM_SEED,
):
    """Hitung UMAP lalu langsung plot hasilnya.

    Args:
        features: Array ``(N, D)``.
        labels: Array ``(N,)``.
        class_names: Urutan nama kelas.
        output_path: Bila diberikan, simpan figur ke path ini.
        title: Judul plot.
        random_seed: Seed proyeksi.

    Returns:
        Objek ``matplotlib.figure.Figure``.
    """
    projection = compute_umap(features, random_seed=random_seed)
    return plot_feature_projection(
        projection, labels, class_names=class_names, title=title, output_path=output_path
    )


def plot_projection_comparison(
    projections: dict[str, tuple[np.ndarray, np.ndarray]],
    class_names: list[str] | None = None,
    output_path: str | Path | None = None,
    suptitle: str = "Perbandingan ruang fitur antar-arsitektur",
):
    """Tampilkan proyeksi beberapa arsitektur berdampingan.

    Args:
        projections: Dict ``nama_model -> (projection (N,2), labels (N,))``.
        class_names: Urutan nama kelas.
        output_path: Bila diberikan, simpan figur ke path ini.
        suptitle: Judul figur.

    Returns:
        Objek ``matplotlib.figure.Figure``.

    Raises:
        ValueError: Bila ``projections`` kosong.
    """
    if not projections:
        raise ValueError("projections kosong.")

    apply_plot_style()
    fig, axes = plt.subplots(
        1, len(projections), figsize=(5.4 * len(projections), 5.2), squeeze=False
    )

    for ax, (model_name, (projection, labels)) in zip(axes[0], projections.items()):
        plot_feature_projection(
            projection, labels, class_names=class_names, title=model_name, ax=ax
        )

    fig.suptitle(suptitle)
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def cluster_separation_score(
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Ukur pemisahan antar-kelas di ruang fitur dengan silhouette score.

    Pendamping kuantitatif untuk plot t-SNE/UMAP: nilai mendekati ``1``
    berarti kelas membentuk klaster rapat dan terpisah, mendekati ``0``
    berarti saling tumpang tindih, dan negatif berarti sampel rata-rata lebih
    dekat ke klaster kelas lain.

    Dihitung pada fitur berdimensi penuh, bukan pada proyeksi 2D — proyeksi
    adalah alat lihat, bukan alat ukur.

    Args:
        features: Array ``(N, D)``.
        labels: Array ``(N,)``.

    Returns:
        Silhouette score, atau ``float("nan")`` bila hanya ada satu kelas.
    """
    unique = np.unique(labels)
    if unique.size < 2:
        logger.warning("Silhouette score butuh >= 2 kelas; ada %d.", unique.size)
        return float("nan")
    return float(silhouette_score(features, labels))

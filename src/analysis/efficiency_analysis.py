"""Analisis trade-off efisiensi vs akurasi antar-arsitektur.

Untuk Tugas Akhir ini, "arsitektur terbaik" tidak otomatis berarti yang paling
akurat: MobileNetV3 yang 5x lebih kecil dengan F1 sedikit di bawah ResNet-50
bisa jadi pilihan yang lebih tepat untuk perangkat tepi. Modul ini
mengumpulkan angka yang dibutuhkan untuk argumen itu: jumlah parameter, ukuran
model di disk, waktu inferensi, dan estimasi FLOPs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from configs.base_config import DEVICE
from src.analysis.style import apply_plot_style, save_figure

import matplotlib.pyplot as plt  # noqa: E402  (style.py sudah menetapkan backend Agg)

logger = logging.getLogger(__name__)

#: Byte per parameter float32.
_BYTES_PER_PARAM: int = 4


def count_flops(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
) -> int:
    """Estimasi FLOPs satu forward pass (hanya Conv2d dan Linear).

    Dihitung dengan forward hook: untuk tiap ``Conv2d`` dijumlahkan
    ``kernel_ops * output_elements``, dan untuk tiap ``Linear``
    ``in_features * out_features``. Operasi lain (BatchNorm, aktivasi,
    pooling) diabaikan karena kontribusinya kecil pada CNN seperti ini.

    Angka yang dikembalikan adalah **MAC** (multiply-accumulate), konvensi yang
    umum dipakai literatur saat menyebut "FLOPs" model. Karena cakupannya
    terbatas, nilainya cocok untuk **membandingkan** arsitektur, bukan untuk
    diklaim sebagai hitungan FLOP yang eksak.

    Args:
        model: Model yang diukur.
        input_shape: Bentuk tensor input dummy, termasuk batch.
        device: Device eksekusi.

    Returns:
        Estimasi MAC per forward pass (untuk batch sesuai ``input_shape``).
    """
    model = model.to(device).eval()
    total = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def conv_hook(module: nn.Conv2d, _inputs, output: torch.Tensor) -> None:
        nonlocal total
        kernel_ops = module.in_channels // module.groups
        for size in module.kernel_size:
            kernel_ops *= size
        total += int(kernel_ops * output.numel())

    def linear_hook(module: nn.Linear, _inputs, output: torch.Tensor) -> None:
        nonlocal total
        # rows = output.numel() / out_features, dan MAC = rows * in * out,
        # sehingga totalnya menyederhana menjadi in_features * output.numel().
        total += int(module.in_features * output.numel())

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    try:
        with torch.no_grad():
            model(torch.randn(*input_shape, device=device))
    finally:
        for handle in handles:
            handle.remove()

    return total


def measure_model_efficiency(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    device: str = DEVICE,
    n_runs: int = 100,
    n_warmup: int = 10,
    checkpoint_path: str | Path | None = None,
) -> dict[str, float | int]:
    """Ukur seluruh metrik efisiensi sebuah model.

    Args:
        model: Model yang diukur.
        input_shape: Bentuk tensor input dummy, termasuk batch.
        device: Device eksekusi.
        n_runs: Jumlah iterasi pengukuran waktu.
        n_warmup: Jumlah iterasi pemanasan (tidak dihitung).
        checkpoint_path: Bila diberikan, ukuran berkas ini dilaporkan sebagai
            ``checkpoint_size_mb`` (ukuran nyata di disk).

    Returns:
        Dict berisi ``params_total``, ``params_trainable``, ``model_size_mb``
        (estimasi dari jumlah parameter float32), ``checkpoint_size_mb``
        (bila ada), ``inference_time_ms``, ``throughput_img_per_s``, dan
        ``flops``.
    """
    model = model.to(device).eval()

    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    per_run_s = elapsed / max(n_runs, 1)
    batch = int(input_shape[0]) if input_shape else 1

    result: dict[str, float | int] = {
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "model_size_mb": round(params_total * _BYTES_PER_PARAM / (1024**2), 3),
        "inference_time_ms": round(per_run_s * 1000.0, 4),
        "throughput_img_per_s": round(batch / per_run_s, 2) if per_run_s > 0 else 0.0,
        "flops": int(count_flops(model, input_shape=input_shape, device=device)),
        "input_size": int(input_shape[-1]) if input_shape else 0,
        "device": device,
    }

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists():
            result["checkpoint_size_mb"] = round(path.stat().st_size / (1024**2), 3)
        else:
            logger.warning("Checkpoint tidak ditemukan: %s", path)

    return result


def efficiency_table(results: dict[str, dict]) -> pd.DataFrame:
    """Susun tabel perbandingan efisiensi + akurasi beberapa model.

    Args:
        results: Dict ``nama_model -> dict metrik``. Kunci yang dikenali:
            ``params_total``, ``params_trainable``, ``model_size_mb``,
            ``inference_time_ms``, ``flops``, ``accuracy``, ``f1``.

    Returns:
        DataFrame berindeks nama model, terurut menurun menurut ``f1`` bila
        kolom itu tersedia.
    """
    frame = pd.DataFrame(results).T
    frame.index.name = "model"
    if "f1" in frame.columns:
        frame = frame.sort_values("f1", ascending=False)
    return frame


def plot_efficiency_accuracy_tradeoff(
    results: dict[str, dict],
    metric: str = "f1",
    output_path: str | Path | None = None,
    title: str = "Trade-off efisiensi vs akurasi",
):
    """Scatter plot parameter (log) vs akurasi, ukuran titik = waktu inferensi.

    Args:
        results: Dict ``nama_model -> dict metrik`` yang memuat
            ``params_total``, ``inference_time_ms``, dan ``metric``.
        metric: Nama metrik akurasi untuk sumbu y (mis. ``"f1"``).
        output_path: Bila diberikan, simpan figur ke path ini.
        title: Judul plot.

    Returns:
        Objek ``matplotlib.figure.Figure``.

    Raises:
        ValueError: Bila ``results`` kosong.
    """
    if not results:
        raise ValueError("results kosong.")

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    times = [float(info.get("inference_time_ms", 1.0)) for info in results.values()]
    max_time = max(times) if times else 1.0

    for (name, info), infer_ms in zip(results.items(), times):
        params = float(info.get("params_total", 0)) or 1.0
        score = float(info.get(metric, 0.0))
        size = 160.0 + 900.0 * (infer_ms / max_time if max_time else 0.0)
        ax.scatter(params, score, s=size, alpha=0.6, edgecolors="black", linewidths=0.8)
        ax.annotate(
            f"{name}\n{infer_ms:.2f} ms",
            (params, score),
            textcoords="offset points",
            xytext=(12, 8),
            fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Jumlah parameter (skala log)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{title}\n(ukuran titik = waktu inferensi)")
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_inference_time_comparison(
    results: dict[str, dict],
    output_path: str | Path | None = None,
    title: str = "Perbandingan waktu inferensi",
):
    """Bar chart waktu inferensi per model.

    Args:
        results: Dict ``nama_model -> dict metrik`` dengan
            ``inference_time_ms``.
        output_path: Bila diberikan, simpan figur ke path ini.
        title: Judul plot.

    Returns:
        Objek ``matplotlib.figure.Figure``.
    """
    if not results:
        raise ValueError("results kosong.")

    names = list(results)
    values = [float(results[n].get("inference_time_ms", 0.0)) for n in names]

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 2.4, 4.2))
    bars = ax.bar(names, values, color="#3498db", alpha=0.85, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Waktu inferensi (ms/citra)")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


def plot_size_vs_metric(
    results: dict[str, dict],
    metric: str = "f1",
    output_path: str | Path | None = None,
):
    """Bar chart ganda: ukuran model (MB) dan metrik akurasi berdampingan.

    Args:
        results: Dict ``nama_model -> dict metrik``.
        metric: Nama metrik akurasi.
        output_path: Bila diberikan, simpan figur ke path ini.

    Returns:
        Objek ``matplotlib.figure.Figure``.
    """
    if not results:
        raise ValueError("results kosong.")

    names = list(results)
    sizes = [float(results[n].get("model_size_mb", 0.0)) for n in names]
    scores = [float(results[n].get(metric, 0.0)) for n in names]

    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    axes[0].bar(names, sizes, color="#9b59b6", alpha=0.85, width=0.55)
    axes[0].set_ylabel("Ukuran model (MB)")
    axes[0].set_title("Ukuran model")

    axes[1].bar(names, scores, color="#2ecc71", alpha=0.85, width=0.55)
    axes[1].set_ylabel(metric.upper())
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title(f"{metric.upper()} pada split test")

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig

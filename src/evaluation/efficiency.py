"""Pengukuran efisiensi model: jumlah parameter dan waktu inferensi.

Metrik ini melengkapi perbandingan arsitektur dari sisi biaya komputasi.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Hitung jumlah parameter total dan yang dapat dilatih.

    Args:
        model: Model yang diperiksa.

    Returns:
        Dict berisi ``total`` dan ``trainable``.

    Example:
        >>> import torch.nn as nn
        >>> count_parameters(nn.Linear(10, 2))
        {'total': 22, 'trainable': 22}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def measure_inference_time(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
    n_runs: int = 100,
    n_warmup: int = 10,
) -> float:
    """Ukur rata-rata waktu inferensi per forward pass (milidetik).

    Args:
        model: Model yang diukur (dipindah ke ``device`` dan diset ``eval``).
        input_shape: Bentuk tensor input dummy, termasuk batch.
        device: Device eksekusi (``"cpu"`` atau ``"cuda"``).
        n_runs: Jumlah iterasi pengukuran.
        n_warmup: Jumlah iterasi pemanasan (tidak dihitung).

    Returns:
        Rata-rata waktu inferensi dalam milidetik.
    """
    model = model.to(device).eval()
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

    return (elapsed / n_runs) * 1000.0

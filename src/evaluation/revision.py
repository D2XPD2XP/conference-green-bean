"""Metrik & agregasi untuk eksperimen revisi reviewer (multi-seed).

Satu run = satu (konfigurasi, model, seed). Tiap run menyimpan **logit** valid
dan test untuk semua varian TTA, sehingga seluruh metrik (termasuk ambang
P(good) dan bootstrap) dapat dihitung ulang tanpa memuat model lagi.

Modul ini hanya berisi fungsi murni (numpy/pandas); orkestrasi ada di
``scripts/run_revision.py`` dan ``scripts/aggregate_revision.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from configs.base_config import CLASS_NAMES, GOOD_CLASS
from src.evaluation.metrics import (
    compute_defect_metrics,
    compute_metrics,
    compute_per_class_metrics,
)
from src.evaluation.threshold import apply_good_threshold

#: Penanda pemisah nama foto sumber pada nama berkas crop.
_BEAN_MARKER = "_bean_"


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax stabil secara numerik per baris."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def photo_id_from_path(path: str | Path) -> str:
    """Ambil id foto sumber dari path crop.

    ``IMG_20260814_050353_jpg.rf.<hash>_bean_012_orig.jpg`` ->
    ``IMG_20260814_050353``.
    """
    name = Path(path).name
    stem = name.split(_BEAN_MARKER)[0]
    return stem.split("_jpg")[0]


def metrics_from_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: np.ndarray | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict:
    """Seluruh metrik tingkat biji untuk satu set prediksi.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.
        probs: Probabilitas ``(N, C)`` (opsional, untuk statistik confidence).
        class_names: Urutan nama kelas.

    Returns:
        Dict berisi metrik makro (``accuracy``, ``precision``, ``recall``,
        ``f1``), ``per_class``, metrik layak-vs-cacat (``defect_recall``,
        ``defect_miss_rate``, ...), ``insect_to_good``, ``n_defect``,
        ``n_defect_missed``, ``confusion`` (list baris = true), dan bila
        ``probs`` ada: ``mean_conf_correct`` / ``mean_conf_incorrect``.
    """
    names = list(class_names)
    y_true = [int(v) for v in y_true]
    y_pred = [int(v) for v in y_pred]
    good = names.index(GOOD_CLASS)
    insect = names.index("insect_damage")
    yt, yp = np.asarray(y_true), np.asarray(y_pred)

    out: dict = dict(compute_metrics(y_true, y_pred))
    out["per_class"] = compute_per_class_metrics(y_true, y_pred, names)
    out.update(compute_defect_metrics(y_true, y_pred, names))
    # Miss rate eksak (tanpa pembulatan) dari hitungan, untuk pelaporan sekali-bulat.
    is_defect = yt != good
    out["n_defect"] = int(is_defect.sum())
    out["n_defect_missed"] = int((is_defect & (yp == good)).sum())
    out["defect_miss_rate"] = out["n_defect_missed"] / max(out["n_defect"], 1)
    out["defect_recall"] = 1.0 - out["defect_miss_rate"]
    out["insect_to_good"] = int(((yt == insect) & (yp == good)).sum())
    out["n_insect"] = int((yt == insect).sum())
    out["confusion"] = confusion_matrix(y_true, y_pred, labels=list(range(len(names)))).tolist()
    if probs is not None:
        conf = probs.max(axis=1)
        correct = yt == yp
        out["mean_conf_correct"] = float(conf[correct].mean()) if correct.any() else float("nan")
        out["mean_conf_incorrect"] = (
            float(conf[~correct].mean()) if (~correct).any() else float("nan")
        )
    return out


def metrics_from_logits(
    y_true: Sequence[int],
    logits: np.ndarray,
    threshold: float = 0.0,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict:
    """Metrik dari logit, opsional dengan ambang P(good).

    Args:
        y_true: Label sebenarnya.
        logits: Logit ``(N, C)`` (rata-rata TTA bila dipakai).
        threshold: Ambang P(good); ``0`` = argmax biasa.
        class_names: Urutan nama kelas.

    Returns:
        Lihat :func:`metrics_from_predictions`.
    """
    probs = softmax(np.asarray(logits, dtype=np.float64))
    names = list(class_names)
    preds = apply_good_threshold(probs, threshold, good_index=names.index(GOOD_CLASS))
    return metrics_from_predictions(y_true, preds, probs=probs, class_names=names)


def flatten_metrics(metrics: dict, class_names: Sequence[str] = CLASS_NAMES) -> dict[str, float]:
    """Ratakan dict metrik menjadi kolom skalar (untuk tabel per run)."""
    row = {
        key: metrics[key]
        for key in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "defect_recall",
            "defect_miss_rate",
            "insect_to_good",
            "n_defect_missed",
        )
    }
    for name in class_names:
        row[f"f1_{name}"] = metrics["per_class"][name]["f1"]
    for key in ("mean_conf_correct", "mean_conf_incorrect"):
        if key in metrics:
            row[key] = metrics[key]
    return row


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Rata-rata dan simpangan baku sampel (ddof=1; 0 bila n=1)."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def fmt_mean_std(values: Sequence[float], digits: int = 3, scale: float = 1.0) -> str:
    """Format ``mean ± std`` (dibulatkan **sekali** dari nilai eksak)."""
    mean, std = mean_std([v * scale for v in values])
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def bootstrap_photo_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    photo_ids: Sequence[str],
    metric: str = "f1",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """CI bootstrap dengan me-resample **foto** (bukan biji).

    Tiap iterasi menarik foto test dengan pengembalian sebanyak jumlah foto,
    menggabungkan seluruh biji dari foto terpilih, lalu menghitung metrik.

    Args:
        y_true: Label sebenarnya ``(N,)``.
        y_pred: Prediksi ``(N,)``.
        photo_ids: Id foto per biji ``(N,)``.
        metric: Kunci metrik dari :func:`compute_metrics` (default ``"f1"``).
        n_boot: Jumlah iterasi.
        seed: Seed generator.
        alpha: Tingkat signifikansi (0,05 -> CI95).

    Returns:
        Dict ``point``, ``lo``, ``hi``, ``mean``, ``std``, ``n_photos``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    photos = np.asarray(photo_ids)
    unique = np.unique(photos)
    index = {p: np.flatnonzero(photos == p) for p in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        pick = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([index[p] for p in pick])
        values.append(compute_metrics(y_true[idx].tolist(), y_pred[idx].tolist())[metric])
    values = np.asarray(values)
    point = compute_metrics(y_true.tolist(), y_pred.tolist())[metric]
    return {
        "point": float(point),
        "lo": float(np.quantile(values, alpha / 2)),
        "hi": float(np.quantile(values, 1 - alpha / 2)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "n_photos": int(unique.size),
        "n_boot": int(n_boot),
    }


def load_run_predictions(npz_path: str | Path) -> dict[str, dict]:
    """Muat ``predictions.npz`` sebuah run menjadi dict per split.

    Returns:
        ``{split: {"y_true": arr, "paths": arr, "logits": {tta: arr}}}``.
    """
    data = np.load(npz_path, allow_pickle=False)
    out: dict[str, dict] = {}
    for key in data.files:
        split, _, rest = key.partition("__")
        entry = out.setdefault(split, {"logits": {}})
        if rest.startswith("logits_"):
            entry["logits"][rest[len("logits_"):]] = data[key]
        else:
            entry[rest] = data[key]
    return out


def summarize_table(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    """Hitung mean, std, dan n per grup untuk kolom-kolom metrik."""
    grouped = frame.groupby(list(group_cols))[list(value_cols)]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).fillna(0.0).add_suffix("_std")
    n = grouped.size().rename("n_seeds")
    return pd.concat([mean, std, n], axis=1).reset_index()

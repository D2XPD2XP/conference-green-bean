"""Test komponen eksperimen revisi: cache, checkpoint F1, BN beku, ambang,
augmentasi pengganti, dan metrik/bootstrap.

Semua citra sintetis dibuat di ``tmp_path`` (tidak ada data dummy di repo).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from configs.base_config import CLASS_NAMES
from src.data.cache import CachedCropDataset
from src.data.dataset import GreenBeanDataset
from src.data.offline_augmentation import generate_augmented_dataset
from src.data.transforms import get_transforms
from src.evaluation.revision import (
    bootstrap_photo_ci,
    mean_std,
    metrics_from_logits,
    photo_id_from_path,
)
from src.evaluation.threshold import select_threshold
from src.training.callbacks import ModelCheckpoint
from src.training.train_cnn import freeze_batchnorm_stats, train_cnn


def _write_crops(root: Path, per_class: int = 3, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for name in CLASS_NAMES:
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            h, w = rng.integers(40, 90, size=2)
            image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            cv2.imwrite(str(folder / f"IMG_{i:03d}_jpg.rf.x_bean_{i:03d}.jpg"), image)


@pytest.mark.parametrize("model_name", ["resnet50", "inceptionv3"])
def test_cached_dataset_matches_pil_pipeline(tmp_path: Path, model_name: str) -> None:
    _write_crops(tmp_path / "split" / "valid")
    root = tmp_path / "split" / "valid"
    size = 299 if model_name == "inceptionv3" else 224
    cached = CachedCropDataset(root, size, cache_dir=tmp_path / "cache")
    reference = GreenBeanDataset(root, transform=get_transforms(model_name, "valid"))
    assert len(cached) == len(reference)
    for i in range(len(reference)):
        a, la = cached[i]
        b, lb = reference[i]
        assert la == lb
        assert torch.equal(a, b)


def test_cache_rebuilt_when_files_change(tmp_path: Path) -> None:
    root = tmp_path / "s" / "train"
    _write_crops(root, per_class=1)
    first = CachedCropDataset(root, 224, cache_dir=tmp_path / "cache").cache_path
    _write_crops(root, per_class=2, seed=1)  # menambah & menimpa berkas
    second = CachedCropDataset(root, 224, cache_dir=tmp_path / "cache").cache_path
    assert first != second


def test_checkpoint_tag_and_tie_keeps_earlier(tmp_path: Path) -> None:
    ckpt = ModelCheckpoint(tmp_path, "m", metric_tag="f1")
    model = nn.Linear(2, 2)
    assert ckpt.maybe_save(model, 0.5, 1) is not None
    assert ckpt.maybe_save(model, 0.5, 2) is None  # seri -> epoch awal dipertahankan
    assert ckpt.best_path.name == "m_best_epoch1_f10.50.pt"


def test_freeze_batchnorm_keeps_running_stats() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Dropout(0.5))
    model.train()
    assert freeze_batchnorm_stats(model) == 1
    before = model[1].running_mean.clone()
    model(torch.randn(4, 3, 8, 8))
    assert torch.equal(before, model[1].running_mean)
    assert model[2].training  # dropout tidak disentuh


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.fc = nn.Linear(4, len(CLASS_NAMES))
        for p in self.bn.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bn(x))


def _loaders(seed: int = 0) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(60, 4, generator=g)
    y = torch.arange(60) % len(CLASS_NAMES)
    ds = TensorDataset(x + y[:, None].float(), y)
    ds.labels = y.tolist()  # bobot kelas membaca atribut ini
    return DataLoader(ds, batch_size=16, shuffle=True), DataLoader(ds, batch_size=16)


@pytest.mark.parametrize("bn_mode", ["adapt", "frozen"])
def test_train_cnn_records_f1_and_bn_mode(tmp_path: Path, bn_mode: str) -> None:
    torch.manual_seed(0)
    model = _Tiny()
    before = model.bn.running_mean.clone()
    train_loader, val_loader = _loaders()
    history = train_cnn(
        model,
        train_loader,
        val_loader,
        config={"epochs": 3, "learning_rate": 0.05},
        output_dir=tmp_path,
        model_name="tiny",
        device="cpu",
        checkpoint_metric="val_f1",
        bn_mode=bn_mode,
    )
    assert len(history["val_f1"]) == 3
    best = history["best_epoch"]
    assert history["best_metric"] == pytest.approx(max(history["val_f1"]))
    assert history["val_f1"].index(max(history["val_f1"])) + 1 == best  # epoch paling awal
    assert history["best_val_acc"] == history["val_acc"][best - 1]
    assert Path(history["best_model_path"]).name.startswith(f"tiny_best_epoch{best}_f1")
    frozen = torch.equal(before, model.bn.running_mean.cpu())
    assert frozen == (bn_mode == "frozen")


def test_train_cnn_rejects_unknown_metric(tmp_path: Path) -> None:
    train_loader, val_loader = _loaders()
    with pytest.raises(ValueError):
        train_cnn(_Tiny(), train_loader, val_loader, {"epochs": 1}, tmp_path,
                  device="cpu", checkpoint_metric="val_loss")


def test_replace_fractions_keeps_count_and_reuses_aug01(tmp_path: Path) -> None:
    src = tmp_path / "raw" / "train"
    _write_crops(src, per_class=6)
    full = generate_augmented_dataset(src, tmp_path / "x3", multipliers={n: 3 for n in CLASS_NAMES})
    bal = generate_augmented_dataset(
        src, tmp_path / "bal", multipliers={n: 1 for n in CLASS_NAMES},
        replace_fractions={"good": 2 / 3},
    )
    assert full["per_class"]["good"]["output"] == 18
    assert bal["per_class"]["good"]["output"] == 6
    assert bal["per_class"]["good"]["replaced"] == 4
    good = sorted(p.name for p in (tmp_path / "bal" / "good").iterdir())
    assert sum(n.endswith("_orig.jpg") for n in good) == 2
    for name in (n for n in good if n.endswith("_aug01.jpg")):
        a = (tmp_path / "bal" / "good" / name).read_bytes()
        b = (tmp_path / "x3" / "good" / name).read_bytes()
        assert a == b  # varian identik dengan konfigurasi x3
    # Kelas lain tidak tersentuh.
    assert bal["per_class"]["black"]["replaced"] == 0


def test_replace_fractions_requires_multiplier_one(tmp_path: Path) -> None:
    src = tmp_path / "raw" / "train"
    _write_crops(src, per_class=2)
    with pytest.raises(ValueError):
        generate_augmented_dataset(src, tmp_path / "o", multipliers={"good": 3},
                                   replace_fractions={"good": 0.5})


def test_select_threshold_argmax_reference() -> None:
    sweep = pd.DataFrame(
        {"f1": [0.70, 0.72, 0.705, 0.69], "miss_rate": [0.20, 0.15, 0.10, 0.05]},
        index=pd.Index([0.0, 0.4, 0.5, 0.6], name="threshold"),
    )
    # argmax: batas 0,69 -> 0.6 lolos (miss terendah).
    assert select_threshold(sweep, 0.01, reference="argmax") == pytest.approx(0.6)
    # best: batas 0,71 -> hanya 0.4.
    assert select_threshold(sweep, 0.01, reference="best") == pytest.approx(0.4)


def test_metrics_from_logits_exact_miss_rate() -> None:
    # 3 cacat, 1 lolos sebagai good -> miss rate 1/3 eksak (tidak dibulatkan).
    y_true = [0, 1, 2, 4]
    logits = np.array([[5, 0, 0, 0, 0], [0, 5, 0, 0, 0], [0, 0, 5, 0, 0], [5, 0, 0, 0, 0]], float)
    m = metrics_from_logits(y_true, logits)
    assert m["defect_miss_rate"] == pytest.approx(1 / 3, abs=1e-15)
    assert m["insect_to_good"] == 1
    assert m["confusion"][4][0] == 1


def test_photo_id_and_bootstrap() -> None:
    assert photo_id_from_path("IMG_20260814_050353_jpg.rf.ab_bean_012_orig.jpg") == "IMG_20260814_050353"
    y = np.array([0, 1, 2, 3, 4] * 4)
    photos = np.repeat(["a", "b", "c", "d"], 5)
    ci = bootstrap_photo_ci(y, y, photos, n_boot=50)
    assert ci["point"] == 1.0 and ci["lo"] == 1.0 and ci["n_photos"] == 4
    assert mean_std([1.0]) == (1.0, 0.0)


def test_bnf_ablations_change_one_factor_from_bn_frozen() -> None:
    """Ablasi di atas bn_frozen hanya mengubah satu faktor dari konfigurasi utama."""
    from scripts.run_revision import CONFIGS, MAIN_SEEDS

    main = CONFIGS["bn_frozen"]
    assert main["bn_mode"] == "frozen" and tuple(main["seeds"]) == MAIN_SEEDS
    expected = {
        "bnf_focal": {"strategy"},
        "bnf_sampler": {"strategy"},
        "bnf_aug_all_x3": {"train_root"},
        "bnf_aug_all_balanced": {"train_root"},
        "bnf_e60": {"epochs"},
    }
    keys = {"train_root", "strategy", "checkpoint_metric", "bn_mode", "epochs"}
    for name, changed in expected.items():
        spec = CONFIGS[name]
        diff = {k for k in keys if spec.get(k, 30 if k == "epochs" else None)
                != main.get(k, 30 if k == "epochs" else None)}
        assert diff == changed, (name, diff)
    # Ablasi lama (BN beradaptasi) tetap ada sebagai arsip / pembanding bn_adapt.
    assert CONFIGS["main"]["bn_mode"] == "adapt"

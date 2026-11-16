"""CLI: jalankan seluruh eksperimen revisi reviewer (multi-seed, ablasi).

Satu run = (konfigurasi, model, seed), disimpan ke
``runs/{model}_{config}_seed{seed}/``:

- ``<model>_best_epoch{N}_f1{X.XX}.pt`` : checkpoint terbaik (kriteria run)
- ``<model>_history.json``              : per epoch train/val loss, acc, F1 makro
- ``train.log``                         : log pelatihan
- ``predictions.npz``                   : logit valid & test, 4 varian TTA
- ``metrics.json``                      : metrik valid & test per varian TTA

Run yang sudah punya ``metrics.json`` dilewati, sehingga perintah yang sama
dapat diulang untuk melanjutkan bila terputus.

Pelatihan memakai :class:`src.data.cache.CachedCropDataset` (crop ter-resize
di-cache di disk); tensornya identik bit-per-bit dengan pipeline biasa, tetapi
JPEG tidak di-decode ulang tiap epoch. Seluruh pemilihan (checkpoint) hanya
memakai valid; test hanya dievaluasi setelah checkpoint dipilih.

Contoh::

    python -m scripts.run_revision                        # semua konfigurasi
    python -m scripts.run_revision --configs main --models resnet50
    python -m scripts.run_revision --dry-run              # daftar run saja
    python -m scripts.run_revision --configs main --seeds 42 --epochs 1 \\
        --runs-dir runs_smoke                             # uji cepat
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # CLI: tanpa jendela.

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from configs.base_config import (  # noqa: E402
    CLASS_NAMES,
    CROPS_DIR,
    DATA_DIR,
    DEVICE,
    NUM_CLASSES,
    PROJECT_ROOT,
)
from configs.cnn_config import CNN_CONFIG  # noqa: E402
from src.data.cache import CachedCropDataset  # noqa: E402
from src.evaluation.revision import flatten_metrics, metrics_from_logits  # noqa: E402
from src.evaluation.tta import TTA_CHOICES, wrap_tta  # noqa: E402
from src.models.model_factory import AVAILABLE_MODELS, get_model  # noqa: E402
from src.training.imbalance import create_weighted_sampler  # noqa: E402
from src.training.train_cnn import train_cnn  # noqa: E402
from src.utils.io import load_checkpoint, save_json  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

#: Seed konfigurasi utama (Tugas 3) dan ablasi (Tugas 4, 5, 7).
MAIN_SEEDS: tuple[int, ...] = (42, 0, 1, 2, 3)
ABLATION_SEEDS: tuple[int, ...] = (42, 0, 1)

#: Definisi konfigurasi. ``train_root`` hanya menentukan split train; valid &
#: test SELALU dari ``data/crops`` (citra asli) agar semua run dinilai pada
#: data yang sama persis.
CONFIGS: dict[str, dict] = {
    # Reproduksi run lama (kriteria val_acc, seed 42): verifikasi bahwa jalur
    # kode baru menghasilkan angka identik dengan results/ 2026-10-15.
    "repro_valacc": {
        "train_root": CROPS_DIR,
        "strategy": "class_weights",
        "checkpoint_metric": "val_acc",
        "bn_mode": "adapt",
        "seeds": (42,),
    },
    "main": {
        "train_root": CROPS_DIR,
        "strategy": "class_weights",
        "checkpoint_metric": "val_f1",
        "bn_mode": "adapt",
        "seeds": MAIN_SEEDS,
    },
    "focal_loss": {
        "train_root": CROPS_DIR,
        "strategy": "focal_loss",
        "checkpoint_metric": "val_f1",
        "bn_mode": "adapt",
        "seeds": ABLATION_SEEDS,
    },
    "weighted_sampler": {
        "train_root": CROPS_DIR,
        "strategy": "weighted_sampler",
        "checkpoint_metric": "val_f1",
        "bn_mode": "adapt",
        "seeds": ABLATION_SEEDS,
    },
    "aug_all_x3": {
        "train_root": DATA_DIR / "crops_aug_all_x3",
        "strategy": "class_weights",
        "checkpoint_metric": "val_f1",
        "bn_mode": "adapt",
        "seeds": ABLATION_SEEDS,
    },
    "aug_all_balanced": {
        "train_root": DATA_DIR / "crops_aug_all_balanced",
        "strategy": "class_weights",
        "checkpoint_metric": "val_f1",
        "bn_mode": "adapt",
        "seeds": ABLATION_SEEDS,
    },
    # Konfigurasi UTAMA sejak brief lanjutan 2026-11-13 (di laporan: "main";
    # konfigurasi "main" di atas dilabeli "bn_adapt").
    "bn_frozen": {
        "train_root": CROPS_DIR,
        "strategy": "class_weights",
        "checkpoint_metric": "val_f1",
        "bn_mode": "frozen",
        "seeds": MAIN_SEEDS,
    },
    # Ablasi di atas bn_frozen: tiap baris hanya mengubah satu faktor dari
    # konfigurasi utama baru. Run ablasi lama (BN beradaptasi) tetap diarsipkan.
    # focal: alpha = bobot kelas balanced, gamma = 2 (identik ablasi lama).
    "bnf_focal": {
        "train_root": CROPS_DIR,
        "strategy": "focal_loss",
        "checkpoint_metric": "val_f1",
        "bn_mode": "frozen",
        "seeds": ABLATION_SEEDS,
    },
    "bnf_sampler": {
        "train_root": CROPS_DIR,
        "strategy": "weighted_sampler",
        "checkpoint_metric": "val_f1",
        "bn_mode": "frozen",
        "seeds": ABLATION_SEEDS,
    },
    "bnf_aug_all_x3": {
        "train_root": DATA_DIR / "crops_aug_all_x3",
        "strategy": "class_weights",
        "checkpoint_metric": "val_f1",
        "bn_mode": "frozen",
        "seeds": ABLATION_SEEDS,
    },
    "bnf_aug_all_balanced": {
        "train_root": DATA_DIR / "crops_aug_all_balanced",
        "strategy": "class_weights",
        "checkpoint_metric": "val_f1",
        "bn_mode": "frozen",
        "seeds": ABLATION_SEEDS,
    },
    # Diagnostik jumlah epoch (hanya dilaporkan; konfigurasi utama tidak diubah).
    "bnf_e60": {
        "train_root": CROPS_DIR,
        "strategy": "class_weights",
        "checkpoint_metric": "val_f1",
        "bn_mode": "frozen",
        "seeds": (42,),
        "epochs": 60,
    },
}

#: Root default hasil run.
RUNS_DIR: Path = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    """Parse argumen CLI."""
    parser = argparse.ArgumentParser(description="Eksperimen revisi multi-seed.")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--models", nargs="+", default=list(AVAILABLE_MODELS), choices=AVAILABLE_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Override seed semua konfigurasi.")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch (HANYA uji cepat).")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Cetak daftar run tanpa menjalankan.")
    parser.add_argument("--force", action="store_true", help="Jalankan ulang walau metrics.json ada.")
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover - lingkungan tanpa git
        return "unknown"


class _RunLog:
    """Handler file sementara untuk logger ``src`` + logger run."""

    def __init__(self, path: Path) -> None:
        self.handler = logging.FileHandler(path, mode="w", encoding="utf-8", errors="replace")
        self.handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        self.loggers = [logging.getLogger("src"), logging.getLogger("revision")]

    def __enter__(self) -> logging.Logger:
        for lg in self.loggers:
            lg.setLevel(logging.INFO)
            lg.addHandler(self.handler)
        return self.loggers[1]

    def __exit__(self, *exc) -> None:
        for lg in self.loggers:
            lg.removeHandler(self.handler)
        self.handler.close()


@torch.no_grad()
def _collect_logits(model: torch.nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    chunks = [model(images.to(DEVICE)).float().cpu().numpy() for images, _ in loader]
    return np.concatenate(chunks, axis=0)


def run_one(config_name: str, model_name: str, seed: int, args: argparse.Namespace) -> Path:
    """Latih + evaluasi satu run. Mengembalikan folder run."""
    spec = CONFIGS[config_name]
    run_dir = Path(args.runs_dir) / f"{model_name}_{config_name}_seed{seed}"
    if (run_dir / "metrics.json").exists() and not args.force:
        print(f"[lewati] {run_dir.name} (sudah selesai)")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    for old in run_dir.glob("*.pt"):
        old.unlink()  # sisa run yang terputus

    config = dict(CNN_CONFIG[model_name])
    if "epochs" in spec:
        config["epochs"] = int(spec["epochs"])
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    config.update(
        imbalance_strategy=spec["strategy"],
        checkpoint_metric=spec["checkpoint_metric"],
        bn_mode=spec["bn_mode"],
        seed=seed,
        config_name=config_name,
        train_root=str(Path(spec["train_root"]).relative_to(PROJECT_ROOT)),
    )
    size = int(config["input_size"])
    batch = int(config["batch_size"])

    with _RunLog(run_dir / "train.log") as log:
        start = time.perf_counter()
        log.info("Run %s | config=%s", run_dir.name, json.dumps(config, default=str))
        train_ds = CachedCropDataset(Path(spec["train_root"]) / "train", size)
        valid_ds = CachedCropDataset(CROPS_DIR / "valid", size)
        test_ds = CachedCropDataset(CROPS_DIR / "test", size)
        log.info("Distribusi train: %s", train_ds.class_counts())
        log.info("Komposisi train: %s", train_ds.augmentation_counts())
        for name, ds in (("valid", valid_ds), ("test", test_ds)):
            if ds.augmentation_counts()["augmented"]:
                raise RuntimeError(f"Split {name} memuat citra teraugmentasi.")

        # Urutan konsumsi RNG sama dengan scripts.train: seed -> sampler ->
        # loader -> model (inisialisasi head) -> pelatihan.
        set_seed(seed)
        sampler = None
        if spec["strategy"] == "weighted_sampler":
            sampler = create_weighted_sampler(train_ds.labels, num_classes=NUM_CLASSES, seed=seed)
        pin = DEVICE.startswith("cuda")
        train_loader = DataLoader(
            train_ds, batch_size=batch, shuffle=sampler is None, sampler=sampler, pin_memory=pin
        )
        valid_loader = DataLoader(valid_ds, batch_size=batch, shuffle=False, pin_memory=pin)
        test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, pin_memory=pin)

        model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=True)
        history = train_cnn(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            config=config,
            output_dir=run_dir,
            model_name=model_name,
            device=DEVICE,
            imbalance_strategy=spec["strategy"],
            checkpoint_metric=spec["checkpoint_metric"],
            bn_mode=spec["bn_mode"],
        )
        train_seconds = time.perf_counter() - start
        del model

        # Evaluasi checkpoint terbaik dari disk (seperti scripts.evaluate).
        best = Path(history["best_model_path"])
        model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        load_checkpoint(model, best, map_location=DEVICE)
        arrays: dict[str, np.ndarray] = {}
        evaluation: dict[str, dict] = {}
        for split, ds, loader in (("valid", valid_ds, valid_loader), ("test", test_ds, test_loader)):
            y_true = np.asarray(ds.labels, dtype=np.int64)
            arrays[f"{split}__y_true"] = y_true
            arrays[f"{split}__paths"] = np.asarray([p.name for p in ds.paths])
            evaluation[split] = {}
            for tta in TTA_CHOICES:
                logits = _collect_logits(wrap_tta(model, tta), loader)
                arrays[f"{split}__logits_{tta}"] = logits
                metrics = metrics_from_logits(y_true, logits)
                evaluation[split][tta] = metrics
                log.info("%s/%s: %s", split, tta, {k: round(v, 4) for k, v in flatten_metrics(metrics).items()})
        np.savez_compressed(run_dir / "predictions.npz", **arrays)

        report = {
            "run": run_dir.name,
            "config_name": config_name,
            "model": model_name,
            "seed": seed,
            "config": config,
            "checkpoint": best.name,
            "best_epoch": history["best_epoch"],
            "best_val_f1": history["best_val_f1"],
            "best_val_acc": history["best_val_acc"],
            "checkpoint_metric": history["checkpoint_metric"],
            "class_weights": history.get("class_weights"),
            "train_counts": train_ds.class_counts(),
            "train_composition": train_ds.augmentation_counts(),
            "train_seconds": round(train_seconds, 1),
            "evaluation": evaluation,
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "python": platform.python_version(),
                "git_commit": _git_commit(),
            },
            "class_names": CLASS_NAMES,
        }
        save_json(report, run_dir / "metrics.json")
        d4 = evaluation["test"]["d4"]
        log.info("Selesai %.0fs | best epoch %d", time.perf_counter() - start, history["best_epoch"])

    print(
        f"[selesai] {run_dir.name}: epoch {history['best_epoch']} | val_f1 {history['best_val_f1']:.4f} "
        f"| test d4 acc {d4['accuracy']:.4f} f1 {d4['f1']:.4f} miss {d4['defect_miss_rate']:.4f} "
        f"| {time.perf_counter() - start:.0f}s",
        flush=True,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_dir


def main() -> None:
    """Jalankan seluruh run yang diminta secara berurutan."""
    args = parse_args()
    plan = [
        (cfg, model, seed)
        for cfg in args.configs
        for seed in (args.seeds or CONFIGS[cfg]["seeds"])
        for model in args.models
    ]
    print(f"{len(plan)} run direncanakan -> {args.runs_dir}")
    if args.dry_run:
        for cfg, model, seed in plan:
            done = (Path(args.runs_dir) / f"{model}_{cfg}_seed{seed}" / "metrics.json").exists()
            print(f"  {'[x]' if done else '[ ]'} {model}_{cfg}_seed{seed}")
        return
    start = time.perf_counter()
    for i, (cfg, model, seed) in enumerate(plan, 1):
        print(f"--- [{i}/{len(plan)}] {model} | {cfg} | seed {seed}", flush=True)
        run_one(cfg, model, seed, args)
    print(f"Semua run selesai dalam {(time.perf_counter() - start) / 60:.1f} menit.")


if __name__ == "__main__":
    main()

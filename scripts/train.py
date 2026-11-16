"""CLI: latih satu arsitektur CNN untuk klasifikasi kualitas green bean.

Dataset diambil dari ``data/crops/`` — keluaran
``python -m scripts.crop_dataset`` lalu ``python -m scripts.augment_dataset``.
Hyperparameter default seragam antar-model (lihat ``configs/cnn_config.py``)
agar perbandingan arsitektur bersifat adil.

Augmentasi **tidak** dilakukan di sini: dataset di ``data/crops/`` sudah
teraugmentasi di disk, jadi transform saat pelatihan hanya resize +
normalisasi ImageNet.

Contoh::

    python -m scripts.train --model resnet50
    python -m scripts.train --model resnet50 --imbalance-strategy focal_loss
    python -m scripts.train --model mobilenetv3 --save-predictions --save-features
    python -m scripts.train --model inceptionv3 --epochs 2 --output-dir results/tmp
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # CLI: figur hanya disimpan ke file, tanpa jendela.

from torch.utils.data import DataLoader  # noqa: E402

from configs.base_config import (  # noqa: E402
    ANALYSIS_DIR,
    BN_MODE,
    BN_MODES,
    CHECKPOINT_METRIC,
    CHECKPOINT_METRICS,
    CLASS_NAMES,
    CLASS_WEIGHTS_STRATEGY,
    CROPS_DIR,
    DEVICE,
    FOCAL_LOSS_GAMMA,
    IMBALANCE_STRATEGIES,
    IMBALANCE_STRATEGY,
    LOGS_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
)
from configs.cnn_config import CNN_CONFIG
from src.analysis.feature_visualization import extract_features, save_features
from src.data.dataset import GreenBeanDataset
from src.data.transforms import get_transforms
from src.evaluation.predict import collect_predictions, save_predictions
from src.models.model_factory import AVAILABLE_MODELS, get_model
from src.training.imbalance import (
    WEIGHT_STRATEGIES,
    create_weighted_sampler,
    describe_imbalance,
)
from src.training.train_cnn import train_cnn
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse argumen CLI."""
    parser = argparse.ArgumentParser(
        description="Latih satu arsitektur CNN untuk klasifikasi kualitas green bean."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=AVAILABLE_MODELS,
        help="Arsitektur yang dilatih.",
    )
    parser.add_argument(
        "--crops-dir",
        default=None,
        help="Root dataset siap latih (default: data/crops).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Folder keluaran checkpoint & history "
            "(default: results/models/<model>). Pakai folder lain untuk "
            "eksperimen cepat agar hasil run utama tidak tertimpa."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="Worker DataLoader.")
    parser.add_argument(
        "--imbalance-strategy",
        choices=IMBALANCE_STRATEGIES,
        default=IMBALANCE_STRATEGY,
        help=f"Strategi penanganan imbalance (default: {IMBALANCE_STRATEGY}).",
    )
    parser.add_argument(
        "--class-weights-strategy",
        choices=WEIGHT_STRATEGIES,
        default=CLASS_WEIGHTS_STRATEGY,
        help=f"Skema perhitungan bobot kelas (default: {CLASS_WEIGHTS_STRATEGY}).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=FOCAL_LOSS_GAMMA,
        help=f"Gamma Focal Loss (default: {FOCAL_LOSS_GAMMA}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=(
            f"Seed pelatihan: inisialisasi head, urutan batch, sampler (default: "
            f"{RANDOM_SEED}). Dataset augmentasi di disk tidak terpengaruh."
        ),
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=CHECKPOINT_METRICS,
        default=CHECKPOINT_METRIC,
        help=f"Metrik valid untuk memilih checkpoint (default: {CHECKPOINT_METRIC}).",
    )
    parser.add_argument(
        "--bn-mode",
        choices=BN_MODES,
        default=BN_MODE,
        help=(
            "adapt = statistik BatchNorm ikut beradaptasi (default); "
            "frozen = BN tetap mode eval selama training (ablasi)."
        ),
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Alias untuk --imbalance-strategy none (kompatibilitas mundur).",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Simpan prediksi split test ke CSV untuk analisis lanjutan.",
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Simpan fitur penultimate split test ke .npz untuk t-SNE/UMAP.",
    )
    parser.add_argument(
        "--augment-on-the-fly",
        action="store_true",
        help=(
            "Tambahkan augmentasi saat pelatihan. HANYA untuk eksperimen "
            "pembanding; proyek ini memakai augmentasi offline."
        ),
    )
    return parser.parse_args()


def _build_loader(
    root: Path,
    model_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    sampler=None,
    augment: bool = False,
) -> DataLoader:
    """Bangun DataLoader untuk sebuah split.

    Args:
        root: Folder split (berisi subfolder kelas).
        model_name: Nama arsitektur (menentukan ukuran input).
        split: ``"train"``, ``"valid"``, atau ``"test"``.
        batch_size: Ukuran batch.
        num_workers: Jumlah worker DataLoader.
        sampler: Sampler custom (mis. ``WeightedRandomSampler``). Bila
            diberikan, ``shuffle`` dimatikan karena PyTorch melarang keduanya
            dipakai bersamaan.
        augment: Augmentasi on-the-fly (default ``False``).

    Returns:
        DataLoader siap pakai.
    """
    dataset = GreenBeanDataset(
        root=root, transform=get_transforms(model_name, split, augment=augment)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=DEVICE.startswith("cuda"),
    )


def main() -> None:
    """Jalankan pelatihan satu arsitektur CNN."""
    args = parse_args()

    set_seed(args.seed)
    logger = setup_logger(
        f"train_{args.model}",
        log_file=f"train_{args.model}.log",
        level=logging.INFO,
        attach_to=["src"],
        # Satu file = satu run. Salinan per eksperimen disimpan di output_dir
        # setelah pelatihan, jadi log run lain tidak hilang saat ditimpa.
        mode="w",
    )

    strategy = "none" if args.no_class_weights else args.imbalance_strategy

    # Susun config efektif (default + override CLI).
    config = dict(CNN_CONFIG[args.model])
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    config["imbalance_strategy"] = strategy
    config["class_weights_strategy"] = args.class_weights_strategy
    config["focal_gamma"] = args.focal_gamma
    config["seed"] = args.seed
    config["checkpoint_metric"] = args.checkpoint_metric
    config["bn_mode"] = args.bn_mode

    crops_dir = Path(args.crops_dir) if args.crops_dir else CROPS_DIR
    batch_size = int(config["batch_size"])

    if not (crops_dir / "train").is_dir():
        raise FileNotFoundError(
            f"Dataset siap latih belum ada di {crops_dir}. Jalankan lebih dulu:\n"
            "  python -m scripts.crop_dataset\n"
            "  python -m scripts.augment_dataset"
        )

    logger.info("Menyiapkan dataloader dari %s ...", crops_dir)

    # Sampler hanya dibuat setelah label train diketahui.
    probe = GreenBeanDataset(root=crops_dir / "train")
    sampler = None
    if strategy == "weighted_sampler":
        sampler = create_weighted_sampler(
            probe.labels, num_classes=NUM_CLASSES, seed=args.seed
        )
        logger.info("WeightedRandomSampler aktif (%d sampel per epoch).", len(sampler))

    train_loader = _build_loader(
        crops_dir / "train",
        args.model,
        "train",
        batch_size,
        args.num_workers,
        sampler=sampler,
        augment=args.augment_on_the_fly,
    )
    val_loader = _build_loader(
        crops_dir / "valid", args.model, "valid", batch_size, args.num_workers
    )

    train_counts = train_loader.dataset.class_counts()
    logger.info("Distribusi kelas (train): %s", train_counts)
    logger.info("Distribusi kelas (valid): %s", val_loader.dataset.class_counts())
    logger.info("Komposisi train: %s", train_loader.dataset.augmentation_counts())
    logger.info("Ketimpangan train: %s", describe_imbalance(train_counts))

    valid_aug = val_loader.dataset.augmentation_counts()
    if valid_aug["augmented"]:
        logger.warning(
            "Split valid memuat %d citra teraugmentasi; metrik validasi jadi "
            "tidak bersih. Periksa --augment-splits pada scripts.augment_dataset.",
            valid_aug["augmented"],
        )

    if args.augment_on_the_fly:
        logger.warning(
            "Augmentasi on-the-fly aktif DI ATAS dataset yang sudah "
            "diaugmentasi offline. Hasil run ini tidak sebanding dengan run "
            "standar; laporkan eksplisit bila dipakai."
        )

    logger.info("Membangun model %s (device=%s) ...", args.model, DEVICE)
    model = get_model(args.model, num_classes=NUM_CLASSES, pretrained=True)

    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR / args.model
    logger.info(
        "Mulai pelatihan (epochs=%s, strategi=%s) -> %s",
        config["epochs"],
        strategy,
        output_dir,
    )
    history = train_cnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        model_name=args.model,
        device=DEVICE,
        imbalance_strategy=strategy,
        checkpoint_metric=args.checkpoint_metric,
        bn_mode=args.bn_mode,
    )
    logger.info(
        "Selesai. %s=%.4f (epoch %s) | model=%s",
        args.checkpoint_metric,
        history["best_metric"],
        history.get("best_epoch"),
        history["best_model_path"],
    )

    # Simpan salinan log run ini di samping checkpoint & history-nya.
    log_path = LOGS_DIR / f"train_{args.model}.log"
    if log_path.exists():
        shutil.copy2(log_path, output_dir / f"{args.model}_train.log")

    print(f"\n=== {args.model} ===")
    print(f"strategi imbalance : {strategy}")
    print(f"kriteria checkpoint: {args.checkpoint_metric} | seed {args.seed} | BN {args.bn_mode}")
    print(
        f"epoch terpilih     : {history.get('best_epoch')} "
        f"(val_f1={history['best_val_f1']:.4f}, val_acc={history['best_val_acc']:.4f})"
    )
    print(f"checkpoint         : {history['best_model_path']}")

    if args.save_predictions or args.save_features:
        _save_analysis_artifacts(args, crops_dir, model, batch_size, logger)

    print("\nLangkah berikutnya - evaluasi di split test:")
    print(
        f"  python -m scripts.evaluate --model {args.model} "
        f"--checkpoint {history['best_model_path'] or '<path .pt>'}"
    )


def _save_analysis_artifacts(
    args: argparse.Namespace,
    crops_dir: Path,
    model,
    batch_size: int,
    logger: logging.Logger,
) -> None:
    """Simpan prediksi dan/atau fitur split test untuk analisis lanjutan.

    Dipanggil setelah pelatihan selesai, memakai bobot model terakhir di
    memori. Untuk memakai checkpoint terbaik, pakai ``scripts.evaluate``
    yang memuat ulang checkpoint dari disk.
    """
    test_dir = crops_dir / "test"
    if not test_dir.is_dir():
        logger.warning("Split test tidak ada di %s; artefak analisis dilewati.", test_dir)
        return

    test_loader = _build_loader(test_dir, args.model, "test", batch_size, args.num_workers)

    if args.save_predictions:
        frame = collect_predictions(model, test_loader, device=DEVICE, class_names=CLASS_NAMES)
        path = save_predictions(frame, ANALYSIS_DIR / f"{args.model}_test_predictions.csv")
        logger.info("Prediksi test tersimpan: %s (%d baris)", path, len(frame))
        print(f"prediksi test      : {path}")

    if args.save_features:
        features, labels = extract_features(
            model, test_loader, model_name=args.model, device=DEVICE
        )
        path = save_features(
            features,
            labels,
            ANALYSIS_DIR / f"{args.model}_test_features.npz",
            paths=[str(p) for p in test_loader.dataset.paths],
        )
        logger.info("Fitur test tersimpan: %s (shape=%s)", path, features.shape)
        print(f"fitur test         : {path} {features.shape}")


if __name__ == "__main__":
    main()

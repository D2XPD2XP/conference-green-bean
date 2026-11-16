"""Training loop generik untuk semua arsitektur CNN.

Hanya parameter dengan ``requires_grad = True`` (yaitu head klasifikasi) yang
dioptimasi, sesuai skema feature extraction. Strategi penanganan imbalance
dapat dipilih lewat ``imbalance_strategy``; history pelatihan disimpan ke JSON
agar bisa dibaca modul analisis (kurva pembelajaran, perbandingan konvergensi).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from configs.base_config import (
    BN_MODE,
    BN_MODES,
    CHECKPOINT_METRIC,
    CHECKPOINT_METRICS,
    CLASS_WEIGHTS_STRATEGY,
    DEVICE,
    FOCAL_LOSS_GAMMA,
    IMBALANCE_STRATEGY,
    NUM_CLASSES,
)
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.imbalance import build_criterion, compute_class_weights
from src.utils.io import save_json

logger = logging.getLogger(__name__)


def class_weights_from_loader(
    loader: DataLoader,
    num_classes: int = NUM_CLASSES,
    strategy: str = CLASS_WEIGHTS_STRATEGY,
) -> torch.Tensor | None:
    """Hitung bobot kelas dari distribusi dataset sebuah DataLoader.

    Bobot dihitung dari dataset yang **benar-benar dimuat** (yaitu sudah
    pasca-augmentasi offline), sehingga tidak terjadi koreksi ganda terhadap
    oversampling yang dilakukan di disk.

    Args:
        loader: DataLoader data latih. ``loader.dataset`` diharapkan menyediakan
            atribut ``labels`` (atau ``samples`` berisi pasangan
            ``(path, label)``).
        num_classes: Jumlah kelas.
        strategy: Skema perhitungan bobot (lihat
            :func:`src.training.imbalance.compute_class_weights`).

    Returns:
        Tensor bobot ``(num_classes,)``, atau ``None`` bila label tidak dapat
        dibaca dari dataset.
    """
    dataset = loader.dataset
    labels = getattr(dataset, "labels", None)
    if labels is None:
        samples = getattr(dataset, "samples", None)
        if samples is None:
            logger.warning(
                "Dataset tidak menyediakan 'labels' maupun 'samples'; "
                "bobot kelas dilewati."
            )
            return None
        labels = [label for _, label in samples]

    if len(labels) == 0:
        logger.warning("Dataset latih kosong; bobot kelas dilewati.")
        return None

    return compute_class_weights(labels, num_classes=num_classes, strategy=strategy)


def freeze_batchnorm_stats(model: nn.Module) -> int:
    """Set seluruh modul BatchNorm ke mode eval (statistik tidak diperbarui).

    Dipanggil **setelah** ``model.train()``. Hanya BatchNorm yang disentuh;
    modul lain (mis. Dropout pada classifier MobileNetV3) tetap mode train,
    sehingga satu-satunya perbedaan terhadap konfigurasi utama adalah
    ``running_mean/var``.

    Args:
        model: Model yang sedang dilatih.

    Returns:
        Jumlah modul BatchNorm yang dibekukan.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    desc: str,
    num_classes: int = NUM_CLASSES,
    bn_mode: str = BN_MODE,
) -> tuple[float, float, float]:
    """Jalankan satu epoch. Bila ``optimizer`` None -> mode evaluasi.

    Args:
        model: Model yang dilatih/dievaluasi.
        loader: DataLoader split terkait.
        criterion: Fungsi loss.
        optimizer: Optimizer, atau ``None`` untuk evaluasi.
        device: Device komputasi.
        desc: Label progress bar.
        num_classes: Jumlah kelas (untuk F1 makro).
        bn_mode: ``"adapt"`` atau ``"frozen"`` (hanya berpengaruh saat train).

    Returns:
        Tuple ``(rata_rata_loss, akurasi, f1_makro)``.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    if is_train and bn_mode == "frozen":
        freeze_batchnorm_stats(model)

    running_loss = 0.0
    correct = 0
    total = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.set_grad_enabled(is_train):
        # ascii=True: hindari UnicodeEncodeError pada konsol Windows (cp1252),
        # yang bisa mematikan pelatihan panjang di tengah jalan.
        for images, labels in tqdm(loader, desc=desc, leave=False, ascii=True):
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_true.extend(labels.tolist())
            all_pred.extend(preds.tolist())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    macro_f1 = 0.0
    if total:
        macro_f1 = float(
            f1_score(
                all_true,
                all_pred,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        )
    return avg_loss, accuracy, macro_f1


def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    output_dir: str | Path,
    model_name: str = "cnn",
    device: str = DEVICE,
    imbalance_strategy: str | None = None,
    class_weights: torch.Tensor | None = None,
    checkpoint_metric: str | None = None,
    bn_mode: str | None = None,
) -> dict:
    """Latih model CNN dan simpan model terbaik + history.

    Args:
        model: Model CNN (head dapat dilatih, backbone beku).
        train_loader: DataLoader data latih.
        val_loader: DataLoader data validasi.
        config: Dict hyperparameter (``epochs``, ``learning_rate``,
            ``num_classes``, ``imbalance_strategy``, dsb.), biasanya dari
            ``CNN_CONFIG[model_name]``.
        output_dir: Direktori penyimpanan checkpoint dan history.
        model_name: Nama model untuk penamaan file & logging.
        device: Device komputasi.
        imbalance_strategy: ``"none"``, ``"class_weights"``,
            ``"weighted_sampler"``, atau ``"focal_loss"``. Bila ``None``,
            diambil dari ``config["imbalance_strategy"]``, lalu dari
            ``configs.base_config.IMBALANCE_STRATEGY``.
        class_weights: Bobot kelas siap pakai. Bila ``None`` dan strategi
            membutuhkannya, bobot dihitung dari distribusi ``train_loader``.
        checkpoint_metric: ``"val_f1"`` (F1 makro valid, default) atau
            ``"val_acc"``. Bila ``None``, diambil dari
            ``config["checkpoint_metric"]`` lalu ``CHECKPOINT_METRIC``.
        bn_mode: ``"adapt"`` (default) atau ``"frozen"``. Bila ``None``,
            diambil dari ``config["bn_mode"]`` lalu ``BN_MODE``.

    Returns:
        Dict ``history`` berisi list per-epoch (``train_loss``, ``train_acc``,
        ``train_f1``, ``val_loss``, ``val_acc``, ``val_f1``) plus
        ``checkpoint_metric``, ``best_metric``, ``best_epoch``,
        ``best_val_acc`` & ``best_val_f1`` (nilai pada epoch terpilih),
        ``best_model_path``, ``bn_mode``, ``imbalance_strategy``,
        ``class_weights``, dan ``config`` efektif.

    Raises:
        ValueError: Bila ``imbalance_strategy``, ``checkpoint_metric``, atau
            ``bn_mode`` tidak dikenal.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    epochs = int(config.get("epochs", 30))
    lr = float(config.get("learning_rate", 1e-3))
    patience = int(config.get("patience", epochs))
    num_classes = int(config.get("num_classes", NUM_CLASSES))
    gamma = float(config.get("focal_gamma", FOCAL_LOSS_GAMMA))
    weights_strategy = str(config.get("class_weights_strategy", CLASS_WEIGHTS_STRATEGY))

    strategy = imbalance_strategy or str(
        config.get("imbalance_strategy", IMBALANCE_STRATEGY)
    )
    metric_name = checkpoint_metric or str(config.get("checkpoint_metric", CHECKPOINT_METRIC))
    if metric_name not in CHECKPOINT_METRICS:
        raise ValueError(
            f"checkpoint_metric '{metric_name}' tidak dikenal. "
            f"Pilihan: {list(CHECKPOINT_METRICS)}"
        )
    bn = bn_mode or str(config.get("bn_mode", BN_MODE))
    if bn not in BN_MODES:
        raise ValueError(f"bn_mode '{bn}' tidak dikenal. Pilihan: {list(BN_MODES)}")
    logger.info("Kriteria checkpoint: %s | mode BatchNorm: %s", metric_name, bn)

    # Bobot kelas hanya relevan untuk dua strategi; menghitungnya untuk yang
    # lain akan menyesatkan saat dibaca di history.
    if class_weights is None and strategy in {"class_weights", "focal_loss"}:
        class_weights = class_weights_from_loader(
            train_loader, num_classes=num_classes, strategy=weights_strategy
        )

    if class_weights is not None:
        logger.info(
            "Bobot kelas (%s): %s",
            weights_strategy,
            [round(w, 4) for w in class_weights.tolist()],
        )
        class_weights = class_weights.to(device)

    criterion = build_criterion(strategy, class_weights=class_weights, gamma=gamma)
    criterion = criterion.to(device)
    logger.info(
        "Strategi imbalance: %s | loss: %s", strategy, type(criterion).__name__
    )

    # Strategi weighted_sampler diterapkan di sisi DataLoader, bukan di sini.
    # Kalau sampler-nya lupa dipasang, pelatihan jalan tanpa penyeimbangan
    # apa pun tanpa gejala yang terlihat — jadi lebih baik diperingatkan.
    if strategy == "weighted_sampler" and not isinstance(
        getattr(train_loader, "sampler", None), WeightedRandomSampler
    ):
        logger.warning(
            "imbalance_strategy='weighted_sampler' tetapi train_loader tidak "
            "memakai WeightedRandomSampler; tidak ada penyeimbangan yang aktif."
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    metric_tag = "f1" if metric_name == "val_f1" else "acc"
    checkpointer = ModelCheckpoint(output_dir, model_name, mode="max", metric_tag=metric_tag)
    early_stopping = EarlyStopping(patience=patience, mode="max")

    history: dict = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            f"[{model_name}] train {epoch}/{epochs}",
            num_classes=num_classes,
            bn_mode=bn,
        )
        val_loss, val_acc, val_f1 = _run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            f"[{model_name}] valid {epoch}/{epochs}",
            num_classes=num_classes,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | "
            "val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_f1,
        )

        monitored = val_f1 if metric_name == "val_f1" else val_acc
        if checkpointer.maybe_save(model, monitored, epoch) is not None:
            best_epoch = epoch
        early_stopping.step(monitored)
        if early_stopping.should_stop:
            logger.info("Early stopping pada epoch %d (patience=%d).", epoch, patience)
            break

    history["checkpoint_metric"] = metric_name
    history["best_metric"] = checkpointer.best_value or 0.0
    history["best_epoch"] = best_epoch
    # Nilai pada epoch terpilih. Untuk kriteria val_acc, best_val_acc identik
    # dengan arti lamanya (val acc maksimum).
    history["best_val_acc"] = history["val_acc"][best_epoch - 1] if best_epoch else 0.0
    history["best_val_f1"] = history["val_f1"][best_epoch - 1] if best_epoch else 0.0
    history["bn_mode"] = bn
    history["best_model_path"] = str(checkpointer.best_path) if checkpointer.best_path else ""
    history["imbalance_strategy"] = strategy
    history["class_weights_strategy"] = weights_strategy if class_weights is not None else None
    history["class_weights"] = class_weights.tolist() if class_weights is not None else None
    history["epochs_ran"] = len(history["train_loss"])
    history["config"] = {
        key: value for key, value in config.items() if isinstance(value, (int, float, str, bool))
    }

    save_json(history, output_dir / f"{model_name}_history.json")
    return history

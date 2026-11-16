"""CLI: pilih ambang keputusan ``good`` pada VALID, terapkan ke TEST.

Tanpa melatih ulang. Membutuhkan checkpoint dan tabel prediksi test hasil
``scripts.evaluate`` (``<results-dir>/analysis/<model>_test_predictions.csv``).

Contoh::

    python -m scripts.tune_threshold --model resnet50 \\
        --checkpoint results/models/resnet50/<ckpt>.pt
    python -m scripts.tune_threshold --model resnet50 \\
        --checkpoint <ckpt>.pt --results-dir results/experiments/focal_loss
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # CLI: figur hanya disimpan ke file, tanpa jendela.

import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from configs.base_config import (  # noqa: E402
    ANALYSIS_DIR,
    CLASS_NAMES,
    CROPS_DIR,
    DEVICE,
    METRICS_DIR,
    NUM_CLASSES,
    TTA_MODE,
)
from configs.cnn_config import CNN_CONFIG  # noqa: E402
from src.data.dataset import GreenBeanDataset  # noqa: E402
from src.data.transforms import get_transforms  # noqa: E402
from src.evaluation.predict import collect_predictions, load_predictions  # noqa: E402
from src.evaluation.threshold import (  # noqa: E402
    probability_matrix,
    select_threshold,
    sweep_thresholds,
)
from src.evaluation.tta import TTA_CHOICES, wrap_tta  # noqa: E402
from src.models.model_factory import AVAILABLE_MODELS, get_model  # noqa: E402
from src.utils.io import load_checkpoint, save_json  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse argumen CLI."""
    parser = argparse.ArgumentParser(
        description="Pilih ambang P(good) pada valid, laporkan efeknya di test."
    )
    parser.add_argument("--model", required=True, choices=AVAILABLE_MODELS)
    parser.add_argument("--checkpoint", required=True, help="Path checkpoint (.pt).")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Root hasil (default: results/). Prediksi test dibaca dari <dir>/analysis.",
    )
    parser.add_argument("--crops-dir", default=None, help="Root dataset (default: data/crops).")
    parser.add_argument(
        "--tta",
        choices=TTA_CHOICES,
        default=TTA_MODE,
        help="TTA untuk prediksi valid; samakan dengan saat scripts.evaluate.",
    )
    parser.add_argument(
        "--max-f1-drop",
        type=float,
        default=0.01,
        help="Penurunan F1 makro valid yang ditoleransi demi miss rate lebih rendah.",
    )
    return parser.parse_args()


def main() -> None:
    """Sweep ambang pada valid, pilih, lalu terapkan ke test."""
    args = parse_args()
    set_seed()
    logger = setup_logger(
        f"threshold_{args.model}", log_file=f"threshold_{args.model}.log", mode="w"
    )

    root = Path(args.results_dir) if args.results_dir else None
    metrics_dir = root / "metrics" if root else METRICS_DIR
    analysis_dir = root / "analysis" if root else ANALYSIS_DIR
    crops_dir = Path(args.crops_dir) if args.crops_dir else CROPS_DIR

    test_path = analysis_dir / f"{args.model}_test_predictions.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} belum ada. Jalankan scripts.evaluate dulu.")

    model = get_model(args.model, num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    load_checkpoint(model, args.checkpoint, map_location=DEVICE)
    valid_ds = GreenBeanDataset(crops_dir / "valid", transform=get_transforms(args.model, "valid"))
    loader = DataLoader(
        valid_ds, batch_size=int(CNN_CONFIG[args.model]["batch_size"]), shuffle=False
    )
    # Valid diprediksi dengan TTA yang sama seperti tabel prediksi test.
    valid = collect_predictions(
        wrap_tta(model, args.tta), loader, device=DEVICE, class_names=CLASS_NAMES
    )
    test = load_predictions(test_path)

    valid_sweep = sweep_thresholds(valid["y_true"], probability_matrix(valid))
    chosen = select_threshold(valid_sweep, max_f1_drop=args.max_f1_drop)
    # Test HANYA dievaluasi pada argmax dan ambang terpilih (bukan di-sweep),
    # agar kurva test tidak terlihat saat memilih.
    test_sweep = sweep_thresholds(
        test["y_true"], probability_matrix(test), thresholds=sorted({0.0, chosen})
    )

    valid_sweep.to_csv(analysis_dir / f"{args.model}_threshold_sweep_valid.csv", encoding="utf-8")

    report = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "selection": f"min miss_rate valid dengan F1 >= F1 argmax - {args.max_f1_drop}",
        "threshold": chosen,
        "valid_argmax": valid_sweep.loc[0.0].to_dict(),
        "valid_threshold": valid_sweep.loc[chosen].to_dict(),
        "test_argmax": test_sweep.loc[0.0].to_dict(),
        "test_threshold": test_sweep.loc[chosen].to_dict(),
    }
    save_json(report, metrics_dir / f"{args.model}_threshold.json")
    logger.info("Ambang terpilih %s: %.2f", args.model, chosen)

    pd.set_option("display.width", 200)
    print(f"\n=== {args.model}: sweep ambang P(good) pada VALID ===")
    print(valid_sweep.round(4).to_string())
    print(f"\nAmbang terpilih (dari VALID): {chosen:.2f}")
    cols = ["acc", "f1", "miss_rate", "f1_insect_damage", "good_recall", "n_rejected_good"]
    summary = pd.DataFrame(
        {
            "valid argmax": valid_sweep.loc[0.0, cols],
            "valid ambang": valid_sweep.loc[chosen, cols],
            "test argmax": test_sweep.loc[0.0, cols],
            "test ambang": test_sweep.loc[chosen, cols],
        }
    )
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()

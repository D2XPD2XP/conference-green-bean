"""CLI: analisis pendukung revisi (Tugas 8, 9, 10).

Subperintah:

- ``photo-stats`` : jumlah biji per kelas per foto sumber (30 foto) + ringkasan
  min/median/maks per kelas. Hanya membaca label YOLO (CPU).
- ``timing``      : waktu per sampel 50 g (= satu foto test): crop otomatis
  dari koordinat bbox anotasi + klasifikasi seluruh crop (GPU, batch 32,
  dengan warm-up), tanpa TTA dan dengan TTA D4. **Jalankan saat GPU tidak
  dipakai proses lain.** Waktu deteksi biji tidak termasuk karena pipeline
  ini tidak memiliki tahap deteksi (crop berasal dari anotasi).
- ``support``     : bootstrap CI95 per foto (F1 makro), silhouette fitur,
  confidence benar vs salah, jumlah insect->good, dan efisiensi (parameter,
  ukuran, MAC, latensi 1 & 8 view). Model representatif = seed 42
  konfigurasi utama; variasi lintas seed ikut dilaporkan bila relevan.

Hasil ditulis ke ``runs/summary/`` dan digabung ke ``revision_results.json``.
Dengan ``--main-config bn_frozen`` (konfigurasi utama sejak 2026-11-13) hasil
ditulis ke ``runs/summary/bnfrozen/`` dan kunci JSON ``timing_bnfrozen`` /
``support_bnfrozen``; varian TTA = pilihan valid dari ``aggregate_bnfrozen``
(jalankan ``scripts.aggregate_revision --main-config bn_frozen`` lebih dulu).

Contoh::

    python -m scripts.revision_extras photo-stats
    python -m scripts.revision_extras timing
    python -m scripts.revision_extras support
    python -m scripts.revision_extras timing --main-config bn_frozen
    python -m scripts.revision_extras support --main-config bn_frozen
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from configs.base_config import (  # noqa: E402
    ANNOTATED_DATA_DIR,
    CLASS_NAMES,
    CROPS_DIR,
    CROPS_RAW_DIR,
    DEVICE,
    GOOD_CLASS,
    NUM_CLASSES,
    PROJECT_ROOT,
    SPLITS,
)
from configs.cnn_config import CNN_CONFIG  # noqa: E402
from src.analysis.efficiency_analysis import measure_model_efficiency  # noqa: E402
from src.analysis.feature_visualization import (  # noqa: E402
    cluster_separation_score,
    extract_features,
    save_features,
)
from src.data.cache import CachedCropDataset  # noqa: E402
from src.data.crop_from_annotation import yolo_to_pixel_bbox  # noqa: E402
from src.data.transforms import get_transforms  # noqa: E402
from src.evaluation.efficiency import measure_inference_time  # noqa: E402
from src.evaluation.revision import (  # noqa: E402
    bootstrap_photo_ci,
    load_run_predictions,
    mean_std,
    photo_id_from_path,
    softmax,
)
from src.evaluation.tta import wrap_tta  # noqa: E402
from src.models.model_factory import get_model  # noqa: E402
from src.utils.io import load_checkpoint, load_yolo_annotations  # noqa: E402

MODELS = ["resnet50", "mobilenetv3", "inceptionv3"]
SEEDS = (42, 0, 1, 2, 3)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
RUNS_DIR = PROJECT_ROOT / "runs"
SUMMARY_DIR = RUNS_DIR / "summary"
RESULTS_JSON = PROJECT_ROOT / "revision_results.json"

#: Profil konfigurasi utama: (folder ringkasan, akhiran kunci JSON, kunci agregat).
PROFILES: dict[str, tuple[Path, str, str]] = {
    "main": (RUNS_DIR / "summary", "", "aggregate"),
    "bn_frozen": (RUNS_DIR / "summary" / "bnfrozen", "_bnfrozen", "aggregate_bnfrozen"),
}
#: Diset di ``main()`` sesuai ``--main-config``.
CONFIG = "main"
JSON_SUFFIX = ""
SEL_TTA = "d4"


def _merge_json(key: str, value: dict) -> None:
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8")) if RESULTS_JSON.exists() else {}
    data[key] = value
    RESULTS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def _run_dir(model: str, seed: int = 42, config: str | None = None) -> Path:
    return RUNS_DIR / f"{model}_{config or CONFIG}_seed{seed}"


def _load_model(model: str, seed: int = 42) -> torch.nn.Module:
    ckpt = next(_run_dir(model, seed).glob("*.pt"))
    net = get_model(model, num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    load_checkpoint(net, ckpt, map_location=DEVICE)
    return net.eval()


def _photos() -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        image_dir = ANNOTATED_DATA_DIR / split / "images"
        label_dir = ANNOTATED_DATA_DIR / split / "labels"
        for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                continue
            boxes = load_yolo_annotations(label)
            counts = Counter(CLASS_NAMES[b[0]] for b in boxes)
            rows.append(
                {
                    "photo_id": image.name.split("_jpg")[0],
                    "split": split,
                    "annotation": "full" if counts.get(GOOD_CLASS, 0) else "defect-only",
                    **{name: counts.get(name, 0) for name in CLASS_NAMES},
                    "total": len(boxes),
                    "image_path": image,
                    "label_path": label,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tugas 8
# --------------------------------------------------------------------------- #
def photo_stats() -> None:
    """Tabel jumlah biji per kelas per foto + ringkasan."""
    photos = _photos()
    # Verifikasi silang dengan crop mentah di disk.
    crop_counts = Counter()
    for split in SPLITS:
        for path in (CROPS_RAW_DIR / split).glob("*/*.jpg"):
            crop_counts[photo_id_from_path(path)] += 1
    photos["crops_on_disk"] = photos["photo_id"].map(crop_counts).fillna(0).astype(int)
    mismatch = photos[photos["crops_on_disk"] != photos["total"]]

    table = photos.drop(columns=["image_path", "label_path"]).sort_values(["split", "photo_id"])
    order = {"train": 0, "valid": 1, "test": 2}
    table = table.sort_values(["split", "photo_id"], key=lambda s: s.map(order) if s.name == "split" else s)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(SUMMARY_DIR / "photo_stats.csv", index=False, encoding="utf-8")

    def summarize(frame: pd.DataFrame, scope: str) -> list[dict]:
        out = []
        for name in [*CLASS_NAMES, "total"]:
            col = frame[name]
            out.append({"scope": scope, "class": name, "n_photos": len(frame), "min": int(col.min()),
                        "median": float(col.median()), "max": int(col.max()), "sum": int(col.sum())})
        return out

    summary = pd.DataFrame(
        summarize(table, "semua foto")
        + summarize(table[table.annotation == "full"], "foto full")
        + summarize(table[table.annotation == "defect-only"], "foto defect-only")
    )
    summary.to_csv(SUMMARY_DIR / "photo_stats_summary.csv", index=False, encoding="utf-8")
    split_totals = table.groupby("split")[[*CLASS_NAMES, "total"]].sum().loc[["train", "valid", "test"]]
    split_modes = table.groupby(["split", "annotation"]).size().unstack(fill_value=0)

    lines = ["| Foto | Split | Anotasi | " + " | ".join(CLASS_NAMES) + " | Total |",
             "|---|---|---|" + "--:|" * (len(CLASS_NAMES) + 1)]
    for r in table.itertuples():
        lines.append(f"| {r.photo_id} | {r.split} | {r.annotation} | "
                     + " | ".join(str(getattr(r, c)) for c in CLASS_NAMES) + f" | {r.total} |")
    lines += ["", "| Cakupan | Kelas | Min | Median | Maks | Jumlah |", "|---|---|--:|--:|--:|--:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.scope} (n={r.n_photos}) | {r[2]} | {r.min} | {r.median:g} | {r.max} | {r.sum} |")
    (SUMMARY_DIR / "photo_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _merge_json("photo_stats", {
        "n_photos": int(len(table)),
        "per_photo": table.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "split_totals": split_totals.reset_index().to_dict(orient="records"),
        "split_annotation_modes": split_modes.reset_index().to_dict(orient="records"),
        "crop_count_mismatches": mismatch[["photo_id", "total", "crops_on_disk"]].to_dict(orient="records"),
    })
    print(f"{len(table)} foto | total biji {int(table.total.sum())} | selisih crop vs label: {len(mismatch)} foto")
    print(split_totals.to_string())
    print(split_modes.to_string())


# --------------------------------------------------------------------------- #
# Tugas 9
# --------------------------------------------------------------------------- #
def _crop_photo(image_path: Path, label_path: Path) -> list[np.ndarray]:
    """Baca foto + label, potong semua bbox di memori (seperti crop_from_annotation)."""
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    crops = []
    for _, x_c, y_c, bw, bh in load_yolo_annotations(label_path):
        x1, y1, x2, y2 = yolo_to_pixel_bbox(x_c, y_c, bw, bh, w, h, padding_ratio=0.0)
        crop = image[y1:y2, x1:x2]
        if crop.size:
            crops.append(crop.copy())
    return crops


def _sync() -> None:
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def _classify(net: torch.nn.Module, crops: list[np.ndarray], transform, batch: int) -> tuple[float, float]:
    """Pra-proses + forward seluruh crop; kembalikan (ms pra-proses, ms forward)."""
    t0 = time.perf_counter()
    tensors = [transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops]
    t1 = time.perf_counter()
    for i in range(0, len(tensors), batch):
        net(torch.stack(tensors[i:i + batch]).to(DEVICE, non_blocking=True)).argmax(1).cpu()
    _sync()
    t2 = time.perf_counter()
    return (t1 - t0) * 1000, (t2 - t1) * 1000


def timing(repeats: int = 5, batch: int = 32) -> None:
    """Waktu per foto test: crop + klasifikasi (tanpa TTA & D4)."""
    photos = _photos()
    test = photos[photos.split == "test"].reset_index(drop=True)
    rows = []
    crops_cache = {}
    for r in test.itertuples():
        _crop_photo(r.image_path, r.label_path)  # warm-up cache disk
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            crops = _crop_photo(r.image_path, r.label_path)
            times.append((time.perf_counter() - t0) * 1000)
        crops_cache[r.photo_id] = crops
        rows.append({"photo_id": r.photo_id, "annotation": r.annotation, "n_beans": len(crops),
                     "crop_ms": statistics.mean(times), "crop_ms_std": statistics.stdev(times)})
    frame = pd.DataFrame(rows).set_index("photo_id")

    for model in MODELS:
        net = _load_model(model)
        transform = get_transforms(model, "test")
        for tta in ("none", SEL_TTA):
            wrapped = wrap_tta(net, tta)
            warm = crops_cache[frame.index[0]]
            for _ in range(3):
                _classify(wrapped, warm, transform, batch)
            for pid in frame.index:
                pre, fwd = [], []
                for _ in range(repeats):
                    a, b = _classify(wrapped, crops_cache[pid], transform, batch)
                    pre.append(a)
                    fwd.append(b)
                frame.loc[pid, f"{model}_{tta}_preprocess_ms"] = statistics.mean(pre)
                frame.loc[pid, f"{model}_{tta}_forward_ms"] = statistics.mean(fwd)
                frame.loc[pid, f"{model}_{tta}_total_ms"] = (
                    frame.loc[pid, "crop_ms"] + statistics.mean(pre) + statistics.mean(fwd)
                )
        del net
        torch.cuda.empty_cache()

    frame.to_csv(SUMMARY_DIR / "timing_per_photo.csv", encoding="utf-8")

    def agg(sub: pd.DataFrame) -> dict:
        out = {"n_photos": int(len(sub)), "beans_per_photo_mean": float(sub.n_beans.mean()),
               "crop_ms_mean": float(sub.crop_ms.mean())}
        for model in MODELS:
            for tta in ("none", SEL_TTA):
                for part in ("preprocess", "forward", "total"):
                    out[f"{model}_{tta}_{part}_ms_mean"] = float(sub[f"{model}_{tta}_{part}_ms"].mean())
                per_bean = (sub[f"{model}_{tta}_total_ms"] / sub.n_beans).mean()
                out[f"{model}_{tta}_total_ms_per_bean"] = float(per_bean)
        return out

    summary = {"all_test_photos": agg(frame), "full_photos": agg(frame[frame.annotation == "full"]),
               "defect_only_photos": agg(frame[frame.annotation == "defect-only"])}
    env = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "batch_size": batch,
        "repeats": repeats,
        "tta": SEL_TTA,
        "checkpoints": f"runs/<model>_{CONFIG}_seed42 (arsitektur sama; bobot tidak memengaruhi waktu)",
        "warmup": "1 baca foto + 3 klasifikasi penuh per model/TTA sebelum diukur",
        "crop_definition": "cv2.imread foto penuh + parsing label YOLO + potong semua bbox di memori "
                           "(tanpa menulis ke disk). Tanpa deteksi: koordinat dari anotasi.",
        "classify_definition": "BGR->RGB, Resize, ToTensor, Normalize per crop (CPU) + forward batch di GPU "
                               "sampai argmax tersalin ke CPU",
    }

    lab = SEL_TTA.upper() if SEL_TTA == "d4" else SEL_TTA
    lines = [f"| Cakupan | Foto | Biji/foto | Crop (ms) | Model | Total tanpa TTA (ms) | Total {lab} (ms) | "
             f"Forward tanpa TTA (ms) | Forward {lab} (ms) | ms/biji tanpa TTA | ms/biji {lab} |",
             "|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|"]
    for scope, label in (("all_test_photos", "semua foto test"), ("full_photos", "foto full"),
                         ("defect_only_photos", "foto defect-only")):
        s = summary[scope]
        for model in MODELS:
            lines.append(
                f"| {label} | {s['n_photos']} | {s['beans_per_photo_mean']:.0f} | {s['crop_ms_mean']:.0f} | {model} | "
                f"{s[f'{model}_none_total_ms_mean']:.0f} | {s[f'{model}_{SEL_TTA}_total_ms_mean']:.0f} | "
                f"{s[f'{model}_none_forward_ms_mean']:.0f} | {s[f'{model}_{SEL_TTA}_forward_ms_mean']:.0f} | "
                f"{s[f'{model}_none_total_ms_per_bean']:.2f} | {s[f'{model}_{SEL_TTA}_total_ms_per_bean']:.2f} |"
            )
    # Rasio kecepatan dihitung dari angka yang DILAPORKAN (detik, 2 desimal)
    # agar konsisten dengan teks; rasio dari ms mentah ikut disimpan.
    full = summary["full_photos"]
    ratios = {}
    for tta in ("none", SEL_TTA):
        r_ms = full[f"resnet50_{tta}_total_ms_mean"]
        m_ms = full[f"mobilenetv3_{tta}_total_ms_mean"]
        ratios[tta] = {"raw_ms": r_ms / m_ms, "from_reported_s": round(r_ms / 1000, 2) / round(m_ms / 1000, 2)}
    summary["speed_ratio_resnet_over_mobilenet_full"] = ratios
    lines += ["", "Rasio waktu ResNet-50 / MobileNetV3-L (foto full): " + "; ".join(
        f"{'tanpa TTA' if k == 'none' else lab}: {v['from_reported_s']:.2f}x dari detik dilaporkan "
        f"({v['raw_ms']:.3f}x dari ms mentah)" for k, v in ratios.items())]
    (SUMMARY_DIR / "timing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _merge_json("timing" + JSON_SUFFIX, {"environment": env, "summary": summary,
                           "per_photo": frame.reset_index().to_dict(orient="records")})
    print(frame.round(1).to_string())


# --------------------------------------------------------------------------- #
# Tugas 10
# --------------------------------------------------------------------------- #
def support(n_boot: int = 1000, latency_repeats: int = 5) -> None:
    """Bootstrap, silhouette, confidence, insect->good, efisiensi."""
    out: dict = {"config": CONFIG, "tta": SEL_TTA, "representative_seed": 42, "bootstrap": {},
                 "silhouette": {}, "confidence": {}, "insect_to_good": {}, "efficiency": {}}
    for model in MODELS:
        preds = {s: load_run_predictions(_run_dir(model, s) / "predictions.npz") for s in SEEDS}
        test42 = preds[42]["test"]
        photos = [photo_id_from_path(p) for p in test42["paths"]]
        out["bootstrap"][model] = {}
        for tta in ("none", SEL_TTA):
            y_pred = test42["logits"][tta].argmax(1)
            out["bootstrap"][model][tta] = bootstrap_photo_ci(test42["y_true"], y_pred, photos, n_boot=n_boot)

        # Confidence benar vs salah (TTA terpilih), seed 42 + lintas seed.
        conf = {}
        for s in SEEDS:
            probs = softmax(preds[s]["test"]["logits"][SEL_TTA].astype(np.float64))
            correct = probs.argmax(1) == preds[s]["test"]["y_true"]
            conf[s] = (float(probs.max(1)[correct].mean()), float(probs.max(1)[~correct].mean()))
        out["confidence"][model] = {
            "seed42_correct": conf[42][0], "seed42_incorrect": conf[42][1],
            "seed42_gap": conf[42][0] - conf[42][1],
            "seeds_correct": mean_std([c[0] for c in conf.values()]),
            "seeds_incorrect": mean_std([c[1] for c in conf.values()]),
        }

        insect, good = CLASS_NAMES.index("insect_damage"), CLASS_NAMES.index(GOOD_CLASS)
        counts = {tta: [int(((preds[s]["test"]["y_true"] == insect)
                             & (preds[s]["test"]["logits"][tta].argmax(1) == good)).sum()) for s in SEEDS]
                  for tta in ("none", SEL_TTA)}
        out["insect_to_good"][model] = {
            "n_insect_test": int((test42["y_true"] == insect).sum()),
            "seed42_none": counts["none"][0], f"seed42_{SEL_TTA}": counts[SEL_TTA][0],
            "seeds_none": mean_std(counts["none"]), f"seeds_{SEL_TTA}": mean_std(counts[SEL_TTA]),
            "per_seed_none": counts["none"], f"per_seed_{SEL_TTA}": counts[SEL_TTA],
        }

        # Silhouette fitur penultimate (test, tanpa TTA) per seed; fitur seed 42 disimpan untuk t-SNE.
        size = int(CNN_CONFIG[model]["input_size"])
        loader = DataLoader(CachedCropDataset(CROPS_DIR / "test", size), batch_size=32, shuffle=False)
        sil = {}
        for s in SEEDS:
            net = _load_model(model, s)
            feats, labels = extract_features(net, loader, model_name=model, device=DEVICE)
            sil[s] = cluster_separation_score(feats, labels)
            if s == 42:
                save_features(feats, labels, SUMMARY_DIR / f"{model}_seed42_test_features.npz",
                              paths=[str(p) for p in loader.dataset.paths])
                out["efficiency"][model] = measure_model_efficiency(
                    net, input_shape=(1, 3, size, size), device=DEVICE,
                    checkpoint_path=next(_run_dir(model, 42).glob("*.pt")),
                )
                # Latensi diulang beberapa kali (masing-masing 100 forward) agar
                # variasi antar-pengukuran ikut dilaporkan.
                shape = (1, 3, size, size)
                one = [measure_inference_time(net, input_shape=shape, device=DEVICE) for _ in range(latency_repeats)]
                multi = [measure_inference_time(wrap_tta(net, SEL_TTA), input_shape=shape, device=DEVICE)
                         for _ in range(latency_repeats)]
                eff = out["efficiency"][model]
                eff["inference_time_ms"], eff["inference_time_ms_std"] = mean_std(one)
                eff["inference_time_ms_tta"], eff["inference_time_ms_tta_std"] = mean_std(multi)
                eff["inference_time_ms_runs"], eff["inference_time_ms_tta_runs"] = one, multi
                eff["latency_protocol"] = (f"batch 1, {DEVICE}, {latency_repeats} pengukuran x 100 forward "
                                           "(warm-up 10 per pengukuran)")
            del net
            torch.cuda.empty_cache()
        out["silhouette"][model] = {"seed42": sil[42], "seeds": mean_std(list(sil.values())),
                                    "per_seed": {str(k): v for k, v in sil.items()}}

    lab = "D4" if SEL_TTA == "d4" else SEL_TTA
    lines = [f"| Model | CI95 F1 ({lab}) | CI95 F1 (tanpa TTA) | Silhouette seed 42 (5 seed) | "
             f"Conf benar / salah (seed 42) | I→G tanpa → {lab} (seed 42) | I→G tanpa → {lab} (5 seed) |",
             "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        b, c, ig, si = out["bootstrap"][m], out["confidence"][m], out["insect_to_good"][m], out["silhouette"][m]
        lines.append(
            f"| {m} | {b[SEL_TTA]['point']:.3f} [{b[SEL_TTA]['lo']:.3f}, {b[SEL_TTA]['hi']:.3f}] | "
            f"{b['none']['point']:.3f} [{b['none']['lo']:.3f}, {b['none']['hi']:.3f}] | "
            f"{si['seed42']:.3f} ({si['seeds'][0]:.3f} ± {si['seeds'][1]:.3f}) | "
            f"{c['seed42_correct']:.3f} / {c['seed42_incorrect']:.3f} | "
            f"{ig['seed42_none']} → {ig[f'seed42_{SEL_TTA}']} / {ig['n_insect_test']} | "
            f"{ig['seeds_none'][0]:.1f} ± {ig['seeds_none'][1]:.1f} → "
            f"{ig[f'seeds_{SEL_TTA}'][0]:.1f} ± {ig[f'seeds_{SEL_TTA}'][1]:.1f} |"
        )
    lines += ["", f"| Model | Parameter total | Trainable | Ukuran (MB) | MAC (G) | Latensi 1 view (ms) | "
              f"Latensi TTA {lab} (ms) |", "|---|--:|--:|--:|--:|--:|--:|"]
    for m in MODELS:
        e = out["efficiency"][m]
        lines.append(f"| {m} | {e['params_total']:,} | {e['params_trainable']:,} | {e['model_size_mb']:.1f} | "
                     f"{e['flops'] / 1e9:.2f} | {e['inference_time_ms']:.3f} ± {e.get('inference_time_ms_std', 0):.3f} | "
                     f"{e['inference_time_ms_tta']:.3f} ± {e.get('inference_time_ms_tta_std', 0):.3f} |")
    lines += ["", f"Latensi: {out['efficiency'][MODELS[0]].get('latency_protocol', 'batch 1')}; "
              "mean ± std antar-pengukuran."]
    (SUMMARY_DIR / "support.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _merge_json("support" + JSON_SUFFIX, out)
    print(f"support selesai; lihat {SUMMARY_DIR / 'support.md'}")


def main() -> None:
    """Dispatcher subperintah."""
    parser = argparse.ArgumentParser(description="Analisis pendukung revisi.")
    parser.add_argument("task", choices=["photo-stats", "timing", "support"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--main-config", default="main", choices=list(PROFILES))
    args = parser.parse_args()
    global CONFIG, SUMMARY_DIR, JSON_SUFFIX, SEL_TTA
    CONFIG = args.main_config
    SUMMARY_DIR, JSON_SUFFIX, agg_key = PROFILES[CONFIG]
    if args.task != "photo-stats" and RESULTS_JSON.exists():
        agg = json.loads(RESULTS_JSON.read_text(encoding="utf-8")).get(agg_key, {})
        SEL_TTA = agg.get("tta_selection", {}).get("uniform_choice", "d4")
    print(f"Profil {CONFIG}: TTA {SEL_TTA}, keluaran {SUMMARY_DIR}")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    if args.task == "photo-stats":
        photo_stats()
    elif args.task == "timing":
        timing(repeats=args.repeats, batch=args.batch_size)
    else:
        support(n_boot=args.n_boot, latency_repeats=args.latency_repeats)


if __name__ == "__main__":
    main()

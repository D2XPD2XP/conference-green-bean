"""CLI: agregasi seluruh run revisi -> tabel mean ± std + revision_results.json.

Membaca ``runs/*/metrics.json`` dan ``runs/*/predictions.npz`` (keluaran
``scripts.run_revision``), lalu menghasilkan (per *profil* konfigurasi utama):

- ``<summary>/per_run.csv``  : satu baris per (run, split, TTA)
- ``<summary>/tables.md``    : tabel siap tempel ke laporan
- ``revision_results.json``  : seluruh angka mentah (kunci per profil)

Profil (``--main-config``):

- ``main``      : konfigurasi utama revisi 2026-11-12 (BN beradaptasi).
  Keluaran ``runs/summary/`` dan kunci JSON ``aggregate``.
- ``bn_frozen`` : konfigurasi utama sejak brief lanjutan 2026-11-13 (BN beku
  total). Keluaran ``runs/summary/bnfrozen/`` dan kunci ``aggregate_bnfrozen``.
  Di laporan, ``bn_frozen`` disebut **main** dan ``main`` lama **bn_adapt**.

Aturan pemilihan (semuanya dari VALID, test hanya pelaporan):

- Varian TTA: F1 makro valid rata-rata lintas 5 seed konfigurasi utama,
  dipilih **seragam** untuk ketiga arsitektur (rata-rata lintas model).
  Varian terpilih dipakai untuk SEMUA tabel berikutnya.
- Ambang P(good): sweep 0,30-0,90 (langkah 0,05) pada valid, kurva
  dirata-ratakan lintas seed per model; dipilih miss rate valid terendah
  dengan F1 makro valid turun <= 0,01 dari argmax. Ambang terpilih lalu
  diterapkan ke test tiap seed.

Contoh::

    python -m scripts.aggregate_revision                         # profil main
    python -m scripts.aggregate_revision --main-config bn_frozen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs.base_config import CLASS_NAMES, PROJECT_ROOT
from src.evaluation.revision import (
    flatten_metrics,
    fmt_mean_std,
    load_run_predictions,
    mean_std,
    metrics_from_logits,
    softmax,
)
from src.evaluation.threshold import DEFAULT_THRESHOLDS, select_threshold, sweep_thresholds
from src.evaluation.tta import TTA_CHOICES

MODELS = ["resnet50", "mobilenetv3", "inceptionv3"]
MODEL_LABELS = {"resnet50": "ResNet-50", "mobilenetv3": "MobileNetV3-L", "inceptionv3": "InceptionV3"}
TTA_LABELS = {"none": "tanpa", "hflip": "flip H", "flip_hv": "flip H+V", "d4": "D4"}
CORE_METRICS = ["accuracy", "precision", "recall", "f1", "defect_recall", "defect_miss_rate"]
PAIRED_SEEDS = (42, 0, 1)
ABBR = {"good": "G", "black": "B", "chipped": "C", "dry_pods": "D", "insect_damage": "I"}

#: Profil konfigurasi utama. Baris ablasi = (label, config, "sel"|"none", seed|None);
#: "sel" = varian TTA terpilih di valid.
PROFILES: dict[str, dict] = {
    "main": {
        "summary_dir": "summary",
        "json_key": "aggregate",
        "main_label": "main",
        "ablation": [
            ("main (class weights, TTA {tta})", "main", "sel", None),
            ("main [seed 42,0,1]", "main", "sel", PAIRED_SEEDS),
            ("main tanpa TTA", "main", "none", None),
            ("focal loss (γ=2)", "focal_loss", "sel", None),
            ("weighted sampler", "weighted_sampler", "sel", None),
            ("aug_all_x3 (good ×3)", "aug_all_x3", "sel", None),
            ("aug_all_balanced (good diganti varian)", "aug_all_balanced", "sel", None),
            ("bn_frozen (BN eval)", "bn_frozen", "sel", None),
        ],
        "compare_config": None,
        "epoch_diag": None,
    },
    "bn_frozen": {
        "summary_dir": "summary/bnfrozen",
        "json_key": "aggregate_bnfrozen",
        "main_label": "main (bn_frozen)",
        "ablation": [
            ("main (bn_frozen, class weights, TTA {tta})", "bn_frozen", "sel", None),
            ("main [seed 42,0,1]", "bn_frozen", "sel", PAIRED_SEEDS),
            ("main tanpa TTA", "bn_frozen", "none", None),
            ("bn_adapt (BN train mode; main lama)", "main", "sel", None),
            ("focal loss (γ=2, α=bobot kelas)", "bnf_focal", "sel", None),
            ("weighted sampler", "bnf_sampler", "sel", None),
            ("aug_all_x3 (good ×3)", "bnf_aug_all_x3", "sel", None),
            ("aug_all_balanced (good diganti varian)", "bnf_aug_all_balanced", "sel", None),
        ],
        # Pembanding berpasangan R1.7: bn_adapt (= config "main", 5 seed).
        "compare_config": ("main", "bn_adapt"),
        "epoch_diag": ("bnf_e60", 42),
    },
}


def parse_args() -> argparse.Namespace:
    """Parse argumen CLI."""
    parser = argparse.ArgumentParser(description="Agregasi run revisi.")
    parser.add_argument("--main-config", default="main", choices=list(PROFILES))
    parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--output-json", default=str(PROJECT_ROOT / "revision_results.json"))
    parser.add_argument("--max-f1-drop", type=float, default=0.01)
    return parser.parse_args()


def load_runs(runs_dir: Path) -> tuple[list[dict], pd.DataFrame]:
    """Muat seluruh metrics.json menjadi list report + tabel panjang."""
    reports, rows = [], []
    for path in sorted(runs_dir.glob("*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["_dir"] = path.parent
        reports.append(report)
        for split, by_tta in report["evaluation"].items():
            for tta, metrics in by_tta.items():
                rows.append(
                    {
                        "config": report["config_name"],
                        "model": report["model"],
                        "seed": report["seed"],
                        "split": split,
                        "tta": tta,
                        "best_epoch": report["best_epoch"],
                        **flatten_metrics(metrics),
                    }
                )
    return reports, pd.DataFrame(rows)


def _sel(df: pd.DataFrame, **kw) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for key, value in kw.items():
        mask &= df[key].isin(value if isinstance(value, (list, tuple, set)) else [value]).to_numpy()
    return df[mask]


def _cell(values, metric: str) -> str:
    if metric == "defect_miss_rate":
        return fmt_mean_std(values, digits=1, scale=100.0) + "%"
    return fmt_mean_std(values, digits=3)


def _seed_rank(seed: int) -> int:
    """Urutan tampilan seed: 42 dulu (seed proposal), lalu 0, 1, 2, 3."""
    return -1 if seed == 42 else int(seed)


def _stats(values) -> dict:
    values = list(values)
    mean, std = mean_std(values)
    return {"mean": mean, "std": std, "n": len(values), "values": [float(v) for v in values]}


def main_table(df: pd.DataFrame, cfg: str, tta: str, split: str = "test") -> tuple[str, dict]:
    """Tabel utama mean ± std untuk satu konfigurasi & varian TTA."""
    metric_names = {
        "accuracy": "Acc",
        "precision": "Macro P",
        "recall": "Macro R",
        "f1": "Macro F1",
        "defect_recall": "Defect recall",
        "defect_miss_rate": "Miss rate",
    }
    lines = ["| Model | n | " + " | ".join(metric_names.values()) + " |",
             "|---|--:|" + "--:|" * len(metric_names)]
    raw: dict = {}
    for model in MODELS:
        sub = _sel(df, config=cfg, split=split, tta=tta, model=model).sort_values("seed")
        raw[model] = {m: _stats(sub[m]) for m in [*CORE_METRICS, *[f"f1_{c}" for c in CLASS_NAMES],
                                                  "insect_to_good"]}
        raw[model]["seeds"] = sub["seed"].tolist()
        cells = [_cell(sub[m], m) for m in metric_names]
        lines.append(f"| {MODEL_LABELS[model]} | {len(sub)} | " + " | ".join(cells) + " |")
    return "\n".join(lines), raw


def valid_table(df: pd.DataFrame, cfg: str, tta: str) -> str:
    """Validasi: accuracy & F1 makro (mean ± std), tanpa TTA dan TTA terpilih."""
    lines = [f"| Model | Val acc tanpa TTA | Val F1 tanpa TTA | Val acc {TTA_LABELS[tta]} | Val F1 {TTA_LABELS[tta]} |",
             "|---|--:|--:|--:|--:|"]
    for m in MODELS:
        a = _sel(df, config=cfg, split="valid", tta="none", model=m)
        b = _sel(df, config=cfg, split="valid", tta=tta, model=m)
        lines.append(f"| {MODEL_LABELS[m]} | {fmt_mean_std(a['accuracy'])} | {fmt_mean_std(a['f1'])} | "
                     f"{fmt_mean_std(b['accuracy'])} | {fmt_mean_std(b['f1'])} |")
    return "\n".join(lines)


def per_seed_table(df: pd.DataFrame, cfg: str, tta: str) -> str:
    """Nilai per seed (test) untuk konfigurasi utama."""
    lines = ["| Model | Seed | Epoch | Acc | Macro P | Macro R | Macro F1 | Defect recall | Miss rate | I→G |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for model in MODELS:
        sub = _sel(df, config=cfg, split="test", tta=tta, model=model)
        sub = sub.sort_values("seed", key=lambda s: s.map(_seed_rank))
        for r in sub.itertuples():
            lines.append(
                f"| {MODEL_LABELS[model]} | {r.seed} | {r.best_epoch} | {r.accuracy:.3f} | {r.precision:.3f} | "
                f"{r.recall:.3f} | {r.f1:.3f} | {r.defect_recall:.3f} | {100 * r.defect_miss_rate:.1f}% | "
                f"{r.insect_to_good} |"
            )
    return "\n".join(lines)


def per_class_table(df: pd.DataFrame, cfg: str, tta: str) -> str:
    """F1 per kelas mean ± std (test)."""
    lines = ["| Kelas | " + " | ".join(MODEL_LABELS[m] for m in MODELS) + " |",
             "|---|" + "--:|" * len(MODELS)]
    for name in CLASS_NAMES:
        cells = [fmt_mean_std(_sel(df, config=cfg, split="test", tta=tta, model=m)[f"f1_{name}"])
                 for m in MODELS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ablation_tables(df: pd.DataFrame, profile: dict, sel_tta: str, main_cfg: str) -> tuple[str, str, dict]:
    """Tabel ablasi test (F1 & miss rate) + tabel val F1 untuk baris yang sama."""
    head = "| Konfigurasi | Seed | " + " | ".join(
        f"{MODEL_LABELS[m]} F1 | {MODEL_LABELS[m]} miss" for m in MODELS) + " | Val F1 rata2 |"
    lines = [head, "|---|---|" + "--:|" * (2 * len(MODELS) + 1)]
    vlines = ["| Konfigurasi | Seed | " + " | ".join(f"{MODEL_LABELS[m]} val F1" for m in MODELS)
              + " | Rata-rata 3 model | Δ vs main (seed sama) |",
              "|---|---|" + "--:|" * (len(MODELS) + 2)]
    raw: dict = {}
    for label_tpl, cfg, tta_kind, seeds in profile["ablation"]:
        tta = sel_tta if tta_kind == "sel" else "none"
        label = label_tpl.format(tta=TTA_LABELS[sel_tta])
        test = _sel(df, config=cfg, split="test", tta=tta)
        valid = _sel(df, config=cfg, split="valid", tta=tta)
        if seeds is not None:
            test, valid = _sel(test, seed=list(seeds)), _sel(valid, seed=list(seeds))
        if test.empty:
            continue
        n = test.groupby("model")["seed"].nunique()
        seed_txt = f"{int(n.min())}" if n.min() == n.max() else f"{int(n.min())}-{int(n.max())}"
        cells, vcells = [], []
        entry = {"config": cfg, "tta": tta, "seeds": sorted(test["seed"].unique().tolist())}
        for m in MODELS:
            t, v = _sel(test, model=m), _sel(valid, model=m)
            cells += [fmt_mean_std(t["f1"]), fmt_mean_std(t["defect_miss_rate"], 1, 100.0) + "%"]
            vcells.append(fmt_mean_std(v["f1"]))
            entry[m] = {"test_f1": _stats(t["f1"]), "test_miss_rate": _stats(t["defect_miss_rate"]),
                        "test_accuracy": _stats(t["accuracy"]), "val_f1": _stats(v["f1"])}
        # Selisih val F1 berpasangan terhadap main pada seed yang sama (TTA sama).
        ref = _sel(df, config=main_cfg, split="valid", tta=tta, seed=entry["seeds"])
        merged = valid.merge(ref, on=["model", "seed"], suffixes=("", "_ref"))
        delta = (merged["f1"] - merged["f1_ref"]).mean() if len(merged) else float("nan")
        entry["val_f1_delta_vs_main"] = float(delta)
        cells.append(f"{valid['f1'].mean():.3f}")
        vcells += [f"{valid['f1'].mean():.3f}", "—" if cfg == main_cfg else f"{delta:+.3f}"]
        raw[label] = entry
        lines.append(f"| {label} | {seed_txt} | " + " | ".join(cells) + " |")
        vlines.append(f"| {label} | {seed_txt} | " + " | ".join(vcells) + " |")
    return "\n".join(lines), "\n".join(vlines), raw


def tta_selection(df: pd.DataFrame, cfg: str) -> tuple[str, dict, str]:
    """Pilih varian TTA dari F1 makro VALID rata-rata lintas seed."""
    valid = _sel(df, config=cfg, split="valid")
    table = valid.groupby(["model", "tta"])["f1"].agg(["mean", "std", "count"])
    per_model_best = {m: table.loc[m]["mean"].idxmax() for m in MODELS}
    overall = valid.groupby("tta")["f1"].mean()
    uniform = overall.idxmax()
    lines = ["| Varian TTA | " + " | ".join(MODEL_LABELS[m] for m in MODELS) + " | Rata-rata 3 model |",
             "|---|" + "--:|" * (len(MODELS) + 1)]
    for tta in TTA_CHOICES:
        cells = []
        for m in MODELS:
            mean, std = table.loc[(m, tta), ["mean", "std"]]
            mark = " **(terbaik)**" if per_model_best[m] == tta else ""
            cells.append(f"{mean:.3f} ± {std:.3f}{mark}")
        best = " **(terpilih)**" if tta == uniform else ""
        lines.append(f"| {TTA_LABELS[tta]} | " + " | ".join(cells) + f" | {overall[tta]:.3f}{best} |")
    raw = {
        "config": cfg,
        "valid_f1": {m: {t: _stats(_sel(valid, model=m, tta=t)["f1"]) for t in TTA_CHOICES} for m in MODELS},
        "best_per_model": per_model_best,
        "uniform_choice": uniform,
        "overall_mean_valid_f1": overall.to_dict(),
    }
    return "\n".join(lines), raw, uniform


def tta_effect(df: pd.DataFrame, cfg: str, tta: str) -> tuple[str, dict]:
    """Efek TTA di test: none -> varian terpilih, mean ± std + jumlah run yang membaik."""
    lines = ["| Model | F1 tanpa → TTA | Acc tanpa → TTA | Miss tanpa → TTA | I→G tanpa → TTA | "
             "Run F1 naik | Run miss turun |", "|---|---|---|---|---|--:|--:|"]
    raw: dict = {"tta": tta}
    for m in MODELS:
        a = _sel(df, config=cfg, split="test", tta="none", model=m).sort_values("seed").set_index("seed")
        b = _sel(df, config=cfg, split="test", tta=tta, model=m).sort_values("seed").set_index("seed")
        up = int((b["f1"] > a["f1"]).sum())
        down = int((b["defect_miss_rate"] < a["defect_miss_rate"]).sum())
        lines.append(
            f"| {MODEL_LABELS[m]} | {fmt_mean_std(a['f1'])} → {fmt_mean_std(b['f1'])} | "
            f"{fmt_mean_std(a['accuracy'])} → {fmt_mean_std(b['accuracy'])} | "
            f"{fmt_mean_std(a['defect_miss_rate'], 1, 100)}% → {fmt_mean_std(b['defect_miss_rate'], 1, 100)}% | "
            f"{fmt_mean_std(a['insect_to_good'], 1)} → {fmt_mean_std(b['insect_to_good'], 1)} | "
            f"{up}/{len(a)} | {down}/{len(a)} |"
        )
        raw[m] = {k: {"none": _stats(a[k]), tta: _stats(b[k])}
                  for k in ("f1", "accuracy", "defect_miss_rate", "insect_to_good")}
        raw[m]["n_f1_up"], raw[m]["n_miss_down"], raw[m]["n_runs"] = up, down, int(len(a))
    return "\n".join(lines), raw


def threshold_selection(reports: list[dict], cfg: str, tta: str, max_drop: float) -> tuple[str, dict]:
    """Sweep ambang P(good) di VALID (rata-rata seed), terapkan ke test."""
    raw: dict = {"config": cfg, "tta": tta, "grid": list(DEFAULT_THRESHOLDS), "criterion":
                 f"min miss rate valid (rata-rata seed) dengan F1 valid >= F1 argmax - {max_drop}"}
    lines = ["| Model | Ambang | Val miss argmax → ambang | Val F1 argmax → ambang | "
             "Test miss argmax → ambang | Test F1 argmax → ambang | Test acc argmax → ambang |",
             "|---|--:|---|---|---|---|---|"]
    for m in MODELS:
        runs = sorted((r for r in reports if r["config_name"] == cfg and r["model"] == m), key=lambda r: r["seed"])
        preds = [load_run_predictions(r["_dir"] / "predictions.npz") for r in runs]
        sweeps = [sweep_thresholds(p["valid"]["y_true"], softmax(p["valid"]["logits"][tta].astype(np.float64)))
                  for p in preds]
        mean_sweep = sum(sweeps) / len(sweeps)
        chosen = select_threshold(mean_sweep, max_f1_drop=max_drop, reference="argmax")
        per_seed_choice = [select_threshold(s, max_f1_drop=max_drop, reference="argmax") for s in sweeps]
        test_arg, test_thr = [], []
        for p in preds:
            y, logits = p["test"]["y_true"], p["test"]["logits"][tta]
            test_arg.append(metrics_from_logits(y, logits, 0.0))
            test_thr.append(metrics_from_logits(y, logits, chosen))

        def pair(key, src_a, src_b, pct=False):
            a = [x[key] for x in src_a]
            b = [x[key] for x in src_b]
            if pct:
                return f"{fmt_mean_std(a, 1, 100)}% → {fmt_mean_std(b, 1, 100)}%"
            return f"{fmt_mean_std(a)} → {fmt_mean_std(b)}"

        v0, vt = mean_sweep.loc[0.0], mean_sweep.loc[chosen]
        lines.append(
            f"| {MODEL_LABELS[m]} | {'argmax' if chosen == 0 else f'{chosen:.2f}'} | "
            f"{100 * v0['miss_rate']:.1f}% → {100 * vt['miss_rate']:.1f}% | {v0['f1']:.3f} → {vt['f1']:.3f} | "
            f"{pair('defect_miss_rate', test_arg, test_thr, True)} | {pair('f1', test_arg, test_thr)} | "
            f"{pair('accuracy', test_arg, test_thr)} |"
        )
        raw[m] = {
            "chosen_threshold": chosen,
            "seeds": [r["seed"] for r in runs],
            "per_seed_choice": per_seed_choice,
            "valid_mean_sweep": mean_sweep.reset_index().to_dict(orient="records"),
            "test_argmax": {k: _stats([x[k] for x in test_arg]) for k in ("f1", "accuracy", "defect_miss_rate")},
            "test_threshold": {k: _stats([x[k] for x in test_thr]) for k in ("f1", "accuracy", "defect_miss_rate")},
            "test_good_rejected": _stats([
                sum(row[1:]) for x in test_thr for row in [x["confusion"][0]]
            ]),
        }
    return "\n".join(lines), raw


def error_patterns(reports: list[dict], cfg: str, tta: str, top: int = 4) -> tuple[str, dict]:
    """Pasangan salah klasifikasi terbanyak (test): seed 42 dan rata-rata seed."""
    lines = ["| Model | Peringkat | Pasangan (true→pred) | Seed 42 | Rata-rata seed (± std) |",
             "|---|--:|---|--:|--:|"]
    raw: dict = {"tta": tta}
    for m in MODELS:
        runs = sorted((r for r in reports if r["config_name"] == cfg and r["model"] == m), key=lambda r: r["seed"])
        cms = {r["seed"]: np.asarray(r["evaluation"]["test"][tta]["confusion"]) for r in runs}
        stack = np.stack(list(cms.values()))
        mean, std = stack.mean(0), stack.std(0, ddof=1) if len(stack) > 1 else np.zeros_like(stack[0], float)
        pairs = [(i, j) for i in range(len(CLASS_NAMES)) for j in range(len(CLASS_NAMES)) if i != j]
        pairs.sort(key=lambda p: -mean[p])
        raw[m] = []
        for rank, (i, j) in enumerate(pairs[:top], 1):
            key = f"{ABBR[CLASS_NAMES[i]]}→{ABBR[CLASS_NAMES[j]]}"
            s42 = int(cms[42][i, j]) if 42 in cms else None
            lines.append(f"| {MODEL_LABELS[m]} | {rank} | {CLASS_NAMES[i]}→{CLASS_NAMES[j]} ({key}) | {s42} | "
                         f"{mean[i, j]:.1f} ± {std[i, j]:.1f} |")
            raw[m].append({"pair": key, "true": CLASS_NAMES[i], "pred": CLASS_NAMES[j], "seed42": s42,
                           "mean": float(mean[i, j]), "std": float(std[i, j])})
    return "\n".join(lines), raw


def epoch_table(df: pd.DataFrame) -> tuple[str, dict]:
    """Epoch checkpoint terpilih per konfigurasi, model, seed."""
    sub = _sel(df, split="test", tta="d4")[["config", "model", "seed", "best_epoch"]].drop_duplicates()
    raw = {}
    lines = ["| Konfigurasi | Model | Epoch per seed (seed: epoch) | Rata-rata |", "|---|---|---|--:|"]
    for cfg in sorted(sub["config"].unique()):
        for m in MODELS:
            s = _sel(sub, config=cfg, model=m).sort_values("seed")
            if s.empty:
                continue
            raw.setdefault(cfg, {})[m] = dict(zip(s["seed"].astype(str), s["best_epoch"].astype(int)))
            txt = ", ".join(f"{a}: {b}" for a, b in zip(s["seed"], s["best_epoch"]))
            lines.append(f"| {cfg} | {MODEL_LABELS[m]} | {txt} | {s['best_epoch'].mean():.1f} |")
    return "\n".join(lines), raw


def repro_check(runs_dir: Path) -> tuple[str, dict]:
    """Bandingkan run repro_valacc (seed 42) dengan history & metrik lama."""
    old_tta = pd.read_csv(PROJECT_ROOT / "results" / "experiments" / "tta_screening.csv")
    raw, lines = {}, ["| Model | History identik | Epoch lama → baru | Test F1 D4 lama → baru | "
                      "Test acc D4 lama → baru |", "|---|---|---|---|---|"]
    for m in MODELS:
        run = runs_dir / f"{m}_repro_valacc_seed42"
        if not (run / "metrics.json").exists():
            continue
        new_h = json.loads((run / f"{m}_history.json").read_text(encoding="utf-8"))
        old_h = json.loads((PROJECT_ROOT / "results" / "models" / m / f"{m}_history.json").read_text(encoding="utf-8"))
        same = all(new_h[k] == old_h[k] for k in ("train_loss", "train_acc", "val_loss", "val_acc"))
        rep = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        old = old_tta[(old_tta.model == m) & (old_tta.split == "test") & (old_tta.tta == "d4")].iloc[0]
        new = rep["evaluation"]["test"]["d4"]
        raw[m] = {"history_identical": bool(same), "old_epoch": old_h["best_epoch"], "new_epoch": rep["best_epoch"],
                  "old_test_f1": float(old.f1), "new_test_f1": new["f1"],
                  "old_test_acc": float(old.acc), "new_test_acc": new["accuracy"]}
        lines.append(f"| {MODEL_LABELS[m]} | {'ya' if same else 'TIDAK'} | {old_h['best_epoch']} → {rep['best_epoch']} | "
                     f"{old.f1:.4f} → {new['f1']:.4f} | {old.acc:.4f} → {new['accuracy']:.4f} |")
    return "\n".join(lines), raw


def ranking_check(df: pd.DataFrame, cfg: str, tta: str) -> tuple[str, dict]:
    """Selisih berpasangan per seed untuk F1 makro dan miss rate."""
    raw: dict = {"tta": tta}
    lines = []
    for metric, title, scale, fmt in (("f1", "F1 makro", 1.0, "{:+.3f}"),
                                      ("defect_miss_rate", "Miss rate (poin %)", 100.0, "{:+.1f}")):
        sub = _sel(df, config=cfg, split="test", tta=tta).pivot(index="seed", columns="model", values=metric)[MODELS]
        sub = sub * scale
        order_ok = (sub["resnet50"] > sub["mobilenetv3"]) & (sub["mobilenetv3"] > sub["inceptionv3"])
        diffs = {
            "resnet50-mobilenetv3": sub["resnet50"] - sub["mobilenetv3"],
            "mobilenetv3-inceptionv3": sub["mobilenetv3"] - sub["inceptionv3"],
            "resnet50-inceptionv3": sub["resnet50"] - sub["inceptionv3"],
        }
        std = {m: float(sub[m].std(ddof=1)) for m in MODELS}
        entry = {"order_resnet_gt_mobilenet_gt_inception": order_ok.astype(bool).to_dict(),
                 "std_per_model": std, "diffs": {}}
        lines += [f"**{title}** (test, TTA {TTA_LABELS[tta]})", "",
                  "| Selisih | Rata-rata (paired) | Std selisih | Seed dengan selisih > 0 |", "|---|--:|--:|--:|"]
        for key, d in diffs.items():
            entry["diffs"][key] = {"mean": float(d.mean()), "std": float(d.std(ddof=1)),
                                   "n_positive": int((d > 0).sum()), "n": int(d.size),
                                   "values": {str(k): float(v) for k, v in d.items()}}
            digits = 1 if metric == "defect_miss_rate" else 3
            lines.append(f"| {key} | {fmt.format(d.mean())} | {d.std(ddof=1):.{digits}f} | "
                         f"{int((d > 0).sum())}/{d.size} |")
        digits = 1 if metric == "defect_miss_rate" else 3
        std_txt = ", ".join(f"{MODEL_LABELS[m]} {std[m]:.{digits}f}" for m in MODELS)
        if metric == "f1":
            lines += ["", f"Urutan F1 ResNet-50 > MobileNetV3-L > InceptionV3 bertahan di "
                      f"{int(order_ok.sum())}/{len(order_ok)} seed. Std antar-seed per model: {std_txt}.", ""]
        else:
            short = {"resnet50": "R", "mobilenetv3": "M", "inceptionv3": "I"}
            orders = {int(s): " < ".join(short[m] for m in row.sort_values().index) for s, row in sub.iterrows()}
            entry["order_low_to_high_per_seed"] = orders
            n_distinct = len(set(orders.values()))
            lines += ["", "Urutan miss rate (terendah → tertinggi) per seed: "
                      + "; ".join(f"{s}: {o}" for s, o in sorted(orders.items(), key=lambda x: _seed_rank(x[0])))
                      + f" ({n_distinct} urutan berbeda). Std antar-seed per model (poin %): {std_txt}. "
                      "Selisih > 0 berarti model pertama lebih banyak meloloskan cacat.", ""]
        raw[metric] = entry
    return "\n".join(lines).rstrip(), raw


def compare_configs(df: pd.DataFrame, cfg_a: str, label_a: str, cfg_b: str, label_b: str,
                    tta: str) -> tuple[str, dict]:
    """Perbandingan berpasangan dua konfigurasi (seed sama): valid & test."""
    lines = [f"| Model | Seed | Val F1 {label_b} → {label_a} | Test F1 {label_b} → {label_a} | "
             f"Test acc {label_b} → {label_a} | Test miss {label_b} → {label_a} | Δ val F1 (paired) | "
             "Seed val F1 lebih tinggi |", "|---|--:|---|---|---|---|--:|--:|"]
    raw: dict = {"a": cfg_a, "b": cfg_b, "tta": tta}
    for m in MODELS:
        out = {}
        for split in ("valid", "test"):
            a = _sel(df, config=cfg_a, split=split, tta=tta, model=m).set_index("seed")
            b = _sel(df, config=cfg_b, split=split, tta=tta, model=m).set_index("seed")
            seeds = sorted(set(a.index) & set(b.index))
            a, b = a.loc[seeds], b.loc[seeds]
            out[split] = (a, b)
        (va, vb), (ta, tb) = out["valid"], out["test"]
        d = va["f1"] - vb["f1"]
        lines.append(
            f"| {MODEL_LABELS[m]} | {len(va)} | {fmt_mean_std(vb['f1'])} → {fmt_mean_std(va['f1'])} | "
            f"{fmt_mean_std(tb['f1'])} → {fmt_mean_std(ta['f1'])} | "
            f"{fmt_mean_std(tb['accuracy'])} → {fmt_mean_std(ta['accuracy'])} | "
            f"{fmt_mean_std(tb['defect_miss_rate'], 1, 100)}% → {fmt_mean_std(ta['defect_miss_rate'], 1, 100)}% | "
            f"{d.mean():+.3f} ± {d.std(ddof=1):.3f} | {int((d > 0).sum())}/{len(d)} |"
        )
        raw[m] = {
            "seeds": [int(s) for s in va.index],
            "val_f1": {cfg_b: _stats(vb["f1"]), cfg_a: _stats(va["f1"])},
            "test_f1": {cfg_b: _stats(tb["f1"]), cfg_a: _stats(ta["f1"])},
            "test_accuracy": {cfg_b: _stats(tb["accuracy"]), cfg_a: _stats(ta["accuracy"])},
            "test_miss_rate": {cfg_b: _stats(tb["defect_miss_rate"]), cfg_a: _stats(ta["defect_miss_rate"])},
            "test_f1_delta": _stats(ta["f1"] - tb["f1"]),
            "test_miss_delta": _stats(ta["defect_miss_rate"] - tb["defect_miss_rate"]),
            "val_f1_delta": _stats(d), "n_val_higher": int((d > 0).sum()),
        }
    return "\n".join(lines), raw


def epoch_diagnostic(df: pd.DataFrame, reports: list[dict], cfg_long: str, cfg_main: str, seed: int,
                     tta: str) -> tuple[str, dict]:
    """Diagnostik 60 epoch: val F1 terbaik <=30 vs <=60 dan test F1 checkpoint masing-masing.

    Checkpoint <=30 diambil dari run utama 30 epoch (seed sama); 30 epoch
    pertama run panjang diverifikasi identik dengan run tersebut.
    """
    by = {(r["config_name"], r["model"], r["seed"]): r for r in reports}
    lines = ["| Model | 30 epoch awal identik | Epoch terbaik ≤30 | Val F1 ≤30 | Epoch terbaik ≤60 | Val F1 ≤60 | "
             "Δ val F1 | Std val F1 antar-seed (main) | Test F1 ckpt ≤30 | Test F1 ckpt ≤60 | "
             "Test miss ≤30 → ≤60 |", "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    raw: dict = {"seed": seed, "tta": tta, "curves": {}}
    for m in MODELS:
        long_r, main_r = by.get((cfg_long, m, seed)), by.get((cfg_main, m, seed))
        if long_r is None or main_r is None:
            continue
        hl = json.loads((long_r["_dir"] / f"{m}_history.json").read_text(encoding="utf-8"))
        hm = json.loads((main_r["_dir"] / f"{m}_history.json").read_text(encoding="utf-8"))
        same = all(np.allclose(hl[k][:30], hm[k][:30], rtol=0, atol=1e-9) for k in ("train_loss", "val_loss", "val_f1"))
        vf = np.asarray(hl["val_f1"])
        e30 = int(np.argmax(vf[:30]) + 1)
        e60 = int(np.argmax(vf) + 1)
        std_seed = float(_sel(df, config=cfg_main, split="valid", tta="none", model=m)["f1"].std(ddof=1))
        t30, t60 = main_r["evaluation"]["test"][tta], long_r["evaluation"]["test"][tta]
        delta = float(vf[e60 - 1] - vf[e30 - 1])
        lines.append(
            f"| {MODEL_LABELS[m]} | {'ya' if same else 'TIDAK'} | {e30} | {vf[e30 - 1]:.3f} | {e60} | {vf[e60 - 1]:.3f} | "
            f"{delta:+.3f} | {std_seed:.3f} | {t30['f1']:.3f} | {t60['f1']:.3f} | "
            f"{100 * t30['defect_miss_rate']:.1f}% → {100 * t60['defect_miss_rate']:.1f}% |"
        )
        raw[m] = {"first30_identical": bool(same), "best_epoch_30": e30, "val_f1_30": float(vf[e30 - 1]),
                  "best_epoch_60": e60, "val_f1_60": float(vf[e60 - 1]), "val_f1_gain": delta,
                  "val_f1_seed_std_main": std_seed, "gain_exceeds_seed_std": bool(delta > std_seed),
                  "test_f1_ckpt30": t30["f1"], "test_f1_ckpt60": t60["f1"],
                  "test_acc_ckpt30": t30["accuracy"], "test_acc_ckpt60": t60["accuracy"],
                  "test_miss_ckpt30": t30["defect_miss_rate"], "test_miss_ckpt60": t60["defect_miss_rate"],
                  "val_f1_ckpt_criterion_note": "val F1 tanpa TTA (kriteria checkpoint)"}
        raw["curves"][m] = {k: hl[k] for k in ("train_loss", "val_loss", "val_f1")}
    lines += ["", "Val F1 di tabel ini = kriteria checkpoint (valid tanpa TTA). Test F1 memakai TTA terpilih."]
    return "\n".join(lines), raw


def main() -> None:
    """Agregasi seluruh run dan tulis tabel + JSON."""
    args = parse_args()
    profile = PROFILES[args.main_config]
    cfg = args.main_config
    runs_dir = Path(args.runs_dir)
    out_dir = runs_dir / profile["summary_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    reports, df = load_runs(runs_dir)
    if df.empty:
        raise SystemExit(f"Tidak ada run di {runs_dir}")
    df.to_csv(out_dir / "per_run.csv", index=False, encoding="utf-8")

    sections: list[tuple[str, str]] = []
    results: dict = {"main_config": cfg, "n_runs": len(reports), "runs": sorted(r["run"] for r in reports)}

    # Pemilihan TTA lebih dulu: varian terpilih dipakai seluruh tabel berikutnya.
    t, raw, sel = tta_selection(df, cfg)
    results["tta_selection"] = raw
    lab = TTA_LABELS[sel]
    tables = [(f"Pemilihan TTA di VALID (F1 makro, {cfg})", t)]

    t, raw = main_table(df, cfg, sel); tables.append((f"Tabel utama test, TTA {lab} (mean ± std)", t)); results["main_sel"] = raw
    results[f"main_{sel}"] = raw  # alias (kompatibel kunci lama "main_d4")
    t, raw = main_table(df, cfg, "none"); tables.append(("Tabel utama test, tanpa TTA (mean ± std)", t)); results["main_none"] = raw
    tables.append(("Validasi (mean ± std)", valid_table(df, cfg, sel)))
    tables.append((f"Nilai per seed, test TTA {lab}", per_seed_table(df, cfg, sel)))
    tables.append(("Nilai per seed, test tanpa TTA", per_seed_table(df, cfg, "none")))
    tables.append((f"F1 per kelas, test TTA {lab} (mean ± std)", per_class_table(df, cfg, sel)))
    tables.append(("F1 per kelas, test tanpa TTA (mean ± std)", per_class_table(df, cfg, "none")))
    t, raw = error_patterns(reports, cfg, sel); tables.append((f"Pola kesalahan terbanyak (test, TTA {lab})", t)); results["errors"] = raw
    t, raw = ranking_check(df, cfg, sel); tables.append(("Peringkat arsitektur vs variasi antar-seed", t)); results["ranking"] = raw
    t, vt, raw = ablation_tables(df, profile, sel, cfg)
    tables.append((f"Ablasi (test, F1 makro & miss rate, TTA {lab})", t))
    tables.append(("Ablasi (val F1 makro, baris yang sama)", vt)); results["ablation"] = raw
    if profile["compare_config"]:
        other, other_label = profile["compare_config"]
        t, raw = compare_configs(df, cfg, "bn_frozen", other, other_label, sel)
        tables.append((f"bn_frozen vs {other_label} (berpasangan, TTA {lab})", t)); results["bn_effect"] = raw
    t, raw = tta_effect(df, cfg, sel); tables.append((f"Efek TTA di test (tanpa → {lab})", t)); results["tta_effect"] = raw
    t, raw = threshold_selection(reports, cfg, sel, args.max_f1_drop)
    tables.append((f"Ambang P(good), dipilih di VALID (TTA {lab})", t)); results["threshold"] = raw
    if profile["epoch_diag"]:
        long_cfg, seed = profile["epoch_diag"]
        t, raw = epoch_diagnostic(df, reports, long_cfg, cfg, seed, sel)
        tables.append((f"Diagnostik 60 epoch (seed {seed})", t)); results["epoch60"] = raw
    t, raw = epoch_table(df); tables.append(("Epoch checkpoint terpilih (semua konfigurasi)", t)); results["checkpoint_epochs"] = raw
    if cfg == "main":
        t, raw = repro_check(runs_dir); tables.append(("Reproduksi run lama (kriteria val_acc, seed 42)", t)); results["repro"] = raw
    # Angka mentah per run x split x TTA.
    results["per_run"] = df.to_dict(orient="records")

    md = "\n\n".join(f"### {title}\n\n{body}" for title, body in tables)
    (out_dir / "tables.md").write_text(md + "\n", encoding="utf-8")

    out_json = Path(args.output_json)
    existing = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    existing[profile["json_key"]] = results
    out_json.write_text(json.dumps(existing, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    # Tabel memuat karakter non-ASCII (±, →); konsol Windows cp1252 tidak bisa
    # mencetaknya, jadi hanya ditulis ke berkas.
    print(f"{len(reports)} run diagregasi (profil {cfg}). TTA seragam terpilih (valid): {sel}")
    print(f"Tersimpan: {out_dir / 'tables.md'}, {out_dir / 'per_run.csv'}, {out_json} [{profile['json_key']}]")


if __name__ == "__main__":
    main()

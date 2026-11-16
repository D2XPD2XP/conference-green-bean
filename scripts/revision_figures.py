"""CLI: regenerasi figur paper dari hasil revisi (Tugas 11 / brief lanjutan §7).

Gaya sama dengan ``notebooks/11_paper_figures`` (serif, palet model biru /
oranye / hijau, lebar satu kolom IEEE 3,5 in) lewat
:mod:`src.analysis.paper_style`. Sejak brief lanjutan 2026-11-13 **seluruh
teks di figur >= 7 pt** pada ukuran cetak satu kolom (``MIN_FS``); tata letak
disesuaikan agar tidak ada teks yang bertumpuk.

Profil (``--main-config``):

- ``main``      : konfigurasi utama 2026-11-12 (BN beradaptasi) ->
  ``results/figures/paper_revision/``.
- ``bn_frozen`` : konfigurasi utama sejak 2026-11-13 (BN beku total) ->
  ``results/figures/paper_revision_bnfrozen/``. Pembanding ``bn_adapt`` =
  config ``main``.

Varian TTA di semua figur = pilihan VALID dari ``scripts.aggregate_revision``
(kunci ``tta_selection.uniform_choice``), label figur mengikutinya.

Model representatif untuk figur visual (kurva, confusion, Grad-CAM,
confidence/t-SNE) = **seed 42**. Figur agregat (F1 per kelas, TTA, trade-off,
efek BN) memakai mean ± std lintas 5 seed.

Contoh::

    python -m scripts.revision_figures --main-config bn_frozen
    python -m scripts.revision_figures --main-config bn_frozen --only fig_curves fig_tta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from configs.base_config import (  # noqa: E402
    CLASS_COLORS,
    CLASS_NAMES,
    CROPS_DIR,
    DEVICE,
    FIGURES_DIR,
    NUM_CLASSES,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.analysis.paper_style import (  # noqa: E402
    CLASS_ABBR,
    CLASS_LABELS,
    CLASS_MARKERS,
    COL_W,
    CONF_COLORS,
    HIGHLIGHT,
    INK,
    INK_2,
    MODEL_COLORS,
    MODEL_LABELS,
    MODELS,
    NEUTRAL,
    panel_label,
    paper_style,
    save_paper,
)
from src.evaluation.revision import load_run_predictions, softmax  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "runs"
SEEDS = (42, 0, 1, 2, 3)
FIGURES = ["fig_curves", "fig_perclass_f1", "fig_confusion", "fig_tta", "fig_tradeoff",
           "fig_gradcam", "fig_confidence", "fig_bn_effect"]

#: Ukuran teks minimum (pt) pada ukuran cetak satu kolom.
MIN_FS = 7.0

#: Profil: folder ringkasan, folder figur, kunci JSON agregat/support/figur, pembanding BN.
PROFILES: dict[str, dict] = {
    "main": {"summary": "summary", "out": "paper_revision", "agg": "aggregate", "support": "support",
             "figures": "figures", "compare": "bn_frozen"},
    "bn_frozen": {"summary": "summary/bnfrozen", "out": "paper_revision_bnfrozen", "agg": "aggregate_bnfrozen",
                  "support": "support_bnfrozen", "figures": "figures_bnfrozen", "compare": "main"},
}
#: Label varian TTA untuk legenda figur.
TTA_NAMES: dict[str, str] = {"none": "Without TTA", "hflip": "H-flip TTA", "flip_hv": "H+V-flip TTA",
                             "d4": "$D_4$ TTA"}
TTA_COLORS: dict[str, str] = {"none": NEUTRAL, "hflip": "#6250d6", "flip_hv": "#6250d6", "d4": "#6250d6"}

# Diset di main() sesuai profil.
CONFIG = "main"
PROFILE = PROFILES["main"]
OUT_DIR = FIGURES_DIR / PROFILE["out"]
SEL_TTA = "d4"


def _run(model: str, seed: int = 42, config: str | None = None) -> Path:
    return RUNS_DIR / f"{model}_{config or CONFIG}_seed{seed}"


def _history(model: str, seed: int = 42) -> dict:
    return json.loads((_run(model, seed) / f"{model}_history.json").read_text(encoding="utf-8"))


def _all_runs() -> pd.DataFrame:
    return pd.read_csv(RUNS_DIR / PROFILE["summary"] / "per_run.csv")


def _per_run(config: str | None = None) -> pd.DataFrame:
    df = _all_runs()
    return df[(df.config == (config or CONFIG)) & (df.split == "test")]


def _results() -> dict:
    return json.loads((PROJECT_ROOT / "revision_results.json").read_text(encoding="utf-8"))


def _metrics(model: str, seed: int = 42) -> dict:
    return json.loads((_run(model, seed) / "metrics.json").read_text(encoding="utf-8"))


def _clean_axes(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_curves() -> None:
    """Loss train/val + val F1 makro per epoch (seed 42), penanda epoch terpilih."""
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(COL_W, 2.05))
    for m in MODELS:
        h, color = _history(m), MODEL_COLORS[m]
        n = min(30, len(h["train_loss"]))
        epochs = np.arange(1, n + 1)
        ax_loss.plot(epochs, h["train_loss"][:n], color=color, lw=1.1, ls="-")
        ax_loss.plot(epochs, h["val_loss"][:n], color=color, lw=1.1, ls=(0, (3, 1.5)))
        ax_f1.plot(epochs, h["val_f1"][:n], color=color, lw=1.1)
        best = int(h["best_epoch"])
        assert abs(h["val_f1"][best - 1] - h["best_metric"]) < 1e-12
        y = h["val_f1"][best - 1]
        ax_f1.plot(best, y, marker="o", ms=4.5, mfc=color, mec="white", mew=0.8, zorder=5)
        ax_f1.annotate(f"ep {best}", (best, y), xytext=(0, 4), textcoords="offset points", ha="center",
                       va="bottom", fontsize=MIN_FS, color=color,
                       bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none"}, zorder=6)
    for ax in (ax_loss, ax_f1):
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, 30)
        ax.set_xticks([1, 10, 20, 30])
    lo, hi = ax_f1.get_ylim()
    ax_f1.set_ylim(lo, hi + 0.12 * (hi - lo))  # ruang label epoch di atas penanda
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_f1.set_ylabel("Validation macro F1")
    panel_label(ax_loss, "(a)")
    panel_label(ax_f1, "(b)")
    models = [Line2D([], [], color=MODEL_COLORS[m], lw=1.4, label=MODEL_LABELS[m]) for m in MODELS]
    styles = [Line2D([], [], color=INK_2, lw=1.1, ls="-", label="train"),
              Line2D([], [], color=INK_2, lw=1.1, ls=(0, (3, 1.5)), label="validation"),
              Line2D([], [], color=INK_2, lw=0, marker="o", ms=4, mfc=INK_2, mec="white",
                     label="selected epoch")]
    # Legenda diisi per kolom: baris 1 = model, baris 2 = gaya garis.
    handles = [h for pair in zip(models, styles) for h in pair]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.97),
               handlelength=1.6, columnspacing=1.0, handletextpad=0.4, fontsize=MIN_FS)
    fig.tight_layout(w_pad=1.2)
    save_paper(fig, OUT_DIR, "fig_curves")


def fig_perclass_f1(ymin: float = 0.4) -> None:
    """F1 per kelas: mean ± std lintas 5 seed (test, TTA terpilih)."""
    df = _per_run()
    sub_tta = df[df.tta == SEL_TTA]
    per_class = _metrics("resnet50")["evaluation"]["test"][SEL_TTA]["per_class"]
    support = {c: int(per_class[c]["support"]) for c in CLASS_NAMES}
    n_seeds = sub_tta.groupby("model")["seed"].nunique().min()
    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    x = np.arange(len(CLASS_NAMES))
    width = 0.27
    for i, m in enumerate(MODELS):
        sub = sub_tta[sub_tta.model == m]
        mean = np.array([sub[f"f1_{c}"].mean() for c in CLASS_NAMES])
        std = np.nan_to_num(np.array([sub[f"f1_{c}"].std(ddof=1) for c in CLASS_NAMES]))
        pos = x + (i - 1) * width
        ax.bar(pos, mean - ymin, width=width - 0.03, bottom=ymin, color=MODEL_COLORS[m],
               label=MODEL_LABELS[m], linewidth=0)
        ax.errorbar(pos, mean, yerr=std, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.4, capthick=0.6)
        for xp, v, s in zip(pos, mean, std):
            ax.text(xp, v + s + 0.012, f"{v:.2f}", ha="center", va="bottom", rotation=90, fontsize=MIN_FS,
                    color=INK)
    labels = {"good": "good", "black": "black", "chipped": "chipped", "dry_pods": "dry\npods",
              "insect_damage": "insect\ndamage"}
    ax.set_xticks(x, [f"{labels[c]}\n(n={support[c]})" for c in CLASS_NAMES])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(ymin, 1.1)
    ax.set_yticks(np.arange(ymin, 1.01, 0.1))
    ax.set_ylabel(f"F1-score (mean ± std, {n_seeds} seeds)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.0, columnspacing=1.2,
              fontsize=MIN_FS)
    fig.tight_layout()
    save_paper(fig, OUT_DIR, "fig_perclass_f1")


def fig_confusion() -> None:
    """Confusion ternormalisasi baris (seed 42, TTA terpilih), sel I->G ditandai."""
    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 1.72), layout="constrained")
    abbr = [CLASS_ABBR[c] for c in CLASS_NAMES]
    for k, (ax, m) in enumerate(zip(axes, MODELS)):
        cm = np.asarray(_metrics(m)["evaluation"]["test"][SEL_TTA]["confusion"], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                v = norm[i, j]
                # Dua desimal tanpa nol di depan agar muat di sel pada 7 pt.
                txt = "1.00" if v >= 0.995 else f"{v:.2f}".lstrip("0")
                ax.text(j, i, txt, ha="center", va="center", fontsize=MIN_FS - 0.5 if v >= 0.995 else MIN_FS,
                        color="white" if v > 0.55 else INK)
        ti, pj = CLASS_NAMES.index("insect_damage"), CLASS_NAMES.index("good")
        ax.add_patch(Rectangle((pj - 0.5, ti - 0.5), 1, 1, fill=False, edgecolor=HIGHLIGHT, linewidth=1.2))
        ax.set_xticks(range(NUM_CLASSES), abbr)
        ax.set_yticks(range(NUM_CLASSES), abbr if k == 0 else [])
        ax.tick_params(length=0, labelsize=MIN_FS, pad=1.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f'({"abc"[k]}) {MODEL_LABELS[m]}', fontsize=MIN_FS, pad=3)
        ax.set_xlabel("Predicted", fontsize=MIN_FS, labelpad=1)
        if k == 0:
            ax.set_ylabel("True", fontsize=MIN_FS, labelpad=1)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=14, pad=0.02, ticks=[0, 0.5, 1])
    cbar.ax.tick_params(labelsize=MIN_FS, length=1.5, width=0.4)
    cbar.outline.set_linewidth(0.4)
    save_paper(fig, OUT_DIR, "fig_confusion")


def fig_tta() -> None:
    """F1 makro & miss rate: tanpa TTA vs TTA terpilih, mean ± std lintas 5 seed."""
    df = _per_run()
    fig, (ax_f1, ax_mr) = plt.subplots(2, 1, figsize=(COL_W, 3.0), sharex=True)
    x = np.arange(len(MODELS))
    width = 0.36
    panels = [(ax_f1, "f1", "Macro F1-score", 1.0, "{:.3f}", 0.5),
              (ax_mr, "defect_miss_rate", "Defect miss rate (%)", 100.0, "{:.1f}", 0.0)]
    for ax, col, ylabel, mult, fmt, base in panels:
        tops = []
        for i, mode in enumerate(["none", SEL_TTA]):
            sub = df[df.tta == mode]
            mean = np.array([sub[sub.model == m][col].mean() * mult for m in MODELS])
            std = np.nan_to_num(np.array([sub[sub.model == m][col].std(ddof=1) * mult for m in MODELS]))
            pos = x + (i - 0.5) * width
            ax.bar(pos, mean - base, bottom=base, width=width - 0.04, color=TTA_COLORS[mode],
                   label=TTA_NAMES[mode], linewidth=0)
            ax.errorbar(pos, mean, yerr=std, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.4, capthick=0.6)
            tops.append(mean + std)
            for xp, v, s in zip(pos, mean, std):
                ax.text(xp, v + s + (0.004 if col == "f1" else 0.3), fmt.format(v), ha="center", va="bottom",
                        fontsize=MIN_FS, color=INK)
        top = max(t.max() for t in tops)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS], fontsize=MIN_FS + 1)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(base, base + (top - base) * 1.18)
        ax.set_ylabel(ylabel)
    panel_label(ax_f1, "(a)")
    panel_label(ax_mr, "(b)")
    handles, labels = ax_f1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.97), handlelength=1.0,
               fontsize=MIN_FS)
    fig.tight_layout(h_pad=0.8)
    save_paper(fig, OUT_DIR, "fig_tta")


def fig_tradeoff() -> None:
    """F1 makro (mean ± std, TTA terpilih) vs ukuran model; luas bubble = jumlah parameter."""
    df = _per_run()
    sub_tta = df[df.tta == SEL_TTA]
    eff = _results()[PROFILE["support"]]["efficiency"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    ys = []
    for m in MODELS:
        xv = eff[m]["model_size_mb"]
        yv, ystd = sub_tta[sub_tta.model == m]["f1"].mean(), sub_tta[sub_tta.model == m]["f1"].std(ddof=1)
        ys += [yv - ystd, yv + ystd]
        pm = eff[m]["params_total"] / 1e6
        ax.errorbar(xv, yv, yerr=ystd, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5, zorder=4)
        ax.scatter(xv, yv, s=pm * 22, color=MODEL_COLORS[m], alpha=0.85, edgecolor="white", linewidth=1.0, zorder=3)
        offset, ha, va = ((14, 0), "left", "center") if m == "mobilenetv3" else ((-20, 0), "right", "center")
        ax.annotate(f"{MODEL_LABELS[m]}\n{pm:.1f}M params", (xv, yv), xytext=offset, textcoords="offset points",
                    ha=ha, va=va, fontsize=MIN_FS, color=INK)
    xs = np.array([eff[m]["model_size_mb"] for m in MODELS])
    pad = (xs.max() - xs.min()) * 0.15
    ax.set_xlim(max(0, xs.min() - pad), xs.max() + pad)
    span = max(ys) - min(ys)
    ax.set_ylim(min(ys) - 0.35 * span, max(ys) + 0.35 * span)
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Macro F1-score\n(test, mean ± std)")
    ax.grid(True, axis="both")
    fig.tight_layout()
    save_paper(fig, OUT_DIR, "fig_tradeoff")


def _joined_predictions() -> pd.DataFrame:
    """Tabel per biji test: prediksi & confidence TTA terpilih seed 42 ketiga model."""
    joined = None
    for m in MODELS:
        p = load_run_predictions(_run(m) / "predictions.npz")["test"]
        probs = softmax(p["logits"][SEL_TTA].astype(np.float64))
        frame = pd.DataFrame({
            "name": p["paths"], "true": [CLASS_NAMES[i] for i in p["y_true"]],
            f"pred_{m}": [CLASS_NAMES[i] for i in probs.argmax(1)],
            f"conf_{m}": probs.max(1), f"pgood_{m}": probs[:, 0],
        })
        joined = frame if joined is None else joined.merge(frame, on=["name", "true"], validate="one_to_one")
    joined["path"] = [str(CROPS_DIR / "test" / t / n) for n, t in zip(joined.name, joined.true)]
    joined["all_correct"] = np.all([joined[f"pred_{m}"] == joined["true"] for m in MODELS], axis=0)
    joined["n_good"] = np.sum([joined[f"pred_{m}"] == "good" for m in MODELS], axis=0)
    joined["min_conf"] = joined[[f"conf_{m}" for m in MODELS]].min(axis=1)
    joined["mean_pgood"] = joined[[f"pgood_{m}" for m in MODELS]].mean(axis=1)
    return joined


def fig_gradcam() -> dict:
    """Grad-CAM contoh benar (a) dan gagal lubang kecil (b), model seed 42.

    Aturan pemilihan (otomatis): (a) biji ``black`` dan ``insect_damage`` yang
    benar di ketiga model dengan confidence minimum antar-model tertinggi;
    (b) dua biji ``insect_damage`` yang diprediksi ``good`` oleh ketiga model
    dengan rata-rata P(good) tertinggi. Bila kurang dari dua, diambil yang
    diprediksi ``good`` oleh minimal dua model (aturan dicatat di JSON).
    """
    from src.analysis.gradcam import GradCAMAnalyzer
    from src.models.model_factory import get_model
    from src.utils.io import load_checkpoint

    joined = _joined_predictions()
    insect = joined[joined.true == "insect_damage"]
    black = joined[(joined.true == "black") & joined.all_correct].sort_values("min_conf", ascending=False)
    ins_ok = insect[insect.all_correct].sort_values("min_conf", ascending=False)
    all3 = insect[insect.n_good == 3].sort_values("mean_pgood", ascending=False)
    if len(all3) >= 2:
        miss, rule_b = all3, "insect_damage -> good di ketiga model, rata-rata P(good) tertinggi"
    else:
        miss = insect[insect.n_good >= 2].sort_values(["n_good", "mean_pgood"], ascending=False)
        rule_b = (f"hanya {len(all3)} biji salah di ketiga model; diambil insect_damage -> good di "
                  ">= 2 model, urut jumlah model lalu rata-rata P(good)")
    row_a = [black.name.iloc[0], ins_ok.name.iloc[0]]
    row_b = list(miss.name.iloc[:2])

    analyzers = {}
    for m in MODELS:
        net = get_model(m, num_classes=NUM_CLASSES, pretrained=False)
        load_checkpoint(net, next(_run(m).glob("*.pt")), map_location=DEVICE)
        analyzers[m] = GradCAMAnalyzer(net, m, device=DEVICE)

    by_name = joined.set_index("name")
    samples = [("(a)", n) for n in row_a] + [("(b)", n) for n in row_b]
    ncols = 1 + len(MODELS)
    cell = COL_W / ncols
    fig, axes = plt.subplots(len(samples), ncols, figsize=(COL_W, cell * len(samples) * 1.36 + 0.2),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.42})
    for r, (group, name) in enumerate(samples):
        info = by_name.loc[name]
        results = {m: analyzers[m].explain(Path(info["path"]), target_class=CLASS_NAMES.index(info[f"pred_{m}"]))
                   for m in MODELS}
        ax = axes[r, 0]
        ax.imshow(results[MODELS[0]]["rgb"])
        ax.set_xlabel(f"true:\n{CLASS_LABELS[info['true']]}", fontsize=MIN_FS, labelpad=1.5, linespacing=1.0)
        if r == 0:
            ax.set_title("Input", fontsize=MIN_FS, pad=2)
        if r == 0 or samples[r - 1][0] != group:
            ax.text(-0.1, 0.5, group, transform=ax.transAxes, ha="right", va="center", fontweight="bold")
        for c, m in enumerate(MODELS, start=1):
            ax = axes[r, c]
            ax.imshow(results[m]["overlay"])
            pred, conf = info[f"pred_{m}"], info[f"conf_{m}"]
            ax.set_xlabel(f"{CLASS_LABELS[pred]}\n{conf:.2f}", fontsize=MIN_FS, labelpad=1.5, linespacing=1.0,
                          color="#c0392b" if pred != info["true"] else INK)
            if r == 0:
                ax.set_title(MODEL_LABELS[m], fontsize=MIN_FS, pad=2)
        for ax in axes[r]:
            _clean_axes(ax)
    save_paper(fig, OUT_DIR, "fig_gradcam", formats=("png", "pdf"))
    return {"tta": SEL_TTA, "row_a": row_a, "row_b": row_b, "rule_a": "black & insect_damage benar di 3 model, "
            "confidence minimum antar-model tertinggi", "rule_b": rule_b,
            "n_insect_good_all3": int(len(all3)), "n_insect_good_ge2": int((insect.n_good >= 2).sum()),
            "row_b_n_models_good": [int(by_name.loc[n, "n_good"]) for n in row_b],
            "labels": {n: {m: [by_name.loc[n, f"pred_{m}"], float(by_name.loc[n, f"conf_{m}"])] for m in MODELS}
                       for n in row_a + row_b}}


def fig_confidence(model: str = "resnet50") -> dict:
    """(a) histogram confidence benar vs salah, (b) t-SNE fitur pooled (seed 42)."""
    from src.analysis.feature_visualization import compute_tsne, load_features

    p = load_run_predictions(_run(model) / "predictions.npz")["test"]
    probs = softmax(p["logits"][SEL_TTA].astype(np.float64))
    conf = probs.max(1)
    correct = probs.argmax(1) == p["y_true"]
    features, labels = load_features(RUNS_DIR / PROFILE["summary"] / f"{model}_seed42_test_features.npz")
    emb = compute_tsne(features, random_seed=RANDOM_SEED)
    sil = _results()[PROFILE["support"]]["silhouette"][model]["seed42"]

    fig, (ax_h, ax_p) = plt.subplots(1, 2, figsize=(COL_W, 2.15), gridspec_kw={"width_ratios": [1.05, 1]})
    bins = np.linspace(0.2, 1.0, 17)
    for key, mask in (("correct", correct), ("incorrect", ~correct)):
        c = conf[mask]
        w = np.full(len(c), 100.0 / len(c))
        ax_h.hist(c, bins=bins, weights=w, color=CONF_COLORS[key], alpha=0.45, linewidth=0)
        ax_h.hist(c, bins=bins, weights=w, histtype="step", color=CONF_COLORS[key], linewidth=1.0,
                  label=f"{key} (n={len(c)})")
        ax_h.axvline(c.mean(), color=CONF_COLORS[key], linestyle=(0, (3, 1.5)), linewidth=1.0)
    ymax = ax_h.get_ylim()[1] * 1.55
    ax_h.set_ylim(0, ymax)
    means = {k: conf[m].mean() for k, m in (("correct", correct), ("incorrect", ~correct))}
    for key, mean in means.items():
        ax_h.text(mean, ymax, f"{mean:.2f}", ha="center", va="bottom", fontsize=MIN_FS, color=CONF_COLORS[key])
    ax_h.set_xlabel("Prediction confidence")
    ax_h.set_ylabel("Share of group (%)")
    ax_h.set_xlim(0.2, 1.0)
    leg = ax_h.legend(loc="upper left", fontsize=MIN_FS, handlelength=1.0, borderaxespad=0.1, handletextpad=0.4,
                      frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    leg.set_zorder(10)
    panel_label(ax_h, "(a)", y=1.08)
    for idx, name in enumerate(CLASS_NAMES):
        sel = labels == idx
        ax_p.scatter(emb[sel, 0], emb[sel, 1], s=3.5, marker=CLASS_MARKERS[name], color=CLASS_COLORS[name],
                     alpha=0.75, linewidths=0, label=CLASS_LABELS[name])
    _clean_axes(ax_p)
    for spine in ax_p.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.5); spine.set_color(NEUTRAL)
    ax_p.set_xlabel("t-SNE 1")
    ax_p.set_ylabel("t-SNE 2")
    panel_label(ax_p, "(b)", y=1.08)
    handles, names = ax_p.get_legend_handles_labels()
    fig.tight_layout(w_pad=1.0, rect=(0, 0.1, 1, 1))
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=MIN_FS,
               markerscale=2.2, handletextpad=0.1, columnspacing=0.8)
    save_paper(fig, OUT_DIR, "fig_confidence")
    return {"model": model, "tta": SEL_TTA, "silhouette_seed42": sil, "n_correct": int(correct.sum()),
            "n_incorrect": int((~correct).sum()), "mean_conf_correct": float(means["correct"]),
            "mean_conf_incorrect": float(means["incorrect"])}


def fig_bn_effect() -> None:
    """Opsional: F1 makro & miss rate bn_adapt vs bn_frozen (mean ± std, TTA terpilih)."""
    df = _all_runs()
    df = df[(df.split == "test") & (df.tta == SEL_TTA)]
    conds = [("main", "BN adapted (bn_adapt)", 0.35), ("bn_frozen", "BN frozen (bn_frozen)", 1.0)]
    fig, (ax_f1, ax_mr) = plt.subplots(2, 1, figsize=(COL_W, 3.0), sharex=True)
    x = np.arange(len(MODELS))
    width = 0.36
    for ax, col, ylabel, mult, fmt, base in ((ax_f1, "f1", "Macro F1-score", 1.0, "{:.3f}", 0.5),
                                             (ax_mr, "defect_miss_rate", "Defect miss rate (%)", 100.0, "{:.1f}", 0.0)):
        tops = []
        for i, (cfg, _, alpha) in enumerate(conds):
            sub = df[df.config == cfg]
            mean = np.array([sub[sub.model == m][col].mean() * mult for m in MODELS])
            std = np.nan_to_num(np.array([sub[sub.model == m][col].std(ddof=1) * mult for m in MODELS]))
            pos = x + (i - 0.5) * width
            ax.bar(pos, mean - base, bottom=base, width=width - 0.04, color=[MODEL_COLORS[m] for m in MODELS],
                   alpha=alpha, linewidth=0)
            ax.errorbar(pos, mean, yerr=std, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.4, capthick=0.6)
            tops.append(mean + std)
            for xp, v, s in zip(pos, mean, std):
                ax.text(xp, v + s + (0.004 if col == "f1" else 0.3), fmt.format(v), ha="center", va="bottom",
                        fontsize=MIN_FS, color=INK)
        top = max(t.max() for t in tops)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS], fontsize=MIN_FS + 1)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(base, base + (top - base) * 1.18)
        ax.set_ylabel(ylabel)
    panel_label(ax_f1, "(a)")
    panel_label(ax_mr, "(b)")
    handles = [Patch(facecolor=INK_2, alpha=a, label=lab) for _, lab, a in conds]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.97), handlelength=1.0,
               fontsize=MIN_FS)
    fig.tight_layout(h_pad=0.8)
    save_paper(fig, OUT_DIR, "fig_bn_effect")


def main() -> None:
    """Buat figur yang diminta (default semua)."""
    parser = argparse.ArgumentParser(description="Figur paper revisi.")
    parser.add_argument("--main-config", default="main", choices=list(PROFILES))
    parser.add_argument("--only", nargs="+", choices=FIGURES, default=None)
    parser.add_argument("--runs-dir", default=str(RUNS_DIR), help="Root run (uji: folder lain).")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    profile = PROFILES[args.main_config]
    results_path = PROJECT_ROOT / "revision_results.json"
    data = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    sel = data.get(profile["agg"], {}).get("tta_selection", {}).get("uniform_choice", "d4")
    globals().update(CONFIG=args.main_config, PROFILE=profile, SEL_TTA=sel, RUNS_DIR=Path(args.runs_dir),
                     OUT_DIR=Path(args.out_dir) if args.out_dir else FIGURES_DIR / profile["out"])
    only = args.only or [f for f in FIGURES if f != "fig_bn_effect" or args.main_config == "bn_frozen"]
    paper_style()
    meta = {"config": CONFIG, "tta": SEL_TTA, "min_font_pt": MIN_FS}
    for name in only:
        result = globals()[name]()
        if isinstance(result, dict):
            meta[name] = result
        print(f"[ok] {name}")
    data = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    data.setdefault(profile["figures"], {}).update(meta)
    results_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"Figur ({CONFIG}, TTA {SEL_TTA}) tersimpan di {OUT_DIR}")


if __name__ == "__main__":
    main()

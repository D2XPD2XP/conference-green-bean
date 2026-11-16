"""Modul analisis: interpretasi, visualisasi, dan diagnosis hasil model.

Analisis diperlakukan setara dengan modul pelatihan, bukan pelengkap. Isinya:

- :mod:`~src.analysis.gradcam` - apa yang dilihat model (Grad-CAM).
- :mod:`~src.analysis.confusion_analysis` - kelas apa tertukar dengan apa.
- :mod:`~src.analysis.confidence_analysis` - apakah model tahu saat ragu.
- :mod:`~src.analysis.error_analysis` - galeri biji yang salah diklasifikasi.
- :mod:`~src.analysis.feature_visualization` - ruang fitur (t-SNE/UMAP).
- :mod:`~src.analysis.efficiency_analysis` - trade-off efisiensi vs akurasi.
- :mod:`~src.analysis.learning_curves` - konvergensi & gejala overfitting.
- :mod:`~src.analysis.style` - warna kelas & gaya figur bersama.

Impor dibuat *lazy* lewat ``__getattr__``: memanggil
``from src.analysis import GradCAMAnalyzer`` tidak akan memuat umap atau
pytorch-grad-cam kalau yang dipakai hanya kurva pembelajaran.
"""

from __future__ import annotations

import importlib
from typing import Any

#: Pemetaan nama publik -> modul asalnya (untuk impor lazy).
_EXPORTS: dict[str, str] = {
    # gradcam
    "GradCAMAnalyzer": "src.analysis.gradcam",
    "compare_architectures": "src.analysis.gradcam",
    "resolve_target_layer": "src.analysis.gradcam",
    "select_sample_images": "src.analysis.gradcam",
    "select_correct_samples": "src.analysis.gradcam",
    # confusion
    "plot_confusion_matrix": "src.analysis.confusion_analysis",
    "plot_confusion_matrices_comparison": "src.analysis.confusion_analysis",
    "generate_classification_report_table": "src.analysis.confusion_analysis",
    "error_pattern_analysis": "src.analysis.confusion_analysis",
    "plot_error_pattern_heatmap": "src.analysis.confusion_analysis",
    # confidence
    "plot_confidence_distribution": "src.analysis.confidence_analysis",
    "plot_confidence_per_class": "src.analysis.confidence_analysis",
    "identify_uncertain_predictions": "src.analysis.confidence_analysis",
    "confidence_summary": "src.analysis.confidence_analysis",
    "accuracy_vs_threshold": "src.analysis.confidence_analysis",
    # error analysis
    "collect_misclassified_samples": "src.analysis.error_analysis",
    "plot_error_gallery": "src.analysis.error_analysis",
    "plot_class_gallery": "src.analysis.error_analysis",
    "error_summary_by_class": "src.analysis.error_analysis",
    # feature space
    "extract_features": "src.analysis.feature_visualization",
    "save_features": "src.analysis.feature_visualization",
    "load_features": "src.analysis.feature_visualization",
    "compute_tsne": "src.analysis.feature_visualization",
    "compute_umap": "src.analysis.feature_visualization",
    "plot_tsne_features": "src.analysis.feature_visualization",
    "plot_umap_features": "src.analysis.feature_visualization",
    "plot_feature_projection": "src.analysis.feature_visualization",
    "plot_projection_comparison": "src.analysis.feature_visualization",
    "cluster_separation_score": "src.analysis.feature_visualization",
    # efficiency
    "measure_model_efficiency": "src.analysis.efficiency_analysis",
    "count_flops": "src.analysis.efficiency_analysis",
    "efficiency_table": "src.analysis.efficiency_analysis",
    "plot_efficiency_accuracy_tradeoff": "src.analysis.efficiency_analysis",
    "plot_inference_time_comparison": "src.analysis.efficiency_analysis",
    "plot_size_vs_metric": "src.analysis.efficiency_analysis",
    # learning curves
    "load_history": "src.analysis.learning_curves",
    "load_all_histories": "src.analysis.learning_curves",
    "plot_learning_curves": "src.analysis.learning_curves",
    "plot_learning_curves_grid": "src.analysis.learning_curves",
    "convergence_summary": "src.analysis.learning_curves",
    # style
    "apply_plot_style": "src.analysis.style",
    "class_palette": "src.analysis.style",
    "save_figure": "src.analysis.style",
    "tensor_to_image": "src.analysis.style",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Impor atribut publik dari modulnya saat pertama kali dipakai."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.analysis' tidak punya atribut '{name}'")
    module = importlib.import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value  # cache agar impor berikutnya langsung.
    return value


def __dir__() -> list[str]:
    """Lengkapi autocomplete dengan seluruh nama publik."""
    return __all__

"""Unit test konsistensi konfigurasi.

Pemetaan kelas tersebar di beberapa dict (nama Indonesia, warna, deskripsi,
multiplier augmentasi). Menambah kelas baru tanpa memperbarui salah satunya
tidak akan error saat impor — gejalanya baru muncul jauh di dalam pipeline
sebagai ``KeyError`` atau warna acak di figur. Test di sini menangkapnya lebih
awal.
"""

from __future__ import annotations

from configs.base_config import (
    AUGMENT_SPLITS,
    AUGMENTATION_MULTIPLIERS,
    CLASS_COLORS,
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    CLASS_NAMES_ID,
    COPY_SPLITS,
    CROPS_DIR,
    CROPS_RAW_DIR,
    DEFECT_CLASSES,
    GOOD_CLASS,
    IMBALANCE_STRATEGIES,
    IMBALANCE_STRATEGY,
    NUM_CLASSES,
    SPLITS,
)
from configs.cnn_config import CLASSIFIER_ATTR, CNN_CONFIG, TARGET_LAYERS


def test_five_classes() -> None:
    """Proyek v2 memakai 5 kelas; NUM_CLASSES diturunkan dari CLASS_NAMES."""
    assert NUM_CLASSES == 5
    assert len(CLASS_NAMES) == 5
    assert NUM_CLASSES == len(CLASS_NAMES)


def test_class_names_unique() -> None:
    """Tidak ada nama kelas yang terduplikasi (indeks label harus unik)."""
    assert len(set(CLASS_NAMES)) == len(CLASS_NAMES)


def test_no_legacy_binary_class_names() -> None:
    """Nama kelas biner lama tidak boleh tersisa di konfigurasi."""
    assert "accepted" not in CLASS_NAMES
    assert "rejected" not in CLASS_NAMES


def test_every_class_has_mappings() -> None:
    """Tiap kelas punya nama Indonesia, deskripsi, warna, dan multiplier."""
    for name in CLASS_NAMES:
        assert name in CLASS_NAMES_ID, f"{name} tidak punya padanan Indonesia"
        assert name in CLASS_DESCRIPTIONS, f"{name} tidak punya deskripsi"
        assert name in CLASS_COLORS, f"{name} tidak punya warna"
        assert name in AUGMENTATION_MULTIPLIERS, f"{name} tidak punya multiplier"


def test_mappings_have_no_extra_keys() -> None:
    """Dict pemetaan tidak memuat kelas yang sudah tidak ada."""
    for mapping, label in (
        (CLASS_NAMES_ID, "CLASS_NAMES_ID"),
        (CLASS_DESCRIPTIONS, "CLASS_DESCRIPTIONS"),
        (CLASS_COLORS, "CLASS_COLORS"),
        (AUGMENTATION_MULTIPLIERS, "AUGMENTATION_MULTIPLIERS"),
    ):
        extra = set(mapping) - set(CLASS_NAMES)
        assert not extra, f"{label} memuat kelas asing: {extra}"


def test_colors_are_distinct_hex() -> None:
    """Warna kelas berupa hex valid dan tidak ada yang kembar."""
    colors = [CLASS_COLORS[name] for name in CLASS_NAMES]
    assert len(set(colors)) == len(colors)
    for color in colors:
        assert color.startswith("#") and len(color) == 7
        int(color[1:], 16)  # memastikan hex-nya valid


def test_multipliers_are_positive() -> None:
    """Multiplier augmentasi minimal 1 (1 = hanya salin citra asli)."""
    for name, value in AUGMENTATION_MULTIPLIERS.items():
        assert isinstance(value, int) and value >= 1, f"{name}: {value}"


def test_good_and_defect_classes_partition() -> None:
    """GOOD_CLASS + DEFECT_CLASSES membentuk seluruh himpunan kelas."""
    assert GOOD_CLASS in CLASS_NAMES
    assert GOOD_CLASS not in DEFECT_CLASSES
    assert set([GOOD_CLASS, *DEFECT_CLASSES]) == set(CLASS_NAMES)
    assert len(DEFECT_CLASSES) == NUM_CLASSES - 1


def test_crops_dirs_are_separate() -> None:
    """crops_raw dan crops wajib folder berbeda.

    Kalau sama, augmentasi akan membaca dan menulis di tempat yang sama dan
    varian hasil augmentasi bisa ikut diaugmentasi lagi saat run berikutnya.
    """
    assert CROPS_RAW_DIR != CROPS_DIR
    assert CROPS_RAW_DIR.name == "crops_raw"
    assert CROPS_DIR.name == "crops"


def test_augment_and_copy_splits_do_not_overlap() -> None:
    """Satu split tidak boleh diaugmentasi sekaligus disalin."""
    assert not set(AUGMENT_SPLITS) & set(COPY_SPLITS)
    assert set(AUGMENT_SPLITS) | set(COPY_SPLITS) == set(SPLITS)


def test_only_train_is_augmented() -> None:
    """Valid dan test tidak boleh ikut diaugmentasi."""
    assert AUGMENT_SPLITS == ("train",)
    assert "valid" in COPY_SPLITS and "test" in COPY_SPLITS


def test_default_imbalance_strategy_is_valid() -> None:
    """Strategi default harus termasuk daftar strategi yang didukung."""
    assert IMBALANCE_STRATEGY in IMBALANCE_STRATEGIES


def test_cnn_config_consistent() -> None:
    """Hyperparameter wajib seragam antar-arsitektur kecuali input_size."""
    assert set(CNN_CONFIG) == {"resnet50", "mobilenetv3", "inceptionv3"}

    shared_keys = ("epochs", "batch_size", "learning_rate", "optimizer", "loss", "num_classes")
    reference = CNN_CONFIG["resnet50"]
    for name, config in CNN_CONFIG.items():
        for key in shared_keys:
            assert config[key] == reference[key], (
                f"{name}.{key} berbeda dari resnet50; perbandingan arsitektur "
                "jadi tidak apple-to-apple"
            )

    assert CNN_CONFIG["inceptionv3"]["input_size"] == 299
    assert CNN_CONFIG["resnet50"]["input_size"] == 224
    assert CNN_CONFIG["mobilenetv3"]["input_size"] == 224


def test_analysis_layer_maps_cover_every_model() -> None:
    """Tiap arsitektur punya target layer Grad-CAM dan atribut head."""
    for name in CNN_CONFIG:
        assert name in TARGET_LAYERS, f"{name} tidak punya target layer Grad-CAM"
        assert name in CLASSIFIER_ATTR, f"{name} tidak punya atribut classifier"


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__, "-v"])

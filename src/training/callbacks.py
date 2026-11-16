"""Callback pelatihan: early stopping dan penyimpanan model terbaik."""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from src.utils.io import save_checkpoint


class EarlyStopping:
    """Hentikan pelatihan bila metrik pantauan tak membaik selama ``patience`` epoch.

    Args:
        patience: Jumlah epoch tanpa perbaikan sebelum berhenti.
        mode: ``"max"`` (metrik makin besar makin baik, mis. akurasi) atau
            ``"min"`` (mis. loss).
        min_delta: Ambang minimum perubahan agar dianggap perbaikan.

    Attributes:
        should_stop: ``True`` bila kriteria berhenti sudah terpenuhi.
        best: Nilai metrik terbaik sejauh ini.
    """

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 0.0) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode harus 'max' atau 'min'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        """Perbarui state dengan metrik epoch terbaru.

        Args:
            value: Nilai metrik yang dipantau.

        Returns:
            ``True`` bila epoch ini merupakan perbaikan (best baru).
        """
        if self._is_improvement(value):
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class ModelCheckpoint:
    """Simpan bobot model terbaik berdasarkan metrik pantauan.

    Penamaan file mengikuti pola:
    ``<model_name>_best_epoch{N}_<metric_tag>{X.XX}.pt`` (mis. ``acc0.80`` atau
    ``f10.75``). Perbandingan bersifat ketat, sehingga bila nilai seri epoch
    yang lebih awal dipertahankan.

    Args:
        output_dir: Direktori penyimpanan checkpoint.
        model_name: Nama model (untuk penamaan file).
        mode: ``"max"`` atau ``"min"``.
        metric_tag: Penanda metrik pada nama file (default ``"acc"``).

    Attributes:
        best_value: Nilai metrik terbaik yang tersimpan.
        best_path: Path checkpoint terbaik terkini.
    """

    def __init__(
        self,
        output_dir: str | Path,
        model_name: str,
        mode: str = "max",
        metric_tag: str = "acc",
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode harus 'max' atau 'min'")
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.mode = mode
        self.metric_tag = metric_tag
        self.best_value: float | None = None
        self.best_path: Path | None = None

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return value > self.best_value if self.mode == "max" else value < self.best_value

    def maybe_save(self, model: nn.Module, value: float, epoch: int) -> Path | None:
        """Simpan checkpoint bila ``value`` merupakan metrik terbaik.

        Checkpoint lama akan dihapus agar hanya menyimpan yang terbaik.

        Args:
            model: Model yang dievaluasi.
            value: Nilai metrik epoch ini (mis. ``val_acc``).
            epoch: Nomor epoch (1-indexed).

        Returns:
            Path checkpoint baru bila tersimpan, atau ``None``.
        """
        if not self._is_improvement(value):
            return None

        filename = f"{self.model_name}_best_epoch{epoch}_{self.metric_tag}{value:.2f}.pt"
        new_path = self.output_dir / filename
        save_checkpoint(model, new_path, epoch=epoch, metric=value, model_name=self.model_name)

        # Hapus checkpoint terbaik sebelumnya.
        if self.best_path is not None and self.best_path.exists():
            self.best_path.unlink()

        self.best_value = value
        self.best_path = new_path
        return new_path

"""Setup logging: keluaran ke konsol dan (opsional) file.

Training loop panjang sebaiknya menulis ke file, bukan hanya stdout.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from configs.base_config import LOGS_DIR

_DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    to_console: bool = True,
    attach_to: list[str] | None = None,
    mode: str = "a",
) -> logging.Logger:
    """Buat logger dengan handler konsol dan/atau file.

    Handler tidak diduplikasi bila logger dengan nama sama sudah dikonfigurasi.

    Args:
        name: Nama logger (biasanya ``__name__`` atau nama eksperimen).
        log_file: Nama file log. Bila relatif, diletakkan di ``results/logs/``.
            Bila ``None``, tidak menulis ke file.
        level: Level logging (default ``logging.INFO``).
        to_console: Bila ``True``, tambahkan handler stdout.
        attach_to: Nama logger lain yang ikut memakai handler yang sama, mis.
            ``["src"]`` agar log per-epoch dari ``src.training.train_cnn`` ikut
            tertulis ke file. Tanpa ini, logger modul di ``src/`` menulis ke root
            logger yang tidak punya handler, sehingga lognya hilang.
        mode: Mode file log. ``"a"`` = tambahkan ke log lama (default),
            ``"w"`` = timpa sehingga file hanya memuat satu run.

    Returns:
        Instance ``logging.Logger`` yang siap dipakai.

    Example:
        >>> logger = setup_logger("train_resnet50", "resnet50.log")
        >>> logger.info("mulai pelatihan")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:  # Sudah dikonfigurasi sebelumnya.
        return logger

    handlers: list[logging.Handler] = []

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    if to_console:
        # Konsol Windows default cp1252 dan melempar UnicodeEncodeError bila pesan
        # memuat karakter non-ASCII. Ganti karakter tak terwakili alih-alih gagal,
        # supaya pelatihan panjang tidak mati hanya karena satu baris log.
        stream = sys.stdout
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):  # pragma: no cover - bergantung terminal
                pass
        console = logging.StreamHandler(stream)
        console.setLevel(level)
        console.setFormatter(formatter)
        logger.addHandler(console)
        handlers.append(console)

    if log_file is not None:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_path = LOGS_DIR / log_path
        file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        handlers.append(file_handler)

    # Pasang handler yang sama ke logger lain (mis. paket "src") supaya log dari
    # modul di dalamnya tidak hilang ke root logger yang tak punya handler.
    for other_name in attach_to or []:
        other = logging.getLogger(other_name)
        other.setLevel(level)
        if not other.handlers:
            for handler in handlers:
                other.addHandler(handler)

    return logger

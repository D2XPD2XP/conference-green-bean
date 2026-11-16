"""Test logger: satu file log = satu run bila mode='w'."""

from __future__ import annotations

from pathlib import Path

from src.utils.logger import setup_logger


def test_logger_mode_w_keeps_only_one_run(tmp_path: Path) -> None:
    """mode='w' menimpa log sehingga satu file hanya memuat satu run."""
    log_file = tmp_path / "run.log"
    for i, name in enumerate(("run_a", "run_b")):
        logger = setup_logger(name, log_file=log_file, to_console=False, mode="w")
        logger.info("run ke-%d", i)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    text = log_file.read_text(encoding="utf-8")
    assert "run ke-1" in text and "run ke-0" not in text

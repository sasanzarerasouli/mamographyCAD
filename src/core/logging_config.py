"""Structured logging configuration for Mammography CAD."""
import logging
import sys
from pathlib import Path

from config import Config


def setup_logging(debug: bool = None) -> logging.Logger:
    """Configure structured logging for production or debug mode."""
    debug = Config.DEBUG_MODE if debug is None else debug
    level = logging.DEBUG if debug else logging.INFO

    log_dir = Path(Config.OUTPUT_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("MammographyCAD")
    root.handlers.clear()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_dir / "mammography_cad.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return root

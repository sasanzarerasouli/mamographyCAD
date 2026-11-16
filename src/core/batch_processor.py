"""Batch processing of multiple mammograms in a background thread.

Each file is analysed with a shared :class:`MammographyCADSystem` (models are
loaded once) and a JSON + HTML report is written to ``outputs/reports``.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from .cad_system import MammographyCADSystem
from ..reporting.report_builder import ReportBuilder

logger = logging.getLogger("MammographyCAD.batch")


class BatchProcessor(QThread):
    """Analyse a list of files sequentially without blocking the GUI."""

    progress = pyqtSignal(int, int, str)      # index, total, filename
    file_done = pyqtSignal(str, str, str)     # path, status, message
    batch_done = pyqtSignal(int, int, list)   # ok, failed, failures

    def __init__(
        self,
        paths: List[str],
        export_reports: bool = True,
        output_dir: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.paths = list(paths)
        self.export_reports = export_reports
        self._cancelled = False
        self.cad_system = MammographyCADSystem()
        self.report_builder = ReportBuilder(output_dir=output_dir)

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        ok = 0
        failures: List[str] = []
        total = len(self.paths)

        for index, path in enumerate(self.paths, start=1):
            if self._cancelled:
                break
            self.progress.emit(index, total, os.path.basename(path))
            try:
                result = self.cad_system.analyze(path)
                message = (
                    f"{len(result.detections)} detection(s), "
                    f"BI-RADS {(result.birads or {}).get('category')}"
                )
                if self.export_reports:
                    paths = self.report_builder.save(result)
                    message += f", report: {os.path.basename(paths['html'])}"
                ok += 1
                self.file_done.emit(path, "ok", message)
            except Exception as exc:  # noqa: BLE001 - reported per file
                logger.exception("Batch item failed: %s", path)
                failures.append(f"{os.path.basename(path)}: {exc}")
                self.file_done.emit(path, "failed", str(exc))

        self.batch_done.emit(ok, len(failures), failures)

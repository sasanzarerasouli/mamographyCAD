"""Background analysis worker.

The worker owns a :class:`MammographyCADSystem` instance and runs the complete
pipeline in a separate thread so the GUI never blocks.  Two signals are
provided:

* ``result_ready`` – emits the full :class:`AnalysisResult` object (preferred).
* ``finished``     – legacy tuple signal kept for backward compatibility with
  older UI code and external scripts.
"""
import logging
import time
from typing import Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .analysis_result import AnalysisResult
from .cad_system import MammographyCADSystem
from .detection_types import Detection

logger = logging.getLogger("MammographyCAD.worker")


class CADWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, list, dict, float)
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, image_path: str, parent=None, cad_system: Optional[MammographyCADSystem] = None):
        super().__init__(parent)
        self.image_path = image_path
        self.cad_system = cad_system or MammographyCADSystem()
        self._is_cancelled = False
        self.result: Optional[AnalysisResult] = None

    def cancel(self) -> None:
        """Request cancellation (checked at every pipeline stage boundary)."""
        self._is_cancelled = True

    def run(self) -> None:  # noqa: D401 - Qt entry point
        try:
            start_time = time.time()

            def progress_callback(value: int) -> None:
                if self._is_cancelled:
                    raise _Cancelled()
                self.progress.emit(int(value))

            result = self.cad_system.analyze(self.image_path, progress_callback)
            processing_time = time.time() - start_time
            result.processing_time = processing_time

            if self._is_cancelled:
                self.cancelled.emit()
                return

            self.result = result
            self.result_ready.emit(result)

            # Legacy signal (explicit signature kept for old consumers)
            self.finished.emit(
                result.original,
                result.enhanced,
                result.breast_mask,
                list(result.detections),
                dict(result.statistics),
                processing_time,
            )
        except _Cancelled:
            logger.info("CAD processing cancelled by user")
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - reported through the signal
            logger.exception("CAD processing failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")


class _Cancelled(Exception):
    """Internal control-flow exception used to abort the pipeline."""

"""Deep-learning detection backend (YOLO family and RT-DETR via ultralytics).

The backend is deliberately optional: the classical detectors remain the
fallback whenever no checkpoint or no runtime is available, so the application
always starts.  When a checkpoint *is* available, this module performs the
detection strategy the task requires:

* **Mass detection** – full-image inference with a second, larger scale.
* **Calcification detection** – overlapping tiled inference (small-object
  strategy), so microcalcifications are never resampled down to a few pixels,
  merged with a full-image pass for context, followed by class-aware NMS.

Everything runs on CPU by default (``Config.DL_DEVICE``).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import Config
from ..core.detection_types import Detection, DetectionType
from .model_registry import ModelInfo, ModelRegistry, deep_inference_available

logger = logging.getLogger("MammographyCAD.deep")


class BackendUnavailable(RuntimeError):
    """Raised when deep inference is requested but cannot be performed."""


@dataclass
class RawDetection:
    """A single detection in full-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int = 0
    class_name: str = ""
    source: str = "deep"

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x2 + self.x1) / 2.0, (self.y2 + self.y1) / 2.0


# ---------------------------------------------------------------------- #
#  NMS                                                                   #
# ---------------------------------------------------------------------- #
def nms(detections: Sequence[RawDetection], iou_threshold: float = 0.45) -> List[RawDetection]:
    """Class-aware greedy non-maximum suppression."""
    if not detections:
        return []
    output: List[RawDetection] = []
    by_class: Dict[int, List[RawDetection]] = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)

    for _class_id, group in by_class.items():
        pending = sorted(group, key=lambda d: d.score, reverse=True)
        while pending:
            best = pending.pop(0)
            output.append(best)
            pending = [
                cand for cand in pending
                if _iou(best, cand) <= iou_threshold
            ]
    return sorted(output, key=lambda d: d.score, reverse=True)


def _iou(a: RawDetection, b: RawDetection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------- #
#  Backend                                                               #
# ---------------------------------------------------------------------- #
class DeepDetectionBackend:
    """Wraps an ultralytics model for mammography inference."""

    def __init__(
        self,
        model_info: ModelInfo,
        task: str,
        device: Optional[str] = None,
    ) -> None:
        self.model_info = model_info
        self.task = task              # "mass" | "calcification"
        self.device = device or Config.DL_DEVICE
        self._model = None
        self._class_ids: List[int] = []
        self._class_names: List[str] = []
        self.loaded = False
        self.load_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    #  Loading                                                           #
    # ------------------------------------------------------------------ #
    def load(self) -> "DeepDetectionBackend":
        if self.loaded:
            return self
        if not deep_inference_available():
            self.load_error = (
                "deep inference unavailable: install torch + ultralytics"
            )
            raise BackendUnavailable(self.load_error)
        if self.model_info is None or not self.model_info.class_names:
            self.load_error = "checkpoint class names are unknown; refusing to guess"
            raise BackendUnavailable(self.load_error)

        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.model_info.path, task="detect")
        except Exception as exc:  # pragma: no cover - depends on runtime
            self.load_error = f"cannot load {self.model_info.filename}: {exc}"
            raise BackendUnavailable(self.load_error) from exc

        # Route only the classes that belong to this task.
        tokens = (
            Config.MASS_CLASS_TOKENS
            if self.task == "mass"
            else Config.CALC_CLASS_TOKENS
        )
        names = self.model_info.class_names
        self._class_ids = [
            idx for idx, name in enumerate(names)
            if any(tok in name.lower() for tok in tokens)
        ]
        self._class_names = [names[i] for i in self._class_ids]
        if not self._class_ids:
            self.load_error = (
                f"checkpoint {self.model_info.filename} has no class matching "
                f"the {self.task} vocabulary ({', '.join(names)})"
            )
            raise BackendUnavailable(self.load_error)

        self.loaded = True
        logger.info(
            "Loaded %s model %s for task '%s' (classes: %s)",
            self.model_info.file_format,
            self.model_info.filename,
            self.task,
            ", ".join(self._class_names),
        )
        return self

    @property
    def available(self) -> bool:
        return self.loaded

    def class_names(self) -> List[str]:
        return list(self._class_names)

    # ------------------------------------------------------------------ #
    #  Inference                                                         #
    # ------------------------------------------------------------------ #
    def predict(
        self,
        gray: np.ndarray,
        *,
        tile: bool = False,
        patch_size: int = 640,
        overlap: float = 0.30,
        scales: Iterable[int] = (1024,),
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[RawDetection]:
        """Run inference and return merged detections in full-image coordinates."""
        if not self.loaded:
            self.load()
        if gray is None or gray.size == 0:
            return []

        gray = _to_uint8(gray)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        height, width = gray.shape[:2]
        conf = Config.DL_CONF_THRESHOLD if conf is None else float(conf)
        iou = Config.DL_IOU_THRESHOLD if iou is None else float(iou)

        raw: List[RawDetection] = []

        if tile:
            raw.extend(
                self._predict_tiles(rgb, patch_size, overlap, conf, iou)
            )
        for scale in scales:
            raw.extend(
                self._predict_image(
                    rgb,
                    imgsz=_fit_imgsz(scale, height, width),
                    offset=(0.0, 0.0),
                    conf=conf,
                    iou=iou,
                )
            )

        merged = nms(raw, iou_threshold=iou)
        if len(merged) > Config.MAX_DETECTIONS_PER_TASK:
            merged = merged[: Config.MAX_DETECTIONS_PER_TASK]
        logger.info(
            "Deep %s inference on %dx%d -> %d detections (tiled=%s)",
            self.task, width, height, len(merged), tile,
        )
        return merged

    def _predict_tiles(
        self,
        rgb: np.ndarray,
        patch_size: int,
        overlap: float,
        conf: float,
        iou: float,
    ) -> List[RawDetection]:
        height, width = rgb.shape[:2]
        patch = int(max(128, min(patch_size, max(height, width))))
        stride = int(round(patch * (1.0 - float(np.clip(overlap, 0.0, 0.8)))))
        stride = max(1, stride)
        detections: List[RawDetection] = []

        y_positions = list(range(0, max(height - patch, 0) + 1, stride)) or [0]
        x_positions = list(range(0, max(width - patch, 0) + 1, stride)) or [0]
        # Always include the far edge so no strip is left unread.
        if y_positions[-1] + patch < height:
            y_positions.append(max(0, height - patch))
        if x_positions[-1] + patch < width:
            x_positions.append(max(0, width - patch))

        total = len(y_positions) * len(x_positions)
        logger.debug("Tiled inference: %d patches of %d px", total, patch)

        for y0 in y_positions:
            for x0 in x_positions:
                y1 = min(height, y0 + patch)
                x1 = min(width, x0 + patch)
                tile_img = rgb[y0:y1, x0:x1]
                detections.extend(
                    self._predict_image(
                        tile_img,
                        imgsz=int(patch),
                        offset=(float(x0), float(y0)),
                        conf=conf,
                        iou=iou,
                    )
                )
        return detections

    def _predict_image(
        self,
        image: np.ndarray,
        imgsz: int,
        offset: Tuple[float, float],
        conf: float,
        iou: float,
    ) -> List[RawDetection]:
        try:
            results = self._model.predict(  # type: ignore[union-attr]
                source=image,
                imgsz=int(imgsz),
                conf=float(conf),
                iou=float(iou),
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            logger.error("Deep inference failed on one window: %s", exc)
            return []

        detections: List[RawDetection] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            names = getattr(result, "names", {}) or {}
            for (bx1, by1, bx2, by2), score, class_id in zip(xyxy, scores, classes):
                if int(class_id) not in self._class_ids:
                    continue
                name = str(names.get(int(class_id), f"class_{class_id}"))
                detections.append(
                    RawDetection(
                        x1=bx1 + offset[0],
                        y1=by1 + offset[1],
                        x2=bx2 + offset[0],
                        y2=by2 + offset[1],
                        score=float(score),
                        class_id=int(class_id),
                        class_name=name,
                        source="deep",
                    )
                )
        return detections

    # ------------------------------------------------------------------ #
    #  Conversion                                                        #
    # ------------------------------------------------------------------ #
    def to_detections(
        self,
        raw: Sequence[RawDetection],
        image_shape: Tuple[int, int],
        breast_mask: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """Convert model output to the project's :class:`Detection` objects."""
        height, width = image_shape[:2]
        detections: List[Detection] = []
        for item in raw:
            x1 = int(round(max(0.0, min(item.x1, width - 1))))
            y1 = int(round(max(0.0, min(item.y1, height - 1))))
            x2 = int(round(max(0.0, min(item.x2, width - 1))))
            y2 = int(round(max(0.0, min(item.y2, height - 1))))
            if x2 <= x1 or y2 <= y1:
                continue

            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if breast_mask is not None and breast_mask.size == width * height:
                region = breast_mask[y1:y2, x1:x2]
                if region.size and not np.any(region):
                    continue

            is_calc = self.task == "calcification"
            if is_calc:
                dtype = (
                    DetectionType.MICROCALCIFICATION
                    if (x2 - x1) <= 40 and (y2 - y1) <= 40
                    else DetectionType.MACROCALCIFICATION
                )
            else:
                dtype = (
                    DetectionType.TUMOR_CONFIRMED
                    if item.score >= Config.CONFIRMED_THRESHOLD
                    else DetectionType.TUMOR_SUSPICIOUS
                )

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            detections.append(
                Detection(
                    type=dtype,
                    location=center,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    mask=mask,
                    confidence=float(item.score),
                    features={
                        "source": "deep",
                        "model": self.model_info.filename,
                        "model_sha256": self.model_info.sha256[:16],
                        "class_name": item.class_name,
                        "class_id": item.class_id,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "is_cluster": bool(is_calc),
                    },
                )
            )
        return detections


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    data = image.astype(np.float32)
    lo, hi = float(np.min(data)), float(np.max(data))
    if hi - lo < 1e-6:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    return ((data - lo) * (255.0 / (hi - lo))).clip(0, 255).astype(np.uint8)


def _fit_imgsz(target: int, height: int, width: int) -> int:
    """Keep the requested inference size a multiple of 32 and sane."""
    target = int(max(320, target))
    longest = max(height, width)
    if longest <= target:
        # Never upscale beyond 1.5x: it only burns CPU without new information
        target = int(min(target, math.ceil(longest * 1.5)))
    return int(max(320, (target // 32) * 32))


# ---------------------------------------------------------------------- #
#  Factory (cached per process)                                          #
# ---------------------------------------------------------------------- #
_BACKENDS: Dict[Tuple[str, str], Any] = {}
_LAST_ERRORS: Dict[str, str] = {}


def get_backend(
    task: str, registry: Optional[ModelRegistry] = None
) -> Optional[DeepDetectionBackend]:
    """Return a loaded backend for *task*, or ``None`` when unavailable."""
    task = task.lower()
    if Config.DETECTION_BACKEND == "off":
        return None

    registry = registry or ModelRegistry()
    info = registry.select_for(task)
    if info is None:
        _LAST_ERRORS[task] = (
            f"no usable {task} checkpoint found in {', '.join(registry.directories)}"
        )
        return None

    key = (task, info.path)
    cached = _BACKENDS.get(key)
    if isinstance(cached, DeepDetectionBackend) and cached.available:
        return cached

    try:
        backend = DeepDetectionBackend(info, task).load()
    except BackendUnavailable as exc:
        _LAST_ERRORS[task] = str(exc)
        logger.warning("Deep backend for '%s' unavailable: %s", task, exc)
        return None
    _BACKENDS[key] = backend
    _LAST_ERRORS.pop(task, None)
    return backend


def last_error(task: str) -> str:
    return _LAST_ERRORS.get(task.lower(), "")


def backend_status() -> Dict[str, Any]:
    """Describe the effective detection configuration (used by the UI)."""
    registry = ModelRegistry()
    status = {
        "configured_mode": Config.DETECTION_BACKEND,
        "deep_runtime": deep_inference_available(),
        "mass": {"backend": "classical", "model": None, "reason": ""},
        "calcification": {"backend": "classical", "model": None, "reason": ""},
    }
    for task in ("mass", "calcification"):
        backend = get_backend(task, registry)
        if backend is not None:
            status[task] = {
                "backend": "deep",
                "model": backend.model_info.filename,
                "classes": backend.class_names(),
                "reason": "",
            }
        else:
            status[task]["reason"] = last_error(task) or "deep backend disabled"
    return status

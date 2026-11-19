"""Unified analysis result container for the full CAD pipeline."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .detection_types import Detection


@dataclass
class AnalysisResult:
    """Complete output of mammography analysis pipeline."""

    original: np.ndarray
    enhanced: np.ndarray
    breast_mask: np.ndarray
    detections: List[Detection]
    statistics: Dict[str, Any]

    pectoral_removed: Optional[np.ndarray] = None
    pectoral_mask: Optional[np.ndarray] = None
    quality_report: Optional[Dict[str, Any]] = None
    segmentation_result: Optional[Dict[str, Any]] = None
    pectoral_result: Optional[Dict[str, Any]] = None
    density_result: Optional[Dict[str, Any]] = None
    risk_results: List[Dict[str, Any]] = field(default_factory=list)
    birads_result: Optional[Dict[str, Any]] = None
    heatmap: Optional[np.ndarray] = None
    dicom_metadata: Optional[Dict[str, Any]] = None
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    confidence_summary: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    backend_info: Optional[Dict[str, Any]] = None

    # ----------------------------
    # New fields (extended output)
    # ----------------------------
    segmentation_confidence: Optional[Dict[str, Any]] = None
    tumor_risk: Optional[Dict[str, Any]] = None
    # BI-RADS (new top-level field; keep birads_result for backward compatibility)
    birads: Optional[Dict[str, Any]] = None
    # Breast density (new top-level field; keep density_result for backward compatibility)
    breast_density: Optional[Dict[str, Any]] = None
    # Overall explainability (new top-level field; per-lesion explainability already exists in Detection)
    explainability: Optional[Dict[str, Any]] = None
    pectoral_confidence: Optional[float] = None
    # Backend (new top-level field; keep backend_info for backward compatibility)
    backend: Optional[Dict[str, Any]] = None

    # ----------------------------
    # Clinical analysis outputs
    # ----------------------------
    calcification_analysis: Optional[Dict[str, Any]] = None
    lesion_characterizations: List[Dict[str, Any]] = field(default_factory=list)
    heatmap_image: Optional[np.ndarray] = None
    density_heatmap: Optional[np.ndarray] = None
    model_status: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    image_metadata: Optional[Dict[str, Any]] = None

    source_pixels: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def to_legacy_tuple(self):
        """Backward-compatible return format for existing UI/worker."""
        return (
            self.original,
            self.enhanced,
            self.breast_mask,
            self.detections,
            self.statistics,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export (excludes numpy arrays)."""
        return {
            "statistics": self.statistics,
            "quality_report": self.quality_report,

            "segmentation_result": _strip_arrays(self.segmentation_result),
            "pectoral_result": _strip_arrays(self.pectoral_result),

            "density_result": self.density_result,
            "risk_results": self.risk_results,

            "birads_result": self.birads_result,

            "dicom_metadata": self.dicom_metadata,
            "measurements": self.measurements,
            "confidence_summary": self.confidence_summary,
            "processing_time": self.processing_time,

            # Backward-compatible backend field
            "backend_info": self.backend_info,
            # New backend field
            "backend": self.backend,

            # ----------------------------
            # New fields in export
            # ----------------------------
            "segmentation_confidence": self.segmentation_confidence,
            "tumor_risk": self.tumor_risk,
            "birads": self.birads,
            "breast_density": self.breast_density,
            "explainability": self.explainability,
            "pectoral_confidence": self.pectoral_confidence,

            # ----------------------------
            # Clinical analysis outputs
            # ----------------------------
            "calcification_analysis": self.calcification_analysis,
            "lesion_characterizations": self.lesion_characterizations,
            "model_status": self.model_status,
            "warnings": self.warnings,
            "image_metadata": self.image_metadata,

            "detections": [
                {
                    "lesion_id": d.lesion_id,
                    "type": d.type.value,
                    "location": d.location,
                    "bbox": d.bbox,
                    "confidence": d.confidence,
                    "is_ignored": d.is_ignored,
                    "diameter_mm": d.diameter_mm,
                    "area_mm2": d.area_mm2,
                    "explainability": d.explainability,
                    "risk_scores": d.risk_scores,
                    "features": _json_safe(d.features),
                    "reliability": d.reliability,
                    "uncertainty": d.uncertainty,
                }
                for d in self.detections
            ],
        }


def _strip_arrays(obj):
    if obj is None:
        return None
    return {k: v for k, v in obj.items() if not isinstance(v, np.ndarray)}

def _json_safe(obj, _depth: int = 0):
    """Best-effort conversion of nested analysis data to JSON-safe values."""
    if _depth > 6:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, _depth + 1) for v in obj]
    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

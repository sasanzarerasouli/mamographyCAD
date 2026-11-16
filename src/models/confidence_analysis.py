"""Confidence calibration and uncertainty analysis."""
import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("MammographyCAD.confidence")


class ConfidenceAnalysisModule:
    """Generate confidence, reliability, and uncertainty scores for predictions."""

    def analyze_detection(self, detection_features: Dict[str, float], base_confidence: float) -> Dict[str, float]:
        """Calibrate confidence for a single detection."""
        feature_values = list(detection_features.values()) if detection_features else [base_confidence]
        variance = float(np.var(feature_values)) if len(feature_values) > 1 else 0.1
        reliability = max(0.0, 1.0 - variance)
        uncertainty = min(1.0, variance + (1.0 - base_confidence) * 0.5)
        calibrated = base_confidence * reliability
        return {
            "confidence": round(calibrated, 3),
            "reliability": round(reliability, 3),
            "uncertainty": round(uncertainty, 3),
        }

    def analyze_module(self, scores: List[float], label: str) -> Dict[str, float]:
        """Analyze confidence for a pipeline module (segmentation, density, etc.)."""
        if not scores:
            return {"confidence": 0.0, "reliability": 0.0, "uncertainty": 1.0, "label": label}
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        reliability = max(0.0, 1.0 - std_score)
        uncertainty = min(1.0, std_score + (1.0 - mean_score) * 0.3)
        return {
            "confidence": round(mean_score * reliability, 3),
            "reliability": round(reliability, 3),
            "uncertainty": round(uncertainty, 3),
            "label": label,
        }

    def summarize(self, module_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate confidence across all modules."""
        return {
            "modules": module_results,
            "overall_confidence": round(
                float(np.mean([m["confidence"] for m in module_results.values()])) if module_results else 0.0,
                3,
            ),
            "overall_uncertainty": round(
                float(np.mean([m["uncertainty"] for m in module_results.values()])) if module_results else 1.0,
                3,
            ),
        }

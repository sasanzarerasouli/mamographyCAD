"""Mammogram image quality assessment before analysis."""
import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger("MammographyCAD.quality")


class MammogramQualityAnalyzer:
    """Detect acquisition and image quality issues."""

    def analyze(self, gray: np.ndarray, breast_mask: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate mammogram quality and return score, warnings, and confidence.
        Uses contrast, exposure, blur, and anatomy coverage metrics.
        """
        warnings: List[str] = []
        scores: List[float] = []

        contrast_score, contrast_warn = self._check_contrast(gray, breast_mask)
        scores.append(contrast_score)
        warnings.extend(contrast_warn)

        blur_score, blur_warn = self._check_blur(gray, breast_mask)
        scores.append(blur_score)
        warnings.extend(blur_warn)

        exposure_score, exposure_warn = self._check_exposure(gray, breast_mask)
        scores.append(exposure_score)
        warnings.extend(exposure_warn)

        anatomy_score, anatomy_warn = self._check_anatomy(gray, breast_mask)
        scores.append(anatomy_score)
        warnings.extend(anatomy_warn)

        artifact_score, artifact_warn = self._check_artifacts(gray, breast_mask)
        scores.append(artifact_score)
        warnings.extend(artifact_warn)

        # Segmentation sanity: a breast mask covering almost the whole frame
        # means the background/page was included. Results then cannot be
        # trusted and must not be reported as a normal assessment.
        coverage = float(np.count_nonzero(breast_mask)) / float(breast_mask.size)
        mask_warning = None
        if coverage > 0.78:
            mask_warning = (
                "Breast segmentation covers {:.0%} of the frame - background "
                "may be included (unsupported image framing or inverted "
                "presentation). Quantitative results are unreliable."
            ).format(coverage)
        elif coverage < 0.03:
            mask_warning = (
                "Breast segmentation covers less than 3% of the frame - the "
                "breast may be cropped or missing."
            )
        if mask_warning:
            warnings.append(mask_warning)
            scores.append(0.2)

        quality_score = float(np.mean(scores)) if scores else 0.0
        quality_confidence = 1.0 - float(np.std(scores)) if len(scores) > 1 else quality_score

        result = {
            "quality_score": round(quality_score, 3),
            "quality_confidence": round(max(0.0, min(1.0, quality_confidence)), 3),
            "analysis_limited": bool(mask_warning),
            "breast_coverage": round(coverage, 4),
            "warnings": warnings,
            "metrics": {
                "contrast": contrast_score,
                "blur": blur_score,
                "exposure": exposure_score,
                "anatomy": anatomy_score,
                "artifacts": artifact_score,
            },
        }
        logger.info("Quality analysis: score=%.2f warnings=%d", quality_score, len(warnings))
        return result

    def _roi(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        roi = gray[mask > 0]
        return roi if roi.size > 0 else gray.ravel()

    def _check_contrast(self, gray: np.ndarray, mask: np.ndarray) -> Tuple[float, List[str]]:
        roi = self._roi(gray, mask)
        std_val = float(np.std(roi))
        if std_val < 15:
            return 0.3, ["Low contrast detected"]
        if std_val < 25:
            return 0.6, ["Moderate contrast - image may be difficult to analyze"]
        return 1.0, []

    def _check_blur(self, gray: np.ndarray, mask: np.ndarray) -> Tuple[float, List[str]]:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        roi = lap[mask > 0] if np.any(mask > 0) else lap.ravel()
        variance = float(np.var(roi))
        if variance < 50:
            return 0.4, ["Motion blur or out-of-focus image suspected"]
        if variance < 100:
            return 0.7, ["Mild blur detected"]
        return 1.0, []

    def _check_exposure(self, gray: np.ndarray, mask: np.ndarray) -> Tuple[float, List[str]]:
        roi = self._roi(gray, mask)
        mean_val = float(np.mean(roi))
        warnings = []
        score = 1.0
        if mean_val < 40:
            score = 0.4
            warnings.append("Underexposure detected")
        elif mean_val > 220:
            score = 0.4
            warnings.append("Overexposure detected")
        elif mean_val < 60 or mean_val > 200:
            score = 0.7
            warnings.append("Suboptimal exposure")
        return score, warnings

    def _check_anatomy(self, gray: np.ndarray, mask: np.ndarray) -> Tuple[float, List[str]]:
        h, w = gray.shape
        coverage = float(np.sum(mask > 0)) / (h * w)
        warnings = []
        if coverage < 0.15:
            return 0.3, ["Missing or insufficient breast anatomy"]
        if coverage < 0.25:
            return 0.6, ["Cropped breast anatomy - partial field of view"]
        return 1.0, warnings

    def _check_artifacts(self, gray: np.ndarray, mask: np.ndarray) -> Tuple[float, List[str]]:
        roi = self._roi(gray, mask)
        saturated = float(np.sum(roi > 250)) / max(len(roi), 1)
        if saturated > 0.05:
            return 0.5, ["Acquisition artifacts or markers detected"]
        return 1.0, []

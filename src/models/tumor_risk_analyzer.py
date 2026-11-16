"""Clinical tumor risk scoring for detected lesions.

This module computes interpretable lesion features and derives a clinical-style
risk score with benign/suspicious/malignant probabilities plus an explanation.
"""
import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix

from config import Config

logger = logging.getLogger("MammographyCAD.risk")


class TumorRiskAnalyzer:
    """Calculate benign/suspicious/malignant probabilities from lesion features."""

    def analyze_lesion(self, gray: np.ndarray, detection, breast_mask: np.ndarray) -> Dict[str, Any]:
        features = self._extract_features(gray, detection, breast_mask)
        norm = self._normalize_features(features)

        weights = getattr(Config, "RISK_WEIGHTS", {}) or {}
        # Weighted sum -> risk_score in [0,1]
        risk_score = float(sum(norm[k] * weights.get(k, 0.0) for k in norm))
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        benign_prob, suspicious_prob, malignant_prob = self._probabilities_from_score(risk_score)
        category = self._risk_category(risk_score)
        explanation = self._generate_explanation(norm, features, risk_score, category)

        return {
            "risk": {
                "risk_score": round(risk_score * 100.0, 1),  # 0-100
                "risk_category": category,
            },
            "probabilities": {
                "benign": round(benign_prob, 3),
                "suspicious": round(suspicious_prob, 3),
                "malignant": round(malignant_prob, 3),
            },
            "features": features,
            "explanation": explanation,
            "disclaimer": getattr(Config, "DISCLAIMER", ""),
        }

    # -------------------------
    # Feature extraction
    # -------------------------
    def _extract_features(self, gray: np.ndarray, detection, breast_mask: np.ndarray) -> Dict[str, float]:
        x, y, w, h = detection.bbox or (0, 0, 10, 10)
        x, y, w, h = int(x), int(y), int(w), int(h)

        H, W = gray.shape[:2]
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        roi = gray[y:y + h, x:x + w]
        bg = gray[breast_mask > 0] if breast_mask is not None else gray.reshape(-1)

        # ---- shape features (from contour if available)
        shape_irregularity = 0.5
        circularity = 0.5
        spiculation = 0.0

        if getattr(detection, "contour", None) is not None and len(detection.contour) >= 3:
            contour = detection.contour

            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))

            if perimeter > 1e-6 and area > 0:
                circularity = float((4.0 * np.pi * area) / (perimeter ** 2))
                circularity = float(np.clip(circularity, 0.0, 1.0))
                shape_irregularity = float(np.clip(1.0 - circularity, 0.0, 1.0))

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 1e-6 and area > 0:
                # higher => more spiculated/less convex
                spiculation = float(np.clip(1.0 - (area / hull_area), 0.0, 1.0))

        # ---- edge sharpness (edge density inside ROI)
        if roi.size > 0:
            roi_u8 = self._to_uint8(roi)
            edges = cv2.Canny(roi_u8, 50, 150)
            edge_sharpness = float(np.count_nonzero(edges)) / float(max(roi.size, 1)) * 100.0
        else:
            edge_sharpness = 0.0

        # ---- contrast & density difference (difference between lesion ROI and breast background)
        if roi.size > 0 and bg.size > 0:
            roi_mean = float(np.mean(roi))
            bg_mean = float(np.mean(bg))
            contrast = float(abs(roi_mean - bg_mean))
        else:
            contrast = 0.0

        density_difference = float(contrast)

        # ---- texture entropy (true entropy from normalized GLCM)
        texture_entropy = 0.0
        if roi.size > 16:
            roi_u8 = self._to_uint8(roi)
            # reduce levels for stability/speed
            roi_q = (roi_u8 // 8).astype(np.uint8)  # 0..31 => 32 levels
            glcm = graycomatrix(
                roi_q,
                distances=[1],
                angles=[0],
                levels=32,
                symmetric=True,
                normed=True,
            )
            p = glcm[:, :, 0, 0].astype(np.float64)
            p = p / (p.sum() + 1e-12)
            texture_entropy = float(-np.sum(p * np.log2(p + 1e-12)))

        return {
            "shape_irregularity": float(shape_irregularity),
            "circularity": float(circularity),
            "spiculation": float(spiculation),
            "edge_sharpness": float(edge_sharpness),
            "contrast": float(contrast),
            "texture_entropy": float(texture_entropy),
            "density_difference": float(density_difference),
        }

    # -------------------------
    # Normalization + scoring
    # -------------------------
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Map features to risk-oriented 0..1 where higher means higher risk.
        """
        # Notes:
        # - circularity: higher circularity => lower risk; convert to (1-circularity)
        # - edge_sharpness/contrast/entropy/density_difference scaled to typical ranges
        norm = {
            "shape_irregularity": float(np.clip(features["shape_irregularity"], 0.0, 1.0)),
            "circularity": float(np.clip(1.0 - features["circularity"], 0.0, 1.0)),
            "spiculation": float(np.clip(features["spiculation"], 0.0, 1.0)),
            "edge_sharpness": float(np.clip(features["edge_sharpness"] / 100.0, 0.0, 1.0)),
            "contrast": float(np.clip(features["contrast"] / 80.0, 0.0, 1.0)),
            "texture_entropy": float(np.clip(features["texture_entropy"] / 5.0, 0.0, 1.0)),  # entropy often ~0..5
            "density_difference": float(np.clip(features["density_difference"] / 60.0, 0.0, 1.0)),
        }
        return norm

    def _probabilities_from_score(self, risk_score: float) -> Tuple[float, float, float]:
        """
        Convert risk_score in [0,1] to (benign, suspicious, malignant) probabilities.
        Uses a smooth mapping then normalizes to sum=1.
        """
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        # Malignant grows with score; benign decreases with score; suspicious peaks in middle.
        malignant = risk_score ** 1.4
        benign = (1.0 - risk_score) ** 1.2
        suspicious = max(0.0, 1.0 - (benign + malignant) / 1.2)

        total = benign + suspicious + malignant + 1e-12
        return benign / total, suspicious / total, malignant / total

    def _risk_category(self, score: float) -> str:
        if score < 0.35:
            return "Low Risk"
        if score < 0.65:
            return "Intermediate Risk"
        return "High Risk"

    # -------------------------
    # Explanation
    # -------------------------
    def _generate_explanation(
        self,
        norm: Dict[str, float],
        features: Dict[str, float],
        risk_score: float,
        category: str,
    ) -> Dict[str, Any]:
        """
        Structured explanation: top drivers + human-readable summary.
        """
        # rank drivers by normalized risk contribution (norm * weight)
        weights = getattr(Config, "RISK_WEIGHTS", {}) or {}
        contributions = []
        for k, v in norm.items():
            contributions.append((k, float(v) * float(weights.get(k, 0.0)), float(v)))
        contributions.sort(key=lambda t: t[1], reverse=True)

        key_map = {
            "shape_irregularity": "Shape irregularity",
            "circularity": "Low circularity",
            "spiculation": "Spiculation",
            "edge_sharpness": "Edge sharpness",
            "contrast": "Contrast",
            "texture_entropy": "Texture entropy",
            "density_difference": "Density difference",
        }

        # thresholded indicators
        indicators: List[str] = []
        if norm["shape_irregularity"] > 0.5:
            indicators.append("Irregular shape")
        if norm["spiculation"] > 0.4:
            indicators.append("Spiculated/non-convex margin")
        if norm["edge_sharpness"] > 0.5:
            indicators.append("Prominent/complex edges")
        if norm["density_difference"] > 0.5:
            indicators.append("Marked density difference vs background")
        if norm["contrast"] > 0.5:
            indicators.append("Strong intensity contrast")
        if norm["texture_entropy"] > 0.5:
            indicators.append("Heterogeneous texture (high entropy)")

        top = []
        for k, contrib, nv in contributions[:3]:
            top.append(
                {
                    "feature": k,
                    "label": key_map.get(k, k),
                    "normalized_risk": round(nv, 3),
                    "contribution": round(contrib, 4),
                    "raw_value": round(float(features.get(k, 0.0)), 4),
                }
            )

        summary = (
            f"{category} based on combined feature score."
            if indicators
            else "No strong high-risk indicators detected from extracted features."
        )

        return {
            "summary": summary,
            "indicators": indicators or ["No significant risk indicators"],
            "top_drivers": top,
            "score_basis": {
                "risk_score_0_1": round(float(risk_score), 4),
                "normalized_features": {k: round(float(v), 3) for k, v in norm.items()},
            },
        }

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """Convert ROI to uint8 safely (handles uint16/float)."""
        if img.dtype == np.uint8:
            return img
        if img.dtype == np.uint16:
            # scale 16-bit to 8-bit
            return (img / 257).astype(np.uint8)
        imgf = img.astype(np.float32)
        mn, mx = float(np.min(imgf)), float(np.max(imgf))
        if mx - mn < 1e-6:
            return np.zeros_like(imgf, dtype=np.uint8)
        out = (imgf - mn) * (255.0 / (mx - mn))
        return np.clip(out, 0, 255).astype(np.uint8)
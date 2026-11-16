"""Breast mass characterization using ACR BI-RADS lexicon descriptors.

Each detected mass is described with the three core lexicon features that drive
the BI-RADS assessment:

* **Shape** – round, oval, irregular
* **Margin** – circumscribed, obscured, microlobulated, indistinct, spiculated
* **Density** – high density, equal density, low density, fat-containing

plus supporting measurements (size, orientation, spiculation index,
lobulation).  All values are *software-derived measurements of image
features*; nothing here is a diagnosis.  Every descriptor carries the score it
was derived from so a radiologist can audit the decision.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("MammographyCAD.characterization")


class LesionCharacterizer:
    """Extract BI-RADS lexicon descriptors for mass detections."""

    # Decision thresholds (documented so they can be audited / tuned)
    SPICULATION_STRONG = 0.22        # convexity deficit
    SPIKE_COUNT_STRONG = 5
    MICROLOBULATION_COUNT = 4
    BOUNDARY_OBSCURED_GRADIENT = 12.0
    DENSITY_DIFFERENCE_HIGH = 18.0   # grey levels above the surrounding ring
    DENSITY_DIFFERENCE_LOW = -18.0

    def characterize(
        self,
        gray: np.ndarray,
        detection,
        breast_mask: Optional[np.ndarray],
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> Dict[str, Any]:
        """Return the lexicon description for one mass detection."""
        mask = self._lesion_mask(detection, gray.shape)
        if mask is None or cv2.countNonZero(mask) == 0:
            return self._empty("no segmentation available for this detection")

        contour = self._largest_contour(mask)
        if contour is None or len(contour) < 3:
            return self._empty("lesion contour could not be extracted")

        geometry = self._geometry(contour)
        radial = self._radial_profile(contour, geometry["centroid"])
        boundary = self._boundary_strength(gray, contour, mask)
        density = self._density(gray, mask, breast_mask, detection)

        shape, shape_conf, shape_scores = self._classify_shape(geometry, radial)
        margin, margin_conf, margin_scores = self._classify_margin(
            geometry, radial, boundary
        )

        row_spacing, col_spacing = self._spacing(pixel_spacing)
        long_axis_mm = geometry["max_diameter_px"] * float(
            np.mean([row_spacing, col_spacing])
        )
        width = geometry["min_area_rect"][1][0] * col_spacing
        height = geometry["min_area_rect"][1][1] * row_spacing
        orientation = self._orientation(width, height, geometry)

        result = {
            "available": True,
            "shape": shape,
            "shape_confidence": round(shape_conf, 3),
            "margin": margin,
            "margin_confidence": round(margin_conf, 3),
            "density": density["density"],
            "density_confidence": round(density["confidence"], 3),
            "density_difference": round(density["difference"], 2),
            "orientation": orientation,
            "spiculation_index": round(geometry["convexity_deficit"], 3),
            "lobulation_count": int(radial["lobule_count"]),
            "spike_count": int(radial["spike_count"]),
            "boundary_gradient": round(boundary, 2),
            "size_mm": round(float(long_axis_mm), 2),
            "size_wide_mm": round(float(max(width, height)), 2),
            "scores": {
                "shape": shape_scores,
                "margin": margin_scores,
                "density": density,
            },
            "descriptors": [
                f"Shape: {shape}",
                f"Margin: {margin}",
                f"Density: {density['density']}",
                f"Orientation: {orientation}",
            ],
            "lexicon_source": "ACR BI-RADS (software-assisted measurement)",
        }
        logger.debug(
            "Characterized lesion: shape=%s margin=%s density=%s",
            shape, margin, density["density"],
        )
        return result

    def characterize_all(
        self,
        gray: np.ndarray,
        detections: List,
        breast_mask: Optional[np.ndarray],
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> Dict[int, Dict[str, Any]]:
        """Characterize every mass-like detection, keyed by detection id."""
        output: Dict[int, Dict[str, Any]] = {}
        for detection in detections:
            if getattr(detection, "is_ignored", False):
                continue
            if not self.is_mass(detection):
                continue
            output[id(detection)] = self.characterize(
                gray, detection, breast_mask, pixel_spacing
            )
        return output

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def is_mass(detection) -> bool:
        name = getattr(getattr(detection, "type", None), "value", "") or ""
        return "tumor" in name or "mass" in name

    @staticmethod
    def _spacing(pixel_spacing) -> Tuple[float, float]:
        try:
            row, col = float(pixel_spacing[0]), float(pixel_spacing[1])
            if row <= 0 or col <= 0:
                raise ValueError
            return row, col
        except (TypeError, ValueError, IndexError):
            return 1.0, 1.0

    @staticmethod
    def _empty(reason: str) -> Dict[str, Any]:
        return {
            "available": False,
            "reason": reason,
            "shape": "not assessed",
            "margin": "not assessed",
            "density": "not assessed",
            "descriptors": [],
        }

    @staticmethod
    def _lesion_mask(detection, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        height, width = shape[:2]
        mask = getattr(detection, "mask", None)
        if mask is not None and mask.size:
            mask = (mask > 0).astype(np.uint8) * 255
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            return mask
        bbox = getattr(detection, "bbox", None)
        if bbox:
            x, y, w, h = [int(v) for v in bbox]
            generated = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(
                generated,
                (max(0, x), max(0, y)),
                (min(width - 1, x + max(1, w)), min(height - 1, y + max(1, h))),
                255,
                -1,
            )
            return generated
        return None

    @staticmethod
    def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def _geometry(contour: np.ndarray) -> Dict[str, Any]:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) or 1e-6
        rect = cv2.minAreaRect(contour)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circularity = (
            float(min(1.0, 4.0 * np.pi * area / (perimeter ** 2))) if perimeter > 0 else 0.0
        )
        pts = hull.reshape(-1, 2).astype(np.float64)
        max_diameter = 0.0
        if len(pts) >= 2:
            diff = pts[:, None, :] - pts[None, :, :]
            max_diameter = float(np.sqrt((diff ** 2).sum(axis=2)).max())
        return {
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "convexity_deficit": float(max(0.0, min(1.0, 1.0 - area / hull_area))),
            "solidity": float(min(1.0, area / hull_area)),
            "min_area_rect": (
                (float(rect[0][0]), float(rect[0][1])),
                (float(rect[1][0]), float(rect[1][1])),
                float(rect[2]),
            ),
            "centroid": (float(np.mean(contour[:, 0, 0])), float(np.mean(contour[:, 0, 1]))),
            "bounding_radius": float(radius),
            "max_diameter_px": max_diameter,
        }

    @staticmethod
    def _radial_profile(
        contour: np.ndarray, centroid: Tuple[float, float], bins: int = 360
    ) -> Dict[str, Any]:
        """Angular radius signature -> spicules, lobules and roughness."""
        pts = contour.reshape(-1, 2).astype(np.float64)
        cx, cy = centroid
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        angles = np.arctan2(dy, dx)
        radii = np.sqrt(dx ** 2 + dy ** 2)

        theta = np.linspace(-np.pi, np.pi, bins, endpoint=False)
        profile = np.zeros(bins, dtype=np.float64)
        idx = np.clip(((angles + np.pi) / (2 * np.pi) * bins).astype(int), 0, bins - 1)
        np.maximum.at(profile, idx, radii)
        # Interpolate empty bins
        valid = profile > 0
        if valid.sum() >= 3:
            profile = np.interp(
                np.arange(bins), np.flatnonzero(valid), profile[valid]
            )
        else:
            return {"lobule_count": 0, "spike_count": 0, "roughness": 0.0, "profile": profile}

        mean_r = float(np.mean(profile)) or 1.0
        smooth = cv2.GaussianBlur(profile.astype(np.float32).reshape(1, -1), (1, 15), 0).ravel()

        diff = np.diff(smooth)
        sign_change = np.diff(np.sign(diff))
        peaks = np.where((sign_change[:-1] > 0))[0] + 1
        valleys = np.where((sign_change[:-1] < 0))[0] + 1

        spike_count = 0
        for peak in peaks:
            prominence = (smooth[peak] - np.min(smooth)) / mean_r
            if prominence > 0.30 and smooth[peak] > mean_r * 1.15:
                spike_count += 1

        lobule_count = 0
        for peak in peaks:
            prominence = (smooth[peak] - np.min(smooth)) / mean_r
            if 0.05 < prominence <= 0.30:
                lobule_count += 1

        roughness = float(np.std(profile - smooth) / mean_r)
        return {
            "lobule_count": int(lobule_count),
            "spike_count": int(spike_count),
            "roughness": roughness,
            "profile": profile,
            "valleys": int(len(valleys)),
        }

    @staticmethod
    def _boundary_strength(
        gray: np.ndarray, contour: np.ndarray, mask: np.ndarray
    ) -> float:
        """Mean gradient magnitude on the lesion border (visibility of margin)."""
        gray8 = gray if gray.dtype == np.uint8 else cv2.normalize(
            gray, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        gx = cv2.Sobel(gray8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray8, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)

        band = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(band, [contour], -1, 255, 3)
        values = magnitude[band > 0]
        if values.size == 0:
            return 0.0
        return float(np.mean(values))

    def _density(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        breast_mask: Optional[np.ndarray],
        detection,
    ) -> Dict[str, Any]:
        """Compare lesion attenuation with the immediately surrounding tissue."""
        gray_f = gray.astype(np.float32)
        lesion_values = gray_f[mask > 0]
        if lesion_values.size == 0:
            return {"density": "not assessed", "confidence": 0.0, "difference": 0.0}

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        dilated = cv2.dilate(mask, kernel)
        ring = cv2.subtract(dilated, mask)
        if breast_mask is not None:
            ring = cv2.bitwise_and(ring, (breast_mask > 0).astype(np.uint8) * 255)
        ring_values = gray_f[ring > 0]
        if ring_values.size < 20:
            return {
                "density": "equal density",
                "confidence": 0.3,
                "difference": 0.0,
                "note": "surrounding tissue too small for a reliable comparison",
            }

        difference = float(np.mean(lesion_values) - np.mean(ring_values))
        if difference > self.DENSITY_DIFFERENCE_HIGH:
            label = "high density"
            confidence = min(1.0, 0.6 + abs(difference) / 120.0)
        elif difference < self.DENSITY_DIFFERENCE_LOW:
            label = "low density"
            confidence = min(1.0, 0.6 + abs(difference) / 120.0)
        else:
            label = "equal density"
            confidence = 0.75

        # Very dark interior compared with the rim suggests fat content
        inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        if cv2.countNonZero(inner) > 20:
            inner_mean = float(np.mean(gray_f[inner > 0]))
            if inner_mean < float(np.mean(ring_values)) - 10.0:
                label = "fat-containing"
                confidence = 0.55

        return {
            "density": label,
            "confidence": float(confidence),
            "difference": difference,
            "lesion_mean": float(np.mean(lesion_values)),
            "ring_mean": float(np.mean(ring_values)),
        }

    def _classify_shape(
        self, geometry: Dict[str, Any], radial: Dict[str, Any]
    ) -> Tuple[str, float, Dict[str, float]]:
        circularity = geometry["circularity"]
        solidity = geometry["solidity"]
        rect = geometry["min_area_rect"][1]
        long_axis = max(rect[0], rect[1], 1e-6)
        short_axis = min(rect[0], rect[1], 1e-6)
        aspect = long_axis / short_axis

        scores = {
            "round": float(np.clip(
                circularity / 0.85, 0.0, 1.0
            ) * (1.0 if aspect < 1.25 else 0.3)),
            "oval": float(np.clip(1.0 - abs(aspect - 1.7) / 1.2, 0.0, 1.0))
            * float(np.clip(solidity / 0.9, 0.0, 1.0)),
            "irregular": float(np.clip(
                (1.0 - circularity) + geometry["convexity_deficit"]
                + radial["roughness"], 0.0, 1.0
            )),
        }
        label = max(scores, key=scores.get)
        if scores[label] < 0.25:
            label = "irregular" if circularity < 0.65 else "oval"
        return label, float(scores[label]), {k: round(v, 3) for k, v in scores.items()}

    def _classify_margin(
        self,
        geometry: Dict[str, Any],
        radial: Dict[str, Any],
        boundary_gradient: float,
    ) -> Tuple[str, float, Dict[str, float]]:
        deficit = geometry["convexity_deficit"]
        spikes = radial["spike_count"]
        lobules = radial["lobule_count"]
        roughness = radial["roughness"]

        scores = {
            "spiculated": float(np.clip(
                deficit / self.SPICULATION_STRONG, 0.0, 1.0
            )) * (1.0 if spikes >= 2 else 0.6),
            "microlobulated": float(np.clip(
                lobules / self.MICROLOBULATION_COUNT, 0.0, 1.0
            )),
            "indistinct": float(np.clip(roughness / 0.18, 0.0, 1.0))
            * float(np.clip(boundary_gradient / 20.0, 0.0, 1.0)),
            "obscured": float(np.clip(
                (self.BOUNDARY_OBSCURED_GRADIENT - boundary_gradient)
                / self.BOUNDARY_OBSCURED_GRADIENT, 0.0, 1.0
            )),
            "circumscribed": float(np.clip(geometry["solidity"] / 0.95, 0.0, 1.0))
            * float(np.clip(boundary_gradient / 25.0, 0.0, 1.0))
            * (1.0 if lobules < 3 and deficit < 0.12 else 0.4),
        }
        # Spiculation and obscured margins are clinically the most specific:
        # they win ties unless clearly absent.
        if scores["spiculated"] >= max(scores.values()) - 0.05:
            label = "spiculated"
        else:
            label = max(scores, key=scores.get)
        if scores[label] < 0.2:
            label = "indistinct" if boundary_gradient < 18 else "circumscribed"
        return label, float(scores[label]), {
            k: round(v, 3) for k, v in scores.items()
        }

    @staticmethod
    def _orientation(width: float, height: float, geometry: Dict[str, Any]) -> str:
        """BI-RADS 'orientation' descriptor: parallel / not parallel to skin."""
        rect = geometry["min_area_rect"][1]
        long_axis, short_axis = max(rect), min(rect)
        if short_axis <= 0:
            return "not assessable"
        return "parallel" if long_axis / short_axis >= 1.2 else "not parallel"


def apply_characterization_to_detections(
    detections: List,
    characterization: Dict[int, Dict[str, Any]],
) -> None:
    """Store lexicon descriptors on the detections (used by BI-RADS + report)."""
    for detection in detections:
        result = characterization.get(id(detection))
        if not result:
            continue
        detection.features["mass_characterization"] = result
        detection.features["shape"] = result.get("shape")
        detection.features["margin"] = result.get("margin")
        detection.features["lesion_density"] = result.get("density")
        detection.features["orientation"] = result.get("orientation")

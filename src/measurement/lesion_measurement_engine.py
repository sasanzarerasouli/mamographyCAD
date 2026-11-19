"""Lesion measurement calculations with DICOM pixel spacing support."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("MammographyCAD.measurement")


# ------------------------------------------------------------------ #
#  Module-level helper                                                 #
# ------------------------------------------------------------------ #

def _max_pairwise_distance(points: np.ndarray) -> float:
    """Maximum Euclidean distance between any pair of 2-D points (Feret diameter)."""
    n = len(points)
    if n < 2:
        return 0.0
    max_d2 = 0.0
    for i in range(n - 1):
        diff = points[i + 1:] - points[i]
        d2 = (diff * diff).sum(axis=1)
        cand = d2.max()
        if cand > max_d2:
            max_d2 = cand
    return float(np.sqrt(max_d2))


# ------------------------------------------------------------------ #
#  Engine                                                              #
# ------------------------------------------------------------------ #

class LesionMeasurementEngine:
    """Calculate lesion dimensions in pixels and millimeters.

    Metrics produced per lesion
    ──────────────────────────
    • Maximum (Feret) Diameter  – longest chord across the convex hull
    • Area                      – contour / mask / bbox area
    • Perimeter                 – contour arc-length or bbox perimeter
    • Equivalent Diameter       – diameter of a circle with the same area
    • Circularity               – 4π·A / P²  (1.0 = perfect circle)
    • Bounding Circle           – minimum enclosing circle
    • Bounding Box              – axis-aligned bbox **and** minimum-area
                                  rotated rectangle (from ``minAreaRect``)

    All values are returned in **pixel** *and* **mm** units when proper
    DICOM ``PixelSpacing`` is supplied.  Every key that existed in the
    previous version is kept so downstream code is unaffected.
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def measure_all(
        self,
        detections: List,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
        spacing_known: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Measure every non-ignored detection and return a list of dicts.

        ``spacing_known`` states whether ``pixel_spacing`` is a measurement
        (DICOM ``PixelSpacing``) or a documented assumption; ``None`` falls back
        to inferring it from the values, which is what the legacy callers do.
        """
        sy, sx = pixel_spacing
        measurements: List[Dict[str, Any]] = []
        for detection in detections:
            if detection.is_ignored:
                continue
            m = self.measure_detection(detection, sx, sy, spacing_known=spacing_known)
            # backward-compatible attribute assignments
            detection.diameter_px = m["diameter_px"]
            detection.diameter_mm = m["diameter_mm"]
            detection.area_px    = m["area_px"]
            detection.area_mm2   = m["area_mm2"]
            measurements.append(m)
        return measurements

    def measure_detection(
        self,
        detection,
        spacing_x: float = 1.0,
        spacing_y: float = 1.0,
        spacing_known: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Return a comprehensive measurement dict for *detection*."""
        geo = self._extract_geometry(detection)

        area_px      = geo["area_px"]
        perimeter_px = geo["perimeter_px"]
        max_diam_px  = geo["max_diameter_px"]
        bc_center    = geo["bc_center"]
        bc_radius_px = geo["bc_radius_px"]
        mar          = geo["min_area_rect"]
        contour      = geo["contour"]

        # -- derived pixel metrics --
        equiv_diam_px = (
            2.0 * np.sqrt(area_px / np.pi) if area_px > 0 else 0.0
        )
        circularity = 0.0
        if perimeter_px > 0:
            circularity = min(
                (4.0 * np.pi * area_px) / (perimeter_px ** 2), 1.0
            )

        # -- mm conversions --
        avg_sp = (spacing_x + spacing_y) / 2.0
        max_diam_mm = (
            self._max_diameter_physical(contour, spacing_x, spacing_y)
            if contour is not None
            else max_diam_px * avg_sp
        )
        equiv_diam_mm = equiv_diam_px * avg_sp
        area_mm2     = area_px * spacing_x * spacing_y
        perimeter_mm = perimeter_px * avg_sp

        # axis-aligned bbox → mm
        bbox_mm = None
        if detection.bbox:
            x, y, w, h = detection.bbox
            bbox_mm = (
                x * spacing_x,
                y * spacing_y,
                w * spacing_x,
                h * spacing_y,
            )

        # bounding circle → mm
        bc_mm = None
        if bc_center is not None:
            bc_mm = {
                "center": (
                    round(bc_center[0] * spacing_x, 2),
                    round(bc_center[1] * spacing_y, 2),
                ),
                "radius_mm": round(bc_radius_px * avg_sp, 2),
            }

        # min-area rotated rect → mm
        mar_mm = None
        if mar is not None:
            mar_mm = {
                "center": (
                    round(mar["center"][0] * spacing_x, 2),
                    round(mar["center"][1] * spacing_y, 2),
                ),
                "size": (
                    round(mar["size"][0] * spacing_x, 2),
                    round(mar["size"][1] * spacing_y, 2),
                ),
                "angle": mar["angle"],
            }

        return {
            # ── backward-compatible keys ──────────────────────────────
            "lesion_id":         detection.lesion_id,
            "diameter_px":       round(equiv_diam_px, 2),
            "diameter_mm":       round(equiv_diam_mm, 2),
            "area_px":           round(area_px, 2),
            "area_mm2":          round(area_mm2, 2),
            "perimeter_px":      round(perimeter_px, 2),
            "perimeter_mm":      round(perimeter_mm, 2),
            "bbox_mm":           bbox_mm,
            "pixel_spacing":     [spacing_y, spacing_x],
            "spacing_available": self._spacing_available(
                spacing_x, spacing_y, spacing_known
            ),
            "spacing_basis": self._spacing_basis(spacing_x, spacing_y, spacing_known),
            # ── extended metrics ──────────────────────────────────────
            "max_diameter_px":        round(max_diam_px, 2),
            "max_diameter_mm":        round(max_diam_mm, 2),
            "equivalent_diameter_px": round(equiv_diam_px, 2),
            "equivalent_diameter_mm": round(equiv_diam_mm, 2),
            "circularity":            round(circularity, 4),
            "bounding_circle_px": {
                "center": (
                    round(bc_center[0], 2),
                    round(bc_center[1], 2),
                ),
                "radius": round(bc_radius_px, 2),
            } if bc_center is not None else None,
            "bounding_circle_mm":  bc_mm,
            "min_area_rect_px":    mar,
            "min_area_rect_mm":    mar_mm,
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _spacing_available(
        spacing_x: float, spacing_y: float, spacing_known: Optional[bool]
    ) -> bool:
        if spacing_known is not None:
            return bool(spacing_known)
        return spacing_x != 1.0 or spacing_y != 1.0

    @classmethod
    def _spacing_basis(
        cls, spacing_x: float, spacing_y: float, spacing_known: Optional[bool]
    ) -> str:
        if spacing_known is False and (spacing_x != 1.0 or spacing_y != 1.0):
            return (
                f"assumed pixel spacing {spacing_x:.4f} mm/px (no DICOM spacing "
                "available) - mm values are estimates"
            )
        if cls._spacing_available(spacing_x, spacing_y, spacing_known):
            return f"DICOM pixel spacing ({spacing_x:.4f} mm/px)"
        return "pixel units (pixel spacing unavailable)"

    @staticmethod
    def _contour_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract the largest contour from a binary mask."""
        binary = (mask > 0).astype(np.uint8)
        # result[-2] works for both OpenCV 3 (img, cnts, hier)
        # and OpenCV 4 (cnts, hier)
        result = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = result[-2]
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    @classmethod
    def _extract_geometry(cls, detection) -> Dict[str, Any]:
        """Collect raw pixel-space geometry from a detection object."""
        area_px      = 0.0
        perimeter_px = 0.0
        max_diam_px  = 0.0
        bc_center: Optional[Tuple[float, float]] = None
        bc_radius_px = 0.0
        mar: Optional[Dict[str, Any]] = None
        contour      = None

        # ---- source: explicit contour ----
        if detection.contour is not None and len(detection.contour) >= 2:
            contour = detection.contour

        # ---- source: binary mask → derive contour ----
        elif detection.mask is not None:
            contour = cls._contour_from_mask(detection.mask)
            # absolute fallback when no contour could be extracted
            if contour is None and np.any(detection.mask > 0):
                area_px = float(np.sum(detection.mask > 0))
                r = np.sqrt(area_px / np.pi) if area_px > 0 else 0.0
                perimeter_px = 2.0 * np.pi * r
                max_diam_px  = 2.0 * r

        # ---- compute everything from contour ----
        if contour is not None:
            area_px = float(np.count_nonzero(detection.mask) if detection.mask is not None else cv2.contourArea(contour))
            perimeter_px = float(cv2.arcLength(contour, True))

            hull = cv2.convexHull(contour)
            pts  = hull.reshape(-1, 2).astype(np.float64)
            max_diam_px = (
                _max_pairwise_distance(pts) if len(pts) >= 2 else 0.0
            )

            center, radius = cv2.minEnclosingCircle(contour)
            bc_center    = (float(center[0]), float(center[1]))
            bc_radius_px = float(radius)

            rect = cv2.minAreaRect(contour)
            mar = {
                "center": (
                    round(float(rect[0][0]), 2),
                    round(float(rect[0][1]), 2),
                ),
                "size": (
                    round(float(rect[1][0]), 2),
                    round(float(rect[1][1]), 2),
                ),
                "angle": round(float(rect[2]), 2),
            }

        # ---- fallback: axis-aligned bbox only ----
        elif detection.bbox:
            x, y, w, h = detection.bbox
            area_px      = float(w * h)
            perimeter_px = float(2 * (w + h))
            max_diam_px  = float(np.hypot(w, h))
            bc_center    = (float(x + w / 2.0), float(y + h / 2.0))
            bc_radius_px = max_diam_px / 2.0
            mar = {
                "center": (
                    round(float(x + w / 2.0), 2),
                    round(float(y + h / 2.0), 2),
                ),
                "size":  (round(float(w), 2), round(float(h), 2)),
                "angle": 0.0,
            }

        return {
            "area_px":         area_px,
            "perimeter_px":    perimeter_px,
            "max_diameter_px": max_diam_px,
            "bc_center":       bc_center,
            "bc_radius_px":    bc_radius_px,
            "min_area_rect":   mar,
            "contour":         contour,
        }

    @staticmethod
    def _max_diameter_physical(
        contour: np.ndarray,
        sx: float,
        sy: float,
    ) -> float:
        """Max Feret diameter in **mm** using anisotropic pixel spacing."""
        if contour is None or len(contour) < 2:
            return 0.0
        hull = cv2.convexHull(contour)
        pts  = hull.reshape(-1, 2).astype(np.float64)
        # scale to physical coordinates *before* distance computation
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        return _max_pairwise_distance(pts)
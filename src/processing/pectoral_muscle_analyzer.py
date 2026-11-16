import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("MammographyCAD.pectoral")


class PectoralMuscleAnalyzer:
    """
    Detects and removes the pectoral muscle from MLO (mediolateral oblique)
    mammography views.

    Pipeline
    --------
    1. Hough line detection  -> finds the (roughly) straight pectoral edge.
    2. Texture analysis      -> pectoral muscle is a comparatively
                                 homogeneous, denser region.
    3. Region growing        -> grows from a seed near the chest-wall
                                 corner, bounded by the breast mask.
    4. Geometric constraint  -> pectoral muscle only ever occupies the
                                 upper triangular corner of the image on
                                 the chest-wall side.

    The four cues are combined with a weighted vote, the resulting mask is
    regularized/smoothed, and a hard area cap guards against
    over-segmentation that would otherwise remove genuine breast tissue.
    """

    # ---- tunable safety / behaviour parameters --------------------------------
    MAX_PECTORAL_AREA_RATIO = 0.35   # pectoral region can never exceed this % of breast area
    VOTE_THRESHOLD = 2.0             # weighted votes needed for a pixel to be "pectoral"
    EROSION_KERNEL_SIZE = 5
    CLOSE_KERNEL_SIZE = 15

    CUE_WEIGHTS = {"hough": 1.4, "texture": 1.0, "region_growing": 0.8, "geometric": 0.6}
    CONFIDENCE_WEIGHTS = {"hough": 0.35, "texture": 0.25, "region_growing": 0.2, "geometric": 0.1}

    def analyze(self, gray: np.ndarray, breast_mask: np.ndarray, view: str = "MLO") -> Dict[str, Any]:
        """Run the full pectoral-muscle detection & removal pipeline.

        Parameters
        ----------
        gray:
            Grayscale mammogram image (uint8).
        breast_mask:
            Binary mask (any nonzero = foreground) of the breast region.
        view:
            Acquisition view. Only ``"MLO"`` is supported -- pectoral
            muscle is not expected on CC views, so any other value
            short-circuits to an empty, zero-confidence result.
        """
        if view is None or view.upper() != "MLO":
            logger.info("Pectoral analysis skipped: view=%s is not MLO", view)
            return self._empty_result(gray)

        if gray is None or breast_mask is None or not np.any(breast_mask):
            logger.warning("Pectoral analysis skipped: empty image or breast mask")
            return self._empty_result(gray)

        h, w = gray.shape
        breast_mask = (breast_mask > 0).astype(np.uint8) * 255
        breast_area = float(np.count_nonzero(breast_mask))

        is_left = self._infer_laterality(breast_mask)

        # ---- collect independent cues -------------------------------------------------
        votes = np.zeros((h, w), dtype=np.float32)
        cue_hits: Dict[str, bool] = {}

        line_mask, line_found = self._hough_detection(gray, breast_mask, is_left)
        if line_found:
            votes += (line_mask > 0).astype(np.float32) * self.CUE_WEIGHTS["hough"]
        cue_hits["hough"] = line_found

        texture_mask = self._texture_region(gray, breast_mask, is_left)
        has_texture = bool(np.any(texture_mask))
        if has_texture:
            votes += (texture_mask > 0).astype(np.float32) * self.CUE_WEIGHTS["texture"]
        cue_hits["texture"] = has_texture

        seed_mask = self._region_growing(gray, breast_mask, is_left)
        has_seed = bool(np.any(seed_mask))
        if has_seed:
            votes += (seed_mask > 0).astype(np.float32) * self.CUE_WEIGHTS["region_growing"]
        cue_hits["region_growing"] = has_seed

        geo_mask = self._geometric_constraint(h, w, is_left)
        votes += (geo_mask > 0).astype(np.float32) * self.CUE_WEIGHTS["geometric"]
        cue_hits["geometric"] = True

        # ---- combine cues into a single candidate mask ---------------------------------
        candidate = (votes >= self.VOTE_THRESHOLD).astype(np.uint8) * 255
        candidate = cv2.bitwise_and(candidate, geo_mask)      # never leave the chest-wall corner
        candidate = cv2.bitwise_and(candidate, breast_mask)   # never exceed the breast region

        pectoral_mask = self._regularize_mask(candidate, breast_mask, breast_area, h, w, is_left)

        # ---- build outputs ---------------------------------------------------------------
        fill_value = int(np.median(gray[breast_mask > 0])) if np.any(breast_mask) else 0
        pectoral_removed = gray.copy()
        pectoral_removed[pectoral_mask > 0] = fill_value

        boundary_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(pectoral_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(boundary_overlay, contours, -1, (255, 100, 0), 2)

        confidence = self._compute_confidence(cue_hits, pectoral_mask, breast_area)
        area_ratio = (np.count_nonzero(pectoral_mask) / breast_area) if breast_area else 0.0
        logger.info(
            "Pectoral analysis: laterality=%s confidence=%.2f area_ratio=%.2f cues=%s",
            "left" if is_left else "right",
            confidence,
            area_ratio,
            cue_hits,
        )

        return {
            "pectoral_mask": pectoral_mask,
            "pectoral_removed": pectoral_removed,
            "boundary_overlay": boundary_overlay,
            "confidence": confidence,
            "is_left": is_left,
        }

    # ------------------------------------------------------------------ helpers -----

    def _empty_result(self, gray: Optional[np.ndarray]) -> Dict[str, Any]:
        if gray is None:
            gray = np.zeros((1, 1), dtype=np.uint8)
        empty_mask = np.zeros_like(gray, dtype=np.uint8)
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
        return {
            "pectoral_mask": empty_mask,
            "pectoral_removed": gray.copy(),
            "boundary_overlay": overlay,
            "confidence": 0.0,
            "is_left": None,
        }

    def _infer_laterality(self, breast_mask: np.ndarray) -> bool:
        """True if the breast (and therefore the pectoral muscle) sits on the left half."""
        h, w = breast_mask.shape
        left_sum = np.sum(breast_mask[:, : w // 2] > 0)
        right_sum = np.sum(breast_mask[:, w // 2:] > 0)
        return left_sum >= right_sum

    def _hough_detection(self, gray: np.ndarray, breast_mask: np.ndarray, is_left: bool) -> Tuple[np.ndarray, bool]:
        """Detect the (roughly straight) pectoral boundary line and return a
        mask of the "pectoral side" of that line. Fully vectorized -- no
        per-pixel Python loops, unlike a naive implementation.
        """
        h, w = gray.shape
        roi = np.zeros_like(gray, dtype=np.uint8)
        if is_left:
            roi[: h // 2, : w // 2] = 1
        else:
            roi[: h // 2, w // 2:] = 1

        edges = cv2.Canny(gray, 50, 150)
        edges = edges * roi * (breast_mask > 0).astype(np.uint8)

        lines = cv2.HoughLines(edges.astype(np.uint8), 1, np.pi / 180, 80)
        mask = np.zeros_like(gray, dtype=np.uint8)
        if lines is None:
            return mask, False

        yy, xx = np.mgrid[0:h, 0:w]
        found_any = False
        for rho, theta in lines[:, 0]:
            # Skip near-horizontal lines: the pectoral boundary is a
            # diagonal/near-vertical edge, never a horizontal one.
            if np.pi / 6 < theta < 5 * np.pi / 6:
                continue
            a, b = np.cos(theta), np.sin(theta)
            if abs(a) < 1e-6:
                continue
            line_x = (rho - yy * b) / a
            side = (xx < line_x) if is_left else (xx > line_x)
            mask[side] = 255
            found_any = True

        return mask, found_any

    def _texture_region(self, gray: np.ndarray, breast_mask: np.ndarray, is_left: bool) -> np.ndarray:
        """Pectoral muscle tends to be a comparatively smooth/homogeneous and
        denser (brighter) region near the chest wall; flag low local-variance,
        higher-intensity pixels in that corner."""
        h, w = gray.shape
        roi_mask = np.zeros_like(gray, dtype=np.uint8)
        if is_left:
            roi_mask[: h // 2, : w // 2] = 255
        else:
            roi_mask[: h // 2, w // 2:] = 255
        roi_mask = cv2.bitwise_and(roi_mask, breast_mask)

        if not np.any(roi_mask):
            return np.zeros_like(gray, dtype=np.uint8)

        gray_f = gray.astype(np.float32)
        mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(9, 9))
        sq_mean = cv2.boxFilter(gray_f * gray_f, ddepth=-1, ksize=(9, 9))
        local_var = np.clip(sq_mean - mean * mean, 0, None)
        local_std = np.sqrt(local_var)

        std_norm = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, low_texture = cv2.threshold(std_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        low_texture = cv2.bitwise_and(low_texture, roi_mask)

        # Pectoral tissue is usually denser/brighter than surrounding fat.
        breast_pixels = gray[breast_mask > 0]
        if breast_pixels.size:
            intensity_cutoff = np.percentile(breast_pixels, 40)
            low_texture[gray <= intensity_cutoff] = 0

        return low_texture

    def _region_growing(self, gray: np.ndarray, breast_mask: np.ndarray, is_left: bool) -> np.ndarray:
        """Grow a region from a seed near the chest-wall corner using
        OpenCV's native ``floodFill`` (fast, C-implemented), strictly bounded
        by the breast mask so it can never leak into surrounding background
        or, more importantly, overrun far into breast tissue.
        """
        h, w = gray.shape
        seed = (w // 6, h // 8) if is_left else (5 * w // 6, h // 8)
        sx, sy = seed
        if not (0 <= sx < w and 0 <= sy < h) or breast_mask[sy, sx] == 0:
            return np.zeros_like(gray, dtype=np.uint8)

        work_img = gray.copy()
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        # Mark everything outside the breast as a barrier so the fill can't leak.
        flood_mask[1:-1, 1:-1][breast_mask == 0] = 1

        diff = 12
        try:
            cv2.floodFill(
                work_img,
                flood_mask,
                seedPoint=(sx, sy),
                newVal=255,
                loDiff=diff,
                upDiff=diff,
                flags=4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8),
            )
        except cv2.error as exc:
            logger.warning("Region growing flood fill failed: %s", exc)
            return np.zeros_like(gray, dtype=np.uint8)

        grown = (flood_mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
        return grown

    def _geometric_constraint(self, h: int, w: int, is_left: bool) -> np.ndarray:
        """Pectoral muscle only ever occupies the upper triangular region on
        the chest-wall side of an MLO view."""
        mask = np.zeros((h, w), dtype=np.uint8)
        apex_h = int(h * 0.6)
        apex_w = int(w * 0.6)
        if is_left:
            pts = np.array([[0, 0], [apex_w, 0], [0, apex_h]], np.int32)
        else:
            pts = np.array([[w, 0], [w - apex_w, 0], [w, apex_h]], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _regularize_mask(
        self,
        candidate: np.ndarray,
        breast_mask: np.ndarray,
        breast_area: float,
        h: int,
        w: int,
        is_left: bool,
    ) -> np.ndarray:
        """Clean up the raw voted mask: keep only the chest-wall-connected
        component, smooth it, and hard-cap its area so real breast tissue is
        never mistakenly erased."""
        if not np.any(candidate):
            return np.zeros_like(candidate)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.CLOSE_KERNEL_SIZE,) * 2)
        closed = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel_close)

        num_labels, labels = cv2.connectedComponents(closed)
        corner = (0, 0) if is_left else (0, w - 1)
        corner_label = labels[corner]
        if corner_label == 0:
            # Fall back to the largest component if the exact corner pixel
            # isn't part of the mask (e.g. due to erosion at the border).
            counts = np.bincount(labels.flatten())
            counts[0] = 0
            if counts.max() == 0:
                return np.zeros_like(candidate)
            corner_label = int(np.argmax(counts))

        pectoral = np.where(labels == corner_label, 255, 0).astype(np.uint8)

        # Erode slightly to leave a safety margin away from real breast tissue.
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.EROSION_KERNEL_SIZE,) * 2)
        pectoral = cv2.erode(pectoral, kernel_erode)

        # Hard area cap: pectoral muscle should never dominate the breast area.
        area = np.count_nonzero(pectoral)
        if breast_area > 0 and area / breast_area > self.MAX_PECTORAL_AREA_RATIO:
            logger.warning(
                "Pectoral candidate too large (%.1f%% of breast); shrinking to safety cap",
                100 * area / breast_area,
            )
            pectoral = self._shrink_to_area_ratio(pectoral, breast_area, is_left, h, w)

        return pectoral

    def _shrink_to_area_ratio(
        self, mask: np.ndarray, breast_area: float, is_left: bool, h: int, w: int
    ) -> np.ndarray:
        """Iteratively erode the mask until it satisfies the maximum allowed
        area ratio, falling back to a small conservative triangle if erosion
        collapses it entirely."""
        target_area = breast_area * self.MAX_PECTORAL_AREA_RATIO
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shrunk = mask.copy()
        for _ in range(20):
            if not np.any(shrunk) or np.count_nonzero(shrunk) <= target_area:
                break
            shrunk = cv2.erode(shrunk, kernel)

        if not np.any(shrunk):
            fallback = np.zeros((h, w), dtype=np.uint8)
            apex = int(min(h, w) * 0.25)
            if is_left:
                pts = np.array([[0, 0], [apex, 0], [0, apex]], np.int32)
            else:
                pts = np.array([[w, 0], [w - apex, 0], [w, apex]], np.int32)
            cv2.fillPoly(fallback, [pts], 255)
            return fallback
        return shrunk

    def _compute_confidence(
        self, cue_hits: Dict[str, bool], pectoral_mask: np.ndarray, breast_area: float
    ) -> float:
        """Confidence grows with the number of agreeing cues and a plausible
        (non-trivial, non-excessive) resulting pectoral area."""
        confidence = sum(self.CONFIDENCE_WEIGHTS[c] for c, hit in cue_hits.items() if hit)

        if breast_area > 0 and np.any(pectoral_mask):
            ratio = np.count_nonzero(pectoral_mask) / breast_area
            # A plausible pectoral area on an MLO view is roughly 5%-35% of the breast area.
            if 0.05 <= ratio <= self.MAX_PECTORAL_AREA_RATIO:
                confidence += 0.1
            else:
                confidence -= 0.1

        contours, _ = cv2.findContours(pectoral_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            confidence += 0.05

        return float(np.clip(confidence, 0.0, 1.0))

"""Mammogram loading, breast segmentation and contrast enhancement.

The preprocessor is the entry point of the analysis pipeline.  It accepts
DICOM or raster files, normalises every input to an 8-bit *display oriented*
grayscale image (bright = dense tissue), extracts the breast mask and produces
the enhanced image used by the detectors.

Performance note: the previous Hough-based pectoral removal used a per-pixel
Python double loop, which took tens of seconds on a 3000x4000 mammogram.  The
implementation below is fully vectorised (same geometry, ~1000x faster).
"""
import logging
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..io.dicom_handler import DICOMHandler

logger = logging.getLogger("MammographyCAD.preprocessor")


class MammographyPreprocessor:

    # ------------------------------------------------------------------ #
    #  Loading                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_image(img_path: str, *, source_pixels=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a mammogram as ``(gray8 display image, metadata)``."""
        if img_path is None or not os.path.exists(img_path):
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        gray, metadata = DICOMHandler().load_grayscale(img_path, source_pixels=source_pixels)
        if gray is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        gray = MammographyPreprocessor.ensure_display_polarity(gray)
        return gray, metadata

    @staticmethod
    def ensure_display_polarity(gray: np.ndarray) -> np.ndarray:
        """Guarantee 'bright tissue on dark background' orientation.

        DICOM ``MONOCHROME1`` is handled in the DICOM layer; this guard covers
        inverted raster exports (and 8-bit conversions that lost the
        photometric tag) by comparing the image border with the image centre.
        """
        if gray is None:
            return gray
        h, w = gray.shape[:2]
        if h < 20 or w < 20:
            return gray
        border = np.concatenate(
            [gray[0:3, :].ravel(), gray[-3:, :].ravel(),
             gray[:, 0:3].ravel(), gray[:, -3:].ravel()]
        )
        interior = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4].ravel()
        if interior.size == 0 or border.size == 0:
            return gray
        if float(np.mean(border)) > float(np.mean(interior)) + 20.0:
            logger.info("Inverted (MONOCHROME1-like) raster detected; flipping")
            return cv2.bitwise_not(gray)
        return gray

    # ------------------------------------------------------------------ #
    #  Main API                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess(img_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward-compatible entry point.

        Returns
        -------
        (original, enhanced, breast_mask)
            ``original`` is the 8-bit display grayscale image (2-D).
        """
        result = MammographyPreprocessor.preprocess_full(img_path)
        return (
            result["original"],
            result["enhanced"],
            result["breast_mask"],
        )

    @staticmethod
    def preprocess_full(
        img_path: str,
        remove_pectoral: bool = True,
    ) -> Dict[str, Any]:
        """Run the complete preprocessing stage and return all intermediates."""
        gray, metadata = MammographyPreprocessor.load_image(img_path)
        out = MammographyPreprocessor.preprocess_array(
            gray, metadata=metadata, remove_pectoral=remove_pectoral
        )
        out["path"] = img_path
        return out

    @staticmethod
    def preprocess_array(
        gray: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        remove_pectoral: bool = True,
    ) -> Dict[str, Any]:
        """Preprocess an in-memory grayscale array."""
        gray = MammographyPreprocessor.to_uint8(gray)
        gray = MammographyPreprocessor.ensure_display_polarity(gray)

        breast_mask = MammographyPreprocessor.extract_breast_mask(gray)
        cleaned = MammographyPreprocessor.remove_artifacts(gray, breast_mask)

        if remove_pectoral:
            removed, pectoral_mask = MammographyPreprocessor.remove_pectoral_muscle_ex(
                cleaned, breast_mask
            )
        else:
            removed, pectoral_mask = cleaned, np.zeros_like(breast_mask)

        enhanced = MammographyPreprocessor.enhance_contrast(removed, breast_mask)

        return {
            "original": gray,
            "cleaned": cleaned,
            "pectoral_removed": removed,
            "enhanced": enhanced,
            "breast_mask": breast_mask,
            "pectoral_mask": pectoral_mask,
            "metadata": metadata or {},
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def to_uint8(image: np.ndarray) -> np.ndarray:
        """Safely convert any numeric array to 8-bit grayscale."""
        if image is None:
            return image
        if image.dtype == np.uint8:
            return image
        data = image.astype(np.float32)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        lo, hi = float(finite.min()), float(finite.max())
        if hi - lo < 1e-6:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        scaled = (data - lo) * (255.0 / (hi - lo))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    @staticmethod
    def extract_breast_mask(gray: np.ndarray) -> np.ndarray:
        """Segment the breast region.

        Two stages are required in practice:

        1. **Frame removal.** Scanned films and screen photographs have a light
           page border around the dark film. Plain Otsu then selects the *page*
           as the foreground and the mask covers almost the whole image, which
           silently ruins every downstream measurement. The content area is
           therefore found first (largest region not connected to a uniform
           image border).
        2. **Breast segmentation inside the content area.** Otsu (on the content
           crop) plus largest-component selection, hole filling and smoothing.
        """
        gray = MammographyPreprocessor.to_uint8(gray)
        content = MammographyPreprocessor._content_region(gray)

        # Nested frames (photo-of-screen exports have a black margin *and* a
        # light page) require more than one pass.  Each pass removes the uniform
        # band that touches the current border, and the loop stops when nothing
        # meaningful is left to remove.
        for _ in range(2):
            previous = np.count_nonzero(content)
            masked = cv2.bitwise_and(gray, gray, mask=content)
            reduced = MammographyPreprocessor._content_region(masked)
            coverage = np.count_nonzero(reduced) / float(reduced.size)
            if coverage < 0.15 or coverage > 0.98:
                break
            if np.count_nonzero(reduced) >= 0.98 * previous:
                break
            content = reduced

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        restricted = np.zeros_like(blurred)
        restricted[content > 0] = blurred[content > 0]
        if np.count_nonzero(restricted) == 0:      # pragma: no cover - degenerate
            restricted = blurred

        _, binary = cv2.threshold(
            restricted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary = cv2.bitwise_and(binary, content)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels < 2:
            return binary

        # Largest bright component inside the content area
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(areas)) + 1
        mask = (labels == largest_label).astype(np.uint8) * 255

        # Close small holes, then fill interior holes so lesions inside the
        # parenchyma are never excluded from analysis.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, contours, -1, 255, -1)
            mask = filled

        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

        # Sanity check: a breast almost never occupies >80% of a mammography
        # frame. If it appears to, the bright *frame/page* was merged into the
        # mask with the breast (typical for photographs of a screen). A
        # watershed seeded on the largest inscribed disk separates the solid
        # breast from the thin surrounding frame.
        coverage = np.count_nonzero(mask) / float(mask.size)
        if coverage > 0.80:
            separated = MammographyPreprocessor._separate_breast_from_frame(
                gray, mask
            )
            if separated is not None:
                separated = cv2.GaussianBlur(separated, (21, 21), 0)
                _, separated = cv2.threshold(separated, 128, 255, cv2.THRESH_BINARY)
                separated_coverage = np.count_nonzero(separated) / float(separated.size)
                if 0.02 < separated_coverage < 0.80:
                    logger.info(
                        "Breast mask refined (frame separation): %.2f -> %.2f coverage",
                        coverage, separated_coverage,
                    )
                    mask = separated
        return mask

    @staticmethod
    def _separate_breast_from_frame(
        gray: np.ndarray, mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """Split the breast from a surrounding bright frame using a watershed."""
        try:
            binary = (mask > 0).astype(np.uint8)
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            if float(dist.max()) < 5:
                return None
            _, _, _, max_loc = cv2.minMaxLoc(dist)
            seed_value = float(dist[max_loc[1], max_loc[0]])

            core = ((dist >= max(3.0, 0.45 * seed_value)) & (binary > 0)).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(core, connectivity=8)
            if num_labels <= 1:
                return None
            seed_label = labels[max_loc[1], max_loc[0]]
            if seed_label == 0:
                return None
            core = (labels == seed_label).astype(np.uint8)

            markers = np.zeros(gray.shape, dtype=np.int32)
            markers[core > 0] = 1
            border_bright = np.zeros_like(binary)
            border_bright[0, :] = binary[0, :]
            border_bright[-1, :] = binary[-1, :]
            border_bright[:, 0] = binary[:, 0]
            border_bright[:, -1] = binary[:, -1]
            markers[(border_bright > 0) & (core == 0)] = 2
            markers[(binary == 0)] = 2

            gradient = cv2.morphologyEx(
                cv2.GaussianBlur(gray, (5, 5), 0),
                cv2.MORPH_GRADIENT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            colour = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
            cv2.watershed(colour, markers)

            breast = ((markers == 1) & (binary > 0)).astype(np.uint8) * 255
            if np.count_nonzero(breast) == 0:
                return None

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            breast = cv2.morphologyEx(breast, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                breast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return None
            filled = np.zeros_like(breast)
            cv2.drawContours(filled, contours, -1, 255, -1)
            return filled
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Frame separation failed: %s", exc)
            return None

    @staticmethod
    def _content_region(gray: np.ndarray, tolerance: int = 12) -> np.ndarray:
        """Return a mask of the film/content area, excluding a uniform border."""
        height, width = gray.shape[:2]
        border = np.concatenate(
            [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
        )
        background_value = int(np.median(border))
        background_std = float(np.std(border))

        # A non-uniform border means there is no framing page to remove.
        if background_std > 25.0:
            return np.ones_like(gray) * 255

        similar = (
            np.abs(gray.astype(np.int16) - background_value) <= tolerance
        ).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(similar, connectivity=4)
        if num_labels <= 1:
            return np.ones_like(gray) * 255

        border_labels = set()
        border_labels.update(labels[0, :].tolist())
        border_labels.update(labels[-1, :].tolist())
        border_labels.update(labels[:, 0].tolist())
        border_labels.update(labels[:, -1].tolist())
        border_labels.discard(0)

        background = np.isin(labels, list(border_labels)).astype(np.uint8)
        content = ((background == 0).astype(np.uint8)) * 255

        coverage = np.count_nonzero(content) / float(height * width)
        if coverage < 0.15 or coverage > 0.995:
            # Nothing meaningful was removed: keep the whole frame.
            return np.ones_like(gray) * 255

        # Keep only the largest content component (the film itself)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            content, connectivity=8
        )
        if num_labels <= 1:
            return content
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = int(np.argmax(areas)) + 1
        return ((labels == keep).astype(np.uint8)) * 255

    @staticmethod
    def remove_artifacts(gray: np.ndarray, breast_mask: np.ndarray) -> np.ndarray:
        """Null the background and flatten saturated markers."""
        cleaned = cv2.bitwise_and(gray, gray, mask=breast_mask)
        if np.count_nonzero(breast_mask) == 0:
            return cleaned
        # Saturated pixels (markers, scratches) are replaced by the breast median
        high_intensity_mask = (cleaned >= 250) & (breast_mask > 0)
        if np.any(high_intensity_mask):
            median_val = int(np.median(cleaned[breast_mask > 0]))
            cleaned = cleaned.copy()
            cleaned[high_intensity_mask] = median_val
        return cleaned

    # ------------------------------------------------------------------ #
    #  Pectoral muscle removal                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def remove_pectoral_muscle(
        gray: np.ndarray, breast_mask: np.ndarray
    ) -> np.ndarray:
        """Backward-compatible wrapper returning only the cleaned image."""
        removed, _ = MammographyPreprocessor.remove_pectoral_muscle_ex(
            gray, breast_mask
        )
        return removed

    @staticmethod
    def remove_pectoral_muscle_ex(
        gray: np.ndarray, breast_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect and suppress the pectoral muscle on MLO views.

        Returns ``(image_without_pectoral, pectoral_mask)``. The mask is all
        zeros when no reliable pectoral boundary is found, so CC views and
        pectoral-free images are left untouched.
        """
        h, w = gray.shape[:2]
        result = gray.copy()
        empty = np.zeros_like(gray)

        if np.count_nonzero(breast_mask) == 0 or h < 40 or w < 40:
            return result, empty

        # Which half holds the breast -> the chest wall is on the opposite side
        left_sum = float(np.sum(breast_mask[:, : w // 2]))
        right_sum = float(np.sum(breast_mask[:, w // 2:]))
        is_left = left_sum > right_sum

        # Search the upper quadrant next to the chest wall
        roi_mask = np.zeros_like(gray)
        if is_left:
            roi_mask[: h // 2, : w // 2] = 1
        else:
            roi_mask[: h // 2, w // 2:] = 1
        roi_mask = cv2.bitwise_and(roi_mask, roi_mask, mask=breast_mask)

        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        edges = cv2.bitwise_and(edges, roi_mask)

        lines = cv2.HoughLines(edges.astype(np.uint8), 1, np.pi / 180, 100)
        if lines is None:
            return result, empty

        best = None
        best_score = -1.0
        for rho, theta in lines[:, 0]:
            # Pectoral edge is oblique: the Hough normal spans roughly
            # 15°..75° for realistic MLO pectoral boundaries.
            if not (np.pi / 12 < theta < 5 * np.pi / 12):
                continue
            a, b = np.cos(theta), np.sin(theta)
            if abs(a) < 1e-3:
                continue
            # Score = how many breast pixels lie on the removable side
            ys = np.arange(h)
            xs_line = (rho - ys * b) / a
            if is_left:
                score = float(
                    np.sum(
                        (np.arange(w)[None, :] < xs_line[:, None])
                        & (breast_mask > 0)
                    )
                )
            else:
                score = float(
                    np.sum(
                        (np.arange(w)[None, :] > xs_line[:, None])
                        & (breast_mask > 0)
                    )
                )
            if score > best_score:
                best_score = score
                best = (rho, theta, xs_line)

        if best is None or best_score <= 0:
            return result, empty

        rho, theta, xs_line = best
        breast_area = float(np.count_nonzero(breast_mask))
        # A pectoral muscle occupies at most ~35% of the breast area
        if best_score > 0.35 * breast_area:
            logger.debug(
                "Pectoral candidate too large (%.1f%% of breast); skipping",
                100.0 * best_score / max(breast_area, 1),
            )
            return result, empty

        xs_grid = xs_line[:, None]
        cols = np.arange(w)[None, :]
        if is_left:
            region = (cols < xs_grid).astype(np.uint8)
        else:
            region = (cols > xs_grid).astype(np.uint8)
        region = cv2.bitwise_and(region, (breast_mask > 0).astype(np.uint8))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        region = cv2.morphologyEx(region * 255, cv2.MORPH_CLOSE, kernel)
        pectoral_mask = region

        # Soft attenuation (do not hard-delete tissue: keeps the image usable)
        mask_f = cv2.GaussianBlur(region, (51, 51), 0).astype(np.float32) / 255.0
        result = (gray.astype(np.float32) * (1.0 - 0.9 * mask_f)).clip(0, 255)
        result = result.astype(np.uint8)
        return result, pectoral_mask

    # ------------------------------------------------------------------ #
    #  Enhancement                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def enhance_contrast(
        gray: np.ndarray, breast_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """CLAHE + unsharp masking + top-hat, restricted to the breast."""
        if breast_mask is None:
            breast_mask = MammographyPreprocessor.extract_breast_mask(gray)

        masked = cv2.bitwise_and(gray, gray, mask=breast_mask)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(masked)

        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

        enhanced = cv2.add(enhanced, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)
        return cv2.bitwise_and(enhanced, enhanced, mask=breast_mask)
